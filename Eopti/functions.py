import requests
from datetime import datetime, timedelta
from entsoe import EntsoePandasClient
import pandas as pd
import numpy as np
from influxdb import DataFrameClient
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mip import *
from pytz import timezone
import math
from io import BytesIO

#forecasting modules
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
# from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import grid_search_forecaster

#############################
## HA influxDB connection
#############################

class Eoptimization:

    def __init__(self, config = ""):
        self.config = config
        self.PVForecastToday={hour:0 for hour in range(0,24)}
        self.PVForecastTomorrow={hour:0 for hour in range(0,24)}
        self.CostPurchaseToday={hour:0 for hour in range(0,24)}
        self.CostPurchaseTomorrow={hour:0 for hour in range(0,24)}
        self.CostFeedBackToday={hour:0 for hour in range(0,24)}
        self.CostFeedBackTomorrow={hour:0 for hour in range(0,24)}
        self.edata=''
        self.Eforecast=''
        self.TempForecast=''
        self.ExogFut=''
        self.Optimization=''
        self.influxconfig=config['Eprediction']
        self.influxclient = DataFrameClient(host=self.influxconfig['influxdb_ip'], port=self.influxconfig['influxdb_port'], username=self.influxconfig['influxdb_username'], password=self.influxconfig['influxdb_password'], database=self.influxconfig['influxdb_database'])
        self.dayondayprice = 0.0
        self.calculatedat = datetime.now()
        #############################
        ## Functions
        #############################

        #load PV forecast
    def loadPVForecast(self):
        for row in self.config['PVinstallations']:
            previoushourT=0.0
            previoushourTom=0.0    
            lat=row['latitude']
            lon=row['longitude']
            dec=row['declination']
            azi=row['azimuth']
            kwp=row['kwp']
            response=requests.get(f'https://api.forecast.solar/estimate/watthours/{lat}/{lon}/{dec}/{azi}/{kwp}').json()
            for key in response['result']:
                a=datetime.strptime(key, '%Y-%m-%d %H:%M:%S')
                if a.date() == datetime.today().date():
                    if a.hour > 1:
                        self.PVForecastToday[a.hour-1]+=(response['result'][key]-previoushourT)
                        previoushourT=response['result'][key]
                        
                else:
                    if a.hour > 1:
                        self.PVForecastTomorrow[a.hour-1]+=(response['result'][key]-previoushourTom)
                        previoushourTom=response['result'][key]

    #get hourly cost from ENTSO-E platform
    def loadPrices(self):
        country_code = self.config['Costs']['country_code']
        start= pd.Timestamp(datetime.today().strftime('%Y-%m-%d'), tz=self.config['Costs']['tz'])
        end= pd.Timestamp((datetime.today()+timedelta(days=2)).strftime('%Y-%m-%d'), tz=self.config['Costs']['tz'])
        client = EntsoePandasClient(self.config['Costs']['api_key'])
        result = client.query_day_ahead_prices(country_code, start=start,end=end)
        for timestamp, price in result.items():
            if timestamp.to_pydatetime().date() == datetime.today().date():
                self.CostPurchaseToday[timestamp.to_pydatetime().hour]=(price/1000.0)*(1.0+self.config['Costs']['btw'])+self.config['Costs']['delivery_cost']+(self.config['Costs']['energy_tax'])
            else:
                self.CostPurchaseTomorrow[timestamp.to_pydatetime().hour]=(price/1000.0)*(1.0+self.config['Costs']['btw'])+self.config['Costs']['delivery_cost']+(self.config['Costs']['energy_tax'])
            if timestamp.to_pydatetime().date() == datetime.today().date():
                self.CostFeedBackToday[timestamp.to_pydatetime().hour]=-1.0*((price/1000.0)*(1.0+self.config['Costs']['btw'])+self.config['Costs']['feedback_rebate']+((self.config['Costs']['energy_tax'])*self.config['Costs']['saldering_percentage']))
            else:
                self.CostFeedBackTomorrow[timestamp.to_pydatetime().hour]=-1.0*((price/1000.0)*(1.0+self.config['Costs']['btw'])+self.config['Costs']['feedback_rebate']+((self.config['Costs']['energy_tax'])*self.config['Costs']['saldering_percentage']))

    #get temperature forecast
    def getTempForecast(self):
        self.TempForecast=pd.DataFrame()
        lat=self.config['TempForecast']['lat']
        lon=self.config['TempForecast']['lon']
        appid=self.config['TempForecast']['appid']
        response=requests.get(f'https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=current,minutely,daily,alerts&appid={appid}').json()
        for row in response['hourly']:
            self.TempForecast = pd.concat([self.TempForecast, pd.DataFrame({'time': datetime.fromtimestamp(row['dt']), 'temperature': row['temp']-272.15}, index=[0])], ignore_index=True)

        self.TempForecast=self.TempForecast.set_index('time')
        self.TempForecast.index = self.TempForecast.index.tz_localize(self.influxconfig['timezone'])



    #create predicted energy consumption forecast
    def loadEdata(self):
        #get consumption
        self.edata=self.influxclient.query('SELECT integral("value",1h)/ 1000 as consumption, time as time from "W" WHERE "entity_id"=\''+self.influxconfig['energy_demand_sensor']+'\' and time <= now() and time >= now() - 365d GROUP BY time(1h)')['W']
        self.edata.index.name='time'
        self.edata.index = self.edata.index.tz_convert(self.influxconfig['timezone'])
        self.edata = self.edata.asfreq('H', fill_value=0.0).sort_index()
        self.edata['weekday'] = self.edata.index.weekday
        self.edata['hour'] = self.edata.index.hour
        #get temperature data
        tdata=self.influxclient.query('SELECT mean("value") as temperature, time as time from "°C" WHERE "entity_id"=\''+self.influxconfig['outside_temperature_sensor']+'\' and time <= now() and time >= now() - 365d GROUP BY time(1h)')['°C']
        tdata.index.name='time'
        tdata.index = tdata.index.tz_convert(self.influxconfig['timezone'])
        tdata = tdata.asfreq('H', fill_value=15.0).sort_index()
        self.edata = self.edata.join(tdata, how='left')
        self.edata['temperature']=self.edata['temperature'].fillna(0.0)
        #add holiday data
        self.edata['holiday']=0
        for row in self.config['Holiday']:
            self.edata.loc[row['start']:row['end'],"holiday"]=1

    #plot edata if necessary
    def plotEdata(self):
        self.edata
        end_data=datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)+timedelta(days=-1)
        begin_data=self.edata.index[0].replace(hour=0, minute=0, second=0, microsecond=0)
        end_train=(end_data+timedelta(days=-28)).replace(hour=23, minute=59, second=59, microsecond=999)
        end_validation=(end_data+timedelta(days=-14)).replace(hour=23, minute=59, second=59, microsecond=999)
        data = self.edata.loc[begin_data.strftime("%Y/%m/%d, %H:%M:%S"): end_data.strftime("%Y/%m/%d, %H:%M:%S")].copy()
        data_train = data.loc[: end_train.strftime("%Y/%m/%d, %H:%M:%S"), :].copy()
        data_val   = data.loc[end_train.strftime("%Y/%m/%d, %H:%M:%S"):end_validation.strftime("%Y/%m/%d, %H:%M:%S"), :].copy()
        data_test  = data.loc[end_validation.strftime("%Y/%m/%d, %H:%M:%S"):, :].copy()

        print(f"Train dates      : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
        print(f"Validation dates : {data_val.index.min()} --- {data_val.index.max()}  (n={len(data_val)})")
        print(f"Test dates       : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")
        # Time series plot
        # ==============================================================================
        fig, ax = plt.subplots(figsize=(8, 3.5))
        data_train.consumption.plot(ax=ax, label='train', linewidth=1)
        data_val.consumption.plot(ax=ax, label='validation', linewidth=1)
        data_test.consumption.plot(ax=ax, label='test', linewidth=1)
        ax.set_title('Electricity demand')
        ax.legend();
        plt.show()

    #create exogenous variable for eforecast
    def getExogFut(self, temp=0):
        self.ExogFut
        self.TempForecast
        self.ExogFut=pd.DataFrame(pd.date_range((datetime.today().replace(minute=0, second=0, microsecond=0)).strftime("%Y/%m/%d, %H:%M:%S"),(datetime.today().replace(minute=0, second=0, microsecond=0)+timedelta(hours=34)).strftime("%Y/%m/%d, %H:%M:%S"),freq='H'),columns=['time'])
        self.ExogFut['time']=pd.to_datetime(self.ExogFut['time'])
        self.ExogFut=self.ExogFut.set_index('time')
        self.ExogFut = self.ExogFut.tz_localize(self.influxconfig['timezone'])
        self.ExogFut = self.ExogFut.asfreq('H', fill_value=0.0).sort_index()
        self.ExogFut['weekday'] = self.ExogFut.index.weekday
        self.ExogFut['hour'] = self.ExogFut.index.hour
        self.ExogFut['holiday']=0   
        #add for temperature if needed 
        if temp==1:
            self.ExogFut = self.ExogFut.join(self.TempForecast, how='left')

    #forecast energy consumption
    def forecastEdata(self,backtest=0,plot=0):
        self.Eforecast
        end_data=(datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)+timedelta(days=-1)).strftime("%Y/%m/%d, %H:%M:%S")
        begin_data=(self.edata.index[0].replace(hour=0, minute=0, second=0, microsecond=0)).strftime("%Y/%m/%d, %H:%M:%S")
        forecaster = ForecasterAutoreg(
                        regressor     = Ridge(random_state=1286,alpha=27.825594022071257),#Ridge(random_state=1286,alpha=27.825594022071257) or RandomForestRegressor(random_state=123)
                        lags          = [1, 2, 3, 23, 24, 25, 47, 48, 49],
                        transformer_y = StandardScaler()
                    )
        exog = [col for col in self.edata.columns if col.startswith(('weekday', 'hour','holiday'))] #add temperature if needed
        end_forecast=(datetime.today().replace(minute=0, second=0, microsecond=0)+timedelta(hours=-1)).strftime("%Y/%m/%d, %H:%M:%S")
        forecaster.fit(y=self.edata.loc[begin_data:end_data, 'consumption'], exog=self.edata.loc[begin_data:end_data, exog])
        self.Eforecast=forecaster.predict(34,self.edata.loc[:end_forecast, 'consumption'],exog=self.ExogFut)
        if backtest==1:
            end_data=(datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)+timedelta(days=-1)).astimezone(timezone(self.influxconfig['timezone']))
            begin_data=end_data+timedelta(days=-60)
            end_validation=(begin_data+timedelta(days=50)).replace(hour=23, minute=59, second=59, microsecond=999)
            metric, predictions = backtesting_forecaster(
                          forecaster         = forecaster,
                          y                  = self.edata['consumption'],
                          steps              = 33,
                          metric             = 'mean_absolute_error',
                          initial_train_size = len(self.edata.loc[:end_validation]),
                          refit              = False,
                          verbose            = True,
                          show_progress      = True
                    )
            if plot==1:
                fig, ax = plt.subplots(figsize=(8, 3.5))
                self.edata.loc[predictions.index, 'consumption'].plot(ax=ax, linewidth=2, label='real')
                predictions.plot(linewidth=2, label='prediction', ax=ax)
                ax.set_title('Prediction vs real demand')
                ax.legend();
                plt.show()
    

    #create complete dataframe for optimization for the nex day at 16:00 the previous day
    def createOptInput(self):
        curhour=datetime.today().hour
        if curhour<16:
            horizon=24-curhour
        else:
            horizon=48-curhour
        self.Optimization=pd.DataFrame(pd.date_range((datetime.today().replace(hour=curhour,minute=0, second=0, microsecond=0)).strftime("%Y/%m/%d, %H:%M:%S"),(datetime.today().replace(hour=curhour,minute=0, second=0, microsecond=0)+timedelta(hours=horizon)).strftime("%Y/%m/%d, %H:%M:%S"),freq='H'),columns=['time'])
        self.Optimization['time']=pd.to_datetime(self.Optimization['time'])
        self.Optimization=self.Optimization.set_index('time')
        self.Optimization = self.Optimization.tz_localize(self.influxconfig['timezone'])
        self.Optimization = self.Optimization.asfreq('H', fill_value=0.0).sort_index()
        self.Optimization=self.Optimization.join(self.Eforecast, how='left')
        self.Optimization=self.Optimization.rename(columns={'pred': 'Eforecast'})
        self.Optimization['PVForecast']=0.0
        self.Optimization['CostPurchase']=0.0
        self.Optimization['CostFeedback']=0.0
        self.Optimization['CostBatIn']=0.0
        self.Optimization['CostBatOut']=0.0
        for index, row in self.Optimization.iterrows():
            timestamp=index.to_pydatetime()
            hour=timestamp.hour
            if timestamp.date() == datetime.today().date():
                self.Optimization.at[index,'PVForecast']=self.PVForecastToday[hour]/1000.0
                self.Optimization.at[index,'CostPurchase']=self.CostPurchaseToday[hour]
                self.Optimization.at[index,'CostFeedback']=self.CostFeedBackToday[hour]
                self.Optimization.at[index,'CostBatIn']=self.CostPurchaseToday[hour]*self.config['Costs']['bat_loss']
                self.Optimization.at[index,'CostBatOut']=self.CostPurchaseToday[hour]*self.config['Costs']['bat_loss']
            else:
                self.Optimization.at[index,'PVForecast']=self.PVForecastTomorrow[hour]/1000.0
                self.Optimization.at[index,'CostPurchase']=self.CostPurchaseTomorrow[hour]
                self.Optimization.at[index,'CostFeedback']=self.CostFeedBackTomorrow[hour]
                self.Optimization.at[index,'CostBatIn']=self.CostPurchaseTomorrow[hour]*self.config['Costs']['bat_loss']
                self.Optimization.at[index,'CostBatOut']=self.CostPurchaseTomorrow[hour]*self.config['Costs']['bat_loss']
    def priceForecast(self):
        tomorrow=0
        dayaftertomorrow=0
        count1=0
        count2=0
        result = requests.get('https://energie.theoxygent.nl/api/prices.php').json()
        for row in result[1]:
            rowdate=datetime.fromtimestamp(row['x']*100).date() 
            if rowdate == datetime.today().date()+timedelta(days=1):
                tomorrow+=row['y']
                count1+=1
            if rowdate == datetime.today().date()+timedelta(days=2):
                dayaftertomorrow+=row['y']
                count2+=1
        ptom=tomorrow/count1
        pdat=dayaftertomorrow/count2
        self.dayondayprice=pdat/ptom
        
    #fixSOC makes sure SOC stays relatively similar over a run (influence by SOCslack in conf), fixSOCt and SOCtarget assume SOC needs to be at a certain target. smartSOC uses forecast for prices to determine high or low SOC at the end of run    
    def createOptimization(self,fixSOC=0,fixSOCt=0, SOCtarget=0.0,smartSOC=0):
        #constants
        n=range(len(self.Optimization.index))
        cPurchase=list(self.Optimization['CostPurchase'])
        cFeedback=list(self.Optimization['CostFeedback'])
        cBatIn=list(self.Optimization['CostBatIn'])
        cBatOut=list(self.Optimization['CostBatOut'])
        PV = list(self.Optimization['PVForecast'])
        PL = list(self.Optimization['Eforecast'])
        PGridIn=[]
        PGridOut=[]
        Pbattin=[]
        Pbattout=[]
        SOCt=[]
        SOCkWh=[]
        BattCapacity=self.config['Battery']['maxcapacity']
        eta=self.config['Battery']['eta']
        minSOC=self.config['Battery']['minSOC']*BattCapacity
        maxSOC=self.config['Battery']['maxSOC']*BattCapacity
        #get SOC from HA
        url = "http://"+self.config['HA']['IP']+":"+self.config['HA']['port']+"/api/states/sensor."+self.config['Battery']['entity_id']
        headers = {
        "Authorization": "Bearer "+self.config['HA']['accesstoken'] ,
        'Content-type' : 'application/json' 
        }
        
        response = requests.get(url, headers=headers)
        SOC=(float(response.json()['state'])/100.0)*BattCapacity
        SOCprev=SOC
        m=Model()
        #PgridIn
        x= [m.add_var(ub=self.config['Optimization']['gridinmax']) for i in n]
        #PgridOut
        y= [m.add_var(ub=self.config['Optimization']['gridoutmax']) for i in n]
        #Pbattin
        w= [m.add_var(ub=self.config['Battery']['maxcharge']) for i in n]
        #Pbattout
        z= [m.add_var(ub=self.config['Battery']['maxdischarge']) for i in n]
        #help vars
        d= [m.add_var(var_type=BINARY) for i in n]
        e= [m.add_var(var_type=BINARY) for i in n]

        #objective
        m.objective = minimize(xsum(cPurchase[i]*x[i]+cFeedback[i]*y[i]+cBatIn[i]*w[i]+cBatOut[i]*z[i] for i in n))

        #Constraints
        for i in n:
            m += PV[i]+x[i]+z[i]-PL[i]-y[i]-w[i] == 0
            m += x[i]-(self.config['Optimization']['gridinmax']*d[i]) <= 0
            m += y[i]-(self.config['Optimization']['gridoutmax']*(1-d[i])) <= 0
            m += w[i]-(self.config['Battery']['maxcharge']*e[i]) <= 0
            m += z[i]-(self.config['Battery']['maxdischarge']*(1-e[i])) <= 0
            m += SOC + xsum(w[j]*eta for j in range(i) ) - xsum(z[j]/eta for j in range(i+1)) <= maxSOC
            m += SOC + xsum(w[j]*eta for j in range(i) ) - xsum(z[j]/eta for j in range(i+1)) >= minSOC

        if fixSOCt>0 and SOCtarget>0.0:
            m += SOC + xsum(w[i]*eta for i in range(fixSOCt+1) ) - xsum(z[i]/eta for i in range(fixSOCt+1)) <= SOCtarget*BattCapacity + 0.025*BattCapacity
            m += SOC + xsum(w[i]*eta for i in range(fixSOCt+1) ) - xsum(z[i]/eta for i in range(fixSOCt+1)) >= SOCtarget*BattCapacity - 0.025*BattCapacity      

        else:
            if smartSOC==1:
                if self.dayondayprice > 1.0:
                    m += SOC + xsum(w[i]*eta for i in n ) - xsum(z[i]/eta for i in n) >= self.config['Optimization']['HighSOC']*BattCapacity 
                    endSOC=self.config['Optimization']['HighSOC']*BattCapacity 
                else:
                    m += SOC + xsum(w[i]*eta for i in n ) - xsum(z[i]/eta for i in n) >= self.config['Optimization']['LowSOC']*BattCapacity 
                    endSOC=self.config['Optimization']['LowSOC']*BattCapacity 

            if fixSOC==1 and smartSOC!=1:
                endSOC=self.config['Optimization']['MinEndSOC']*BattCapacity 
                if self.config['Optimization']['SOCSlack']>0.0:
                    m += SOC + xsum(w[i]*eta for i in n ) - xsum(z[i]/eta for i in n) <= endSOC + self.config['Optimization']['SOCSlack']*BattCapacity
                    m += SOC + xsum(w[i]*eta for i in n ) - xsum(z[i]/eta for i in n) >= endSOC - self.config['Optimization']['SOCSlack']*BattCapacity
                else:
                    m += SOC + xsum(w[i]*eta for i in n ) - xsum(z[i]/eta for i in n) == endSOC
            
            #find max amount to be taken from GRID if settings don't allow unlmited GRID use
            if self.config['Optimization']['GRIDSlack']!=100.0:
                SOCsim=SOC
                DifFromMinSOC=SOC-minSOC
                GRIDallowance=0.0
                

                for index, row in self.Optimization.iterrows():
                    BATout=0.0
                    BATin=0.0
                    prevDifFromMinSOC=DifFromMinSOC
                    if row['Eforecast']>row['PVForecast']:
                        BATout=(row['Eforecast']-row['PVForecast'])/eta
                    if row['PVForecast']>row['Eforecast']:
                        BATin=row['PVForecast']-row['Eforecast']*eta
                    if (SOCsim + BATin) > BattCapacity:
                        SOCsim=BattCapacity
                    else:
                        SOCsim=SOCsim+BATin-BATout
                    DifFromMinSOC=SOCsim-minSOC
                    if DifFromMinSOC<0.0 and DifFromMinSOC<prevDifFromMinSOC:
                        GRIDallowance+=(prevDifFromMinSOC-DifFromMinSOC)*-1.0
                GRIDallowance=GRIDallowance+((endSOC-SOCsim) if SOCsim<endSOC else 0.0)
                m += xsum(x[i] for i in n) <= (1.0+self.config['Optimization']['GRIDSlack'])*(GRIDallowance/eta)          

        m.max_gap = 0.05
        status = m.optimize(max_seconds=60)
        if status == OptimizationStatus.OPTIMAL:
            self.calculatedat=datetime.now()
            print('optimal solution cost {} found'.format(m.objective_value))
        elif status == OptimizationStatus.FEASIBLE:
            print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
        elif status == OptimizationStatus.NO_SOLUTION_FOUND:
            print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))

        for i in n:
            PGridIn.append(x[i].x)
            PGridOut.append(y[i].x)
            Pbattin.append(w[i].x)
            Pbattout.append(z[i].x)
            SOCkWh.append((SOCprev+w[i].x*eta-z[i].x/eta))
            SOCprev=(SOCprev+w[i].x*eta-z[i].x/eta)
            SOCt.append((SOCprev/BattCapacity)*100.0)
        self.Optimization['PGridIn']=PGridIn
        self.Optimization['PGridOut']=PGridOut
        self.Optimization['PBattIn']=Pbattin
        self.Optimization['PBattOut']=Pbattout
        self.Optimization['GridSetPoint']=self.Optimization['PGridIn']-self.Optimization['PGridOut']
        self.Optimization['Balance']=round(self.Optimization['PVForecast']+self.Optimization['PGridIn']+self.Optimization['PBattOut']-self.Optimization['Eforecast']-self.Optimization['PGridOut']-self.Optimization['PBattIn'],3)
        self.Optimization['SOC']=SOCt
        self.Optimization['SOCkWh']=SOCkWh
        self.Optimization['hour']=self.Optimization.index
        self.Optimization['hour']=self.Optimization['hour'].dt.hour
        self.Optimization['Cost']=self.Optimization['CostPurchase']*self.Optimization['PGridIn']+self.Optimization['CostFeedback']*self.Optimization['PGridOut']
        self.Optimization['Cumcost']=self.Optimization['Cost'].cumsum()
        self.Optimization['UnoptimizedCost']=self.Optimization['CostPurchase']*self.Optimization['Eforecast']+self.Optimization['CostFeedback']*self.Optimization['PVForecast']
        self.Optimization['CumUnoptimizedCost']=self.Optimization['UnoptimizedCost'].cumsum()

    def plotOptimization(self,plot=0,show=0):
        df=self.Optimization
        if plot==1 or plot==0:
            options = {
                "title" : "Calculated at: " + self.calculatedat.strftime('%Y-%m-%d %H:%M'),
                "haxis" : {
                    "values": "hour",
                    "title": "hour from start"
                },
                "vaxis" :[{
                    "title" : "kWh"
                    }
                ],
                "series":[ {"column" : "PGridIn",
                    "label": "GRID In",
                    "type": "stacked",
                    "color": '#00bfff'
                    },
                    {"column": "PVForecast",
                        "label": "PV",
                        "type": "stacked",
                        "color": 'green'
                        },
                    {"column": "PBattOut",
                        "label": "Batt Out",
                        "type": "stacked",
                        "color": 'red'
                    },
                    {"column": "Eforecast",
                    "label": "Consumption",
                    "type": "stacked",
                    "color": "#f1a603",
                    "negative": 1
                    },
                    {"column": "PGridOut",
                    "label": "GRID out",
                    "type": "stacked",
                    "color": '#0080ff',
                    "negative": 1
                    },
                    {"column": "PBattIn",
                    "label": "Batt In",
                    "type": "stacked",
                    "color": '#ff8000',
                    "negative": 1
                    },
                    ]
            }        
            fig, axis = plt.subplots(figsize=(7, 5))  # , sharex= True)
            ind = np.arange(len(df.index))
            stacked_plus= np.zeros( shape=(len(df.index)) )
            stacked_neg = np.zeros( shape=(len(df.index)) )
            for serie in options["series"]:
                data_array = df[serie['column']]
                type = serie["type"]
                color = serie["color"]
                if "label" in serie:
                    label = serie["label"]
                else:
                    label = serie["column"].capitalize()
                if type=="bar":
                    axis.bar(ind, data_array, label=label, color=color)
                elif type=="line":
                    linestyle = serie["linestyle"]
                    axis.plot(ind, data_array, label=label, linestyle=linestyle, color=color)
                else: #stacked bar
                    if "negative" in serie:
                        data_array = np.negative(data_array)
                    sum = np.sum(data_array)
                    if sum > 0:
                        axis.bar(ind, data_array, bottom=stacked_plus, label=label, color=color)
                        stacked_plus = stacked_plus + data_array
                    elif sum < 0:
                        axis.bar(ind, data_array, bottom=stacked_neg, label=label, color=color)
                        stacked_neg = stacked_neg + data_array

            xlabels = df[options["haxis"]["values"]].values.tolist()
            axis.set_xticks(ind, labels=xlabels)
            axis.set_xlabel(options["haxis"]["title"])
            if len(df.index)>15:
                axis.xaxis.set_major_locator(ticker.MultipleLocator(2))
                axis.xaxis.set_minor_locator(ticker.MultipleLocator(1))

            ylim = math.ceil(max(np.max(stacked_plus), - np.min(stacked_neg)))
            if np.min(stacked_neg) < 0:
                axis.set_ylim([-ylim, ylim])
            else:
                axis.set_ylim([0,ylim])
            axis.set_ylabel(options["vaxis"][0]["title"])

            axis.set_title(options["title"])
            # Shrink current axis by 20%
            box = axis.get_position()
            axis.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            #axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axis.legend(loc='upper left', bbox_to_anchor=(1.05, 1.00))
            fig1= BytesIO()
            fig.savefig(fig1, format="png")
            fig1.seek(0)
            return fig1
        if plot==2 or plot==0:
            #2nd graph
            options = {
                "title" : "Calculated at: " + self.calculatedat.strftime('%Y-%m-%d %H:%M'),
                "haxis" : {
                    "values": "hour",
                    "title": "hour from start"
                },
                "vaxis" :[{
                    "title" : "%SOC"
                    }
                ],
                "series":[ {"column" : "SOC",
                    "label": "SOC",
                    "type": "line",
                    "color": 'red',
                    "linestyle": 'solid' 
                    }, 
                    ]
            }        
            fig, axis = plt.subplots(figsize=(7, 5))  # , sharex= True)
            ind = np.arange(len(df.index))
            stacked_plus= np.zeros( shape=(len(df.index)) )
            stacked_neg = np.zeros( shape=(len(df.index)) )
            for serie in options["series"]:
                data_array = df[serie['column']]
                type = serie["type"]
                color = serie["color"]
                if "label" in serie:
                    label = serie["label"]
                else:
                    label = serie["column"].capitalize()
                if type=="bar":
                    axis.bar(ind, data_array, label=label, color=color)
                elif type=="line":
                    linestyle = serie["linestyle"]
                    axis.plot(ind, data_array, label=label, linestyle=linestyle, color=color)
                else: #stacked bar
                    if "negative" in serie:
                        data_array = np.negative(data_array)
                    sum = np.sum(data_array)
                    if sum > 0:
                        axis.bar(ind, data_array, bottom=stacked_plus, label=label, color=color)
                        stacked_plus = stacked_plus + data_array
                    elif sum < 0:
                        axis.bar(ind, data_array, bottom=stacked_neg, label=label, color=color)
                        stacked_neg = stacked_neg + data_array

            xlabels = df[options["haxis"]["values"]].values.tolist()
            axis.set_xticks(ind, labels=xlabels)
            axis.set_xlabel(options["haxis"]["title"])
            if len(df.index)>15:
                axis.xaxis.set_major_locator(ticker.MultipleLocator(2))
                axis.xaxis.set_minor_locator(ticker.MultipleLocator(1))

            axis.set_ylim([0,101.0])
            axis.set_ylabel(options["vaxis"][0]["title"])

            axis.set_title(options["title"])
            # Shrink current axis by 20%
            box = axis.get_position()
            axis.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            #axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axis.legend(loc='upper left', bbox_to_anchor=(1.05, 1.00))
            fig2= BytesIO()
            fig.savefig(fig2, format="png")
            fig2.seek(0)
            return fig2
        if show==1:
            plt.show()

    def getActuals(self):
        #consumption
        self.Optimization=self.Optimization.drop(columns=['Consumption','PVreal','GRID','SOCact'],errors='ignore')
        consumption=self.influxclient.query('SELECT integral("value",1h)/ 1000 as Consumption, time as time from "W" WHERE "entity_id"=\''+self.config['Sensors']['Consumption']+'\' and time <= now() and time >= now() - 2d GROUP BY time(1h)')['W']
        consumption.index.name='time'
        consumption.index = consumption.index.tz_convert(self.influxconfig['timezone'])
        consumption = consumption.asfreq('H', fill_value=0.0).sort_index()
        self.Optimization=self.Optimization.join(consumption, how='left')
        #PV
        PV=self.influxclient.query('SELECT integral("value",1h)/ 1000 as PVreal, time as time from "W" WHERE "entity_id"=\''+self.config['Sensors']['PV']+'\' and time <= now() and time >= now() - 2d GROUP BY time(1h)')['W']
        PV.index.name='time'
        PV.index = PV.index.tz_convert(self.influxconfig['timezone'])
        PV = PV.asfreq('H', fill_value=0.0).sort_index()
        self.Optimization=self.Optimization.join(PV, how='left')
        #GRID
        GRID=self.influxclient.query('SELECT integral("value",1h)/ 1000 as GRID, time as time from "W" WHERE "entity_id"=\''+self.config['Sensors']['GRID']+'\' and time <= now() and time >= now() - 2d GROUP BY time(1h)')['W']
        GRID.index.name='time'
        GRID.index = GRID.index.tz_convert(self.influxconfig['timezone'])
        GRID = GRID.asfreq('H', fill_value=0.0).sort_index()
        self.Optimization=self.Optimization.join(GRID, how='left')
        #SOC
        SOC=self.influxclient.query('SELECT mean("value") as SOCact, time as time from "%" WHERE "entity_id"=\''+self.config['Sensors']['SOC']+'\' and time <= now() and time >= now() - 2d GROUP BY time(1h)')['%']
        SOC.index.name='time'
        SOC.index = SOC.index.tz_convert(self.influxconfig['timezone'])
        SOC = SOC.asfreq('H', fill_value=0.0).sort_index()
        self.Optimization=self.Optimization.join(SOC, how='left')  
        self.Optimization['Consumption']= self.Optimization['Consumption'].fillna(0.0) 
        self.Optimization['PVreal']= self.Optimization['PVreal'].fillna(0.0) 
        self.Optimization['GRID']= self.Optimization['GRID'].fillna(0.0) 
        self.Optimization['SOCact']= self.Optimization['SOCact'].fillna(0.0) 
        self.Optimization['CostReal']=np.where(self.Optimization['GRID']>0.0, self.Optimization['CostPurchase']*self.Optimization['GRID'], self.Optimization['CostFeedback']*self.Optimization['GRID'])
        self.Optimization['CostRealCum']=self.Optimization['CostRealCum'].cumsum()




     




