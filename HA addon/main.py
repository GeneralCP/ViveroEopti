from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
from functions import Eoptimization
import uvicorn
import yaml, json
from datetime import datetime,timedelta,date

#Import configuration file
config='json'
if config=='json':
    with open("./options.json") as stream:
        try:
            config = json.load(stream)
            print(config)
        except ValueError:
            print('Loading JSON has failed')

else:
    with open("config.yaml", "r") as stream:
        try:
            config=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

Eopti=Eoptimization(config)
app = FastAPI()

@app.get("/")
def root():
    return [{"Status": "Vivero E-optimization is online"},
            {"Error Codes": "None"},
    ]

@app.get("/calculate")
def root():
    try:
        Eopti.loadPVForecast()
    except:
        return {"status": "Error trying to load PV Forecast"}
    
    try:
        Eopti.loadPrices()
    except:
        return {"status": "Error trying to load prices from entso-e platform"}
    
    try:
        Eopti.getTempForecast()
    except:
        return {"status": "Error trying to get temperature forecast from OpenWeatherMap"}   

    try:
        Eopti.loadEdata()
    except:
        return {"status": "Error trying to load historical energy data from HA sensor and influxDB"}     
    
    try:
        Eopti.getExogFut()
    except:
        return {"status": "Error trying to get future external variables"}  

    try:
        Eopti.forecastEdata(backtest=0,plot=0)
    except:
        return {"status": "Error trying to create forecast data for energy"} 
    
    try:
        Eopti.createOptInput()
    except:
        return {"status": "Error trying to greate dataframe from previous input"} 

    try:
        Eopti.priceForecast()
    except:
        return {"status": "Error trying to get price forecast >24h ahead from "} 
    
    try:
        Eopti.createOptimization(smartSOC=1)
    except:
        return {"status": "Error trying to greate dataframe from previous input"}     
    
    return Response(Eopti.Optimization.to_json(orient="index"), media_type="application/json")

@app.get("/plot1")
def root():
    try:
        buf=Eopti.plotOptimization(plot=1)
    except:
        return {"status": "Error trying to greate dataframe from previous input"} 
    
    return StreamingResponse(buf, media_type="image/png")

@app.get("/plot2")
def root():
    try:
        buf=Eopti.plotOptimization(plot=2)
    except:
        return {"status": "Error trying to greate dataframe from previous input"} 
    
    return StreamingResponse(buf, media_type="image/png")

@app.get("/GRIDSetpoint")
def root():
    if len(Eopti.Optimization.index)>0:
        df=Eopti.Optimization
        df=df[(df.index.date == date.today()) & (df.index.hour == datetime.now().hour) ]
        print(df)
        return {"status": "success", "data": {'GridSetPoint': df['GridSetPoint'].values[0]*1000.0},"message": "null"}
                

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)