# HA

IP: IP address of your local home assistant instance
port: Port of your local HA instance. Usually 8123
accesstoken: Accestoken for connection to HA. Link(https://community.home-assistant.io/t/how-to-get-long-lived-access-token/162159)

# PVinstallations

For each installation or set of PV panels with a different orientation, create a new set
The PV forecast will be calculated for each set. The back-end used is forecast.solar. You can query 12 times per hour with the same IP-address so without a subscription do not supply more than 12 installations

latitude: latitude of the installation
longitude: longitude of the installation
declination: declination of the set of PV panels
azimuth: azimuth of the PV panels
kwp: kilo watt peak total of the set op PV panels

# Battery
    
maxcapacity: max capacity of battery 
maxdischarge: maximum discharge power to use for optimization in kW
maxcharge: maximum charge power to use for the optimization in kW    
minSOC: min SOC you want to use for optimization. OPtimization will never go above or below this SOC
maxSOC: max SOC you want to use for optimization. OPtimization will never go above or below this SOC
eta: charge and discharge efficiency for the battery. 
entity_id: entity id for getting SOC in HA

# Costs

energy_tax: the tax per kWh in including BTW
delivery_cost: delivery cost per kWh of your dynamic energy supplier
feedback_rebate: rebate cost per kWh of your dynamic energy supplier
btw: VAT percentage in the Netherlands
saldering_percentage: specific for the Netherlands. Currently 100% (1.0) might go down when laws change in the future
country_code: 'NL'
api_key: Entso-e API key. You can get your own API key by following the instructions from this link(https://thesmartinsights.com/how-to-query-data-from-the-entso-e-transparency-platform-using-python/#:~:text=Request%20an%20API%20key%20by,%E2%80%9CWeb%20API%20Security%20Token%E2%80%9D.)
tz: timezone used for Entso-e prices. For Netherlands use Europe/Amsterdam
bat_loss: percentage to calculate as lost cost when using battery. Only use when you want to cost in battery use. The eta value under battery already calculates in losses in efficiency

# Eprediction
energy_demand_sensor: name of the sensor in HA that shows total energy consumption of the house
outside_temperature_sensor: name of the sensor in HA that registers the outside temperature at your location. This temperature is used to increase forecast accuracy (i.e. when you use AC at high temps or a heatpump at low temps)
timezome: 'Europe/Amsterdam'
influxdb_ip: IP adrress of your influx DB (usually same as HA instance)
influxdb_port: port of influxdb instance
influxdb_username: 
influxdb_password:
influxdb_database: name of the influxdb database. usually homeassistant

# Holiday
Specify each holiday in the past. Used for increasing accuray of the e-forecast. 
list each holiday seperately

start: start of holiday in 'YYYY/MM/DD'
end: end of holiday in same format

#TempForecast
lat: latitude of your home location
lon: longitude of your home location
appid: openweathermap appid. link(https://openweathermap.org/appid)

#Optimization

gridinmax: maximum amount in kW that can be taken from GRID
gridoutmat: maximum amount in kW that can be supplied to GRID
GRIDSlack: This percentage of the daily load is allowed to be taken from the GRID instead of solar. Set this to 100.0 to allow unlimited GRID usage. If this is set to 0. Model will only take from GRID when solar or battery is insufficient. Minimum is 0.005 or model might not find a feasible solution
SOCSlack: When optimization is run with fixSOC option. This option allows some slack. Settings this to 0.1 allows end SOC to be 10% lower or higher.
MinEndSOC: MIN ending SOC. Prevents SOC going below this state at the end of optimization run. Is ignored when option smartSOC is used
HighSOC:  Smart endSOC. if forecast dynamic prices are higher for the day after tomorrow. use HighSOC else use LOWSOC.
LowSOC: Smart endSOC. if forecast dynamic prices are higher for the day after tomorrow. use HighSOC else use LOWSOC.

#Sensors
The sensors are used for the actuals endpoint. Data is retrieved from HA via influxdb

SOC: SOC sensor
PV: ac power sensor of PV installation
Consumption: sensor which shows total consumption of the house
GRID: sensor showing GRID point
