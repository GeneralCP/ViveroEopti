# ViveroEopti
A Home Assistant addon for optimizing home battery use in the Netherlands. The addon presents an API which, based of a large amount of input data, will present a minimum cost solution for using a home battery storage in combitation with solar and a dynamic tariffs contract. 

# Working principle
The addon itself is a RESTapi which runs in the background. So far the rest API has 4 endpoints

* /calculate , this wil recalculate the full optimization starting from the current time. Calculations always run as far as Entso-e pricing for the Netherlands is available. Before 4 pm this is till the end of today. After 4 pm this is till the end of tomorrow
* /plot1, shows the optimization in a plotted graph
* /plot2, shows the calculated SOC of the battery for the hours of the optimization
* /GRIDSetpoint, the calculated GRID setpoint for the current hour. This should then be forwarded to your battery system using an automation in Home Assistant (NodeRed for example)

# Optimization principle
Optimization is reached in a number of steps. The goal is to have the lowest possible cost based on a Solar PV forecast, a forecast of energy consumption and the dynamic tariffs gathered from the ENTSO-E platform adjusted for your specific supplier. This is not a trading algorithm which means that the intended use is to have the most economic use of your own energy (PV and Battery storage) and when PV is insufficient, buy from the GRID at the most economic time. Excess PV is sold at the most economic time as well (within the possibilities of your system). 

The steps used for optimizing are:

* Calculate PV forecast based on the parameters of your PV installation (setup in config file)
* Get prices from the ENTSO-E platform and convert them to prices for your own contract (based on tariffs from the config file)
* Gather historical date on the electricity consumption in your home. Also get exogenous variables: temperature, day of the week (weekend vs weekday) and holiday (less power consumption)
* This historical data is used to create an hourly energy forecast for the next couple of days
* A Linear optimization problem is used to find the most optimal solution for using your battery. The model find the minimum cost within several fixed parameters:
* usage of GRID is minimized and is only used when no PV is available
* based on a forecast of the prices of the day after tomorrow the battery is either left to discharge to a minimum SOC (because electricity will be cheaper the next day) or to a maximum SOC (because electricity is more expensive the next day)

The output of the optimization is a table for the next 32 hours that lists all variables of the optimization. The most important variable is the GRID Setpoint which is used to determine if you should be feeding in or back from the GRID in a certain hour. This is based on the common 'GRID Setpoint' used in most ESS system (e.g. Victron).

# Requirements

* Home Assistant OS on a network within your home
* Addons on Home Assistant: NodeRed, Influxdb
* Required sensor input from Home Assistant: A sensor showing totall energy consumption of your house in Watts, A sensor for outside temperature of your house and a sensor showing SOC of the battery
* Information about your PV system (watt peak, azimuth, declination etc.)
* Entso-e API key (can be attained for free from contact with them)

# Installation

* To install add the E-opti Add-on repository in the Home Assistant store, follow these steps: https://www.home-assistant.io/common-tasks/os/#installing-third-party-add-ons
This will be: Configuration > Add-ons & Backups open the add-on store > Add the URL of the repository and then press "Add".
Look for the Vivero e-optimization Add-on tab and when inside the Add-on click on `install`.
Be patient, the installation may take some time depending on your hardware.
After installation set the configuration parameters before starting your addon. Addon/API should then be running on the http://{HA IP address}:8000

* Make sure to specify configuration parameters in your addon. See the config.md file for a description of all the variables. 

* The addon can also be run stand-alone as a python application. This is not recommended since it will still require a connection to a Home assistant instance (to get the data from). To run the app stand-alone. Download the whole repository to a folder on the PC you would like to run the app on. Make sure python 3 and pip is installed. In the folder run "pip install -r requirements.txt" to install all required python modules. After this you can run "uvicorn main:app --host 0.0.0.0 --port 8000" from the same folder. It will start the application on the server. The API will be available at the IP address of the server

* Install the NodeRed flow on your cerbo device's NodeRed installation make sure to change the IP address to the IP of your HA installation in the HTTP request nodes

# Usage

* Once the API is up and running (either as HA add-on or as a standalone API) the NodeRed flow in your cerbo will periodically:
    * daily at 4 pm (once new day-ahead prices are known) it will recalculate the whole uptimization until the end of the next day
    * hourly at 1 minute past the whole hour it will update the GRID setpoint in your ESS to the new required setpoint

* To check if the API is running and what the output is, you can check either of the following endpoints
    * http://{HA IP address}:8000 -> shows if the API is running
    * http://{HA IP address}:8000/calculate -> runs the calculation again from the current time
    * http://{HA IP address}:8000/plot/1 -> shows .png showing the output of the optimization. This can also be used to set as a picture in your HA dashboard to check the optimization results
    * http://{HA IP address}:8000/plot/2 -> shows .png showing a graph with the predicted battery SOC
    * http://{HA IP address}:8000/GRIDSetpoint -> get GRID setpoint for current hour to supply to your battery system (used in NodeRed flow for Victron)
    * http://{HA IP address}:8000/forecast/{entity} -> used to get reply of forecasted values for a certain entity. This endpoint can be used to create a sensor in HA that can be shown in an APEX charts graph. Available entities:
         * PVForecast
         * CostPurchase
         * CostFeedback
         * CostBatIn
         * CostBatOut
         * Eforecast
         * PGridIn
         * PGridOut
         * PBattIn
         * PBattOut
         * GridSetpoint
         * Balance
         * SOC
         * SOCkWH
         * hour
         * Cost -> optimized cost prediction
         * CumCost
         * UnoptimizedCost
         * CumUnoptimizedCost
    * http://{HA IP address}:8000/actuals/{entity} -> retries actuals since start of optimization and fills the entities. Returns JSON with selected entity. Available entities:
         *  Consumption
         * PVreal
         * GRID
         * SOCact
         * CostReal
         * CostRealCum



