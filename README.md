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
