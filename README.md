# ViveroEopti
A Home Assistant addon for optimizing home battery use in the Netherlands. The addon presents an API which, based of a large amount of input data, will present a minimum cost solution for using a home battery storage in combitation with solar and a dynamic tariffs contract. 

# Working principle
The addon itself is a RESTapi which runs in the background. So far the rest API has 4 endpoints

* /calculate , this wil recalculate the full optimization starting from the current time. Calculations always run as far as Entso-e pricing for the Netherlands is available. Before 4 pm this is till the end of today. After 4 pm this is till the end of tomorrow
* /plot1, shows the optimization in a plotted graph
* /plot2, shows the calculated SOC of the battery for the hours of the optimization
* /GRIDSetpoint, the calculated GRID setpoint for the current hour. This should then be forwarded to your battery system using an automation in Home Assistant (NodeRed for example) 
