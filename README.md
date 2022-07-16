# KP_timber_trade

This repo contains the scripts necessary for a paper looking at the impact of the Kyoto Protocol on the structure of the global timber trade network and the competition graph derived therefrom. Data comes from FAOstat (the file name is: Forestry_Trade_Flows_E_All_Data.csv) and can be downloaded freely from them.

To replicate this project, simply run:

1. KP_timber_trade_network.py (this runs the network analyses but must call the script competition_graph.py from my (Competition Graph repo)[https://github.com/cat-astrophic/competition_graph])
2. KP_timber_trade.R (this runs the econometric analyses)
3. KP_timber_trade_results_tables.py (this creates nicely formatted LaTeX tables for the paper)

*Be sure to update relevant working directory info at the beginning of each script!*

The paper is currently under review at *Renewable and Sustainable Energy Reviews*.
