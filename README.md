# KP_timber_trade

This repo contains the scripts necessary for a paper studying the impact of the Kyoto Protocol on the structure of the global timber trade network and the competition graph derived therefrom. The paper has been published in [*Renewable and Sustainable Energy Reviews*](https://www.sciencedirect.com/science/article/pii/S136403212300727X). Data comes from FAOstat (the file name is: Forestry_Trade_Flows_E_All_Data.csv) and can be downloaded freely from them.

To replicate this project, simply run:

1. KP_timber_trade_network.py (this runs the network analyses but calls the script competition_graph.py from my [Competition Graph repo](https://github.com/cat-astrophic/competition_graph))
2. KP_timber_trade.R (this runs the econometric analyses)
3. KP_timber_trade_results_tables.py (this creates nicely formatted LaTeX tables for the paper)

*Be sure to update relevant working directory info at the beginning of each script!*

## Citation

### APA:

Cary, M. (2023). Climate policy boosts trade competitiveness: Evidence from timber trade networks. Renewable and Sustainable Energy Reviews, 188, 113869.

### MLA:

Cary, Michael. "Climate policy boosts trade competitiveness: Evidence from timber trade networks." Renewable and Sustainable Energy Reviews 188 (2023): 113869.

### Bibtex:

@article{cary2023climate,
&nbsp;&nbsp;&nbsp;&nbsp;title={Climate policy boosts trade competitiveness: Evidence from timber trade networks},
&nbsp;&nbsp;&nbsp;&nbsp;author={Cary, Michael},
&nbsp;&nbsp;&nbsp;&nbsp;journal={Renewable and Sustainable Energy Reviews},
&nbsp;&nbsp;&nbsp;&nbsp;volume={188},
&nbsp;&nbsp;&nbsp;&nbsp;pages={113869},
year={2023},
&nbsp;&nbsp;&nbsp;&nbsp;publisher={Elsevier}
}

