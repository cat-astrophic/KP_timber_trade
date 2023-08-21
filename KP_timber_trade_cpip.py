# This script performs an intertemporal core-periphery analysis on the international trade network for timber products

# Importing required modules

import pandas as pd
import numpy as np
import networkx as nx
import competition_graph as cg
from cpip import cpip
from matplotlib import pyplot as plt

# Directroy info

direc = 'D:/KP_timber_trade/'

# Loading the data

data = pd.read_csv(direc + 'data/Forestry_Trade_Flows_E_All_Data.csv', encoding = 'latin-1')

# Data prep

# Replace all NaN with 0

data = data.fillna(0)

# Create a list of all nations represented in the raw trade data

nations = list(data['Reporter Countries'].unique()) + list(data['Partner Countries'].unique())
nations = pd.Series(nations)
nations = nations.unique()
nations = nations[:173]

# A list of the annual trade data variables

netvars = ['Y' + str(y) for y in range(1997,2018)]

# A list of the different traded items included in the raw data

items = list(data.Item.unique())

# Data storage

export_results = []
import_results = []
trade_results = []
comp_results = []

# Running cpip

item = items[0]

# Subset data for the desired category

subdata = data[data.Item == item]
subdata = subdata[subdata.Element == 'Export Value']

for nv in netvars:
    
    # Initialize a trade matrix
    
    M = np.zeros((len(nations),len(nations)))
    
    for i in range(len(nations)):
        
        # Visualizing progress
        
        print('Year ' + str(nv[1:]) + ' of 2017 :: Nation ' + str(i+1) + ' of ' + str(len(nations)))
        
        for j in range(len(nations)):
            
            # Subset for nations i and j
            
            tmp = subdata[subdata['Reporter Countries'] == nations[i]].reset_index(drop = True)
            tmp = tmp[tmp['Partner Countries'] == nations[j]].reset_index(drop = True)
            
            # Extract values
            
            if len(tmp) > 0:
                
                M[i][j] = tmp[nv][0]
                
    # Create the trade network
    
    N = nx.DiGraph(M)
    
    # Create the competition graph
    
    A = cg.competition_graph(M)
    A = A - np.eye(len(A)) * np.diag(A)
    G = nx.Graph(A)
    
    # Running cpip for the trade network - exports
    
    exports_core = cpip(np.matrix(M), theta = 1.1, psi = 1)
    exports_core = [nations[int(x[1:])] for x in exports_core]
    
    # Running cpip for the trade network - imports
    
    imports_core = cpip(np.transpose(np.matrix(M)), theta = 1.1, psi = 1)
    imports_core = [nations[int(x[1:])] for x in imports_core]
    
    # Dual-trade-core members
    
    trade_core = [x for x in exports_core if x in imports_core]
    
    # Running cpip for the competition graph
    
    comp_core = cpip(np.matrix(A), theta = 1, psi = 1)
    comp_core = [nations[int(x[1:])] for x in comp_core]
    
    # Storing data
    
    export_results.append(exports_core)
    import_results.append(imports_core)
    trade_results.append(trade_core)
    comp_results.append(comp_core)

# Saving these results

yr = pd.Series([y for y in range(1997,2018)], name = 'Year')
er = pd.Series(export_results, name = 'Exports')
ir = pd.Series(import_results, name = 'Imports')
tr = pd.Series(trade_results, name = 'Trade')
cr = pd.Series(comp_results, name = 'Competition')
core_df = pd.concat([yr, er, ir, tr, cr], axis = 1)
core_df.to_csv(direc + 'results/cores.csv', index = False)

# Plotting the time series of core sizes for the trade network and its derived competition graph

plot_df = pd.concat([pd.Series([i for i in range(1997,2018)], name = 'Year'), pd.Series([len(c) for c in core_df.Trade], name = 'Trade'), pd.Series([len(c) for c in core_df.Competition], name = 'Competition')], axis = 1)

plt.figure(figsize = (6, 6), dpi = 1000)
plt.scatter(plot_df.Year, plot_df.Trade, color = 'maroon')
plt.scatter(plot_df.Year, plot_df.Competition, color = 'orange')
plt.plot(plot_df.Year, plot_df.Trade, color = 'maroon')
plt.plot(plot_df.Year, plot_df.Competition, color = 'orange')
plt.title("Core Size over Time for the Timber Trade Network and its Competition Graph")
plt.xlabel('Year')
plt.ylabel('Core Size')
plt.axvline(x = 2007.5, color = 'black', linestyle = 'dotted')
plt.yticks([0, 10, 20, 30, 40, 50], [0, 10, 20, 30, 40, 50])
plt.xticks([1997, 2001, 2005, 2009, 2013, 2017], [1997, 2001, 2005, 2009, 2013, 2017]) 
plt.legend(['Trade Network', 'Competition Graph'], bbox_to_anchor = (1.4321, 0.666))
plt.savefig(direc + 'figures/core_sizes.png', bbox_inches = 'tight', dpi = 1000)

