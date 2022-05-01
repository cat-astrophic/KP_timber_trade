# This script analyzes the impact of the Kyoto Protocol on the international trade network for timber products

# Importing required modules

import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import competition_graph as cg

# Directroy info

username = ''
direc = 'C:/Users/' + username + '/Documents/Data/KP_timber_trade/'

# Loading the data

data = pd.read_csv(direc + 'data/Forestry_Trade_Flows_E_All_Data.csv')

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

# Initializing dataframes

df_forest_products = pd.DataFrame()
df_plywood = pd.DataFrame()
df_paper_paperboard = pd.DataFrame()
df_industrial_roundwood_non_coniferous_non_tropical = pd.DataFrame()
df_sawnwood_coniferous = pd.DataFrame()
df_veneer_sheets = pd.DataFrame()
df_newsprint = pd.DataFrame()
df_fibreboard = pd.DataFrame()
df_industrial_roundwood_coniferous = pd.DataFrame()
df_sawnwood_non_coniferous = pd.DataFrame()
df_wood_pulp = pd.DataFrame()
df_industrial_roundwood_non_coniferous_tropical = pd.DataFrame()
df_wood_chips_and_particles = pd.DataFrame()

# Main loop

for xxx in range(len(items)):
    
    item = items[xxx]
    
    # Subset data for the desired category
    
    subdata = data[data.Item == item]
    subdata = subdata[subdata.Element == 'Export Value']
    
    for nv in netvars:
        
        # Initialize a trade matrix
        
        M = np.zeros((len(nations),len(nations)))
        
        for i in range(len(nations)):
            
            # Visualizing progress
            
            print('Item ' + str(1+xxx) + ' of ' + str(len(items)) + ' :: Year ' + str(nv[1:]) + ' of 2017 :: Nation ' + str(i+1) + ' of ' + str(len(nations)))
            
            for j in range(len(nations)):
                
                # Subset for nations i and j
                
                tmp = subdata[subdata['Reporter Countries'] == nations[i]].reset_index(drop = True)
                tmp = tmp[tmp['Partner Countries'] == nations[j]].reset_index(drop = True)
                
                # Extract values
                
                if len(tmp) > 0:
                    
                    M[i][j] = tmp[nv][0]
                    
        # Create the trade network
        
        N = nx.DiGraph(M)
        
        # Visualize the trade network
        
        #plt.figure()
        #nx.draw(N)
        #plt.title('World Trade Network for ' + item.replace(' (export/import)','') + ' - ' + str(nv[1:]))
        #plt.savefig(direc + 'figures/' + item.replace('(','_').replace(')','_').replace('/','_').replace('-','_') + '__' + str(nv[1:]) + '.eps')
        #plt.savefig(direc + 'figures/' + item.replace('(','_').replace(')','_').replace('/','_').replace('-','_') + '__' + str(nv[1:]) + '.png')
        
        # Create the competition graph
        
        A = cg.competition_graph(M)
        G = nx.Graph(A)
        
        # Visualize the competition graph
        
        #plt.figure()
        #nx.draw(G)
        #plt.title('Competition Graph for ' + item.replace(' (export/import)','') + ' - ' + str(nv[1:]))
        #plt.savefig(direc + 'figures/' + item.replace('(','_').replace(')','_').replace('/','_').replace('-','_') + '__' + str(nv[1:]) + '.eps')
        #plt.savefig(direc + 'figures/' + item.replace('(','_').replace(')','_').replace('/','_').replace('-','_') + '__' + str(nv[1:]) + '.png')
        
        # Calculate centrality measures for both graphs
        
        in_degree_centrality_trade = nx.in_degree_centrality(N)
        out_degree_centrality_trade = nx.out_degree_centrality(N)
        betweenness_centrality_trade = nx.betweenness_centrality(N)
        eigenvector_centrality_trade = nx.eigenvector_centrality(N)
        closeness_centrality_trade = nx.closeness_centrality(N)
        clustering_coef_trade = nx.clustering(N)
        expos = np.sum(M, axis = 0)
        imps = np.sum(M, axis = 1)
        net_exports = expos - imps
        #cpip_imports_core = 
        #cpip_exports_core = 
        betweenness_centrality_comp = nx.betweenness_centrality(G)
        eigenvector_centrality_comp = nx.eigenvector_centrality(G)
        closeness_centrality_comp = nx.closeness_centrality(G)
        clustering_coef_comp = nx.clustering(G)
        #cpip_comp_core = 
        
        # Maybe also compute some network level statistics?
        
        avg_clustering_trade = pd.Series([nx.average_clustering(N)]*len(nations), name = 'Average Clustering Coefficient - Trade')
        avg_clustering_comp = pd.Series([nx.average_clustering(G)]*len(nations), name = 'Average Clustering Coefficient - Comp')
        
        # Update dataframe
        
        nats = pd.Series(nations, name = 'Nation')
        year = pd.Series([nv[1:]]*len(nations), name = 'Year')
        in_degree_centrality_trade = pd.Series(in_degree_centrality_trade, name = 'In Degree Centrality - Trade')
        out_degree_centrality_trade = pd.Series(out_degree_centrality_trade, name = 'Out Degree Centrality - Trade')
        betweenness_centrality_trade = pd.Series(betweenness_centrality_trade, name = 'Betweenness Centrality - Trade')
        eigenvector_centrality_trade = pd.Series(eigenvector_centrality_trade, name = 'Eigenvector Centrality - Trade')
        closeness_centrality_trade = pd.Series(closeness_centrality_trade, name = 'Closeness Centrality - Trade')
        clustering_coef_trade = pd.Series(clustering_coef_trade, name = 'Clustering Coefficient - Trade')
        net_exports = pd.Series(net_exports, name = 'Net Exports')
        #cpip_imports_core = pd.Series(cpip_imports_core, name = 'Imports Core')
        #cpip_exports_core = pd.Series(cpip_exports_core, name = 'Exports Core')
        betweenness_centrality_comp = pd.Series(betweenness_centrality_comp, betweenness_centrality_comp, name = 'Betweenness Centrality - Competition')
        eigenvector_centrality_comp = pd.Series(eigenvector_centrality_comp, name = 'Eigenvector Centrality - Competition')
        closeness_centrality_comp = pd.Series(closeness_centrality_comp, name = 'Closeness Centrality - Competition')
        clustering_coef_comp = pd.Series(clustering_coef_comp, name = 'Clustering Coefficient - Competition')
        #cpip_comp_core = pd.Series(cpip_comp_core, name = 'Competition Core')
        
        if xxx == 0:
            
            df_forest_products_tmp = pd.concat([nats, year, in_degree_centrality_trade, out_degree_centrality_trade,
                                            betweenness_centrality_trade,
                                            eigenvector_centrality_trade, closeness_centrality_trade, clustering_coef_trade,
                                            net_exports, betweenness_centrality_comp,
                                            eigenvector_centrality_comp, closeness_centrality_comp, clustering_coef_comp,
                                            avg_clustering_trade, avg_clustering_comp], axis = 1) # cpip_imports_core, cpip_exports_core, cpip_comp_core
            
            df_forest_products = pd.concat([df_forest_products, df_forest_products_tmp], axis = 0)
            
        elif xxx == 1:
            
            df_plywood_tmp = pd.concat([nats, year, in_degree_centrality_trade, out_degree_centrality_trade,
                                    betweenness_centrality_trade,
                                    eigenvector_centrality_trade, closeness_centrality_trade, clustering_coef_trade,
                                    net_exports, betweenness_centrality_comp,
                                    eigenvector_centrality_comp, closeness_centrality_comp, clustering_coef_comp,
                                    avg_clustering_trade, avg_clustering_comp], axis = 1) # cpip_imports_core, cpip_exports_core, cpip_comp_core
            
            df_plywood = pd.concat([df_plywood, df_plywood_tmp], axis = 0)
            
        elif xxx == 2:
            
            df_paper_paperboard_tmp = pd.concat([nats, year, in_degree_centrality_trade, out_degree_centrality_trade,
                                            betweenness_centrality_trade,
                                            eigenvector_centrality_trade, closeness_centrality_trade, clustering_coef_trade,
                                            net_exports, betweenness_centrality_comp,
                                            eigenvector_centrality_comp, closeness_centrality_comp, clustering_coef_comp,
                                            avg_clustering_trade, avg_clustering_comp], axis = 1) # cpip_imports_core, cpip_exports_core, cpip_comp_core
            
            df_paper_paperboard = pd.concat([df_paper_paperboard, df_paper_paperboard_tmp], axis = 0)
            
        elif xxx == 3:
            
            df_industrial_roundwood_non_coniferous_non_tropical_tmp = pd.concat([nats, year, in_degree_centrality_trade, out_degree_centrality_trade,
                                                                            betweenness_centrality_trade,
                                                                            eigenvector_centrality_trade, closeness_centrality_trade, clustering_coef_trade,
                                                                            net_exports, betweenness_centrality_comp,
                                                                            eigenvector_centrality_comp, closeness_centrality_comp, clustering_coef_comp,
                                                                            avg_clustering_trade, avg_clustering_comp], axis = 1) # cpip_imports_core, cpip_exports_core, cpip_comp_core
            
            df_industrial_roundwood_non_coniferous_non_tropical = pd.concat([df_industrial_roundwood_non_coniferous_non_tropical, df_industrial_roundwood_non_coniferous_non_tropical_tmp], axis = 0)
            
        elif xxx == 4:
            
            df_sawnwood_coniferous_tmp = pd.concat([nats, year, in_degree_centrality_trade, out_degree_centrality_trade,
                                                                        betweenness_centrality_trade,
                                                                        eigenvector_centrality_trade, closeness_centrality_trade, clustering_coef_trade,
                                                                        net_exports, betweenness_centrality_comp,
                                                                        eigenvector_centrality_comp, closeness_centrality_comp, clustering_coef_comp,
                                                                        avg_clustering_trade, avg_clustering_comp], axis = 1) # cpip_imports_core, cpip_exports_core, cpip_comp_core
            
            df_sawnwood_coniferous = pd.concat([df_sawnwood_coniferous, df_sawnwood_coniferous_tmp], axis = 0)
            
        elif xxx == 5:
            
            df_veneer_sheets_tmp = pd.concat([nats, year, in_degree_centrality_trade, out_degree_centrality_trade,
                                                            betweenness_centrality_trade,
                                                            eigenvector_centrality_trade, closeness_centrality_trade, clustering_coef_trade,
                                                            net_exports, betweenness_centrality_comp,
                                                            eigenvector_centrality_comp, closeness_centrality_comp, clustering_coef_comp,
                                                            avg_clustering_trade, avg_clustering_comp], axis = 1) # cpip_imports_core, cpip_exports_core, cpip_comp_core
            
            df_veneer_sheets = pd.concat([df_veneer_sheets, df_veneer_sheets_tmp], axis = 0)
            
        elif xxx == 6:
            
            df_newsprint_tmp = pd.concat([nats, year, in_degree_centrality_trade, out_degree_centrality_trade,
                                                    betweenness_centrality_trade,
                                                    eigenvector_centrality_trade, closeness_centrality_trade, clustering_coef_trade,
                                                    net_exports, betweenness_centrality_comp,
                                                    eigenvector_centrality_comp, closeness_centrality_comp, clustering_coef_comp,
                                                    avg_clustering_trade, avg_clustering_comp], axis = 1) # cpip_imports_core, cpip_exports_core, cpip_comp_core
            
            df_newsprint = pd.concat([df_newsprint, df_newsprint_tmp], axis = 0)
            
        elif xxx == 7:
            
            df_fibreboard_tmp = pd.concat([nats, year, in_degree_centrality_trade, out_degree_centrality_trade,
                                                betweenness_centrality_trade,
                                                eigenvector_centrality_trade, closeness_centrality_trade, clustering_coef_trade,
                                                net_exports, betweenness_centrality_comp,
                                                eigenvector_centrality_comp, closeness_centrality_comp, clustering_coef_comp,
                                                avg_clustering_trade, avg_clustering_comp], axis = 1) # cpip_imports_core, cpip_exports_core, cpip_comp_core
                
            df_fibreboard = pd.concat([df_fibreboard, df_fibreboard_tmp], axis = 0)
            
        elif xxx == 8:
            
            df_industrial_roundwood_coniferous_tmp = pd.concat([nats, year, in_degree_centrality_trade, out_degree_centrality_trade,
                                            betweenness_centrality_trade,
                                            eigenvector_centrality_trade, closeness_centrality_trade, clustering_coef_trade,
                                            net_exports, betweenness_centrality_comp,
                                            eigenvector_centrality_comp, closeness_centrality_comp, clustering_coef_comp,
                                            avg_clustering_trade, avg_clustering_comp], axis = 1) # cpip_imports_core, cpip_exports_core, cpip_comp_core
            
            df_industrial_roundwood_coniferous = pd.concat([df_industrial_roundwood_coniferous, df_industrial_roundwood_coniferous_tmp], axis = 0)
            
        elif xxx == 9:
            
            df_sawnwood_non_coniferous_tmp = pd.concat([nats, year, in_degree_centrality_trade, out_degree_centrality_trade,
                                        betweenness_centrality_trade,
                                        eigenvector_centrality_trade, closeness_centrality_trade, clustering_coef_trade,
                                        net_exports, betweenness_centrality_comp,
                                        eigenvector_centrality_comp, closeness_centrality_comp, clustering_coef_comp,
                                        avg_clustering_trade, avg_clustering_comp], axis = 1) # cpip_imports_core, cpip_exports_core, cpip_comp_core
            
            df_sawnwood_non_coniferous = pd.concat([df_sawnwood_non_coniferous, df_sawnwood_non_coniferous_tmp], axis = 0)
            
        elif xxx == 10:
            
            df_wood_pulp_tmp = pd.concat([nats, year, in_degree_centrality_trade, out_degree_centrality_trade,
                                        betweenness_centrality_trade,
                                        eigenvector_centrality_trade, closeness_centrality_trade, clustering_coef_trade,
                                        net_exports, betweenness_centrality_comp,
                                        eigenvector_centrality_comp, closeness_centrality_comp, clustering_coef_comp,
                                        avg_clustering_trade, avg_clustering_comp], axis = 1) # cpip_imports_core, cpip_exports_core, cpip_comp_core
            
            df_wood_pulp = pd.concat([df_wood_pulp, df_wood_pulp_tmp], axis = 0)
            
        elif xxx == 11:
            
            df_industrial_roundwood_non_coniferous_tropical_tmp = pd.concat([nats, year, in_degree_centrality_trade, out_degree_centrality_trade,
                                                            betweenness_centrality_trade,
                                                            eigenvector_centrality_trade, closeness_centrality_trade, clustering_coef_trade,
                                                            net_exports, betweenness_centrality_comp,
                                                            eigenvector_centrality_comp, closeness_centrality_comp, clustering_coef_comp,
                                                            avg_clustering_trade, avg_clustering_comp], axis = 1) # cpip_imports_core, cpip_exports_core, cpip_comp_core
            
            df_industrial_roundwood_non_coniferous_tropical = pd.concat([df_industrial_roundwood_non_coniferous_tropical, df_industrial_roundwood_non_coniferous_tropical_tmp], axis = 0)
            
        elif xxx == 12:
            
            df_wood_chips_and_particles_tmp = pd.concat([nats, year, in_degree_centrality_trade, out_degree_centrality_trade,
                                                        betweenness_centrality_trade,
                                                        eigenvector_centrality_trade, closeness_centrality_trade, clustering_coef_trade,
                                                        net_exports, betweenness_centrality_comp,
                                                        eigenvector_centrality_comp, closeness_centrality_comp, clustering_coef_comp,
                                                        avg_clustering_trade, avg_clustering_comp], axis = 1) # cpip_imports_core, cpip_exports_core, cpip_comp_core
            
            df_wood_chips_and_particles = pd.concat([df_wood_chips_and_particles, df_wood_chips_and_particles_tmp], axis = 0)
            
# Saving results to file

df_forest_products.to_csv(direc + 'data/forest_products.csv', index = False)
df_plywood.to_csv(direc + 'data/plywood.csv', index = False)
df_paper_paperboard.to_csv(direc + 'data/paper_and_paperboard.csv', index = False)
df_industrial_roundwood_non_coniferous_non_tropical.to_csv(direc + 'data/industrial_roundwood_non_coniferous_non_tropical.csv', index = False)
df_sawnwood_coniferous.to_csv(direc + 'data/sawnwood_coniferous.csv', index = False)
df_veneer_sheets.to_csv(direc + 'data/veneer_sheets.csv', index = False)
df_newsprint.to_csv(direc + 'data/newsprint.csv', index = False)
df_fibreboard.to_csv(direc + 'data/fibreboard.csv', index = False)
df_industrial_roundwood_coniferous.to_csv(direc + 'data/industrial_roundwood_coniferous.csv', index = False)
df_sawnwood_non_coniferous.to_csv(direc + 'data/sawnwood_non_coniferous.csv', index = False)
df_wood_pulp.to_csv(direc + 'data/wood_pulp.csv', index = False)
df_industrial_roundwood_non_coniferous_tropical.to_csv(direc + 'data/industrial_roundwood_non_coniferous_tropical.csv', index = False)
df_wood_chips_and_particles.to_csv(direc + 'data/wood_chips_and_particles.csv', index = False)

