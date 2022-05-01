# This just makes clean LaTeX formatted tables

# Importing required modules

import pandas as pd

# Directory info

username = ''
direc = 'C:/Users/' + username + '/Documents/Data/KP_timber_trade/results/'

# Reading in the tables

t0 = pd.read_csv(direc + 'trade.csv')
t1 = pd.read_csv(direc + 'trade_1.csv')
t2 = pd.read_csv(direc + 'trade_2.csv')
t3 = pd.read_csv(direc + 'trade_3.csv')
c0 = pd.read_csv(direc + 'competition.csv')
c1 = pd.read_csv(direc + 'competition_1.csv')
c2 = pd.read_csv(direc + 'competition_2.csv')
c3 = pd.read_csv(direc + 'competition_3.csv')

# Defining a function to caluclate statistical significance

def signif(inp1,inp2):
    
    if abs(inp1 / inp2) >= 2.576:
        
        stars = '$^{***}$'
        
    elif abs(inp1 / inp2) >= 1.960:
        
        stars = '$^{**}$'
        
    elif abs(inp1 / inp2) >= 1.645:
        
        stars = '$^{*}$'
        
    else:
        
        stars = ''
    
    return stars

# Defining functions to make formatted results tables

def trade_fx(val,df):
    
    if val%2 == 0:
        
        output = str(df.Market[val]) + ' & ' + str(df['Betweenness Centrality'][val]) + str(signif(df['Betweenness Centrality'][val],df['Betweenness Centrality'][val+1])) + ' & ' + str(df['Closeness Centrality'][val]) + str(signif(df['Closeness Centrality'][val],df['Closeness Centrality'][val+1])) + ' & ' + str(df['Eigenvector Centrality'][val]) + str(signif(df['Eigenvector Centrality'][val],df['Eigenvector Centrality'][val+1])) + ' & ' + str(df['In Degree Centrality'][val]) + str(signif(df['In Degree Centrality'][val],df['In Degree Centrality'][val+1])) + ' & ' + str(df['Out Degree Centrality'][val]) + str(signif(df['Out Degree Centrality'][val],df['Out Degree Centrality'][val+1])) + ' & ' + str(df['Clustering Coefficient'][val]) + str(signif(df['Clustering Coefficient'][val],df['Clustering Coefficient'][val+1])) + '\\\\\n'
        
    else:
        
        output = ' & (' + str(df['Betweenness Centrality'][val]) + ') & (' + str(df['Closeness Centrality'][val]) + ') & (' + str(df['Eigenvector Centrality'][val]) + ') & (' + str(df['In Degree Centrality'][val]) + ') & (' + str(df['Out Degree Centrality'][val]) + ') & (' + str(df['Clustering Coefficient'][val]) + ')\\\\\n'
        
    return output

def comp_fx(val,df):
    
    if val%2 == 0:
        
        output = str(df.Market[val]) + ' & ' + str(df['Betweenness Centrality'][val]) + str(signif(df['Betweenness Centrality'][val],df['Betweenness Centrality'][val+1])) + ' & ' + str(df['Closeness Centrality'][val]) + str(signif(df['Closeness Centrality'][val],df['Closeness Centrality'][val+1])) + ' & ' + str(df['Eigenvector Centrality'][val]) + str(signif(df['Eigenvector Centrality'][val],df['Eigenvector Centrality'][val+1])) + ' & ' + str(df['Clustering Coefficient'][val]) + str(signif(df['Clustering Coefficient'][val],df['Clustering Coefficient'][val+1])) + '\\\\\n'
        
    else:
        
        output = ' & (' + str(df['Betweenness Centrality'][val]) + ') & (' + str(df['Closeness Centrality'][val]) + ') & (' + str(df['Eigenvector Centrality'][val]) + ') & (' + str(df['Clustering Coefficient'][val]) + ')\\\\\n'
        
    return output

def label_maker(lab):
    
    label = '\\label{' + lab + '}\n'
    
    return label

# Table headers and footers

header = ['\\begin{table}[h!]\n', '\\centering\n']

trade_header = ['\\begin{tabular}{lcccccc}\n', '\\hline\\hline\n', '\\rule{0pt}{3ex}\n',
          'Market & \\begin{tabular}{c}Betweenness\\\\Centrality\\end{tabular} & \\begin{tabular}{c}Closeness\\\\Centrality\\end{tabular} & \\begin{tabular}{c}Eigenvector\\\\Centrality\\end{tabular} & \\begin{tabular}{c}In Degree\\\\Centrality\\end{tabular} & \\begin{tabular}{c}Out Degree\\\\Centrality\\end{tabular} & \\begin{tabular}{c}Clustering\\\\Coefficient\\end{tabular}\\\\\n'
          '\\hline\n', '\\rule{0pt}{0ex}\\\\\n']

comp_header = ['\\begin{tabular}{lcccc}\n', '\\hline\\hline\n', '\\rule{0pt}{3ex}\n',
          'Market & \\begin{tabular}{c}Betweenness\\\\Centrality\\end{tabular} & \\begin{tabular}{c}Closeness\\\\Centrality\\end{tabular} & \\begin{tabular}{c}Eigenvector\\\\Centrality\\end{tabular} & \\begin{tabular}{c}Clustering\\\\Coefficient\\end{tabular}\\\\\n'
          '\\hline\n', '\\rule{0pt}{0ex}\\\\\n']

footer = ['\\hline\n', '\\end{tabular}\n', '\\end{table}']

# Definining functions for writing tables to .txt

def tex_writer(saveas, table_label, df, table_type):
    
    with open(direc + saveas + '.txt', 'w') as f:
        
        for head in header:
            
            f.write(head)
            
        f.write(label_maker(table_label))
        
        if table_type == 'trade':
            
            for head in trade_header:
                
                f.write(head)
                
            for i in range(len(df)):
                
                f.write(trade_fx(i,df))
            
        else:
            
            for head in comp_header:
                
                f.write(head)
                
            for i in range(len(df)):
                
                f.write(comp_fx(i,df))
                
        for foot in footer:
            
            f.write(foot)
            
    f.close()

# Writing tables

save_locs = ['trade', 'trade_1', 'trade_2', 'trade_3', 'competition', 'competition_1', 'competition_2', 'competition_3']
table_labels = ['t0', 't1', 't2', 't3', 'c0', 'c1', 'c2', 'c3']
dfs = [t0, t1, t2, t3, c0, c1, c2, c3]
table_types = ['trade', 'trade', 'trade', 'trade', 'comp', 'comp', 'comp', 'comp']

for x in range(len(save_locs)):
    
    tex_writer(save_locs[x], table_labels[x], dfs[x], table_types[x])
    
