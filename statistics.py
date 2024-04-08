"""
This script is part of PMS (Python for Medical Statistics) project
The following resources were used during the build of this project
https://www.geeksforgeeks.org/how-to-perform-a-one-way-anova-in-python/
https://www.statology.org/one-way-anova-python/
https://www.reneshbedre.com/blog/anova.html

The project has 2 classes
- TableNormalizer which is used to filter the initial data-sets (provided as XLSX files), removing noise, comments 
  as well as some filled cells which are not part of valid tables
  This class must be improved an parametrized in order to allow researchers apply their own filters
- MedicalStatistics which is used to perform various computations on the normalized tables and get graphics and
  statistics out of the data
  This class must also be improved in order to allow researchers decide the type of statistics and customize the way the graphics are built
"""

import statistics
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats
import pandas.plotting
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from bioinfokit.analys import stat

#a class for normalizing the data inside the excel tables
class TableNormalizer:
    def __init__(self, df):
        self.df = df
  
    #remove white-spaces from left/right of column-names/header items
    def trim_all_columns(self):
        header_initial = list(self.df.iloc[1:,:])        
        trim_strings = lambda x: x.strip() if isinstance(x, str) else x
        header_trim = list(map(trim_strings, header_initial))           
        self.df = self.df.applymap(trim_strings)
        self.df.rename(columns=dict(zip(header_initial,header_trim)), inplace=True)

    #get two lists of numbers and string having 1 for valid element and 0 for missing or invalid element
    def get_val_types(self, values):
        numbers = []
        strings = []
        for val in values:
            if type(val) == str:  
                strings += [1]          
                if val.isnumeric():
                    numbers += [1] 
                else:
                    numbers += [0] 
            elif type(val) == int:
                numbers += [1]
                strings += [0]
            elif type(val) == float:
                strings += [0]
                if pd.isna(val):
                    numbers += [0]
                else:
                    numbers += [1]
            else:
                numbers += [0]    
                strings += [0]
        return numbers, strings

    #remove empty lines and rename the table using as header, the first row of the valid sub-table
    def remove_lines(self):    
        header = list(self.df.iloc[1:,:])    
        invalid_header = [1 if 'unnamed' in val.lower() else 0 for val in header]    
        invalid = invalid_header.count(1) == len(invalid_header)
        i = 0
        remove_indexes = []
        while True:
            line = list(self.df.iloc[i,:])
            nrs, strs = self.get_val_types(line)
            s1 = sum(nrs)
            s2 = sum(strs)
            if s1 == 0 and s2 == 0:
                remove_indexes += [i]
            else:
                break
            i += 1
        if len(remove_indexes) > 0:
            self.df = self.df.iloc[max(remove_indexes) + 1:]
        if invalid:
            first_valid = list(self.df.iloc[0,:])
            self.df = self.df.iloc[1:]
            self.df.rename(columns=dict(zip(header,first_valid)), inplace=True)

    #remove any column which do not have valid strings or numbers, including those with NaN items
    def remove_columns(self):    
        remove_indexes = []
        i = 0
        while True:
            col = self.df.iloc[:,i]
            nrs, strs = self.get_val_types(col)
            s1 = sum(nrs)
            s2 = sum(strs)
            if s1 == 0 and s2 == 0:
                remove_indexes += [i]
            else:
                break
            i += 1
        if len(remove_indexes) > 0:
            self.df = self.df.drop(self.df.columns[remove_indexes], axis = 1)                

    #extract and normalize the data from a single table, trimming any possible blank rows or columns
    def normalize_data_single_table(self):
        self.remove_columns()    
        self.remove_lines()
        self.trim_all_columns()
        self.df.index = range(len(self.df.index))
        return [self.df]

    #search for group of rows defining sub-tables with blank lines between
    def find_tables(self):
        i = 0
        specs = []
        spec = []
        start_table = False
        last_s1 = 0
        last_s2 = 0
        for i in range(len(self.df.index)):
            #get current line as a list of items
            line = list(self.df.iloc[i,:])
            #get the valid numbers and strings in the line
            nrs, strs = self.get_val_types(line)
            #sum the valid numbers in s1 and valid strings in s2
            s1 = sum(nrs)
            s2 = sum(strs)
            #if there are no numbers but we have at least 5 strings, then this is a header
            if s1 == 0 and s2 > 5:
                spec += [i]
                start_table = True
            else:
                #if we found a header, then we store the other rows after the header and stop considering a new table
                if start_table:
                    last_s1 = s1
                    last_s2 = s2
                    start_table = False
                    spec += [i]
                else:
                    #if the total number of numbers and strings are the same as for the previous line then the line is from the sub-table
                    if s1 == last_s1 and s2 <= last_s2 and sum([s1,s2]) > 2:
                        spec += [i]
                    else:
                        #if there are mismatches between the new line and the previous one, then the sub-table has finished and something else starts
                        if len(spec) > 0:
                            specs += [spec]
                            spec = []
            i += 1                
        if len(spec) > 0:
            specs += [spec]
            spec = []
        return specs
    
    #remove any full empty or partial empty column
    def remove_empty_columns(self):
        remove_indexes = []
        for i in range(len(self.df.columns)):
            #get current column as a list of items
            col = list(self.df.iloc[:,i])
            #get the list of valid numbers inside the column
            nrs, _ = self.get_val_types(col)      
            #if no number in the column, prepare to remove it
            if sum(nrs) == 0:
                remove_indexes += [i]      
        #remove the invalid columns
        if len(remove_indexes) > 0:
            self.df = self.df.drop(self.df.columns[remove_indexes], axis = 1)   

    #discard any added computations which are not part of the original dataset
    def remove_light_computations(self):
        flsum = []        
        for i in range(len(self.df.columns)):
            #get current column as a list of items
            col = list(self.df.iloc[:,i])
            #highlight with 1 the items which are actual valid floats and with 0 all the others
            floats = [1 if isinstance(el, float) and not pd.isna(el) else 0 for el in col]
            #store the actual number of valid floats for each column
            flsum += [sum(floats)]
        #compute median (the most used value)
        median = statistics.median(flsum)
        #buld a list with indexes of columns with a lot more floats than the median
        remove_indexes = [flsum.index(el) for el in flsum if el > 2 * median]
        #remove these columns
        if len(remove_indexes) > 0:
            self.df = self.df.drop(self.df.columns[remove_indexes], axis = 1)                   

    #use a single header for all sub-tables, the header of the first table
    def uniform_rename_tables(self):
        #the initial header
        header_wrong = list(self.df.iloc[1:,:])    
        #the header starting at the first valid line found in specs
        header_corect = self.df.iloc[specs[0][0],:]
        #rename all columns inplace
        self.df.rename(columns=dict(zip(header_wrong, header_corect)), inplace=True)

    #build individual normalized sub-tables, from the original dataset
    def normalize_data_multiple_tables(self):
        #get the specifications of possible sub-tables as a list of lists with row indexes
        specs = self.find_tables()
        #remove the last line of the table if the total number of rows including the header is even
        #if the number of rows excluding the header is odd, then the last line is some kind of non-mandatory computation
        #the actual number of rows must be even, as the data was split into 2 groups and is always a multiple of 2
        specs = [tab[:-1] if len(tab) % 2 == 0 else tab for tab in specs]
                
        self.remove_empty_columns()       
        self.remove_light_computations()
        self.uniform_rename_tables()

        #split sub-tables
        tables = []
        for sp in specs:        
            #a table is defined by the group of rows starting after the header
            table = self.df.iloc[sp[1:],:]
            #reset the index for current sub-table to start at 0
            table.index = range(len(table.index))
            #add sub-table
            tables += [table]

        return tables        
#----------------------------------------------------------------------------------------------        

class MedicalStatistics:
    def __init__(self, suffix, tables, indivizi_per_lot, table_names = None, focus_cols = None):
        #prima coloane identifica indivizii si loturile din care fac parte
        self.suffix = suffix
        self.tables = tables
        self.table = tables[0]
        self.names = list(self.table.iloc[:,0])
        self.indivizi = indivizi_per_lot
        self.loturi = len(self.names) // self.indivizi 
        self.coloane = len(self.table.columns) - 1
        if table_names != None:
            self.named_tables = { tm : table for (tm, table) in list(zip(table_names, self.tables)) }   
        if focus_cols != None:
            self.focus_col_header = focus_cols         
    
    def SingleTable_Stats(self):
        print(self.table.describe())
        plt.gcf().subplots_adjust(wspace=1, hspace=1)
        self.table.plot(kind='box', subplots=True, layout=(4,3), sharex=False, sharey=False, figsize=[10,10])
        plt.savefig(f'{self.suffix}_box-plot.png')
        self.table.hist(figsize=[10,10])
        plt.savefig(f'{self.suffix}_histograms.png')    
        pandas.plotting.scatter_matrix(self.table, figsize=[10,10])    
        plt.savefig(f'{self.suffix}_scatter-matrix.png')

    def SingleTable_Individual(self):
        #fix-me - sub-plots must be auto-configured depending on the size of the data-set        
        fig, ax = plt.subplots(4, 3)
        fig.set_size_inches([10, 10])
        fig.tight_layout(pad=1.5)    
        for i in range(self.coloane):
            ax[i//3,i%3].plot(list(self.table.iloc[i,:]))
            ax[i//3,i%3].grid()
            ax[i//3,i%3].set_title(self.names[i])
        plt.savefig(f'{self.suffix}_individual.png')        

    def SingleTable_PerLot(self):
        #fig, ax = plt.subplots(3, 2)
        fig,ax = plt.subplots(figsize =(15, 15)) 
        #fig.set_size_inches([10, 10])
        fig.tight_layout(pad=1.5)            
        for i in range(self.loturi):
            sub_table = self.table.iloc[i*self.indivizi:(i+1)*self.indivizi,:]
            sub_table.index = range(len(sub_table.index))
            means = [sub_table.iloc[:,j].mean() for j in range(1, len(sub_table.columns))]
            ax.plot(range(1,13), means, label=self.names[i*self.indivizi][:-1], linewidth=5)
            #ax[i//2,i%2].plot(means)
            #ax[i//2,i%2].grid()
            #ax[i//2,i%2].set_title(self.names[i*self.indivizi][:-1])        
        ax.grid()        
        plt.xlabel("Timeline (Week 1-12)", fontsize=30)
        plt.ylabel("Weight", fontsize = 30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=24)
        plt.savefig(f'{self.suffix}_per_lot.png', bbox_inches='tight', dpi=300)
        #fig.savefig(f'{self.suffix}_per_lot.svg', format='svg', dpi=1200)

    def SingleTable_PerSex(self):
        fig1, ax1 = plt.subplots(3, 2)
        fig2, ax2 = plt.subplots(3, 2)
        fig1.set_size_inches([10, 10])
        fig1.tight_layout(pad=1.5)    
        fig2.set_size_inches([10, 10])
        fig2.tight_layout(pad=1.5)    
        for i in range(self.loturi):
            sub_table_m = self.table.iloc[i*self.indivizi:i*self.indivizi+3,:]
            sub_table_f = self.table.iloc[i*self.indivizi+3:(i+1)*self.indivizi,:]
            sub_table_m.index = range(len(sub_table_m.index))
            sub_table_f.index = range(len(sub_table_f.index))
            means_m = [sub_table_m.iloc[:,j].mean() for j in range(1, len(sub_table_m.columns))]
            means_f = [sub_table_f.iloc[:,j].mean() for j in range(1, len(sub_table_f.columns))]
            ax1[i//2,i%2].plot(means_m)
            ax1[i//2,i%2].grid()
            ax1[i//2,i%2].set_title(self.names[i*self.indivizi][:-1]+'_M')
            ax2[i//2,i%2].plot(means_f)
            ax2[i//2,i%2].grid()
            ax2[i//2,i%2].set_title(self.names[i*self.indivizi][:-1]+'_F')    
        fig1.savefig(f'{self.suffix}_masculin.png')
        fig2.savefig(f'{self.suffix}_feminin.png')     

    def SingleTable_KMeans(self, clusters):
        y = []    
        x = []
        for i in range(len(self.table.index)):
            y += list(self.table.iloc[i,1:])
            x += [i] * len(self.table.iloc[i,1:])

        inertia = []
        for i in range(1,15):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(list(zip(x, y)))
            inertia += [kmeans.inertia_]

        fig, [ax1, ax2] = plt.subplots(2, 1)
        fig.set_size_inches([10, 10])
        fig.tight_layout(pad=1.5)
        ax1.plot(range(1,15), inertia, marker='o')
        ax1.grid()
        ax1.set_title(f'{self.suffix}_kmeans-inertia.png')

        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(list(zip(x, y)))
        ax2.scatter(x, y, c=kmeans.labels_)
        ax2.grid()   
        ax2.set_title(f'{self.suffix}_grupuri_greutate.png')
        fig.savefig(f'{self.suffix}_KMeans.png')

    def ProcessSingleTable(self):
        #self.SingleTable_Stats()
        #self.SingleTable_Individual()
        self.SingleTable_PerLot()
        #self.SingleTable_PerSex()
        #self.SingleTable_KMeans(8)

    #--------------------------------------------------------------------------------
    def ArrangeFocusedTables(self):
        self.focus_tables = {}
        #iterate header for the required columns indexes and names (names here will be new table names)
        for i, (k, v) in enumerate(self.focus_col_header.items()):
            #we prepare a new table
            crt_table = pd.DataFrame()
            #iterate all tables and grab v-indexed columns for each table and push them to our new table, usign the table name as the name for the column
            for j, (kk, vv) in enumerate(self.named_tables.items()):                
                #grab the full column with index v in table j
                crt_table[kk] = vv.iloc[:, v]
            self.focus_tables[k] = crt_table

    def MultipleTable_Stats(self):
        #general statisticis for each of the initial set of tables
        for i, (k, v) in enumerate(self.named_tables.items()):
            v=v.astype(float)
            #print(v.describe())
            plt.gcf().subplots_adjust(wspace=1, hspace=1)
            v.plot(kind='box', subplots=True, layout=(4,3), sharex=False, sharey=False, figsize=[10,10])
            plt.savefig(f"{self.suffix}_{k}_box-plot.png")
            v.hist(figsize=[10,10])
            plt.savefig(f"{self.suffix}_{k}_histograms.png")    
            pandas.plotting.scatter_matrix(v, figsize=[10,10])    
            plt.savefig(f"{self.suffix}_{k}_scatter-matrix.png")

        #general statistics for each of the mixed set of tables
        for i, (k, v) in enumerate(self.focus_tables.items()):            
            v=v.astype(float)
            print(v.describe())
            plt.gcf().subplots_adjust(wspace=1, hspace=1)
            v.plot(kind='box', subplots=True, layout=(3,2), sharex=False, sharey=False, figsize=[10,10])
            plt.savefig(f"{self.suffix}_{k}_box-plot.png")
            v.hist(figsize=[10,10])
            plt.savefig(f"{self.suffix}_{k}_histograms.png")    
            pandas.plotting.scatter_matrix(v, figsize=[10,10])    
            plt.savefig(f"{self.suffix}_{k}_scatter-matrix.png")

    def MultipleTable_GeneralMeanStd(self):
        for i, (k, v) in enumerate(self.focus_tables.items()): 
            avgs = v.mean()
            std = v.std()
            barWidth = 0.25
            fig = plt.subplots(figsize =(12, 6)) 
            br1 = np.arange(len(avgs)) 
            br2 = [x + barWidth for x in br1] 
            plt.bar(br1, std, color ='r', width = barWidth, edgecolor ='grey', label ='STD') 
            plt.bar(br2, avgs, color ='b', width = barWidth, edgecolor ='grey', label ='MEAN')            
            plt.xlabel("Group", fontweight ='bold', fontsize = 15) 
            plt.ylabel(k, fontweight ='bold', fontsize = 15) 
            plt.xticks([r + barWidth for r in range(len(avgs))], list(self.named_tables.keys()))            
            plt.legend()
            plt.savefig(f"{self.suffix}_{k}_mean_std_regroup.png")

    def MultipleTable_SexMeanStdPerGroup(self):        
        for i, (k, v) in enumerate(self.focus_tables.items()): 
            print(f"male_female_mean_std_per_timp_intrari [{k}]")
            boys_sub_table = v.iloc[0:self.indivizi//2,:]
            girls_sub_table = v.iloc[self.indivizi//2:self.indivizi,:]
            boy_avgs = list(boys_sub_table.mean())
            girl_avgs = list(girls_sub_table.mean())
            boy_std = list(boys_sub_table.std())
            girls_std = list(girls_sub_table.std())
            mins = v.min()
            maxs = v.max()
            min_val = min(mins)
            max_val = max(maxs)
            ind = np.arange(self.indivizi) 
            width = 0.35 

            print('male_sub_table')
            print(boys_sub_table)
            print(' ')
            print('female_sub_table')
            print(girls_sub_table)
            print(' ')
            print('male_mean')
            print(boy_avgs)
            print(' ')
            print('female mean')
            print(girl_avgs)
            print(' ')
            print('male_std')
            print(boy_std)
            print(' ')
            print('female std')
            print(girls_std)
            print(' ')

            fig = plt.subplots(figsize =(12, 6)) 
            p1 = plt.bar(ind, boy_avgs, width, yerr = boy_std)
            p2 = plt.bar(ind, girl_avgs, width, bottom = boy_avgs, yerr = girls_std)

            plt.ylabel(k)
            plt.title(f"Male/Female for [{k}]")
            plt.xticks(ind, list(self.named_tables.keys()))
            plt.yticks(np.arange(0, max_val*2, max_val*2/10))
            plt.legend((p1[0], p2[0]), ('male', 'female'))
            plt.savefig(f"{self.suffix}_{k}_male_female_mean_std_per_timp_intrari.png")     

    def MultipleTable_Anova1Way(self, y_labels):        
        for i, (k, v) in enumerate(self.focus_tables.items()):    
            v=v.astype(float)
            df_melt = pd.melt(v.reset_index(), id_vars=['index'], value_vars=list(self.named_tables.keys()))
            df_melt.columns = ['index', 'treatments', 'value']
            print(k)
            print(v)
            print(df_melt)
            
            fig, ax = plt.subplots(1, 4, figsize =(40, 10), dpi=300)
            ax[3].xaxis.set_tick_params(labelsize=20)
            ax[2].xaxis.set_tick_params(labelsize=20)
            ax[1].xaxis.set_tick_params(labelsize=20)
            ax[0].xaxis.set_tick_params(labelsize=20)            
            ax[3].yaxis.set_tick_params(labelsize=20)
            ax[2].yaxis.set_tick_params(labelsize=20)
            ax[1].yaxis.set_tick_params(labelsize=20)
            ax[0].yaxis.set_tick_params(labelsize=20)

            sns.boxplot(x='treatments', y='value', data=df_melt, color='#99c2a2', ax=ax[2])
            sns.swarmplot(x='treatments', y="value", data=df_melt, color='#7d0013', ax=ax[2])      
            ax[2].set_xlabel('Groups', fontsize=20)
            ax[2].set_ylabel(y_labels[2][i], fontsize=20)      
            ax[2].grid()
                        
            #print(f"Tabelul {k} pentru anova")
            #print(v)

            table_columns = list(self.named_tables.keys())
            fvalue, pvalue = stats.f_oneway(v[table_columns[0]], v[table_columns[1]], v[table_columns[2]], v[table_columns[3]], v[table_columns[4]], v[table_columns[5]])            
            ax[3].bar(['F', 'p'], [fvalue, pvalue])
            for el in zip(['F', 'p'], [fvalue, pvalue]):
                ax[3].text(el[0], int(el[1] * 1000)/1000, int(el[1] * 1000)/1000, ha = 'center', fontsize=24)
            ax[3].set_xlabel('Anova Variables', fontsize=20)
            if len(k) == 3 and k[2] == '%':
                ax[3].set_ylabel('Anova Values (' + k[:-1] + ')', fontsize=20)
            else:
                ax[3].set_ylabel('Anova Values (' + k + ')', fontsize=20)

            #print(fvalue, pvalue)
            model = ols("value ~ C(treatments)", data=df_melt).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            #print(anova_table)
            res = stat()
            res.anova_stat(df=df_melt, res_var='value', anova_model="value ~ C(treatments)")
            #print(res.anova_summary)

            #fig1, ax1 = plt.subplots(figsize =(10, 10))            
            #anova residuals
            print(f"rezidurile anova QQPLOT [{k}]")            
            print(res.anova_std_residuals)
            sm.qqplot(res.anova_std_residuals, line='45', ax=ax[1])
            ax[1].set_xlabel('Theoretical Quantiles', fontsize=20)  
            if len(k) == 3 and k[2] == '%':
                ax[1].set_ylabel(y_labels[1] + ' (' + k[:-1] + ')', fontsize=20)      
            else:
                ax[1].set_ylabel(y_labels[1] + ' (' + k + ')', fontsize=20)      
            ax[1].grid()
            #plt.xlabel("Theoretical Quantiles")
            #plt.ylabel("Standardized Residuals")
            #plt.grid()
            #plt.savefig(f"{self.suffix}_{k}_anova_residuals.png")
            
            # histogram
            #fig2, ax2 = plt.subplots(figsize =(10, 10))
            #ax2.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')             
            ax[0].hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')             
            ax[0].set_xlabel('Anova Residuals', fontsize=20)
            if len(k) == 3 and k[2] == '%':
                ax[0].set_ylabel(y_labels[0] + ' (' + k[:-1] + ')', fontsize=20)      
            else:
                ax[0].set_ylabel(y_labels[0] + ' (' + k + ')', fontsize=20)      
            print(f"Histograme anova [{k}]")
            print(res.anova_model_out.resid)
            #plt.xlabel("Residuals")
            #plt.ylabel('Frequency')
            #fig2.savefig(f"{self.suffix}_{k}_anova_model.png")            
            plt.savefig(f"{self.suffix}_{k}_anova_model.png", bbox_inches='tight')

    def ProcessMultipleTables(self, y_labels):
        self.ArrangeFocusedTables()
        #self.MultipleTable_Stats()
        #self.MultipleTable_GeneralMeanStd()
        #self.MultipleTable_SexMeanStdPerGroup()
        self.MultipleTable_Anova1Way(y_labels)

    def PerformNORTest(self, y_labels):
        self.RI_per_grup_tables = {}                
        new_col_names = ['OF%', 'ON1', 'ON24']

        for i, (k, v) in enumerate(self.named_tables.items()):
            print(k)
            print(v)
            print(' ')
        print(' ')
        print(' ')

        for i, (k, v) in enumerate(self.named_tables.items()):
            crt_table = pd.DataFrame()
            for j in range(len(new_col_names)):
                c1 = v.iloc[:,j*2]
                c2 = v.iloc[:,j*2 + 1]
                crt_table[new_col_names[j]] = (c2/(c1+c2))*100
            self.RI_per_grup_tables[k] = crt_table
        
        for i, (k, v) in enumerate(self.RI_per_grup_tables.items()):
            print(k)
            print(v)
            print(' ')
        print(' ')
        print(' ')

        #-------------------------------------------------------------
        #create mixed tables to bring all group RIs together in 3 tables
        self.focus_tables = {}
        #iterate header for the required columns indexes and names (names here will be new table names)
        for i in range(len(new_col_names)):
            #we prepare a new table
            crt_table = pd.DataFrame()
            #iterate all tables and grab v-indexed columns for each table and push them to our new table, usign the table name as the name for the column
            for j, (kk, vv) in enumerate(self.RI_per_grup_tables.items()):
                #grab the full column with index v in table j
                crt_table[kk] = vv.iloc[:, i]
            self.focus_tables[new_col_names[i]] = crt_table

        for i, (k, v) in enumerate(self.focus_tables.items()):
            print(k)
            print(v)
            print(' ')
        print(' ')
        print(' ')
        #---------------------------------------------------------------

        #general statistics for each of the mixed set of tables
        #for i, (k, v) in enumerate(self.focus_tables.items()):            
        #    v=v.astype(float)
        #    print(v.describe())
        #    plt.gcf().subplots_adjust(wspace=1, hspace=1)
        #    v.plot(kind='box', subplots=True, layout=(3,2), sharex=False, sharey=False, figsize=[10,10])
        #    plt.savefig(f"{self.suffix}_{k}_box-plot.png")
        #    v.hist(figsize=[10,10])
        #    plt.savefig(f"{self.suffix}_{k}_histograms.png")    
        #    pandas.plotting.scatter_matrix(v, figsize=[10,10])    
        #    plt.savefig(f"{self.suffix}_{k}_scatter-matrix.png")

        #self.MultipleTable_GeneralMeanStd()
        #self.MultipleTable_SexMeanStdPerGroup()
        self.MultipleTable_Anova1Way(y_labels)

sheets = {
    "Weights": "data\\Weight_Behavior.xlsx",
    "Maze": "data\\Elevated_Maze.xlsx",
    "Nor": "data\\NOR-Test.xlsx"
}

plt.rcParams.update({'axes.titlesize': 'large'})

for i, (name, path) in enumerate(sheets.items()):
    df = pd.DataFrame(pd.read_excel(path))
    tn = TableNormalizer(df)
    specs = tn.find_tables()
    if len(specs) == 1:
        tables = tn.normalize_data_single_table()
        statistic = MedicalStatistics(name, tables, 6, table_names = ['ConG', 'JG', 'JDG', 'PG', 'PDG', 'WT'])
        statistic.ProcessSingleTable()
    else:
        tables = tn.normalize_data_multiple_tables()
        if name != "Nor":
            statistic = MedicalStatistics(name, tables, 6, table_names = ['ConG', 'JG', 'JDG', 'PG', 'PDG', 'WT'], focus_cols = {'Bdi':0, 'Bdt': 1, 'C': 2, 'Bii': 3, 'Bit': 4})
            statistic.ProcessMultipleTables(['Anova Residuals Frequency', 'Standardized Anova Residuals', ['Number of Entrances (Bdi)\nfor individuals (red dots) in each group', 'Time Spent (Bdt)\nFor individuals (red dots) in each group', 'Time Spent (C)\nFor individuals (red dots) in each group', 'Number of Entrances (Bii)\nfor individuals (red dots) in each group', 'Time Spent (Bit)\nFor individuals (red dots) in each group', ]])
        else:
            statistic = MedicalStatistics(name, tables, 6, table_names = ['ConG', 'JG', 'JDG', 'PG', 'PDG', 'WT'])
            statistic.PerformNORTest(['Anova Residuals Frequency', 'Standardized Anova Residuals', ['Object Explore Time (OF)\nfor individuals (red dots) in each group', 'Object Explore Time (ON1)\nfor individuals (red dots) in each group', 'Object Explore Time (ON24)\nfor individuals (red dots) in each group']])

