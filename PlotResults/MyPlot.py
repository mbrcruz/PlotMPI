import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import csv
import random
import statistics
from enum import Enum
from matplotlib.ticker import ScalarFormatter, LogLocator



class TypeEvaluation(Enum):
    JUST_COMUNICATION = 1,
    COMUNICATION_AND_IO = 2,
    JUST_SEND = 3,
   

    
class MyPlot(object):
    """description of class"""


    def __init__(self, base_directory,number_nodes, number_scenarios_per_node, number_scenarios, typeEvaluation= TypeEvaluation.JUST_COMUNICATION, onlyRemote=False, limit=-1):
        self.number_nodes = number_nodes 
        self.number_scenarios_per_nodes = number_scenarios_per_node
        self.number_scenarios =  number_scenarios
        self.onlyRemote = onlyRemote
                
        self.base_directory = base_directory
        self.limit=limit
        self.df_vec= []
        self.df_master= []
        self.colors={}
        self.X1=[]
        self.X2=[]
        self.X3=[]
        self.X4=[]
        self.records=[]
        self.Simulations=[]
        self.df_mpiCollective=[]
        self.mpiCollective=[]
        self.localScenarios=[]   
        self.bestScenario=0
        self.worstScenario=0 
        self.categories=[0.001,0.128,1,50]
        self.records1=[]
        self.records2=[]
        self.records3=[]
        self.records4=[]
        self.badScenarios=[]

        self.start_moment = 0       
        self.sizesPerScenario=[]     
        self.bandwidths={}
        self.numberBuffers=0
        self.start_moment=1000000000000000
        self.typeEvaluation= typeEvaluation
        self.dicionarioScenarios={}

    def random_color(self,k):
        if k not in self.colors:
            self.colors[k]='#{:06x}'.format(random.randint(0, 0xFFFFFF))            
        return self.colors[k]

    def load_data(self,filter_scenario=0,filter_experiment=0,number_experiments=5,min_size=0,max_size=0):

        if ( not self.typeEvaluation == TypeEvaluation.JUST_COMUNICATION 
            and not self.typeEvaluation == TypeEvaluation.COMUNICATION_AND_IO        
            and not self.typeEvaluation == TypeEvaluation.JUST_SEND):          
                raise Exception("Bad configuration.")
    
        for experiment in range(number_experiments): 
            
            if filter_experiment == 0 or filter_experiment == experiment+1:
                print(f'Loading Experiment {experiment+1}')
                '#initialize data structures'
                self.X1= {i: {} for i in range(1, self.number_scenarios+1 )}
                self.X2= {i: {} for i in range(1, self.number_scenarios+1 )}
                self.X3= {i: {} for i in range(1, self.number_scenarios+1 )}
                self.X4= {i: {} for i in range(1, self.number_scenarios+1 )}
                self.localScenarios=[]

                initial_rank = 2

                for i in range(self.number_nodes * self.number_scenarios_per_nodes):
                    rank= initial_rank +i      
                    if number_experiments == 1 or filter_experiment != 0:  
                        print(f'Loading Processes {i+1}')
                    df= pd.read_csv(os.path.join(self.base_directory, str(experiment+1),  f"mpiio-{rank}.log"),header=None)
                        
                    if ( self.start_moment > df.iloc[0,5] ):
                        self.start_moment= df.iloc[0,5]
                
                    for k in range(len(df)):
                        self.numberBuffers += 1
                        stage = df.iloc[k,1]
                        scenario = df.iloc[k,2]
                        file = df.iloc[k,3]
                        block = df.iloc[k,4] 
                        time = df.iloc[k,5]
                        time2 = df.iloc[k,6]
                        size = df.iloc[k,8]
                        diff= time2 - time
                        
                        if self.onlyRemote and i < self.number_scenarios_per_nodes:
                           continue
                        else:
                            if filter_scenario == 0 or scenario == filter_scenario:     
                                        record= { "experiment": experiment+1, "scenario": scenario, "stage":stage, 'file': file, 'block': block, 'sizeBytes': size  , 'timeSec': diff, 'time_start': time , "rank": rank }
                                        self.records.append(record)  
                                        #separating records by size categories   
                                        if size / (1024 * 1024) < self.categories[0]:
                                            self.records1.append(record)
                                        elif size / (1024 * 1024) < self.categories[1]: 
                                            self.records2.append(record)   
                                        elif size / (1024 * 1024)< self.categories[2]: 
                                            self.records3.append(record)
                                        else:
                                            self.records4.append(record)  

                    self.df_times= pd.read_csv(os.path.join(self.base_directory , str(experiment+1), f"sddptimer{rank:04d}.log"),header=None)
                    
                    simulation = { "experiment":  experiment+1, "rank": rank, "simulation": 0, "hourly_simulation": 0 }
                    for k in range(len(self.df_times)):
                      
                        if  self.df_times.iloc[k,0] == "Simulation":
                            simulation["simulation"]= float(self.df_times.iloc[k,1])
                        if  self.df_times.iloc[k,0] == "Hourly simulation":
                            simulation["hourly_simulation"]= float(self.df_times.iloc[k,1])
                    simulation["comunication"] = simulation["simulation"] - simulation["hourly_simulation"]
                    self.Simulations.append(simulation)

                    # load mpi collective times — tenta os dois padrões de nome
                    _exp_dir = os.path.join(self.base_directory, str(experiment+1))
                    _collective_path = os.path.join(_exp_dir, f"mpiio-collective-{rank}.log")
                    _open_path       = os.path.join(_exp_dir, f"mpiio-open-{rank}.log")
                    if os.path.exists(_collective_path):
                        # colunas: [..., col3=start, col4=end]
                        self.df_mpiCollective = pd.read_csv(_collective_path, header=None)
                        for k in range(len(self.df_mpiCollective)):
                            diff = self.df_mpiCollective.iloc[k,7] - self.df_mpiCollective.iloc[k, 6]
                            self.mpiCollective.append({"experiment": experiment+1, "timeSec": diff, "rank": rank})
                    elif os.path.exists(_open_path):
                        # colunas: [0, 1, open_time, close_time]
                        self.df_mpiCollective = pd.read_csv(_open_path, header=None)
                        for k in range(len(self.df_mpiCollective)):
                            diff = self.df_mpiCollective.iloc[k, 3] - self.df_mpiCollective.iloc[k, 2]
                            self.mpiCollective.append({"experiment": experiment+1, "timeSec": diff, "rank": rank})

        print(f'Number of Records: {len(self.records)}')             

    def show_config(self):
        print( f" Number Scenario {self.number_scenarios}")
        print( f" Base Directory {self.base_directory}")
   
    def computerMetrics(self,desabilitaEscreverCsv=False):     
        
        
            # as 2 contagens nao sao relevantes, porque os buffers tem tamanhos diferentes.
        pd.set_option("display.max_rows", None) 
        df= pd.DataFrame(self.records)
        df1= df
        df2= df
        df3= df
        df4= df
        df1= pd.DataFrame(self.records1)    
        df2= pd.DataFrame(self.records2)
        df3= pd.DataFrame(self.records3)
        df4= pd.DataFrame(self.records4)
        df_simulation = pd.DataFrame(self.Simulations)
        if self.mpiCollective is not None and len(self.mpiCollective) > 0:
            self.df_mpiCollective= pd.DataFrame(self.mpiCollective)
        else:
            self.df_mpiCollective= None
        

        sum_scenarios= df.groupby(["experiment","scenario"])["timeSec"].sum()

        #record_por_cenarios= df[~df["scenario"].isin(self.badScenarios)].groupby("scenario")["timeSec"].size().reset_index(name="num_registros").sort_values("num_registros", ascending=False)            
        # record_por_cenarios= ~df.isin(self.badScenarios).size()
        #print(record_por_cenarios)                
        avg_time_per_scenario=  sum_scenarios.groupby("experiment").mean().mean()       
        stdev_time_per_scenario = sum_scenarios.groupby("experiment").mean().std()        

        self.worstScenario= sum_scenarios.idxmax()
        self.bestScenario= sum_scenarios.idxmin()
        max_time_per_scenario = sum_scenarios.max()
        min_time_per_scenario = sum_scenarios.min() 
        
    
        #avg_simulation = statistics.mean(self.Simulations)
        avg_simulation = df_simulation.groupby("experiment")["hourly_simulation"].mean().mean()         
        stdev_simulation  = df_simulation.groupby("experiment")["hourly_simulation"].mean().std()
        

        Avg_comunication_per_process = df_simulation.groupby("experiment")["comunication"].mean().mean()   
        stdev_comunication_per_process =df_simulation.groupby("experiment")["comunication"].mean().std()

        print(f'Avg Comunication per Process(s): {Avg_comunication_per_process}')
        print(f'StdDev Comunication per Process(s): {stdev_comunication_per_process}')

        #Falta desvio padrao de comunicacao por processo
        print(f'Avg Simulation per Process(s): {avg_simulation}') 
        print(f'StdDev Simulation per Process(s): {stdev_simulation}') 
        #print(f'Number Buffers: {self.records.count}')                
        print(f'AVG per Scenarios: {avg_time_per_scenario}')        
        print(f'Stdev per Scenarios: {stdev_time_per_scenario}')  
        print(f'Max per Scenarios: {self.worstScenario} {max_time_per_scenario}') 
        print(f'Min per Scenarios: {self.bestScenario} {min_time_per_scenario}') 


        Avg_time_per_record1=0
        Stdev_time_per_record1=0    
        Avg_time_per_record2=0
        Stdev_time_per_record2=0            
        Avg_time_per_record3=0
        Stdev_time_per_record3=0
        Avg_time_per_record4=0
        Stdev_time_per_record4=0
        
        if len(self.records1) >0:
            Size_time_per_record1 = df1["timeSec"].count()/10
            Avg_time_per_record1 = df1.groupby("experiment")["timeSec"].mean().mean()
            Stdev_time_per_record1= df1.groupby("experiment")["timeSec"].mean().std()
            print(f'Count per record1 < {self.categories[0]} MB: {Size_time_per_record1}')
            print(f'AVG per record1 < {self.categories[0]} MB: {Avg_time_per_record1}') 
            print(f'Stdev per record1 < {self.categories[0]} MB: {Stdev_time_per_record1}')

        if len(self.records2) >0:
            Size_time_per_record2 = df2["timeSec"].count()/10
            Avg_time_per_record2 = df2.groupby("experiment")["timeSec"].mean().mean()
            Stdev_time_per_record2= df2.groupby("experiment")["timeSec"].mean().std()
            print(f'Count per record2 < {self.categories[1]} MB: {Size_time_per_record2}')  
            print(f'AVG per record2 < {self.categories[1]} MB: {Avg_time_per_record2}') 
            print(f'Stdev per record2 < {self.categories[1]} MB: {Stdev_time_per_record2}')

        if len(self.records3) > 0:
            Size_time_per_record3 = df3["timeSec"].count()/10
            Avg_time_per_record3 = df3.groupby("experiment")["timeSec"].mean().mean()
            Stdev_time_per_record3= df3.groupby("experiment")["timeSec"].mean().std()
            print(f'Count per record3 < {self.categories[2]} MB: {Size_time_per_record3}')
            print(f'AVG per record3 < {self.categories[2]} MB: {Avg_time_per_record3}') 
            print(f'Stdev per record3 < {self.categories[2]} MB: {  Stdev_time_per_record3}')
        if len(self.records4) > 0:
            Size_time_per_record4 = df4["timeSec"].count()/10
            Avg_time_per_record4 = df4.groupby("experiment")["timeSec"].mean().mean()
            Stdev_time_per_record4= df4.groupby("experiment")["timeSec"].mean().std()
            print(f'Count per record4 >= {self.categories[2]} MB: {Size_time_per_record4}')
            print(f'AVG per record4 >= {self.categories[2]} MB: {Avg_time_per_record4}') 
            print(f'Stdev per record4 >= {self.categories[2]} MB: {Stdev_time_per_record4}')

        
        #sum_time= sum(self.diffs)
        #Calcula o tempo medio de cada cenario e depois multiplica pelo numero de cenario executado por processo.
        Avg_io_per_process = df.groupby(["experiment","rank"])["timeSec"].sum().mean()        
        std_io_per_process = df.groupby(["experiment","rank"])["timeSec"].sum().std()        
        # # Average time per process)
        print(f'Avg IO per Process(s): {Avg_io_per_process}') 
        print(f'Std IO per Process(s): {std_io_per_process}') 
        
        
     

        
        # Banda agregada com janelas de tempo fixas para capturar concorrência real entre ranks
        WINDOW_SEC = 300
        results_bw = []
        for exp, df_exp in df.groupby("experiment"):
            t0 = df_exp["time_start"].min()
            windows = ((df_exp["time_start"] - t0) // WINDOW_SEC).astype(int)                               
            df_exp_w = df_exp.assign(window=windows)
            bytes_per_window    = df_exp_w.groupby("window")["sizeBytes"].sum()
            max_time_per_window = df_exp_w.groupby(["window","rank"])["timeSec"].sum().groupby("window").max()
            bw_per_window = bytes_per_window * 8 / 1e9 / max_time_per_window  # Gb/s
            results_bw.append({
                "experiment": exp,
                "mean_bw": bw_per_window.mean(),
                "std_bw": bw_per_window.std(ddof=1) if len(bw_per_window) > 1 else 0.0
            })
        df_bw = pd.DataFrame(results_bw).set_index("experiment")
        print(f"Banda agregada por experimento (janelas de {WINDOW_SEC}s):")
        for exp, row in df_bw.iterrows():
            print(f'  Experimento {exp}: {row["mean_bw"]:.2f} ± {row["std_bw"]:.2f} Gb/s')
        avg_agregate_bandwidth = df_bw["mean_bw"].mean()
        stddev_bandwidth       = df_bw["mean_bw"].std(ddof=1) if len(df_bw) > 1 else 0.0
        agrupados = df.groupby(["experiment","rank"])[["sizeBytes","timeSec"]].sum()
        size_por_rank = agrupados["sizeBytes"].mean() / 1e9  # em GB
        total_size_per_nodes = size_por_rank * self.number_scenarios_per_nodes  # em GB
        print(f'Total Size per Node (GB): {total_size_per_nodes:.2f}')
        print(f'AVG Aggregate Bandwidth (Gb/s): {avg_agregate_bandwidth:.2f}')
        print(f'Stdev Aggregate Bandwidth (Gb/s): {stddev_bandwidth:.2f}')


        if self.df_mpiCollective is None:
            avg_mpiCollective= 0
            stdev_mpiCollective= 0
            stdev_mpiopen=0
        else:
            sum_mpiCollective= self.df_mpiCollective.groupby(["experiment","rank"])["timeSec"].sum()
            avg_mpiCollective = sum_mpiCollective.groupby("experiment").mean().mean()
            stdev_mpiCollective = sum_mpiCollective.groupby("experiment").mean().std()
              
        print(f'AVG MPIOpen (s): {avg_mpiCollective:.2f}')        
        print(f'Stdev MPIOpen (s)): {stdev_mpiCollective:.2f}')  
        
        if not desabilitaEscreverCsv:
            print("Writing CSV file...")
            self.escreveCsv({ 'Nodes': self.number_nodes, 
                                'Avg_Simulation': avg_simulation , 'Stdev_simulation': stdev_simulation,
                                'Avg_io_per_process': Avg_io_per_process, 'Stdev_io_per_process': std_io_per_process,
                                'Avg_time_per_scenario': avg_time_per_scenario ,'Stdev_time_per_scenario': stdev_time_per_scenario,
                                'Avg_bandwidth': avg_agregate_bandwidth, 'Stddev_bandwidth': stddev_bandwidth,
                                'worstScenario': self.worstScenario, 'max_time_per_scenario': max_time_per_scenario,
                                'bestScenario': self.bestScenario, 'min_time_per_scenario': min_time_per_scenario ,
                                'Avg_time_per_record1': Avg_time_per_record1 ,'Stdev_time_per_record1': Stdev_time_per_record1,
                                'Avg_time_per_record2': Avg_time_per_record2 ,'Stdev_time_per_record2': Stdev_time_per_record2,
                                'Avg_time_per_record3': Avg_time_per_record3 ,'Stdev_time_per_record3': Stdev_time_per_record3,
                                'Avg_time_per_record4': Avg_time_per_record4 ,'Stdev_time_per_record4': Stdev_time_per_record4,
                                'Avg_comunication_per_process': Avg_comunication_per_process, 'std_comunication_per_process': stdev_comunication_per_process,
                                'Avg_mpiCollective_per_process': avg_mpiCollective, 'std_mpiCollective_per_process': stdev_mpiCollective

                                } )  
        

    def escreveCsv(self,linha):
        path_csv = os.path.join(self.base_directory,"../plot.csv")
        cabecalho = ["Nodes", 
                     "Avg_Simulation", "Stdev_simulation",
                     "Avg_io_per_process","Stdev_io_per_process",
                     "Avg_time_per_scenario",'Stdev_time_per_scenario',                     
                     "Avg_bandwidth", "Stddev_bandwidth",
                     "worstScenario", "max_time_per_scenario",
                     "bestScenario", "min_time_per_scenario",
                     "Avg_time_per_record1", "Stdev_time_per_record1",
                     "Avg_time_per_record2", "Stdev_time_per_record2",
                     "Avg_time_per_record3", "Stdev_time_per_record3",
                     "Avg_time_per_record4", "Stdev_time_per_record4",
                     "Avg_comunication_per_process", "std_comunication_per_process",
                     "Avg_mpiCollective_per_process", "std_mpiCollective_per_process"
                     ]
        escrever_cabecalho = not os.path.exists(path_csv) or os.path.getsize(path_csv) == 0
        with open(path_csv,mode='a',newline='',encoding='utf-8') as arquivo_csv:
            writer = csv.DictWriter(arquivo_csv, fieldnames=cabecalho)
            if escrever_cabecalho:
                writer.writeheader()
            writer.writerow(linha)
            arquivo_csv.close()      
            
    def plotBandwidth(self, experiments, plotLabel):
        """
        Banda agregada por configuração de nós, comparando múltiplos experimentos.

        experiments : list of (csv_path, label)
            Ex: [(r"...\\plot.csv", "Original"), (r"...\\plot.csv", "MPI-IO")]
        """
        exp_colors  = ['#1976d2', '#e53935', '#43a047', '#fb8c00', '#8e24aa']
        exp_hatches = ['', '///', '...', 'xxx']

        # Carrega todos os CSVs
        dfs = [(pd.read_csv(p, index_col='Nodes'), lbl) for p, lbl in experiments]
        all_nodes = dfs[0][0].index.tolist()
        n_nodes   = len(all_nodes)
        n_exp     = len(experiments)
        bar_w     = 0.7 / n_exp
        X         = np.arange(n_nodes)

        fig, ax = plt.subplots(figsize=(max(8, n_nodes * n_exp * 1.2), 5))

        for j, (df, lbl) in enumerate(dfs):
            color   = exp_colors[j % len(exp_colors)]
            hatch   = exp_hatches[j % len(exp_hatches)]
            bar_x   = X + (j - (n_exp - 1) / 2) * bar_w
            avg_bw  = df['Avg_bandwidth'].values
            std_bw  = df['Stddev_bandwidth'].values
            ax.bar(bar_x, avg_bw, bar_w,
                   yerr=std_bw, capsize=6,
                   color=color, edgecolor='white', hatch=hatch,
                   label=lbl,
                   error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor='#333'))
            # Valor no topo de cada barra
            for i, (bx, v) in enumerate(zip(bar_x, avg_bw)):
                ax.text(bx, v + std_bw[i] + ax.get_ylim()[1] * 0.01,
                        f'{v:.1f}', ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold', color=color)

        ax.set_xticks(X)
        ax.set_xticklabels([f'{n} Nodes' for n in all_nodes])
        ax.set_ylabel('Banda agregada média (Gb/s)')
        ax.set_xlabel('Configuração')
        ax.set_ylim(bottom=0)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.legend(fontsize=9, framealpha=0.9)
        plt.tight_layout()
        plt.show()

    def plotScenarios(self,base_directory,plotLabel):
        
        df_csv = pd.read_csv(os.path.join(base_directory,"../plot.csv"),index_col='Nodes')
        df_len = len(df_csv)
        X = np.zeros(df_len)
        categorias =  np.empty(df_len, dtype=object)
        avgLatency = np.zeros(df_len)
        stdLatency = np.zeros(df_len)
        
        
        for i in range(len(df_csv)):
            X[i]= i
            categorias[i]= f"{df_csv.index[i]} Nodes"
            avgLatency[i]= df_csv.iloc[i]['Avg_time_per_scenario']
            stdLatency[i]= df_csv.iloc[i]['Stdev_time_per_scenario']
        # Plotando com barras de erro vindas da outra série
        plt.figure(figsize=(8,5))      
        plt.bar(X, avgLatency, yerr=stdLatency, label="Tempo envio(s)", capsize=8, color='lightgreen', edgecolor='black') 

        plt.xticks(X, categorias)
        plt.ylabel('Tempo envio médio de um cenário(s)')
        plt.title(f'Tempo envio médio de um cenário(s) {plotLabel} com erro padrão')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()     
        plt.legend()  
        plt.show()

    
    def plotBlocks(self, experiments, plotLabel, number_blocks=4):
        """
        Tempo médio por categoria de tamanho de mensagem, comparando experimentos.

        experiments : list of (csv_path, label)
            Ex: [(r"...\\plot.csv", "Original"), (r"...\\plot.csv", "MPI-IO")]

        Organização do eixo X:
            Grupos = contagem de nós.
            Dentro de cada grupo: sub-grupos por categoria de tamanho,
            dentro de cada sub-grupo: uma barra por experimento.
        """
        block_colors  = ['#bdbdbd', '#4caf50', '#29b6f6', '#e53935']
        block_labels  = [
            f'até {self.categories[0]*1000:.0f} KB',
            f'até {self.categories[1]*1000:.0f} KB',
            f'até {self.categories[2]*1000:.0f} MB',
            f'até {self.categories[3]:.0f} MB',
        ]
        block_cols    = [
            ('Avg_time_per_record1', 'Stdev_time_per_record1'),
            ('Avg_time_per_record2', 'Stdev_time_per_record2'),
            ('Avg_time_per_record3', 'Stdev_time_per_record3'),
            ('Avg_time_per_record4', 'Stdev_time_per_record4'),
        ]
        exp_hatches = ['', '///']

        dfs       = [(pd.read_csv(p, index_col='Nodes'), lbl) for p, lbl in experiments]
        all_nodes = dfs[0][0].index.tolist()
        n_nodes   = len(all_nodes)
        n_exp     = len(experiments)

        bar_w       = 0.18                   # largura de cada barra individual
        blk_gap     = 0.05                   # espaço entre categorias de tamanho
        grp_gap     = 0.6                    # espaço entre grupos de nós
        blk_span    = n_exp * bar_w + blk_gap
        grp_span    = number_blocks * blk_span + grp_gap

        # Centro de cada grupo de nós
        grp_centers = np.arange(n_nodes) * grp_span

        legend_rows = number_blocks + n_exp
        fig_h = max(7.5, 6 + legend_rows * 0.35)
        fig, ax = plt.subplots(figsize=(max(10, n_nodes * number_blocks * n_exp * 0.55), fig_h))

        # Ticks no centro de cada grupo
        xtick_pos    = []
        xtick_labels = []

        for ni, node in enumerate(all_nodes):
            grp_x = grp_centers[ni]
            xtick_pos.append(grp_x + (number_blocks * blk_span) / 2 - blk_span / 2)
            xtick_labels.append(f'{node} Nodes')

            for bi in range(number_blocks):
                blk_x = grp_x + bi * blk_span
                avg_col, std_col = block_cols[bi]
                color = block_colors[bi]

                for j, (df, _) in enumerate(dfs):
                    hatch  = exp_hatches[j % len(exp_hatches)]
                    bar_x  = blk_x + (j - (n_exp - 1) / 2) * bar_w
                    avg_v  = df.loc[node, avg_col] if node in df.index else 0
                    std_v  = df.loc[node, std_col] if node in df.index else 0
                    # Entra na legenda só na primeira ocorrência
                    blk_lbl = block_labels[bi] if ni == 0 and j == 0 else '_nolegend_'
                    ax.bar(bar_x, avg_v, bar_w,
                           yerr=std_v, capsize=4,
                           color=color, edgecolor='white', hatch=hatch,
                           label=blk_lbl,
                           error_kw=dict(elinewidth=1.2, capthick=1.2, ecolor='#444'))

        # Legenda de categorias (cores) + experimentos (hachuras)
        from matplotlib.patches import Patch
        cat_h = [Patch(facecolor=c, edgecolor='white', label=l)
                 for c, l in zip(block_colors, block_labels)]
        exp_h = [Patch(facecolor='#ddd', edgecolor='#555',
                       hatch=exp_hatches[j % len(exp_hatches)], label=lbl)
                 for j, (_, lbl) in enumerate(experiments)]
        ax.legend(handles=cat_h + exp_h,
                  loc='upper center', bbox_to_anchor=(0.5, 1.12),
                  borderaxespad=0.0, fontsize=8, framealpha=0.9, ncol=2)

        ax.set_yscale('log')
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, rotation=25, ha='right')
        ax.set_ylabel('Tempo médio de envio (s) — escala logarítmica')
        ax.set_title(plotLabel)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show()


    def plotExecutionTime(self, experiments, plotLabel):
        """
        Stacked bar com alturas reais (s), barras lado a lado por experimento.

        experiments : list of (csv_path, label)
            Ex: [(r"...\\plot.csv", "Implementação atual"), (r"...\\plot.csv", "MPI-IO")]
        plotLabel   : título do gráfico

        Segmentos grandes  (>= large_threshold) → tempo dentro da pilha
        Percentuais de todos os segmentos       → zona acima das barras
        """
        from matplotlib.patches import Patch

        seg_colors      = ['#4caf50', '#1976d2', '#e57373', '#fbc02d']
        seg_labels_txt  = ['Computação', 'Comunicação', 'E/S', 'Coletiva MPI']
        exp_hatches     = ['', '///']
        exp_txt_colors  = ['#1a1a1a', '#c62828', '#1565c0', '#2e7d32']
        dark_bg         = {'#1976d2'}
        label_box       = dict(boxstyle='round,pad=0.18',
                               facecolor='white', edgecolor='none',
                               alpha=0.92)

        def _col(df, *names):
            """Retorna df[name] para o primeiro nome encontrado nas colunas."""
            for name in names:
                if name in df.columns:
                    return df[name].values
            raise KeyError(f'Nenhuma das colunas encontrada: {names}')

        def _load(csv_path):
            df         = pd.read_csv(csv_path, index_col='Nodes')
            avg_sim    = _col(df, 'Avg_Simulation')
            avg_io     = _col(df, 'Avg_io_per_process')
            avg_coll   = _col(df, 'Avg_mpiCollective_per_process',
                                   'Avg_mpiopen_per_process') * 6
            avg_comm   = _col(df, 'Avg_comunication_per_process')
            std_sim    = _col(df, 'Stdev_simulation')
            std_io     = _col(df, 'Stdev_io_per_process')
            std_comm   = _col(df, 'std_comunication_per_process')
            std_coll   = _col(df, 'std_mpiCollective_per_process',
                                   'std_mpiopen_per_process')
            avg_comp   = np.maximum(avg_sim - avg_io - avg_coll, 0)
            std_comp   = np.sqrt(np.maximum(std_sim**2 - std_io**2 - std_coll**2, 0))
            total      = avg_comp + avg_comm + avg_io + avg_coll
            return (df.index.tolist(),
                    [avg_comp, avg_comm, avg_io, avg_coll],
                    [std_comp, std_comm, std_io, std_coll],
                    total)

        loaded     = [_load(p) for p, _ in experiments]
        all_nodes  = loaded[0][0]
        n_nodes    = len(all_nodes)
        n_exp      = len(experiments)
        bar_w      = 0.35
        grp_gap    = 0.5
        X          = np.arange(n_nodes) * (n_exp * bar_w + grp_gap)

        global_max    = max(float(t.max()) for _, _, _, t in loaded)
        # Thresholds baseados no percentual da própria barra (pct),
        # não no valor absoluto — assim segmentos grandes em barras curtas
        # (ex: 32 nodes) também recebem o label completo.
        PCT_LARGE = 12.0   # pct >= 12% → tempo dentro da pilha

        legend_mask = [
            any(np.any(loaded[j][1][s] > 0) for j in range(n_exp))
            for s in range(len(seg_labels_txt))
        ]

        # Máximo total por grupo de nós — baseline para a zona de labels
        group_max = np.zeros(n_nodes)
        for _, _, _, total in loaded:
            group_max = np.maximum(group_max, total)

        line_h    = global_max * 0.06
        clearance = global_max * 0.08

        fig, ax = plt.subplots(figsize=(max(10, n_nodes * (n_exp * bar_w + grp_gap) * 2.2), 9.5))
        ax2 = ax.twinx()

        # pct_labels   : bar_x → {'gi': group_index, 'items': [(txt, color)]}
        # totals_list  : [(bar_x_array, total_array, j)] — renderizados por último
        pct_labels   = {}
        totals_list  = []

        for j, ((_, exp_label), (nodes, seg_v, seg_s, total)) in enumerate(
                zip(experiments, loaded)):
            hatch  = exp_hatches[j % len(exp_hatches)]
            bar_x  = X + (j - (n_exp - 1) / 2) * bar_w
            bots   = np.zeros(n_nodes)

            for s, (vals, stds, color, slbl, ok) in enumerate(
                    zip(seg_v, seg_s, seg_colors, seg_labels_txt, legend_mask)):
                lbl = slbl if (ok and j == 0) else '_nolegend_'
                ax.bar(bar_x, vals, bar_w, bottom=bots,
                       color=color, edgecolor='white', linewidth=0.5,
                       hatch=hatch, label=lbl)

                for i in range(n_nodes):
                    if vals[i] <= 0:
                        continue
                    pct = vals[i] / total[i] * 100 if total[i] > 0 else 0
                    cy  = bots[i] + vals[i] / 2
                    std = max(float(stds[i]), 0.0)
                    if std > 0:
                        # Draw segment std above its own box, offset from the
                        # centered time label.
                        seg_yerr = min(std, float(vals[i]) * 0.45)
                        err_x = bar_x[i] + bar_w * 0.30
                        err_y = bots[i] + vals[i]
                        ax.errorbar(err_x, err_y,
                                    yerr=np.array([[0.0], [seg_yerr]]),
                                    fmt='none', ecolor='#444',
                                    elinewidth=1.2, capthick=1.2,
                                    capsize=3, zorder=9, clip_on=True)
                    key = round(bar_x[i], 8)
                    if key not in pct_labels:
                        pct_labels[key] = {'gi': i, 'items': []}
                    pct_labels[key]['items'].append((f'{pct:.1f}%', color))

                    if pct >= PCT_LARGE:
                        # Segmento grande: apenas o tempo dentro da pilha.
                        tc = 'white' if color in dark_bg else 'black'
                        ax.text(bar_x[i], cy, f'{vals[i]:.0f}s',
                                ha='center', va='center',
                                fontsize=8.5, fontweight='bold', color=tc,
                                clip_on=True, zorder=10)
                bots += vals

            totals_list.append((bar_x.copy(), total.copy(), j))

        # ── Passo 2: zona de percentuais, rastreando topo por grupo ───────────
        # group_top[i] = próxima y disponível para o grupo i após pct_labels
        group_top = group_max + clearance   # ponto de partida de cada grupo

        for bx, data in pct_labels.items():
            gi = data['gi']
            y  = group_max[gi] + clearance
            for txt, color in data['items']:
                ax.text(bx, y, txt,
                        ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold', color='black',
                        zorder=11,
                        bbox={**label_box, 'edgecolor': color, 'linewidth': 0.9})
                y += line_h * 0.85
            if y > group_top[gi]:
                group_top[gi] = y

        # ── Passo 3: labels de total acima da zona de percentuais ─────────────
        # Empilhados por experimento dentro de cada grupo, sem colidir com nada.
        for bar_x, total, j in totals_list:
            tc_exp = exp_txt_colors[j % len(exp_txt_colors)]
            for i, tot in enumerate(total):
                y = group_top[i] + j * line_h * 0.85
                ax.text(bar_x[i], y, f'{tot:.0f}s',
                        ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold', color=tc_exp)

        max_label_top = max(
            group_top[i] + n_exp * line_h * 0.85
            for i in range(n_nodes)
        )

        # ── Eixos ─────────────────────────────────────────────────────────────
        ylim_top = max(global_max * 1.4, max_label_top + line_h)
        ax.set_ylim(0, ylim_top)
        ax2.set_ylim(0, ylim_top / global_max * 100)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
        ax2.set_ylabel('Proporção do tempo total (%)', color='#555')
        ax2.tick_params(axis='y', labelcolor='#555')

        ax.set_xticks(X)
        ax.set_xticklabels([f'{nd} Nodes' for nd in all_nodes])
        ax.set_ylabel('Tempo médio por processo (s)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        # ── Legenda ───────────────────────────────────────────────────────────
        seg_h = [Patch(facecolor=c, edgecolor='white', label=l)
                 for c, l, m in zip(seg_colors, seg_labels_txt, legend_mask) if m]
        exp_h = [Patch(facecolor='#ddd', edgecolor='#555',
                       hatch=exp_hatches[j % len(exp_hatches)], label=lbl)
                 for j, (_, lbl) in enumerate(experiments)]
        ax.legend(handles=seg_h + exp_h,
                  loc='upper left', fontsize=8, framealpha=0.9, ncol=2)

        plt.tight_layout()
        plt.show()

    # def plotExecutionTimeComparison(self, experiments, plotLabel):
    #     """
    #     Compara tempo de execução entre experimentos (ex: centralizado vs descentralizado).

    #     Parameters
    #     ----------
    #     experiments : list of (csv_path, label)
    #         Cada entrada é o caminho direto para o plot.csv e um rótulo descritivo.
    #     plotLabel : str
    #         Título do gráfico.

    #     Exemplo de uso
    #     --------------
    #     p.plotExecutionTimeComparison([
    #         (r"...\\AWS\\Lustre - 1024 Series - Sem rede\\plot.csv",       "Original"),
    #         (r"...\\AWS\\Lustre - 1024 Series - Sem rede - MPIO\\plot.csv", "MPIO"),
    #         (r"...\\AWS\\Lustre - 1024 Series - Sem rede - MPIO ASYNC\\plot.csv", "MPIO Async"),
    #     ], "AWS")
    #     """
    #     from matplotlib.patches import Patch

    #     dfs = {}
    #     for csv_path, label in experiments:
    #         dfs[label] = pd.read_csv(csv_path, index_col='Nodes')

    #     all_nodes   = sorted(set.union(*[set(df.index) for df in dfs.values()]))
    #     n_nodes     = len(all_nodes)
    #     n_exp       = len(experiments)
    #     width       = 0.7 / n_exp
    #     X           = np.arange(n_nodes)

    #     seg_colors  = ['lightgreen', 'steelblue', 'salmon', 'gold']
    #     seg_labels  = ['Computação', 'Comunicação', 'E/S', 'Coletiva MPI']
    #     exp_hatches = ['', '///', '...', 'xxx']

    #     fig, ax = plt.subplots(figsize=(13, 6))

    #     for j, (_, label) in enumerate(experiments):
    #         df      = dfs[label]
    #         offset  = (j - n_exp / 2 + 0.5) * width
    #         hatch   = exp_hatches[j % len(exp_hatches)]

    #         comp_v = []; comm_v = []; io_v = []; mpio_v = []
    #         for node in all_nodes:
    #             if node in df.index:
    #                 row      = df.loc[node]
    #                 avg_sim  = row['Avg_Simulation']
    #                 avg_io   = row['Avg_io_per_process']
    #                 avg_mpio = row['Avg_mpiopen_per_process']
    #                 avg_comm = row['Avg_comunication_per_process']
    #                 avg_comp = max(avg_sim - avg_io - avg_mpio, 0)
    #             else:
    #                 avg_comp = avg_comm = avg_io = avg_mpio = 0
    #             comp_v.append(avg_comp); comm_v.append(avg_comm)
    #             io_v.append(avg_io);     mpio_v.append(avg_mpio)

    #         segs    = [np.array(v) for v in [comp_v, comm_v, io_v, mpio_v]]
    #         totals  = sum(segs)
    #         bottoms = np.zeros(n_nodes)

    #         for vals, color in zip(segs, seg_colors):
    #             ax.bar(X + offset, vals, width, bottom=bottoms,
    #                    color=color, edgecolor='black', hatch=hatch,
    #                    label='_nolegend_')
    #             bottoms += vals

    #         # Total e rótulo do experimento no topo de cada barra
    #         for i, (x, tot) in enumerate(zip(X + offset, totals)):
    #             if tot > 0:
    #                 ax.text(x, tot * 1.005, f'{label}\n{tot:.0f}s',
    #                         ha='center', va='bottom', fontsize=7)

    #     # Legenda: cores = componentes, hachuras = experimentos
    #     color_handles = [Patch(facecolor=c, edgecolor='black', label=l)
    #                      for c, l in zip(seg_colors, seg_labels)]
    #     hatch_handles = [Patch(facecolor='white', edgecolor='black',
    #                            hatch=exp_hatches[j % len(exp_hatches)],
    #                            label=lbl)
    #                      for j, (_, lbl) in enumerate(experiments)]
    #     ax.legend(handles=color_handles + hatch_handles,
    #               loc='upper right', fontsize=8, ncol=2)

    #     ax.set_xticks(X)
    #     ax.set_xticklabels([f"{n} Nodes" for n in all_nodes])
    #     ax.set_ylabel('Tempo médio por processo (s)')
    #     ax.set_title(f'Comparação de Experimentos — {plotLabel}')
    #     ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    #     plt.tight_layout()
    #     plt.show()
    
    def PlotHistogram(self,max_size_kb=0):

        smallSizes=[]
        for row in self.records:
            smallSizesInKb= row['sizeBytes']/1024
            if max_size_kb == 0 or smallSizesInKb < max_size_kb:
                smallSizes.append(smallSizesInKb)                
        plt.figure(figsize=(8,5))
        plt.hist(smallSizes, bins=500, color='blue', alpha=0.7, edgecolor='black')

        ax = plt.gca()

        # Coloca ticks principais em potências de 10
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=None))
        # Formata os ticks como números decimais normais
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='x')   # evita notação científica
        plt.xscale('log')  
        plt.yscale('log')   
        #plt.xlim(left=0.6, right=50000)
        #plt.xticks([1, 1000, 10000, 20000,30000,50000])
        ticks = [1, 1000, 50000]
        plt.xticks(ticks, [str(t) for t in ticks])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plotScatter(self,displotLabel,records1, records2):
        
        plt.figure(figsize=(8,6))

        
        sizes1=[]
        diffs1=[]

        for row1 in records1:
            sizes1.append(row1['sizeBytes'])
            diffs1.append(row1['timeSec'])
        sizes2=[]
        diffs2=[]
        for row2 in records2:
            sizes2.append(row2['sizeBytes'])
            diffs2.append(row2['timeSec'])

        # # Scatter plot
        jitter_x = np.array(diffs1) + (np.random.rand(len(diffs1)) - 0.5) * 0.1  # Adiciona um pequeno jitter no eixo y
        plt.scatter( jitter_x,sizes1, color="green", label="Melhor caso com 2 nós", s=10, alpha=0.7, edgecolors='k')
        jitter_y = np.array(sizes2) + (np.random.rand(len(sizes2)) - 0.5)  # Adiciona um pequeno jitter no eixo x
        plt.scatter(diffs2, jitter_y,  color="yellow",  label="Pior caso com 32 nós", s=10, alpha=0.7, edgecolors='k')

        plt.title("Tempo envio x tamanho do Buffer no "+ displotLabel )
        plt.xlabel("Tempo de envio (s)")
        plt.ylabel("Tamanho do Buffer (Bytes)")

        # Escala log no eixo X ajuda a visualizar melhor (opcional)
        plt.xscale('log')
        #plt.ticklabel_format(axis='y', style='sci')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.show()














