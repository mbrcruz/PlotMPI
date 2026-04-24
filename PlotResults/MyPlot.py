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
        self.df_mpiOpenComunication=[]
        self.mpiOpenComunication=[]
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

    def load_data(self,filter_scenario=0,filter_experiment=0,number_experiments=10,min_size=0,max_size=0):

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

                    # load mpi comunication times if exist
                    if os.path.exists(os.path.join(self.base_directory , str(experiment+1), f"mpiio-open-{rank}.log")):
                        self.df_mpiOpenComunication= pd.read_csv(os.path.join(self.base_directory , str(experiment+1), f"mpiio-open-{rank}.log"),header=None)
                        for k in range(len(self.df_mpiOpenComunication)):
                            mpiopenDiff= self.df_mpiOpenComunication.iloc[k,3] - self.df_mpiOpenComunication.iloc[k,2]
                            mpiOpenTimeRecord={}
                            mpiOpenTimeRecord= { "experiment": experiment+1, 'timeSec': mpiopenDiff , "rank": rank }  
                            self.mpiOpenComunication.append(mpiOpenTimeRecord)  

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
        if self.mpiOpenComunication is not None and len(self.mpiOpenComunication) > 0:
            self.df_mpiOpenComunication= pd.DataFrame(self.mpiOpenComunication)
        else:
            self.df_mpiOpenComunication= None
        

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


        if self.df_mpiOpenComunication is None:
            avg_mpiopen= 0
            stdev_mpiopen=0
        else:
            sum_mpiopen= self.df_mpiOpenComunication.groupby(["experiment","rank"])["timeSec"].sum()
            avg_mpiopen = sum_mpiopen.groupby("experiment").mean().mean()
            stdev_mpiopen = sum_mpiopen.groupby("experiment").mean().std()      
              
        print(f'AVG MPIOpen (s): {avg_mpiopen:.2f}') 
        print(f'Stdev MPIOpen (s)): {stdev_mpiopen:.2f}')        
        
        
        
        
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
                                'Avg_mpiopen_per_process': avg_mpiopen, 'std_mpiopen_per_process': stdev_mpiopen

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
                     "Avg_mpiopen_per_process", "std_mpiopen_per_process"
                     ]
        escrever_cabecalho = not os.path.exists(path_csv) or os.path.getsize(path_csv) == 0
        with open(path_csv,mode='a',newline='',encoding='utf-8') as arquivo_csv:
            writer = csv.DictWriter(arquivo_csv, fieldnames=cabecalho)
            if escrever_cabecalho:
                writer.writeheader()
            writer.writerow(linha)
            arquivo_csv.close()      
            
    def plotBandwidth(self,base_directory,plotLabel):
        
        if self.records is None or len(self.records) == 0:
            self.load_data(number_experiments=1)
            
        df = pd.DataFrame(self.records)

        # calcula tamanho médio por cenário (GB)
        agrupados = df.groupby("scenario")[["sizeBytes"]].sum()
        size_por_scenario = agrupados["sizeBytes"].mean() / 1e9

        # carrega CSV com métricas por configuração
        df_csv = pd.read_csv(os.path.join(base_directory, "../plot.csv"), index_col='Nodes')
        df_len = len(df_csv)
        X = np.arange(df_len)
        categorias = [f"{n} Nodes" for n in df_csv.index]        
        avgBandwidth = df_csv['Avg_bandwidth'].to_numpy()
        #avgBandwidth = avgBandwidthCenario
        stdDevBandwidthScenario = df_csv['Stddev_bandwidth'].to_numpy()
        stdDevBandwidth = stdDevBandwidthScenario

        # calcula volume de dados por nó (GB) para segunda eixo y
        sizePerNode = np.zeros(df_len)
        for i in range(df_len):
            total_size_per_nodes = (size_por_scenario * self.number_scenarios) / 2 ** ( i+1)
            sizePerNode[i] = total_size_per_nodes

        # plot com dois eixos y
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.bar(X, avgBandwidth, yerr=stdDevBandwidth, capsize=8, color='blue', edgecolor='black', label='Banda (Gb/s)')
        ax1.set_ylabel('Banda agregada média (Gb/s)', color='green')
        ax1.set_xlabel('Configuração')
        ax1.set_xticks(X)
        ax1.set_xticklabels(categorias)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax1.set_ylim(bottom=0)

        plt.tight_layout()
        plt.show()
        return        

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

    
    def plotBlocks(self,base_directory,plotLabel,number_blocks=4):
        
        df_csv = pd.read_csv(os.path.join(base_directory,"plot.csv"),index_col='Nodes')
        number_conf = len(df_csv)
        categorias =  np.empty(number_conf, dtype=object)       
        xTicks = np.zeros(number_conf)      
       

        X= np.zeros((number_blocks,number_conf))
        avgLatency = np.zeros((number_blocks,number_conf))
        stdLatency = np.zeros((number_blocks,number_conf))

        # Plotando com barras de erro vindas da outra série
        space_between=5
        for i in range(0,number_conf):     
            xTicks[i]= i*space_between
            categorias[i]= f"{df_csv.index[i]} Nodes"       
            X[0][i] = i*space_between
            X[1][i] = i*space_between+1
            X[2][i] = i*space_between+2
            X[3][i] = i*space_between+3 
            avgLatency[0][i] = df_csv.iloc[i]['Avg_time_per_record1']
            stdLatency[0][i] = df_csv.iloc[i]['Stdev_time_per_record1']    
            avgLatency[1][i] = df_csv.iloc[i]['Avg_time_per_record2']
            stdLatency[1][i] = df_csv.iloc[i]['Stdev_time_per_record2']
            avgLatency[2][i] = df_csv.iloc[i]['Avg_time_per_record3']
            stdLatency[2][i] = df_csv.iloc[i]['Stdev_time_per_record3']
            avgLatency[3][i] = df_csv.iloc[i]['Avg_time_per_record4']
            stdLatency[3][i] = df_csv.iloc[i]['Stdev_time_per_record4'] 
           


        plt.figure(figsize=(8,5))
        for i in range(0,number_blocks):
            if i == 0:
                color='lightyellow'
                label=f"Messagem de até 1 KB" 
            elif i == 1:
                color='lightgreen'
                label=f"Messagem de até 128 KB" 
            elif i == 2:
                color='deepskyblue' 
                label=f"Messagem de até {self.categories[2]} MB" 
            else:
                color='darkred' 
                label=f"Messagem de até {self.categories[3]} MB" 
                     
            plt.bar(X[i], avgLatency[i], yerr=stdLatency[i], label=label, width=0.9,capsize=8, color=color, edgecolor='black') 

       
        plt.yscale("log")
        plt.xticks(xTicks, categorias)
        plt.ylabel('Tempo médio(s) do envio em escala logaritmica')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)

        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)
        plt.tight_layout()

        plt.show()


    def plotExecutionTime(self, experiments, plotLabel):
        """
        Stacked bar com alturas reais (s), barras lado a lado por experimento.

        Parameters
        ----------
        experiments : list of (csv_path, label)
            Caminho direto para plot.csv e rótulo descritivo.
            Exemplo:
                [
                    (r"...\\Sem rede\\plot.csv",       "Centralizado"),
                    (r"...\\Sem rede - MPIO\\plot.csv", "MPI-IO"),
                ]
        plotLabel : str  — título do gráfico

        Cada grupo no eixo X representa uma contagem de nós.
        Dentro do grupo há uma barra por experimento, lado a lado.
        """
        seg_colors = ['#4caf50', '#1976d2', '#e57373', '#fbc02d']
        seg_labels = ['Computação', 'Comunicação', 'E/S', 'Coletiva MPI']
        exp_hatches = ['', '///', '...', 'xxx']
        dark_text_colors = {'#1976d2'}   # segmentos com fundo escuro → texto branco

        def _load(csv_path):
            df = pd.read_csv(csv_path, index_col='Nodes')
            avg_sim    = df['Avg_Simulation'].values
            avg_io     = df['Avg_io_per_process'].values
            avg_mpio   = df['Avg_mpiopen_per_process'].values * 6
            avg_comm   = df['Avg_comunication_per_process'].values
            stdev_sim  = df['Stdev_simulation'].values
            stdev_io   = df['Stdev_io_per_process'].values
            stdev_comm = df['std_comunication_per_process'].values
            stdev_mpio = df['std_mpiopen_per_process'].values
            avg_comp   = np.maximum(avg_sim - avg_io - avg_mpio, 0)
            stdev_comp = np.sqrt(np.maximum(stdev_sim**2 - stdev_io**2 - stdev_mpio**2, 0))
            total      = avg_comp + avg_comm + avg_io + avg_mpio
            seg_v = [avg_comp, avg_comm, avg_io, avg_mpio]
            seg_s = [stdev_comp, stdev_comm, stdev_io, stdev_mpio]
            return df.index.tolist(), seg_v, seg_s, total

        # Carrega todos os CSVs e determina nós em comum (ordem do primeiro)
        loaded = [_load(p) for p, _ in experiments]
        all_nodes = loaded[0][0]   # preserva ordem do primeiro CSV
        n_nodes   = len(all_nodes)
        n_exp     = len(experiments)
        width     = 0.75 / n_exp
        gap       = 0.05           # espaço entre grupos
        X         = np.arange(n_nodes) * (0.75 + gap + 0.1)

        # Limiar global para decidir label dentro vs seta
        global_max = max(t.max() for _, _, _, t in loaded)
        vis_threshold = global_max * 0.04

        # Máscara de legenda de segmento: só entra se algum experimento tem valor > 0
        legend_mask = [
            any(np.any(loaded[j][1][s] > 0) for j in range(n_exp))
            for s in range(len(seg_labels))
        ]

        fig, ax = plt.subplots(figsize=(max(11, n_nodes * n_exp * 1.8), 7))
        ax2 = ax.twinx()

        # Coleta todos os bar_info para depois empilhar os labels acima das barras
        # sem cruzamentos: cada barra tem sua própria fila de labels pequenos
        all_small = {}   # (bar_x_val, i) -> lista de (y_base, label_txt, color)

        for j, ((_, exp_label), (nodes, seg_v, seg_s, total)) in enumerate(
                zip(experiments, loaded)):
            hatch   = exp_hatches[j % len(exp_hatches)]
            bar_off = (j - n_exp / 2 + 0.5) * width
            bar_x   = X + bar_off

            bottoms  = np.zeros(n_nodes)
            bar_info = []

            for s_idx, (vals, stds, color, slbl, in_leg) in enumerate(
                    zip(seg_v, seg_s, seg_colors, seg_labels, legend_mask)):
                legend_lbl = slbl if (in_leg and j == 0) else '_nolegend_'
                ax.bar(bar_x, vals, width, bottom=bottoms,
                       color=color, edgecolor='white', linewidth=0.6,
                       hatch=hatch, label=legend_lbl,
                       yerr=stds, capsize=4,
                       error_kw=dict(elinewidth=1.2, capthick=1.2, ecolor='#555'))
                bar_info.append((vals, bottoms.copy()))
                bottoms += vals

            for s_idx, (vals, bot) in enumerate(bar_info):
                color = seg_colors[s_idx]
                for i in range(n_nodes):
                    if vals[i] == 0:
                        continue
                    pct = vals[i] / total[i] * 100 if total[i] > 0 else 0
                    label_txt = f'{pct:.1f}%\n{vals[i]:.1f}s'
                    seg_cy = bot[i] + vals[i] / 2

                    if vals[i] >= vis_threshold:
                        # Label dentro do segmento
                        tc = 'white' if color in dark_text_colors else 'black'
                        ax.text(bar_x[i], seg_cy, label_txt,
                                ha='center', va='center',
                                fontsize=7, fontweight='bold', color=tc)
                    else:
                        # Acumula para renderizar empilhado acima da barra
                        key = (round(bar_x[i], 6), i)
                        all_small.setdefault(key, {'total': total[i], 'items': []})
                        all_small[key]['items'].append((label_txt, color))

            # Rótulo do experimento + total no topo de cada barra
            for i, tot in enumerate(total):
                ax.text(bar_x[i], tot + global_max * 0.005,
                        f'{exp_label}  {tot:.0f}s',
                        ha='center', va='bottom', fontsize=7,
                        fontweight='bold', color='#222')

        # Renderiza labels pequenos empilhados acima do topo, sem setas nem cruzamentos
        line_h = global_max * 0.055   # altura de cada linha de texto
        for key, data in all_small.items():
            bx    = key[0]
            y_cur = data['total'] + global_max * 0.09   # começa logo acima do rótulo do total
            for label_txt, color in data['items']:
                ax.text(bx, y_cur, label_txt,
                        ha='center', va='bottom',
                        fontsize=6.5, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                  ec=color, lw=0.8, alpha=0.85))
                y_cur += line_h * (label_txt.count('\n') + 1)

        # Eixo direito espelhado
        ax.set_ylim(0, global_max * 1.55)
        ax2.set_ylim(0, 100 * 1.55)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
        ax2.set_ylabel('Proporção do tempo total (%)', color='#555')
        ax2.tick_params(axis='y', labelcolor='#555')

        ax.set_xticks(X)
        ax.set_xticklabels([f'{nd} Nodes' for nd in all_nodes])
        ax.set_ylabel('Tempo médio por processo (s)')
        ax.set_title(f'Decomposição do Tempo de Execução — {plotLabel}', fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.35)

        # Legenda: cores = segmentos + hachuras = experimentos
        from matplotlib.patches import Patch
        seg_handles = [Patch(facecolor=c, edgecolor='white', label=l)
                       for c, l, m in zip(seg_colors, seg_labels, legend_mask) if m]
        exp_handles = [Patch(facecolor='#ccc', edgecolor='black',
                             hatch=exp_hatches[j % len(exp_hatches)], label=lbl)
                       for j, (_, lbl) in enumerate(experiments)]
        ax.legend(handles=seg_handles + exp_handles,
                  loc='upper right', fontsize=8, framealpha=0.9, ncol=2)

        plt.tight_layout()
        plt.show()

    def plotExecutionTimeComparison(self, experiments, plotLabel):
        """
        Compara tempo de execução entre experimentos (ex: centralizado vs descentralizado).

        Parameters
        ----------
        experiments : list of (csv_path, label)
            Cada entrada é o caminho direto para o plot.csv e um rótulo descritivo.
        plotLabel : str
            Título do gráfico.

        Exemplo de uso
        --------------
        p.plotExecutionTimeComparison([
            (r"...\\AWS\\Lustre - 1024 Series - Sem rede\\plot.csv",       "Centralizado"),
            (r"...\\AWS\\Lustre - 1024 Series - Sem rede - MPIO\\plot.csv", "MPIO"),
            (r"...\\AWS\\Lustre - 1024 Series - Sem rede - MPIO ASYNC\\plot.csv", "MPIO Async"),
        ], "AWS")
        """
        from matplotlib.patches import Patch

        dfs = {}
        for csv_path, label in experiments:
            dfs[label] = pd.read_csv(csv_path, index_col='Nodes')

        all_nodes   = sorted(set.union(*[set(df.index) for df in dfs.values()]))
        n_nodes     = len(all_nodes)
        n_exp       = len(experiments)
        width       = 0.7 / n_exp
        X           = np.arange(n_nodes)

        seg_colors  = ['lightgreen', 'steelblue', 'salmon', 'gold']
        seg_labels  = ['Computação', 'Comunicação', 'E/S', 'Coletiva MPI']
        exp_hatches = ['', '///', '...', 'xxx']

        fig, ax = plt.subplots(figsize=(13, 6))

        for j, (_, label) in enumerate(experiments):
            df      = dfs[label]
            offset  = (j - n_exp / 2 + 0.5) * width
            hatch   = exp_hatches[j % len(exp_hatches)]

            comp_v = []; comm_v = []; io_v = []; mpio_v = []
            for node in all_nodes:
                if node in df.index:
                    row      = df.loc[node]
                    avg_sim  = row['Avg_Simulation']
                    avg_io   = row['Avg_io_per_process']
                    avg_mpio = row['Avg_mpiopen_per_process']
                    avg_comm = row['Avg_comunication_per_process']
                    avg_comp = max(avg_sim - avg_io - avg_mpio, 0)
                else:
                    avg_comp = avg_comm = avg_io = avg_mpio = 0
                comp_v.append(avg_comp); comm_v.append(avg_comm)
                io_v.append(avg_io);     mpio_v.append(avg_mpio)

            segs    = [np.array(v) for v in [comp_v, comm_v, io_v, mpio_v]]
            totals  = sum(segs)
            bottoms = np.zeros(n_nodes)

            for vals, color in zip(segs, seg_colors):
                ax.bar(X + offset, vals, width, bottom=bottoms,
                       color=color, edgecolor='black', hatch=hatch,
                       label='_nolegend_')
                bottoms += vals

            # Total e rótulo do experimento no topo de cada barra
            for i, (x, tot) in enumerate(zip(X + offset, totals)):
                if tot > 0:
                    ax.text(x, tot * 1.005, f'{label}\n{tot:.0f}s',
                            ha='center', va='bottom', fontsize=7)

        # Legenda: cores = componentes, hachuras = experimentos
        color_handles = [Patch(facecolor=c, edgecolor='black', label=l)
                         for c, l in zip(seg_colors, seg_labels)]
        hatch_handles = [Patch(facecolor='white', edgecolor='black',
                               hatch=exp_hatches[j % len(exp_hatches)],
                               label=lbl)
                         for j, (_, lbl) in enumerate(experiments)]
        ax.legend(handles=color_handles + hatch_handles,
                  loc='upper right', fontsize=8, ncol=2)

        ax.set_xticks(X)
        ax.set_xticklabels([f"{n} Nodes" for n in all_nodes])
        ax.set_ylabel('Tempo médio por processo (s)')
        ax.set_title(f'Comparação de Experimentos — {plotLabel}')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    
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
        plt.title('Histograma do tamanho das mensagens enviadas dentro de 1 cenário.')
        plt.xlabel('Tamanho (KBytes) em escala logarítmica')
        plt.ylabel('Frequência')
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














