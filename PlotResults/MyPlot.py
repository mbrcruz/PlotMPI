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
        self.df_mpiComunication=[]
        self.mpiComunication=[]
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
                    for k in range(len(self.df_times)):
                        if  self.df_times.iloc[k,0] == "Simulation":
                            self.Simulations.append(float(self.df_times.iloc[k,1]))   

                    # load mpi comunication times if exist
                    if os.path.exists(os.path.join(self.base_directory , str(experiment+1), f"mpiio-open-{rank}.log")):
                        self.df_mpiComunication= pd.read_csv(os.path.join(self.base_directory , str(experiment+1), f"mpiio-open-{rank}.log"),header=None)
                        for k in range(len(self.df_mpiComunication)):
                            mpiopenDiff= self.df_mpiComunication.iloc[k,3] - self.df_mpiComunication.iloc[k,2]
                            mpiOpenTimeRecord={}
                            mpiOpenTimeRecord= { "experiment": experiment+1, 'timeSec': mpiopenDiff , "rank": rank }  
                            self.mpiComunication.append(mpiOpenTimeRecord)  

        print(f'Number of Records: {len(self.records)}')             

    def show_config(self):
        print( f" Number Scenario {self.number_scenarios}")
        print( f" Base Directory {self.base_directory}")
   
    def computerMetrics(self):     
        
        
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
        if self.mpiComunication is not None and len(self.mpiComunication) > 0:
            self.df_mpiComunication= pd.DataFrame(self.mpiComunication)
        else:
            self.df_mpiComunication= None

        sum_scenarios= df.groupby(["experiment","scenario"])["timeSec"].sum()

        #record_por_cenarios= df[~df["scenario"].isin(self.badScenarios)].groupby("scenario")["timeSec"].size().reset_index(name="num_registros").sort_values("num_registros", ascending=False)            
        # record_por_cenarios= ~df.isin(self.badScenarios).size()
        #print(record_por_cenarios)                
        avg_time_per_scenario=  sum_scenarios.groupby("experiment").mean().mean()
        print(avg_time_per_scenario)
        stdev_time_per_scenario = sum_scenarios.groupby("experiment").mean().std()
        print(stdev_time_per_scenario)

        self.worstScenario= sum_scenarios.idxmax()
        self.bestScenario= sum_scenarios.idxmin()
        max_time_per_scenario = sum_scenarios.max()
        min_time_per_scenario = sum_scenarios.min() 
        
    
        avg_simulation = statistics.mean(self.Simulations)
        stdev_simulation  = statistics.stdev(self.Simulations)
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
        avg_per_process = df.groupby(["experiment","rank"])["timeSec"].sum().mean()        
        std_per_process = df.groupby(["experiment","rank"])["timeSec"].sum().std()        
        # # Average time per process)
        print(f'Avg Comunication per Process(s): {avg_per_process}') 
        print(f'Std Comunication per Process(s): {std_per_process}') 
        #Falta desvio padrao de comunicacao por processo
        print(f'Avg Simulation per Process(s): {avg_simulation}') 
        print(f'StdDev Simulation per Process(s): {stdev_simulation}') 

        
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


        if self.df_mpiComunication is None:
            avg_mpiopen= 0
            stdev_mpiopen=0
        else:
            sum_mpiopen= self.df_mpiComunication.groupby(["experiment","rank"])["timeSec"].sum()
            avg_mpiopen = sum_mpiopen.groupby("experiment").mean().mean()
            stdev_mpiopen = sum_mpiopen.groupby("experiment").mean().std()      
              
        print(f'AVG MPIOpen (s): {avg_mpiopen:.2f}') 
        print(f'Stdev MPIOpen (s)): {stdev_mpiopen:.2f}')        
        
        
        
        print("Writing CSV file...")
        self.escreveCsv({ 'Nodes': self.number_nodes, 
                            'Avg_Simulation': avg_simulation , 'Stdev_simulation': stdev_simulation,
                            'Avg_comunication_time_per_process': avg_per_process, 'std_per_process': std_per_process,
                            'Avg_time_per_scenario': avg_time_per_scenario ,'Stdev_time_per_scenario': stdev_time_per_scenario,
                            'Avg_bandwidth': avg_agregate_bandwidth, 'Stddev_bandwidth': stddev_bandwidth,
                            'worstScenario': self.worstScenario, 'max_time_per_scenario': max_time_per_scenario,
                            'bestScenario': self.bestScenario, 'min_time_per_scenario': min_time_per_scenario ,
                            'Avg_time_per_record1': Avg_time_per_record1 ,'Stdev_time_per_record1': Stdev_time_per_record1,
                            'Avg_time_per_record2': Avg_time_per_record2 ,'Stdev_time_per_record2': Stdev_time_per_record2,
                            'Avg_time_per_record3': Avg_time_per_record3 ,'Stdev_time_per_record3': Stdev_time_per_record3,
                            'Avg_time_per_record4': Avg_time_per_record4 ,'Stdev_time_per_record4': Stdev_time_per_record4,
                            "Avg_mpiComunication": avg_mpiopen, 'Stdev_mpiComunication': stdev_mpiopen
                            } )  
        

    def escreveCsv(self,linha):
        path_csv = os.path.join(self.base_directory,"../plot.csv")
        cabecalho = ["Nodes", 
                     "Avg_Simulation", "Stdev_simulation",
                     "Avg_comunication_time_per_process","std_per_process",
                     "Avg_time_per_scenario",'Stdev_time_per_scenario',                     
                     "Avg_bandwidth", "Stddev_bandwidth",
                     "worstScenario", "max_time_per_scenario",
                     "bestScenario", "min_time_per_scenario",
                     "Avg_time_per_record1", "Stdev_time_per_record1",
                     "Avg_time_per_record2", "Stdev_time_per_record2",
                     "Avg_time_per_record3", "Stdev_time_per_record3",
                     "Avg_time_per_record4", "Stdev_time_per_record4",
                     "Avg_mpiComunication", 'Stdev_mpiComunication'
                     ]
        escrever_cabecalho = not os.path.exists(path_csv) or os.path.getsize(path_csv) == 0
        with open(path_csv,mode='a',newline='',encoding='utf-8') as arquivo_csv:
            writer = csv.DictWriter(arquivo_csv, fieldnames=cabecalho)
            if escrever_cabecalho:
                writer.writeheader()
            writer.writerow(linha)
            arquivo_csv.close()      
            
    def plotBandwidth(self,base_directory,plotLabel):
        


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
        ax1.bar(X, avgBandwidth, yerr=stdDevBandwidth, capsize=8, color='lightgreen', edgecolor='black', label='Banda (Gb/s)')
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


    def plotExecutionTime(self,base_directory,plotLabel):
        
        df_csv = pd.read_csv(os.path.join(base_directory,"../plot.csv"),index_col='Nodes')
        df_len = len(df_csv)
        X = np.zeros(df_len)
        largura = 0.25        
        categorias =  np.empty(df_len, dtype=object)
        AvgSimulation = np.zeros(df_len)
        Stdev_simulation = np.zeros(df_len)
        Avg_io_process = np.zeros(df_len)
        Stdev_time_per_process = np.zeros(df_len) 
        Avg_mpiComunication = np.zeros(df_len)
        Std_mpiComunication = np.zeros(df_len)


        timePerScenarioBase = 0      
        comunicationEstimate = 0         
        for i in range(len(df_csv)):
            X[i]= i
            categorias[i]= f"{df_csv.index[i]} Nodes"            
            AvgSimulation[i]= df_csv.iloc[i]['Avg_Simulation']                 
            Stdev_simulation[i]= df_csv.iloc[i]['Stdev_simulation']
            Avg_io_process[i] = df_csv.iloc[i]['Avg_comunication_time_per_process']
            Stdev_time_per_process[i]= df_csv.iloc[i]['std_per_process']
            # if i== 0:
            #         timePerScenarioBase = df_csv.iloc[i]['Avg_Simulation']
            # else:                
            #     comunicationEstimate = df_csv.iloc[i]['Avg_Simulation'] - timePerScenarioBase/( 2**i) - Avg_io_process[i]
            #     print(f"Comunication Estimate for {categorias[i]} Nodes: {comunicationEstimate:.2f} s")
            if 'Avg_mpiComunication' in df_csv.columns:                
                Avg_mpiComunication[i]= 2 * df_csv.iloc[i]['Avg_mpiComunication'] + comunicationEstimate
                Std_mpiComunication[i]= 2 * df_csv.iloc[i]['Stdev_mpiComunication']               
            else:
                Avg_mpiComunication[i] = comunicationEstimate
            AvgSimulation[i]= AvgSimulation[i]- Avg_io_process[i] - Avg_mpiComunication[i]
        
        # Plotando com barras de erro vindas da outra série
        plt.figure(figsize=(8,5))      
        #error_kw = dict(elinewidth=1.5, capthick=1.5)
        error_kw = dict(elinewidth=2.5, capthick=2.5, ecolor='black')
        plt.bar(X , AvgSimulation, yerr=Stdev_simulation, label="Computação", width=largura, capsize=8, error_kw=error_kw, color='lightgreen', edgecolor='black')
        plt.bar(X , Avg_mpiComunication, bottom=AvgSimulation, yerr=Std_mpiComunication, label="Comunicação", width=largura, capsize=8, error_kw=error_kw, color='blue', edgecolor='black')
        plt.bar(X , Avg_io_process, bottom=AvgSimulation+Avg_mpiComunication, yerr=Stdev_time_per_process, label="E/S", width=largura, capsize=8, error_kw=error_kw, color='red', edgecolor='black')


        plt.xticks(X, categorias)
        plt.ylabel('tempo de execução médio (s)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)

        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)
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














