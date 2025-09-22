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


    def __init__(self, base_directory,number_nodes, number_scenarios_per_node, number_scenarios, typeEvaluation= TypeEvaluation.JUST_COMUNICATION, onlyRemote=True, limit=-1):
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
        self.localScenarios=[]   
        self.bestScenario=0
        self.worstScenario=0 
        self.categories=[1,10,30]
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

    def load_data(self, filter_scenario=0,experiments=10,min_size=0,max_size=0):

        if ( not self.typeEvaluation == TypeEvaluation.JUST_COMUNICATION 
            and not self.typeEvaluation == TypeEvaluation.COMUNICATION_AND_IO        
            and not self.typeEvaluation == TypeEvaluation.JUST_SEND):          
                raise Exception("Bad configuration.")
    
        for experiment in range(experiments):        
            print(f'Loading Experiment {experiment+1}')
            '#initialize data structures'
            self.X1= {i: {} for i in range(1, self.number_scenarios+1 )}
            self.X2= {i: {} for i in range(1, self.number_scenarios+1 )}
            self.X3= {i: {} for i in range(1, self.number_scenarios+1 )}
            self.X4= {i: {} for i in range(1, self.number_scenarios+1 )}

            initial_rank = 2

            for i in range(self.number_nodes * self.number_scenarios_per_nodes):
                sufix= initial_rank +i        
                print(f'Loading Processes {i+1}')
                df= pd.read_csv(os.path.join(self.base_directory, str(experiment+1),  f"mpiio-{sufix}.log"),header=None)
                    
                if ( self.start_moment > df.iloc[0,5] ):
                    self.start_moment= df.iloc[0,5]
            
                for k in range(len(df)):
                    self.numberBuffers += 1
                    scenario = df.iloc[k,2]
                    file = df.iloc[k,3]
                    seq = df.iloc[k,4] 
                    time = df.iloc[k,5]
                    time2 = df.iloc[k,6]
                    size = df.iloc[k,8]
                    
                    if self.onlyRemote and i < self.number_scenarios_per_nodes:
                        if scenario not in self.localScenarios:
                            self.localScenarios.append(scenario)                       
                    if file in self.X1[scenario]:
                        if seq in self.X1[scenario][file] and self.X1[scenario][file][seq] < time:
                            continue
                        else: 
                            self.X1[scenario][file][seq]=  ( time , size )                       
                            self.X2[scenario][file][seq]= ( time2 , size )  
                            self.dicionarioScenarios[(i+1,seq)]= (scenario,file)
                    else:
                        self.X1[scenario][file]={}
                        self.X1[scenario][file][seq]= ( time , size )                   
                        self.X2[scenario][file]={}
                        self.X2[scenario][file][seq]= ( time2 , size ) 
                        self.dicionarioScenarios[(i+1,seq)]= (scenario,file)


                
                self.df_times= pd.read_csv(os.path.join(self.base_directory , str(experiment+1), f"sddptimer{sufix:04d}.log"),header=None)
                for k in range(len(self.df_times)):
                    if  self.df_times.iloc[k,0] == "Simulation":
                        self.Simulations.append(float(self.df_times.iloc[k,1]))
            
            self.df_master=pd.read_csv(os.path.join(self.base_directory , str(experiment+1), "mpiio-master.log"),header=None)    

            for k in range(len(self.df_master)):           

                cpu = self.df_master.iloc[k,0]
                seq = self.df_master.iloc[k,1]           
                time = self.df_master.iloc[k,2] 
                time2= self.df_master.iloc[k,6]
            
                try:
                    scenario = self.dicionarioScenarios.get((cpu,seq))[0]
                    file = self.dicionarioScenarios.get((cpu,seq))[1] 
                    if self.X1[scenario][file][seq]:
                        if file in self.X3[scenario]:
                            if seq in self.X3[scenario][file] and self.X3[scenario][file][seq] > time:
                                continue
                            else: 
                                self.X3[scenario][file][seq]=  time     
                                self.X4[scenario][file][seq]=  time2
                        else:
                            self.X3[scenario][file]={}
                            self.X3[scenario][file][seq]= time
                            self.X4[scenario][file]={}
                            self.X4[scenario][file][seq]= time2
                except KeyError:
                    #print(f"KeyError {scenario} {file} {seq}")
                    continue   
                except TypeError:
                    #print(f"TypeError {scenario} {file} {seq}")
                    continue
            
            for i in range(self.number_scenarios):    
                scenario = i+1        
                
                tdiff = 0                  
                last_x1=0
                if scenario in self.localScenarios:     
                    print( f"Ignoring local scenario {scenario} ...")         
                    continue
                else:
                    print( f"Computing scenario {scenario} ...")                  
                    for file in self.X1[scenario]:
                        for block in self.X1[scenario][file]:               
                            x1 = ( self.X1[scenario][file][block][0] - self.start_moment )  
                            size = self.X1[scenario][file][block][1]
                            try:                    
                                x2 = ( self.X2[scenario][file][block][0] - self.start_moment )  
                                x3 = ( self.X3[scenario][file][block] - self.start_moment )
                                x4 = ( self.X4[scenario][file][block] - self.start_moment )                            
                            except KeyError:
                                print(f"KeyError {scenario} {file} {block}")
                                exit(-1)   
                            diff = 0 

                            if self.typeEvaluation == TypeEvaluation.JUST_SEND:
                                diff= x2 - x1
                            elif self.typeEvaluation == TypeEvaluation.JUST_COMUNICATION:
                                diff= (x3 - x1)
                            else:
                                diff= (x4 - x3 ) + ( x2 - x1)
                                            
                            if diff <= 0:                           
                                if scenario not in self.badScenarios: 
                                    print(f"Diff is zero or negative: {diff} for {scenario} {file} {block}")  
                                    self.badScenarios.append(scenario)     
                                continue              
                            else:                  
                                # The following elifs were empty and are removed for clarity
                                # If you want to add logic for sizeMB ranges, add code here
                                if filter_scenario == 0 or scenario == filter_scenario:     
                                    record= { "experiment": experiment, "scenario": scenario, 'file': file, 'block': block, 'sizeBytes': size  , 'timeSec': diff }   
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

        
        Size_time_per_record1 = df1["timeSec"].count()/10
        Avg_time_per_record1 = df1.groupby("experiment")["timeSec"].mean().mean()
        Stdev_time_per_record1= df1.groupby("experiment")["timeSec"].mean().std()
        print(f'Count per record1 < {self.categories[0]} MB: {Size_time_per_record1}')
        print(f'AVG per record1 < {self.categories[0]} MB: {Avg_time_per_record1}') 
        print(f'Stdev per record1 < {self.categories[0]} MB: {Stdev_time_per_record1}')

        Size_time_per_record2 = df2["timeSec"].count()/10
        Avg_time_per_record2 = df2.groupby("experiment")["timeSec"].mean().mean()
        Stdev_time_per_record2= df2.groupby("experiment")["timeSec"].mean().std()
        print(f'Count per record2 < {self.categories[1]} MB: {Size_time_per_record2}')  
        print(f'AVG per record2 < {self.categories[1]} MB: {Avg_time_per_record2}') 
        print(f'Stdev per record2 < {self.categories[1]} MB: {Stdev_time_per_record2}')

        Size_time_per_record3 = df3["timeSec"].count()/10
        Avg_time_per_record3 = df3.groupby("experiment")["timeSec"].mean().mean()
        Stdev_time_per_record3= df3.groupby("experiment")["timeSec"].mean().std()
        print(f'Count per record3 < {self.categories[2]} MB: {Size_time_per_record3}')
        print(f'AVG per record3 < {self.categories[2]} MB: {Avg_time_per_record3}') 
        print(f'Stdev per record3 < {self.categories[2]} MB: {  Stdev_time_per_record3}')

        Size_time_per_record4 = df4["timeSec"].count()/10
        Avg_time_per_record4 = df4.groupby("experiment")["timeSec"].mean().mean()
        Stdev_time_per_record4= df4.groupby("experiment")["timeSec"].mean().std()
        print(f'Count per record4 >= {self.categories[2]} MB: {Size_time_per_record4}')
        print(f'AVG per record4 >= {self.categories[2]} MB: {Avg_time_per_record4}') 
        print(f'Stdev per record4 >= {self.categories[2]} MB: {Stdev_time_per_record4}')

        
        #sum_time= sum(self.diffs)
        #Calcula o tempo medio de cada cenario e depois multiplica pelo numero de cenario executado por processo.
        avg_per_process = ( avg_time_per_scenario * self.number_scenarios/ ( ( self.number_nodes - 1) * self.number_scenarios_per_nodes) )
        # # Average time per process)
        print(f'Avg Comunication per Process(s): {avg_per_process}') 
        print(f'Avg Simulation per Process(s): {avg_simulation}') 
        print(f'StdDev Simulation per Process(s): {stdev_simulation}') 

        

        agrupados = df.groupby("scenario")[["sizeBytes","timeSec"]].sum()
        bandwidth_per_scenario = ( agrupados["sizeBytes"] * 8 / 1000000000 ) / agrupados["timeSec"]  # em Gb/s
        avg_bandwidth = bandwidth_per_scenario.mean()
        stddev_bandwidth = bandwidth_per_scenario.std()
        # total_size_per_nodes = statistics.mean(self.sizesPerScenario.values()) * self.number_scenarios_per_nodes/ 1000000000
        # print(f'AVG size Scenario (MB): {avg_size:.2f}') 
        # print(f'Total size per node (GB): {total_size_per_nodes:.2f}') 
        print(f'AVG Bandwidth per scenario (Gb/s): {avg_bandwidth:.2f}') 
        print(f'Stdev Bandwidth per scenario (Gb/s): {stddev_bandwidth:.2f}')
        print("Writing CSV file...")
        self.escreveCsv({ 'Nodes': self.number_nodes, 
                            'Avg_Simulation': avg_simulation , 'Stdev_simulation': stdev_simulation,
                            'Avg_comunication_time_per_process': avg_per_process, 
                            'Avg_time_per_scenario': avg_time_per_scenario ,'Stdev_time_per_scenario': stdev_time_per_scenario,
                            'Avg_bandwidth': avg_bandwidth, 'Stddev_bandwidth': stddev_bandwidth,
                            'worstScenario': self.worstScenario, 'max_time_per_scenario': max_time_per_scenario,
                            'bestScenario': self.bestScenario, 'min_time_per_scenario': min_time_per_scenario ,
                            'Avg_time_per_record1': Avg_time_per_record1 ,'Stdev_time_per_record1': Stdev_time_per_record1,
                            'Avg_time_per_record2': Avg_time_per_record2 ,'Stdev_time_per_record2': Stdev_time_per_record2,
                            'Avg_time_per_record3': Avg_time_per_record3 ,'Stdev_time_per_record3': Stdev_time_per_record3,
                            'Avg_time_per_record4': Avg_time_per_record4 ,'Stdev_time_per_record4': Stdev_time_per_record4 } )  
        

    def escreveCsv(self,linha):
        path_csv = os.path.join(self.base_directory,"../plot.csv")
        cabecalho = ['Nodes', 
                     'Avg_Simulation', 'Stdev_simulation',
                     'Avg_comunication_time_per_process',
                     "Avg_time_per_scenario",'Stdev_time_per_scenario',                     
                     "Avg_bandwidth", "Stddev_bandwidth",
                     "worstScenario", "max_time_per_scenario",
                     "bestScenario", "min_time_per_scenario",
                     "Avg_time_per_record1", "Stdev_time_per_record1",
                     "Avg_time_per_record2", "Stdev_time_per_record2",
                     "Avg_time_per_record3", "Stdev_time_per_record3",
                     "Avg_time_per_record4", "Stdev_time_per_record4"
                     ]
        escrever_cabecalho = not os.path.exists(path_csv) or os.path.getsize(path_csv) == 0
        with open(path_csv,mode='a',newline='',encoding='utf-8') as arquivo_csv:
            writer = csv.DictWriter(arquivo_csv, fieldnames=cabecalho)
            if escrever_cabecalho:
                writer.writeheader()
            writer.writerow(linha)
            arquivo_csv.close()      
            
    def plotBandwidth(self,base_directory,plotLabel):
        
        df_csv = pd.read_csv(os.path.join(base_directory,"../plot.csv"),index_col='Nodes')
        df_len = len(df_csv)
        X = np.zeros(df_len)
        categorias =  np.empty(df_len, dtype=object)
        avgBandwidth = np.zeros(df_len)
        stdDevBandwidth = np.zeros(df_len)
        
        for i in range(len(df_csv)):
            X[i]= i
            categorias[i]= f"{df_csv.index[i]} Nodes"
            avgBandwidth[i]= df_csv.iloc[i]['Avg_bandwidth']
            stdDevBandwidth[i]= df_csv.iloc[i]['Stddev_bandwidth']
        # Plotando com barras de erro vindas da outra série
        plt.figure(figsize=(8,5))      
        plt.bar(X, avgBandwidth, yerr=stdDevBandwidth, label="Banda Gb/s", capsize=8, color='lightgreen', edgecolor='black') 

        plt.xticks(X, categorias)
        plt.ylabel('Banda Média')
        plt.title(f'Banda média por envio de cenário no {plotLabel} com erro padrão')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()     
        plt.legend()  
        plt.show()

    def plotLatency(self,base_directory,plotLabel):
        
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



    def plotExecutionTime(self,base_directory,plotLabel):
        
        df_csv = pd.read_csv(os.path.join(base_directory,"../plot.csv"),index_col='Nodes')
        df_len = len(df_csv)
        X = np.zeros(df_len)
        largura = 0.25        
        categorias =  np.empty(df_len, dtype=object)
        AvgSimulation = np.zeros(df_len)
        Stdev_simulation = np.zeros(df_len)
        Avg_time_per_process = np.zeros(df_len)
        
        for i in range(len(df_csv)):
            X[i]= i
            categorias[i]= f"{df_csv.index[i]} Nodes"
            AvgSimulation_total= df_csv.iloc[i]['Avg_Simulation']            
            Stdev_simulation[i]= df_csv.iloc[i]['Stdev_simulation']
            Avg_time_per_process[i]= df_csv.iloc[i]['Avg_comunication_time_per_process']
            AvgSimulation[i]= AvgSimulation_total - Avg_time_per_process[i]
        
        # Plotando com barras de erro vindas da outra série
        plt.figure(figsize=(8,5))      
        plt.bar(X , AvgSimulation, yerr=Stdev_simulation, label="Computação", width=largura, color='lightgreen', edgecolor='black') 
        plt.bar(X , Avg_time_per_process, bottom=AvgSimulation, label="Comunicação", width=largura, color='blue', edgecolor='black') 
        #plt.bar(X , Avg_time_per_process, label="Comunicação", width=largura, color='blue', edgecolor='black') 

        plt.xticks(X, categorias)
        plt.ylabel('tempo de simulação médio (s)')
        plt.title(f'tempo de simulação médio (s) no {plotLabel} com desvio padrão.')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()   
        plt.legend()    
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
        plt.title('Histograma do tamanho das mensagens enviadas')
        plt.xlabel('Tamanho (KBytes) em escala logarítmica')
        plt.ylabel('Frequência')
        plt.xscale('log')     
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
        plt.scatter( jitter_x,sizes1, color="green", label="Sem concorrencia", s=10, alpha=0.7, edgecolors='k')
        jitter_y = np.array(sizes2) + (np.random.rand(len(sizes2)) - 0.5)  # Adiciona um pequeno jitter no eixo x
        plt.scatter(diffs2, jitter_y,  color="red",  label="Com concorrencia", s=10, alpha=0.7, edgecolors='k')

        plt.title("Tempo envio x tamanho do Buffer no "+ displotLabel )
        plt.xlabel("Tempo de envio (s)")
        plt.ylabel("Tamanho do Buffer (Bytes)")

        # Escala log no eixo X ajuda a visualizar melhor (opcional)
        plt.xscale('log')
        #plt.ticklabel_format(axis='y', style='sci')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.show()














