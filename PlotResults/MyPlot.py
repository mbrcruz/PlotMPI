import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import random
import statistics
from enum import Enum


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
        self.Simulations=[]
        self.localScenarios=[]      

        self.diffs=[]
        self.sizes=[]
        self.bandwidths=[]
        self.numberBuffers=0
        self.start_moment=1000000000000000
        self.typeEvaluation= typeEvaluation
        self.dicionarioScenarios={}

    def random_color(self,k):
        if k not in self.colors:
            self.colors[k]='#{:06x}'.format(random.randint(0, 0xFFFFFF))            
        return self.colors[k]

    def load_data(self):
        
        if ( not self.typeEvaluation == TypeEvaluation.JUST_COMUNICATION 
        and not self.typeEvaluation == TypeEvaluation.COMUNICATION_AND_IO        
        and not self.typeEvaluation == TypeEvaluation.JUST_SEND):          
            raise Exception("Bad configuration.")
        

        self.X1= {i: {} for i in range(1, self.number_scenarios+1 )}
        self.X2= {i: {} for i in range(1, self.number_scenarios+1 )}
        self.X3= {i: {} for i in range(1, self.number_scenarios+1 )}
        self.X4= {i: {} for i in range(1, self.number_scenarios+1 )}

        start = 2
        for i in range(self.number_nodes * self.number_scenarios_per_nodes):
            sufix= start +i        
            print(f'Loading Processes {i+1}')
            df= pd.read_csv(os.path.join(self.base_directory ,f"mpiio-{sufix}.log"),header=None)
                
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


            
            self.df_times= pd.read_csv(os.path.join(self.base_directory ,f"sddptimer{sufix:04d}.log"),header=None)
            for k in range(len(self.df_times)):
                if  self.df_times.iloc[k,0] == "Simulation":
                    self.Simulations.append(float(self.df_times.iloc[k,1]))  

        self.df_master=pd.read_csv(os.path.join(self.base_directory ,f"mpiio-master.log"),header=None)    
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
                print(f"KeyError {scenario} {file} {seq}")
                continue   
            except TypeError:
                print(f"TypeError {scenario} {file} {seq}")
                continue
       

    def show_config(self):
        print( f" Number Scenario {self.number_scenarios}")
        print( f" Base Directory {self.base_directory}")
   
    def computerMetrics(self,escrever_csv=True):     
        
        small_length=0.1
        scale = 10**6            
        #plt.ylim(0, self.number_scenarios+1)
        #plt.xscale('log')
        #plt.yscale('linear')
        if self.limit==0:
            self.limit=3000
        # plt.xlim(0, self.limit*scale)
        # plt.xlabel('Time in microseconds(ms)')
        # plt.ylabel('Core Id')

        # plt.grid(True, linestyle='--', color='gray', alpha=0.5)   
        # plt.axhline(y=0, color='k', linewidth=0.1)
        # plt.axvline(x=0, color='k', linewidth=0.1)
       
        

        #max= 0
        ind= 0
        self.start_moment = 0
        for i in range(self.number_scenarios):    
            scenario = i+1        
            print( f"Ploting scenario {scenario} ...")
            tdiff = 0                  
            last_x1=0
            if scenario in self.localScenarios:              
                continue
            else:
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
                            print(f"Diff is zero or negative: {diff} for {scenario} {file} {block}")        
                            continue              
                        else:
                            self.diffs.append(diff)  
                            self.sizes.append(size / 1000000)  # Convert to MB
                            self.bandwidths.append( ( size * 8 / 1000000000 ) / diff )
                        #if x1 < self.limit:
                        #    plt.scatter(x1*scale,scenario+1, color=self.random_color(file),s=1)
                        #plt.plot([x1*scale,x2*scale],[scenario+1,scenario+1], color=self.random_color(file),marker="o",markersize=1)
                        #plt.plot([x1*scale,x2*scale],[scenario+1,scenario+1], color=self.random_color(file),linewidth=1) 
            
            # as 2 contagens nao sao relevantes, porque os buffers tem tamanhos diferentes.            
        avg_time_per_buffer= statistics.mean(self.diffs)
        stdev_time_per_buffer = statistics.stdev(self.diffs)
        max_time_per_buffer = max(self.diffs)
        sum_time= sum(self.diffs)
        count_buffer= len(self.diffs)
        avg_simulation = statistics.mean(self.Simulations)
        stdev_simulation  = statistics.stdev(self.Simulations)
        print(f'Number Buffers: {count_buffer}')                
        print(f'AVG per Buffers: {avg_time_per_buffer}')        
        print(f'Stdev per Buffers: {stdev_time_per_buffer}')  
        print(f'Max per Buffers: {max_time_per_buffer}') 

        avg_per_process = sum_time/self.number_scenarios
        print(f'Avg Comunication per Process(s): {avg_per_process}') 
        print(f'Avg Simulation per Process(s): {avg_simulation}') 
        print(f'StdDev Simulation per Process(s): {stdev_simulation}') 

        avg_size = statistics.mean(self.sizes)
        total_size_per_nodes = sum(self.sizes) / 1000 / self.number_nodes
        print(f'AVG size buffer (MB): {avg_size:.2f}') 
        print(f'Total size per node (GB): {total_size_per_nodes:.2f}') 

        avg_bandwidth = statistics.mean(self.bandwidths)
        stddev_bandwidth = statistics.stdev(self.bandwidths)
        print(f'AVG Bandwidth (Gb/s): {avg_bandwidth:.2f}') 
        print(f'Stdev Bandwidth (Gb/s): {stddev_bandwidth:.2f}')

        if escrever_csv:
            print("Writing CSV file...")
            self.escreveCsv({ 'Nodes': self.number_nodes, 
                             'Avg_Simulation': avg_simulation , 'Stdev_simulation': stdev_simulation,
                             'Avg_comunication_time_per_process': avg_per_process, 
                             'Avg_time_per_buffer': avg_time_per_buffer ,'Stdev_time_per_buffer': stdev_time_per_buffer,
                             'Avg_bandwidth': avg_bandwidth, 'Stddev_bandwidth': stddev_bandwidth }   )  
            
   
    def escreveCsv(self,linha):
        path_csv = os.path.join(self.base_directory,"../plot.csv")
        cabecalho = ['Nodes', 
                     'Avg_Simulation', 'Stdev_simulation',
                     'Avg_comunication_time_per_process',
                     "Avg_time_per_buffer",'Stdev_time_per_buffer', 
                     "Avg_bandwidth", "Stddev_bandwidth"]
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
        plt.title(f'Banda média por envio de buffer no {plotLabel} com erro padrão')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()     
        plt.legend()  
        plt.show()

    def plotLatency(self,base_directory,plotLabel):
        
        df_csv = pd.read_csv(os.path.join(base_directory,"../plot.csv"),index_col='Nodes')
        df_len = len(df_csv)
        X = np.zeros(df_len)
        categorias =  np.empty(df_len, dtype=object)
        avg_time_per_buffer = np.zeros(df_len)
        stdev_time_per_buffer = np.zeros(df_len)
        
        for i in range(len(df_csv)):
            X[i]= i
            categorias[i]= f"{df_csv.index[i]} Nodes"
            avg_time_per_buffer[i]= df_csv.iloc[i]['Avg_time_per_buffer']
            stdev_time_per_buffer[i]= df_csv.iloc[i]['Stdev_time_per_buffer']
        # Plotando com barras de erro vindas da outra série
        plt.figure(figsize=(8,5))      
        plt.bar(X, avg_time_per_buffer, yerr=stdev_time_per_buffer, label="Latencia (s)", capsize=8, color='lightgreen', edgecolor='black') 

        plt.xticks(X, categorias)
        plt.ylabel('Latencia média')
        plt.title(f'Latencia Média de envio do buffer no {plotLabel} com erro padrão')
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
            AvgSimulation[i]= df_csv.iloc[i]['Avg_Simulation']
            Stdev_simulation[i]= df_csv.iloc[i]['Stdev_simulation']
            Avg_time_per_process[i]= df_csv.iloc[i]['Avg_comunication_time_per_process']
        
        # Plotando com barras de erro vindas da outra série
        plt.figure(figsize=(8,5))      
        plt.bar(X , AvgSimulation, yerr=Stdev_simulation, label="Computação", width=largura, color='lightgreen', edgecolor='black') 
        plt.bar(X , Avg_time_per_process, bottom=AvgSimulation, label="Comunicação", width=largura, color='blue', edgecolor='black') 

        plt.xticks(X, categorias)
        plt.ylabel('tempo de simulação médio (s)')
        plt.title(f'tempo de simulação médio (s) no {plotLabel} com desvio padrão.')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()   
        plt.legend()    
        plt.show()
    
    def PlotHistogram(self):

        plt.figure(figsize=(8,5))
        plt.hist(self.sizes, bins=50, color='blue', alpha=0.7, edgecolor='black')
        plt.title('Histograma Tamanho arquivos')
        plt.xlabel('Tamanho (MB)')
        plt.ylabel('Frequência')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    





          



     


