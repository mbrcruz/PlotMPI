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


    def __init__(self, base_directory,number_nodes, number_scenarios_per_nodes,typeEvaluation= TypeEvaluation.JUST_COMUNICATION,limit=-1):
        self.number_scenarios = number_nodes * number_scenarios_per_nodes
        self.number_scenarios_per_nodes = number_scenarios_per_nodes
        self.number_nodes = number_nodes 
        self.base_directory = base_directory
        self.limit=limit
        self.df_vec= []
        self.df_master= []
        self.colors={}
        self.X1=[]
        self.X2=[]
        self.X3=[]
        self.X4=[]
      

        self.diffs=[]
        self.sizes=[]
        self.bandwidths=[]
        self.start_moment=1000000000000000
        self.typeEvaluation= typeEvaluation

    def random_color(self,k):
        if k not in self.colors:
            self.colors[k]='#{:06x}'.format(random.randint(0, 0xFFFFFF))            
        return self.colors[k]

    def load_data(self):
        start = 2
        if not self.typeEvaluation == TypeEvaluation.JUST_COMUNICATION and not self.typeEvaluation == TypeEvaluation.COMUNICATION_AND_IO and not self.typeEvaluation == TypeEvaluation.JUST_SEND:       
            raise Exception("Bad configuration.")

        for i in range(self.number_scenarios):
            print(f'Loading scenario {i+1}')
            sufix= start +i
            self.X1.append(dict())
            self.X2.append(dict())   
            self.X3.append(dict()) 
            self.X4.append(dict()) 
            df= pd.read_csv(os.path.join(self.base_directory ,f"mpiio-{sufix}.log"),header=None)
            
            if ( self.start_moment > df.iloc[0,4] ):
                self.start_moment= df.iloc[0,4]
           
            for k in range(len(df)):
                scenario = i
                file = df.iloc[k,2]
                block = df.iloc[k,3] 
                time = df.iloc[k,4]
                time2 = df.iloc[k,5]
                size = df.iloc[k,7]
             
                if file in self.X1[scenario]:
                    if block in self.X1[i][file] and self.X1[scenario][file][block] < time:
                        continue
                    else: 
                        self.X1[scenario][file][block]=  ( time , size )                       
                        self.X2[scenario][file][block]= ( time2 , size )  
                else:
                    self.X1[scenario][file]={}
                    self.X1[scenario][file][block]= ( time , size )                   
                    self.X2[scenario][file]={}
                    self.X2[scenario][file][block]= ( time2 , size ) 
                    

        self.df_master=pd.read_csv(os.path.join(self.base_directory ,f"mpiio-master.log"),header=None)    
        for k in range(len(self.df_master)):
            if self.df_master.iloc[k].count() >= 8:
                time = self.df_master.iloc[k,7]
            else:
                raise Exception("Bad formatted file.")

            scenario = self.df_master.iloc[k,0]-1
            file = self.df_master.iloc[k,1]
            block= self.df_master.iloc[k,2]
            time = self.df_master.iloc[k,4] 
            time2= self.df_master.iloc[k,7]
            
            if scenario < self.number_scenarios:
                try:
                    if self.X1[scenario][file][block]:
                        if file in self.X3[scenario]:
                            if block in self.X3[scenario][file] and self.X3[scenario][file][block] > time:
                                continue
                            else: 
                                self.X3[scenario][file][block]=  time     
                                self.X4[scenario][file][block]=  time2
                        else:
                            self.X3[scenario][file]={}
                            self.X3[scenario][file][block]= time
                            self.X4[scenario][file]={}
                            self.X4[scenario][file][block]= time2
                except KeyError:
                    continue    

    def show_config(self):
        print( f" Number Scenario {self.number_scenarios}")
        print( f" Base Directory {self.base_directory}")
   
    def plot(self):     
        
        small_length=0.1
        scale = 10**6            
        plt.ylim(0, self.number_scenarios+1)
        #plt.xscale('log')
        #plt.yscale('linear')
        if self.limit==0:
            self.limit=3000
        plt.xlim(0, self.limit*scale)
        plt.xlabel('Time in microseconds(ms)')
        plt.ylabel('Core Id')

        plt.grid(True, linestyle='--', color='gray', alpha=0.5)   
        plt.axhline(y=0, color='k', linewidth=0.1)
        plt.axvline(x=0, color='k', linewidth=0.1)
       
        

        #max= 0
        self.start_moment = 0
        for i in range(self.number_scenarios):         
         print( f"Ploting scenario {i+1} ...")
         tdiff = 0 
         scenario = i            
         last_x1=0
         for file in self.X1[i].keys():    
             for block in self.X1[i][file].keys():               
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
                self.diffs.append(diff)  
                self.sizes.append(size)  
                if diff <= 0:
                    print(f"Diff is zero or negative: {diff} for {scenario} {file} {block}")
                    continue
                else:
                    self.bandwidths.append( ( size * 8 / 1000000000 ) / diff )

                if x1 < self.limit:
                    plt.scatter(x1*scale,scenario+1, color=self.random_color(file),s=1)
                    #plt.plot([x1*scale,x2*scale],[scenario+1,scenario+1], color=self.random_color(file),marker="o",markersize=1)
                    #plt.plot([x1*scale,x2*scale],[scenario+1,scenario+1], color=self.random_color(file),linewidth=1) 
          
        # as 2 contagens nao sao relevantes, porque os buffers tem tamanhos diferentes.            
        avg_time_per_buffer= statistics.mean(self.diffs)
        stdev_time_per_buffer = statistics.stdev(self.diffs)
        max_time_per_buffer = max(self.diffs)
        sum_time= sum(self.diffs)
        count_buffer= len(self.diffs)
        print(f'Number Buffers: {count_buffer}')         
        print(f'AVG per Buffers: {avg_time_per_buffer:.2e}')        
        print(f'Stdev per Buffers: {stdev_time_per_buffer:.2e}')  
        print(f'Max per Buffers: {max_time_per_buffer:.2e}') 

        avg_per_process = sum_time/self.number_scenarios
        print(f'Avg per Process(s): {avg_per_process}') 

        avg_size = statistics.mean(self.sizes) / 1000000
        total_size_per_nodes = sum(self.sizes) / 1000000000 / self.number_nodes
        print(f'AVG size buffer (MB): {avg_size:.2f}') 
        print(f'Total size per node (GB): {total_size_per_nodes:.2f}') 

        avg_bandwidth = statistics.mean(self.bandwidths)
        stddev_bandwidth = statistics.stdev(self.bandwidths)
        print(f'AVG Bandwidth (Gb/s): {avg_bandwidth:.2f}') 
        print(f'Stdev Bandwidth (Gb/s): {stddev_bandwidth:.2f}')

        self.escreveCsv({ 'Nodes': self.number_nodes, 'Avg_time_per_process': avg_per_process,  'Avg_bandwidth': avg_bandwidth, 'Stddev_bandwidth': stddev_bandwidth }   )
        
        #plt.xlim(0, max) 
        if self.limit != -1:
            plt.show()

    def plotMean(self,base_directory,plotLabel):
        
        df_csv = pd.read_csv(os.path.join(base_directory,"plot.csv"))
        df_len = len(df_csv)
        X = np.zeros(df_len)
        Means = np.zeros(df_len)
        StdDev = np.zeros(df_len)
        
        for i in range(len(df_csv)):
            X[i]= df_csv.iloc[i,0]
            Means[i]= df_csv.iloc[i,1]
            StdDev[i]= df_csv.iloc[i,2]
        # Plotando com barras de erro vindas da outra série
        plt.figure(figsize=(8,5))        
        plt.plot(X, Means, '-o',label='Média de envio dos blocos', color='blue')
        # Faixa do desvio padrão
        plt.fill_between(X, Means - StdDev, Means + StdDev, color='blue', alpha=0.2, label='Desvio Padrão')  
        #plt.ylim(-2,2)
        plt.title(plotLabel)        
        plt.xlabel('Número de nós')
        plt.ylabel('Média(s)')
        plt.legend()
        plt.grid(True)
        plt.show()


    def escreveCsv(self,linha):
        path_csv = os.path.join(self.base_directory,"../plot.csv")
        cabecalho = ['Nodes', 'Avg_time_per_process', "Avg_bandwidth", "Stddev_bandwidth"]
        escrever_cabecalho = not os.path.exists(path_csv) or os.path.getsize(path_csv) == 0
        with open(path_csv,mode='a',newline='',encoding='utf-8') as arquivo_csv:
            writer = csv.DictWriter(arquivo_csv, fieldnames=cabecalho)
            if escrever_cabecalho:
                writer.writeheader()
            writer.writerow(linha)
            arquivo_csv.close()





          



     


