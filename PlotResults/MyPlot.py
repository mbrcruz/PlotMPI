import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import statistics
from enum import Enum


class TypeEvaluation(Enum):
    JUST_COMUNICATION = 1,
    COMUNICATION_AND_IO = 2,
    JUST_SEND = 3,

    
class MyPlot(object):
    """description of class"""


    def __init__(self, base_directory,number_scenarios,typeEvaluation= TypeEvaluation.JUST_COMUNICATION,limit=-1):
        self.number_scenarios = number_scenarios
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
             
                if file in self.X1[scenario]:
                    if block in self.X1[i][file] and self.X1[scenario][file][block] < time:
                        continue
                    else: 
                        self.X1[scenario][file][block]= time                       
                        self.X2[scenario][file][block]= time2 
                else:
                    self.X1[scenario][file]={}
                    self.X1[scenario][file][block]=time                   
                    self.X2[scenario][file]={}
                    self.X2[scenario][file][block]= time2 
                    

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
        #self.start_moment = self.start_moment -  60
        for i in range(self.number_scenarios):         
         print( f"Ploting scenario {i+1} ...")
         tdiff = 0 
         scenario = i            
         last_x1=0
         for file in self.X1[i].keys():    
             for block in self.X1[i][file].keys():               
                x1 = ( self.X1[scenario][file][block] - self.start_moment )  
                try:                    
                    x2 = ( self.X2[scenario][file][block] - self.start_moment )  
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
                   diff= x4 - x1
                self.diffs.append(diff)                
                if x1 < self.limit:
                    plt.scatter(x1*scale,scenario+1, color=self.random_color(file),s=1)
                    #plt.plot([x1*scale,x2*scale],[scenario+1,scenario+1], color=self.random_color(file),marker="o",markersize=1)
                    #plt.plot([x1*scale,x2*scale],[scenario+1,scenario+1], color=self.random_color(file),linewidth=1)                          
        avg_time= statistics.mean(self.diffs)        
        stdev_time = statistics.stdev(self.diffs)
        max_time = max(self.diffs)
        sum_time= sum(self.diffs)
        count_time= len(self.diffs)
        print(f'Number blocks {count_time}')         
        print(f'Mean per block {avg_time:.2e}')        
        print(f'Stdev per block {stdev_time:.2e}')  
        print(f'Max per block {max_time:.2e}') 
        mean_per_process = sum_time/self.number_scenarios
        print(f'Mean per node {mean_per_process}')   
        #plt.xlim(0, max) 
        if self.limit != -1:
            plt.show()

   


