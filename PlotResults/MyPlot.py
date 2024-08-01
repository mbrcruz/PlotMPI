import pandas as pd
import matplotlib.pyplot as plt
import os
import random
class MyPlot(object):
    """description of class"""


    def __init__(self, base_directory,number_scenarios,max_lines):
        self.number_scenarios = number_scenarios
        self.base_directory = base_directory
        self.max_lines=max_lines
        self.df_vec= []
        self.colors=[]

    def random_color(self,k):
        if (k >= len(self.colors) ):
            self.colors= '#{:06x}'.format(random.randint(0, 0xFFFFFF))
        return self.colors[k]

    def load_data(self):
        start = 2
        for i in range(self.number_scenarios):
            sufix= start +i
            self.df_vec.append(pd.read_csv(os.path.join(self.base_directory ,f"mpiio-{sufix}.log")))

    def show_config(self):
        print( f" Number Scenario {self.number_scenarios}")
        print( f" Base Directory {self.base_directory}")
   




    def plot(self):        
               
        plt.ylim(0, self.number_scenarios+1)
        #plt.xscale('log')
        #plt.yscale('linear')

        plt.grid(True, linestyle='--', color='gray', alpha=0.5)   
        plt.axhline(y=0, color='k', linewidth=1)
        plt.axvline(x=0, color='k', linewidth=1)
        small_length=0
        scale = 1
        

        max= 0
        for i in range(self.number_scenarios):
         start_moment = self.df_vec[i].iloc[0,2]
         print( f" Ploting process {i} ...")
         tdiff = 0 
         if self.max_lines > 0:
             max_lines=self.max_lines
         else:
            max_lines = len(self.df_vec[i])
         for k in range(max_lines):          
          x1 = ( self.df_vec[i].iloc[k,2] - start_moment ) * scale
          x2 = ( self.df_vec[i].iloc[k,3] - start_moment )* scale
          diff= x2 - x1
          tdiff= tdiff + diff
          plt.plot([x1,x2+small_length],[i+1,i+1], color=self.random_color(k),marker='o',markersize=1)
          if x2 > max:
              max = x2          
         print(f"Total diff {i} {tdiff} {max}...")   
        plt.xlim(0, max *scale) 
        plt.show()

   


