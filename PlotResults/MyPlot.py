import pandas as pd
import matplotlib.pyplot as plt
import os
import random
class MyPlot(object):
    """description of class"""


    def __init__(self, base_directory,number_scenarios):
        self.number_scenarios = number_scenarios
        self.base_directory = base_directory
        self.df_vec= []

    def random_color(self):
        return '#{:06x}'.format(random.randint(0, 0xFFFFFF))


    def load_data(self):
        start = 2
        for i in range(self.number_scenarios):
            sufix= start +i
            self.df_vec.append(pd.read_csv(os.path.join(self.base_directory ,f"mpiio-{sufix}.log")))

    def show_config(self):
        print( f" Number Scenario {self.number_scenarios}")
        print( f" Base Directory {self.base_directory}")
   




    def plot(self):
        x_max= 50**10
        plt.xlim(0, x_max)
        plt.ylim(0, self.number_scenarios+1)
        plt.grid(True, linestyle='--', color='gray', alpha=0.5)   
        plt.axhline(y=0, color='k', linewidth=1)
        plt.axvline(x=0, color='k', linewidth=1)
        

        max= 0
        for i in range(len(self.df_vec)):
         start_moment = self.df_vec[i].iloc[0,2]
         print( f" Ploting process {i} ...")
         tdiff = 0 
         for k in range(len(self.df_vec[i])):
          diff= self.df_vec[i].iloc[k,3] - self.df_vec[i].iloc[k,2]
          tdiff= tdiff + diff
          x1 = ( self.df_vec[i].iloc[k,2] - start_moment ) **10 
          x2 = ( self.df_vec[i].iloc[k,3] - start_moment ) **10
          plt.plot([x1,x2],[i+1,i+1], color=self.random_color(), linestyle='-')
          if x2 > max:
              max = x2          
         print(f"Total diff {i} {tdiff} {max}...")        
        plt.show()

   


