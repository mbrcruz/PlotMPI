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
        self.ioExtra=[]   # tempos de wait/test: {"experiment", "rank", "timeSec"}
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

    def _confidence_interval_95(self, values):
        values = pd.Series(values, dtype="float64").dropna()
        n = len(values)
        if n < 2:
            return 0.0
        df = n - 1
        try:
            from scipy.stats import t
            t_critical = t.ppf(0.975, df)
        except ImportError:
            t_critical_by_df = {
                1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
                16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
                21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
                26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
                40: 2.021, 60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980,
            }
            if df in t_critical_by_df:
                t_critical = t_critical_by_df[df]
            elif df > max(t_critical_by_df):
                t_critical = 1.96
            else:
                next_df = min(key for key in t_critical_by_df if key > df)
                t_critical = t_critical_by_df[next_df]
        return t_critical * values.std(ddof=1) / np.sqrt(n)

    def _sample_stdev(self, values):
        values = pd.Series(values, dtype="float64").dropna()
        if len(values) < 2:
            return 0.0
        return values.std(ddof=1)

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
                    _exp_dir = os.path.join(self.base_directory, str(experiment+1))
                    simulation["comunication"] = simulation["simulation"] - simulation["hourly_simulation"]
                    self.Simulations.append(simulation)

                    # Arquivos auxiliares de E/S: formato rank,scenario,tempo_acumulado
                    # Não têm timestamps nem tamanho — armazenados separadamente em ioExtra
                    if not (self.onlyRemote and i < self.number_scenarios_per_nodes):
                        for _suffix in ("wait", "test"):
                            _aux_path = os.path.join(_exp_dir, f"mpiio-{rank}-{_suffix}.log")
                            if not os.path.exists(_aux_path):
                                continue
                            df_aux = pd.read_csv(_aux_path, header=None)
                            for k in range(len(df_aux)):
                                scenario = df_aux.iloc[k, 1]
                                t_acc    = float(df_aux.iloc[k, 2])
                                if filter_scenario == 0 or scenario == filter_scenario:
                                    self.ioExtra.append({
                                        "experiment": experiment + 1,
                                        "rank": rank,
                                        "timeSec": t_acc,
                                    })

                    # load mpi collective times — tenta os dois padrões de nome
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
        mean_time_per_scenario_by_exp = sum_scenarios.groupby("experiment").mean()
        avg_time_per_scenario=  mean_time_per_scenario_by_exp.mean()       
        stdev_time_per_scenario = self._sample_stdev(mean_time_per_scenario_by_exp)
        ci95_time_per_scenario = self._confidence_interval_95(mean_time_per_scenario_by_exp)

        self.worstScenario= sum_scenarios.idxmax()
        self.bestScenario= sum_scenarios.idxmin()
        max_time_per_scenario = sum_scenarios.max()
        min_time_per_scenario = sum_scenarios.min() 
        
    
        #avg_simulation = statistics.mean(self.Simulations)
        mean_simulation_by_exp = df_simulation.groupby("experiment")["simulation"].mean()
        avg_simulation = mean_simulation_by_exp.mean()         
        stdev_simulation = self._sample_stdev(mean_simulation_by_exp)
        ci95_simulation = self._confidence_interval_95(mean_simulation_by_exp)
        

        mean_comunication_by_exp = df_simulation.groupby("experiment")["comunication"].mean()
        Avg_comunication_per_process = mean_comunication_by_exp.mean()   
        stdev_comunication_per_process = self._sample_stdev(mean_comunication_by_exp)
        ci95_comunication_per_process = self._confidence_interval_95(mean_comunication_by_exp)

        print(f'Avg Comunication per Process(s): {Avg_comunication_per_process}')
        print(f'Stdev Comunication per Process(s): {stdev_comunication_per_process}')
        print(f'CI95 Comunication per Process(s): {ci95_comunication_per_process}')

        #Falta desvio padrao de comunicacao por processo
        print(f'Avg Simulation per Process(s): {avg_simulation}') 
        print(f'Stdev Simulation per Process(s): {stdev_simulation}')
        print(f'CI95 Simulation per Process(s): {ci95_simulation}') 
        #print(f'Number Buffers: {self.records.count}')                
        print(f'AVG per Scenarios: {avg_time_per_scenario}')        
        print(f'Stdev per Scenarios: {stdev_time_per_scenario}')
        print(f'CI95 per Scenarios: {ci95_time_per_scenario}')  
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
        CI95_time_per_record1=0
        CI95_time_per_record2=0
        CI95_time_per_record3=0
        CI95_time_per_record4=0
        
        if len(self.records1) >0:
            Size_time_per_record1 = df1["timeSec"].count()/ 5
            mean_time_per_record1_by_exp = df1.groupby("experiment")["timeSec"].mean()
            Avg_time_per_record1 = mean_time_per_record1_by_exp.mean()
            Stdev_time_per_record1= self._sample_stdev(mean_time_per_record1_by_exp)
            CI95_time_per_record1= self._confidence_interval_95(mean_time_per_record1_by_exp)
            print(f'Count per record1 < {self.categories[0]} MB: {Size_time_per_record1}')
            print(f'AVG per record1 < {self.categories[0]} MB: {Avg_time_per_record1}') 
            print(f'Stdev per record1 < {self.categories[0]} MB: {Stdev_time_per_record1}')
            print(f'CI95 per record1 < {self.categories[0]} MB: {CI95_time_per_record1}')

        if len(self.records2) >0:
            Size_time_per_record2 = df2["timeSec"].count()/ 5 
            mean_time_per_record2_by_exp = df2.groupby("experiment")["timeSec"].mean()
            Avg_time_per_record2 = mean_time_per_record2_by_exp.mean()
            Stdev_time_per_record2= self._sample_stdev(mean_time_per_record2_by_exp)
            CI95_time_per_record2= self._confidence_interval_95(mean_time_per_record2_by_exp)
            print(f'Count per record2 < {self.categories[1]} MB: {Size_time_per_record2}')  
            print(f'AVG per record2 < {self.categories[1]} MB: {Avg_time_per_record2}') 
            print(f'Stdev per record2 < {self.categories[1]} MB: {Stdev_time_per_record2}')
            print(f'CI95 per record2 < {self.categories[1]} MB: {CI95_time_per_record2}')

        if len(self.records3) > 0:
            Size_time_per_record3 = df3["timeSec"].count()/ 5
            mean_time_per_record3_by_exp = df3.groupby("experiment")["timeSec"].mean()
            Avg_time_per_record3 = mean_time_per_record3_by_exp.mean()
            Stdev_time_per_record3= self._sample_stdev(mean_time_per_record3_by_exp)
            CI95_time_per_record3= self._confidence_interval_95(mean_time_per_record3_by_exp)
            print(f'Count per record3 < {self.categories[2]} MB: {Size_time_per_record3}')
            print(f'AVG per record3 < {self.categories[2]} MB: {Avg_time_per_record3}') 
            print(f'Stdev per record3 < {self.categories[2]} MB: {Stdev_time_per_record3}')
            print(f'CI95 per record3 < {self.categories[2]} MB: {CI95_time_per_record3}')
        if len(self.records4) > 0:
            Size_time_per_record4 = df4["timeSec"].count()/ 5
            mean_time_per_record4_by_exp = df4.groupby("experiment")["timeSec"].mean()
            Avg_time_per_record4 = mean_time_per_record4_by_exp.mean()
            Stdev_time_per_record4= self._sample_stdev(mean_time_per_record4_by_exp)
            CI95_time_per_record4= self._confidence_interval_95(mean_time_per_record4_by_exp)
            print(f'Count per record4 >= {self.categories[2]} MB: {Size_time_per_record4}')
            print(f'AVG per record4 >= {self.categories[2]} MB: {Avg_time_per_record4}') 
            print(f'Stdev per record4 >= {self.categories[2]} MB: {Stdev_time_per_record4}')
            print(f'CI95 per record4 >= {self.categories[2]} MB: {CI95_time_per_record4}')

        
        #sum_time= sum(self.diffs)
        #Calcula o tempo medio de cada cenario e depois multiplica pelo numero de cenario executado por processo.
        sum_io_by_process = df.groupby(["experiment","rank"])["timeSec"].sum()
        # Adiciona tempos de wait/test (ioExtra) ao I/O por processo
        if self.ioExtra:
            df_extra = pd.DataFrame(self.ioExtra)
            sum_extra = df_extra.groupby(["experiment","rank"])["timeSec"].sum()
            mean_ioextra_by_exp = sum_extra.groupby("experiment").mean()
            avg_ioextra_all_experiments = mean_ioextra_by_exp.mean()
            print(f'Avg IOExtra across Experiments(s): {avg_ioextra_all_experiments}')
            sum_io_by_process = sum_io_by_process.add(sum_extra, fill_value=0)
        Avg_io_per_process = sum_io_by_process.mean()
        std_io_per_process = self._sample_stdev(sum_io_by_process)
        ci95_io_per_process = self._confidence_interval_95(sum_io_by_process)
        print(f'Avg IO per Process(s): {Avg_io_per_process}')
        print(f'Stdev IO per Process(s): {std_io_per_process}')
        print(f'CI95 IO per Process(s): {ci95_io_per_process}')
        
        
     

        
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
                "std_bw": self._sample_stdev(bw_per_window),
                "ci95_bw": self._confidence_interval_95(bw_per_window)
            })
        df_bw = pd.DataFrame(results_bw).set_index("experiment")
        print(f"Banda agregada por experimento (janelas de {WINDOW_SEC}s):")
        for exp, row in df_bw.iterrows():
            print(f'  Experimento {exp}: {row["mean_bw"]:.2f} stdev {row["std_bw"]:.2f} CI95 {row["ci95_bw"]:.2f} Gb/s')
        avg_agregate_bandwidth = df_bw["mean_bw"].mean()
        stddev_bandwidth = self._sample_stdev(df_bw["mean_bw"])
        ci95_bandwidth = self._confidence_interval_95(df_bw["mean_bw"])
        agrupados = df.groupby(["experiment","rank"])[["sizeBytes","timeSec"]].sum()
        size_por_rank = agrupados["sizeBytes"].mean() / 1e9  # em GB
        total_size_per_nodes = size_por_rank * self.number_scenarios_per_nodes  # em GB
        print(f'Total Size per Node (GB): {total_size_per_nodes:.2f}')
        print(f'AVG Aggregate Bandwidth (Gb/s): {avg_agregate_bandwidth:.2f}')
        print(f'Stdev Aggregate Bandwidth (Gb/s): {stddev_bandwidth:.2f}')
        print(f'CI95 Aggregate Bandwidth (Gb/s): {ci95_bandwidth:.2f}')


        if self.df_mpiCollective is None:
            avg_mpiCollective= 0
            stdev_mpiCollective= 0
            ci95_mpiCollective= 0
            stdev_mpiopen=0
        else:
            sum_mpiCollective= self.df_mpiCollective.groupby(["experiment","rank"])["timeSec"].sum()
            mean_mpiCollective_by_exp = sum_mpiCollective.groupby("experiment").mean()
            avg_mpiCollective = mean_mpiCollective_by_exp.mean()
            stdev_mpiCollective = self._sample_stdev(mean_mpiCollective_by_exp)
            ci95_mpiCollective = self._confidence_interval_95(mean_mpiCollective_by_exp)
              
        print(f'AVG MPIOpen (s): {avg_mpiCollective:.2f}')        
        print(f'Stdev MPIOpen (s): {stdev_mpiCollective:.2f}')
        print(f'CI95 MPIOpen (s)): {ci95_mpiCollective:.2f}')  
        
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
                                'Avg_mpiCollective_per_process': avg_mpiCollective, 'std_mpiCollective_per_process': stdev_mpiCollective,
                                'CI95_simulation': ci95_simulation,
                                'CI95_io_per_process': ci95_io_per_process,
                                'CI95_time_per_scenario': ci95_time_per_scenario,
                                'CI95_bandwidth': ci95_bandwidth,
                                'CI95_time_per_record1': CI95_time_per_record1,
                                'CI95_time_per_record2': CI95_time_per_record2,
                                'CI95_time_per_record3': CI95_time_per_record3,
                                'CI95_time_per_record4': CI95_time_per_record4,
                                'CI95_comunication_per_process': ci95_comunication_per_process,
                                'CI95_mpiCollective_per_process': ci95_mpiCollective

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
                     "Avg_mpiCollective_per_process", "std_mpiCollective_per_process",
                     "CI95_simulation",
                     "CI95_io_per_process",
                     "CI95_time_per_scenario",
                     "CI95_bandwidth",
                     "CI95_time_per_record1",
                     "CI95_time_per_record2",
                     "CI95_time_per_record3",
                     "CI95_time_per_record4",
                     "CI95_comunication_per_process",
                     "CI95_mpiCollective_per_process"
                     ]
        escrever_cabecalho = not os.path.exists(path_csv) or os.path.getsize(path_csv) == 0
        if not escrever_cabecalho:
            with open(path_csv, mode='r', newline='', encoding='utf-8') as arquivo_csv:
                reader = csv.DictReader(arquivo_csv)
                linhas_existentes = list(reader)
                cabecalho_atual = reader.fieldnames
            if cabecalho_atual != cabecalho:
                with open(path_csv, mode='w', newline='', encoding='utf-8') as arquivo_csv:
                    writer = csv.DictWriter(arquivo_csv, fieldnames=cabecalho)
                    writer.writeheader()
                    for linha_existente in linhas_existentes:
                        writer.writerow({campo: linha_existente.get(campo, "") for campo in cabecalho})
        with open(path_csv,mode='a',newline='',encoding='utf-8') as arquivo_csv:
            writer = csv.DictWriter(arquivo_csv, fieldnames=cabecalho)
            if escrever_cabecalho:
                writer.writeheader()
            writer.writerow(linha)
            arquivo_csv.close()      

    def _filter_nodes(self, df, nodes_filter):
        if nodes_filter is None:
            return df
        if isinstance(nodes_filter, (int, np.integer)):
            nodes_filter = [nodes_filter]
        filtered_nodes = [node for node in nodes_filter if node in df.index]
        return df.reindex(filtered_nodes)
            
    def plotBandwidth(self, experiments, plotLabel, nodes_filter=None, implementation_formats=None):
        """
        Banda agregada por configuração de nós, comparando múltiplos experimentos.

        experiments : list of (csv_path, label)
            Ex: [(r"...\\plot.csv", "Original"), (r"...\\plot.csv", "MPI-IO")]
        """
        bandwidth_color = '#e57373'
        hatch_by_format = {
            0: '',
            1: '///',
            2: '---',
        }
        if implementation_formats is None:
            implementation_formats = [0, 1]

        # Carrega todos os CSVs
        dfs = [(self._filter_nodes(pd.read_csv(p, index_col='Nodes'), nodes_filter), lbl) for p, lbl in experiments]
        all_nodes = dfs[0][0].index.tolist()
        if not all_nodes:
            raise ValueError("Nenhuma configuração de nós encontrada para o filtro informado.")
        n_nodes   = len(all_nodes)
        n_exp     = len(experiments)
        bar_w     = 0.7 / n_exp
        X         = np.arange(n_nodes)

        fig, ax = plt.subplots(figsize=(max(8, n_nodes * n_exp * 1.2), 5))

        for j, (df, lbl) in enumerate(dfs):
            color   = bandwidth_color
            implementation_format = implementation_formats[j % len(implementation_formats)]
            hatch   = hatch_by_format.get(implementation_format)
            if hatch is None:
                raise ValueError(f"Formato de implementaÃ§Ã£o invÃ¡lido: {implementation_format}. Use 0, 1 ou 2.")
            bar_x   = X + (j - (n_exp - 1) / 2) * bar_w
            df_plot = df.reindex(all_nodes).fillna(0)
            avg_bw  = df_plot['Avg_bandwidth'].values
            std_bw  = df_plot['Stddev_bandwidth'].values
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
        ax.set_xticklabels([f'{n} nós' for n in all_nodes])
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
            categorias[i]= f"{df_csv.index[i]} nós"
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

    
    def plotBlocks(self, experiments, plotLabel, nodes_filter=None, number_blocks=4, implementation_formats=None):
        """
        Tempo médio por categoria de tamanho de mensagem, comparando experimentos.

        experiments : list of (csv_path, label)
            Ex: [(r"...\\plot.csv", "Original"), (r"...\\plot.csv", "MPI-IO")]

        Organização do eixo X:
            Subplots = categorias de tamanho.
            Dentro de cada subplot: barras por implementação ao longo dos nós.
        """
        from matplotlib.patches import Patch

        block_colors  = ['#6B6B6B', '#8E44AD', '#C2185B', '#795548']
        block_labels  = [
            'até 1 KB',
            'até 128 KB',
            'até 1 MB',
            'até 50 MB',
        ]
        block_cols    = [
            ('Avg_time_per_record1', 'Stdev_time_per_record1'),
            ('Avg_time_per_record2', 'Stdev_time_per_record2'),
            ('Avg_time_per_record3', 'Stdev_time_per_record3'),
            ('Avg_time_per_record4', 'Stdev_time_per_record4'),
        ]
        hatch_by_format = {
            0: '',
            1: '///',
            2: '---',
        }
        if implementation_formats is None:
            implementation_formats = [0, 1]
        y_min = 1e-5

        if isinstance(nodes_filter, (int, np.integer)):
            number_blocks = int(nodes_filter)
            nodes_filter = None

        dfs       = [(self._filter_nodes(pd.read_csv(p, index_col='Nodes'), nodes_filter), lbl) for p, lbl in experiments]
        all_nodes = dfs[0][0].index.tolist()
        if not all_nodes:
            raise ValueError("Nenhuma configuração de nós encontrada para o filtro informado.")
        n_nodes   = len(all_nodes)
        n_exp     = len(experiments)
        n_blocks  = min(number_blocks, len(block_labels))

        def _format_seconds(value):
            value = float(value)
            if value >= 100:
                return f"{value:.0f} s"
            if value >= 10:
                return f"{value:.1f} s"
            if value >= 1:
                return f"{value:.2f} s"
            if value >= 0.01:
                return f"{value:.3f} s"
            return f"{value:.1e} s"

        bar_w = min(0.32, 0.75 / max(n_exp, 1))
        X = np.arange(n_nodes)
        fig, axes = plt.subplots(
            2, 2,
            figsize=(max(11, n_nodes * n_exp * 1.05), 8.8),
            sharex=True
        )
        axes = axes.ravel()

        for bi, ax_cat in enumerate(axes):
            if bi >= n_blocks:
                ax_cat.axis('off')
                continue

            avg_col, std_col = block_cols[bi]
            category = block_labels[bi]
            color = block_colors[bi]
            category_max = y_min
            bars_by_exp = []

            for j, (df, lbl) in enumerate(dfs):
                df_plot = df.reindex(all_nodes).fillna(0)
                avg_v = np.maximum(df_plot[avg_col].astype(float).values, y_min)
                std_v = np.maximum(df_plot[std_col].astype(float).values, 0.0)
                lower_err = np.minimum(std_v, np.maximum(avg_v - y_min, 0.0))
                upper_err = std_v
                yerr = np.vstack([lower_err, upper_err])
                bar_x = X + (j - (n_exp - 1) / 2) * bar_w
                implementation_format = implementation_formats[j % len(implementation_formats)]
                hatch = hatch_by_format.get(implementation_format)
                if hatch is None:
                    raise ValueError(f"Formato de implementaÃ§Ã£o invÃ¡lido: {implementation_format}. Use 0, 1 ou 2.")

                ax_cat.bar(
                    bar_x, avg_v, bar_w,
                    yerr=yerr, capsize=4,
                    color=color, edgecolor='#555', linewidth=0.45,
                    hatch=hatch, label=lbl,
                    error_kw=dict(elinewidth=1.0, capthick=1.0, ecolor='#444')
                )
                bars_by_exp.append((bar_x, avg_v, std_v, lbl))

                positive_tops = avg_v + std_v
                category_max = max(category_max, float(np.nanmax(positive_tops)))

            ax_cat.set_yscale('log')
            ax_cat.set_ylim(y_min, max(category_max * 3.0, y_min * 10))
            ax_cat.grid(True, axis='y', which='both', linestyle='--', alpha=0.35)
            ax_cat.tick_params(axis='y', labelsize=9)

            if bi in (0, 2):
                ax_cat.set_ylabel('Tempo médio de envio (s)', fontsize=10)

        for ax_cat in axes:
            if ax_cat.has_data():
                ax_cat.set_xticks(X)
                ax_cat.set_xticklabels([f'{node} nós' for node in all_nodes],
                                       rotation=25, ha='right', fontsize=10)


        cat_h = [Patch(facecolor=c, edgecolor='#555', label=l)
                 for c, l in zip(block_colors[:n_blocks], block_labels[:n_blocks])]
        exp_h = [Patch(facecolor='#ddd', edgecolor='#555',
                       hatch=hatch_by_format[implementation_formats[j % len(implementation_formats)]], label=lbl)
                 for j, (_, lbl) in enumerate(experiments)]

        fig.legend(handles=cat_h, title='Categoria',
                   loc='upper left', bbox_to_anchor=(0.01, 0.995),
                   fontsize=9, title_fontsize=9, framealpha=0.92, ncol=2)
        fig.legend(handles=exp_h, title='Implementação',
                   loc='upper right', bbox_to_anchor=(0.99, 0.995),
                   fontsize=9, title_fontsize=9, framealpha=0.92, ncol=min(n_exp, 3))

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()
        return

    def plotExecutionTime(self, experiments, plotLabel, nodes_filter=None):
        """
        Stacked bar com alturas reais (s), barras lado a lado por experimento.

        experiments : list of (csv_path, label)
            Ex: [(r"...\\plot.csv", "Implementação atual"), (r"...\\plot.csv", "MPI-IO")]
        plotLabel   : título do gráfico

        Segmentos grandes  (>= 12% do total) → tempo dentro da pilha
        Segmentos pequenos                    → sem rótulo
        """
        from matplotlib.patches import Patch

        seg_colors      = ['#4caf50', '#1976d2', '#e57373', '#fbc02d']
        seg_labels_txt  = ['Computação', 'Comunicação', 'E/S', 'MPI_File_open/MPI_File_close']
        exp_hatches     = ['', '///']
        def _col(df, *names):
            """Retorna df[name] para o primeiro nome encontrado nas colunas."""
            for name in names:
                if name in df.columns:
                    return df[name].values
            raise KeyError(f'Nenhuma das colunas encontrada: {names}')

        def _load(csv_path):
            df         = self._filter_nodes(pd.read_csv(csv_path, index_col='Nodes'), nodes_filter)
            avg_sim    = _col(df, 'Avg_Simulation')
            avg_io     = _col(df, 'Avg_io_per_process')
            avg_coll   = _col(df, 'Avg_mpiCollective_per_process',
                                   'Avg_mpiopen_per_process')
            avg_comm   = _col(df, 'Avg_comunication_per_process')
            std_sim    = _col(df, 'Stdev_simulation')
            std_io     = _col(df, 'Stdev_io_per_process')
            std_comm   = _col(df, 'std_comunication_per_process')
            std_coll   = _col(df, 'std_mpiCollective_per_process',
                                   'std_mpiopen_per_process')
            avg_comp   = np.maximum(avg_sim - avg_comm - avg_io - avg_coll, 0)
            std_comp   = np.sqrt(np.maximum(std_sim**2 - std_io**2 - std_coll**2 - std_comm**2, std_sim))
            total      = avg_sim
            total_std  = std_sim
            return (df.index.tolist(),
                    [avg_comp, avg_comm, avg_io, avg_coll],
                    [std_comp, std_comm, std_io, std_coll],
                    total,
                    total_std)

        loaded     = [_load(p) for p, _ in experiments]
        all_nodes  = loaded[0][0]
        if not all_nodes:
            raise ValueError("Nenhuma configuração de nós encontrada para o filtro informado.")
        n_nodes    = len(all_nodes)
        n_exp      = len(experiments)
        bar_w      = 0.35
        grp_gap    = 0.5
        X          = np.arange(n_nodes) * (n_exp * bar_w + grp_gap)

        global_max    = max(float(t.max()) for _, _, _, t, _ in loaded)
        legend_mask = [
            any(np.any(loaded[j][1][s] > 0) for j in range(n_exp))
            for s in range(len(seg_labels_txt))
        ]

        fig, ax = plt.subplots(figsize=(max(10, n_nodes * (n_exp * bar_w + grp_gap) * 2.2), 9.5))
        for j, ((_, exp_label), (nodes, seg_v, seg_s, total, total_std)) in enumerate(
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

                bots += vals

            ax.errorbar(bar_x, total, yerr=total_std,
                        fmt='none', ecolor='#333',
                        elinewidth=1.25, capthick=1.25,
                        capsize=4, zorder=11, clip_on=True)
        # ── Eixos ─────────────────────────────────────────────────────────────
        global_errmax = max(float((t + np.nan_to_num(ts, nan=0.0)).max())
                            for _, _, _, t, ts in loaded)
        ylim_top = max(global_max, global_errmax) * 1.15
        ax.set_ylim(0, ylim_top)
        ax.set_xticks(X)
        ax.set_xticklabels([f'{nd} nós' for nd in all_nodes])
        ax.set_ylabel('Tempo médio de execução (s)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        # ── Legenda ───────────────────────────────────────────────────────────
        seg_h = [Patch(facecolor=c, edgecolor='white', label=l)
                 for c, l, m in zip(seg_colors, seg_labels_txt, legend_mask) if m]
        exp_h = [Patch(facecolor='#ddd', edgecolor='#555',
                       hatch=exp_hatches[j % len(exp_hatches)], label=lbl)
                 for j, (_, lbl) in enumerate(experiments)]
        cat_legend = ax.legend(handles=seg_h, title='Categoria',
                               loc='upper left', fontsize=8, title_fontsize=8,
                               framealpha=0.9, ncol=2)
        ax.add_artist(cat_legend)
        ax.legend(handles=exp_h, title='Implementação',
                  loc='upper right', fontsize=8, title_fontsize=8,
                  framealpha=0.9)

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
    
    def plotIO(self, experiments, block_counts=None, nodes_filter=None,
               small_segment_pct=8, show_small_labels=False,
               show_percent_panel=False, separate_category_plots=True):
        """
        Compara o tempo estimado de E/S usando os tempos medios do plot.csv.

        O segundo argumento pode ser um rotulo do grafico, como nas outras
        funcoes de plot, ou a lista block_counts para compatibilidade.
        block_counts segue a ordem: ate 1 KB, ate 128 KB, ate 1 MB, ate 50 MB.
        experiments segue o formato: [(plot_csv_ou_diretorio, label), ...].
        small_segment_pct define o limite para rotulos internos, baseado no
        percentual da propria pilha, mas os valores exibidos ficam em segundos.
        show_small_labels e mantido por compatibilidade; tempos pequenos ficam
        acima da barra empilhada.
        separate_category_plots exibe um subplot separado para cada categoria
        de bloco; use False para manter a barra empilhada original.
        """
        plotLabel = None
        if isinstance(block_counts, str):
            plotLabel = block_counts
            block_counts = None

        block_labels = ["Até 1 KB", "Até 128 KB", "Até 1 MB", "Até 50 MB"]
        block_colors = ["#6B6B6B", "#8E44AD", "#C2185B", "#795548"]
        block_cols = [
            "Avg_time_per_record1",
            "Avg_time_per_record2",
            "Avg_time_per_record3",
            "Avg_time_per_record4",
        ]
        block_counts = [1801728, 152082, 267264, 27648] if block_counts is None else block_counts
        if len(block_counts) != len(block_labels):
            raise ValueError("block_counts deve ter 4 valores: [ate 1 KB, ate 128 KB, ate 1 MB, ate 50 MB].")
        exp_hatches = ["", "///"]

        def _load_plot_csv(source):
            if source is None:
                source = os.path.join(self.base_directory, "../plot.csv")
            elif isinstance(source, pd.DataFrame):
                return source.copy()
            elif hasattr(source, "base_directory"):
                source = os.path.join(source.base_directory, "../plot.csv")

            source = os.fspath(source)
            if os.path.isdir(source):
                direct_csv = os.path.join(source, "plot.csv")
                parent_csv = os.path.join(source, "../plot.csv")
                source = direct_csv if os.path.exists(direct_csv) else parent_csv
            return pd.read_csv(source, index_col="Nodes")

        if not experiments:
            raise ValueError("Informe experiments no formato [(plot_csv_ou_diretorio, label), ...].")

        dfs = [(self._filter_nodes(_load_plot_csv(path), nodes_filter), label)
               for path, label in experiments]
        all_nodes = dfs[0][0].index.tolist()
        if not all_nodes:
            raise ValueError("Nenhuma configuração de nós encontrada para o filtro informado.")

        def _mpi_processes_from_node(node):
            try:
                return max(float(node), 1.0)
            except (TypeError, ValueError):
                digits = "".join(ch for ch in str(node) if ch.isdigit() or ch == ".")
                return max(float(digits), 1.0) if digits else 1.0

        def _format_seconds(value):
            value = float(value)
            if value >= 100:
                return f"{value:.0f}s"
            if value >= 10:
                return f"{value:.1f}s"
            if value >= 1:
                return f"{value:.2f}s"
            return f"{value:.3f}s"

        rows = []
        for df, label in dfs:
            df_plot = df.reindex(all_nodes).fillna(0)
            for node, row in df_plot.iterrows():
                mpi_processes = _mpi_processes_from_node(node) * self.number_scenarios_per_nodes
                estimated_total_times = np.array([
                    float(row[col]) * count for col, count in zip(block_cols, block_counts)
                ])
                estimated_times_per_process = estimated_total_times / mpi_processes
                total_time = estimated_times_per_process.sum()
                impacts = (estimated_times_per_process / total_time * 100) if total_time > 0 else np.zeros(len(block_labels))
                for category, count, avg_col, est_total_time, est_time, impact in zip(
                        block_labels, block_counts, block_cols,
                        estimated_total_times, estimated_times_per_process, impacts):
                    rows.append({
                        "Experimento": label,
                        "Nodes": node,
                        "Processos_MPI": mpi_processes,
                        "Categoria": category,
                        "Mensagens": count,
                        "Avg_time_record_s": float(row[avg_col]),
                        "Tempo_E/S_estimado_total_s": est_total_time,
                        "Tempo_E/S_estimado_s": est_time,
                        "Impacto_tempo_pct": impact,
                    })

        df_metrics = pd.DataFrame(rows)

        n_nodes = len(all_nodes)
        n_exp = len(dfs)
        bar_w = min(0.32, 0.75 / n_exp)
        X = np.arange(n_nodes)

        if separate_category_plots:
            fig, axes = plt.subplots(
                2, 2,
                figsize=(max(11, n_nodes * n_exp * 1.15), 8.5),
                sharex=True
            )
            axes = axes.ravel()

            for ax_cat, category, color in zip(axes, block_labels, block_colors):
                rows_category = df_metrics[df_metrics["Categoria"] == category]
                category_max = (
                    float(rows_category["Tempo_E/S_estimado_s"].max())
                    if not rows_category.empty
                    else 0
                )
                label_gap = category_max * 0.035 if category_max > 0 else 1
                y_top = category_max * 1.12 + label_gap if category_max > 0 else 1

                for j, (_, label) in enumerate(dfs):
                    bar_x = X + (j - (n_exp - 1) / 2) * bar_w
                    values = []
                    for node in all_nodes:
                        row = df_metrics[
                            (df_metrics["Experimento"] == label) &
                            (df_metrics["Nodes"] == node) &
                            (df_metrics["Categoria"] == category)
                        ].iloc[0]
                        values.append(row["Tempo_E/S_estimado_s"])

                    values = np.array(values)
                    hatch = exp_hatches[j % len(exp_hatches)]
                    ax_cat.bar(
                        bar_x, values, bar_w,
                        color=color, edgecolor="#555", linewidth=0.35,
                        hatch=hatch, label=label
                    )

                    for x, value in zip(bar_x, values):
                        if value > 0:
                            ax_cat.text(
                                x, value + label_gap, _format_seconds(value),
                                ha="center", va="bottom", fontsize=7.2,
                                fontweight="bold", color=color
                            )

                ax_cat.set_title(category, fontsize=11)
                ax_cat.set_ylim(0, y_top)
                ax_cat.grid(True, axis="y", linestyle="--", alpha=0.35)

            axes[0].set_ylabel("Tempo de E/S estimado por processo MPI (s)")
            axes[2].set_ylabel("Tempo de E/S estimado por processo MPI (s)")
            for ax_cat in axes:
                ax_cat.set_xticks(X)
                ax_cat.set_xticklabels([f"{node} nós" for node in all_nodes], rotation=25, ha="right", fontsize=10)

            if plotLabel:
                fig.suptitle(f"Tempo estimado de E/S por categoria - {plotLabel}", fontsize=13)
            else:
                fig.suptitle("Tempo estimado de E/S por categoria", fontsize=13)

            axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
            plt.tight_layout()
            plt.show()
            return df_metrics

        fig_h = 7.5
        fig, ax = plt.subplots(figsize=(max(10, n_nodes * n_exp * 1.15), fig_h))

        total_by_exp = {}
        for _, label in dfs:
            totals = []
            for node in all_nodes:
                rows_node = df_metrics[
                    (df_metrics["Experimento"] == label) &
                    (df_metrics["Nodes"] == node)
                ]
                totals.append(rows_node["Tempo_E/S_estimado_s"].sum())
            total_by_exp[label] = np.array(totals)

        max_total = max(
            (float(totals.max()) for totals in total_by_exp.values() if len(totals)),
            default=0
        )
        label_gap = max_total * 0.035 if max_total > 0 else 1
        max_label_rows = 1

        for j, (_, label) in enumerate(dfs):
            bar_x = X + (j - (n_exp - 1) / 2) * bar_w
            bottoms = np.zeros(n_nodes)
            small_label_counts = np.zeros(n_nodes, dtype=int)
            hatch = exp_hatches[j % len(exp_hatches)]

            for category, color in zip(block_labels, block_colors):
                values = []
                impacts = []
                for node in all_nodes:
                    row = df_metrics[
                        (df_metrics["Experimento"] == label) &
                        (df_metrics["Nodes"] == node) &
                        (df_metrics["Categoria"] == category)
                    ].iloc[0]
                    values.append(row["Tempo_E/S_estimado_s"])
                    impacts.append(row["Impacto_tempo_pct"])

                values = np.array(values)
                ax.bar(bar_x, values, bar_w, bottom=bottoms,
                       color=color, edgecolor="#555", linewidth=0.35,
                       hatch=hatch, label=category if j == 0 else "_nolegend_")

                for i, (x, bottom, value, impact) in enumerate(zip(bar_x, bottoms, values, impacts)):
                    if value > 0 and impact >= small_segment_pct:
                        ax.text(x, bottom + value / 2, _format_seconds(value),
                                ha="center", va="center", fontsize=7.5,
                                fontweight="bold", color="white")
                    elif value > 0:
                        label_row = small_label_counts[i]
                        total = total_by_exp[label][i]
                        x_text = x + ((label_row % 2) - 0.5) * bar_w * 0.55
                        y_text = total + (label_row + 1) * label_gap
                        ax.text(x_text, y_text, _format_seconds(value),
                                ha="center", va="bottom", fontsize=7.5,
                                fontweight="bold", color=color)
                        small_label_counts[i] += 1

                bottoms += values

            max_label_rows = max(max_label_rows, int(small_label_counts.max()) + 2)
        ax.set_ylabel("Tempo de E/S estimado por processo MPI (s)")
        if plotLabel:
            ax.set_title(f"Impacto estimado de E/S - {plotLabel}")
        ax.set_xticks(X)
        ax.set_xticklabels([f"{node} nós" for node in all_nodes], rotation=25, ha="right", fontsize=11)
        y_margin = max(max_label_rows * label_gap, max_total * 0.06)
        ax.set_ylim(0, max_total + y_margin if max_total > 0 else 1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        from matplotlib.patches import Patch
        cat_h = [Patch(facecolor=c, edgecolor="#555", label=l)
                 for c, l in zip(block_colors, block_labels)]
        exp_h = [Patch(facecolor="#ddd", edgecolor="#555",
                       hatch=exp_hatches[j % len(exp_hatches)], label=label)
                 for j, (_, label) in enumerate(dfs)]
        ax.legend(handles=cat_h + exp_h, loc="upper left",
                  fontsize=9, framealpha=0.9, ncol=2)

        plt.tight_layout()
        plt.show()
        return df_metrics

    def PlotHistogram(self,max_size_kb=0):

        labels = ["Até 1 KB", "Até 128 KB", "Até 1 MB", "Até 50 MB"]
        colors = ["#6B6B6B", "#8E44AD", "#C2185B", "#795548"]
        counts = [0, 0, 0, 0]

        for row in self.records:
            size_bytes = row["sizeBytes"]
            size_kb = size_bytes / 1024
            if max_size_kb != 0 and size_kb >= max_size_kb:
                continue

            if size_bytes <= 1024:
                counts[0] += 1
            elif size_bytes <= 32 * 1024:
                counts[1] += 1
            elif size_bytes <= 1024 * 1024:
                counts[2] += 1
            elif size_bytes <= 50 * 1024 * 1024:
                counts[3] += 1

        total = sum(counts)
        percentages = [(count / total * 100) if total > 0 else 0 for count in counts]
        visible_counts = [count if count > 0 else np.nan for count in counts]

        def format_count(value):
            return f"{value:,}".replace(",", ".")

        fig, ax = plt.subplots(figsize=(11, 5))
        y_positions = np.arange(len(labels))
        ax.barh(y_positions, visible_counts, color=colors, height=0.62)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("Quantidade de blocos de dados (escala logaritmica)")

        max_count = max(counts) if counts else 0
        right_limit = max(max_count * 2, 10)
        ax.set_xlim(left=1, right=right_limit)

        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=None))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style="plain", axis="x")
        ax.tick_params(axis="x", which="both", labelbottom=False)
        ax.grid(True, axis="x", which="both", linestyle="-", alpha=0.25)
        ax.set_axisbelow(True)

        for y, count, pct in zip(y_positions, counts, percentages):
            x = count * 1.08 if count > 0 else 1.08
            ax.text(x, y, f"{format_count(count)} ({pct:.1f}%)",
                    va="center", ha="left", fontsize=10)

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














