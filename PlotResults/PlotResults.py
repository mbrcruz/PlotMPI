from MyPlot import *
import numpy as np
import matplotlib.pyplot as plt


path_dados = r"D:\Marcelo\Dropbox (Personal)\Planejamento\Mestrado\Pesquisas\Results\AWS\Lustre - 112 Series - Com rede\2 Nodes"

p =  MyPlot(path_dados,
            2 , 7, 112 ,TypeEvaluation.JUST_SEND)
p.show_config()
p.load_data()
p.computerMetrics()
# p.plotBandwidth(path_dados,"LoboC")
# p.plotExecutionTime(path_dados,"LoboC")