from MyPlot import *
import numpy as np
import matplotlib.pyplot as plt


path_dados = r"D:\Marcelo\Dropbox (Personal)\Planejamento\Mestrado\Pesquisas\Results\AWS\Lustre - 448 Series - Com rede\16 Nodes"

p =  MyPlot(path_dados,
            16 , 7, 448 ,TypeEvaluation.JUST_SEND)
p.show_config()
p.load_data()
p.computerMetrics()
p.PlotHistogram()
# p.plotBandwidth(path_dados,"LoboC")
# p.plotExecutionTime(path_dados,"LoboC")