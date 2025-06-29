from MyPlot import *
import numpy as np
import matplotlib.pyplot as plt



p =  MyPlot(r"D:\Marcelo\Dropbox (Personal)\Planejamento\Mestrado\Pesquisas\Results\LoboC\Lustre - 2 Series\2-nodes",2,2
            ,TypeEvaluation.COMUNICATION_AND_IO)
p.show_config()
p.load_data()
p.plot()
#p.plotMean(r"D:\Marcelo\Dropbox (Personal)\Planejamento\Mestrado\Pesquisas\Results\LoboC\Lustre - 3 series","LoboC")
