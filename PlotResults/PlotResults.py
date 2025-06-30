from MyPlot import *
import numpy as np
import matplotlib.pyplot as plt



p =  MyPlot(r"D:\Marcelo\Dropbox (Personal)\Planejamento\Mestrado\Pesquisas\Results\LoboC\Lustre - 2 Series\16-nodes", 8,2
            ,TypeEvaluation.JUST_SEND)
p.show_config()
p.load_data()
p.plot()
#p.plotMean(r"D:\Marcelo\Dropbox (Personal)\Planejamento\Mestrado\Pesquisas\Results\LoboC\Lustre - 3 series","LoboC")
