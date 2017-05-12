##CUSTOM PLOTTER
import sys
import os
#from PyQt5 import QtGui
#from PyQt5 import QtCore

import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtGui, QtWidgets
from serial_utils import get_serial_ports
from serial_com import Serial_com

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from mpldatacursor import datacursor

import numpy as np
import json
import datetime
import os

colorList = ['red', 'blue', 'green', 'purple', 'yellow', 'black', 'gray']
colorInd = 0

class Window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__()
        
        self.initUI()

    def initUI(self):
        
        # Setup initial path and file
        self.workingDir = os.getcwd()
        self.workingFile = r"data.csv"


        
        # Create UI
        self.build_main_panel()
        #self.plot()


    def build_main_panel(self):

        plt.ion()
        # open file dialog
        self.btn1 = QtWidgets.QPushButton("Open File")
        self.btn1.clicked.connect(self.getfiles)
      
        self.btn2 = QtWidgets.QPushButton('Change Colour')
        self.btn2.clicked.connect(self.changeColour)
        
        self.btn3 = QtWidgets.QPushButton('Remove Signal')
        self.btn3.clicked.connect(self.removeSignal)
        
        # a figure instance to plot on
        self.figure = Figure(figsize=(5,4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel('User Score')
        self.axes.set_ylabel('Calculated Score')
        self.axes.set_xlim([0,11])
        self.axes.set_ylim([0,300000])

        # a second figure instance to plot on
        self.figure2 = Figure(figsize=(5,4), dpi=100)
        self.axes2 = self.figure2.add_subplot(111)
        self.axes2.set_xlabel('Time(ms)')
        self.axes2.set_ylabel('Measured Voltage')
        self.axes2.set_xlim([0,3000])
        self.axes2.set_ylim([0,300])

        
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.canvas2 = FigureCanvas(self.figure2)
        self.canvas2.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        self.canvas2.updateGeometry()

        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.addWidget(self.btn1)
        self.vbox.addWidget(self.btn2)
        self.vbox.addWidget(self.canvas)
        self.vbox.addWidget(self.canvas2)
        self.vbox.addWidget(self.btn3)
        #self.vbox.addStretch(1)

        self.setLayout(self.vbox) 


        self.setGeometry(500, 500, 500, 500)
        self.setWindowTitle('20/20 Armor - Data Plotter')    
        self.show()

    def getfiles(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)
            self.plot(fileName)

    def changeColour(self):
        global colorInd
        colorInd = colorInd + 1
        if(colorInd > 6):
            colorInd = 0

    def plot(self, file):
        #plot the oldest data in the queue
        fp = open(file)
        data = json.load(fp)
        print(data['area_score'])
        print('plotting')

        score = data['area_score']
            
        strength = data['strength'].split(' ')[0]
        self.axes.scatter(strength,score, marker='.', label = file, color = colorList[colorInd],picker=5)
        
    
        
        # refresh canvas
        self.canvas.draw()
        self.canvas.mpl_connect('pick_event', self.onpick)

        

    def onpick(self, event):
        print("Got here")
        fileName = event.artist.get_label()
        
        

        print (fileName)
        fp = open(fileName)
        data = json.load(fp)
#
        ##print (data['raw_data'])
        
        #self.axes2.plot(data['raw_data'], label=fileName + ' ' + data['kicker'] +' '+ data['type']+' Strength '+ data['strength'], color = colorList[colorInd])
        self.axes2.plot(data['raw_data'], label=data['kicker'] +' '+ data['type']+' Strength '+ data['strength'], color = colorList[colorInd])

        self.axes2.legend()
        self.canvas2.draw()
        
    def removeSignal(self):

        print(self.axes2.lines)
        self.axes2.lines.pop()
        self.axes2.legend_.remove()
        self.axes2.legend()
        self.canvas2.draw()
 
        

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
