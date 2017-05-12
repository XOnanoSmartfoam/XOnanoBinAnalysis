##########################################################################################
# 
#                     ____   ___    ______   ___  
#                    |___ \ / _ \  / /___ \ / _ \ 
#                      __) | | | |/ /  __) | | | |
#                     / __/| |_| / /  / __/| |_| |
#                    |_____|\___/_/  |_____|\___/ 
#                                                
#                        _    ____  __  __  ___  ____  
#                       / \  |  _ \|  \/  |/ _ \|  _ \ 
#                      / _ \ | |_) | |\/| | | | | |_) |
#                     / ___ \|  _ <| |  | | |_| |  _ < 
#                    /_/   \_\_| \_\_|  |_|\___/|_| \_\
#
#
#           Wireless Data Logger
#           version 1.0
#
#           Written by G. Kenworthy
#           For 20/20 Armor
#
#





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
#import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
import json
import datetime
import os


sample_interval = 1.0/40000 


class Window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__()
        
        self.initUI()
        
    def initUI(self):
        
        # Setup initial path and file
        self.workingDir = os.getcwd()
        self.workingFile = r"data.csv"

        self.data_q = []
        
        # UI updates on a timer
        self.refreshInternal = 500
        self.refresh_timer = QtCore.QTimer()
        self.refresh_timer.timeout.connect(self.on_refresh_timer)
        self.refresh_timer.start(self.refreshInternal) 

        self.active_port = ''

        self.current_event = 0
        self.last_event = 0

        # state inits
        self.connection_state = False # Active connection
        
        # Create UI
        self.build_main_panel()
        self.update_interface_state()

        self.ready_state = True # Currently ready to plot new data

    def build_main_panel(self):


        self.lbl_port = QtWidgets.QLabel('Port')
        self.cmb_port = QtWidgets.QComboBox(self)
        self.cmb_port.activated[str].connect(self.on_select_port_cmb)


        self.lbl_status = QtWidgets.QLabel('Status: ')
        self.txt_status = QtWidgets.QLineEdit()
        self.txt_status.setFixedWidth(100)

        self.btn_connect = QtWidgets.QPushButton('Connect')
        self.btn_connect.clicked.connect(self.on_connect_btn)
        self.btn_disconnect = QtWidgets.QPushButton('Disconnect')
        self.btn_disconnect.clicked.connect(self.on_disconnect_btn)

        self.lbl_buffer = QtWidgets.QLabel('Buffer: ')
        self.txt_buffer = QtWidgets.QLineEdit()

        # Kick info fields
        self.lbl_kicker = QtWidgets.QLabel('Kicker')
        self.txt_kicker = QtWidgets.QLineEdit()
        self.lbl_kickee = QtWidgets.QLabel('Kickee')
        self.txt_kickee = QtWidgets.QLineEdit()
        self.lbl_type   = QtWidgets.QLabel()
        self.cmb_type   = QtWidgets.QComboBox()
        self.cmb_type.addItems(['Back Kick',
                                'Round Kick',
                                'Cut Kick',
                                'Punch',
                                'Noise'])
        self.lbl_strength = QtWidgets.QLabel('Strength')
        self.cmb_strength = QtWidgets.QComboBox()
        self.cmb_strength.addItems(['1 (soft)', '2', '3', '4', '5','6','7','8','9','10 (hard)'])

        self.lbl_notes = QtWidgets.QLabel('Notes:')
        self.txt_notes = QtWidgets.QLineEdit()

        # File saving stuff
        self.lbl_working_dir = QtWidgets.QLabel('Working Directory: ')
        self.txt_working_dir = QtWidgets.QLineEdit()
        self.txt_working_dir.setText(os.getcwd())
        self.btn_working_dir = QtWidgets.QPushButton('...')
        self.btn_working_dir.setFixedWidth(30)
        self.btn_working_dir.clicked.connect(self.on_working_dir_btn)

        self.btn_save = QtWidgets.QPushButton('Save')
        self.btn_save.clicked.connect(self.on_save_btn)
        self.btn_skip = QtWidgets.QPushButton('Skip')
        self.btn_skip.clicked.connect(self.on_skip_btn)

        # a figure instance to plot on
        self.figure = Figure(figsize=(5,4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel('time [ms]')
        self.axes.set_ylabel('magnitude')
        self.axes.axis([0, 0.07, -128, 128])

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        #self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        #self.button = QtGui.QPushButton('Plot')
        #self.button.clicked.connect(self.plot)

        # set the layout
        #grid = QtGui.QVBoxLayout()
        #grid.addWidget(self.toolbar)
        #grid.addWidget(self.canvas)
        #grid.addWidget(self.button)              



        # Port and connection buttons across the top
        self.hbox1 = QtWidgets.QHBoxLayout()
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.lbl_port)
        self.hbox1.addWidget(self.cmb_port)
        self.hbox1.addWidget(self.lbl_status)
        self.hbox1.addWidget( self.txt_status)
        self.hbox1.addWidget(self.btn_connect)
        self.hbox1.addWidget(self.btn_disconnect)

        self.hbox1.addWidget(self.lbl_buffer)
        self.hbox1.addWidget(self.txt_buffer)

        self.hbox2 = QtWidgets.QHBoxLayout()
        #self.hbox2.addStretch(1)
        self.hbox2.addWidget(self.lbl_kicker)
        self.hbox2.addWidget(self.txt_kicker)
        self.hbox2.addWidget(self.lbl_kickee)
        self.hbox2.addWidget(self.txt_kickee)
        self.hbox2.addWidget(self.lbl_type)
        self.hbox2.addWidget(self.cmb_type)
        self.hbox2.addWidget(self.lbl_strength)
        self.hbox2.addWidget(self.cmb_strength)

        self.hbox3 = QtWidgets.QHBoxLayout()
        #self.hbox4.addStretch(1)
        self.hbox3.addWidget(self.lbl_notes)
        self.hbox3.addWidget(self.txt_notes)

        self.hbox4 = QtWidgets.QHBoxLayout()
        #self.hbox4.addStretch(1)
        self.hbox4.addWidget(self.lbl_working_dir)
        self.hbox4.addWidget(self.txt_working_dir)
        self.hbox4.addWidget(self.btn_working_dir)
        self.hbox4.addWidget(self.btn_save)
        self.hbox4.addWidget(self.btn_skip)
        
        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.addLayout(self.hbox1)
        self.vbox.addLayout(self.hbox2)
        self.vbox.addLayout(self.hbox3)        
        self.vbox.addLayout(self.hbox4)        

        self.vbox.addWidget(self.canvas)
        #self.vbox.addStretch(1)

        self.setLayout(self.vbox) 

        self.set_interface_state()


        self.setGeometry(500, 500, 500, 500)
        self.setWindowTitle('20/20 Armor - Data Logger')    
        self.show()        

    def on_refresh_timer(self):

        self.txt_buffer.setText(str(len(self.data_q)))
        
        if(self.ready_state and len(self.data_q)):
            self.ready_state = False
            self.plot()

        # if(len(self.data_q)):
        #     #data available
        #     self.plot()
        #     self.refresh_timer.stop()

        self.update_interface_state()
        
    def on_select_port_cmb(self):
        
        if(self.cmb_port.currentText() != self.active_port):
            self.on_disconnect_btn
        self.active_port = self.cmb_port.currentText()
        self.set_interface_state()
            
        
    def on_connect_btn(self):
        
        self.active_port = self.cmb_port.currentText()
        self.connection_state = True

        self.data_q = []

        self.com_monitor = Serial_com(
            data_q = self.data_q,
            port_num = self.active_port,
            port_baud = 115200)

        self.com_monitor.start()
        # start polling

        self.set_interface_state()

        
    def on_disconnect_btn(self):
        print('disconnect')
        self.connection_state = False
        self.com_monitor.alive.clear()
        self.set_interface_state()

    def on_working_dir_btn(self):
        #set the working directory
        self.txt_working_dir.setText (str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")))

    def on_save_btn(self):
        print ('save data')
        if len(self.data_q):
            data_set = dict()
            data_set['kicker'] = self.txt_kicker.text()
            data_set['kickee'] = self.txt_kickee.text()
            data_set['type']   = self.cmb_type.currentText()
            data_set['strength'] = self.cmb_strength.currentText()
            data_set['notes'] = self.txt_notes.text()
            data_set['date'] = datetime.datetime.today().isoformat()
            data_set['sensor_zero'] = self.data_q[0][0]
            data_set['raw_data'] = self.data_q[0][2]
            data_set['area_score'] = self.data_q[0][1]
            data_set['zeroed_data'] = [x - self.data_q[0][0] for x in self.data_q[0][2]]
                    

            filename = os.path.join(self.txt_working_dir.text(),datetime.datetime.today().strftime('%Y-%m-%d %Hh%Mm%Ss'))
            filename += '.json'
            
            jsondata = json.dumps(data_set, ensure_ascii=False)
            f = open(filename, "w")
            f.write(jsondata)
            f.close()

            self.txt_notes.clear()

            #pop the last data set off the stack
            self.data_q.pop(0)
            self.ready_state = True
        #self.refresh_timer.start


    def on_skip_btn(self):
        print('skip data')
        if len(self.data_q):
            self.data_q.pop(0)
            self.axes.cla()
            self.canvas.draw()
            self.ready_state = True
            #self.refresh_timer.start
        

    def update_interface_state(self):
        # update port list
        ports = get_serial_ports()
        self.cmb_port.clear()
        self.cmb_port.addItems(ports)
        index = self.cmb_port.findText(self.active_port)
        if index >= 0:
            self.cmb_port.setCurrentIndex(index)
        

    def set_interface_state(self):
        self.update_interface_state()
        if self.cmb_port.currentText() == '':
            self.btn_connect.setEnabled( False)
            self.btn_disconnect.setEnabled(False)
            #self.cmdStartRecord.Enabled = False
            #self.cmdStopRecord.Enabled = False
        else:            
            if self.connection_state:
                self.btn_connect.setEnabled(False)
                self.btn_disconnect.setEnabled(True) 

                self.txt_status.setText("Connected")
                self.btn_connect.setEnabled(False)
                self.btn_disconnect.setEnabled(True)

                # if self.record_active:
                #     self.txtStatus.SetValue("Recording") 
                #     self.cmdStartRecord.Enabled = False
                #     self.cmdStopRecord.Enabled = True
                # else:
                #     self.txtStatus.SetValue("Connected") 
                #     self.cmdStartRecord.Enabled = True
                #     self.cmdStopRecord.Enabled = False 
                
            else:
                self.txt_status.setText("Disconnected")
                self.btn_connect.setEnabled(True)
                self.btn_disconnect.setEnabled(False)

                # self.cmdStartRecord.Enabled = False
                # self.cmdStopRecord.Enabled = False  

    def plot(self):
        #plot the oldest data in the queue
        print('plotting')
        self.axes.cla()
        self.axes.plot(np.linspace(0,len(self.data_q[0][2])*sample_interval,len(self.data_q[0][2])), 
                       self.data_q[0][2],color='r', linewidth=2.0)
        self.axes.set_ylim([-0,255])
        #self.axes.plot(self.data_q[0],color='r', linewidth=2.0)
        

        
        # refresh canvas
        self.canvas.draw()




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())