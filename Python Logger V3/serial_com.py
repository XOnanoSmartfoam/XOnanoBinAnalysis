import serial
import threading
#import struct
#import re
import csv

class Serial_com(threading.Thread):

    def __init__(   self,
                    data_q,
                    port_num,
                    port_baud,
                    port_bytesize=serial.EIGHTBITS,
                    port_parity=serial.PARITY_NONE,
                    port_stopbits=serial.STOPBITS_ONE,                    
                    port_timeout=0.01,
                    port_xonxoff=0, # software flow control
                    port_rtscts=0, # hardware (RTS/CTS) flow control   
                    port_dsrdtr=True # hardware(DSR/DTR) flow control
                    ):
                        
        threading.Thread.__init__(self)
        
        self.serial_port = None
        self.serial_arg = dict( port=port_num,
                                baudrate=port_baud,
                                bytesize=port_bytesize,
                                parity=port_parity,
                                stopbits=port_stopbits,                                
                                timeout=port_timeout,
                                xonxoff=port_xonxoff,
                                rtscts=port_rtscts,
                                dsrdtr=port_dsrdtr
                                )
        self.data_q = data_q      
        
        self.new_data = False

        #self.dataBuffer = bytearray()
        self.dataBuffer_list = []
        self.lineBuffer = ''
        
        #self.start_seq = bytearray([0,0,0])
        self.start_seq_str = [0,0,0]

        self.alive = threading.Event()
        self.alive.set()

    def run(self):
        try:
            if self.serial_port: 
                self.serial_port.close()
            self.serial_port = serial.Serial(**self.serial_arg)
            
            self.serial_port.setDTR(True)
            
        except serial.SerialException as e:
            #self.error_q.put(e.message)
            print ('serial error')
            return
        
        
        while self.alive.isSet(): 
            # read data
            
            while(self.serial_port.inWaiting()):
                char = self.serial_port.read(1).decode("utf-8")
                if char == '\r':
                    self.dataBuffer_list.append(self.lineBuffer)
                    self.lineBuffer = ''
                elif char != ' ':
                    self.lineBuffer += char
                    
                    
            if len(self.dataBuffer_list) > 1:
                line = list(map(int,self.dataBuffer_list[0].split(',')))
                self.dataBuffer_list.pop(0)
                if line[0:3] == [0,0,0]:
                    # start line
                    num_bytes = line[3]
                    sensor_zero = line[4]
                    area_score = line[5]
                    data = list(map(int,self.dataBuffer_list[0].split(',')))
                    self.dataBuffer_list.pop(0)
                    if len(data) != num_bytes:
                        print('bad packet? data length is ' + str(len(data)) + ' and num bytes expected is ' + str(num_bytes))
                        
                    self.data_q.append([sensor_zero, area_score, data])
                    self.new_data = True
                    print('received event')
                    
                    
                    
                
            
                
#            
#            if (self.serial_port.inWaiting()):
#                # Read the data from the line
#                data = self.serial_port.read(self.serial_port.inWaiting()).decode("utf-8")
#                # split based on carriage returns
#                data_lines = data.split('\r')
#                self.dataBuffer_list[-1] += data_lines[0]
#                if len(data_lines) > 1:
#                    self.dataBuffer_list += data_lines[1:]
#            
#                for line in csv.reader(data_lines):
#                    #data = re.compile(',\s?').split(line)
#                    
#                    #print(data)
#                    #if line[0:3] == [0, 0, 0]:
#                    if line != '':
#                        print(line)
#                    
            
#            index = self.dataBuffer.find(self.start_seq) 
#            if index >= 0 and len(self.dataBuffer) >= index + 5:
#                data_length = struct.unpack(">H",self.dataBuffer[index+3:index+5])[0]
#                # TODO check if data length is sensible
#                if len(self.dataBuffer) >= index + 5 + data_length:
#                    self.data_q.append(struct.unpack(str(self.data_length)+'c', \
#                        self.dataBuffer[index+5:index+5+data_length]))
#                    self.dataBuffer = self.dataBuffer[index+5+data_length:]
            
        # clean up
        if self.serial_port:
            self.serial_port.close()