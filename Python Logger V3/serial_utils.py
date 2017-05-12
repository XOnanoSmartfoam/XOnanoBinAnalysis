import threading
import serial
import serial.tools.list_ports


#serial_port = serial.Serial(port, baud, timeout=0)

def handle_data(data):
    print(data)

def read_from_port(ser):
    while not connected:
        #serin = ser.read()
        connected = True

        while True:
           print("test")
           reading = ser.readline().decode()
           handle_data(reading)


#
#thread = threading.Thread(target=read_from_port, args=(serial_port,))
#thread.start()
#

def get_serial_ports():
    ports = []
    for i in serial.tools.list_ports.comports():
        ports.append(i[0])
    return ports