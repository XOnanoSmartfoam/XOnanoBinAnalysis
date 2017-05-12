# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:47:06 2017

@author: gkenworthy
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import json
import os
#conda install -c bjornfjohansson mpldatacursor=0.6.2
from mpldatacursor import datacursor

def CalcStrikeForce_Area(data):
    # make up zero level
    zero = np.average(data[0:500])
    
    area = 0
    peak_index = np.argmax(data)
    area = np.sum(np.abs(data[peak_index-350:peak_index+1000] - zero))

    return area

# Using a closure to access data. Ideally you'd use a "functor"-style class.
def onNote(**kwargs):
    #dist = abs(np.array(x) - kwargs['x'])
    #i = dist.argmin()
    
    for root, subdirs, files in os.walk(mypath):    
        for f in files:    
            if f == kwargs['label']:
                with open(os.path.join(root,f), 'r') as fp:
                    data = json.load(fp)                
                    ax1.plot(data['zeroed_data'], label=f + ' ' + data['kicker'] +' '+ data['type']+' Strength '+ data['strength'])
    leg = ax1.legend()
    ##return kwargs['label']

    for root, subdirs, files in os.walk(mypath2):    
        for f in files:    
            if f == kwargs['label']:
                with open(os.path.join(root,f), 'r') as fp:
                    data = json.load(fp)                
                    ax1.plot(data['zeroed_data'], label=f + ' ' + data['kicker'] +' '+ data['type']+' Strength '+ data['strength'])
    leg = ax1.legend()

    for root, subdirs, files in os.walk(mypath3):    
        for f in files:    
            if f == kwargs['label']:
                with open(os.path.join(root,f), 'r') as fp:
                    data = json.load(fp)                
                    ax1.plot(data['zeroed_data'], label=f + ' ' + data['kicker'] +' '+ data['type']+' Strength '+ data['strength'])
    leg = ax1.legend()
    
    return kwargs['label']

mypath = r'C:\Users\2020 Armor\Documents\Python Logger V3\May 10 2017\Christian'
mypath2 = r'C:\Users\2020 Armor\Documents\Python Logger V3\May 10 2017\Dylan'
mypath3 = r'C:\Users\2020 Armor\Documents\Python Logger V3\May 10 2017\Josh'
mypath4 = r'C:\Users\2020 Armor\Documents\Python Logger V3\May 10 2017\Taylon'

fig, (ax1, ax2) = plt.subplots(1,2, sharex=False, sharey=False)

ax1.set_title("Collected Data")    
ax1.set_xlabel('time')
ax1.set_ylabel('Measured Voltage')

ax2.set_title("Scoring Correlation")    
ax2.set_xlabel('User Score')
ax2.set_ylabel('Algorithm Calculated Score')
ax2.set_xlim([0,10])


filenames = next(os.walk(mypath))[2]


for root, subdirs, files in os.walk(mypath):    
    
    for f in files:
        #print(os.path.join(root,f))
        filename, file_extension = os.path.splitext(f)
        if file_extension != '.json':
            continue
        
        with open(os.path.join(root,f), 'r') as fp:
            print(filename)
            data = json.load(fp)
            #print (data)
                    
            #ax1.plot(data['raw_data'], label=data['kicker'] +' '+ data['type']+' Strength '+ data['strength'])
            
            # Uncomment to use the python calculation
            #score = CalcStrikeForce_Area(data['zeroed_data'])  
            
            # Use the score calculated by the uC
            score = data['area_score']
            
            strength = data['strength'].split(' ')[0]
            ax2.scatter(strength,score, marker='.', label = f, color = 'red')

for root, subdirs, files in os.walk(mypath2):    
    
    for f in files:
        #print(os.path.join(root,f))
        filename, file_extension = os.path.splitext(f)
        if file_extension != '.json':
            continue
        
        with open(os.path.join(root,f), 'r') as fp:
            print(filename)
            data = json.load(fp)
            #print (data)
                    
            #ax1.plot(data['raw_data'], label=data['kicker'] +' '+ data['type']+' Strength '+ data['strength'])
            
            # Uncomment to use the python calculation
            #score = CalcStrikeForce_Area(data['zeroed_data'])  
            
            # Use the score calculated by the uC
            score = data['area_score']
            
            strength = data['strength'].split(' ')[0]
            ax2.scatter(strength,score, marker='.', label = f, color = 'blue')     


for root, subdirs, files in os.walk(mypath3):    
    
    for f in files:
        #print(os.path.join(root,f))
        filename, file_extension = os.path.splitext(f)
        if file_extension != '.json':
            continue
        
        with open(os.path.join(root,f), 'r') as fp:
            print(filename)
            data = json.load(fp)
            #print (data)
                    
            #ax1.plot(data['raw_data'], label=data['kicker'] +' '+ data['type']+' Strength '+ data['strength'])
            
            # Uncomment to use the python calculation
            #score = CalcStrikeForce_Area(data['zeroed_data'])  
            
            # Use the score calculated by the uC
            score = data['area_score']
            
            strength = data['strength'].split(' ')[0]
            ax2.scatter(strength,score, marker='.', label = f, color = 'green')

for root, subdirs, files in os.walk(mypath4):    
    
    for f in files:
        #print(os.path.join(root,f))
        filename, file_extension = os.path.splitext(f)
        if file_extension != '.json':
            continue
        
        with open(os.path.join(root,f), 'r') as fp:
            print(filename)
            data = json.load(fp)
            #print (data)
                    
            #ax1.plot(data['raw_data'], label=data['kicker'] +' '+ data['type']+' Strength '+ data['strength'])
            
            # Uncomment to use the python calculation
            #score = CalcStrikeForce_Area(data['zeroed_data'])  
            
            # Use the score calculated by the uC
            score = data['area_score']
            
            strength = data['strength'].split(' ')[0]
            ax2.scatter(strength,score, marker='.', label = f, color = 'purple')

leg = ax1.legend()

y_min, y_max = max_area = ax2.get_ylim()
ax2.set_ylim([0,y_max])
ax2.set_xlim([0,11])

datacursor(formatter=onNote, display='multiple', draggable=True)

plt.show()
