import numpy as np
import random, math
import matlab.engine
import matplotlib.pyplot as plt
import datetime, pause
import mysql.connector
from mysql.connector import MySQLConnection, Error
from ConfigParser import ConfigParser
import requests, urllib

#PID2?tempset=53&Kp=1.8&Ki=0.1&Kd=0.1&valve_lower_limit=10&valve_upper_limit=120&valve_center=40&windup=20&sampletime=30&dcmultiplier=2
def execRequest(tempset):
    params = urllib.urlencode({'tempset': tempset, 'Kp': 1.3, 'Ki': 0.05, 'Kd': 0.4, 'valve_lower_limit': 10, 'valve_upper_limit': 150, 'valve_center': 40, 'windup': 450, 'sampletime':45, 'dcmultiplier':2})
    response = requests.get("http://deep.outtter.space:51880/ilaniwai/PID2?%s" % params, auth=('user', 'Mat5uda$$'))
    return response

"""
Simple algorithm that starts 2 minutes after being called and
sends requests to the server to update control setting temperature.

Inputs:
    interval - minutes
    t_low - low temperature bound
    t_high - upper temperature bound
    t_steps - number of total temperature steps equally spaced
    iters - number of iterations to run this
"""
def schedule(interval, t_low, t_high, t_set):
    #temps = np.linspace(t_low, t_high, t_steps)
    temps = [t_high, t_set, t_low, t_set]
    """
    if iters > t_steps and iters <= 2*t_steps:
        currentTemp = np.concatenate((np.random.choice(temps, t_steps, replace=False), np.random.choice(temps, iters-t_steps, replace=False)))
    else :
        currentTemp = np.random.choice(temps, iters, replace=False)
    print currentTemp
    """
    #currentTemp = [60, 52, 56, 53, 58, 54]
    startTime = datetime.datetime.now()  #+ datetime.timedelta(minutes=3)
    for i in range(len(temps)):
        dt = i*interval
        nextTime = startTime + datetime.timedelta(minutes=dt)
        print "nextTime: " , nextTime
        pause.until(nextTime)
        print "executing request with temp= ", temps[i]
        execRequest(temps[i])

if __name__ == '__main__':
    schedule(30, 48, 66, 54)
    execRequest(54)
