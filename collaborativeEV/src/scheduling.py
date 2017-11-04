import numpy as np
import random, math
import matlab.engine
import matplotlib.pyplot as plt
import datetime, pause
import mysql.connector
from mysql.connector import MySQLConnection, Error
from ConfigParser import ConfigParser
import requests, urllib

def execRequest(tempset):
    params = urllib.urlencode({'tempset': tempset, 'Kp': 7.0, 'valve_lower_limit': 10, 'valve_upper_limit': 255, 'valve_center': 35, 'dcmultiplier':2})
    response = requests.get("http://deep.outtter.space:51880/kahuku_farms/PID_container2/?%s" % params, auth=('user', 'Mat5uda$$'))
    print response

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
def schedule(interval, t_low, t_high, t_steps, iters):
    temps = np.linspace(t_low, t_high, t_steps)
    if iters > t_steps and iters <= 2*t_steps:
        currentTemp = np.concatenate((np.random.choice(temps, t_steps, replace=False), np.random.choice(temps, iters-t_steps, replace=False)))
    else :
        currentTemp = np.random.choice(temps, iters, replace=False)
    print currentTemp
    currentTemp = [60, 52, 56, 53, 58, 54]
    startTime = datetime.datetime.now()  #+ datetime.timedelta(minutes=3)
    for i in range(iters):
        dt = i*interval
        nextTime = startTime + datetime.timedelta(minutes=dt)
        print "nextTime: " , nextTime
        pause.until(nextTime)
        print "executing request with temp= ", currentTemp[i]
        execRequest(currentTemp[i])

if __name__ == '__main__':
    schedule(30, 52, 60, 3, 6)
    execRequest(54)
