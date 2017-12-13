#! /usr/bin/env python2

"""

Title: EV algorithm for PID tuning
Author: Holm Smidt
Version: 0.9
Date: 10-25-2017

Overview:




"""
import sqlCon
#import matlabEng
import evUtils
#import ctrlUtils

import numpy as np
import random
import math
#import matlab.engine
import matplotlib.pyplot as plt
import datetime
import csv
#import mysql.connector
#from mysql.connector import MySQLConnection, Error
#from ConfigParser import ConfigParser
#import requests
#import urllib


def run_EV():
    """ runs the ev and plots fitness at each run ...
        optionally set debug = true and print the
        population after each mutation
    """

    outputName = 'data/12122017_test1.csv'
    outputFile = open(outputName, 'wb')
    with outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(['Generationg', 'PJmin', 'PJmax', 'PJmean',
                        'OJmin', 'OJmax', 'OJmean',
                        'Parent', 'Offspring'])

    # Control Parameters
    tmax = 65                                       # Fahrenheit
    tmin = 48                                      # Fahrenheit
    setpoint = 53                                   # Fahrenheit
    #ctrl_temps = [tmax, setpoint, setpoint+10, tmin, tmax, setpoint]
    #ctrl_temps = [tmax, setpoint, tmin]
    ctrl_temps = [tmax, setpoint]
    ctrl_params = [10, 200, 45, 350, 60, 2]         # valve_lower, valve_upper, valve_center, windup, sample time, dcmultiplier
    ctrl_interval = [25, 10]                              # mins

    # Evolutionary Parameters
    runtime = 8                                    # iterations
    debug = 1                                       # for printing to screen, turn off on server.

    # init ev operators class, which contains: parents, children, , just one until parallelized
    evops = evUtils.ev_operators()
    #evUtils.ev_fitness(evops.parents, ctrl_temps, ctrl_intergacl, ctrl_params, evops.fit_weights)

    # Fitness values per iterations
    Jparents =  np.zeros((evops.pop_size+1, 1))
    Jchildren = np.zeros((evops.pop_size+1, 1))

    # Fitness average and max over iterations
    Jave = np.zeros((runtime, 1))
    Jmin = np.zeros((runtime, 1))
    Jmax = np.zeros((runtime, 1))

    # calc first fitness
    evfit = evUtils.ev_fitness(evops.parents, ctrl_temps, ctrl_interval, ctrl_params, evops.fit_weights, debug)
    if debug:
        print "Population 0: ", evops.parents
    Jparent = evfit.run_eval()
    Jave[0] = np.mean(Jparent)
    Jmin[0] = np.min(Jparent)
    Jmax[0] = np.max(Jparent)

    if debug:
        print "Population 0: ", evops.parents
        print "Fitness : ", Jparent
        print "Mean: ", Jave[0], " Min", Jmin[0]

    for i in range(runtime):

        # when i == 0, this is

        # Add immigrants to the parents
        #if not i % 2:
        #    if debug: print "Should query graph DB"
        #    evops.queryGraph(1)

        # mutation
        evops.children = evops.recombine()
        evops.children = evops.mutate(evops.parents)
        if debug: print "Offspring Population ", i, " : ", evops.children
        evfit = evUtils.ev_fitness(evops.children, ctrl_temps, ctrl_interval, ctrl_params, evops.fit_weights, debug)
        Jchildren = evfit.run_eval()
        if debug:
            print "Offspring Population", i, " : ", evops.children
            print "Fitness : ", Jchildren

        outputFile = open(outputName, 'a')
        with outputFile:
            writer = csv.writer(outputFile)
            writer.writerow([i, np.min(Jparent), np.max(Jparent), np.mean(Jparent),
                            np.min(Jchildren), np.max(Jchildren), np.mean(Jchildren),
                            str(evops.parents),
                            str(evops.children)
                            ])
        # increase generation count,
        i += 1

        # selection of new generation (parents)
        evops.parents, Jparent = evops.selection_comp(Jparent, Jchildren)

        # calculate fitness
        #evfit = evUtils.ev_fitness(evops.parents, ctrl_temps, ctrl_interval, ctrl_params, evops.fit_weights, debug)
        #Jparent = evfit.run_eval()
        Jave[i] = np.mean(Jparent)
        Jmin[i] = np.min(Jparent)

        if debug:
            print "Parent Population", i, " : ", evops.parents
            print "Fitness : ", Jparent
            print "Mean: ", Jave[i], " Min: ", Jmin[i]



    #plot_simout(simout_init, simout_mid, simout, "../sim_results/simout_30s5r6m.png")
    #plot_aveFitness(Jave, "../sim_results/objective_30s5r6m.png")

    if debug:
        print 'Final pop: ', evops.parents
        print 'Final J: ', Jparent
        print 'Jave: ', Jave
        print 'Jmin: ', Jmin

    with outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(["Final", np.min(Jparent), np.max(Jparent), np.mean(Jparent),
                        "", "", "",
                        evops.parents,
                        ""
                        ])

if __name__ == "__main__":

    run_EV()
