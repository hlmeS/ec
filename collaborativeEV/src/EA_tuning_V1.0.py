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
#import mysql.connector
#from mysql.connector import MySQLConnection, Error
#from ConfigParser import ConfigParser
#import requests
#import urllib




def plot_fitness(xarr):
    """ given a range of genotypes, plot
        the fitness as a function of the genotypes
    """
    return 0

def plot_simout(sim0, simm, simf, filename):
    """
    pick one gene at random and plot it's response
    """

    psize = len(sim0[:, 0, 0])
    idx = np.random.randint(0, psize)

    sample0 = sim0[idx, :, : ]
    t = sample0[:,2]
    u = sample0[:,1]
    sim0_y = sample0[:,0]

    samplem = simm[idx, :, :]
    simm_y = samplem[:, 0]

    samplef = simf[idx, :, :]
    simf_y = samplef[:,0]


    """ plot """
    title = 'System Response with intial and final populations '
    plot_init(title, [0, max(t)], [0,20], "Time", "Position")
    plt.plot(t, u , 'k--', label="Input")
    plt.plot(t, sim0_y, 'g--', label="Initial")
    plt.plot(t, simm_y, 'b--', label="Midway")
    plt.plot(t, simf_y, 'r--', label="Final")
    plt.legend(loc=1)
    plt.savefig( filename, bbox_inches='tight')


    return 0

def plot_aveFitness(J, filename):
    """

    """
    i = J[:,0]
    j = J[:,1]

    title = 'Average Fitness after Selection at Each Iteration'
    plot_init(title, [0, max(i)], [min(j)-0.5, max(j)+0.5], "Iteration", "Error ")
    plt.plot(i, j , 'b--', label="Objective Function")
    #plt.plot(t, sim0_y, 'g--', label="Initial")
    #plt.plot(t, simf_y, 'r--', label="Final")
    #plt.legend(loc=1)
    plt.savefig( filename, bbox_inches='tight')

    return 0

def plot_init(title, xlim, ylim, xlabel, ylabel):
    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111)
    ax.grid(True, linestyle = '-', color = '0.75')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def run_EV():
    """ runs the ev and plots fitness at each run ...
        optionally set debug = true and print the
        population after each mutation
    """


    # Control Parameters
    tmax = 60                                       # Fahrenheit
    tmin = 48                                       # Fahrenheit
    setpoint = 54                                   # Fahrenheit
    #ctrl_temps = [tmax, setpoint, setpoint+10, tmin, tmax, setpoint]
    ctrl_temps = [tmax, setpoint, tmin]
    ctrl_params = [10, 150, 45, 350, 60, 2]         # valve_lower, valve_upper, valve_center, windup, sample time, dcmultiplier
    ctrl_interval = 20                              # mins

    # Evolutionary Parameters
    runtime = 10                                    # iterations
    debug = 1                                       # for printing to screen, turn off on server.

    # init ev operators class, which contains: parents, children, , just one until parallelized
    evops = evUtils.ev_operators()
    #evUtils.ev_fitness(evops.parents, ctrl_temps, ctrl_intergacl, ctrl_params, evops.fit_weights)

    # Fitness values per iterations
    Jparents =  10000 * np.ones((evops.pop_size+1, 1))
    Jchildren = 10000 * np.ones((evops.pop_size+1, 1))

    # Fitness average and max over iterations
    Jave = np.zeros((runtime, 1))
    Jmax = np.zeros((runtime, 1))

    # calc first fitness
    evfit = evUtils.ev_fitness(evops.parents, ctrl_temps, ctrl_interval, ctrl_params, evops.fit_weights, debug)
    if debug:
        print "Population 0: ", evops.parents
    Jparent = evfit.run_eval()
    Jave[0] = np.mean(Jparent)
    Jmax[0] = np.mean(Jparent)

    if debug:
        print "Population 0: ", evops.parents
        print "Fitness : ", Jparent
        print "Mean: ", Jave[0], " Max: ", Jmax[0]

    for i in range(runtime):

        # when i == 0, this is

        # Add immigrants to the parents
        if not i % 10:
            if debug: print "Should query graph DB"

        # mutation
        evops.children = evops.mutate(evops.parents)
        if debug: print "Offspring Population ", i, " : ", evops.children
        evfit = evUtils.ev_fitness(evops.children, ctrl_temps, ctrl_interval, ctrl_params, evops.fit_weights, debug)
        Jchildren = evfit.run_eval()
        if debug:
            print "Offspring Population", i, " : ", evops.children
            print "Fitness : ", Jchildren

        # increase generation count,
        i += 1

        # selection of new generation (parents)
        evops.parents = evops.selection_trunc(Jparent, Jchildren)

        # calculate fitness
        evfit = evUtils.ev_fitness(evops.parents, ctrl_temps, ctrl_interval, ctrl_params, evops.fit_weights, debug)
        Jparent = evfit.run_eval()
        Jave[i] = np.mean(Jparent)
        Jmax[i] = np.mean(Jparent)

        if debug:
            print "Parent Population", i, " : ", evops.parents
            print "Fitness : ", Jparent
            print "Mean: ", Jave[i], " Max: ", Jmax[i]



    #plot_simout(simout_init, simout_mid, simout, "../sim_results/simout_30s5r6m.png")
    #plot_aveFitness(Jave, "../sim_results/objective_30s5r6m.png")

    if debug:
        print 'Final pop: ', evops.parents
        print 'Final J: ', Jparent
        print 'Jave: ', Jave[-5:]
        print 'Jmax: ', Jmax[-5:]

if __name__ == "__main__":

    run_EV()
