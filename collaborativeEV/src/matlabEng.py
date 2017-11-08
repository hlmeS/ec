#! /usr/bin/env python2

"""

Title: matlabEng.py
Author: Holm Smidt
Version: 1.0
Date: 11-04-2017

Overview:
Class wrapper for functions that allow creation
of matlab engines for simulation.



"""

import numpy as np
import random
import math
import matlab.engine

class matlabEng:
    """MATLAB simulation engine
    """

    def __init__(self):

        self.engine = matlab.engine.start_matlab()


    def matlab_pid_sim(self, control_params, u, t):
        """
        Run the simulation in matlab using the controller
        constants in each gene and the given 'input' series [r,t]
        and return simulation output as [y, t].

        Then return input an output for each simulations in array [r,y,t] for each
        index.
        """

        #convert float to matlab double
        kp_mat = matlab.double(control_params[:, 0].tolist())
        ki_mat = matlab.double(control_params[:, 0].tolist())
        kd_mat = matlab.double(control_params[:, 0].tolist())
        u_mat = matlab.double(u.tolist())
        t_mat = matlab.double(t.tolist())

        #placeholder for simulation output: idx,
        simout = np.zeros((len(control_params[:,0]), len(t), 3))

        for i in range(len(control_params[:,0])):
            output = self.eng.pid_step(kp_mat[i], ki_mat[i], kd_mat[i], u_mat, t_mat)
            #output = eng.pid_step(Kp, Ki, Kd, u, t)

            #summarize outputs in single matrix
            u = np.asarray(u).reshape(len(t), 1)
            t = np.asarray(t).reshape(len(t), 1)
            out = np.asarray(output)[:,0].reshape(len(t), 1)
            simout[i] = np.concatenate((out, u, t), axis = 1)

        #print simout

        return simout

    def matlab_gensig(self, type, time, step):
        """
        Generates lsim signal data (time t and values u )

        Types:
        1 - unit step input (0 at t=0, 1 at t>0, A = 1)
        2 - step with amplitude 20 ((0 at t=0, 1 at t>0, A = 100))
        3 - square wave
        4 - square wave, twice the frequency
        """
        t  = np.arange(0, time+step, step)
        #t = t.reshape(len(t), 1)

        u = np.zeros(len(t))
        if type == 1 :
            for i, ts in enumerate(t):
                u[i] = 1
        elif type == 2:
            for i, ts in enumerate(t):
                u[i] = 15
        elif type == 3:
            for i, ts in enumerate(t):
                u[i] = 1 if ts < (0.5 * time) else 0
        elif type == 4:
            for i,ts in enumerate(t):
                if ts < 0.2*time:
                    u[i] = 5.0
                elif 0.2*time <= ts < 0.5*time:
                    u[i] = 10.0
                elif  0.5*time <= ts < 0.8*time:
                    u[i] = 3.0
                else :
                    u[i] = 7.0
                #u[i] = 1 if ((ts < 0.25*time) or (0.5*time <= ts < 0.75*time)) else 0


        return (u, t)
        #return (np.concatenate((u,t), axis=1))
