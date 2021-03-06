#! /usr/bin/env python2

"""

Title: evUtils.py
Author: Holm Smidt
Version: 1.0
Date: 11-04-2017

Overview:
Class wrapper for functions that allow help
with the evolutionary algorithms


"""

import sqlCon
#import ctrlUtils

import numpy as np
import random
import math
import datetime, pause
import requests, urllib

class ev_fitness:

    def __init__(self, population, ctrl_temps, ctrl_interval, ctrl_params, fit_weights, debug):
        """
        Some persistne values

        - Population is m x d, where m is number of genes, and d is the # of genes (e.g. Kp, Ki, Kd, center)
        - ctrl_temps is the sequence of temepratures to test ... initially, it's [max, setpoint, setpoint+10, min, max, setpoint]
        - weights for fitness eval : w0 * Energy + w1 * ISE + w2 * ITAE
        - ctrl_params : 'tempset': tempset, 'Kp': controls[0], 'Ki': controls[1], 'Kd': controls[2], 'valve_lower_limit': controls[3],
            'valve_upper_limit': controls[4], 'valve_center': controls[5], 'windup': controls[6], 'sampletime': controls[7], 'dcmultiplier': controls[8]
        - k is the number of cooling cycles sum(x < 0 for x in np.diff(ctrl_temps))
        """

        # population is kp, ki, kd
        self.population = population
        self.ctrl_temps = ctrl_temps
        self.ctrl_interval = ctrl_interval
        self.t_max = max(ctrl_temps)
        self.t_setpoint = ctrl_temps[1]
        self.t_min = min(ctrl_temps)

        # placeholder for fitness values
        self.k = sum(x < 0 for x in np.diff(ctrl_temps))
        self.m = len(population)
        self.J = 10000*np.ones((self.m,self.k))
        self.J_ave = 10000*np.ones((self.m, 1))
        #J = np.concatenate((pop[:,0].reshape(psize,1), np.zeros((psize,1))), axis = 1)
        self.j_weights = fit_weights

        # ctrl_params are: valve_lower, valve_upper, valve_center, windup, sampletime, dcmultiplier
        self.ctrl_params = ctrl_params

        self.con = sqlCon.sqlCon()

        self.debug = debug
        self.debug2 = 0

    def execRequest(self, tempset, controls):
        #params = urllib.urlencode({'tempset': controls[0], 'Kp': 1.3, 'Ki': 0.05, 'Kd': 0.4, 'valve_lower_limit': 10, 'valve_upper_limit': 150, 'valve_center': 40, 'windup': 450, 'sampletime':45, 'dcmultiplier':2})
        params = urllib.urlencode({'tempset': tempset, 'Kp': controls[0], 'Ki': controls[1], 'Kd': controls[2], 'valve_lower_limit': controls[3],
            'valve_upper_limit': controls[4], 'valve_center': controls[5], 'windup': controls[6], 'sampletime': controls[7], 'dcmultiplier': controls[8] })
        response = requests.get("http://deep.outtter.space:51880/ilaniwai/PID2?%s" % params, auth=('user', 'Mat5uda$$'))
        return response

    def ise_calc(self, simout):
        """
        Calculates the integral squared error of the timeseries
        output y with respect to the desired input u and delta t.

        ISE = sum ((y - u)^2 * delta t )
        """
        u = simout[:, 0]
        y = simout[:, 1]
        t = simout[:, 2]
        ise = 0
        for i in range(1, len(t)-1):
            ise += (y[i] - u[i])**2 * (t[i]-t[i-1])

        return ise

    def iae_calc(self, simout):
        """
        Calculates the integral absolute error of the timeseries
        output y with respect to the desired input u and delta t.

        u -> simout[:, 0]
        y -> simout[:, 1]
        t -> simput[:, 2]

        IAE = sum (abs(y - r) * delta t )
        """

        u = simout[:, 0]
        y = simout[:, 1]
        t = simout[:, 2]
        iae = 0
        for i in range(1, len(t)-1):
            iae += abs(y[i] - u[i]) * (t[i]-t[i-1])

        return iae

    def itae_calc(self, simout):
        """
        Calculates the integrated time absolute error of the timeseries
        output y with respect to the desired input r and delta t.

        ITAE = sum (t * abs(y - u) * delta t )
        """
        u = simout[:, 1]
        y = simout[:, 2]
        t = simout[:, 0]

        itae = 0
        for i in range(1, len(t)-1):
            itae += t[i] * abs(y[i] - u[i]) * abs(t[i]-t[i-1])

        return itae

    def energy_calc(self, simout):
        """
        Calculate the sum of the energy consumed during the
        cooling interval.

        """
        return abs(np.sum(simout[:, 4]))

    def J_eval(self, i, k, simout):
        """
        J = w0 * Energy + (w1 * ISE + w2 * ITAE)
        simout columns: [time, tempset, tempactu
        al, power, energy]
        """
        self.J[i,k] = self.j_weights[0] * self.energy_calc(simout) + self.j_weights[1] * abs(self.iae_calc(simout) ) #+ self.j_weights[2] * 0.01* self.itae_calc(simout)

    def run_eval(self):
        """
        Run the evaluation process.

        For each gene, we evaluate fitness over the ctrl_temps array with ctrl_interval between them.

        """

        """ gene population loop """
        self.ctrl_temps[1] = np.random.normal(self.ctrl_temps[1], 1/math.sqrt(2.0/math.pi))

        for i in range(0, self.m):
            # put together control parameters , same across the different temperatures
            control_param = [self.population[i, 0], self.population[i, 1], self.population[i,2]] + self.ctrl_params

            """ temp setpoint loop """
            k = 0
            for j, temp in enumerate(self.ctrl_temps):

                if self.debug:
                    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    print "Iteration: " , i , " Setpoint: ", temp, " Interval: " , self.ctrl_interval, " Timestamp: " ,
                    print datetime.datetime.now(), " , ",  datetime.datetime.now() + datetime.timedelta(minutes=self.ctrl_interval)

		        # update the controls
                if j == 0 or np.diff(self.ctrl_temps)[j-1] >= 0:
                    resp = self.execRequest(temp, control_param)
                    if self.debug: print resp, "HEATING,  ", j
                    pause.until(datetime.datetime.now() + datetime.timedelta(minutes=(self.ctrl_interval-10)))

                else:
                    resp  = self.execRequest(temp, control_param)
                    if self.debug: print resp, "COOLING, Part 1, with ... ", self.population[i,:]
                    pause.until(datetime.datetime.now() + datetime.timedelta(minutes=0.6*self.ctrl_interval))
                    resp = self.execRequest(temp, control_param)

                    query = ("SELECT timestamp, tempSet, tempActual, (realP+dcP), (realE+dcE) FROM measurementHR "
                             "WHERE containerID=2 ORDER BY timestamp DESC LIMIT 30 ; " )

                    cnx = self.con.db_connect()
                    nptime, simout = self.con.db_query(cnx, query)

                    # break if we don't get to the temperature within half the time at least
                    #if not 0.95*temp < np.mean(simout[:, 2]) < 1.08*temp :
                    if sum( 0.98*temp > y for y in simout[:,2]) > 5 :
                        self.J_ave[i] = 10000
                        break

                    if self.debug: print resp, "COOLING, Part 2, with ... ", self.population[i,:]
                    pause.until(datetime.datetime.now() + datetime.timedelta(minutes=(0.4*self.ctrl_interval-0.5)))
                    if self.debug: print "PAU Waiting"


                #query = ("SELECT timestamp, tempSet, tempActual, (realP+dcP), (realE+dcE) FROM measurementHR "
                #         "WHERE deviceID=2 and timestamp >= (NOW() - INTERVAL " + str(self.ctrl_interval*4) + " MINUTE) " )
                query = ("SELECT timestamp, tempSet, tempActual, (realP+dcP), (realE+dcE) FROM measurementHR "
                         "WHERE containerID=2 ORDER BY timestamp DESC LIMIT " + str(self.ctrl_interval*4) + " ; " )

                cnx = self.con.db_connect()
                nptime, simout = self.con.db_query(cnx, query)
                # if it's cooling ... cooling next iteration has higher value
                if j > 0 and np.diff(self.ctrl_temps)[j-1] < 0:
                    #cnx = self.con.db_connect()
                    #nptime, simout = self.con.db_query(cnx, query)
                    self.J_eval(i, k, simout)
                    if self.debug: print self.J[i, :]

                    if self.debug2:
                        print nptime, simout
                        print self.J

                    # put code here to break if gene shows nonconstructive behavior.

                    # too much error
                    #if abs(self.J[i,k]) > 7500:
                    #    self.J_ave[i] = 10000
                    #    break

                    # not getting close enough to target
                    if sum( 0.95*temp < y < 1.05*temp for y in simout[:,2]) < 8 :
                        self.J_ave[i] = 10000
                        break

                    # too much overshoot (10%) (can have 3 outlier due to error reading)
                    elif sum( 0.98*temp > y for y in simout[:,2]) > 5 :
                        self.J[i,k] *= 1.3
                        #break

                    k += 1

                    if self.debug: print self.J

                else:
                    """
                    start_e = abs(temp - np.mean(simout[0:5, 2]))
                    end_e = abs(temp - np.mean(simout[-5:, 2]))
                    if start_e < end_e :
                        self.J_ave[i] = 20000
                        break
                    """
                    if sum( 1.15*temp < y for y in simout[:,2]) > 3 :
                        self.J_ave[i] = 10000
                        break



            """ get average fitness across all cooling cycles """
            self.J_ave[i] = np.mean(self.J[i, :])


        return self.J_ave


class ev_operators:

    #population, ctrl_temps, ctrl_interval, ctrl_params, fit_weights
    def __init__(self):

        #initial population
        self.pop_size = 6              # 5 parents
        self.gene_size = 3              # Kp, Ki, Kd
        self.rand_seed = 1023         # random number seed
        self.spread = 2                # spreading factor for intial guess
        self.max_iter = 20            # max iterations

        # variation operators
        #self.recomb_rate = 0.5
        self.mut_rate = 0.4
        self.mut_oper = "cauchy"
        self.mut_gauss_step = 4


        #0.5 < kp < 20 , 0.5 < ki < 10 , 0 < kd < 10 ?
        self.gain_limits = np.array([[0.5, 40], [0.0, 5], [0.5, 400.0]])
        #self.alpha = 1.0self.recomb_rate

        self.fit_weights = [0.3, 0.7] #, 0.1]    # energy, ise, itae

        self.parents = self.pop_rand_init()
        self.children = np.zeros((np.shape(self.parents)))

    def pop_rand_init(self):
        """
        Generates a population as an array
        with gene_size as the columns (e.g. gene_size = 3 for Kp, Ki, Kd )
        and population_size as the number of genes in the population

        The 'seed' let's us control the seed for random number
        generation.

        The 'spread' let's us scale our random numbers by the
        'spread' factor.
        """

        #np.random.seed(self.rand_seed)
        #idx = np.arange(size).reshape(size, 1)
        #pop = self.spread * np.random.standard_cauchy((self.pop_size, self.gene_size))
        Ku = 15 * (1 + np.random.random(self.pop_size) - 0.5)
        pop = np.ones((self.pop_size, self.gene_size))
        pop[:, 0] = 0.2* Ku
        pop[:, 1] = 0.4*Ku/300
        pop[:, 2] = 0.2*Ku*300/3
        for i in range(self.pop_size):
            for j in range(self.gene_size):
                if pop[i,j] > self.gain_limits[j, 1]:
                    pop[i,j] = self.gain_limits[j, 1]
                elif pop[i,j] < self.gain_limits[j, 0]:
                    pop[i,j] =self.gain_limits[j, 0]

        return pop

    def pop_gdb_init(self, env_params):
        """
        Find similar environmental parameters in the graph and
        query respective genes.
        """

    # not using recombination here.
    def recombine(self):
        """
        Recombination of parent population to create
        same number of offsprings as there are parents.

        psize = len(pop[:,0])
        idx = np.arange(psize).reshape(psize, 1)
        offspring = np.concatenate((idx, np.zeros((psize, 3))), axis = 1)

        for i in range(psize):
            if float(np.random.random(1) < recomb_rate):
                offspring[i] = pop[i]
            else :
                parent1 = pop[i]
                #print 'parent1: ', parent1
                parent2 = pop[np.random.randint(0, psize)]
                #print 'parent2: ', parent2
                XO_pt = np.random.randint(1, 3)
                for j in range(1, 4):
                    #print 'xo-pt / j: ', XO_pt, j
                    offspring[i,j] = parent1[j] if j<= XO_pt else parent2[j]
                #print 'offspring: ', offspring[i]

        return offspring
        """

    def mutate(self, pop):
        """
        Given a population, mutate each genotype stochastically using a
            * "gaussian" operator
            * "cauchy" operator
        """

        for i in range(len(pop[:,0])):
            for j in range(self.gene_size):
                if float(np.random.random(1)) >= self.mut_rate:
                    if self.mut_oper == "gaussian" :
                        dist =np.random.normal(0, step/math.sqrt(2.0/math.pi))
                        if j == 1: pop[i,j] +=  dist / 10
                        else: pop[i,j] += dist
                    elif self.mut_oper == "cauchy" :
                        dist = 5*float(np.random.standard_cauchy(1))
                        if j == 1: pop[i,j] +=  dist / 10
                        else: pop[i,j] += dist

                if pop[i,j] > self.gain_limits[j, 1]:
                    pop[i,j] = self.gain_limits[j, 1]
                elif pop[i,j] < self.gain_limits[j, 0]:
                    pop[i,j] =self.gain_limits[j, 0]
        return pop

    def selection_trunc(self, Jparent, Jchildren):
        """

        select the fittest population from the parent and offspring
        """

        # selected genes, keeping population size the same
        selection = np.zeros((self.pop_size, self.gene_size))

        # all genes, all fitnesses
        totalPop = np.vstack((self.parents, self.children))
        totalJ = np.vstack((Jparent, Jchildren))

        # combine fitness and genes
        genes_J = np.concatenate((totalJ, totalPop), axis =1 )
        #print "before ordering: ", genomes
        genes_J = genes_J[genes_J[:,0].argsort()] # sort by opjective function
        #print "after ordering: ", genes_J
        #cumJ = np.sum(genomes[:,1])
        #print np.shape(genomes)
        #print

        #truncation
        Jout = np.ones((np.shape(Jparent)))
        for i in range(self.pop_size):
            selection[i][:] = genes_J[i][1:1+self.gene_size]
            Jout[i, 0] = genes_J[i, 0]

        #proportional
        '''
        for i in range(2*psize):
            sampleP[i] = genomes[i,1] / cumJ
            CDF[i] = sampleP[i] if i == 0 else (CDF[i-1]+sampleP[i])

        for i in range(psize):
            for k in range(2*psize):
                if float(np.random.random(1)) > CDF[k]:
                    selection[i][1:4] = genomes[k][2:5]
                    break

        '''
        return selection, Jout

    def selection_sus(self, parent, Jparent, children, Jchildren, iter):
        """
        Stochastic Universal samplingTime
        """

    def queryGraph(self, num):

        gdb_result = np.array([ [10.57, 0.0423, 39.65],
                            [7.5, 0.03, 85],
                            [9.41, 0.01, 15.5],
                            [5.13, 0.021, 105],
                            [14, 0.045, 245],
                            [3, 0.02, 300],
                            [4, 0, 50.1]
                         ])

        for i in range(1, num+1):
            self.parents[-i,:] = gdb_result[np.random.randint(len(gdb_result)), :]
