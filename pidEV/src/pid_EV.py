#! /usr/bin/env python2

"""

Title: EV algorithm for PID tuning
Author: Holm Smidt
Version: 0.9
Date: 09-18-2017

Overview:




"""
import numpy as np
import random
import math
import matlab.engine
import matplotlib.pyplot as plt

def matlab_init(n):
    """ initialize n matlab engines
    """
    engines = {}
    for key in range(n):
        engines[key] = matlab.engine.start_matlab()

    return engines

def matlab_sim(eng, pop, u, t):
    """
    Run the simulation in matlab using the controller
    constants in each gene and the given 'input' series [r,t]
    and return simulation output as [y, t].

    Then return input an output for each simulations in array [r,y,t] for each
    index.
    """

    #convert float to matlab double
    pop_mat = matlab.double(pop.tolist())
    u_mat = matlab.double(u.tolist())
    t_mat = matlab.double(t.tolist())

    #placeholder for simulation output: idx,
    simout = np.zeros((len(pop[:,0]), len(t), 3))

    for i in range(len(pop[:,0])):
        #i = int(idx)
        output = eng.pid_step(pop_mat[i][1], pop_mat[i][2], pop_mat[i][3], u_mat, t_mat)
        #output = eng.pid_step(Kp, Ki, Kd, u, t)

        #print "idx: ", i
        #print output
        u = np.asarray(u).reshape(len(t), 1)
        t = np.asarray(t).reshape(len(t), 1)
        out = np.asarray(output)[:,0].reshape(len(t), 1)
        simout[i] = np.concatenate((out, u, t), axis = 1)

    #print simout

    return simout

def matlab_gensig(type, time, step):
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


def population_init(size, seed, spread):
    """
    Generates a population as an array
    with Kp, Ki, Kd as the columns and
    'size' number of gene pairs.

    The 'seed' let's us control the seed for random number
    generation.

    The 'spread' let's us scale our random numbers by the
    'spread' factor.
    """

    np.random.seed(seed)
    idx = np.arange(size).reshape(size, 1)
    genes = spread*np.random.random((size, 3))
    genes = np.concatenate((idx, genes), axis = 1)
    return genes

def ise_calc(simout):
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

def iae_calc(simout):
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

def itae_calc(simout):
    """
    Calculates the integrated time absolute error of the timeseries
    output y with respect to the desired input r and delta t.

    ITAE = sum (t * abs(y - u) * delta t )
    """
    u = simout[:, 0]
    y = simout[:, 1]
    t = simout[:, 2]

    itae = 0
    for i in range(1, len(t)-1):
        itae += t[i] * abs(y[i] - u[i]) * (t[i]-t[i-1])

    return itae

def fitness(pop, simout, weights):
    """
    calculate the fitness for each gene in the population
    based on the objective functions:

    J = w1 * ISE + w2 * IAE + w3 * itae,
    where sum of w_i = 1

    Returns the fitness of each gene in nx2 matrix [idx, J]
    """
    psize = len(pop[:,0])
    J = np.concatenate((pop[:,0].reshape(psize,1), np.zeros((psize,1))), axis = 1)
    for i in range(len(pop[:,0])):
        J[i, 1] = weights[0] * ise_calc(simout[i, :, :]) + weights[1] * iae_calc(simout[i, :, :]) + weights[2] * itae_calc(simout[i, :, :])

    return J


def orgIze(genotype):
    """ given a genotype population, return the
        fitness of the popoulation and the population itself
    """
    return np.column_stack(genotype, fitness(genotype))

def recombine(pop, recomb_rate):
    """
    Recombination of parent population to create
    same number of offsprings as there are parents.
    """
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

def mutate(pop, mut_rate, oper, step, min_k, max_k):
    """
    Given a population, mutate each genotype stochastically using a
    "gaussian" operator
    "cauchy" operator  --- not yet
    """

    for i in range(len(pop[:,0])):
        for j in range(1,4):
            if float(np.random.random(1)) >= mut_rate:
                if oper == "gaussian" :
                    pop[i,j] += np.random.normal(0, step/math.sqrt(2.0/math.pi))
                elif oper == "cauchy" :
                    pop[i,j] += float(np.random.standard_cauchy(1))

            if pop[i,j] > max_k:
                pop[i,j] = max_k
            elif pop[i,j] < min_k:
                pop[i,j] = min_k

    return pop


def select(parent, Jparent, children, Jchildren, iter):
    """

    given a genotype, create a mutation and return
        the fittest ones
    """

    psize = len(parent[:,0])
    idx = np.arange(psize).reshape(psize, 1)
    selection = np.concatenate((idx, np.zeros((psize, 3))), axis = 1)
    #selection = np.empty([psize, 4])

    CDF = np.empty([2*psize,1])
    sampleP = np.empty([2*psize,1])

    # all genomes, delete index
    totalPop = np.delete(np.vstack((parent, children)), 0, axis=1)

    # all objective function
    totalJ = np.vstack((Jparent, Jchildren))
    genomes = np.concatenate((totalJ, totalPop), axis =1 )
    #print "before ordering: ", genomes
    genomes = genomes[genomes[:,1].argsort()] # sort by opjective function
    #print "after ordering: ", genomes
    cumJ = np.sum(genomes[:,1])
    #print np.shape(genomes)
    #print

    #truncation

    for i in range(psize):
        selection[i][1:4] = genomes[i][2:5]

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
    return selection

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

    #initial population
    genes = 20           # 5 parents
    seed = np.random.randint(1,1000)         # random number seed
    factor = 10       # initial guess spread
    max_iter = 20      # max iterations

    #time
    sigType = 4         # 1 - unit step, 2 - step with A=15, 3- square wave, 4 - square wave
    simT = 12           # simulation time
    simDT = 0.01        # simulation time step
    min_k = 0           # control gain lower limit
    max_k = 400         # control gain upper limit

    # weights
    weights = [0.4, 0.3, 0.3]    # ise, iae, itae

    # variation operators
    recomb_rate = 0.5
    mut_rate = 0.3
    mut_oper = "gaussian"
    mut_step = 30
    alpha = 0.98

    # init matlab connection, just one until parallelized
    eng = matlab_init(1)
    #print eng[0]

    # init population, 5 genes, seed = 20, factor 10
    pop = population_init(genes, seed, factor)


    # let's simulate our controller-plant system
    # using step input, t:=5s, dt:=0.1
    (u,t) = matlab_gensig(sigType, simT, simDT)
    simout = matlab_sim(eng[0], pop, u, t)
    simout_init = simout
    #calc fitness
    Jp = fitness(pop, simout, weights)

    print 'Initial: ', pop
    print 'Initial J :', Jp

    #average fitness
    iter = 0
    Jave = np.empty((max_iter+1, 2))
    Jave[:,0] = np.arange(max_iter+1)
    Jave[0, 1] = np.mean(Jp[:,1])

    while (iter < max_iter):
        mut_step = alpha * mut_step

        iter += 1
        print "-----------------------------------------"
        print "ITERATION: ", iter

        # create new gerenation
        offspring = recombine(pop, recomb_rate)
        offspring = mutate(offspring, mut_rate, mut_oper, mut_step, min_k, max_k)
        #print 'offspring: ', offspring

        simout = matlab_sim(eng[0], offspring, u, t)
        Jc = fitness(pop, simout, weights)
        #print 'Jc: ', Jc

        pop_new = select(pop, Jp, offspring, Jc, iter)

        simout = matlab_sim(eng[0], pop_new, u, t)
        Jp = fitness(pop_new, simout, weights)
        #print 'Selected: ', pop
        #print 'Selected J :', Jp
        Jave[iter, 1] = np.mean(Jp[:,1])
        print np.mean(Jp[:,1])
        #print Jp[:,1]
        pop = pop_new
        if iter == 0.5*max_iter:
            simout_mid = simout

    plot_simout(simout_init, simout_mid, simout, "results/simout.png")
    plot_aveFitness(Jave, "results/objective.png")
    print 'Fintal pop: ', pop
    print 'Final J: ', Jp
    print 'Jave: ', Jave
    #print 'simout: ' , simout
    print 'simout: ' , simout[0][:][:]

if __name__ == "__main__":

    run_EV()
