# PID-Tuning using Evolutionary Computation

## Abstract

## Introduction
### Motivation

PID (Proportional, Integral, Derivative) Controllers are a common feedback control method for controlling dynamic systems, given their well-known form, simplicity when compared to modern algorithms, and wide range of applications, i.e. about 90% of the industry still uses PID controllers [3]. Yet, despite its well-known form, the tuning of a PID controller may seem daunting when the behavior of the dynamic system is not known and difficult to estimate, and when external conditions vary from those experienced during the tuning process. A variety of PID-tuning approaches have been proposed and implemented to address these challenges, one being an evolutionary computation approach [3]. This project tests the use of a genetic search algorithm to find optimal PID control gains for a sample mechanical system.

## Problem Definition
In bridging the disciplines of evolutionary computation and control engineering, this project attempts the tuning of a PID-controller (i.e. it's proportional, integral, and derivative gains) for the textbook control system example: a mass-spring-damper system. More specifically, a generic search algorithm is applied in search for an optimal set of control gains using concepts of recombination, mutation, and selection.

## Dynamical System

### Model

The model system chosen is that of a classical mass-spring-damper, as illustrated in Figure 1. A mass of mass _m_ is attached to a wall with a spring with spring constant _m_ and a damper with damping coefficient _c_. A force _F_ is then applied to displace the block in its horizontal position. The dynamics of the system can simply be described in the time domain as `f(t) = m x''(t) + c mx'(t) + k(x) ` or by its transfer function `P(s) = F(s)/X(s) = 1 / (ms^2 + cs + k )`. The variables _m_, _c_, and _k_ were chosen as `1 kg`, `20 Ns/m`, and `10 N/m` respectively.

![Figure 1](figures/spring_damper_fbd.png)

**Figure 1:** Mass spring damper system.

### PID Controller

In this project, a PID-controller was considered for feedback control system with unary feedback signal shown in Figure 2. The plant model is the aforementioned mass-spring-damper system. The controller can be described as `u(t) = Kp * e(t) + Ki \int e(t)dt + Kp de(t)/dt` in the time domain or as `C(s) = Kp + 1/s  Ki + s Kd` in its trasnfer form [4,5].


![Figure 2](figures/feedback_block.png)

**Figure 2:** Feedback Control Loop.

### Simulation Engine

The above model and controller were modeled and simulated using the MATLAB script described in Listing 1. The script takes the three control gain, and system inputs _u_ and _t_ as its inputs, then models the plant, _G_, the controller, _C_, and the overall system, _T_, and ultimately simulates and returns the _T_'s response to the input (_u_, _t_), where _t_ is a time series and _u_ the reference point. 

**Listing 1:** MATLAB code for model simulation.
``` c
function resp = pid_step(Kp, Ki, Kd, u, t)

s = tf('s');
G = 1/(s^2 + 10*s + 20);

C = pid(Kp,Ki,Kd);
T = feedback(C*G,1);

[y, t] = lsim(T, u, t);
resp = [y, t];
```

## GA Design & Implementation

## GA Testing

## Results

## Conclusion and Future workspace

## Rerefences

[1.] L. Altenberg, "ICS674 - Evolutionary Computation". Class Presentation. 2017.

[2.] K. De Jong. "Evolutionary Computation: a unified approach". MIT Press. 2006

[3.] A. Jayachitra and R. Vinodha, “Genetic Algorithm Based PID Controller Tuning Approach for Continuous Stirred Tank Reactor,” Advances in Artificial Intelligence, vol. 2014, Article ID 791230, 8 pages, 2014. doi:10.1155/2014/791230

[4.] [Introduction: PID Controller Design, Control Tutorial](http://ctms.engin.umich.edu/CTMS/index.php?example=Introduction&section=ControlPID)

[5.] [Extras: Generating a Step Response in MATLAB, Control Tutorial](http://ctms.engin.umich.edu/CTMS/index.php?aux=Extras_step)
