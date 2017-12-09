## Network Coordinated Evolution
#### Modeling and Control of Distributed System through Online Genetic Search Optimization

---

## Motivation

Can we modernize the grid without reinventing the wheel?

How can we take advantage of existing technologies and known computational methods to increase efficiency and renewable integration?


---

## Use Case

Using a LEZETi Hybrid Solar PV Mini Split Air Conditioner for Produce Cooling

+++

| ![Top](https://raw.githubusercontent.com/hlmes/ec/master/collaborativeEV/images/Kahuku_ContainerTop.png)  | ![Inside](https://raw.githubusercontent.com/hlmes/ec/master/collaborativeEV/images/Kahuku_Inside.png) |
|:---:|:---:|
| Container Top View | Indoor Ducted Units |

+++

| ![LEZETI](https://raw.githubusercontent.com/hlmes/ec/master/collaborativeEV/images/Kahuku_LEZETi.png)  | ![Inside](https://raw.githubusercontent.com/hlmes/ec/master/collaborativeEV/images/Kahuku_MIAO.png) |
|:---:|:---:|
| Outdoor Unit | Control Unit |

+++


| ![Schematic View](https://raw.githubusercontent.com/hlmes/ec/master/collaborativeEV/images/EVAPI_Lezeti_control_diagram_DWG_MT-1.png) |
|:---:|
|Schematic view of dynamic system |


---

## Controller Design

_Objective:_ Find control gains Kp, Ki, and Kd that best control the system.

_Desired properties:_ Stability, low error, low energy usage.

_Proposed Method:_ Genetic search algorithm.


+++

## Background

* PID controller is the most common controller in the industry
* Can be tuned by hand or computer software (if dynamics are known); difficult when system dynamics are unknown, when there are time delays, and when there are changes in the environment, etc.

+++

## Background

* Numerous works on PID controller tuning using EV: genetic search, particle swarm optimization, differential evolution, ...
* Most works simply consider simulations

+++

## Challenges

How to interface with the system?

How to overcome time constraints? <br> (One passive heating & active cooling cycle takes about 20-30 minutes.)

+++

## Approach

IoT system design to communicate with learning agents over the network.

Network multiple learning agents.

Store environmental and control parameters of each agent in a Graph Database.

---

## Genetic Search Algorithm

#### Design

+++

| ![LEZETI](https://raw.githubusercontent.com/hlmes/ec/master/collaborativeEV/images/CollaborativeEV_Process_V1.png)  | ![Inside](https://raw.githubusercontent.com/hlmes/ec/master/collaborativeEV/images/CollaborativeEV_FitnessEval_V1.png) |
|:---:|:---:|
| Main Loop | Fitness Evaluation |

+++

## Fitness Function

| ![Schematic View](https://raw.githubusercontent.com/hlmes/ec/master/collaborativeEV/images/fitnessJ.png) |
|:---:|
| ![Schematic View](https://raw.githubusercontent.com/hlmes/ec/master/collaborativeEV/images/iae.png) |
| Fitness Function J, Integrated Absolute Error IAE, and consumed Energy E |

+++

## Graph Database

| ![Schematic View](https://raw.githubusercontent.com/hlmes/ec/master/collaborativeEV/images/ev_graphdb_2.png) |
|:---:|
| Graph representation in neo4j |

---

## Initial Implementation

+++

## Parameters

| ![LEZETI](https://raw.githubusercontent.com/hlmes/ec/master/collaborativeEV/images/ga_param.png)  | ![Inside](https://raw.githubusercontent.com/hlmes/ec/master/collaborativeEV/images/ga_seq.png) |
|:---:|:---:|
| GA parameters | Setpoint sequence |

+++

## Initial Results

| ![LEZETI](https://raw.githubusercontent.com/hlmes/ec/master/collaborativeEV/images/GA_run1_gen1_annot_2.png) |
|:---:|
| Sample run analysis, promising observations did not lead to satisfactory convergence |


---

## Conclusions

+++

## So far ...

* Infrastructure for genetic search over the network and across agents is in place
* Promising components: constructive selection, graph query
* Initial tests run with gained insights but little success

+++

## What's next ...

- Add **virtual nodes** to the graph database and update them through simulations
- Add actual learning agents
- Revisit **fitness function, mutation, and selection**
