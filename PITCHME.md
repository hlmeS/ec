## Evolutionary Approach to PID Tuning
##### Applied to a Mass-Spring-Damper system

---

## Model

The goal is to model a dynamical system and evolve the PID gains to control the system. 


---

## Mass Spring Damper System



`f(t) = m x''(t) + c mx'(t) + kx(t)`

`P(s) = F(s)/X(s) = 1 / (ms^2 + cs + k )`

![Figure 1](pidEV/figures/spring_damper_fbd.png)


+++

#### Beautiful Math Rendered Beautifully


+++

`f(t) = m x''(t) + c mx'(t) + kx(t)$$`

+++

`\begin{align}
\dot{x} & = \sigma(y-x) \\
\dot{y} & = \rho x - y - xz \\
\dot{z} & = -\beta z + xy
\end{align}`

+++
