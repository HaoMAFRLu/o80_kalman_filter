# o80_kalman_filter

## Introduction
This is a Kalman Filter written for Pamy's vision system. Its primary functions include: 

* Using the Kalman Filter to estimate the states (position and velocity, we ignore the spin) of a ping-pong ball within the field of view. 
* Detecting impacts between the ping-pong ball and the surrounding environment (currently including the table and the ground) and predicting the states of the ball after the impact. 
* Predicting the future trajectory of the ping-pong ball based on the given states of the ball.

## The motion model of a ball
We adopt the following continuous motion model for the ball:
\[
\dot{v} = - \frac{1}{2m} C_{\text{d}} \rho A \left| v \right| v + \frac{1}{2m} C_{\text{m}} \rho A r \omega \times v + g,
\]
where
* $m$ denotes the mass of a ball, in kg,
* $r$ denote the radius of a ball, in m,
* $A$ denotes the corss sectional area, in m^2,
* $\rho$ denotes the air density, in kg/m^3,
* $C_{\text{d}}$ denotes the coefficient of air resistance,
* $C_{\text{m}}$ denotes the coefficient of Magnus force,
* $v \in \mathbb{R}^3$ and $\omega \in \mathbb{R}^{3}$ denote the velocity and spin of a ball, respectively,
* $g\in \mathbb{R}^3$ denotes the gravity along the negative $z$-axis,
* $|\cdot|$ denotes the $\ell_2$-norm.

and we calculate the coefficients $k_{\text{d}}$ and $k_{\text{m}}$ as follows:
$$
k_{\text{d}} = \frac{1}{2m} C_{\text{d}} \rho A,~k_{\text{m}} = \frac{1}{2m} C_{\text{m}} \rho A r.
$$

In calculation, we adopt the following discrete model,
$$
s_{k+1} = s_k + \Delta t 
\begin{pmatrix}
v_k\\
-k_{\text{d}} \left|v_k\right| v_k + k_{\text{m}} \omega_k \times v_k + g
\end{pmatrix},~k \in \mathbb{Z}_{+},
$$
where 
* the states $s_k \in \mathbb{R}^6$ includes the position $r \in \mathbb{R}^3$ and velocity $v$ of a ball, 
* $k$ denotes the number of step, 
* $\Delta t$ denotes a single time step, which should be small enough and variable.

It should be noted that currently we cannot measure the spin of the ball, therefore $\omega$ is always zero.

## Impact detection
Currently, we can only detect impacts within an extremely small time step $\Delta t$ with the table or the ground. In the future, we might introduce impacts with the racket. 

The impact detection logic is as follows: Given the state $s_k$ of the ball, and fixing the ball's $x$ and $y$ coordinates, we only move in the $z$ direction to detect the time required $\Delta t_{\text{req}}$ for a impact with the object right below the ball (table or ground). 

If the time required $\Delta t_{\text{req}}$ for a impact is greater than a single time step $\Delta t$, then no impact occurs in that simulation step; otherwise, we introduce the impact model and calculate the states before and after the impact separately. 

It is not difficult to see that we assume the $x$ and $y$ coordinates of the ball remain constant during the impact detection process, therefore this method is only effective for very small time steps.