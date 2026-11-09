## Task description

Based on the results from Task 4.1.2.0, actuator strength changes affect the covariance of FB Bellman residuals, while their mean remains nearly unchanged. This suggests that the **residual covariance may capture action-dependent transition sensitivity**.

In this experiment, we train an **action-conditioned dynamics model** on baseline environment transitions to investigate whether explicit transition prediction can detect dynamics shifts more effectively than FB Bellman residuals.

$e^{Dyn}_t = (s_{t+1}-s_t)-f_{\theta}(s_t, a_t)$ 

$\widehat{\Delta s}_t=f_\theta(s_t,a_t)$ where $\Delta s_t=s_{t+1}-s_t$

The central research question is:

> Can an action-conditioned dynamics model detect environment dynamics shifts that are difficult to identify using FB Bellman residuals?
> 

result would be just like 4.1.2.0 with a table that describe the model prediction and threshold

## Sub-tasks

- [ ]  Train an action-conditioned dynamics model
    1. Train an MLP to predict state displacement using baseline transition
    2. Evaluate prediction errors on testing set
- [ ]  Evaluate residual distribution shifts
    1. Reuse data collected from 4.1.2.0
    2. Find the threshold that best describe dynamic shift happened
