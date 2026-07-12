# Add $V_q$ Regularization Loss
### Objective
- In both fixed and [learnable ratio case](./260708_learnable_ratio.md), the explosion of $V_q$ was observed.
- What if we add regularization loss on the $V_q$
  - Suggested Loss
    - $\mathcal{L} = \underbrace{\Vert \mathbf{x}_t - \text{stop\_grad}(\mathbf{x}_t + V_p^+(\mathbf{x}_t) - \lambda_q V_q^-(\mathbf{x}_t)) \Vert}_{\text{Drift Loss}} + \lambda_{\text{reg}} \underbrace{\Vert V_q(\mathbf{x}_t) \Vert^2}_{\text{Reg. Loss}}$

<br><br>

### 2D-Arrow Diagram
|Initital $V_p/V_q$|0.3|0.3|-|
|:-:|:-:|:-:|:-:|
|$\lambda_{\text{reg}}$|0.05|0.01 (epoch 10K~20K)| -|
|GIF|![](../results/plot/generate_gif/wandb-260708_2208-5dqexctj-vp_vq_ratio_0.3-reg_lambda_0.05-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_1024.gif)|![](../results/plot/generate_gif/wandb-260708_2208-jc0iieav-vp_vq_ratio_0.3-reg_lambda_0.01-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_1024.gif)|-|
