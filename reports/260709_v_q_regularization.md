# Add $V_q$ Regularization Loss
### Objective
- In both fixed and [learnable ratio case](./260708_learnable_ratio.md), the explosion of $V_q$ was observed.
- What if we add regularization loss on the $V_q$
  - Suggested Loss
    - $\mathcal{L} = \underbrace{ \left\Vert \mathbf{x}_t - \text{stop\_grad}\left(\mathbf{x}_t + \frac{\lambda_p}{\lambda_p+1} V_p^+(\mathbf{x}_t) - \frac{1}{\lambda_p+1} V_q^-(\mathbf{x}_t)\right) \right\Vert}_{\text{Drift Loss}} + \lambda_{\text{reg}} \underbrace{\Vert V_q(\mathbf{x}_t) \Vert^2}_{\text{Reg. Loss}}$

<br><br>

## Fixed $\lambda_p$ ratio
||$\lambda_p=0.3$||-|
|:-:|:-:|:-:|:-:|
|$\lambda_{\text{reg}}=0.05$|<img src="../results/plot/generate_gif/drifting_tarflow_two_dimensional_ratio_reg_loss/wandb-260708_2208-5dqexctj-vp_vq_ratio_0.3-reg_lambda_0.05-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_1024.gif" width="300px">| | -|
|$\lambda_{\text{reg}}=0.01$ <br> (epoch 10K~20K)|<img src="../results/plot/generate_gif/drifting_tarflow_two_dimensional_ratio_reg_loss/wandb-260708_2208-jc0iieav-vp_vq_ratio_0.3-reg_lambda_0.01-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_1024.gif" width="300px">||-|



<br><br>

## Learnable $\lambda_p$ with $V_q$ regularization
### $\text{lr}=1e-4$
|$\lambda_{\text{reg}}$ \ $\lambda_{\text{q}}$|$\lambda_p=1.5$|$\lambda_p=1.0$|$\lambda_p=0.5$|
|:-:|:-:|:-:|:-:|
|$\lambda_{\text{reg}}=0.01$|![](../)|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260713_0510-hwt85aug-init_vp_vq_ratio_1.0-reg_lambda_0.01-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_2048.gif)|![](../)|
|$\lambda_{\text{reg}}=0.05$|![](../)|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260713_0512-6nunoevj-init_vp_vq_ratio_1.0-reg_lambda_0.05-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_2048.gif)|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260713_0512-6gd17ves-init_vp_vq_ratio_0.5-reg_lambda_0.05-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_2048.gif)|
|$\lambda_{\text{reg}}=0.1$|![](../)|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260713_0606-27zoknft-init_vp_vq_ratio_1.0-reg_lambda_0.1-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_2048.gif)|![](../)|
|$\lambda_{\text{reg}}=0.5$|![](../)|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260713_1252-29b46fyw-init_vp_vq_ratio_1.0-reg_lambda_0.5-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_2048.gif)|![](../)|
|$\lambda_{\text{reg}}=1.0$|![](../)|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260713_1253-ev8zjxej-init_vp_vq_ratio_1.0-reg_lambda_1.0-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_2048.gif)|![](../)|

<br>

### $\text{lr}=5e-5$
|$\lambda_{\text{reg}}$ \ $\lambda_{\text{q}}$|$\lambda_p=1.5$|$\lambda_p=1.0$|$\lambda_p=0.5$|
|:-:|:-:|:-:|:-:|
|$\lambda_{\text{reg}}=0.01$|![](../)|![](../)|![](../)|
|$\lambda_{\text{reg}}=0.05$|![](../)|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260713_1926-mshbq7f9-init_vp_vq_ratio_1.0-reg_lambda_0.05-lr_5e-05-batch_sz_2048.gif)|![](../)|
|$\lambda_{\text{reg}}=0.1$|![](../)|![](../)|![](../)|
|$\lambda_{\text{reg}}=0.5$|![](../)|![](../)|![](../)|

<br>

### $\text{lr}=3e-5$
|$\lambda_{\text{reg}}$ \ $\lambda_{\text{q}}$|$\lambda_p=1.5$|$\lambda_p=1.0$|$\lambda_p=0.5$|$\lambda_p=0.1$|
|:-:|:-:|:-:|:-:|:-:|
|$\lambda_{\text{reg}}=0.01$|![](../)|![](../)|![](../)|![](../)|
|$\lambda_{\text{reg}}=0.05$|![](../)|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260713_1655-msu3wh7k-init_vp_vq_ratio_1.0-reg_lambda_0.05-lr_3e-05-batch_sz_2048.gif)|![](../)|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260713_2139-zpo3px6d-init_vp_vq_ratio_0.1-reg_lambda_0.05-lr_3e-05-batch_sz_2048.gif)|
|$\lambda_{\text{reg}}=0.1$|![](../)|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260713_1254-plu7ag3m-init_vp_vq_ratio_1.0-reg_lambda_0.1-lr_3e-05-batch_sz_2048.gif)|![](../)|![](../)|
|$\lambda_{\text{reg}}=0.5$|![](../)|![](../)|![](../)|![](../)|

<br>

### $\text{lr}=1e-5$
|$\lambda_{\text{reg}}$ \ $\lambda_{\text{q}}$|$\lambda_p=1.5$|$\lambda_p=1.0$|$\lambda_p=0.5$|
|:-:|:-:|:-:|:-:|
|$\lambda_{\text{reg}}=0.01$|![](../)|![](../)|![](../)|
|$\lambda_{\text{reg}}=0.05$|![](../)|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260713_1947-s7rx7ppt-init_vp_vq_ratio_1.0-reg_lambda_0.05-lr_1e-05-batch_sz_2048.gif)|![](../)|
|$\lambda_{\text{reg}}=0.1$|![](../)|![](../)|![](../)|
|$\lambda_{\text{reg}}=0.5$|![](../)|![](../)|![](../)|