# Add $V_q$ Regularization Loss
### Objective
- What if the problem is the sparsity of the score?
- Add noise to the scores

<br><br>


<br><br>

### Result
3. $\text{lr}=3e-5$
   |$\lambda_{\text{reg}}$ \ $\lambda_{\text{q}}$|$\lambda_p=1.5$|$\lambda_p=1.0$|$\lambda_p=0.5$|$\lambda_p=0.1$|
   |:-:|:-:|:-:|:-:|:-:|
   |$\lambda_{\text{reg}}=0.01$|![](../)|![](../)|![](../)|![](../)|
   |$\lambda_{\text{reg}}=0.05$|![](../)|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_noisy_learnable_ratio_reg/wandb-260714_0155-lqpb1py7-init_vp_vq_ratio_1.0-reg_lambda_0.05-lr_3e-05-batch_sz_2048.gif) ![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260713_1655-msu3wh7k-init_vp_vq_ratio_1.0-reg_lambda_0.05-lr_3e-05-batch_sz_2048.gif)|![](../)|![](../)|
   |$\lambda_{\text{reg}}=0.1$ |![](../)|![](../)|![](../)|![](../)|
   |$\lambda_{\text{reg}}=0.5$ |![](../)|![](../)|![](../)|![](../)|