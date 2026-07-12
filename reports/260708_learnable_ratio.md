# Learnable $V_p/V_q$ Ratio Application
### Training Dynamics   
![](./images/260708/001_training_dynamics.png)
- Observations
    - Model reduces the $V_p/V_q$ ratio until some point.
        - This may mean, increasing the power of $V_q$ is necessary at the beginning of the training
        - Aligns with the fixed $V_p/V_q=0.3$ experiment that learned more complex circular shape
    - But at some point, model increases the $V_p/V_q$ ratio.
        - This relates with the training failure case where maintaining low $V_p/V_q$ ratio leads to $V_q$ explosion.

<br><br>

### 2D-Arrow Diagram
|Initital $V_p/V_q$|1.0|0.6|0.5|
|:-:|:-:|:-:|:-:|
|GIF|![](../results/plot/generate_gif/wandb-260707_2105-gutgqj4k-init_vp_vq_ratio_1.0-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_1024.gif)|![](../results/plot/generate_gif/wandb-260708_0214-830t9vac-init_vp_vq_ratio_0.6-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_1024.gif)|![](../results/plot/generate_gif/wandb-260708_0214-p3o7x8fp-init_vp_vq_ratio_0.5-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_1024.gif)|
|Final Shape|![](../results/train/drifting_tarflow_two_dimensional_learnable_ratio/spiral/wandb-260707_2105-gutgqj4k-init_vp_vq_ratio_1.0-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_1024/lr_schedule_type_ws/inference-epoch_10000.png)|![](../results/train/drifting_tarflow_two_dimensional_learnable_ratio/spiral/wandb-260708_0214-830t9vac-init_vp_vq_ratio_0.6-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_1024/lr_schedule_type_ws/inference-epoch_14400.png)|![](../results/train/drifting_tarflow_two_dimensional_learnable_ratio/spiral/wandb-260708_0214-p3o7x8fp-init_vp_vq_ratio_0.5-n_flowblck_4-n_attnblck_4-n_attnheads_4-attnhead_dim_4-batch_sz_1024/lr_schedule_type_ws/inference-epoch_10200.png)|