
- Previous Code
    - Softmax both on y and x
    - Assumed $\frac{1}{Z}k(x,y)$
- Current Code
    - **[Exp.1]** Softmax only on the postive examples.
        - Effect : Learning Center?
    - **[Exp.2]** Assume $k(x,y)$
        - Not effective.

|Exp.1 ($\lambda_q=0.5$)|Exp.2 ($\lambda_q=1.0$)|Exp.1+2 ($\lambda_q=0.5$)|
|:-:|:-:|:-:|
|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260717_1846-ddqhiqj0-init_vp_vq_ratio_0.5-reg_lambda_0.05-n_flowblck_4-n_attnblck_4-n_attnheads_8-attnhead_dim_64-lr_3e-05-batch_sz_2048.gif)|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260717_1931-i0zja6bp-init_vp_vq_ratio_1.0-reg_lambda_0.05-n_flowblck_4-n_attnblck_4-n_attnheads_8-attnhead_dim_64-lr_3e-05-batch_sz_2048.gif)|![](../results/plot/generate_gif/drifting_tarflow_two_dimensional_learnable_ratio_reg/wandb-260718_0437-juzawdfq-init_vp_vq_ratio_0.5-reg_lambda_0.05-n_flowblck_4-n_attnblck_4-n_attnheads_8-attnhead_dim_64-lr_3e-05-batch_sz_2048.gif)|


- code
    ```python
        def _calculate_positive_force(self, fixed_gen_samples, pos_samples, eps, temperature_list):
            B, C_gen, S = fixed_gen_samples.shape
            device = fixed_gen_samples.device                       
            dist = cdist(fixed_gen_samples, pos_samples, eps=eps)    # (B, C_gen, C_pos)        
                    
            scale = dist.mean()
            scale_inputs = torch.clip(scale / math.sqrt(S), min=eps)
            scale_dist = torch.clip(scale, min=eps)
            
            fixed_gen_samples_scaled = fixed_gen_samples / scale_inputs
            pos_samples_scaled = pos_samples / scale_inputs
            dist_normalized = dist / scale_dist        
            
            accumulated_force = torch.zeros_like(fixed_gen_samples_scaled, device=device)
            for curr_tau in temperature_list:
                logits = -dist_normalized / curr_tau    # (B, C_gen, C_gen+C_neg+C_pos)            
                softmax_y = torch.softmax(logits, dim=-1)        # Softmax over y (tgt_samples). (B, C_gen, C_pos)
                # softmax_x = torch.softmax(logits, dim=-2)        # Softmax over x (gen_samples). (B, C_gen, C_pos)
                # kernel_pos = torch.sqrt(torch.clip(softmax_y * softmax_x, min=eps))  # harm.avg. (B, C_gen, C_pos)
                
                # Exp. 3) Exp.1 + Exp.2
                kernel_pos = softmax_y
                    
                # [Derivation]
                # V_p = \sum_{y+}[ k(x,y+) * (y+ - x) ] / Z_pos
                #     = \sum_{y+}[ k(x,y+)/Z_pos * y+]  - x     (Because 1 = \sum_{y+}[(k(x,y+)/Z_pos)])
                #     = kernel_pos * pos_samples_scaled - fixed_gen_samples_scaled
                
                # Exp. 1) Excluding softmax_x
                # kernel_pos_y = torch.einsum("bcC,bCs->bcs", kernel_pos, pos_samples_scaled)     # (B, C_gen, S)
                # kernel_pos_y = torch.einsum("bcC,bCs->bcs", softmax_y, pos_samples_scaled)     # (B, C_gen, S)
                
                # Exp. 2) Multiply by sum_coeffs
                # curr_force = kernel_pos_y - fixed_gen_samples_scaled                            # (B, C_gen, S)
                Z_pos = torch.sum(kernel_pos, dim=-1, keepdim=True)                             # (B, C_gen, 1)
                coeff_pos = kernel_pos * Z_pos                                                  # (B, C_gen, C_pos)
                sum_coeffs = torch.sum(kernel_pos, dim=-1, keepdim=True)                        # (B, C_gen, 1)
                curr_force = torch.einsum("bcC,bCs->bcs", coeff_pos, pos_samples_scaled)        # (B, C_gen, S)
                curr_force = curr_force - sum_coeffs * fixed_gen_samples_scaled                 # (B, C_gen, S)
                
                scale_curr_force = torch.sqrt(torch.clip(torch.mean(curr_force ** 2), min=eps))
                curr_force = curr_force / scale_curr_force
                accumulated_force = accumulated_force + curr_force
                    
            return accumulated_force, scale_inputs    # (B, C_gen, S), scalar
    ```