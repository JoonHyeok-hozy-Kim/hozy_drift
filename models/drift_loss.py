from collections.abc import Iterable
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cdist(x, y, eps=1e-8):
    xy_dot = torch.einsum("bnd,bmd->bnm", x, y)
    x_norm = torch.einsum("bnd,bnd->bn", x, x)
    y_norm = torch.einsum("bmd,bmd->bm", y, y)
    dist_squared = x_norm.unsqueeze(-1) + y_norm.unsqueeze(1) - 2 * xy_dot
    dist_squared = torch.clip(dist_squared, min=eps)
    return torch.sqrt(dist_squared)


class DriftLossFunction(nn.Module):
    def __init__(self, temperature_list=(0.02, 0.05, 0.2)):
        super().__init__()
        assert isinstance(temperature_list, Iterable)
        self.temperature_list = temperature_list
    
    def forward(self, gen_samples, pos_samples, neg_samples=None):
        if neg_samples is None:
            neg_samples = torch.zeros_like(gen_samples[:, :0, :], device=gen_samples.device)   # (B, 1, S)
            neg_samples = neg_samples.detach()
        C_neg, C_pos = neg_samples.shape[1], pos_samples.shape[1]
        
        fixed_gen_samples = gen_samples.detach()                # Stop Grad           (B, C_gen, S)
        target_samples = torch.cat([gen_samples, neg_samples, pos_samples], dim=1)  # (B, C_gen+C_neg+C_pos, S)
        
        V, scale_inputs = self._calculate_force(fixed_gen_samples, target_samples, C_neg, C_pos)
        V, scale_inputs = V.detach(), scale_inputs.detach()     # Stop Grad
        
        gen_samples_scaled = gen_samples / scale_inputs         # Grad
        goal_scaled = fixed_gen_samples / scale_inputs + V      # Stop Grad
        
        diff = gen_samples_scaled - goal_scaled       # (B, C_gen, S)
        loss = torch.mean(diff ** 2, dim=(-1, -2))    # (B, )
        return loss        
        
    def _calculate_force(self, fixed_gen_samples, tgt_samples, C_neg, C_pos, eps=1e-6):
        B, C_gen, S = fixed_gen_samples.shape
        device = fixed_gen_samples.device
        dist = cdist(fixed_gen_samples, tgt_samples, eps=eps)    # (B, C_gen, C_gen+C_neg+C_pos)
        
        # Scaling for the unconditional case
        scale = dist.mean()
        scale_inputs = torch.clip(scale / math.sqrt(S), min=eps)
        scale_dist = torch.clip(scale, min=eps)
        
        fixed_gen_samples_scaled = fixed_gen_samples / scale_inputs
        tgt_samples_scaled = tgt_samples / scale_inputs
        dist_normalized = dist / scale_dist        
        
        # Mask gen_samples to itself
        mask_val = 100.0
        diag_block = torch.eye(C_gen, device=device)                     # (C_gen, C_gen)
        zero_block = torch.zeros((C_gen, C_neg+C_pos), device=device)    # (C_gen, C_neg+C_pos)
        block_mask = torch.cat([diag_block, zero_block], dim=1)          # (C_gen, C_gen+C_neg+C_pos)
        dist_normalized = dist_normalized + block_mask * mask_val
        
        accumulated_force = torch.zeros_like(fixed_gen_samples_scaled, device=device)
        for tau in self.temperature_list:
            logits = -dist_normalized / tau    # (B, C_gen, C_gen+C_neg+C_pos)
            
            softmax_y = torch.softmax(logits, dim=-1)        # Softmax over y (tgt_samples). (B, C_gen, C_gen+C_neg+C_pos)
            # kernel_all = softmax_y
            softmax_x = torch.softmax(logits, dim=-2)        # Softmax over x (gen_samples). (B, C_gen, C_gen+C_neg+C_pos)
            kernel_all = torch.sqrt(torch.clip(softmax_y * softmax_x, min=eps))  # harm.avg. (B, C_gen, C_gen+C_neg+C_pos)
            
            # kernel_all = \sum_{y+}\sum_{y-}[(k(x,y+)/Z_pos) * (k(x,y-)/Z_neg) * (y+ - y-)]
            #            = \sum_{y+}\sum_{y-}[(k(x,y+)/Z_pos)*(k(x,y-)/Z_neg)*(y+)] - \sum_{y+}\sum_{y-}[ (k(x,y+)/Z_pos)*(k(x,y-)/Z_neg)*(y-)]
            #            = \sum_{y+}[(k(x,y+)/Z_pos)*(y+) * \sum_{y-}[(k(x,y-)/Z_neg)]] - \sum_{y-}[(k(x,y-)/Z_neg)*(y-) * \sum_{y+}[(k(x,y+)/Z_pos)]]
            #            = \sum_{y+}[(k(x,y+)/Z_pos)*(y+)] - \sum_{y-}[(k(x,y-)/Z_neg)*(y-)]    (Because \sum_{y}[(k(x,y)/Z_pos)] = 1)
            
            # kernel_all = \sum_{y+}\sum_{y-}[k(x,y+) * k(x,y-) * (y+ - y-)] / (Z_pos*Z_neg)
            #            = \sum_{y+}\sum_{y-}[(k(x,y+)/Z_pos)*(k(x,y-)/Z_neg)*(y+)] - \sum_{y+}\sum_{y-}[(k(x,y+)/Z_pos)*(k(x,y-)/Z_neg)*(y-)]
            #            = \sum_{y+}[(k(x,y+)/Z_pos)*(y+)] - \sum_{y-}[(k(x,y-)/Z_neg)*(y-)]    (Because 1 = \sum_{y+}[(k(x,y+)/Z_pos)] = \sum_{y-}[(k(x,y-)/Z_neg)])
                        
            pos_start = C_gen + C_neg
            kernel_neg = kernel_all[:, :, :pos_start]   # k(x,y-) : (B, C_gen, C_gen+C_neg)
            kernel_pos = kernel_all[:, :, pos_start:]   # k(x,y+) : (B, C_gen, C_pos)
            
            # kernel_all * (Z_pos*Z_neg) = (\sum_{y+}[(k(x,y+)/Z_pos)*(y+)] - \sum_{y-}[(k(x,y-)/Z_neg)*(y-)]) * (Z_pos*Z_neg)
            #                            =  \sum_{y+}[k(x,y+)*(y+)] * Z_neg - \sum_{y-}[k(x,y-)*(y-)]) * Z_pos
            #                            =  coeff_pos + (-coeff_neg)
            Z_neg = torch.sum(kernel_neg, dim=-1, keepdim=True)   # (B, C_gen, 1)
            Z_pos = torch.sum(kernel_pos, dim=-1, keepdim=True)   # (B, C_gen, 1)
            coeff_neg = -kernel_neg * Z_pos        # -k(x,y-)*Z_pos : (B, C_gen, C_gen+C_neg)
            coeff_pos =  kernel_pos * Z_neg        #  k(x,y+)*Z_neg : (B, C_gen, C_pos)            
            coeff_cat = torch.cat([coeff_neg, coeff_pos], dim=-1)   # (B, C_gen, C_gen+C_neg+C_pos)            
            sum_coeff = torch.sum(coeff_cat, dim=-1, keepdim=True)  # (B, C_gen, 1) : k(x,y+)*Z_neg - k(x,y-)*Z_pos
            
            curr_force = torch.einsum("bcC,bCs->bcs", coeff_cat, tgt_samples_scaled)   # (B, C_gen, S)            
            curr_force = curr_force - sum_coeff * fixed_gen_samples_scaled
            
            scale_curr_force = torch.sqrt(torch.clip(torch.mean(curr_force ** 2), min=eps))
            accumulated_force = accumulated_force + curr_force / scale_curr_force
        
        
        return accumulated_force, scale_inputs    # (B, C_gen, S), scalar
            

if __name__ == '__main__':
    temp_list = [1,2,3]
    loss_func = DriftLossFunction(temperature_list=temp_list)
    