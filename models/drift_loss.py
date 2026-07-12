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
    dist_squared = torch.sqrt(torch.clip(dist_squared, min=eps))
    return dist_squared


class DriftLossFunction(nn.Module):
    def __init__(self, temperature_list=(0.02, 0.05, 0.2)):
        super().__init__()
        assert isinstance(temperature_list, Iterable), f"Provided temperature_list is not an Iterable object, instead: {type(temperature_list)}"
        self.temperature_list = temperature_list        
        
    
    def forward(self, gen_samples, pos_samples, neg_samples=None, eps=1e-6, temperature_list=None):
        if neg_samples is None:
            neg_samples = torch.zeros_like(gen_samples[:, :0, :], device=gen_samples.device)   # (B, 1, S)
            neg_samples = neg_samples.detach()
        C_neg, C_pos = neg_samples.shape[1], pos_samples.shape[1]
        
        gen_samples_detached = gen_samples.detach()             # Stop Grad           (B, C_gen, S)
        target_samples = torch.cat([gen_samples, neg_samples, pos_samples], dim=1)  # (B, C_gen+C_neg+C_pos, S)
        
        if temperature_list is None:
            temperature_list = self.temperature_list
        else:
            assert isinstance(temperature_list, Iterable), f"Provided temperature_list is not an Iterable object, instead: {type(temperature_list)}"
        V, scale_inputs = self._calculate_force(gen_samples_detached, target_samples, C_neg, C_pos, eps, temperature_list)
        V, scale_inputs = V.detach(), scale_inputs.detach()     # Stop Grad
        
        gen_samples_scaled = gen_samples / scale_inputs         # Grad
        goal_scaled = gen_samples_detached / scale_inputs + V      # Stop Grad
        
        diff = gen_samples_scaled - goal_scaled       # (B, C_gen, S)
        loss = torch.mean(diff ** 2, dim=(-1, -2))    # (B, )
        return loss
        
    def _calculate_force(self, fixed_gen_samples, tgt_samples, C_neg, C_pos, eps, temperature_list):
        B, C_gen, S = fixed_gen_samples.shape
        device = fixed_gen_samples.device
        dist = cdist(fixed_gen_samples, tgt_samples, eps=eps)    # (B, C_gen, C_gen+C_neg+C_pos)
        
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
        for curr_tau in temperature_list:
            logits = -dist_normalized / curr_tau    # (B, C_gen, C_gen+C_neg+C_pos)
            
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
            
            pos_start_idx = C_gen + C_neg
            kernel_neg = kernel_all[:, :, :pos_start_idx]   # k(x,y-) : (B, C_gen, C_gen+C_neg)
            kernel_pos = kernel_all[:, :, pos_start_idx:]   # k(x,y+) : (B, C_gen, C_pos)
            
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
            curr_force = curr_force / scale_curr_force
            accumulated_force = accumulated_force + curr_force         
            
        return accumulated_force, scale_inputs    # (B, C_gen, S), scalar    
    

class NFDriftLossFunction_Ratio(DriftLossFunction):
    def forward(self, gen_samples, pos_samples, nf_score, eps=1e-6, temperature_list=None, vp_vq_ratio=1.0):
        fixed_gen_samples = gen_samples.detach()  # Stop Grad : (B, C_gen, S)
        
        if temperature_list is None:
            temperature_list = self.temperature_list
        else:
            assert isinstance(temperature_list, Iterable), f"Provided temperature_list is not an Iterable object, instead: {type(temperature_list)}"
        
        V_p, scale_inputs = self._calculate_positive_force(fixed_gen_samples, pos_samples, eps, temperature_list)
        V_p, scale_inputs = V_p.detach(), scale_inputs.detach()     # Stop Grad (B, C_gen, S)
        V_q = nf_score.detach()                                     # Stop Grad (B, C_gen, S)
        
        ratio_total = vp_vq_ratio + 1.0
        weight_p = 2.0 * (vp_vq_ratio / ratio_total)
        weight_q = 2.0 * (1.0 / ratio_total)
        V_p = V_p * weight_p
        V_q = V_q * weight_q
        
        gen_samples_scaled = gen_samples / scale_inputs             # Grad
        # with torch.no_grad:
        #     goal_scaled = fixed_gen_samples / scale_inputs + V_p - V_q  # Stop Grad
        goal_scaled = fixed_gen_samples / scale_inputs + V_p - V_q  # Stop Grad
        
        diff = gen_samples_scaled - goal_scaled       # (B, C_gen, S)
        loss = torch.mean(diff ** 2, dim=(-1, -2))    # (B, )
        return loss        
        
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
            softmax_x = torch.softmax(logits, dim=-2)        # Softmax over x (gen_samples). (B, C_gen, C_pos)
            kernel_pos = torch.sqrt(torch.clip(softmax_y * softmax_x, min=eps))  # harm.avg. (B, C_gen, C_pos)
                
            # [Derivation]
            # V_p = \sum_{y+}[ k(x,y+) * (y+ - x) ] / Z_pos
            #     = \sum_{y+}[ k(x,y+)/Z_pos * y+]  - x     (Because 1 = \sum_{y+}[(k(x,y+)/Z_pos)])
            #     = kernel_pos * pos_samples_scaled - fixed_gen_samples_scaled
            kernel_pos_y = torch.einsum("bcC,bCs->bcs", kernel_pos, pos_samples_scaled)     # (B, C_gen, S)
            curr_force = kernel_pos_y - fixed_gen_samples_scaled                            # (B, C_gen, S)
                
            scale_curr_force = torch.sqrt(torch.clip(torch.mean(curr_force ** 2), min=eps))
            curr_force = curr_force / scale_curr_force
            accumulated_force = accumulated_force + curr_force
                
        return accumulated_force, scale_inputs    # (B, C_gen, S), scalar


class NFDriftLossFunction_LearnableRatio(NFDriftLossFunction_Ratio):
    def __init__(self, temperature_list=(0.02, 0.05, 0.2), init_vp_vq_ratio=0.3):
        super().__init__(temperature_list=temperature_list)
        
        init_ratio_inv_softplus = math.log(math.exp(init_vp_vq_ratio) - 1.0)   # Softplus inverse
        
        # V_p-V_q ratio as a learnable parameter : Softplus for stability.
        self.inverted_softplus_ratio = nn.Parameter(torch.tensor(init_ratio_inv_softplus))
        
    
    def forward(self, gen_samples, pos_samples, nf_score, eps=1e-6, temperature_list=None):
        actual_ratio = F.softplus(self.inverted_softplus_ratio)
        
        return super().forward(
            gen_samples, pos_samples, nf_score, eps, temperature_list, actual_ratio
        )


class NFDriftLossFunction_Ratio_Regularize(NFDriftLossFunction_Ratio):
    def __init__(self, temperature_list=(0.02, 0.05, 0.2), reg_lambda=0.01):
        super().__init__(temperature_list=temperature_list)
        self.reg_lambda = reg_lambda
    
    def forward(self, gen_samples, pos_samples, nf_score, eps=1e-6, temperature_list=None, vp_vq_ratio=1.0):
        fixed_gen_samples = gen_samples.detach()  # Stop Grad : (B, C_gen, S)
        
        if temperature_list is None:
            temperature_list = self.temperature_list
        else:
            assert isinstance(temperature_list, Iterable), f"Provided temperature_list is not an Iterable object, instead: {type(temperature_list)}"
        
        V_p, scale_inputs = self._calculate_positive_force(fixed_gen_samples, pos_samples, eps, temperature_list)
        V_p, scale_inputs = V_p.detach(), scale_inputs.detach()     # Stop Grad (B, C_gen, S)
        V_q = nf_score.detach()                                     # Stop Grad (B, C_gen, S)
        
        ratio_total = vp_vq_ratio + 1.0
        weight_p = 2.0 * (vp_vq_ratio / ratio_total)
        weight_q = 2.0 * (1.0 / ratio_total)
        V_p = V_p * weight_p
        V_q = V_q * weight_q
        
        gen_samples_scaled = gen_samples / scale_inputs             # Grad
        # with torch.no_grad:
        #     goal_scaled = fixed_gen_samples / scale_inputs + V_p - V_q  # Stop Grad
        goal_scaled = fixed_gen_samples / scale_inputs + V_p - V_q  # Stop Grad        
        diff = gen_samples_scaled - goal_scaled       # (B, C_gen, S)
        drift_loss = torch.mean(diff ** 2, dim=(-1, -2))    # (B, )
        
        reg_loss = self.reg_lambda * torch.mean(nf_score ** 2, dim=(-1, -2))        
        loss = drift_loss + reg_loss
        
        return loss


class NFDriftLossFunction_LearnableRatio_Regularize(NFDriftLossFunction_Ratio):
    def __init__(self, temperature_list=(0.02, 0.05, 0.2), init_vp_vq_ratio=0.3, reg_lambda=0.01):
        super().__init__(temperature_list=temperature_list)
        self.reg_lambda = reg_lambda
        
        # Learnable Ratio 파라미터 초기화
        init_ratio_inv_softplus = math.log(math.exp(init_vp_vq_ratio) - 1.0)
        self.inverted_softplus_ratio = nn.Parameter(torch.tensor(init_ratio_inv_softplus))
        
    def forward(self, gen_samples, pos_samples, nf_score, eps=1e-6, temperature_list=None):
        if temperature_list is None:
            temperature_list = self.temperature_list
        else:
            assert isinstance(temperature_list, Iterable), f"Provided temperature_list is not an Iterable object, instead: {type(temperature_list)}"
        
        fixed_gen_samples = gen_samples.detach()  # Stop Grad : (B, C_gen, S)
        actual_ratio = F.softplus(self.inverted_softplus_ratio)
        
        V_p, scale_inputs = self._calculate_positive_force(fixed_gen_samples, pos_samples, eps, temperature_list)
        V_p, scale_inputs = V_p.detach(), scale_inputs.detach()     # Stop Grad (B, C_gen, S)
        V_q = nf_score.detach()                                     # Stop Grad (B, C_gen, S)
        
        ratio_total = actual_ratio + 1.0
        weight_p = 2.0 * (actual_ratio / ratio_total)
        weight_q = 2.0 * (1.0 / ratio_total)
        V_p = V_p * weight_p
        V_q = V_q * weight_q
        
        gen_samples_scaled = gen_samples / scale_inputs             # Grad
        goal_scaled = fixed_gen_samples / scale_inputs + V_p - V_q  # Stop Grad        
        diff = gen_samples_scaled - goal_scaled       # (B, C_gen, S)
        drift_loss = torch.mean(diff ** 2, dim=(-1, -2))    # (B, )
        
        reg_loss = self.reg_lambda * torch.mean(nf_score ** 2, dim=(-1, -2))        
        loss = drift_loss + reg_loss
        
        return loss



if __name__ == '__main__':
    temp_list = [1,2,3]
    loss_func = NFDriftLossFunction_Ratio(temperature_list=temp_list)
    