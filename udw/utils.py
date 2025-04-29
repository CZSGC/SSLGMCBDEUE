# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.



import torch
import torch.nn as nn

from semilearn.algorithms.utils import concat_all_gather
from semilearn.algorithms.hooks import MaskingHook
import torch.nn.functional as F
from semilearn.nets.wrn.wrn import BasicBlock

class udwMcDropoutHook(MaskingHook):
    """
    Pseudo Labeling Hook
    """
    def __init__(self, num_classes, n_sigma=2, momentum=0.999, per_class=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.n_sigma = n_sigma
        self.per_class = per_class



    @torch.no_grad()
    def update(self, algorithm, probs_x_ulb):
        if algorithm.distributed and algorithm.world_size > 1:
            probs_x_ulb = concat_all_gather(probs_x_ulb)
        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        if not self.per_class:
            prob_max_mu_t = torch.mean(max_probs) # torch.quantile(max_probs, 0.5)
            prob_max_var_t = torch.var(max_probs, unbiased=True)
            self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t.item()
            self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t.item()
        else:
            prob_max_mu_t = torch.zeros_like(self.prob_max_mu_t)
            prob_max_var_t = torch.ones_like(self.prob_max_var_t)
            for i in range(self.num_classes):
                prob = max_probs[max_idx == i]
                if len(prob) > 1:
                    prob_max_mu_t[i] = torch.mean(prob)
                    prob_max_var_t[i] = torch.var(prob, unbiased=True)
            self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t
            self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t
        return max_probs, max_idx
    


    @torch.no_grad()
    def mcDropoutPredict(self, algorithm, inputs, n_iter=10):
        """Perform Monte Carlo Dropout predictions."""
        probs = []
        
        algorithm.model.train()  
        with torch.no_grad(): 
            for _ in range(n_iter):
                outputs = algorithm.model(inputs)
                logits = torch.clamp(outputs['logits'], min=-100, max=100)  
                prob = torch.softmax(logits, dim=-1)
                probs.append(prob)


        probs = torch.stack(probs)  
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print("probs contains NaN or Inf!")
            probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)


        mean_probs = probs.mean(dim=0)  
        entropy_of_mean = - (mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1) 

        entropy_per_sample = - (probs * torch.log(probs + 1e-10)).sum(dim=-1)  
        mean_entropy = entropy_per_sample.mean(dim=0)  

     
        uncertainty = entropy_of_mean + mean_entropy  

        if torch.isnan(uncertainty).any() or torch.isinf(uncertainty).any():
            print("uncertainty contains NaN or Inf!")
            uncertainty = torch.zeros_like(uncertainty)

 
        normalized_uncertainty = uncertainty / torch.log(torch.tensor(self.num_classes, dtype=torch.float))
        normalized_uncertainty = torch.clamp(normalized_uncertainty, min=0.0, max=1.0) 

        print("normalized_uncertainty min:", normalized_uncertainty.min().item(), 
            "max:", normalized_uncertainty.max().item())
        return normalized_uncertainty

    @torch.no_grad()
    def masking(self, algorithm, probs_x_ulb_w, inputs):

        max_probs, max_idx = probs_x_ulb_w.max(dim=-1)


        normalized_uncertainty = self.mcDropoutPredict(algorithm, inputs)
        if torch.isnan(normalized_uncertainty).any() or torch.isinf(normalized_uncertainty).any():
            print("normalized_uncertainty contains NaN or Inf!")
            normalized_uncertainty = torch.ones_like(normalized_uncertainty) 


        ezmask1 = max_probs.ge(algorithm.p_cutoff) * (normalized_uncertainty > 0.5).to(max_probs.dtype)
        ezmask2 = (max_probs.lt(algorithm.p_cutoff) & max_probs.gt(0.6)) * (normalized_uncertainty < 0.3).to(max_probs.dtype)


        print("max_probs min:", max_probs.min().item(), "max:", max_probs.max().item())
        print("normalized_uncertainty:", normalized_uncertainty)
        print("ezmask1 mean:", ezmask1.float().mean().item())
        print("ezmask2 mean:", ezmask2.float().mean().item())

        return ezmask1, ezmask2

class GCELoss(nn.Module):
    def __init__(self, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q  

    def forward(self, logits, targets,mask):

        softmax_output = torch.softmax(logits, dim=1)  
        pred = softmax_output.gather(1, targets.view(-1, 1)).squeeze()  
        loss = (1 - pred ** self.q) / self.q 
        loss =  loss*mask
        return loss.mean() 