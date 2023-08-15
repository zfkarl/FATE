from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import math

class FairSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(FairSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    
    def forward(self, features, labels,sensitive_labels,group_norm,method,mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: target classes of shape [bsz].
            sensitive_labels: sensitive attributes of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = features.device
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        #print('features: ', features.shape[0], ' labels: ', labels.shape[0])

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            sensitive_labels = sensitive_labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)   # [bsz,bsz]
            sensitive_mask = torch.eq(sensitive_labels, sensitive_labels.T).float().to(device) # [bsz,bsz]
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  #[2*bsz, dim]
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature #[2*bsz, dim]
            anchor_count = contrast_count  # 2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)                                      #[2*bsz, 2*bsz]
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) #[2*bsz,1]
        logits = anchor_dot_contrast - logits_max.detach() #[2*bsz, 2*bsz]

        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) #[2*bsz, 2*bsz]
        sensitive_mask = sensitive_mask.repeat(anchor_count, contrast_count) #[2*bsz, 2*bsz]
        n_sensitive_mask=(~sensitive_mask.bool()).float() #[2*bsz, 2*bsz]
        # mask-out self-contrast cases
        logits_mask = torch.scatter(     #[2*bsz, 2*bsz]
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        # compute log_prob
        if method=="FSCL":
            mask = mask * logits_mask #[2*bsz, 2*bsz]
            logits_mask_fair=logits_mask*(~mask.bool()).float()*sensitive_mask #[2*bsz, 2*bsz]
            #logits_mask_fair=logits_mask*sensitive_mask
            exp_logits_fair = torch.exp(logits) * logits_mask_fair  #[2*bsz, 2*bsz]
            exp_logits_sum=exp_logits_fair.sum(1, keepdim=True)  #[2*bsz, 1]
            #log_prob = logits - torch.log(exp_logits_sum+((exp_logits_sum==0)*1))  #[2*bsz, 2*bsz]

            # For hard negatives, code adapted from HCL (https://github.com/joshr17/HCL)
            # =============== hard neg params =================
            tau_plus = 0.1
            beta = 1.0
            temperature = 0.5
            N = (batch_size - 1) * contrast_count
            # =============== reweight neg =================
            # for numerical stability
            exp_logits = torch.exp(logits)
            exp_logits_neg = exp_logits  * logits_mask_fair
            exp_logits_pos = exp_logits * mask
            pos = exp_logits_pos.sum(dim=1) / mask.sum(1)
            imp = (beta * (exp_logits_neg + 1e-9).log()).exp()
            reweight_logits_neg = (imp * exp_logits_neg) / imp.mean(dim=-1)
            Ng = (-tau_plus * N * pos + reweight_logits_neg.sum(dim=-1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
            #log_prob = -torch.log(exp_logits / (((exp_logits_sum==0)*1) + Ng))
            log_prob = logits - torch.log(0.1*Ng+exp_logits_sum+((exp_logits_sum==0)*1))
            # ===============================================
   

        elif method=="SupCon":
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            

        elif method=="FSCL*":
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
            mask=mask*logits_mask
                
            logits_mask_fair=logits_mask*sensitive_mask
            exp_logits_fair = torch.exp(logits) * logits_mask_fair
            exp_logits_sum=exp_logits_fair.sum(1, keepdim=True)
            log_prob = logits - torch.log(exp_logits_sum+((exp_logits_sum==0)*1))

        elif method=="SimCLR":
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
            mask=mask*logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

              
        # compute mean of log-likelihood over positive
        #apply group normalization
        if group_norm==1:
            mean_log_prob_pos = ((mask*log_prob)/((mask*sensitive_mask).sum(1))).sum(1)
           
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        #apply group normalization
        if group_norm==1:
            C=loss.size(0)/8
            norm=(1/(((mask*sensitive_mask).sum(1)+1).float()))
            loss=(loss*norm)*C
            
        loss = loss.view(anchor_count, batch_size).mean()

        return loss





