import numpy as np
import torch

def normalize_contributions(model_contributions, scaling="minmax", resultant_norm=None):
    """Normalization of the matrix of contributions/weights extracted from the model."""
    normalized_model_contributions = torch.zeros(model_contributions.size())
    for l in range(0,model_contributions.size(0)):

        if scaling == 'min_max':
            ## Min-max normalization
            min_importance_matrix = model_contributions[l].min(-1, keepdim=True)[0]
            max_importance_matrix = model_contributions[l].max(-1, keepdim=True)[0]
            normalized_model_contributions[l] = (model_contributions[l] - min_importance_matrix) / (max_importance_matrix - min_importance_matrix)
            normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)

        elif scaling == 'sum_one':
            normalized_model_contributions[l] = model_contributions[l] / model_contributions[l].sum(dim=-1,keepdim=True)
            #normalized_model_contributions[l] = normalized_model_contributions[l].clamp(min=0)

        # For l1 distance between resultant and transformer vectors we apply min_sum
        elif scaling == 'min_sum':
            if resultant_norm == None:
                min_importance_matrix = model_contributions[l].min(-1, keepdim=True)[0]
                normalized_model_contributions[l] = model_contributions[l] + torch.abs(min_importance_matrix)
                normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)
            else:
                normalized_model_contributions[l] = model_contributions[l] + torch.abs(resultant_norm[l].unsqueeze(1))
                normalized_model_contributions[l] = torch.clip(normalized_model_contributions[l],min=0)
                normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)
        else:
            print('No normalization selected!')
    return normalized_model_contributions

def compute_joint_attention(att_mat):
    """Compute attention rollout given contributions or attn weights + residual."""
    joint_attentions = torch.zeros(att_mat.size()).to(att_mat.device)
    layers = joint_attentions.shape[0]
    joint_attentions = att_mat[0].unsqueeze(0)

    for i in range(1, layers):
        C_roll_new = torch.matmul(att_mat[i], joint_attentions[i - 1])
        joint_attentions = torch.cat([joint_attentions, C_roll_new.unsqueeze(0)], dim=0)

    return joint_attentions

def compute_rollout(att_mat):
    """ Compute rollout method for raw attention weights."""
    # Add residual connection
    res_att_mat = att_mat + np.eye(att_mat.shape[1])[None,...]
    # Normalize to sum 1
    res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[...,None]
    res_att_mat_torch = torch.tensor(res_att_mat,dtype=torch.float32)
    joint_attentions = compute_joint_attention(res_att_mat_torch) # (num_layers,src_len,src_len)
    return joint_attentions
