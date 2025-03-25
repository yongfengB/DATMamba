from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import math
import sys
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch.nn as nn
from torch_geometric.utils import coalesce 

# ------------------- Utility Functions ------------------- #

def merge_edges_over_time(graph_data):
    """
    Merge the edge_index of multiple time steps into one edge set.
    """
    edge_l = [dt.edge_index for dt in graph_data]
    edge_all = torch.concat(edge_l, -1)
    return coalesce(edge_all)

def normalize(val, v_min, v_max):
    """
    Normalize a value to the range [0, 1].
    """
    return (val - v_min) / (v_max - v_min)

def scale_back(val, v_min, v_max):
    """
    Scale a normalized value [0,1] back to its original range.
    """
    return val * (v_max - v_min) + v_min

def node_feature_to_adj_list(node_features, max_gap, attend_lane_dist, self_loop=True, return_weight=False, xy_scale=1):
    """
    Calculate an adjacency list from node features (including Mileage and lane).
    Node features are expected to be in the order:
      ["Lane", "Class", "x", "Speed", "Acceleration", "Mileage"].
    
    Args:
        node_features (ndarray): Input node features.
        max_gap (float): Maximum longitudinal distance (in normalized miles) to consider neighbors.
        attend_lane_dist (float): Maximum lateral (lane) distance to consider.
        self_loop (bool): Whether to include self-loops.
        return_weight (bool): If True, return edge weights computed from distance.
        xy_scale (float): Scaling factor to convert differences in x/y to comparable units.
    
    Returns:
        adj_list (ndarray): Adjacency list as a 2xE numpy array.
        edge_dict (dict): Dictionary containing edge weights if requested.
    """
    n_nodes = node_features.shape[0]
    # Use column 2 ("x") for longitudinal info and last column ("Mileage") for relative distance
    lane_all = node_features[:, 2]
    dist_all = node_features[:, -1]
    speed_all = node_features[:, 3]
    lane_adj = np.expand_dims(lane_all, 1) - np.expand_dims(lane_all, 0)
    dist_adj = np.expand_dims(dist_all, 1) - np.expand_dims(dist_all, 0)
    # For dist_adj, positive means node i is ahead of node j.
    dist_adj_bool = np.abs(dist_adj) <= max_gap
    lane_adj_bool = np.abs(lane_adj) <= attend_lane_dist 
    adj_matrix = lane_adj_bool & dist_adj_bool
    if not self_loop:
        adj_matrix &= ~np.eye(n_nodes, dtype=bool)
    # Build adjacency list
    adj_list = [[i, j] for i in range(n_nodes) for j in range(n_nodes) if adj_matrix[i][j]]
    adj_list = np.array(adj_list)
    edge_dict = {}
    
    if return_weight:
        # Compute Euclidean distance in (longitudinal, lane) space
        d_norm = np.sqrt((dist_adj * xy_scale) ** 2 + lane_adj ** 2)
        # Exponential kernel weight and similarity weight
        adj_exp = np.exp(-d_norm)
        adj_sim = 1 / (d_norm + np.eye(d_norm.shape[0]))  # add eye to avoid division by zero
        attr_exp = adj_exp[adj_matrix]
        attr_sim = adj_sim[adj_matrix]
        edge_dict['weight'] = (attr_exp, attr_sim)
    
    return adj_list.T, edge_dict

# ------------------- Loss Functions ------------------- #

# Define loss criteria with default reduction='mean'
criterion_NLL = nn.GaussianNLLLoss()
criterion_CEL = nn.CrossEntropyLoss()

def cal_loss(model_output, graph_x, rate_lane, rate_speed=0, rate_acc=0, mask=[], n_lane=4):
    """
    Compute the training loss for a sample graph.
    
    model_output and graph_x have shape [n_nodes, time, feature],
    where model_output's feature vector is structured as:
      [mu_dist, log_var_dist, p_lane1, ..., p_lanen, mu_v, log_var_v, mu_a, log_var_a].
      
    The loss is a weighted sum of:
      - Negative log-likelihood for distance (Gaussian)
      - Cross-entropy loss for lane classification
      - (Optionally) NLL loss for speed and acceleration.
    
    Args:
        model_output (Tensor): The network's predicted output.
        graph_x (Tensor): The ground-truth input features.
        rate_lane (float): Weight for lane classification loss.
        rate_speed (float): Weight for speed loss.
        rate_acc (float): Weight for acceleration loss.
        mask (Tensor or list): Boolean mask for valid (non-padded) entries.
        n_lane (int): Number of lane classes.
        
    Returns:
        loss (Tensor): A scalar loss value.
    """
    # Split the predictions
    # Distance: first two dims (mean and log variance)
    dist_pred = model_output[:, :, :2]
    # Lane predictions: next n_lane dimensions
    lane_pred = model_output[:, :, 2:2+n_lane]
    # Ground truth: lane (as integer, from graph_x first column) and distance (last column)
    lane = graph_x[:, :, [0]].long()
    dist = graph_x[:, :, [-1]]
    
    # Reshape lane for CE loss
    lane = lane.view(-1)  # [n_nodes * time]
    lane_pred = lane_pred.view(-1, lane_pred.shape[-1])
    
    # Compute Gaussian NLL for distance prediction
    mean_dist = dist_pred[:, :, [0]]
    var_dist = torch.exp(dist_pred[:, :, [1]])
    if len(mask) > 0:
        loss_nll = criterion_NLL(mean_dist[mask], dist[mask], var_dist[mask])
        loss_cel = criterion_CEL(lane_pred[mask.flatten()], lane[mask.flatten()])
    else:
        loss_nll = criterion_NLL(mean_dist, dist, var_dist)
        loss_cel = criterion_CEL(lane_pred, lane)
        
    loss = loss_nll + rate_lane * loss_cel

    # Speed loss (if applicable)
    if rate_speed > 0:
        # Speed predictions assumed to be located immediately after lane predictions
        speed_pred = model_output[:, :, 2+n_lane:4+n_lane]
        speed = graph_x[:, :, [3]]
        mean_sp = speed_pred[:, :, [0]]
        var_sp = torch.exp(speed_pred[:, :, [1]])
        if len(mask) > 0:
            loss_speed = criterion_NLL(mean_sp[mask], speed[mask], var_sp[mask])
        else:
            loss_speed = criterion_NLL(mean_sp, speed, var_sp)
        loss += rate_speed * loss_speed

    # Acceleration loss (if applicable)
    if rate_acc > 0:
        acc_pred = model_output[:, :, 4+n_lane:6+n_lane]
        acc = graph_x[:, :, [4]]
        mean_acc = acc_pred[:, :, [0]]
        var_acc = torch.exp(acc_pred[:, :, [1]])
        if len(mask) > 0:
            loss_acc = criterion_NLL(mean_acc[mask], acc[mask], var_acc[mask])
        else:
            loss_acc = criterion_NLL(mean_acc, acc, var_acc)
        loss += rate_acc * loss_acc

    return loss

# For evaluation, we compute loss per vehicle (car)
NLL_loss = nn.GaussianNLLLoss(reduction='none')
CE_loss = nn.CrossEntropyLoss(reduction='none')

def cal_loss_car(model_output, graph_x, rate_lane, rate_speed=0, rate_acc=0, time_agg_loss=True, mask=[], n_lane=4):
    """
    Compute the per-vehicle loss for evaluation.
    
    model_output and graph_x have shape [n_nodes, time, feature],
    with model_output structured as:
      [mu_dist, log_var_dist, p_lane1, ..., p_lanen, mu_v, log_var_v, mu_a, log_var_a].
      
    If time_agg_loss is True, loss is aggregated (e.g. averaged) over time for each vehicle.
    
    Args:
        model_output (Tensor): Predicted output.
        graph_x (Tensor): Ground-truth features.
        rate_lane (float): Loss weight for lane prediction.
        rate_speed (float): Loss weight for speed.
        rate_acc (float): Loss weight for acceleration.
        time_agg_loss (bool): Whether to aggregate loss over time.
        mask (Tensor or list): Boolean mask for valid entries.
        n_lane (int): Number of lane classes.
    
    Returns:
        loss (Tensor): Loss per vehicle (if aggregated over time, shape [n_vehicle]).
    """
    # Split predictions
    dist_pred = model_output[:, :, :2]
    lane_pred = model_output[:, :, 2:2+n_lane]
    lane = graph_x[:, :, [0]].long()
    dist = graph_x[:, :, [-1]]
    lane = lane.view(-1)
    lane_pred = lane_pred.view(-1, lane_pred.shape[-1])
    
    mean_dist = dist_pred[:, :, [0]]
    var_dist = torch.exp(dist_pred[:, :, [1]])
    
    n_car = model_output.shape[0]
    loss_nll = NLL_loss(mean_dist, dist, var_dist).squeeze(-1)  # [n_car, time]
    loss_cel = CE_loss(lane_pred, lane).view(n_car, -1)  # [n_car, time]
    loss = loss_nll + rate_lane * loss_cel

    # Speed loss
    if rate_speed > 0:
        speed_pred = model_output[:, :, 2+n_lane:4+n_lane]
        speed = graph_x[:, :, [3]]
        mean_s = speed_pred[:, :, [0]]
        var_s = torch.exp(speed_pred[:, :, [1]])
        loss_speed = NLL_loss(mean_s, speed, var_s).squeeze(-1)
        loss += rate_speed * loss_speed

    # Acceleration loss
    if rate_acc > 0:
        acc_pred = model_output[:, :, 4+n_lane:6+n_lane]
        acc = graph_x[:, :, [4]]
        mean_acc = acc_pred[:, :, [0]]
        var_acc = torch.exp(acc_pred[:, :, [1]])
        loss_acc = NLL_loss(mean_acc, acc, var_acc).squeeze(-1)
        loss += rate_acc * loss_acc

    if len(mask) > 0:
        loss[~mask] = np.nan

    if time_agg_loss:
        # Aggregate loss per vehicle over time (e.g., using nanmean to ignore missing entries)
        loss = loss.nanmean(1)
    return loss

# --------------- Functions for Bivariate Normal Trajectory Modeling --------------- #

def pred_to_distribution(traj_pred):
    """
    Convert model output distribution parameters to a multivariate Normal distribution.
    traj_pred: [n_node, time_step, dim_feat2] with dim_feat2 = 5, representing
               [mu_x, mu_y, log_std_x, log_std_y, corr]
    """
    sx = torch.exp(traj_pred[:, :, 2])  # Standard deviation in x
    sy = torch.exp(traj_pred[:, :, 3])  # Standard deviation in y
    corr = torch.tanh(traj_pred[:, :, 4])  # Correlation coefficient
    
    cov = torch.zeros(traj_pred.shape[0], traj_pred.shape[1], 2, 2, device=traj_pred.device)
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy
    mean = traj_pred[:, :, 0:2]
    
    return MultivariateNormal(mean, cov)

def bivariate_loss(traj, traj_pred):
    """
    Compute the loss for bivariate trajectory prediction.
    traj: [n_node, time_step, 2] ground-truth positions [x, y].
    traj_pred: [n_node, time_step, 5] predicted parameters [mu_x, mu_y, log_std_x, log_std_y, corr].
    Loss is defined as the negative log likelihood under the predicted distribution.
    """
    dist = pred_to_distribution(traj_pred)
    loss = -dist.log_prob(traj).mean().mean()
    return loss
