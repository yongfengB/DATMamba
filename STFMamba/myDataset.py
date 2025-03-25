#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import sys
import random
from datetime import datetime, timedelta
import matplotlib
import pandas as pd
from random import shuffle
import pickle
import time
import math
import matplotlib as mpl
from tqdm import tqdm

from util import node_feature_to_adj_list

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--attend_dist', type=float, default=0.1, help='Maximum Mileage difference to attend to as neighbors')
parser.add_argument('--attend_lane', type=float, default=1, help='Maximum lane distance to attend to neighbors')
parser.add_argument('--total_lane', type=int, default=4, help='Total number of lanes')
parser.add_argument('--time_window', type=int, default=30, help='Time window length (s)')
parser.add_argument('--save_dir_name', type=str, help='Directory name to save processed data')
parser.add_argument('--edge_weight', action='store_true', default=False, help='Calculate edge weights (based on position)')
parser.add_argument('--edge_weight_headway', action='store_true', default=False, help='Calculate following car headway as edge weights (based on position)')
parser.add_argument('--save_meta', action='store_true', default=False, help='Save metadata along with processed files')
# Specify scenario to process: comprehensive, training, stalled_car, speeding, tailgating, slow
parser.add_argument('--process_scenario', type=str, default='comprehensive', help='Scenario type: comprehensive, training, stalled_car, speeding, tailgating, slow')

args = parser.parse_args()
print(args)

# Read global min and max values for normalization from file
read_dir = '../Data/TRAREAL_df/'
all_maxs, all_mins = pickle.load(open(read_dir + "min_max.pkl", "rb"))

# Compute maximum gap normalized by Mileage range
max_gap = args.attend_dist / (all_maxs.Mileage - all_mins.Mileage)

# Buffer added for lane-based normalization of distances
attend_lane_dist = (args.attend_lane + 0.1) / (args.total_lane - 1)
d_time = args.time_window  # total duration of the time window in seconds

# Create directory to save processed files
save_dir_name = args.save_dir_name
save_path = '../Data/{}/training/'.format(save_dir_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Save normalization parameters for later use
pickle.dump([all_maxs, all_mins], open(save_path + "min_max.p", "wb"))

# Compute scaling factor for spatial dimensions (Mileage and x coordinate)
xy_scale = (all_maxs.Mileage - all_mins.Mileage) * 5280 / (all_maxs.x - all_mins.x)

def process_df(df_orig):
    """
    Normalize specified columns of the input dataframe to the range [0,1].
    The columns normalized are: "x", "Speed", "Acceleration", "Mileage".
    """
    df_orig.loc[:, ["x", "Speed", "Acceleration", "Mileage"]] = (df_orig.loc[:, ["x", "Speed", "Acceleration", "Mileage"]] - all_mins) / (all_maxs - all_mins)
    return df_orig

# The following functions compute two types of edge attributes:
#   edge_attr_sim: similarity edge weight computed as 1/distance.
#   edge_attr_exp: exponential kernel edge weight computed as exp(-distance^2 / std^2).

def save_sample(sample, count, times, df_label=False, save_meta=False, df_n=None):
    """
    Save a sample into graph features.
    
    Args:
        sample (DataFrame): Contains all trajectories in a given time window.
        times (list): List of time steps in the sample.
        df_label (bool): If True, the dataframe contains label information (for testing sets).
        save_meta (bool): If True, save metadata.
        df_n: Dataframe number to store as metadata.
    
    Returns:
        bool: True if the sample is successfully saved, False otherwise.
    """
    # Select only vehicles that are present at all time steps
    count_ID = sample.groupby('ID').Time.count()
    IDs = count_ID[count_ID == d_time].index

    sample = sample[sample.ID.isin(IDs)]
    if IDs.shape[0] == 0:
        return False

    # Save metadata if required
    if save_meta:
        np.save(save_path + "time_stamp_{}".format(count), times)
        np.save(save_path + "IDs_{}".format(count), IDs)
        np.save(save_path + "df_{}".format(count), df_n)
        return True

    # Save graph data for each time step in the window
    for t in range(d_time):
        sample_t = sample[sample.Time == times[t]]
        sample_t = sample_t.set_index('ID')
        sample_t = sample_t.reindex(IDs)
    
        # Extract node features
        # Columns: ["Lane", "Class", "x", "Speed", "Acceleration", "Mileage"]
        node_features = sample_t.loc[:, ["Lane", "Class", "x", "Speed", "Acceleration", "Mileage"]].values
        n_nodes = node_features.shape[0]
        # Compute the adjacency list (edge index) from node features
        adj_list, edge_dict = node_feature_to_adj_list(node_features,
                                                       max_gap,
                                                       attend_lane_dist,
                                                       self_loop=True,
                                                       return_weight=args.edge_weight,
                                                       xy_scale=xy_scale)
        # Save node features and edge index for the current time step
        np.save(save_path + "x_{}_{}".format(count, t), node_features)
        np.save(save_path + "adj_list_{}_{}".format(count, t), adj_list)
        if args.edge_weight:
            np.save(save_path + "edge_attr_exp_{}_{}".format(count, t), edge_dict['weight'][0])
            np.save(save_path + "edge_attr_sim_{}_{}.npy".format(count, t), edge_dict['weight'][1])
        if df_label:
            # Save label for each car at this time step
            label_t = sample_t['label'].values
            np.save(save_path + "label_{}_{}".format(count, t), label_t)
    
    return True

###############  Save Processed Data  ###############

# Processing training samples
if args.process_scenario == 'training':
    count = 0
    train_dfs = os.listdir(os.path.join(read_dir, 'training'))
    for df_name in train_dfs:
        df = pd.read_csv(os.path.join(read_dir, 'training', df_name))
        df = process_df(df)
        for t0 in tqdm(range(0, 299, 1)):
            # 300s (5 minutes) in total
            times = list(range(t0, t0 + d_time))
            time_mask = (df.Time >= t0) & (df.Time < t0 + d_time)
            sample = df[time_mask]
            success = save_sample(sample, count, times)
            if success:
                count += 1
    train_list = list(np.arange(count))
    np.save(save_path + 'train_list', train_list)
else:
    # Processing testing samples
    scenario = args.process_scenario
    save_path = '../Data/{}/testing_{}/'.format(save_dir_name, scenario)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    count = 0
    test_labels = []
    df_label_all = pd.DataFrame()
    for sample_n in [1, 2]:
        # Two dataframes for testing (each 5 minutes, total 10 minutes per scenario)
        path = os.path.join(read_dir, 'testing', '{}_{}.csv'.format(scenario, sample_n))
        df = pd.read_csv(path)
        df = process_df(df)
        for t0 in tqdm(range(0, 299, 1)):
            times = list(range(t0, t0 + d_time))
            time_mask = (df.Time >= t0) & (df.Time < t0 + d_time)
            sample = df[time_mask]
            success = save_sample(sample, count, times, df_label=True, save_meta=args.save_meta, df_n=sample_n)
            if success:
                count += 1
        if args.save_meta:
            df_label = df.loc[:, ["ID", "Time", "label"]]
            df_label["df_n"] = sample_n
            df_label_all = pd.concat([df_label_all, df_label])
    
    df_label_all.to_csv(save_path + 'lable.csv')
    test_list = list(np.arange(count))
    np.save(save_path + 'test_list', test_list)
    print('Scenario:', scenario, ', count:', count)
