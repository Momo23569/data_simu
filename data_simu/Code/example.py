"""
Created on Sep 15th 18:05:53 2020

Author: Qizhong Zhang

Main Function
"""

import csv
import datetime
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Subset

from data_prepare import MYDATA, reform

from Timegeo import TimeGeo

torch.set_default_tensor_type(torch.DoubleTensor)


class parameters(object):
    def __init__(self, args) -> None:
        super().__init__()    
        self.data_type = args.data_type

    # Data-related information
    def data_info(self, GPS):
        self.GPS = GPS
        self.tim_size = 1440

def gen_gps():
    # 设置随机种子以确保结果的可重复性
    np.random.seed(42)

    # 定义生成GPS数据的数量
    num_gps_points = 4528

    # 假定一个基准点，围绕这个点生成随机的GPS坐标
    base_latitude = 35.51168469
    base_longitude = 139.6733776

    # 定义坐标变动的范围
    latitude_variation = 0.0005  # 纬度变化范围
    longitude_variation = 0.012  # 经度变化范围

    # 生成随机的GPS坐标数据
    gps_data = np.zeros((num_gps_points, 2))  # 初始化数组
    for i in range(num_gps_points):
        random_latitude_shift = np.random.uniform(-latitude_variation, latitude_variation)
        random_longitude_shift = np.random.uniform(-longitude_variation, longitude_variation)
        gps_data[i][0] = base_latitude + random_latitude_shift
        gps_data[i][1] = base_longitude + random_longitude_shift

    return gps_data

def gen_data():
    # 设置随机种子以确保结果的可重复性
    np.random.seed(42)
    # 初始化数据字典
    data = {}

    # 定义生成数据的用户数量和每个用户的轨迹数范围
    num_users = 46
    min_trajectories_per_user = 1
    max_trajectories_per_user = 6

    # 为了确保所有轨迹长度相同，设置一个固定的轨迹点数
    fixed_points_per_trajectory = 7

    # 定义位置、时间和状态数据的可能范围
    loc_range = (2000, 4000)
    tim_range = (136000, 137000)
    sta_range = (10, 100)

    # 生成数据
    for user_id in range(1, num_users + 1):
        num_trajectories = np.random.randint(min_trajectories_per_user, max_trajectories_per_user + 1)
        data[user_id] = {}
        for trajectory_id in range(num_trajectories):
            
            locs = np.array([2653, 2744, 2654, 2653, 2745, 2744, 3075])
            # locs = np.random.randint(loc_range[0], loc_range[1], fixed_points_per_trajectory)
            tims = np.array([11740.61666667, 11757.53333333, 11778.65, 11819.31666667, 11833.56666667, 12035.68333333, 12064.13333333])
            stas = np.array([16.91666667, 21.11666667, 40.66666667, 14.25, 202.11666667, 28.45, 1383.16666667])

            # tims = np.random.uniform(tim_range[0], tim_range[1], fixed_points_per_trajectory)
            
            # stas = np.random.uniform(sta_range[0], sta_range[1], fixed_points_per_trajectory)

            # 为当前轨迹分配生成的随机数据
            data[user_id][trajectory_id] = {
                'loc': locs,
                'tim': tims,
                'sta': stas
            }

    return data

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_type', type=str, default='FourSquare_TKY', choices=['FourSquare_NYC', 'FourSquare_TKY'])
    
    gps_data = gen_gps()

    args = parser.parse_args()
    param = parameters(args)
    param.data_info(gps_data)
   
    data = gen_data()
    
    TG = []
    TG.append(TimeGeo(data, param))
   
    print(len(TG)) # 1
    print(len(TG[0])) # 46
    print(TG[0][1].keys()) # dict_keys([0, 1, 2, 3, 4, 5, 6])
    print(TG[0][1][0].keys()) # dict_keys(['loc', 'tim', 'sta'])
    

    
