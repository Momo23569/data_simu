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

# Hyperparameters
class parameters(object):
    def __init__(self, args) -> None:
        super().__init__()    
        # Data-related
        self.data_type = args.data_type
        self.location_mode = args.location_mode
        self.trainsize = args.trainsize

    # Data-related information
    def data_info(self, data):
        self.POI = data.POI
        self.GPS = data.GPS
        self.USERLIST = data.USERLIST

        self.loc_size = data.loc_size
        self.tim_size = data.tim_size
        self.usr_size = data.usr_size
        self.poi_size = data.poi_size

        self.infer_maxlast = data.infer_maxlast 
        

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()

    # Data-related
    # parser.add_argument('-d', '--data_type', type=str, default='ISP', choices=['ISP', 'GeoLife', 'FourSquare_NYC', 'FourSquare_TKY'])
    # parser.add_argument('-d', '--data_type', type=str, default='FourSquare_NYC', choices=['FourSquare_NYC', 'FourSquare_TKY'])
    # parser.add_argument('-d', '--data_type', type=str, default='GeoLife', choices=['FourSquare_NYC', 'FourSquare_TKY'])
    parser.add_argument('-d', '--data_type', type=str, default='FourSquare_TKY', choices=['FourSquare_NYC', 'FourSquare_TKY'])
    
    parser.add_argument('-l', '--location_mode', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('-t', '--trainsize', type=float, default=0.9)

    

    # hyperparameters and dataset
    args = parser.parse_args()
    param = parameters(args)

    data = MYDATA(param.data_type, param.location_mode)
    param.data_info(data)

    trainid, validid, testid = data.split(validprop=0.9 - param.trainsize)
    trainset, validset, testset = Subset(data, trainid), Subset(data, validid), Subset(data, testid)

    reform(trainset, 'train')
    reform(testset, 'test')
    print(len(data.REFORM['train']))
    print(len(data.REFORM['test']))

   
    TG = []
    SM = []

    
    TG.append(TimeGeo(data.REFORM['train'], param))
    print(len(TG))
    print(len(TG[0]))

    
