import os
import numpy as np
import time
import torch
from utils.data_utils import *
from utils.blob_detection import *
from utils.losses import *
import logging
from models.unet import UNet
from datasets.tem import get_tem_Data
import argparse
import json

args = None
def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--model-dir', default="./models/best_model_0209(1).pt",
                        help='model directory')
    parser.add_argument('--device', default='cuda', help='testing device, choose in cpu or cuda')
    parser.add_argument('--method', type=str, default='NMS', help='method for testing, choose in LoG, DoG, DoH or NMS')
    parser.add_argument('--crop_size', type=int, default='512', help='crop size for testing, choose in [128, 256, 512, 800, 1024, 1280, 1440]')
    args = parser.parse_args() 
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)
    
    '''Settings'''
    crop_size = args.crop_size if args.crop_size%32==0 else 800
    method = args.method
    
    '''Load Data'''
    train_loader, test_loader, input_dim, output_dim = get_tem_Data()
 
    '''Get Model'''
    model = UNet(1, 1)
    model.load_state_dict(torch.load(args.model_dir), strict=False)
    model = model.to(device)
    metric = cal_score

    model.eval()
    val_start = time.time()
    epoch_score = EpochScoreRecorder('precision', 'recall', 'cd' ,'jaccard', 'f1', 'lt')
    log_dict = {'method':args.method, 'device':args.device, 'size':args.crop_size, 'fname':[], 'pre_num':[], 'gt':[], 'lt':[], 'ch':[], 'ja':[], 'f1':[]}
    for idx, (inputs, labels, names) in enumerate(test_loader):
        points_list = torch.where(labels==1.0)
        split_idx = [torch.sum(points_list[0]==i).item() for i in range(inputs.size(0))]
        points_list = [torch.split(points_list[2], split_idx),
                    torch.split(points_list[3], split_idx)]
        points_list = [torch.cat((hs.unsqueeze(1), ws.unsqueeze(1)), dim=1) for hs, ws in zip(*points_list)]
        
        # Left-Top Crop or Right-Bottom Padding
        if crop_size in [128, 256, 512, 800]:
            inputs = inputs[:,:,:crop_size, :crop_size]
            labels = labels[:,:,:crop_size, :crop_size]
            points_list = [points[~((points[:,0]>=crop_size) + (points[:,1]>=crop_size))] for points in points_list]
        elif crop_size>800:
            inputs = F.pad(inputs, (0, crop_size-800, 0, crop_size-800))
            labels = F.pad(labels, (0, crop_size-800, 0, crop_size-800))
        
        # To cuda
        inputs = inputs.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.FloatTensor).to(device)
        points_list = [points.type(torch.FloatTensor).to(device) for points in points_list]
                    
        out_diam, out_map = model(inputs)
                    
        gray_th = 0.1
        iou_th = 0.3
        window_size = 8
        locate_time_start = time.time()
        if method in ["log", "Log", "LoG", "LOG"]:
            keep_centers = LoG(out_map, gray_threshold=gray_th, max_sigma=torch.sqrt(out_diam).item(), num_sigma=10, threshold=0.1).to(device)
        elif method in ["dog", "Dog", "DoG", "DOG"]:
            keep_centers = DoG(out_map, gray_threshold=gray_th, max_sigma=torch.sqrt(out_diam).item(), threshold=0.1).to(device)
        elif method in ["doh", "Doh", "DoH", "DOH"]:
            keep_centers = DoH(out_map, gray_threshold=gray_th, max_sigma=10, threshold=0.004).to(device)
        else:
            method = "NMS"
            keep_centers = visualize(inputs, out_diam, out_map, points_list, names, gray_th, iou_th, window_size, None, device)
        
        method_name="{}-{}-{}".format(method, args.device, crop_size)
        lt = (time.time()-locate_time_start) * 1000.0
        precision, recall, cd ,jaccard, f1 = metric(keep_centers, points_list[0].squeeze())
        epoch_score.update(precision.item(), recall.item(), cd.item(), jaccard.item(), f1.item(), lt)
        
        # Output External Log
        log_dict['fname'].append(names[0])
        log_dict['pre_num'].append(len(keep_centers))
        log_dict['gt'].append(len(points_list[0]))
        log_dict['lt'].append(lt)
        log_dict['ch'].append(cd.item())
        log_dict['ja'].append(jaccard.item())
        log_dict['f1'].append(f1.item())
        
    means = epoch_score.cal_mean()
    stds = epoch_score.cal_std()
    log_str = '{} Total Score pre: {} | re: {} | ch: {} | ja: {} | f1: {} | lt: {}ms | Cost {}s'.format(method_name,
                                                                                                        (means['precision'], stds['precision']),
                                                                                                        (means['recall'], stds['recall']),
                                                                                                        (means['cd'], stds['cd']),
                                                                                                        (means['jaccard'], stds['jaccard']),
                                                                                                        (means['f1'], stds['f1']),
                                                                                                        (means['lt'], stds['lt']),
                                                                                                        time.time()-val_start)
    print(log_str)
    #log_dict.update({'log_str': log_str})
    #with open("./response_test_log.json", "r",  encoding="utf-8") as f:
    #    data_dict = json.load(f)
    #data_dict.update({method_name: log_dict})
    #with open("./response_test_log.json", "w",  encoding="utf-8") as f1:
    #    data_dict = json.dump(data_dict, f1)
    