# coding:utf-8
import numpy as np
import os
import cv2
import csv
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torchvision.ops import nms
import torch.nn.functional as F


COLOR_LIST = [np.array([0,0,0]).astype('uint8'),    # black represents the background on default
              np.array([255,0,0]).astype('uint8'),
              np.array([0,255,0]).astype('uint8'),
              np.array([0,0,255]).astype('uint8')]

def csv2array(path:str):
    array = []
    with open(path, 'r') as f:
        content = csv.reader(f)
        next(content)
        for line in content:
            array.append(line)
    
    return np.array(array)

def getFileList(path:str, endwith:str):
    iter = os.walk(path)
    path_list = []
    for p, d, filelist in iter:
        for name in filelist:
            if name.endswith(endwith):
                path_list.append(os.path.join(p, name))
                
    path_list.sort()
    return path_list

def getImgList(path:str, endwith:str):
    filelist = getFileList(path, endwith)
    img_list = []
    for path in filelist:
        img = cv2.imread(path, 0)
        img_list.append(img)
    
    return img_list

def getArrayList(path:str, endwith:str='csv'):
    filelist = getFileList(path, endwith)
    array_list = []
    for path in filelist:
        array = csv2array(path)
        array_list.append(array[:,[1,2,-1]].astype('float').astype('int'))
    
    return array_list


def nms_by_diam(out_map, diam=3, score_th=0.1, iou_th=0.5, device='cuda'):
    out_map = out_map.detach().squeeze()
    out_map[out_map<score_th] *= 0.0
    cood = torch.nonzero(out_map)
    if cood.size(0) == 0:
        return cood

    scores = out_map[cood[:,0],cood[:,1]]
    
    boxes = torch.zeros(size=(cood.shape[0],4), device=device)
    boxes[:,0] = cood[:,0] - diam
    boxes[:,1] = cood[:,1] - diam
    boxes[:,2] = cood[:,0] + diam
    boxes[:,3] = cood[:,1] + diam
    
    keep = nms(boxes.to(device), scores.to(device), iou_th).long()
    keep_centers = cood[keep,:].long()

    return keep_centers

class EpochScoreRecorder():
    def __init__(self, *names:str):
        self.names = names
        self.recorder = {name:[] for name in names}
        self.means = {name:-1.0 for name in names}
        self.stds = {name:0.0 for name in names}
    
    def update(self, *scores:float):
        for name, score in zip(self.names, scores):
            self.recorder[name].append(score)
        return self.recorder
            
    def cal_mean(self):
        for name in self.names:
            self.means[name]  = np.mean(self.recorder[name])
        return self.means
    
    def cal_std(self):
        for name in self.names:
            self.stds[name]  = np.std(self.recorder[name])
        return self.stds

def cal_score(points_1, points_2):
    if len(points_1.shape)<2: points_1 = points_1[None,:]
    if len(points_2.shape)<2: points_2 = points_2[None,:]
    
    x_dis = (points_1[:,0][:,None] - points_2[:,0][None,:]) ** 2
    y_dis = (points_1[:,1][:,None] - points_2[:,1][None,:]) ** 2
    dis = torch.sqrt(x_dis + y_dis)

    rematch_indices = dis[dis.argmin(0), :].argmin(1)
    matches = torch.sum(rematch_indices - torch.arange(dis.size(1), device=dis.device) == 0)

    '''Precision and Recall'''
    precision = matches / max(len(points_1),  1)
    recall = matches / max(len(points_2), 1)
    
    '''Chamfer Distance'''
    cd = torch.mean(dis.min(1)[0]) + torch.mean(dis.min(0)[0])
    
    '''Jaccard Score'''
    jaccard = matches / (len(points_1) + len(points_1) - matches)
    
    '''F1 score'''
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, cd ,jaccard, f1

def visualize(inputs, out_diams, out_maps, points_list, names, gray_th:float=0.1, iou_th:float=0.3, window_size:int=8, save_path=None, device='cuda'):
    if points_list is None:
        points_list = [None] * len(inputs)
    
    for input_img, out_diam, out_map, name, points in zip(inputs, out_diams, out_maps, names, points_list):
        # Refine predicted average spacing
        diam = torch.floor(out_diam).int().item()
        
        # Local spacial maximum filtering
        out_map, maxpool_indices = F.max_pool2d(out_map.unsqueeze(0), window_size, window_size, return_indices=True)
        out_map = F.max_unpool2d(out_map, maxpool_indices, window_size, window_size)
        
        # Non-maximum suppression
        keep_centers = nms_by_diam(out_map, diam, gray_th, iou_th, device)
        
        if save_path is not None:
            # Restore input image
            image = input_img.clone().squeeze().detach().cpu()
            image -= torch.min(image)
            image /= torch.max(image)
            image *= 255.0
        
            # Draw predicted output
            mark_map = image.clone().squeeze().detach().cpu().numpy()
            mark_map = cv2.cvtColor(mark_map, cv2.COLOR_GRAY2BGR)
            for center in keep_centers:
                cv2.circle(mark_map, (center[1].item(), center[0].item()), max(1, diam//4), (0, 95, 255), -1)
            cv2.imwrite(os.path.join(save_path, str(name)+"_predict.png"), mark_map.astype('uint8'))
            
            # Draw ground truth
            if points is not None:
                points = points.type(torch.LongTensor)
                gt = image.clone().squeeze().detach().cpu().numpy()
                gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
                for center in points:
                    cv2.circle(gt, (center[1].item(), center[0].item()), max(1, diam//4), (0, 255, 95), -1)
                cv2.imwrite(os.path.join(save_path, str(name)+"_gt.png"), gt.astype('uint8'))
        
            '''
            out_map = out_map / (out_map.max() + 1e-6) * 255.0
            cv2.imwrite(os.path.join(save_path, str(name)+"_map.png"), out_map.astype('uint8'))
            '''
        return keep_centers
        
def lr_scheduler(epoch, optimizer, decay_eff=0.1, decayEpoch=[], is_print:bool=False):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch in decayEpoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_eff
            if is_print:
                print('New learning rate is: ', param_group['lr'])
    return optimizer

def sobel2D(img:np.ndarray, ksize:int=3):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    gm = cv2.sqrt(sobelx**2 + sobely**2)
    return gm

def vacuum2peaks(inputs:np.ndarray, points:np.ndarray, save_path:str):
    '''convert vacuum areas to visible peaks'''
    vpeaks = np.zeros_like(inputs).astype('float')
    
    for i in range(vpeaks.shape[0]):
        for j in range(vpeaks.shape[1]):
            dis = (np.array([[i,j]]) - points)**2
            dis = np.sum(dis, axis=-1)
            vpk = dis.min()
            vpeaks[i,j] = vpk if vpk <= 300 else 300
    
    the_mean = vpeaks.mean()
    the_std = vpeaks.std()
    # vpeaks[vpeaks>the_mean+3*the_std] = the_mean+3*the_std
    
    vpeaks = 1 - np.exp(-vpeaks / (2*the_std**2))
    # vpeaks = sobel2D(vpeaks, 5)
    # vpeaks[vpeaks<0.0] = 0.0
    vpeaks = vpeaks/vpeaks.max() * 255
    vpeaks  = cv2.Canny(vpeaks.astype(np.uint8), -100, 10)
    #vpeaks = cv2.erode(vpeaks, np.ones((3,3), np.uint8), iterations=1)
    
    cv2.imwrite(save_path.replace("\\tem", "\\v").replace(".bmp", "vc.png"), vpeaks)
    return vpeaks

def get_gt(points:np.ndarray, target:np.ndarray, save_path:str):
    '''Draw grondtruth from points'''
    x_dis = (points[None,:, 0] - points[:, None, 0]) ** 2
    y_dis = (points[None,:, 1] - points[:, None, 1]) ** 2
    dis = np.sqrt(x_dis + y_dis)
    dis = dis + np.eye(len(dis)) * 1e3   # erase self-distance = 0
    dis = dis.min(1)[0].mean().item()
    
    gaussianBlur = transforms.GaussianBlur(round(dis/2) * 2 + 1, dis/4.0)
    target = gaussianBlur(torch.from_numpy(target)[None, :, :])
    target /= target.max()
    target = target.numpy() * 255
    
    cv2.imwrite(save_path.replace("\\tem", "\\gt").replace(".bmp", "_gt.png"), target.squeeze().astype("uint8"))
    return target

def get_noise(img:np.ndarray, target:np.ndarray, save_path:str):
    '''Extract noise from samples'''
    img = img.astype('float')
    target = target.astype('float')
    i_mean = img.mean()
    i_std = img.std()
    
    target = cv2.erode(target, np.ones([3,3]), iterations=3)
    target[target>0] = 1
    target[target<0] = 0
    img[target>0] = 0
    sub = np.random.randn(*img.shape) * i_std + i_mean
    img = img + target * sub
    img[img<0] = 0
    img[img>255] = 255
    img = img.astype(np.uint8)
    
    cv2.imwrite(save_path.replace(".bmp", "noise.png"), img)
    return img
    

def shift_coorelation(img:np.ndarray):
    height, width = img.shape
    dxs = np.arange(-width, width)
    dys = np.arange(-height, height)
    corr_x = []
    corr_y = []
    for dx in dxs:
        M  = np.float32([[1, 0, dx], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (width,height))
        corr = 1-np.exp(-np.sum(dst * img.astype('float'))/1e8)
        corr_x.append(corr)
        
    for dy in dys:
        M  = np.float32([[1, 0, 0], [0, 1, dy]])
        dst = cv2.warpAffine(img, M, (width,height))
        corr = 1-np.exp(-np.sum(dst * img.astype('float'))/1e8)
        corr_y.append(corr)
    
    return dxs, corr_x, dys, corr_y

def rotate_coorelation(img:np.ndarray):
    height, width = img.shape
    das = np.arange(-180, 180)
    corr_r = []
    for da in das:
        M  = np.getRotationMatrix2D((height/2, width/2), da, 1.0)
        dst = cv2.warpAffine(img, M, (width,height))
        corr = 1-np.exp(-np.sum(dst * img.astype('float'))/1e8)
        corr_r.append(corr)
        
    return das, corr_r


if __name__ == '__main__':
    name_list = getFileList("E:\\project\\AtomIDNet\\datasets\\tem", ".bmp")
    points_list = [csv2array(path.replace(".bmp", ".csv")) for path in name_list]
    for i in tqdm(range(len(name_list))):
        path = name_list[i]
        img = cv2.imread(path, 0)
        target = cv2.imread(path.replace(".bmp", "_gt.png"), 0)
        
        get_noise(img, target, path)
        exit()
        
        points = points_list[i]
        points = points[:, [2,1]].astype('int')
        
        gt = np.zeros_like(img).astype('float')
        points = points[points[:,0]<img.shape[0]]
        points = points[points[:,1]<img.shape[1]]
        gt[points[:,0], points[:,1]] = 1.0
        
        vacuum2peaks(img, points, path)