import os
import numpy as np
import time
import torch
from datetime import datetime
import logging
from utils.logger import setlogger
from utils.data_utils import *
from utils.losses import *
from models.unet import get_model
from datasets.tem import get_tem_Data
from tensorboardX import SummaryWriter

root_path = os.getcwd()
model_save_dir = root_path + '\\models'
results_save_dir = root_path + '\\results'


class Trainer():
    def __init__(self, args):
        
        '''Set Logger'''
        sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
        self.save_dir = os.path.join(args.save_dir, sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        setlogger(os.path.join(self.save_dir, 'train.log'))  # set logger
        self.writer = SummaryWriter(self.save_dir)
        
        for k, v in args.__dict__.items():  # save args
            logging.info("{}: {}".format(k, v))
        self.args = args
        self.max_epoch = args.max_epoch
        
        '''Set Device'''
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        '''Load Data'''
        self.train_loader, self.test_loader, self.input_dim, self.output_dim = get_tem_Data(crop_size=args.crop_size,
                                                                                            train_bs=args.batch_size,
                                                                                            is_syn = bool(args.is_syn))
        
        '''Load Model'''
        if args.resume:
            self.model = torch.load(args.resume, map_location=self.device)
        else:
            flag = True if args.pretrained == "True" else False
            self.model = get_model(pretrained=flag)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.metric = cal_score
        self.criterion = torch.nn.MSELoss()
        self.criterion = PixelWiseLoss()
        self.best_score = np.inf
        
    def labels2poitns(self, labels):
        points_list = torch.where(labels==1.0)
        split_idx = [torch.sum(points_list[0]==i).item() for i in range(labels.size(0))]
        points_list = [torch.split(points_list[2], split_idx),
                       torch.split(points_list[3], split_idx)]
        points_list = [torch.cat((hs.unsqueeze(1), ws.unsqueeze(1)), dim=1) for hs, ws in zip(*points_list)]
        return points_list


    def train(self):

        for epoch in range(self.max_epoch):    # epoch
            # Train
            self.model.train()
            train_start = time.time()
            
            epoch_loss = []
            
            for inputs, labels, _ in self.train_loader:    # batch
                inputs = inputs.type(torch.FloatTensor).to(self.device)
                labels = labels.type(torch.FloatTensor).to(self.device)
                points_list = [points.type(torch.FloatTensor).to(self.device) for points in self.labels2poitns(labels)]
                
                out_diams, out_maps = self.model(inputs)
                loss = self.criterion(out_diams, out_maps, labels, points_list)
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                    
                epoch_loss.append(loss.item())
            
            train_loss = np.mean(epoch_loss)
            self.writer.add_scalar('train_loss', train_loss, global_step=epoch)
            logging.info('Epoch {} Train | train loss: {:.3f} | Cost {:.1f}s' .format(epoch, train_loss, time.time()-train_start))
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'training_model.pt'))
            
            # Validation
            if epoch >= self.args.val_start:
                self.model.eval()
                epoch_loss = []
                val_start = time.time()
                epoch_score = EpochScoreRecorder('precision', 'recall', 'cd' ,'jaccard', 'f1')
                for inputs, labels, names in self.test_loader:
                    points_list = torch.where(labels==1.0)
                    split_idx = [torch.sum(points_list[0]==i).item() for i in range(inputs.size(0))]
                    points_list = [torch.split(points_list[2], split_idx),
                                torch.split(points_list[3], split_idx)]
                    points_list = [torch.cat((hs.unsqueeze(1), ws.unsqueeze(1)), dim=1) for hs, ws in zip(*points_list)]
                    
                    inputs = inputs.type(torch.FloatTensor).to(self.device)
                    labels = labels.type(torch.FloatTensor).to(self.device)
                    points_list = [points.type(torch.FloatTensor).to(self.device) for points in points_list]
                    
                    out_diam, out_map = self.model(inputs)
                    loss = self.criterion(out_diam, out_map, labels, points_list) / 16.0
                    epoch_loss.append(loss.item())
                    
                    gray_th = 0.1
                    iou_th = 0.3
                    window_size = 8
                    keep_centers = visualize(inputs, out_diam, out_map, points_list, names, gray_th, iou_th, window_size, results_save_dir, self.device)
                    precision, recall, cd ,jaccard, f1 = self.metric(keep_centers, points_list[0].squeeze())
                    epoch_score.update(precision.item(), recall.item(), cd.item(), jaccard.item(), f1.item())
                
                valid_loss = np.mean(epoch_loss)
                self.writer.add_scalar('valid_loss', valid_loss, global_step=epoch)       
                
                scores = epoch_score.cal_mean()
                self.writer.add_scalar('valid_precision', scores['precision'], global_step=epoch)
                self.writer.add_scalar('valid_recall', scores['recall'], global_step=epoch)
                self.writer.add_scalar('valid_cd', scores['cd'], global_step=epoch)
                self.writer.add_scalar('valid_jaccard', scores['jaccard'], global_step=epoch)
                self.writer.add_scalar('valid_f1', scores['f1'], global_step=epoch)
                    
                logging.info('Epoch {} Score precision: {:.3f} | recall: {:.3f} | chamfer: {:.3f} | jaccard: {:.3f} | f1: {:.3f} | Cost {:.1f}s' .format(epoch,
                                                                                                                                                        scores['precision'],
                                                                                                                                                        scores['recall'],
                                                                                                                                                        scores['cd'],
                                                                                                                                                        scores['jaccard'],
                                                                                                                                                        scores['f1'],
                                                                                                                                                        time.time()-val_start))             
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                    
                if scores['cd'] < self.best_score:
                    self.best_score = scores['cd']
                    logging.info("Epoch {} save best model" .format(epoch))
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pt'))

            self.optimizer = lr_scheduler(epoch, self.optimizer, decay_eff=0.1, decayEpoch=self.args.lr_decay_epoch, is_print=False)
            

if __name__ == '__main__':
    a = np.array([[0, 1],
                  [2, 3],
                  [5, 6],
                  [7, 8]])
    b = np.array([[1, 1],
                  [8, 8]])
    

    