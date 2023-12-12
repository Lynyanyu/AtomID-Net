import torch
import torch.nn as nn
from torchvision import transforms
    
class PixelWiseLoss(nn.Module):
    def __init__(self):
        super(PixelWiseLoss, self).__init__()
        
    def forward(self, diams, maps, targets, points_list):
        loss = 0.0
        for idx, points in enumerate(points_list):
            if len(points) > 1:
                x_dis = (points[:, 0].unsqueeze(0) - points[:, 0].unsqueeze(1)) ** 2
                y_dis = (points[:, 1].unsqueeze(0) - points[:, 1].unsqueeze(1)) ** 2
                dis = torch.sqrt(x_dis + y_dis)
                dis = dis + torch.eye(dis.size(0), device=dis.device) * 1e3   # erase self-distance = 0
                dis = dis.min(1)[0].mean().item()
                
                # diam loss
                try:
                    loss += (diams[idx] - dis) ** 2
                except:
                    pass
                
                # map loss
                gaussianBlur = transforms.GaussianBlur(round(dis/2) * 2 + 1, dis/4.0)
                target = gaussianBlur(targets[idx])
                target /= target.max()
                loss += torch.sum((maps[idx] - target) ** 2) * 0.1
                
            else:
                loss += torch.sum((maps[idx] - 0) ** 2) * 0.1
                

        return loss / max(1, len(points_list))
            
            
            
            