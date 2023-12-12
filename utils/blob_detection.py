from skimage.feature import blob_dog, blob_log, blob_doh
import torch

def LoG(tensor, gray_threshold, max_sigma=30, num_sigma=10, threshold=.1):
    tensor[tensor<gray_threshold] *= 0.0
    image = tensor.squeeze().detach().cpu().numpy()
    blobs_log = blob_log(image, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    blobs_log = torch.from_numpy(blobs_log[:,:2]).type(torch.LongTensor)
    
    return blobs_log

def DoG(tensor, gray_threshold, max_sigma=30, threshold=.1):
    tensor[tensor<gray_threshold] *= 0.0
    image = tensor.squeeze().detach().cpu().numpy()
    blobs_dog = blob_dog(image, max_sigma=max_sigma, threshold=threshold)
    blobs_dog = torch.from_numpy(blobs_dog[:,:2]).type(torch.LongTensor)
    
    return blobs_dog

def DoH(tensor, gray_threshold, max_sigma=30, threshold=.02):
    tensor[tensor<gray_threshold] *= 0.0
    image = tensor.squeeze().detach().cpu().numpy()
    blobs_doh = blob_doh(image, max_sigma=max_sigma, threshold=threshold)
    blobs_doh = torch.from_numpy(blobs_doh[:,:2]).type(torch.LongTensor)
    
    return blobs_doh
