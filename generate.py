# from abtem import __version__
# print('current version:', __version__)

import os
from ase.io import read as io_read
from ase import Atoms
import cv2
from tqdm import tqdm
import numpy as np
import csv

def getFilelist(path:str, endwith:str='.cif'):
    iter = os.walk(path)
    path_list = []
    for p, d, filelist in iter:
        for name in filelist:
            if name.endswith(endwith):
                path_list.append(os.path.join(p, name))
    
    return path_list 


class SyntheticTool():
    '''
    Warning: only positions and atomic numbers are considered, other attributes will
             be twisted during transformations 
    '''
    def __init__(self, size):
        self.size = size
        self.img_center = np.array([size[0]/2, size[1]/2, 0])
        pass
        
    def _pull(self, atoms:Atoms, targets:Atoms, pull_in:bool=True, eff:float=0.3):
        target_centers = targets.get_positions()
        atom_centers = atoms.get_positions()
        
        vectors = np.concatenate(((target_centers[:,0][:,None] - atom_centers[:,0][None,:])[:,:,None], 
                                  (target_centers[:,1][:,None] - atom_centers[:,1][None,:])[:,:,None]),
                                  axis=-1)
        dis = np.sum(vectors ** 2, axis=-1)
        moves = np.sqrt(dis) * np.exp(- dis / (2 * np.sort(dis, axis=1)[:,1,None])) * eff
        vectors = vectors * (moves / (np.sqrt(dis)+1e-6))[:,:,None]
        vectors = np.sum(vectors, axis=0)
        vectors = np.concatenate((vectors, np.zeros((vectors.shape[0],1))), axis=1)
        vectors = vectors if pull_in else -vectors
        
        atoms.set_positions(atoms.get_positions() + vectors)
        atoms = self._filter_inside(atoms)
        return atoms
    
    def _filter_inside(self, atoms:Atoms, size=None):
        if size is None:
            size = self.size
        atoms = atoms[atoms.get_positions()[:,0]>=1]
        atoms = atoms[atoms.get_positions()[:,0]>=1]
        atoms = atoms[atoms.get_positions()[:,0]<=self.size[0]-1]
        atoms = atoms[atoms.get_positions()[:,1]<=self.size[1]-1]
        return atoms
    
    def _generate_low_number_atomic_centers(self, atoms:Atoms, neighbors:int=3, rate:float=0.5):
        if neighbors<1 or rate == 0:
            return atoms
        
        dis= atoms.get_all_distances()
        indices = np.argsort(dis, axis=1)[:,1:neighbors+1]
        centers = atoms.get_positions()
        low_number_centers = []
        for n in range(neighbors):
            low_number_centers.append((centers + centers[indices[:,n]])/2)
        
        low_number_centers = np.concatenate(low_number_centers, axis=0)
        shuffle_indices = np.arange(len(low_number_centers))
        np.random.shuffle(shuffle_indices)
        low_number_centers = low_number_centers[shuffle_indices[:round(len(low_number_centers)*rate)],:]
        return low_number_centers
    
    def read_atoms(self, path:str, max_only:bool=True):
        atoms = io_read(path)
        atoms.center(vacuum=2, axis=2)
        if max_only:
            atoms = atoms[atoms.numbers == max(atoms.numbers)]
            
        return atoms
    
    def expand_atoms(self, atoms:Atoms, expand:float=5.0, repetitions=(200, 200, 1)):
        atoms *= repetitions
        atoms.set_positions(atoms.get_positions() * expand)
        atoms = self._filter_inside(atoms)
        return atoms

    def add_vacancy(self, atoms:Atoms, rate:float=0.05, eff:float=0.3):
        if rate == 0 or eff == 0:
            return atoms

        indices = np.arange(len(atoms.get_positions()))
        np.random.shuffle(indices)
        vacancies = atoms[indices[round(len(indices)*(1-rate)):]].copy()
        
        # vacancies will be given atomic numbers smaller than 0
        atomic_numbers = atoms.get_atomic_numbers()
        atomic_numbers[indices[round(len(indices)*(1-rate)):]] = np.array([-1]*len(vacancies))
        atoms.set_atomic_numbers(atomic_numbers)
        atoms = self._pull(atoms, vacancies, pull_in=True, eff=eff)
        
        return atoms

    def add_dopants(self, atoms:Atoms, rate:float=0.01, add_number:int=1, eff:float=0.3):
        if rate == 0 or add_number == 0 or eff == 0:
            return atoms
        
        indices = np.arange(len(atoms.get_positions()))
        np.random.shuffle(indices)
        dopants = atoms[indices[round(len(indices)*(1-rate)):]].copy()
        
        atomic_numbers = atoms.get_atomic_numbers()
        atomic_numbers[indices[round(len(indices)*(1-rate)):]] = dopants.get_atomic_numbers() + add_number
        atoms.set_atomic_numbers(atomic_numbers)
        
        is_pull_in = False if add_number>0 else True
        
        atoms = self._pull(atoms, dopants, pull_in=is_pull_in, eff=eff)
        
        return atoms
    
    def add_hole(self, atoms:Atoms, index=None, radius:float=10.0, eff:float=1e-3):
        if radius == 0:
            return atoms
        if index is None:
            index = np.random.randint(len(atoms.get_positions()))
        
        hole_center = atoms[index].get('position') 
        dis = (atoms.get_positions()[:,0] - hole_center[0]) ** 2 + (atoms.get_positions()[:,1] - hole_center[1]) ** 2
        hole_atoms = atoms[dis<radius**2]
        atoms = atoms[dis>=radius**2]
        
        atoms = self._pull(atoms, hole_atoms[:10], pull_in=True, eff=eff)
        return atoms
    
    def add_bias(self, atoms:Atoms, bias:int=1):
        bias = 1.0 if bias>0 else bias
        bias = np.random.randint(-bias, bias+1, size=atoms.get_positions().shape)
        atoms.set_positions(atoms.get_positions() + bias)
        atoms = self._filter_inside(atoms)
        return atoms
    
    def add_background(self, img, radius_ratio:float=1/16, bright_ratio:float=0.25, dark_ratio:float=0.25, iteration:int=10):
        def get_background(img):
            h, w = img.shape[-2:]
            t1 = np.linspace(1, h, h)
            t2 = np.linspace(1, w, w)
            x, y = np.meshgrid(t1,t2)
            x -= np.mean(x.flatten()) + (np.random.random() - 0.5) * h
            y -= np.mean(y.flatten()) + (np.random.random() - 0.5) * w
            d = np.sqrt(x**2+y**2)
            sig = (np.random.random()+1) * (h+w) * radius_ratio
            mesh = np.exp(-d**2/(2*sig**2))
            return mesh
        
        noise = np.zeros(img.shape)
        for _ in range(iteration):
            noise += (get_background(img) * bright_ratio - get_background(img) * dark_ratio) * img.max()
            
        img = img + noise
        return img
    
    def seperate(self, atoms:Atoms):
        atomic_numbers = atoms.get_atomic_numbers()
        max_number = np.max(atomic_numbers)
        
        basics = atoms[atomic_numbers>0].copy()
        # basics = basics[basics.get_atomic_numbers()<max_number]
        vacancies = atoms[atomic_numbers<0].copy()
        dopants = atoms[atomic_numbers==max_number].copy()
        
        return basics, vacancies, dopants
    
    def check_visibility(self, atoms:Atoms, image, threshold:float=0.02):
        centers = np.round(atoms.get_positions()).astype('int')
        image_blured = cv2.blur(image, (5,5))
        center_gray = image_blured[centers[:,0], centers[:,1]]
        atoms = atoms[center_gray>image.max()*threshold]
        
        return atoms
    
    def write_atoms_to_csv(self, atoms:Atoms, save_path:str):
        with open(save_path, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Peak #', 'X', 'Y'])
            for idx, center in enumerate(atoms.get_positions()):
                csv_writer.writerow([str(idx+1)] + list(center[[1,0]]))
    
    def generate(self, atoms:Atoms, save_path:str, sigma=[4.0, 3.0], noise_rate:float=5e-3,
                 substrate_mag:float=0.5, background_ratio=[0.5, 0.5], visibility_lower:float=0.1):

        # generate image
        atoms = atoms[atoms.get_atomic_numbers()>0]  # rid of vacancy
        centers = atoms.get_positions()
        centers = np.round(centers).astype('int')
        atomic_numbers = atoms.get_atomic_numbers()
        
        # draw atoms
        image = np.zeros(self.size)
        image[centers[:,0], centers[:,1]] = 1.0
        image[centers[atomic_numbers==np.max(atomic_numbers),0], centers[atomic_numbers==np.max(atomic_numbers),1]] = 2.0
        
        # add fake substrate
        low_number_centers = self._generate_low_number_atomic_centers(atoms, rate=0.8)
        low_number_centers = np.round(low_number_centers).astype('int')
        image[low_number_centers[:,0], low_number_centers[:,1]] = substrate_mag
        
        # blur
        try:
            sigmaX, sigmaY = sigma
        except:
            sigmaX, sigmaY = sigma, sigma
        image = cv2.GaussianBlur(image, (51,51), sigmaX=sigmaX, sigmaY=sigmaY)

        # add basic noise
        gassian_noise = np.random.normal(0, 1.0, image.shape)
        image += gassian_noise * noise_rate
        
        # add bright strip
        mask = np.ones(image.shape)
        mask[:,np.random.randint(mask.shape[1]//4, mask.shape[1]//2):np.random.randint(mask.shape[1]//2, 3*mask.shape[1]//4)] = np.random.randint(15,31)/10.0
        mask = cv2.blur(mask, (21,21))
        image *= mask 
        
        # add background noise
        image /= image.max()
        image = self.add_background(image, 1/32, background_ratio[0], background_ratio[1], 100)
        
        # output image
        image *= 255.0
        image = np.clip(image, 0, 255)
        cv2.imwrite(save_path, image.astype('uint8'))
        
        return image
        
    def generate_gt_mask(self, atoms:Atoms, save_path:str, show_basic:bool=True, show_vacancy:bool=True, show_dopant:bool=True):
        basics, vacancies, dopants = self.seperate(atoms)
        mask = np.zeros((1024, 1024, 3))
        if show_basic:
            for b in basics.get_positions():
                cv2.circle(mask, (round(b[1]), round(b[0])), 3, (0, 255, 97), -1)
        if show_vacancy:
            for v in vacancies.get_positions():
                cv2.circle(mask, (round(v[1]), round(v[0])), 2, (205, 0, 97), -1)
        if show_dopant:
            for d in dopants.get_positions():
                cv2.circle(mask, (round(d[1]), round(d[0])), 6, (97, 155, 155), -1)
        
        mask = cv2.GaussianBlur(mask, (21,21), 2.0)
        mask *= 255.0 / mask.max()
        mask = np.clip(mask, 0, 255)
        cv2.imwrite(save_path, mask.astype('uint8'))
        
        return mask

if __name__ == '__main__':
    root_path = '.'
    save_path = root_path + './datasets/syn'
    cif_path = root_path + './datasets/cif'
    cif_path_list = getFilelist(cif_path, 'cif')
    
    NUM = 8
    
    # output image size
    size = (1024, 1024)
    STool = SyntheticTool(size)
    
    # create folder
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    for cif_file in tqdm(cif_path_list):
        for num in range(NUM):
            name = os.path.basename(cif_file).replace(".cif", "_"+str(num))
            atoms = STool.read_atoms(cif_file)
            atoms = STool.expand_atoms(atoms, expand=np.random.randint(45, 75)/10)
            atoms = STool.add_vacancy(atoms, rate=0.05, eff=0.4)
            atoms = STool.add_dopants(atoms, rate=0.02, eff=0.3)
            
            atoms = STool.add_hole(atoms, radius=np.random.randint(15, 25), eff=0.1)
            atoms = STool.add_hole(atoms, radius=np.random.randint(25, 45), eff=0.1)
            atoms = STool.add_hole(atoms, radius=np.random.randint(45, 75), eff=0.1)
            atoms = STool.add_bias(atoms, bias=1)
            
            image = STool.generate(atoms, os.path.join(save_path, name+"_syn.bmp"),
                                sigma = [2.5, 2.5],
                                noise_rate = np.random.randint(5, 15) / 1000.0,
                                substrate_mag = 0.2,
                                background_ratio = [0.1, 0.05])
            
            atoms = STool.check_visibility(atoms, image)
            basics, vacancies, dopants = STool.seperate(atoms)
            mask = STool.generate_gt_mask(atoms, os.path.join(save_path, name+"_syn_gt.png"))
            STool.write_atoms_to_csv(basics, os.path.join(save_path, name+"_syn.csv"))
            #STool.write_atoms_to_csv(vacancies, os.path.join(save_path, name+"_syn_vacancy.csv"))
            #STool.write_atoms_to_csv(dopants, os.path.join(save_path, name+"_syn_dopant.csv"))