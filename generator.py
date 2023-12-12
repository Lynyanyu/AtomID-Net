import os
from ase.io import read
from ase.build import mx2
import abtem
from abtem.noise import poisson_noise
from abtem.structures import orthogonalize_cell
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import io
import matplotlib.pyplot as plt

device = 'gpu'
cif_path = "./datasets/cif"
out_path = "./datasets/syn"
OUT_SHAPE = 480
NUM_PER_CIF = 20

def getFilelist(path:str, endwith:str='.cif'):
    iter = os.walk(path)
    path_list = []
    for p, d, filelist in iter:
        for name in filelist:
            if name.endswith(endwith):
                path_list.append(os.path.join(p, name))
    
    return path_list 

def get_index(lst, item):
    return [index for (index, value) in enumerate(lst) if value == item]

def format_digit(number:int, max_digit:int=3):
    num_str = str(number)
    while len(num_str) < max_digit:
        num_str = '0'+num_str
    
    return num_str

def spread_atoms(atoms):
    # ================= load and draw
    numbers = np.unique(atoms.numbers)
    numbers.sort()
    atoms_collect = []
    atoms_1 = atoms[atoms.numbers==numbers[-1]]
    atoms_2 = atoms[atoms.numbers==numbers[-2]] if len(numbers)>1 else None
    atoms_3 = atoms[atoms.numbers<numbers[-2]] if len(numbers)>2 else None
    
    atoms_collect.append(atoms_1)
    atoms_collect.append(atoms_2)
    atoms_collect.append(atoms_3)
    
    return atoms_collect
    
path_list = getFilelist(cif_path, '.cif')


##### Set up atomic structure #####

for i, path in enumerate(path_list[0:1]):
    
    name = (path.split("\\")[-1]).split(".")[0]
    print("\n"+name+":")
    for j in range(NUM_PER_CIF):
        
        SAMPLING = np.clip(np.random.normal(0.1, 0.05), 0.05, 0.3)
        atoms = read(path)
        # atoms = mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19, size=(1, 1, 1), vacuum=None)
        atoms = orthogonalize_cell(atoms)
        atoms = atoms.repeat((2, 2, 1))
        atoms.center(vacuum=2, axis=2)
        
        maxnumber = max(atoms.numbers)

        # bias
        bias_range = 2
        bias = np.random.randint(-bias_range, bias_range+1, size=atoms.positions.shape) / 10.0
        new_positions = atoms.positions + bias
        new_positions = np.clip(new_positions, 0, OUT_SHAPE-1)
        atoms.set_positions(new_positions)

        '''dopants'''
        '''vacancy'''
        atom_count = len(atoms)
        del_count = np.random.choice(np.arange(1, 4), 1)[0]
        atom_del = np.random.choice(np.arange(atom_count), atom_count-del_count, replace=False)
        atom_del.sort()
        atoms = atoms[atom_del]
        
        
        atoms1 = atoms.repeat((10, 10, 1))
        #atoms1 = atoms1[atoms1.numbers == maxnumber] # Maintain the main atoms only
        centers = np.concatenate((atoms1.positions, atoms1.numbers[:,None]), axis=-1)
        
        centers[:,:3] = centers[:,:3]/SAMPLING
        keep_indices = (centers[:,0]<OUT_SHAPE) & (centers[:,1]<OUT_SHAPE)
        centers = centers[keep_indices]
        df = pd.DataFrame(centers, columns=['X', 'Y', 'Z', 'N'])
        
        
        '''input'''
        defocus = np.random.randint(50, 100)
        astigmatism = np.random.randint(-5, 10)
        astigmatism_angle = np.random.uniform(0, 4)
        probe = abtem.SMatrix(
                    energy=50e3,
                    interpolation=1,
                    expansion_cutoff=32,
                    semiangle_cutoff=30,
                    defocus=defocus,
                    focal_spread=60,
                    gaussian_spread=0,
                    angular_spread=0,
                    astigmatism=astigmatism,
                    astigmatism_angle=astigmatism_angle,
                    device=device)

        potential = abtem.Potential(atoms,
                        gpts=128,
                        projection='finite',
                        slice_thickness=1,
                        parametrization='kirkland',
                        device=device)

        end = OUT_SHAPE * SAMPLING
        detector = abtem.AnnularDetector(inner=72, outer=190)
        gridscan = abtem.GridScan(start=[0,0], end=[end,end], sampling=SAMPLING)

        try:
            measurement = probe.scan(gridscan, [detector], potential)
            measurement = poisson_noise(measurement, np.random.randint(1, 10) * 10 ** np.random.randint(4, 6))
            # measurement = poisson_noise(measurement, np.random.randint(1, 10) * 10 ** np.random.randint(3, 6))
            measurement.save_as_image(os.path.join(out_path, name + "_" + format_digit(j+1) + "_input.png"))
        except:
            continue
        
        df.to_csv(os.path.join(out_path, name + "_" + format_digit(j+1) + "_note.csv"))
        
        '''target'''
        image_collect = []
        target_save_path = os.path.join(out_path, name + "_" + format_digit(j+1) + "_target.png")
        for idx, atoms_spread in enumerate(spread_atoms(atoms)):
            if atoms_spread is None:
                image_collect.append(np.zeros((OUT_SHAPE, OUT_SHAPE, 1), dtype='uint8'))
            else:
                potential = abtem.Potential(atoms_spread,
                            gpts=128,
                            projection='finite',
                            slice_thickness=1,
                            parametrization='kirkland',
                            device=device)
                
                probe = abtem.SMatrix(
                    energy=20e3,
                    expansion_cutoff=50 + idx * 5,
                    semiangle_cutoff=50 + idx * 5,
                    defocus=0,
                    focal_spread=0,
                    device=device)
        
                
                measurement = probe.scan(gridscan, [detector], potential)
                image = measurement.save_as_image(target_save_path)[:,:,None]
                image_collect.append(image)
        
        image = np.concatenate(image_collect, axis=-1)
        cv2.imwrite(target_save_path, image)
        
