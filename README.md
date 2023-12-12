# AtomID-Net
Project codes for AtomID-Net (https://doi.org/10.1002/smsc.202300031)

# Installation
```
pip install -r requirements.txt
```

# Run UI
Double click `run_ui.bat` on Windows or run:
```
python demo.py
```
in your command line.

# Train
Place your datasets with images and coordinates under
```
./datasets/tem/
```
in the form like those in the demostration data. Fill in the catalog files `./datasets/tem_unet_train.csv` and `./datasets/tem_unet_test.csv` for training and testing data respectively. And run:
```
python train.py
```
Please modify `train.py` for detailed training hyper-parameters. Trained model would be saved under `./models`

# Train with synthetic data
Install extra packages for simulation
```
pip install abtem==1.0.0b34
```
Abtem version later than 1.0.0b34 is not supported in this work. And install `Cupy` according to https://docs.cupy.dev/en/stable/install.html
In this demo
```
pip install cupy-cuda117
```

Place your meta-data `.cif` files under
```
./datasets/cif/
```
Modify parameters `OUT_SHAPE` the heights and widths of output synthetic images and `NUM_PER_CIF` counts of images for each cif file in `./generator.py`. Run it
```
python generator.py
```
to generate synthetic training images and coordiantes.
Fill in the catalog file `./datasets/tem_unet_syn_train.csv` for training data.
Run
```
python train.py --is_syn 1
```
If you want to train with both datasets, merge the two training catalog files and run training depending on how you merged them.


# Inference
Modify the model path in `demo.py` at around 14th Line
```
MODEL_PATH = "{Your Model Path}"
```
and
```
python demo.py
```

# Citation
Please cite us if you are using our codes in your work. 
```
Lin, Yanyu & Yan, Zhangyuan & Tsang, Chi Shing Ben & Wong, Lok Wing & Xiaodong, Zheng & Zheng, Fangyuan & Zhao, Jiong & Chen, Ke. (2023). A Multiscale Deep‐Learning Model for Atom Identification from Low‐Signal‐to‐Noise‐Ratio Transmission Electron Microscopy Images. Small Science. 3. 10.1002/smsc.202300031.
```
