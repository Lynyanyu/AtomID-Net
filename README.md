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
as the demostration data. Fill in the catalog files `tem_unet_train.csv` and `tem_unet_test.csv` for training and testing data respectively. And run:
```
python train.py
```
Please modify `train.py` for detailed training hyper-parameters.

# Inference
Modify the model path in `demo.py` at around 14th Line
```
MODEL_PATH = "{Your Model Path}"
```
and
```
python demo.py
```
