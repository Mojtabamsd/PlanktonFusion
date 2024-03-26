# Edge Plankton Classification
Realtime plankton classification


## Prerequisites
- Linux or macOS or windows
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Getting started
- Clone this repo:
```bash
git clone https://github.com/Mojtabamsd/PlanktonFusion PF
cd PF
```

- Install [PyTorch](http://pytorch.org) and other dependencies (e.g., torchvision).

  For pip users, please type the command `pip install -r requirements.txt`.

  For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.


### Train a model
```bash
python main.py training -c ./configs/config.yaml -i 'sampling_path' -o 'output_path'
```

### Test a model
```bash
python main.py prediction -c ./configs/config.yaml -i 'training_path' -o 'output_path'
```