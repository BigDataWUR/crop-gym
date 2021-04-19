# CropGym

### Introduction
This is the code base for the paper ["CropGym: a Reinforcement Learning Environment for Crop Management"](https://arxiv.org/abs/2104.04326) by Hiske Overweg, Herman N.C. Berghuijs and Ioannis N. Athanasiadis. 

### Installation
The code has been tested using python 3.8.5. To install all required packages, do the following:

Clone this repository

Install the crop gym environment with the following command
    
```
pip install -e gym_crop/.
```

Install required packages for the training script by running:
    
```
pip install -r requirements.txt
```

### Training an agent
Agents can be trained using the following command:
```
python scripts/training_script.py --name repr --beta 10 --tensorboard /path/to/tensorboard/save/dir --log /path/to/model/save/dir --n_steps=10000
```