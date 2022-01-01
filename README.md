# SuperMario_DeepQ
A Deep Q RL agent based on "Playing Atari with Deep Reinforcement Learning" By Mnih et al to solve Super Mario Game 

# Setup
In a python 3.7+ environment, run:

```
conda create --name mario python=3.9
conda activate mario
conda install pytorch
pip install -r requirements.txt
```

# Training
To train the model, run:

```
python main_DQ.py
```

The training hyperparameters are stored in ./config/config_DeepQ.yaml

# Run Pretrained Model
A pretrained RL agent model in available in ./checkpoints. To play mario with this model, run:

```
python play_DQ.py
```
