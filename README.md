# Dependencies

```python
conda env create -f requirement.yaml -n DRAC
```

## How to use my code

First, replace the `model` to your wanted algorithm

### Train from scratch

```bash
python model.py
```

where the default enviroment is 'Pendulum'.

### Play with trained model

```bash
python model.py --EnvIdex 0 --render True --Loadmodel True --ModelIdex 10
```

which will render the 'Pendulum'.

### Change Enviroment

If you want to train on different enviroments, just run

```bash
python model.py --EnvIdex 1
```

The ``--EnvIdex`` can be set to be 0~5, where

```bash
'--EnvIdex 0' for 'Pendulum-v1'  
'--EnvIdex 1' for 'LunarLanderContinuous-v2'  
'--EnvIdex 2' for 'Humanoid-v4'  
'--EnvIdex 3' for 'HalfCheetah-v4'  
'--EnvIdex 4' for 'BipedalWalker-v3'  
'--EnvIdex 5' for 'BipedalWalkerHardcore-v3' 
```

### Visualize the training curve

You can use the [tensorboard](https://pytorch.org/docs/stable/tensorboard.html) to record anv visualize the training curve.

- Record (the training curves will be saved at '**\runs**'):

```bash
python main.py --write True
```

- Visualization:

```bash
tensorboard --logdir runs
```

### Reference

[Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)
