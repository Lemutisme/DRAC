#!/bin/bash
# Example script to demonstrate different Hydra configurations

# Create necessary directories if they don't exist
mkdir -p config/env
mkdir -p config/robust
mkdir -p config/noise
mkdir -p models/SAC_model

# Run standard training on Pendulum environment
echo "Running default configuration (Pendulum)..."
python sac_hydra.py

# Run with a different environment
echo "Running LunarLander environment..."
python sac_hydra.py env=lunarlander

# Run with robust policy
echo "Running with robust policy..."
python sac_hydra.py env=pendulum robust=true delta=0.1

# Run with noise
echo "Running with noise..."
python sac_hydra.py env=pendulum noise=true std=0.1

# Run evaluation only
echo "Running evaluation mode..."
python sac_hydra.py eval_model=true

# Change hyperparameters
echo "Running with custom hyperparameters..."
python sac_hydra.py env=pendulum batch_size=512 a_lr=0.001

# Override output directory
echo "Running with custom output directory..."
python sac_hydra.py hydra.run.dir=./custom_output

# Run a small multirun experiment 
echo "Running multirun with different environments..."
python sac_hydra.py --multirun env=pendulum,cartpole max_train_steps=10000