# SAC with Hydra Configuration

This repository contains a Soft Actor-Critic (SAC) implementation. The implementation includes support for robust policies and various environments.

## Setup

1. Install required dependencies:
   ```
   conda env create -f requirement.yaml -n DRAC
   conda activate DRAC
   ```
2. Make sure all required files are in the correct directory structure:
   ```
   .
   ├── sac_hydra.py
   ├── utils.py
   ├── continuous_cartpole.py
   ├── config
   │   ├── config.yaml
   │   ├── env
   │   │   ├── pendulum.yaml
   │   │   ├── cartpole.yaml
   │   │   ├── lunarlander.yaml
   │   │   ├── humanoid.yaml
   │   │   └── halfcheetah.yaml
   │   ├── robust
   │   │   └── standard.yaml
   │   └── noise
   │       └── medium.yaml
   └── models
       └── SAC_model
   ```

## Running the Code

### Basic Training

Train with default configuration (Pendulum environment):

```
python sac_hydra.py
```

### Changing Environments

Train on a different environment:

```
python sac_hydra.py env=lunarlander
```

Available environments:

* `pendulum` (default)
* `cartpole`
* `lunarlander`
* `humanoid`
* `halfcheetah`

### Using Robust Policy

Train with robust policy:

```
python sac_hydra.py robust=true
```

Or use the standard robust configuration:

```
python sac_hydra.py +robust=standard
```

### Adding Noise

Add noise to the environment:

```
python sac_hydra.py noise=true std=0.1
```

Or use predefined noise configuration:

```
python sac_hydra.py +noise=medium
```

### Changing Hyperparameters

Change any hyperparameter directly from command line:

```
python sac_hydra.py batch_size=512 a_lr=0.001 net_width=128
```

### Evaluation Mode

Run in evaluation mode:

```
python sac_hydra.py eval_model=true
```

### Multirun with Different Configurations

Run multiple configurations in parallel:

```
python sac_hydra.py --multirun env=pendulum,lunarlander robust=true,false
```

This will run 4 experiments with all combinations of environments and robust settings.

## Configuration Structure

### Main Configuration

The main configuration file (`config.yaml`) contains default settings for the algorithm.

### Environment Configurations

Environment-specific configurations are stored in the `config/env/` directory. They define:

* Environment name and index
* Recommended training steps

### Robust and Noise Configurations

* `config/robust/` contains configurations for robust policies
* `config/noise/` contains configurations for adding noise to environments

## Output Structure

When running with Hydra:

* Logs, configs, and outputs are saved to `outputs/YYYY-MM-DD/HH-MM-SS/`
* For multirun experiments, outputs are saved to `multirun/YYYY-MM-DD/HH-MM-SS/`
* TensorBoard logs are in the `tensorboard/` subdirectory
* Models are saved to `models/SAC_model/{ENV_NAME}/`

## Advanced Usage

### Adding New Environments

To add a new environment:

1. Create a YAML file in `config/env/`
2. Define environment parameters (name, index, training steps)
3. Update the environment lists in `sac_hydra.py` if needed

### Creating Custom Configuration Groups

You can create custom configuration groups for different experiments:

1. Create a directory in `config/` (e.g., `config/experiment/`)
2. Add YAML files with different configurations
3. Run with: `python sac_hydra.py +experiment=my_config`
