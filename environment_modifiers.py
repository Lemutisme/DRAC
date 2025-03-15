"""
Environment modifiers for SAC training.
This module provides functions to create environments with various distribution shifts.
"""

import logging
import numpy as np
from numpy.random import normal
import gymnasium as gym

from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from gymnasium.envs.box2d.lunar_lander import LunarLander
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

class CustomPendulumEnv(PendulumEnv):
    def __init__(self, render_mode=None, std=0.0):
        super().__init__(render_mode=render_mode)
        self.noise_std = std
        print(f"Penludum Env with Observation Noise STD={std}.")
        
    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt 
        #############################################################	
        # Universal Normal noise
        # newth += normal(0, self.noise_std)
        
        # Always adverse Normal noise 
        newth += abs(normal(0, self.noise_std)) * np.sign(newth) 
        #############################################################	

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -costs, False, False, {}

register(
    id="CustomPendulum-v1",
    entry_point="environment_modifiers:CustomPendulumEnv",
    max_episode_steps=200,
    kwargs={'std': 0.0}
)

# Parameter Shifted Env
class ParameterShiftedPendulum(PendulumEnv):
    def __init__(self, render_mode=None, g_factor=1.0, m_factor=1.0, l_factor=1.0):
        """
        Pendulum environment with shifted physical parameters.
        
        Args:
            g_factor: Factor to multiply gravity by
            m_factor: Factor to multiply mass by
            l_factor: Factor to multiply length by
        """
        super().__init__(render_mode=render_mode)
        # Modify the physical parameters
        self.g = self.g * g_factor  # Gravity
        self.m = self.m * m_factor  # Mass
        self.l = self.l * l_factor  # Length
        
        print(f"Parameter Shifted Pendulum: g={self.g:.2f}, m={self.m:.2f}, l={self.l:.2f}")

# Register the environments
register(
    id="ParameterShiftedPendulum-v1",
    entry_point="param_shifted_envs:ParameterShiftedPendulum",
    max_episode_steps=200,
)

class ParameterShiftedLunarLander(LunarLander):
    def __init__(self, render_mode=None, gravity_factor=1.0, wind_power=0.0, turbulence_power=0.0):
        """
        LunarLander environment with shifted parameters.

        Args:
            gravity_factor: Factor to multiply gravity by
            wind_power: Constant wind force applied
            turbulence_power: Random turbulence amplitude
        """
        super().__init__(render_mode=render_mode)
        self.gravity_factor = gravity_factor
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power
        print(f"Parameter Shifted LunarLander: gravity_factor={gravity_factor:.2f}, "
                f"wind_power={wind_power:.2f}, turbulence_power={turbulence_power:.2f}")

    def step(self, action):
        # Apply wind as a constant force in the x direction
        if self.wind_power != 0:
            self.lander.ApplyForceToCenter(
                (self.wind_power, 0),
                True
            )

        # Apply random turbulence
        if self.turbulence_power > 0:
            turbulence = np.random.normal(0, self.turbulence_power, size=2)
            self.lander.ApplyForceToCenter(
                (turbulence[0], turbulence[1]),
                True
            )

        # Modify gravity
        original_gravity = self.world.gravity
        self.world.gravity = (original_gravity[0], original_gravity[1] * self.gravity_factor)

        # Call original step function
        obs, reward, terminated, truncated, info = super().step(action)

        # Restore original gravity for consistency
        self.world.gravity = original_gravity
        
        return obs, reward, terminated, truncated, info

register(
    id="ParameterShiftedLunarLander-v1",
    entry_point="param_shifted_envs:ParameterShiftedLunarLander",
    max_episode_steps=1000,
)

# Observation Noise Wrapper
class ObservationNoiseWrapper(gym.Wrapper):
    def __init__(self, env, noise_type='gaussian', noise_level=0.1, noise_freq=1.0):
        """
        Add noise to observations.

        Args:
            env: The environment to wrap
            noise_type: Type of noise ('gaussian', 'uniform', 'saltpepper', 'adversarial')
            noise_level: Standard deviation/magnitude of the noise
            noise_freq: Frequency of applying noise (1.0 = every step)
        """
        super().__init__(env)
        self._max_episode_steps = env._max_episode_steps
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.noise_freq = noise_freq
        self.observation_space = env.observation_space
        print(f"Observation {noise_type} noise with level {noise_level}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._add_noise(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Apply noise based on frequency
        if np.random.random() < self.noise_freq:
            obs = self._add_noise(obs)
        return obs, reward, terminated, truncated, info

    def _add_noise(self, obs):
        if self.noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_level, size=obs.shape)
            return obs + noise
        elif self.noise_type == 'uniform':
            noise = np.random.uniform(-self.noise_level, self.noise_level, size=obs.shape)
            return obs + noise
        elif self.noise_type == 'saltpepper':
            # Salt and pepper noise
            mask = np.random.random(size=obs.shape) < self.noise_level / 2
            obs_copy = obs.copy()
            obs_copy[mask] = np.max(obs)
            mask = np.random.random(size=obs.shape) < self.noise_level / 2
            obs_copy[mask] = np.min(obs)
            return obs_copy
        elif self.noise_type == 'adversarial':
            # Simple adversarial noise - pushes values away from center
            direction = np.sign(obs)
            noise = np.abs(np.random.normal(0, self.noise_level, size=obs.shape)) * direction
            return obs + noise
        else:
            return obs

# Action Perturbation Wrapper
class ActionPerturbationWrapper(gym.Wrapper):
    def __init__(self, env, perturb_type='drop', perturb_prob=0.1, perturb_level=0.1):
        """
        Perturb actions before they are executed.

        Args:
            env: The environment to wrap
            perturb_type: Type of perturbation ('drop', 'noise', 'delay', 'stuck')
            perturb_prob: Probability of applying the perturbation
            perturb_level: Magnitude of the perturbation
        """
        super().__init__(env)
        self.perturb_type = perturb_type
        self.perturb_prob = perturb_prob
        self.perturb_level = perturb_level
        self.last_action = None
        self.stuck_action = None
        self.stuck_count = 0
        self.delay_buffer = []
        self._max_episode_steps = env._max_episode_steps
        print(f"Action perturbation: {perturb_type} with prob {perturb_prob}")

    def reset(self, **kwargs):
        self.last_action = None
        self.stuck_action = None
        self.stuck_count = 0
        self.delay_buffer = []
        return self.env.reset(**kwargs)

    def step(self, action):
        # Process action if perturbation occurs
        if np.random.random() < self.perturb_prob:
            if self.perturb_type == 'drop':
                # Drop action (use last action or zero)
                action = self.last_action if self.last_action is not None else np.zeros_like(action)

            elif self.perturb_type == 'noise':
                # Add random noise to action
                action = action + np.random.normal(0, self.perturb_level, size=action.shape)
                action = np.clip(action, self.action_space.low, self.action_space.high)

            elif self.perturb_type == 'delay':
                # Delay action - apply an old action
                self.delay_buffer.append(action.copy())
                if len(self.delay_buffer) > int(1/self.perturb_level):
                    action = self.delay_buffer.pop(0)
                elif self.last_action is not None:
                    action = self.last_action
                else:
                    action = np.zeros_like(action)

            elif self.perturb_type == 'stuck':
                # Action gets stuck for several timesteps
                if self.stuck_count > 0:
                    action = self.stuck_action
                    self.stuck_count -= 1
                else:
                    self.stuck_action = action.copy()
                    self.stuck_count = int(np.random.exponential(scale=1/self.perturb_level))

        self.last_action = action.copy()
        return self.env.step(action)

# Reward Shift Wrapper
class RewardShiftWrapper(gym.Wrapper):
    def __init__(self, env, shift_type='scale', shift_param=0.9, noise_level=0.1):
        """
        Modify rewards to create distribution shifts.

        Args:
            env: The environment to wrap
            shift_type: Type of shift ('scale', 'delay', 'noise', 'sparse', 'sign_flip')
            shift_param: Parameter for the specific shift type
            noise_level: Standard deviation of noise (for 'noise' type)
        """
        super().__init__(env)
        self.shift_type = shift_type
        self.shift_param = shift_param
        self.noise_level = noise_level
        self.reward_buffer = []
        print(f"Reward shift: {shift_type} with param {shift_param}")

    def reset(self, **kwargs):
        self.reward_buffer = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply reward shift
        if self.shift_type == 'scale':
            # Scale rewards by a factor
            modified_reward = reward * self.shift_param

        elif self.shift_type == 'delay':
            # Delayed reward signal
            self.reward_buffer.append(reward)
            if len(self.reward_buffer) > int(self.shift_param):
                modified_reward = self.reward_buffer.pop(0)
            else:
                modified_reward = 0

        elif self.shift_type == 'noise':
            # Add noise to rewards
            modified_reward = reward + np.random.normal(0, self.noise_level)

        elif self.shift_type == 'sparse':
            # Make rewards more sparse by zeroing out small rewards
            if abs(reward) < self.shift_param:
                modified_reward = 0
            else:
                modified_reward = reward

        elif self.shift_type == 'sign_flip':
            # Randomly flip the sign of the reward with some probability
            if np.random.random() < self.shift_param:
                modified_reward = -reward
            else:
                modified_reward = reward

        else:
            modified_reward = reward

        return obs, modified_reward, terminated, truncated, info

# Transition Perturbation Wrapper
class TransitionPerturbationWrapper(gym.Wrapper):
    """
    This wrapper perturbs the transition dynamics by applying forces,
    teleporting the agent, or other dynamics modifications.
    """
    def __init__(self, env, perturb_type='teleport', perturb_prob=0.05, perturb_level=0.1):
        """
        Args:
            env: The environment to wrap
            perturb_type: Type of perturbation ('teleport', 'random_force', 'friction')
            perturb_prob: Probability of applying the perturbation
            perturb_level: Magnitude of the perturbation
        """
        super().__init__(env)
        self.perturb_type = perturb_type
        self.perturb_prob = perturb_prob
        self.perturb_level = perturb_level
        self.state_dim = env.observation_space.shape[0]
        print(f"Transition perturbation: {perturb_type} with prob {perturb_prob}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Only apply perturbation with some probability
        if np.random.random() < self.perturb_prob:
            if self.perturb_type == 'teleport':
                # Teleport the agent by modifying the observation
                # Note: This doesn't actually change internal state, just what agent observes
                perturbation = np.random.normal(0, self.perturb_level, size=obs.shape)
                obs = obs + perturbation
                obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

            elif self.perturb_type == 'random_force' and hasattr(self.env.unwrapped, 'apply_force'):
                # Apply a random force if the environment supports it
                force = np.random.normal(0, self.perturb_level, size=2)
                self.env.unwrapped.apply_force(force)

            elif self.perturb_type == 'friction' and hasattr(self.env.unwrapped, 'world'):
                # Temporarily change friction (for Box2D environments)
                if hasattr(self.env.unwrapped, 'lander'):
                    for fixture in self.env.unwrapped.lander.fixtures:
                        fixture.friction = fixture.friction * (1 + np.random.normal(0, self.perturb_level))

        return obs, reward, terminated, truncated, info

def create_env_with_mods(env_name, env_config):
    """
    Create environment with modifications based on configuration.
    
    Args:
        env_name (str): Name of the environment to create
        env_config (DictConfig): Configuration for environment modifications

    Returns:
        tuple: (train_env, eval_env) - Training and evaluation environments
    """
    logger.info(f"Creating environment: {env_name}")

    # Create base environments
    train_env = gym.make(env_name)
    eval_env = gym.make(env_name)

    # If no modifications, return base environments
    if not env_config.use_mods:
        logger.info("No environment modifications applied")
        return train_env, eval_env

    # Apply modifications to training environment
    logger.info("Applying modifications to training environment")

    # Apply observation noise if configured
    if env_config.observation_noise.enabled:
        logger.info(f"Adding observation noise: {env_config.observation_noise.type} with level {env_config.observation_noise.level}")
        train_env = ObservationNoiseWrapper(
            train_env,
            noise_type=env_config.observation_noise.type,
            noise_level=env_config.observation_noise.level,
            noise_freq=env_config.observation_noise.frequency
        )

    # Apply action perturbation if configured
    if env_config.action_perturb.enabled:
        logger.info(f"Adding action perturbation: {env_config.action_perturb.type}")
        train_env = ActionPerturbationWrapper(
            train_env,
            perturb_type=env_config.action_perturb.type,
            perturb_prob=env_config.action_perturb.probability,
            perturb_level=env_config.action_perturb.level
        )

    # Apply reward shift if configured
    if env_config.reward_shift.enabled:
        logger.info(f"Adding reward shift: {env_config.reward_shift.type}")
        train_env = RewardShiftWrapper(
            train_env,
            shift_type=env_config.reward_shift.type,
            shift_param=env_config.reward_shift.param,
            noise_level=env_config.reward_shift.noise_level
        )

    # Apply transition perturbation if configured
    if env_config.transition_perturb.enabled:
        logger.info(f"Adding transition perturbation: {env_config.transition_perturb.type}")
        train_env = TransitionPerturbationWrapper(
            train_env,
            perturb_type=env_config.transition_perturb.type,
            perturb_prob=env_config.transition_perturb.probability,
            perturb_level=env_config.transition_perturb.level
        )

    # Create evaluation environment with potentially different modifications
    logger.info("Applying modifications to evaluation environment")

    # Decide evaluation environment configuration based on settings
    if env_config.eval.use_modified:
        logger.info("Using modified environment for evaluation")

        # Create evaluation environment with same modifications but potentially different parameters
        if env_config.observation_noise.enabled:
            eval_noise_level = env_config.observation_noise.level
            if env_config.eval.scale_noise:
                eval_noise_level *= env_config.eval.noise_scale_factor

            eval_env = ObservationNoiseWrapper(
                eval_env,
                noise_type=env_config.observation_noise.type,
                noise_level=eval_noise_level,
                noise_freq=env_config.observation_noise.frequency
            )
            logger.info(f"Evaluation environment using observation noise level: {eval_noise_level}")

        # Apply other modifications to eval env if needed...
        if env_config.action_perturb.enabled and env_config.eval.include_action_perturb:
            eval_env = ActionPerturbationWrapper(
                eval_env,
                perturb_type=env_config.action_perturb.type,
                perturb_prob=env_config.action_perturb.probability,
                perturb_level=env_config.action_perturb.level
            )

        if env_config.reward_shift.enabled and env_config.eval.include_reward_shift:
            eval_env = RewardShiftWrapper(
                eval_env,
                shift_type=env_config.reward_shift.type,
                shift_param=env_config.reward_shift.param,
                noise_level=env_config.reward_shift.noise_level
            )

        if env_config.transition_perturb.enabled and env_config.eval.include_transition_perturb:
            eval_env = TransitionPerturbationWrapper(
                eval_env,
                perturb_type=env_config.transition_perturb.type,
                perturb_prob=env_config.transition_perturb.probability,
                perturb_level=env_config.transition_perturb.level
            )
    else:
        logger.info("Using standard environment for evaluation")

    return train_env, eval_env
