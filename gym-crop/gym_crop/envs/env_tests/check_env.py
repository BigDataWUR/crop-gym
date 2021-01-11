import gym

from stable_baselines3.common.env_checker import check_env

env = gym.make('gym_crop:irrigation-v0', intervention_interval=7)
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)