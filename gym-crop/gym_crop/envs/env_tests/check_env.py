import gym

from stable_baselines3.common.env_checker import check_env

for env_name in ['gym_crop:fertilization-v0']:
    env = gym.make(env_name, intervention_interval=7)
    # If the environment doesn't follow the interface, an error will be thrown
    check_env(env, warn=True)
    print(f"{env_name} checked")
