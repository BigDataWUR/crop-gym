import gym
import time
import os
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

beta = 10
name = 'late_v3_long'
env_kwargs = {'intervention_interval':7, 'beta':beta}
env_id = 'gym_crop:fertilization-v0'
num_cpu = 1  # Number of processes to use
# Create the vectorized environment

# kwargs = {
#  'policy_kwargs': {'net_arch': [dict(pi=[256, 256], vf=[256, 256])]}
# }
tensorboard_log = f"/home/hiske/misc/logs/tensorboard/{name}_beta_{beta}"


model = PPO.load(f"/home/hiske/misc/logs/models/{name}_beta_{beta}", tensorboard_log=tensorboard_log)
stats_path = f"/home/hiske/misc/logs/models/{name}_beta_{beta}.pkl"

# Load the saved statistics
env = gym.make(env_id, **env_kwargs)
env = Monitor(env, tensorboard_log)
env = DummyVecEnv([lambda:env])
env = VecNormalize.load(stats_path, env)
env.reset()
model.set_env(env)
print('reloaded')

n_timesteps = 5E5


model.learn(n_timesteps, reset_num_timesteps=False)

log_dir = "/home/hiske/misc/logs/models/"
model.save(os.path.join(log_dir, f"{name}_beta_{beta}_2" ))
stats_path = os.path.join(log_dir, f"{name}_beta_{beta}.pkl")
env.save(stats_path)