import argparse
import os
import gym
from torch import nn as nn

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor



def train(name, log_dir, tensorboard_dir, beta, n_steps, resume):
    """
    Train a PPO agent

    Parameters
    ----------
    name: string, model name
    log_dir: directory where the model will be saved
    tensorboard_dir: directory where the tensorboard data will be saved
    beta: float, penalty for fertilization application
    n_steps: int, number of timesteps the agent spends in the environment
    resume: bool, indicates continuation from previous training run
    """
    if not os.path.isdir(log_dir):
        raise ValueError(f'Log directory {log_dir} does not exist')
    if tensorboard_dir and not os.path.isdir(tensorboard_dir):
        raise ValueError(f'Tensorboard directory {tensorboard_dir} does not exist')

    tag = f"{name}_beta_{str(beta).replace('.', '_')}"
    tensorboard_log = os.path.join(tensorboard_dir, tag) if tensorboard_dir else None
    model_path = os.path.join(log_dir, tag)
    stats_path = os.path.join(log_dir, f"{tag}.pkl")

    # setup environment
    env_kwargs = {'intervention_interval':7, 'beta':beta}
    env_id = 'gym_crop:fertilization-v0'
    env = gym.make(env_id, **env_kwargs)
    if tensorboard_log:
        env = Monitor(env, tensorboard_log)
    env = DummyVecEnv([lambda: env])

    hyperparams = {'batch_size': 64,
                   'n_steps': 1024,
                   'learning_rate': 0.0002,
                   'ent_coef': 5.2e-05,
                   'clip_range': 0.3,
                   'n_epochs': 5,
                   'gae_lambda': 0.98,
                   'max_grad_norm': 0.3,
                   'vf_coef': 0.0888,
                   'policy_kwargs': dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                                         activation_fn=nn.Tanh,
                                         ortho_init=False)
                  }

    # train model
    if not resume:
        env = VecNormalize(env, norm_obs=True, norm_reward=False,
                           clip_obs=10., gamma=1,)
        model = PPO('MlpPolicy', env, gamma=1, seed=0, verbose=1, **hyperparams, tensorboard_log=tensorboard_log)
    else:
        model = PPO.load(model_path, tensorboard_log=tensorboard_log)
        env = VecNormalize.load(stats_path, env)
        env.reset()
        model.set_env(env)
    model.learn(n_steps, reset_num_timesteps=not resume)

    model.save(model_path)
    env.save(stats_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="model name", default="", type=str, required=False)
    parser.add_argument("--beta", type=float, default=10., help="penalty for fertilization")
    parser.add_argument("--tensorboard", help="Tensorboard log dir", default="", type=str)
    parser.add_argument("--log", help="directory to save model", default="", type=str)
    parser.add_argument("--n_steps", type=int, default=5e5, help="number of timesteps to train for")
    parser.add_argument("--resume", help="resume training", action='store_true')
    args = parser.parse_args()

    train(args.name, args.log, args.tensorboard, args.beta, args.n_steps, args.resume)
