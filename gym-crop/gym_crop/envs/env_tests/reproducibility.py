import numpy as np
import gym

from concurrent.futures import ProcessPoolExecutor


def do(it, env_name):
    env = gym.make(env_name)
    env.seed(it)
    env.action_space.seed(it)
    env.reset()
    observations = []
    for i in range(3):
        while True:
            action = env.action_space.sample()
            ob, reward, done, _ = env.step(action)
            observations.append(ob)
            if done:
                break
    return observations

def do_wrapper(args):
    do(*args)


if __name__ == "__main__":
    maxit = 20
    for env_name in ['gym_crop:fertilization-v0']:
        iterations = range(2, maxit)
        args = {(iteration, env_name) for iteration in iterations}
        with ProcessPoolExecutor() as executor:
            results1 = executor.map(do_wrapper, args)
            results2 = executor.map(do_wrapper, args)
        for a,b in zip(results1, results2):
            if not np.array_equiv(a, b):
                raise ValueError(f"Instances of {env_name} not equal")
        print(f"{env_name} all good!")
