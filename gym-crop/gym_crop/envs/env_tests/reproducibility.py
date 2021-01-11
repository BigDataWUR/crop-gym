import numpy as np
import gym

from concurrent.futures import ProcessPoolExecutor


def do(it):
    env = gym.make("gym_crop:irrigation-v0")
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


if __name__ == "__main__":
    maxit = 20
    with ProcessPoolExecutor() as executor:
        results1 = executor.map(do, range(2, maxit))
        results2 = executor.map(do, range(2, maxit))
    for a,b in zip(results1, results2):
        if np.array_equiv(a, b):
            print("equal, yay")
        else:
            print("not equal :(")