from env_gym import PushT
import time,numpy as np

env = PushT()
env.reset()

while True:
    action = np.random.uniform(low=-1, high=1, size=(5,))
    obs, reward, done,_ , info = env.step(action)
    