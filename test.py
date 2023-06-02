import gym
import numpy as np

alpha = 0.5
gamma = 0.99
epsilon = 0.01
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.9
counter = 0
num_steps = 100

cur_Q_tabel = np.load(file="Q_table_taxi.npy")

for episode in range(50):
    env = gym.make('Taxi-v3', render_mode="human")
    state = env.reset()[0]
    done = False
    reward = 0
    
    for cur_step in range(num_steps):

        print("current step is {}".format(cur_step+1))

        cur_action = np.argmax(cur_Q_tabel[state, : ])
        new_state, reward, isdone, _, info = env.step(cur_action)
        reward += reward

        print("current score: {}".format(reward))
        state = new_state
        
        if isdone:
            break

