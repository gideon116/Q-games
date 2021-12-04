import numpy as np
from collections import deque
import cv2
import random
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.lib.io import file_io

from agent import AgentModel
from players import Players
from environment import MyEnv
from wall import Wall
from lava import Lava

# google research generously gave me access to some of their fastest TPUs

random.seed(1)

def learn(episodes, wall_type, lava_type, train=True, tpu_train=False, graph=True, display=True, cloud_storage=False,
          loaded_model=None):
    size = 10
    train = train
    tpu_train = tpu_train

    if tpu_train:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='gideon116')  # gideon116 is the my TPU's name
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))

    # episodes
    episodes = episodes
    limit_steps_per_episode = True
    steps_per_episode = 200

    # epsilons
    if train:
        initial_epsilon = 0.9
        decay_reset = True

    else:
        initial_epsilon = 0.0
        decay_reset = False

    decay_rate = 0.92

    # penalties and reward
    move_penalty = -1
    enemy_penalty = -400
    lava_penalty = -400
    food_reward = 500

    # model stuff
    gamma = 0.99
    loaded_model = loaded_model

    if tpu_train:
        replay_memory_size = 10_000_000
        start_replay_at = 100_000
        mini_batch_size = 50_000
        target_update = 1

    else:
        replay_memory_size = 10_000_000
        start_replay_at = 100
        mini_batch_size = 64
        target_update = 5

    # begin
    agent = AgentModel(size, replay_memory_size, start_replay_at, mini_batch_size, gamma, target_update, loaded_model)
    rewards_list = []
    env = MyEnv(size, enemy_penalty, move_penalty, food_reward, lava_penalty, wall_type=wall_type, lava_type=lava_type)

    # training loop
    for i in range(episodes):
        step = 1
        done = False
        reward_val = 0

        current_space = env.reset()

        while not done:

            rand_eps = np.random.random()
            if rand_eps > initial_epsilon:

                # if we are using the image of the environment to train
                if current_space.shape == (size, size, 1):
                    action = np.argmax(agent.model.predict(np.reshape(current_space, (1, size, size, 1))))

                elif current_space.shape == (100,):
                    action = np.argmax(agent.model.predict(np.reshape(current_space, (1, 100))))

                else:
                    action = None

            else:
                action = random.choice(range(9))

            reward, done, future_space = env.move(action, limit_steps_per_episode, steps_per_episode)

            reward_val += reward

            if train:
                agent.update_replay_memory([current_space, action, reward, future_space, done])
                agent.fit_model(done)

            current_space = future_space
            step += 1

            # display
            if display:
                if train:
                    if i > episodes - 10 or i < -10:
                        show = True
                    else:
                        show = False

                    env.view(show)

                else:
                    show = True
                    env.view(show)

        # append rewards
        rewards_list.append(reward_val)

        # save
        if i == episodes - 1:
            max_reward = max(rewards_list)
            min_reward = min(rewards_list)
            final_file = f'models/FINAL_MAX-({int(max_reward)})_MIN-({int(min_reward)}).h5'
            agent.model.save(final_file)

            # save to google cloud
            if cloud_storage:
                export_path1 = os.path.join('gs://gideon116', final_file)
                with file_io.FileIO(final_file, mode='rb') as input_f:
                    with file_io.FileIO(export_path1, mode='wb+') as output_f:
                        output_f.write(input_f.read())

        if i % 500 == 0 and i != 0:
            ave_reward = sum(rewards_list[i - 49:]) / len(rewards_list[i - 49:])
            file = f'models/Episode=>{i}__Ave=>{int(ave_reward)}).h5'
            agent.model.save(file)

            if cloud_storage:
                export_path2 = os.path.join('gs://gideon116', file)
                with file_io.FileIO(file, mode='rb') as input_f:
                    with file_io.FileIO(export_path2, mode='wb+') as output_f:
                        output_f.write(input_f.read())

        # decay epsilon
        initial_epsilon *= decay_rate

        if decay_reset:
            if initial_epsilon < 0.0001:
                initial_epsilon = 0.2
                print(f'Epsilon has been reset to {initial_epsilon}')

        # verbosity
        print(f"Episode: {i}/{episodes}  Episode Reward: {int(reward_val)}")

    if graph:
        avg = np.convolve(rewards_list, np.ones((100,)) / 100, mode='valid')
        plt.plot([i for i in range(len(avg))], avg)
        plt.show()

"""
wall options = ['special', 'bottom', 'top']
lava options = ['one', 'two', 'three']
"""

learn(100_000, wall_type='bottom', lava_type='one', loaded_model='models/model3.h5', train=True, graph=False,
              cloud_storage=True, display=False)

