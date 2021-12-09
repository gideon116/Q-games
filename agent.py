import numpy as np
from collections import deque
import random

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


class AgentModel:

    def __init__(self, size, replay_memory_size, min_replay_size, minbatch_size, gamma, update_target_every,
                 loadmodel=None, train_by_image=False):
        self.train_by_image = train_by_image
        self.size = size
        self.replay_memory_size = replay_memory_size
        self.min_replay_size = min_replay_size
        self.minbatch_size = minbatch_size
        self.gamma = gamma
        self.update_target_every = update_target_every
        self.loadmodel = loadmodel

        # main model
        self.model = self.make_model()

        # target model
        self.target_model = self.make_model()
        self.target_model.set_weights(self.model.get_weights())

        # we will take batches from this to fit the model
        self.replay_memory = deque(maxlen=self.replay_memory_size)

        self.target_update_counter = 0

    def make_model(self):

        if self.loadmodel is not None:
            network_model = load_model(self.loadmodel)

        elif self.train_by_image:

            network_model = Sequential()

            # input layer
            network_model.add(Conv2D(128, (3, 3), input_shape=(self.size, self.size, 1), activation='relu',
                                     padding="same"))

            # hidden layers
            network_model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))
            network_model.add(MaxPooling2D())
            network_model.add(Dropout(0.1))

            network_model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))
            network_model.add(MaxPooling2D())
            network_model.add(Dropout(0.1))

            network_model.add(Flatten())
            network_model.add(Dense(256, activation='relu'))

            # output layer, 9 IS THE NUMBER OF POSSIBLE MOVES
            network_model.add(Dense(9, activation='linear'))

            # compile
            network_model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        else:
            network_model = Sequential()

            # input layer
            network_model.add(Dense(128, activation="relu", input_shape=(100,)))

            # hidden layers
            network_model.add(Dense(512, activation="relu"))
            network_model.add(Dense(512, activation="relu"))
            network_model.add(Dropout(0.2))
            network_model.add(Dense(512, activation="relu"))
            network_model.add(Dropout(0.2))

            # output layer
            network_model.add(Dense(9, activation='linear'))

            # compile
            network_model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        return network_model

    # transition = obs, action, reward, new obs, done?
    def update_replay_memory(self, movement):
        self.replay_memory.append(movement)

    def fit_model(self, terminal_state):

        if len(self.replay_memory) < self.min_replay_size:
            return

        minibatch = random.sample(self.replay_memory, self.minbatch_size)

        current_state_per_fit = np.array([movement[0] for movement in minibatch])
        current_qs_list = self.model.predict(current_state_per_fit)

        future_state_per_fit = np.array([movement[3] for movement in minibatch])
        future_qs_list = self.target_model.predict(future_state_per_fit)

        features = []
        labels = []

        for index, (current_space_per_mini, action_per_mini, reward_per_mini,
                    future_space_per_mini, done_per_mini) in enumerate(minibatch):

            if not done_per_mini:
                max_future_q = np.max(future_qs_list[index])

                new_q = reward_per_mini + self.gamma * max_future_q

            else:

                new_q = reward_per_mini

            current_qs_per_mini = current_qs_list[index]
            current_qs_per_mini[action_per_mini] = new_q

            features.append(current_space_per_mini)
            labels.append(current_qs_per_mini)

        self.model.fit(np.array(features), np.array(labels), batch_size=self.minbatch_size, verbose=0,
                       shuffle=False)

        # CHANGE THIS
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
