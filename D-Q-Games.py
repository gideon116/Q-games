import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from collections import deque
import cv2
import random
import matplotlib.pyplot as plt
from matplotlib import style
import tensorflow as tf
from tensorflow.python.lib.io import file_io

# google research generously gave me access to some of their fastest TPUs
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='google-research-gift')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

random.seed(1)


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
            pass

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


class Players:

    def __init__(self, size, restriction=None):

        self.size = size
        self.restriction = restriction

        if self.restriction is not None:
            retry = True

            while retry:
                potential_x = random.choice(range(self.size))
                potential_y = random.choice(range(self.size))

                for ip in self.restriction:

                    # if rand number lands on restriction
                    if potential_x == ip[0] and potential_y == ip[1]:
                        retry = True
                        break

                    else:
                        retry = False

                self.x = potential_x
                self.y = potential_y

        else:
            self.x = random.choice(range(self.size))
            self.y = random.choice(range(self.size))

    def pos(self):

        position = (self.x, self.y)
        return position

    def movement(self, choice):

        if choice == 0:
            self.new_pos(x=1, y=1)
        elif choice == 1:
            self.new_pos(x=1, y=-1)

        elif choice == 2:
            self.new_pos(x=-1, y=-1)
        elif choice == 3:
            self.new_pos(x=-1, y=1)

        elif choice == 4:
            self.new_pos(x=0, y=0)
        elif choice == 5:
            self.new_pos(x=1, y=0)

        elif choice == 6:
            self.new_pos(x=0, y=1)
        elif choice == 7:
            self.new_pos(x=-1, y=0)

        elif choice == 8:
            self.new_pos(x=0, y=-1)

    def new_pos(self, x=None, y=None):

        if self.restriction is not None:

            # if x (and so y) is not specified, choose random numbers avoiding the restriction
            if x is None:
                retry = True
                potential_x = None
                potential_y = None

                while retry:
                    potential_x = self.x + random.choice([-1, 0, 1])  # either -1 or 1
                    potential_y = self.x + random.choice([-1, 0, 1])

                    # if it tries to go out of bounds
                    if potential_x < 0:
                        potential_x = 0
                    elif potential_x > self.size - 1:
                        potential_x = self.size - 1

                    # if it tries to go out of bounds
                    if potential_y < 0:
                        potential_y = 0
                    elif potential_y > self.size - 1:
                        potential_y = self.size - 1

                    for ix in self.restriction:

                        # if it tries to pass some restriction
                        if potential_x == ix[0] and potential_y == ix[1]:
                            retry = True
                            break

                        else:
                            retry = False

                self.x = potential_x
                self.y = potential_y

            else:
                on_restriction = False

                potential_x = self.x + x
                potential_y = self.y + y

                # if it tries to go out of bounds
                if potential_x < 0:
                    potential_x = 0
                elif potential_x > self.size - 1:
                    potential_x = self.size - 1

                # if it tries to go out of bounds
                if potential_y < 0:
                    potential_y = 0
                elif potential_y > self.size - 1:
                    potential_y = self.size - 1

                for ixx in self.restriction:

                    # if it tries to pass some restriction
                    if potential_x == ixx[0] and potential_y == ixx[1]:
                        on_restriction = True
                        break

                    else:
                        on_restriction = False

                if on_restriction:
                    self.x += 0
                    self.y += 0
                else:
                    self.x = potential_x
                    self.y = potential_y

        else:
            if x is None:
                self.x += random.choice([-1, 0, 1])
            else:
                self.x += x

            if y is None:
                self.y += random.choice([-1, 0, 1])  # either -1 or 1
            else:
                self.y += y

            # if it tries to go out of bounds
            if self.x < 0:
                self.x = 0
            elif self.x > self.size - 1:
                self.x = self.size - 1

            # if it tries to go out of bounds
            if self.y < 0:
                self.y = 0
            elif self.y > self.size - 1:
                self.y = self.size - 1


class Wall:

    def __init__(self, env_size, area=None):

        self.area_covered = area
        self.env_size = env_size
        self.pos_range = None

    def location(self, location):

        if location == 'right':

            # NOT USED FOR NOW
            self.pos_range = [(0, 0), (1, 0), (2, 0), (3, 0)]

        elif location == 'left':

            # NOT USED FOR NOW
            self.pos_range = [(0, self.env_size - 1), (1, self.env_size - 1), (2, self.env_size - 1),
                              (3, self.env_size - 1)]

        elif location == 'top':

            # NO 4
            self.pos_range = [(3, 2), (3, 3), (3, 4), (3, 5)]

        elif location == 'bottom':

            # NO 2, 3, 4, 5
            self.pos_range = [(7, 0), (7, 1), (7, 6), (7, 7), (7, 8), (7, 9)]

        elif location == 'middle':

            # NO 6
            self.pos_range = [(self.env_size//2, 0), (self.env_size//2, 1), (self.env_size//2, 2),
                              (self.env_size//2, 3), (self.env_size//2, 4), (self.env_size//2, 5),
                              (self.env_size//2, 7), (self.env_size//2, 8), (self.env_size//2, 9)]


class Lava:

    #  THE Q-TABLE IS DESIGNED FOR ONLY 4 LAVA BLOCKS. IF YOU WANT TO CHANGE THE SIZE OF THE LAVA, REDESIGN THE Q-TABLE
    def __init__(self, env_size):
        self.env_size = env_size
        self.pos_range = None

    def number_of_lava(self, location):
        if location == 'one' or location == 'One' or location == 'ONE':
            self.pos_range = [(1, 2), (1, 3), (2, 2), (2, 3)]

        elif location == 'two' or location == 'Two' or location == 'TWO':
            self.pos_range = [(1, 2), (1, 3), (2, 2), (2, 3), (6, 2), (6, 3), (7, 2), (7, 3)]

        elif location == 'three' or location == 'Three' or location == 'THREE':
            self.pos_range = [(1, 2), (1, 3), (2, 2), (2, 3), (6, 2), (6, 3), (7, 2), (7, 3),
                              (1, 8), (1, 9), (2, 8), (2, 9)]


class MyEnv:

    def __init__(self, size, enemy_p, move_p, food_r, lava_p, train_by_image=False):

        # training mode
        self.train_by_image = train_by_image

        # environment size
        self.size = size

        # penalties
        self.enemy_p = enemy_p
        self.move_p = move_p
        self.lava_p = lava_p
        self.food_r = food_r

        self.reward = 0

        # where the wall is
        self.wall = Wall(env_size=self.size)
        self.wall.location('bottom')

        # the lava
        self.lava = Lava(env_size=self.size)
        self.lava.number_of_lava('one')

        # where food, enemy, and guy are player objects. They have the wall's location as a restriction
        self.enemy = Players(self.size, restriction=self.wall.pos_range)
        self.food = Players(self.size, restriction=self.wall.pos_range)
        self.guy = Players(self.size, restriction=self.wall.pos_range)

        self.sub_episode = 0

    def reset(self):

        # reset sub episode count
        self.sub_episode = 0

        # reset players
        self.enemy = Players(self.size, restriction=self.wall.pos_range)
        self.food = Players(self.size, restriction=self.wall.pos_range)
        self.guy = Players(self.size, restriction=self.wall.pos_range)

        # if we are using the image of the environment to train
        if self.train_by_image:

            current_space_per_reset = np.zeros((self.size, self.size, 1))
            current_space_per_reset[self.guy.x][self.guy.y] = [1]
            current_space_per_reset[self.food.x][self.food.y] = [0]
            current_space_per_reset[self.enemy.x][self.enemy.y] = [0.5]

            for wall in self.wall.pos_range:
                current_space_per_reset[wall[0]][wall[1]] = [0.25]

            for lava in self.lava.pos_range:
                current_space_per_reset[lava[0]][lava[1]] = [0.75]

        else:

            current_space_per_reset = np.array([self.guy.x - self.food.x, self.guy.y - self.food.y,
                                                self.guy.x - self.enemy.x, self.guy.y - self.enemy.y])

            # get the distance from each wall block
            for wall in self.wall.pos_range:
                current_space_per_reset = np.append(current_space_per_reset, self.guy.x - wall[0])
                current_space_per_reset = np.append(current_space_per_reset, self.guy.y - wall[1])

            # get the distance from each lava block
            for lava in self.lava.pos_range:
                current_space_per_reset = np.append(current_space_per_reset, self.guy.x - lava[0])
                current_space_per_reset = np.append(current_space_per_reset, self.guy.y - lava[1])

            # additional space if we want to have a total of 50 things
            for index in range(50 - len(current_space_per_reset)//2):
                current_space_per_reset = np.append(current_space_per_reset, 0)
                current_space_per_reset = np.append(current_space_per_reset, 0)

        return current_space_per_reset

    def move(self, action_passed, limit_number_of_episodes=True, limit_number=None):
        self.sub_episode += 1
        done_per_move = False

        if limit_number_of_episodes:
            if limit_number is not None:
                if self.sub_episode > limit_number:
                    done_per_move = True

            else:
                if self.sub_episode > 500:
                    done_per_move = True

        # when players move
        self.guy.movement(action_passed)
        # self.food.new_pos()
        self.enemy.new_pos()

        if self.guy.x == self.enemy.x and self.guy.y == self.enemy.y:
            done_per_move = True
            reward_per_move = self.enemy_p

        elif self.guy.x == self.food.x and self.guy.y == self.food.y:
            done_per_move = True
            reward_per_move = self.food_r

        else:

            # the penalty for each move is the initial move penalty multiplied by the distance from the food and stuff
            reward_per_move = self.move_p * ((self.guy.x - self.food.x) ** 2 + (self.guy.y - self.food.y) ** 2)

            for pos in self.lava.pos_range:

                # if the player gets on one of the lava blocks, it dies
                if (self.guy.x, self.guy.y) == pos:
                    reward_per_move = self.lava_p
                    done_per_move = True
                    break

                # if the player gets close to the lava, it gets hurt
                else:

                    # the penalty increases exponentially if the player is close to the lava
                    reward_per_move += (0.01 * self.lava_p) // ((np.sqrt((self.guy.x - pos[0])**2 +
                                                                         (self.guy.y - pos[1])**2))**2)

        self.reward = reward_per_move

        # if we are using the image of the environment to train
        if self.train_by_image:

            future_space_per_move = np.zeros((self.size, self.size, 1))
            future_space_per_move[self.guy.x][self.guy.y] = [1]
            future_space_per_move[self.food.x][self.food.y] = [0]
            future_space_per_move[self.enemy.x][self.enemy.y] = [0.5]

            for wall in self.wall.pos_range:
                future_space_per_move[wall[0]][wall[1]] = [0.25]

            for lava in self.lava.pos_range:
                future_space_per_move[lava[0]][lava[1]] = [0.75]

        else:

            future_space_per_move = np.array([self.guy.x - self.food.x, self.guy.y - self.food.y,
                                              self.guy.x - self.enemy.x, self.guy.y - self.enemy.y])

            # get the distance from each wall block
            for pos in self.wall.pos_range:
                future_space_per_move = np.append(future_space_per_move, self.guy.x - pos[0])
                future_space_per_move = np.append(future_space_per_move, self.guy.y - pos[1])

            # get the distance from each lava block
            for pos in self.lava.pos_range:
                future_space_per_move = np.append(future_space_per_move, self.guy.x - pos[0])
                future_space_per_move = np.append(future_space_per_move, self.guy.y - pos[1])

            # additional space if we want to have a total of 50 things
            for index in range(50 - len(future_space_per_move)//2):
                future_space_per_move = np.append(future_space_per_move, 0)
                future_space_per_move = np.append(future_space_per_move, 0)

        return reward_per_move, done_per_move, future_space_per_move

    def view(self, show_img=False):

        resize = 100

        food_image = cv2.imread('images/food.jpg')
        food_image = cv2.resize(food_image, (resize, resize))

        player_image = cv2.imread('images/player.jpeg')
        player_image = cv2.resize(player_image, (resize, resize))

        enemy_image = cv2.imread('images/enemy.jpeg')
        enemy_image = cv2.resize(enemy_image, (resize, resize))

        lava_image = cv2.imread('images/volcano.png')
        lava_image = cv2.resize(lava_image, (resize * 2, resize * 2))

        if show_img:

            # define the environment
            environment = np.zeros((self.size * resize, self.size * resize, 3))

            # the guy
            for index, r in enumerate(range(self.guy.x * resize, self.guy.x * resize + resize)):
                for eindex, c in enumerate(range(self.guy.y * resize, self.guy.y * resize + resize)):
                    player_pixel = [elm/255 for elm in player_image[index][eindex]]

                    # remove the background - NOT USED HERE
                    if player_pixel == [0, 0, 0]:
                        environment[r][c] = [0, 0, 0]
                    else:
                        environment[r][c] = player_pixel

            # the food
            for index, r in enumerate(range(self.food.x * resize, self.food.x * resize + resize)):
                for eindex, c in enumerate(range(self.food.y * resize, self.food.y * resize + resize)):
                    food_pixel = [elm/255 for elm in food_image[index][eindex]]

                    # remove the background
                    if food_pixel == [1, 1, 1]:
                        environment[r][c] = [0, 0, 0]
                    else:
                        environment[r][c] = food_pixel

            # the enemy
            for index, r in enumerate(range(self.enemy.x * resize, self.enemy.x * resize + resize)):
                for eindex, c in enumerate(range(self.enemy.y * resize, self.enemy.y * resize + resize)):
                    enemy_pixel = [elm/255 for elm in enemy_image[index][eindex]]

                    # remove the background - NOT USED HERE
                    if enemy_pixel == [0, 0, 0]:
                        environment[r][c] = [0, 0, 0]
                    else:
                        environment[r][c] = enemy_pixel

            # the wall
            for pos in self.wall.pos_range:
                for index, r in enumerate(range(pos[0] * resize, pos[0] * resize + resize)):
                    for eindex, c in enumerate(range(pos[1] * resize, pos[1] * resize + resize)):
                        environment[r][c] = [0.8, 0.8, 0.8]

            # the lava
            block1 = lava_image[0:resize * 1, 0:resize * 1]
            block2 = lava_image[0:resize * 1, resize * 1:resize * 2]
            block3 = lava_image[resize * 1:resize * 2, 0:resize * 1]
            block4 = lava_image[resize * 1:resize * 2, resize * 1:resize * 2]

            blocks = [block1, block2, block3, block4]

            # after dividing the image into blocks...
            ix = 0
            for posindex, pos in enumerate(self.lava.pos_range):
                for index, r in enumerate(range(pos[0] * resize, pos[0] * resize + resize)):
                    for eindex, c in enumerate(range(pos[1] * resize, pos[1] * resize + resize)):
                        lava_pixel = [elm/255 for elm in blocks[ix][index][eindex]]

                        if lava_pixel == [1, 1, 1]:
                            environment[r][c] = [0, 0, 0]
                        else:
                            environment[r][c] = lava_pixel
                if ix == 3:
                    ix = 0
                else:
                    ix += 1

            row = [[]]
            col = []

            # the white lines
            for i1 in range(self.size * resize):
                col.append([1, 1, 1])

            for i2 in range(self.size * resize + self.size + 1):
                row[0].append([1, 1, 1])

            for i3 in range(self.size + 1):
                environment = np.insert(environment, i3 * (resize + 1), col, axis=1)

            for i4 in range(self.size + 1):
                environment = np.insert(environment, i4 * (resize + 1), row, axis=0)

            cv2.imshow("", environment)

            cv2.waitKey(1)


SIZE = 10
train = True

TPU_TRAIN = True

# episodes
EPISODES = 10_000
LIMIT_STEPS_PER_EPISODE = True
STEPS_PER_EPISODE = 200

# epsilons
if train:
    INITIAL_EPSILON = 0.9
    DECAY_RESET = True
    DECAY_RATE = 0.92
else:
    INITIAL_EPSILON = 0.0
    DECAY_RESET = False
    DECAY_RATE = 0.92

# penalties and reward
MOVE_P = -1
ENEMY_P = -400
FOOD_R = 500
LAVA_P = -400

# model stuff
if TPU_Train:
    LIMIT_STEPS_PER_EPISODE = False
    GAMMA = 0.99
    REPLAY_MEMORY_SIZE = 10_000_000
    MIN_REPLAY_SIZE = 100_000
    MINIBATCH_SIZE = 50_000
    UPDATE_TARGET_EVERY = 1
    LOAD_MODEL = 'models/model1.h5'

else:
    GAMMA = 0.99
    REPLAY_MEMORY_SIZE = 100_000
    MIN_REPLAY_SIZE = 100
    MINIBATCH_SIZE = 64
    UPDATE_TARGET_EVERY = 5
    LOAD_MODEL = 'models/model1.h5'

# begin
agent = AgentModel(SIZE, REPLAY_MEMORY_SIZE, MIN_REPLAY_SIZE, MINIBATCH_SIZE, GAMMA, UPDATE_TARGET_EVERY,
                   LOAD_MODEL)
rewards_list = []
env = MyEnv(SIZE, ENEMY_P, MOVE_P, FOOD_R, LAVA_P)

# training loop
for i in range(EPISODES):
    step = 1
    done = False
    reward_val = 0

    current_space = env.reset()

    while not done:

        rand_eps = np.random.random()
        if rand_eps > INITIAL_EPSILON:

            # if we are using the image of the environment to train
            if current_space.shape == (SIZE, SIZE, 1):
                action = np.argmax(agent.model.predict(np.reshape(current_space, (1, SIZE, SIZE, 1))))

            elif current_space.shape == (100,):
                action = np.argmax(agent.model.predict(np.reshape(current_space, (1, 100))))

            else:
                action = None

        else:
            action = random.choice(range(9))

        reward, done, future_space = env.move(action, LIMIT_STEPS_PER_EPISODE, STEPS_PER_EPISODE)

        reward_val += reward

        if train:
            agent.update_replay_memory([current_space, action, reward, future_space, done])
            agent.fit_model(done)

        current_space = future_space
        step += 1

        # display
        if train:
            if i > EPISODES - 10 or i < -10:
                show = True
            else:
                show = False

            env.view(show)

        else:
            if i > 3:
                show = True
            else:
                show = False

            env.view(show)

    # append rewards
    rewards_list.append(reward_val)

    # save
    if i == EPISODES - 1:
        max_reward = max(rewards_list)
        min_reward = min(rewards_list)
        final_file = f'models/FINAL_MAX-({int(max_reward)})_MIN-({int(min_reward)}).h5'
        agent.model.save(final_file)
        
        # save to google cloud
        export_path1 = os.path.join('gs://gideon116', final_file)
        with file_io.FileIO(final_file, mode='rb') as input_f:
            with file_io.FileIO(export_path1, mode='wb+') as output_f:
                output_f.write(input_f.read())

    if i % 50 == 0:
        ave_reward = sum(rewards_list[i-49:])/len(rewards_list[i-49:])
        file = f'models/Episode=>{i}__Ave=>{int(ave_reward)}).h5'
        agent.model.save(file)
        
        export_path2 = os.path.join('gs://gideon116', file)
        with file_io.FileIO(file) as input_f:
            with file_io.FileIO(export_path2, mode='wb+') as output_f:
                output_f.write(input_f.read())

    # decay epsilon
    INITIAL_EPSILON *= DECAY_RATE

    if DECAY_RESET:
        if INITIAL_EPSILON < 0.0001:
            INITIAL_EPSILON = 0.2
            print(f'Epsilon has been reset to {INITIAL_EPSILON}')

    # verbosity
    print(f"Episode: {i}/{EPISODES}  Episode Reward: {int(reward_val)}")

avg = np.convolve(rewards_list, np.ones((100,))/100, mode='valid')
plt.plot([i for i in range(len(avg))], avg)
plt.show()
