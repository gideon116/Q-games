import numpy as np
import cv2

from players import Players
from wall import Wall
from lava import Lava


class MyEnv:

    def __init__(self, size, enemy_p, move_p, food_r, lava_p, wall_type, lava_type, train_by_image=False):

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
        self.wall.location(wall_type)

        # the lava
        self.lava = Lava(env_size=self.size)
        self.lava.number_of_lava(lava_type)

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

            for wall_b in self.wall.pos_range:
                current_space_per_reset[wall_b[0]][wall_b[1]] = [0.25]

            for lava_b in self.lava.pos_range:
                current_space_per_reset[lava_b[0]][lava_b[1]] = [0.75]

        else:

            current_space_per_reset = np.array([self.guy.x - self.food.x, self.guy.y - self.food.y,
                                                self.guy.x - self.enemy.x, self.guy.y - self.enemy.y])

            # get the distance from each wall block
            for wall_b in self.wall.pos_range:
                current_space_per_reset = np.append(current_space_per_reset, self.guy.x - wall_b[0])
                current_space_per_reset = np.append(current_space_per_reset, self.guy.y - wall_b[1])

            # get the distance from each lava block
            for lava_b in self.lava.pos_range:
                current_space_per_reset = np.append(current_space_per_reset, self.guy.x - lava_b[0])
                current_space_per_reset = np.append(current_space_per_reset, self.guy.y - lava_b[1])

            # additional space if we want to have a total of 50 things
            for index in range(50 - len(current_space_per_reset) // 2):
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
                    reward_per_move += (0.01 * self.lava_p) // ((np.sqrt((self.guy.x - pos[0]) ** 2 +
                                                                         (self.guy.y - pos[1]) ** 2)) ** 2)

        self.reward = reward_per_move

        # if we are using the image of the environment to train
        if self.train_by_image:

            future_space_per_move = np.zeros((self.size, self.size, 1))
            future_space_per_move[self.guy.x][self.guy.y] = [1]
            future_space_per_move[self.food.x][self.food.y] = [0]
            future_space_per_move[self.enemy.x][self.enemy.y] = [0.5]

            for wall_b in self.wall.pos_range:
                future_space_per_move[wall[0]][wall_b[1]] = [0.25]

            for lava_b in self.lava.pos_range:
                future_space_per_move[lava[0]][lava_b[1]] = [0.75]

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
            for index in range(50 - len(future_space_per_move) // 2):
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
                    player_pixel = [elm / 255 for elm in player_image[index][eindex]]

                    # remove the background - NOT USED HERE
                    if player_pixel == [0, 0, 0]:
                        environment[r][c] = [0, 0, 0]
                    else:
                        environment[r][c] = player_pixel

            # the food
            for index, r in enumerate(range(self.food.x * resize, self.food.x * resize + resize)):
                for eindex, c in enumerate(range(self.food.y * resize, self.food.y * resize + resize)):
                    food_pixel = [elm / 255 for elm in food_image[index][eindex]]

                    # remove the background
                    if food_pixel == [1, 1, 1]:
                        environment[r][c] = [0, 0, 0]
                    else:
                        environment[r][c] = food_pixel

            # the enemy
            for index, r in enumerate(range(self.enemy.x * resize, self.enemy.x * resize + resize)):
                for eindex, c in enumerate(range(self.enemy.y * resize, self.enemy.y * resize + resize)):
                    enemy_pixel = [elm / 255 for elm in enemy_image[index][eindex]]

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
                        lava_pixel = [elm / 255 for elm in blocks[ix][index][eindex]]

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
            
