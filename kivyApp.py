import kivy
from kivy.uix.image import Image
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.animation import Animation
from kivy.uix.stacklayout import StackLayout
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from random import randint
from kivy.clock import Clock
from kivy.graphics import *
from tensorflow.keras.models import load_model
from environment import MyEnv
from wall import Wall
from lava import Lava
import numpy as np
from agent import AgentModel
import os
import random


os.chdir('/Users/gideon/Desktop/Reinforcement Learning/Qgames/')


class Agent(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rows = 4
        # self.row_force_default = True
        #  size_hint_x=None, width=self.width
        # self.row_default_height = 40
        self.but1 = TextInput(text='r')
        self.but2 = Button(text='a')
        self.but3 = Button(text='a')
        self.but4 = Button(text='a')

        self.add_widget(self.but1)
        self.add_widget(self.but2)
        self.add_widget(self.but3)
        self.add_widget(self.but4)


class ThreeCol(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 3


class WelcomeButton(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rows = 3

        self.but1 = Label()
        self.but2 = Button(text='Get Started', size_hint=(0, 0.3))
        self.but3 = Label()

        self.add_widget(self.but1)
        self.add_widget(self.but2)
        self.add_widget(self.but3)


# actual UI stuff
class Welcome(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rows = 2

        self.labelW = Label(text="Chase World", halign='center', valign='middle', font_size='100')
        self.labelW.bold = 10
        self.add_widget(self.labelW)

        self.welcome_button = WelcomeButton()
        self.welcome_button.but2.bind(on_release=self.begin_things)

        self.get_started = ThreeCol()
        self.get_started.add_widget(Label())
        self.get_started.add_widget(self.welcome_button)
        self.get_started.add_widget(Label())

        self.add_widget(self.get_started)

    def begin_things(self, instance):
        app.screen_manager.current = 'first'


class First(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rows = 3

        self.title = Label(text='Choose your chaser', font_size=50, bold=90)
        self.add_widget(self.title)

        self.spongebob = Button(background_normal='images/chaserp.png', background_down='images/chaserp.png')
        self.spongebob.bind(on_release=self.begin_things_spongebob)

        self.patrick = Button(background_normal='images/chasers.png', background_down='images/chasers.png')
        self.patrick.bind(on_release=self.begin_things_patrick)

        self.players = ThreeCol()
        self.players.add_widget(self.spongebob)
        self.players.add_widget(self.patrick)

        self.add_widget(self.players)

    def begin_things_spongebob(self, instance):
        app.screen_manager.current = 'second'

    def begin_things_patrick(self, instance):
        app.screen_manager.current = 'second_s'


class Second(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.back = Button(text='Back')
        self.back.bind(on_release=self.begin_things)
        self.add_widget(self.back)

        # Agent
        move_penalty = -1
        enemy_penalty = -400
        lava_penalty = -400
        food_reward = 500
        size = 10

        env = MyEnv(size, enemy_penalty, move_penalty, food_reward, lava_penalty, wall_type='special', lava_type='three')
        current_space = env.reset()
        agent = AgentModel(size, replay_memory_size=1000, min_replay_size=30, minbatch_size=30,
                           gamma=0.99, update_target_every=5, loadmodel='models/models_special_three-special_three_25000.h5')
        reward_val = 0
        done = False

        self.food_x = 100 + 100 * env.food.x
        self.food_y = 100 + 100 * env.food.y

        self.lava_x = []
        self.lava_y = []
        for lava in env.lava.pos_range:
            self.lava_x.append(100 + 100 * lava[0])
            self.lava_y.append(100 + 100 * lava[1])

        self.chaser_x = []
        self.chaser_y = []

        self.enemy_x = []
        self.enemy_y = []

        while not done:
            action = np.argmax(agent.model.predict(np.reshape(current_space, (1, 100))))
            rand_eps = np.random.random()

            if rand_eps > 0:
                reward, done, future_space = env.move(action)
            else:
                reward, done, future_space = env.move(random.choice(range(9)))

            reward_val += reward
            current_space = future_space

            self.chaser_x.append(100 + 100 * env.guy.x)
            self.chaser_y.append(100 + 100 * env.guy.y)

            self.enemy_x.append(100 + 100 * env.enemy.x)
            self.enemy_y.append(100 + 100 * env.enemy.y)

            with self.canvas:
                Color(1, 1, 1)

                self.floor = Line(points=[100, 100,
                                          1100, 100,
                                          1100, 1100,
                                          100, 1100,
                                          100, 100], width=2)

                # lines in between
                for i in range(1, 11):
                    Line(points=[i * 100 + 100, 100,
                                 i * 100 + 100, 1100])

                    Line(points=[100, i * 100 + 100,
                                 1100, i * 100 + 100])

        # load the food
        self.food = Image(source='images/burger.png', height=70, width=70)
        self.food.x = self.food_x
        self.food.y = self.food_y
        self.add_widget(self.food)

        # load the lava
        lava_height = 200
        lava_width = 200

        for index, lava in enumerate(env.lava.pos_range):
            if index % 4 == 0:
                self.lava = Image(source='images/volcano2.png', height=lava_height, width=lava_width)
                self.lava.x = self.lava_x[index]
                self.lava.y = self.lava_y[index]
                self.add_widget(self.lava)

        # load the players
        self.chaser = Image(source='images/patrick.png', height=75, width=75)
        self.chaser.x = 100 + 100 * self.chaser_x[0]
        self.chaser.y = 100 + 100 * self.chaser_y[1]
        self.add_widget(self.chaser)

        self.enemy = Image(source='images/spongebob.png', height=75, width=75)
        self.enemy.x = 100 + 100 * self.enemy_x[0]
        self.enemy.y = 100 + 100 * self.enemy_y[1]
        self.add_widget(self.enemy)

        for wall_b in env.wall.pos_range:
            xi = 100 + 100 * wall_b[0]
            yi = 100 + 100 * wall_b[1]

            with self.canvas:
                Color(1, 1, 1)
                self.wall_block = Rectangle()
                self.wall_block.size = (100, 100)
                self.wall_block.pos = (xi, yi)

        # move the enemy
        self.animation = Animation(x=self.enemy_x[0], y=self.enemy_y[0], duration=1, t='in_out_elastic')
        for index, i in enumerate(self.enemy_x):
            self.animation += Animation(x=i, y=self.enemy_y[index], duration=1, t='in_out_elastic')
        self.animation.start(self.enemy)

        # move the chaser
        self.animation = Animation(x=self.chaser_x[0], y=self.chaser_y[0], duration=1, t='in_out_elastic')
        for index, i in enumerate(self.chaser_x):
            self.animation += Animation(x=i, y=self.chaser_y[index], duration=1, t='in_out_elastic')

        # if the chaser get the good
        if self.chaser_x[-1] == self.food.x and self.chaser_y[-1] == self.food.y:

            self.winner = Label(text='Winner')
            self.winner.pos = (self.food.x, self.food.y)
            self.add_widget(self.winner)

            self.animation.on_complete(self.winner)
            self.animation.stop(self.chaser)
            self.animation.start(self.chaser)



        # if the chaser hits the lava
        for lava in env.lava.pos_range:
            if self.chaser_x[-1] == lava[0] and self.chaser_y[-1] == lava[1]:
                self.animation += Animation(source="v1.png", t='in_out_elastic')
                self.animation.start(self.chaser)
                break



    def begin_things(self, instance):
        app.screen_manager.current = 'first'


class SecondS(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.back = Button(text='Back')
        self.back.bind(on_release=self.begin_things)
        self.add_widget(self.back)

        # Agent
        move_penalty = -1
        enemy_penalty = -400
        lava_penalty = -400
        food_reward = 500
        size = 10

        env = MyEnv(size, enemy_penalty, move_penalty, food_reward, lava_penalty, wall_type='special', lava_type='three')
        current_space = env.reset()
        agent = AgentModel(size, replay_memory_size=1000, min_replay_size=30, minbatch_size=30,
                           gamma=0.99, update_target_every=5, loadmodel='models/models_special_three-special_three_25000.h5')
        reward_val = 0
        done = False

        self.food_x = 100 + 100 * env.food.x
        self.food_y = 100 + 100 * env.food.y

        self.lava_x = []
        self.lava_y = []
        for lava in env.lava.pos_range:
            self.lava_x.append(100 + 100 * lava[0])
            self.lava_y.append(100 + 100 * lava[1])

        self.chaser_x = []
        self.chaser_y = []

        self.enemy_x = []
        self.enemy_y = []

        while not done:
            action = np.argmax(agent.model.predict(np.reshape(current_space, (1, 100))))
            rand_eps = np.random.random()

            if rand_eps > 1:
                reward, done, future_space = env.move(action)
            else:
                reward, done, future_space = env.move(random.choice(range(9)))

            reward_val += reward
            current_space = future_space

            self.chaser_x.append(100 + 100 * env.guy.x)
            self.chaser_y.append(100 + 100 * env.guy.y)

            self.enemy_x.append(100 + 100 * env.enemy.x)
            self.enemy_y.append(100 + 100 * env.enemy.y)

            with self.canvas:
                Color(1, 1, 1)

                self.floor = Line(points=[100, 100,
                                          1100, 100,
                                          1100, 1100,
                                          100, 1100,
                                          100, 100], width=2)

                # lines in between
                for i in range(1, 11):
                    Line(points=[i * 100 + 100, 100,
                                 i * 100 + 100, 1100])

                    Line(points=[100, i * 100 + 100,
                                 1100, i * 100 + 100])

        # load the food
        self.food = Image(source='images/burger.png', height=70, width=70)
        self.food.x = self.food_x
        self.food.y = self.food_y
        self.add_widget(self.food)

        # load the lava
        lava_height = 200
        lava_width = 200

        for index, lava in enumerate(env.lava.pos_range):
            if index % 4 == 0:
                self.lava = Image(source='images/volcano2.png', height=lava_height, width=lava_width)
                self.lava.x = self.lava_x[index]
                self.lava.y = self.lava_y[index]
                self.add_widget(self.lava)

        # load the players
        self.chaser = Image(source='images/spongebob.png', height=75, width=75)
        self.chaser.x = 100 + 100 * self.chaser_x[0]
        self.chaser.y = 100 + 100 * self.chaser_y[1]
        self.add_widget(self.chaser)

        self.enemy = Image(source='images/patrick.png', height=75, width=75)
        self.enemy.x = 100 + 100 * self.enemy_x[0]
        self.enemy.y = 100 + 100 * self.enemy_y[1]
        self.add_widget(self.enemy)

        for wall_b in env.wall.pos_range:
            xi = 100 + 100 * wall_b[0]
            yi = 100 + 100 * wall_b[1]

            with self.canvas:
                Color(1, 1, 1)
                self.wall_block = Rectangle()
                self.wall_block.size = (100, 100)
                self.wall_block.pos = (xi, yi)

        # move the enemy
        self.animation = Animation(x=self.enemy_x[0], y=self.enemy_y[0], duration=1, t='in_out_elastic')
        for index, i in enumerate(self.enemy_x):
            self.animation += Animation(x=i, y=self.enemy_y[index], duration=1, t='in_out_elastic')
        self.animation.start(self.enemy)

        # move the chaser
        self.animation = Animation(x=self.chaser_x[0], y=self.chaser_y[0], duration=1, t='in_out_elastic')
        for index, i in enumerate(self.chaser_x):
            self.animation += Animation(x=i, y=self.chaser_y[index], duration=1, t='in_out_elastic')

        # if the chaser hits the enemy
        if self.chaser.x == self.enemy.x and self.chaser.y == self.enemy.y:
            self.animation &= Animation(source="vv.png", t='in_out_elastic')

        # if the chaser get the good
        elif self.chaser.x == self.food.x and self.chaser.y == self.food.y:
            self.animation &= Animation(source="vv.png", t='in_out_elastic')

        # if the chaser hits the lava
        for lava in env.lava.pos_range:
            if self.chaser.x == lava[0] and self.chaser.y == lava[1]:
                self.animation &= Animation(source="vv.png", t='in_out_elastic')
                break

        self.animation.start(self.chaser)

    def begin_things(self, instance):
        app.screen_manager.current = 'first'


class Third(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 2

        self.begin = Button(text='Begin3')
        self.begin.bind(on_press=self.begin_things)
        self.add_widget(self.begin)

    def begin_things(self, instance):
        txt = self.agent.but1.text
        self.begin.text = txt


class CWGame(Widget):

    def __call__(self):
        return First()

    def serve_ball(self, vel=(4, 0)):
        self.ball.center = self.center
        self.ball.velocity = vel

    def on_touch_move(self, touch):
        if touch.x < self.width / 3:
            self.player1.center_y = touch.y
        if touch.x > self.width - self.width / 3:
            self.player2.center_y = touch.y

    def update(self, dt):
        self.ball.move()
        self.player1.bounce_ball(self.ball)
        self.player2.bounce_ball(self.ball)

        if (self.ball.y < self.y) or (self.ball.top > self.top):
            self.ball.velocity_y *= -1

        if self.ball.x < self.x:
            self.player2.score += 1
            self.serve_ball(vel=(4, 0))
        if self.ball.x > self.width:
            self.player1.score += 1
            self.serve_ball(vel=(4, 0))


class CWApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.screen_manager = ScreenManager()
        self.welcome = Welcome()
        self.first = First()
        self.second = Second()
        self.second_s = SecondS()
        self.third = Third()

    def build(self):
        # Clock.schedule_interval(game.update, 1.0 / 60.0)

        screen = Screen(name='welcome')
        screen.add_widget(self.welcome)
        self.screen_manager.add_widget(screen)

        screen = Screen(name='first')
        screen.add_widget(self.first)
        self.screen_manager.add_widget(screen)

        screen = Screen(name='second')
        screen.add_widget(self.second)
        self.screen_manager.add_widget(screen)

        screen = Screen(name='second_s')
        screen.add_widget(self.second_s)
        self.screen_manager.add_widget(screen)

        screen = Screen(name='third')
        screen.add_widget(self.third)
        self.screen_manager.add_widget(screen)

        return self.screen_manager


if __name__ == "__main__":
    app = CWApp()
    app.run()
