import kivy
from kivy.uix.image import Image
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.colorpicker import Color
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

		env = MyEnv(size, enemy_penalty, move_penalty, food_reward, lava_penalty, wall_type='bottom', lava_type='one')
		current_space = env.reset()
		agent = AgentModel(size, replay_memory_size=1000, min_replay_size=30, minbatch_size=30,
						gamma=0.99, update_target_every=5, loadmodel='models/model3.h5')
		reward_val = 0
		done = False

		self.food_x = 150 + 100 * env.food.y + 100 * env.food.x
		self.food_y = 125 + 50 * env.food.x

		self.chaser_x = []
		self.chaser_y = []

		self.enemy_x = []
		self.enemy_y = []

		while not done:
			action = np.argmax(agent.model.predict(np.reshape(current_space, (1, 100))))
			reward, done, future_space = env.move(action)
			reward_val += reward
			current_space = future_space

			self.chaser_x.append(150 + 100 * env.guy.x + 100 * env.guy.y)
			self.chaser_y.append(125 + 50 * env.guy.y)

			self.enemy_x.append(150 + 100 * env.enemy.x + 100 * env.enemy.y)
			self.enemy_y.append(125 + 50 * env.enemy.y)

		with self.canvas:
			Color(1, 1, 1)

			self.floor = Line(points=[100, 100,
									1100, 100,
									2100, 600,
									1100, 600,
									100, 100], width=2)


			# lines in between
			for i in range(1, 11):
				Line(points=[i * 100 + 100, 100,
							i * 100 + 1100, 600])

				Line(points=[i * 100 + 100, i * 50 + 100,
							i * 100 + 1100, i * 50 + 100])

		# the chaser
		for e in range(100):
			with self.canvas:
				if e < 5:
					Color(0, 0, 0)
				else:
					Color(1, 1, 1)

				vertices = []
				indices = []

				for i in range(100):
					x = 100 + e + (100 * self.chaser_x[0] + 100 * self.chaser_y[0]) / 1
					y = 100 + i + (100 * self.chaser_y[0]) / 2
					vertices.extend([x, y, 0, 0])
					indices.append(i)

				self.chaser = Mesh(vertices=vertices, indices=indices)

				for loop in range(3):
					animated_vert = []
					animated_indi = []
					for i in range(100):
						x = 100 + e + (100 * self.chaser_x[loop] + 100 * self.chaser_y[0]) / 1
						y = 100 + i + (100 * self.chaser_y[loop]) / 2
						animated_vert.extend([x, y, 0, 0])
						animated_indi.append(i)

					self.animation = Animation(vertices=animated_vert, indices=animated_indi)

					self.animation.start(self.chaser)

		# the enemy
		for e in range(100):
			with self.canvas:
				if e < 5:
					Color(0, 0, 0)
				else:
					Color(1, 1, 1)

				vertices = []
				indices = []

				for i in range(100):
					x = 100 + e + (100 * self.enemy_x[0] + 100 * self.enemy_y[0]) / 1
					y = 100 + i + (100 * self.enemy_y[0]) / 2
					vertices.extend([x, y, 0, 0])
					indices.append(i)

				self.enemy = Mesh(vertices=vertices, indices=indices)

				for loop in range(3):
					animated_vert = []
					animated_indi = []
					for i in range(100):
						x = 100 + e + (100 * self.enemy_x[loop] + 100 * self.enemy_y[0]) / 1
						y = 100 + i + (100 * self.enemy_y[loop]) / 2
						animated_vert.extend([x, y, 0, 0])
						animated_indi.append(i)

					self.animation = Animation(vertices=animated_vert, indices=animated_indi)

					self.animation.start(self.enemy)

		self.food = Image(source='images/burger.png', height=70, width=70)
		self.food.x = self.food_x
		self.food.y = self.food_y
		self.add_widget(self.food)

		if env.guy.x == env.food.x and env.guy.y == env.food.y:
			self.remove_widget(self.food)
			self.add_widget(Label(text='Winner'))
		elif env.guy.x == env.enemy.x and env.guy.y == env.enemy.y:
			self.remove_widget(self.chaser)
			self.add_widget(Label(text='Try again'))

		for wall_b in env.wall.pos_range:
			xi = wall_b[1]
			yi = wall_b[0]

			for e in range(100):
				with self.canvas:
					if e < 5:
						Color(0, 0, 0)
					else:
						Color(1, 1, 1)

					vertices = []
					indices = []

					for i in range(100):
						x = 100 + e + (100 * xi + 100 * yi)/1
						y = 100 + i + (100 * yi)/2
						vertices.extend([x, y, 0, 0])
						indices.append(i)

					self.wallfront = Mesh(vertices=vertices, indices=indices)

		for wall_b in env.wall.pos_range:
			xi = wall_b[1]
			yi = wall_b[0]

			for e in range(100):
				with self.canvas:
					if e < 5:
						Color(0, 0, 0)
					else:
						Color(1, 1, 1)

					vertices = []
					indices = []

					for i in range(100):

						x = 200 + e + (100 * xi + 100 * yi)/1
						y = 100 + i + (e + 100 * yi)/2
						vertices.extend([x, y, 0, 0])
						indices.append(i)

					self.wallfront = Mesh(vertices=vertices, indices=indices)

		for wall_b in env.wall.pos_range:
			xi = wall_b[1]
			yi = wall_b[0]

			for e in range(100):
				with self.canvas:
					if e < 5:
						Color(0, 0, 0)
					else:
						Color(0, 1, 1)

					vertices = []
					indices = []

					for i in range(100):
						x = 100 + i + e + (100 * xi + 100 * yi)/1
						y = 200 + (e + 100 * yi)/2
						vertices.extend([x, y, 0, 0])
						indices.append(i)

					self.wallfront = Mesh(vertices=vertices, indices=indices)

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

		env = MyEnv(size, enemy_penalty, move_penalty, food_reward, lava_penalty, wall_type='bottom', lava_type='one')
		current_space = env.reset()
		agent = AgentModel(size, replay_memory_size=1000, min_replay_size=30, minbatch_size=30,
						gamma=0.99, update_target_every=5, loadmodel='models/model3.h5')
		reward_val = 0
		done = False

		self.food_x = 150 + 100 * (9 - env.food.x) + 100 * env.food.y
		self.food_y = 125 + 50 * env.food.y

		self.chaser_x = []
		self.chaser_y = []

		self.enemy_x = []
		self.enemy_y = []

		while not done:
			action = np.argmax(agent.model.predict(np.reshape(current_space, (1, 100))))
			reward, done, future_space = env.move(action)
			reward_val += reward
			current_space = future_space

			self.chaser_x.append(150 + 100 * env.guy.x + 100 * env.guy.y)
			self.chaser_y.append(125 + 50 * env.guy.y)

			self.enemy_x.append(150 + 100 * env.enemy.x + 100 * env.enemy.y)
			self.enemy_y.append(125 + 50 * env.enemy.y)

		with self.canvas:
			Color(1, 1, 1)

			self.floor = Line(points=[100, 100,
									1100, 100,
									2100, 600,
									1100, 600,
									100, 100], width=2)


			# lines in between
			for i in range(1, 11):
				Line(points=[i * 100 + 100, 100,
							i * 100 + 1100, 600])

				Line(points=[i * 100 + 100, i * 50 + 100,
							i * 100 + 1100, i * 50 + 100])

		self.chaser = Image(source=f'images/spongebob.png', height=100, width=100)
		self.add_widget(self.chaser)

		self.enemy = Image(source=f'images/patirck.png', height=100, width=100)
		self.add_widget(self.enemy)

		self.food = Image(source='images/burger.png', height=70, width=70)
		self.food.x = self.food_x
		self.food.y = self.food_y
		self.add_widget(self.food)

		self.animation = Animation(x=self.chaser_x[0], y=self.chaser_y[0], duration=10) + \
						Animation(x=self.chaser_x[1], y=self.chaser_y[1]) + \
						Animation(x=self.chaser_x[2], y=self.chaser_y[2])
		self.animation.start(self.chaser)

		self.animation = Animation(x=self.enemy_x[0], y=self.enemy_y[0], duration=10) + \
						Animation(x=self.enemy_x[1], y=self.enemy_y[1]) + \
						Animation(x=self.enemy_x[2], y=self.enemy_y[2])
		self.animation.start(self.enemy)

		if env.guy.x == env.food.x and env.guy.y == env.food.y:
			self.remove_widget(self.food)
			self.add_widget(Label(text='Winner'))
		elif env.guy.x == env.enemy.x and env.guy.y == env.enemy.y:
			self.remove_widget(self.chaser)
			self.add_widget(Label(text='Try again'))

		for wall_b in env.wall.pos_range:
			xi = 9 - wall_b[0]
			yi = wall_b[1]

			for e in range(100):
				with self.canvas:
					if e < 5:
						Color(0, 0, 0)
					else:
						Color(1, 1, 1)

					vertices = []
					indices = []

					for i in range(100):
						x = 100 + e + (100 * xi + 100 * yi)/1
						y = 100 + i + (100 * yi)/2
						vertices.extend([x, y, 0, 0])
						indices.append(i)

					self.wallfront = Mesh(vertices=vertices, indices=indices)

		for wall_b in env.wall.pos_range:
			xi = 9 - wall_b[0]
			yi = wall_b[1]

			for e in range(100):
				with self.canvas:
					if e < 5:
						Color(0, 0, 0)
					else:
						Color(1, 1, 1)

					vertices = []
					indices = []

					for i in range(100):

						x = 200 + e + (100 * xi + 100 * yi)/1
						y = 100 + i + (e + 100 * yi)/2
						vertices.extend([x, y, 0, 0])
						indices.append(i)

					self.wallfront = Mesh(vertices=vertices, indices=indices)

		for wall_b in env.wall.pos_range:
			xi = 9 - wall_b[0]
			yi = wall_b[1]

			for e in range(100):
				with self.canvas:
					if e < 5:
						Color(0, 0, 0)
					else:
						Color(0, 1, 1)

					vertices = []
					indices = []

					for i in range(100):
						x = 100 + i + e + (100 * xi + 100 * yi)/1
						y = 200 + (e + 100 * yi)/2
						vertices.extend([x, y, 0, 0])
						indices.append(i)

					self.wallfront = Mesh(vertices=vertices, indices=indices)

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
