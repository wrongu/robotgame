# create brain to observe match
# run match between existing bots
	# set inputs and ideal output according to teacher's decision
	# (also take into account outcome??)
# run match for brain

import evolution_bot
import corners3
import numpy as np

class ObservationBot(evolution_bot.Robot):

	def __init__(self, teacher, population):
		super(ObservationBot, self).__init__(population)
		self._teacher = teacher

	def act(self, game):
		self._teacher.hp        = self.hp
		self._teacher.player_id = self.player_id
		self._teacher.robot_id  = self.robot_id
		self._teacher.location  = self.location
		wise_decision = self._teacher.act(game)

		x,y = self.location
		brain_choices = [
			['move', (x-1, y)],   ['move', (x+1, y)],   ['move', (x, y-1)],   ['move', (x, y+1)],
			['attack', (x-1, y)], ['attack', (x+1, y)], ['attack', (x, y-1)], ['attack', (x, y+1)],
			['guard'], ['suicide']]
		brain_index = brain_choices.index(wise_decision)

		if brain_index < 0:
			print "error!", wise_decision, "not an option in", brain_choices
		else:
			ideal_output = np.zeros((len(brain_choices), 1))
			ideal_output[brain_index] = 1.0

			self.population.group_lesson(self.sense_environment(game), ideal_output, 0.1)
		return wise_decision

from rgkit.run import Runner, Options
from rgkit.game import Player
from rgkit.settings import settings

brain_kwargs = {
	"layer_sizes" : [100, 25, 10],
	"random" : 0.01
}

population = evolution_bot.Collective(1, 0, 0, **brain_kwargs)
teacher_class = corners3.Robot

# run the observation match #
settings["max_turns"] = 1000
red = Player(name="Corners 3", robot=corners3.Robot())
blue = Player(name="Sempai", robot=ObservationBot(teacher_class(), population))
r = Runner(players=[red,blue])
r.run()

# run a koohai match
settings["max_turns"] = 100
opts = Options(print_info=True)
red  = Player(name="koohai", robot=evolution_bot.Robot(population))
blue = Player(name="koohai", robot=evolution_bot.Robot(population))
r = Runner(players=[red,blue], options=opts)
r.run()
