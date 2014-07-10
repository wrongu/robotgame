#!/usr/bin/python
from evolution_bot import Perceptron, Robot
from rgkit.run import Runner, Options
from rgkit.game import Player

bot0 = "brains/perceptron_36_9_10_generation800.dat"
bot1 = "brains/perceptron_36_9_10_generation700.dat"

robot0 = Robot()
robot0.set_brain(Perceptron.load(bot0))
player0 = Player(name="gen %s" % (bot0.split("generation")[1]), robot=robot0)

robot1 = Robot()
robot1.set_brain(Perceptron.load(bot1))
player1 = Player(name="gen %s" % (bot1.split("generation")[1]), robot=robot1)

opts = Options(print_info=True)

r = Runner(players=[player0, player1], options=opts)
r.run()