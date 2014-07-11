#!/usr/bin/python
from evolution_bot import ScoredPerceptron, Robot, Family
from rgkit.run import Runner, Options
from rgkit.game import Player

g = 99
loadfile0 = "brains33/perceptron_36_9_10_red_generation%d.dat" % g
loadfile1 = "brains33/perceptron_36_9_10_blu_generation%d.dat" % g

sz = 8

brain0 = ScoredPerceptron.load(loadfile0)
seed0 = [brain0.copy() for _ in range(sz)]
pop0 = Family(sz, 0, 0, seed0)
robot0 = Robot(pop0)
player0 = Player(name="red %s" % (loadfile0.split("generation")[1]), robot=robot0)

brain1 = ScoredPerceptron.load(loadfile1)
seed1 = [brain1.copy() for _ in range(sz)]
pop1 = Family(sz, 0, 0, seed1)
robot1 = Robot(pop1)
player1 = Player(name="blu %s" % (loadfile1.split("generation")[1]), robot=robot1)

opts = Options(print_info=True)

r = Runner(players=[player0, player1], options=opts)
r.run()

# r = Runner(players=[player1, player0], options=opts)
# r.run()