#!/usr/bin/python
from evolution_bot import ScoredPerceptron, Robot, Collective
from rgkit.run import Runner, Options
from rgkit.game import Player

snapshots = [1000, 2000, 3000, 4000, 5000]

for t in snapshots:
	loadfile0 = "evolved_brains/sim0/red/perceptron_100_25_10_turn%d_gen0.npz" % (t - 1)
	loadfile1 = "evolved_brains/sim0/blue/perceptron_100_25_10_turn%d_gen0.npz" % (t - 1)

	try:
		brain0 = ScoredPerceptron.load(loadfile0)
		pop0 = Collective(1, 0, 0, [brain0])
		robot0 = Robot(pop0)
		player0 = Player(name="red %d" % t, robot=robot0)

		brain1 = ScoredPerceptron.load(loadfile1)
		pop1 = Collective(1, 0, 0, [brain1])
		robot1 = Robot(pop1)
		player1 = Player(name="blu %d" % t, robot=robot1)

		opts = Options(print_info=True)

		r = Runner(players=[player0, player1], options=opts)
		r.run()
	except:
		print "failed to run turn %d" % t
