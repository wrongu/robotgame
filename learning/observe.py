import mlpbot
import neuralnets
import corners3
import numpy as np
# debug imports
import rg
from termcolor import cprint

def feature_display(feature_vector):
	for x in range(5):
		for y in range(5):
			block_index = 4 * (5 * x + y)
			bgcolor = "on_white"
			txcolor  = "blue"
			ally_hp  = feature_vector[block_index + 0]
			enemy_hp = feature_vector[block_index + 1]
			spawn    = feature_vector[block_index + 2]
			obstacle = feature_vector[block_index + 3]
			content = "  "
			if ally_hp > 0:
				txcolor = "green"
				content = "%2d" % (ally_hp * rg.settings.robot_hp)
			elif enemy_hp > 0:
				txcolor = "red"
				content = "%2d" % (enemy_hp * rg.settings.robot_hp)
			if spawn > 0:
				bgcolor = "on_grey"
			if obstacle > 0:
				content = "##"
			cprint(" "+content+" ", txcolor, bgcolor, end="")
		print ""

class ObservationBot(object):

	def __init__(self, teacher, student, rate=0.02):
		self._teacher = teacher
		self._student = student
		self._rate    = rate
		self.samples = []

	def init_tracking(self, game):
		if not hasattr(self, 'last_update'):
			self.last_update = -1
		if self.last_update < game.turn:
			# if game.turn > 1:
			# 	print "%f,%f" % (self.average_error,self.percent_correct)
			# else:
			# 	print "average error,percent correct"
			print game.turn
			self.last_update = game.turn
			self.n_observations = 0
			self.average_error = 0.0
			self.percent_correct = 0.0

	def act(self, game):
		# init data tracking
		self.init_tracking(game)
		self.n_observations += 1

		# copy state to teacher to get the "ground truth" decision
		self._teacher.hp        = self.hp
		self._teacher.player_id = self.player_id
		self._teacher.robot_id  = self.robot_id
		self._teacher.location  = self.location
		wise_decision = self._teacher.act(game)
		# copy state to student to see what it thinks is correct
		self._student.hp        = self.hp
		self._student.player_id = self.player_id
		self._student.robot_id  = self.robot_id
		self._student.location  = self.location
		student_decision = self._student.act(game)

		p = 1.0 if student_decision == wise_decision else 0.0
		self.percent_correct += (p - self.percent_correct) / self.n_observations

		x,y = self.location
		brain_choices = [
			['move', (x-1, y)],   ['move', (x+1, y)],   ['move', (x, y-1)],   ['move', (x, y+1)],
			['attack', (x-1, y)], ['attack', (x+1, y)], ['attack', (x, y-1)], ['attack', (x, y+1)],
			['guard'], ['suicide']]
		brain_index = brain_choices.index(wise_decision)

		if brain_index < 0:
			print "error!", wise_decision, "not an option in", brain_choices
		else:
			student_brain = self._student.recall.get(self.robot_id, {}).get("brain")
			if student_brain:
				ideal_output = np.zeros((len(brain_choices), 1))
				ideal_output[brain_index] = 1.0
				senses = self._student.sense_environment(game)
				self.samples.append((senses, ideal_output))
				# student_brain.set_input_vector(senses)
				# student_brain.propagate()
				# err = student_brain.backpropagate(ideal_output, learning_rate=self._rate)
				# self.average_error += (err - self.average_error) / self.n_observations
				# debug
				# feature_display(senses)
				# print sorted(zip(student_brain.get_output_vector(), brain_choices), reverse=True)
				# student_brain.propagate()
				# print sorted(zip(student_brain.get_output_vector(), brain_choices), reverse=True)
				# raw_input()
			else:
				print "no brain!"
		return wise_decision

if __name__ == '__main__':
	from rgkit.run import Runner, Options
	from rgkit.game import Player
	from rgkit.settings import settings
	from sys import argv

	rt = 0.2
	if len(argv) > 1:
		rt = float(argv[1])

	teacher_class = corners3.Robot

	student_brain = neuralnets.MultiLayerPerceptron([100,25,10], random=0.1)
	mlp = mlpbot.Robot(False)
	mlp.override_next_brain(lambda s: student_brain)

	ob = ObservationBot(teacher_class(), mlp, rate=rt)

	# run the observation match
	settings["max_turns"] = 400
	red = Player(name="Corners 3", robot=corners3.Robot())
	blue = Player(name="senpai", robot=ob)
	opts = Options()
	r = Runner(players=[red,blue], options=opts)
	r.run()

	student_brain.save(".", "incubated_%d_%f" % (settings["max_turns"], rt))

	print "~~observation done~~"
	print "collected %d training examples" % (len(ob.samples))

	deep_brain = neuralnets.MultiLayerPerceptron([100,25,10], random=0.1)
	deep_brain.deep_learn(ob.samples)

	print "training shallow brain"
	shallow_brain = neuralnets.MultiLayerPerceptron([100,25,10], random=0.1)
	shallow_brain.train(ob.samples)

	# match between student_brain and deep_brain
	mlp_shallow  = mlpbot.Robot(False)
	mlp_deep     = mlpbot.Robot(False)

	mlp_shallow.override_next_brain(lambda s: shallow_brain)
	mlp_deep.override_next_brain(lambda s: deep_brain)

	opts.print_info = True
	r = Runner(players=[Player(name="shallow", robot=mlp_shallow), Player(name="deep", robot=mlp_deep)], options=opts)
	r.run()