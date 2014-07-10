import rg
import random

def around(loc):
	return rg.locs_around(loc, filter_out=('invalid', 'obstacle'))

class Robot:
	def act(self, game):
		moveable = [loc for loc in around(self.location) if loc not in game['robots']]
		if len(moveable) > 0:
			return ['move', random.choice(moveable)]
		else:
			return ['attack', random.choice(around(self.location))]
