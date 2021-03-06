from neuralnets import MultiLayerPerceptron
import numpy as np

p = MultiLayerPerceptron([2,3,2], random=True)

target0 = np.matrix([ 0.8, 0.1]).transpose()
target1 = np.matrix([0.4, 0.9]).transpose()

p.set_input(0,1.0)
p.set_input(1,0.0)
p.propagate()
print p.get_output(0), p.get_output(1)
p.set_input(0,0.0)
p.set_input(1,1.0)
p.propagate()
print p.get_output(0), p.get_output(1)
print "------------------------"

for i in range(1000):
	p.set_input(0,1.0)
	p.set_input(1,0.0)
	p.propagate()
	p.backpropagate(target0, learning_rate=0.8)
	p.set_input(0,0.0)
	p.set_input(1,1.0)
	p.propagate()
	p.backpropagate(target1, learning_rate=0.8)

p.set_input(0,1.0)
p.set_input(1,0.0)
p.propagate()
print p.get_output(0), p.get_output(1)
p.set_input(0,0.0)
p.set_input(1,1.0)
p.propagate()
print p.get_output(0), p.get_output(1)