from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

net = FeedForwardNetwork()

inLayer = LinearLayer(4)
hiddenLayer = SigmoidLayer(4)
outLayer = LinearLayer(3)

net.addInputModule(inLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

net.addConnection(in_to_hidden)
net.addConnection(hidden_to_out)

net.sortModules()

print(in_to_hidden.params)
print(hidden_to_out.params)
