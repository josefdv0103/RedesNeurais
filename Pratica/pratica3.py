from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

net = buildNetwork(3, 3, 3)

ds = SupervisedDataSet(3, 3)

#Praticamente cores puras
ds.addSample((1, 3, 3), (1, 1, 1))#Maduro
ds.addSample((3, 1, 3), (0, 1, 0))#Não-maduro
ds.addSample((3, 3, 1), (1, 0, 0))#Verde

#Divisão de cores
ds.addSample((2, 2, 3), (0, 0, 1))#Vermelho-Amarelo 50/50(Pré-maduro)
ds.addSample((2, 3, 3), (0, 0, 1))#Vermelho-Amarelo 75/25 (Pré-maduro)
ds.addSample((3, 2, 2), (0, 1, 0))#Amarelo-verde 50/50 (Não-maduro)
ds.addSample((3, 2, 3), (0, 1, 0))#Amarelo-verde 75/25 (Não-maduro)
ds.addSample((3, 2, 2), (0, 1, 0))#Verde-amarelo 50/50 (Não-maduro)
ds.addSample((3, 3, 2), (1, 0, 0))#Verde-amarelo 75/25 (Verde)

trainer = BackpropTrainer(net, ds, learningrate = 0.01)

for i in range(0, 300000):
    erro = trainer.train()
    if i % 100000 == 0:
        print(trainer.train())

print(net.activate([1, 3, 3]))
print(net.activate([3, 3, 1]))