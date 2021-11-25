from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

net = buildNetwork(4, 4, 3)

ds = SupervisedDataSet(4,3)

#ENTRADA
#Vitoria dos ultimos 8 jogos
#Posição na tabela 
#Posição do adversário na Tabela
#Desfalques
#Bias: Casa ou fora
#SAIDA
#(1,0,0) Alto
#(0,1,0) Médio
#(0,0,1) Baixo

ds.addSample((1, 2, 3, 1), (0, 0, 1)) #Risco Baixo
ds.addSample((1, 2, 3, 2), (0, 0, 1)) #Risco Baixo
ds.addSample((1, 2, 3, 3), (0, 1, 0)) #Risco Médio
ds.addSample((2, 2, 3, 3), (0, 1, 0)) #Risco Médio
ds.addSample((1, 1, 2, 3), (1, 0, 0)) #Risco Baixo
ds.addSample((2, 1, 3, 3), (0, 0, 1)) #Risco Baixo
ds.addSample((2, 1, 3, 2), (0, 0, 1)) #Risco Baixo
ds.addSample((2, 1, 3, 1), (0, 0, 1)) #Risco Baixo
ds.addSample((3, 2, 1, 3), (1, 0, 0)) #Risco Alto
ds.addSample((3, 2, 1, 2), (1, 0, 0)) #Risco Alto
ds.addSample((3, 2, 1, 1), (1, 0, 0)) #Risco Alto
ds.addSample((3, 3, 3, 3), (1, 0, 0)) #Risco Alto
ds.addSample((2, 2, 2, 2), (0, 1, 0)) #Risco Médio
ds.addSample((1, 1, 1, 1), (0, 1, 0)) #Risco Médio

trainer = BackpropTrainer(net, ds, learningrate = 0.1)

for i in range(0, 30000):
    erro = trainer.train()
    if i % 1000 == 0:
        print(trainer.train())

print(net.activate([1, 1, 1, 1]))
print(net.activate([2, 2, 2, 2]))