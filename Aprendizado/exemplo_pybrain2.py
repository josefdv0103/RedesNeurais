from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer

#Nossa rede em que stamos definindo as entradas, camadas
#ocultas e saídas
rede  = buildNetwork(2, 3, 1)

#Especificando funcoes
#rede  = buildNetwork(2, 3, 1, outclass = SoftmaxLayer
# hiddenclass = SigmoidLayer, bias = False)

#Entrada, oculta, saída, e baias
#print(rede['in'])
#print(rede['hidden0'])
#print(rede['out'])
#print(rede['bias'])

#Dois atributos previsores e uma classe(operador XOR)
#0,0 quer dizer 0; 0,1 quer dizer 1;....

base = SupervisedDataSet(2, 1)
base.addSample((0, 0), (0,))
base.addSample((0, 1), (1,))
base.addSample((1, 0), (1,))
base.addSample((1, 1), (0,))
#print(base['input'])
#print(base['target'])

treinamento = BackpropTrainer(rede, dataset = base, learningrate = 0.01,
momentum = 0.06)

for i in range(0, 3000):
    erro = treinamento.train()
    if i % 1000 == 0:
        print(f'Erro: {erro} %')

#Os valores que foram calculados
print(rede.activate([0, 0]))
print(rede.activate([1, 0]))
print(rede.activate([0, 1]))
print(rede.activate([1, 1]))