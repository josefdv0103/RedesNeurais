from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

rede = FeedForwardNetwork()

#Camada de entrada com 2 neuronios
camadaEntrada = LinearLayer(2)

#Camada Oculta com 3 neuronios
camadaOculta = SigmoidLayer(3)

#Camada de Saida com 1 neuronio
camadaSaida = SigmoidLayer(1)

#Unidades de Bias
bias1 = BiasUnit()
bias2 = BiasUnit()

rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

#Fullconection quer dizer uma ligação de um neuronio com
#todos os outros da camada seguinte
entradaOculta = FullConnection(camadaEntrada, camadaOculta)
ocultaSaida = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2, camadaSaida)

#Garante que as ligações sejam feitas
rede.sortModules()

#Os pesos são gerados automaticamente 
print(rede)
print(entradaOculta.params)
print(ocultaSaida.params)
print(biasOculta.params)
print(biasSaida.params)

