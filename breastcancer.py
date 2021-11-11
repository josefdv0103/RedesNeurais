import numpy as np
from sklearn import datasets


def sigmoid (soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)

base = datasets.load_breast_cancer()
entradas = base.data
valoresSaida = base.target
#Transformar valoresSaidas em um formato que condiza
#com o nosso algoritmo
saidas = np.empty([569, 1], dtype = int)

for i in range(569):
    saidas[i] = valoresSaida[i]

#PESOS ALEATORIAMENTE SELECIONADOS
#Randomizar a entrada 2 neuronios na entrada 3 na camada oculta
pesos0 = 2 * (np.random.random((30, 5))) - 1
#Randomizar a entrada 3 neuronios na camada oculta 1 na saida
pesos1 = 2 * (np.random.random((5, 1))) - 1

#Quantidades de vezes que iremos atualizar os pesos
epocas = 10
taxaAprendizagem = 0.3
momento = 1

for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)

    somaSinapses1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapses1)

    erroCamadaSaida = (saidas - camadaSaida)
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print(f'Erro = {mediaAbsoluta}')

    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida

    pesos1Transposta = pesos1.T
    deltaSaidaXPeso = np.dot(deltaSaida, pesos1Transposta)
    deltaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)

    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = np.dot(camadaOcultaTransposta, deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)

    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = np.dot(camadaEntradaTransposta, deltaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)
    
