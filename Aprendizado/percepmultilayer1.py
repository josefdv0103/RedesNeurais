import numpy as np

def sigmoid (soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)


entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([[0], [1] ,[1] ,[0]])

#PESOS PRÉ-DETERMINADOS
#Pesos da entrada para a camada oculta
#pesos0 = np.array([[-0.424, -0.740, -0.961], [0.358, -0.577, -0.469]])
#Pesos da camada oculta para a saída
#pesos1 = np.array([[-0.017], [-0.893], [0.148]])

#PESOS ALEATORIAMENTE SELECIONADOS
#Randomizar a entrada 2 neuronios na entrada 3 na camada oculta
pesos0 = 2 * (np.random.random((2, 3))) - 1
#Randomizar a entrada 3 neuronios na camada oculta 1 na saida
pesos1 = 2 * (np.random.random((3, 1))) - 1

#Quantidades de vezes que iremos atualizar os pesos
epocas = 1000000
taxaAprendizagem = 0.6
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
    
