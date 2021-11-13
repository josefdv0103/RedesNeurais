entradas = [-1, 7, 5]
pesos = [0.8, 0.1, 0]

#Função soma com parametros de entradas e pesos(a rede mais 
# simples que podemos ver)
def soma(e, p):
    s = 0
    for i in range(3):
        s += e[i]*p[i]    
    return s

s = soma(entradas, pesos)
print(s)
#Função de ativação nesse caso a ativação vem de uma condição
#de ser maior que um ou não
def stepfunction(soma):
    if soma >= 1:
        return 1
    return 0

r = stepfunction(s)
print(r)