import pandas as pd
import numpy as np

def gerar():
    # Criando dados fictícios
    np.random.seed(42)
    dias = np.arange(1, 31)
    promocoes = np.random.randint(0, 2, size=30)  # 0 ou 1, indicando se houve promoção
    visitantes = np.random.randint(50, 200, size=30)  # Número de visitantes diários
    vendas = 10 * promocoes + 0.5 * visitantes + np.random.normal(0, 5, size=30)  # Fórmula com ruído

    # Criando DataFrame
    dados = pd.DataFrame({
        'Dia': dias,
        'Promocoes': promocoes,
        'Visitantes': visitantes,
        'Vendas': vendas
    })
    print(dados.head())
    return dados