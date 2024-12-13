import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def treinar(dados):
    # Seleção de variáveis
    X = dados[['Promocoes', 'Visitantes']]
    y = dados['Vendas']

    # Divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinando o modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Fazendo previsões
    y_pred = modelo.predict(X_test)

    # Avaliando o modelo
    erro_medio = mean_absolute_error(y_test, y_pred)
    print(f'Erro Médio Absoluto: {erro_medio:.2f}')
    
    #Interpretando resultados 
    print('Coeficientes:', modelo.coef_)
    print('Intercepto:', modelo.intercept_)
    
    return modelo
    
def fazer_previsoes(modelo):
    # Exemplo de previsão
    nova_entrada = pd.DataFrame({'Promocoes': [1], 'Visitantes': [150]})
    previsao = modelo.predict(nova_entrada)
    print(f'Previsão de Vendas: {previsao[0]:.2f}')