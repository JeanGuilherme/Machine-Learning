import matplotlib.pyplot as plt

def executar(dados):
    # Relacionamento entre visitantes e vendas
    plt.scatter(dados['Visitantes'], dados['Vendas'])
    plt.xlabel('Visitantes')
    plt.ylabel('Vendas')
    plt.title('Relação entre Visitantes e Vendas')
    plt.show()

    # Relacionamento entre promoções e vendas
    plt.scatter(dados['Promocoes'], dados['Vendas'])
    plt.xlabel('Promoções')
    plt.ylabel('Vendas')
    plt.title('Relação entre Promoções e Vendas')
    plt.show()