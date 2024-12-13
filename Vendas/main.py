import dados_ficticios 
import visualizar_dados
import construir_modelo

dados = dados_ficticios.gerar()

visualizar_dados.executar(dados)

modelo = construir_modelo.treinar(dados)

construir_modelo.fazer_previsoes(modelo)

