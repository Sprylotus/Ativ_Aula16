import polars as pl
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt

ENDERECO_DADOS = r'./dados/'

# LENDO OS DADOS DO ARQUIVO PARQUET
try:
    print('\nIniciando leitura do arquivo parquet...')

    # Pega o tempo inicial
    inicio = datetime.now()

    # Scan_parquet: Cria um plano de execução preguiçoso para a leitura do parquet
    df_bolsa_fam_plano = pl.scan_parquet(ENDERECO_DADOS + 'bolsa_familia.parquet')

    # Executa as operações lazys e coleta os resultados
    df_bolsa_fam = df_bolsa_fam_plano.collect()

    print(df_bolsa_fam)

    # Pega o tempo final
    fim = datetime.now()

    print(f'Tempo de execução para leitura do parquet: {fim - inicio}')
    print('\nArquivo parquet lido com sucesso!')

except ImportError as e: 
    print(f'Erro ao ler os dados do parquet: {e}')

try:
    print('Calculando medidas de posição...')
    
    array_val_parcela = np.array(df_bolsa_fam['VALOR PARCELA'])

    media_val_parcela = np.mean(array_val_parcela)
    mediana_val_parcela = np.median(array_val_parcela)
    dist_media_mediana = abs((media_val_parcela - mediana_val_parcela) / mediana_val_parcela) * 100

    max = np.max(array_val_parcela)
    min = np.min(array_val_parcela)
    amplitude_total = max - min

    q1 = np.quantile(array_val_parcela, 0.25, method='weibull')
    q2 = np.quantile(array_val_parcela, 0.50, method='weibull')
    q3 = np.quantile(array_val_parcela, 0.75, method='weibull')

    iqr = q3 - q1

    limite_sup = q3 + (1.5 * iqr)
    limite_inf = q1 - (1.5 * iqr)

    print('\nMEDIDAS DE TENDÊNCIA CENTRAL:')
    print(30*'-')
    print(f'Média: {media_val_parcela:.2f}')
    print(f'Mediana: {mediana_val_parcela:.2f}')
    print(f'Distância: {dist_media_mediana:.2f}')

    print('\nMEDIDAS DE DISPERSÃO:')
    print(30*'-')
    print('Máximo: ', max)
    print('Mínimo: ', min)
    print('Amplitude Total: ', amplitude_total)

    print('\nMEDIDAS DE POSIÇÃO:')
    print(30*'-')
    print('Mínimo: ', min)
    print(f'Limite Inferior: {limite_inf:.2f}')
    print('Q1 (25%): ', q1)
    print('Q2 (50%): ', q2)
    print('Q3 (75%): ', q3)
    print(f'IQR: {iqr:.2f}')
    print(f'Limite Superior: {limite_sup:.2f}')
    print('Máximo: ', max)

except ImportError as e:
    print(f'Erro ao obter informações sobre medidas estatísticas: {e}')
    exit()


# Processamento e visualização 
# (12 estados com maior valor de parcela)
try:
    print('Calculando os 12 estados com maior valor de parcelas e gerando gráfico de barras e boxplot...')

    # Agrupar por UF e somar o valor das parcelas
    df_estado_parcelas = df_bolsa_fam.group_by('UF').agg(pl.col('VALOR PARCELA').sum().alias('TOTAL PARCELA'))

    # Ordenar de forma decrescente e pegar os 12 primeiros
    df_estado_parcelas = df_estado_parcelas.sort('TOTAL PARCELA', descending=True).head(12)

    # Exibir o DataFrame resultante
    print(df_estado_parcelas)

    print('\nAo analisar as estatísticas de distância entre a média e a mediana (1%) e o valor do IQR abaixo do limite inferior, é possível concluir que existe uma tendência de homogeneidade nos dados, o que sugere que a média pode ser uma medida confiável para a análise. No entanto, apesar dessa tendência de baixa dispersão, ao observar a amplitude total (R$4.639,00), que resulta da diferença entre os valores máximo e mínimo, nota-se a presença de valores extremos (outliers), tanto para cima quanto para baixo, o que pode ser facilmente verificado no gráfico. Assim, com base nessa análise, é recomendada uma revisão dos valores distribuídos, com o objetivo de reduzir ou investigar mais a fundo essas discrepâncias, verificando, por exemplo, se existem situações específicas que justifiquem esses valores. Isso ajudaria a tornar a distribuição do benefício mais justa e transparente para os beneficiários. Para fins de informação, conforme mostrado no gráfico, entre os 12 estados que mais distribuem o benefício, destacam-se São Paulo e Bahia, enquanto Paraná e Rio Grande do Sul são os que distribuem em menor quantidade entre os 12.')

    # Criar a figura com dois subgráficos lado a lado
    plt.subplots(1, 2, figsize=(15, 6))

    # Gerar gráfico de barras
    plt.subplot(1, 2, 1)

    plt.bar(df_estado_parcelas['UF'], df_estado_parcelas['TOTAL PARCELA'])
    plt.xlabel('Estado (UF)', fontsize=12)
    plt.ylabel('Valor das Parcelas', fontsize=12)
    plt.title('Top 12 Estados com Maior Total de Parcelas', fontsize=14)
    plt.xticks(df_estado_parcelas['UF'], rotation=45, ha='right')

    # Gerar boxplot
    plt.subplot(1, 2, 2)
    array_valor_parcela = np.array(df_bolsa_fam['VALOR PARCELA'])
    plt.boxplot(array_valor_parcela, vert=False)
    plt.title('Distribuição dos Valores das Parcelas', fontsize=14)

    # Ajustar layout
    plt.tight_layout()

    # Exibir gráficos
    plt.show()

    print('Gráficos gerados com sucesso!')

except ImportError as e:
    print(f'Erro ao visualizar os dados: {e}')
