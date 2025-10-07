# A . Elaboração de um conjunto de scripts e funções em Python, NumPy e SciPy para realizar as tarefas de preparação dos dados e Feature Engineering 

# X Crie um script e grave-o com o nome ‘mainActivity.py’. Este script será utilizado na chamada de todas as funções indicadas abaixo.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print("Ambiente configurado com sucesso!")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")


# Descarregue os dados do site https://github.com/spl-icsforth/FORTH_TRACE_DATASET. 
# 🖮 Elabore uma rotina que carregue os dados relativos a um indivíduo e os devolva num Array NumPy. 


# Cada ficheiro CSV segue o formato seguinte:

# Column1: Device ID
# Column2: accelerometer x
# Column3: accelerometer y
# Column4: accelerometer z
# Column5: gyroscope x
# Column6: gyroscope y
# Column7: gyroscope z
# Column8: magnetometer x
# Column9: magnetometer y
# Column10: magnetometer z
# Column11: Timestamp
# Column12: Activity Label

# Table 1: LOCATIONS
# 1 Left Wrist
# 2 Right Wrist
# 3 Torso
# 4 Right Thigh
# 5 Left Ankle

# Table 2: ACTIVITY LABELS
# (Arrows (->) indicate transitions between activities)

# 1 stand
# 2 sit
# 3 sit and talk
# 4 walk
# 5 walk and talk
# 6 climb stairs (up/down)
# 7 climb stairs (up/down) and talk
# 8 stand -> sit
# 9 sit -> stand
# 10 stand -> sit and talk
# 11 sit and talk -> stand
# 12 stand -> walk
# 13 walk -> stand
# 14 stand -> climb stairs (up/down), stand -> climb stairs (up/down) and talk
# 15 climb stairs (up/down) -> walk
# 16 climb stairs (up/down) and talk -> walk and talk


# path: ./FORTH_TRACE_DATASET/part0

# %%
def load_data(file_path):
	# Carregar os dados do ficheiro CSV para um DataFrame do Pandas
	df = pd.read_csv(file_path)
	df.columns = ['Device ID', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'Timestamp', 'Activity Label']
	# Converter o DataFrame para um Array NumPy
	# data_array = df.to_numpy()
	return df

# %%
# Carregar os dados de um indivíduo (exemplo)
file_path = './FORTH_TRACE_DATASET/part0/part0dev1.csv'  # Substitua pelo caminho do

df = load_data(file_path)
# add headers
df.tail()


# %%
# Carregar todos os dados de um participante
base_path='./FORTH_TRACE_DATASET/'

def load_all_data(participant_id):
	df_final = pd.DataFrame()

	for part in range(1, 6):  # part1 a part5
		part_folder = f'part{participant_id}'
		file_path = f'{base_path}{part_folder}/{part_folder}dev{part}.csv'
		try:
			df = load_data(file_path)
			df_final = pd.concat([df_final, df], ignore_index=True)
		except FileNotFoundError:
			pass
	return df_final

# example
df_all = load_all_data(0)
print(f"Total de amostras carregadas: {len(df_all)}")
# df_all.tail()

# %%
# Carregar em um só dataframe todos os participantes de 0 a 14

def load_all_participants():
	df_final = pd.DataFrame()

	for participant_id in range(0, 15):  # part0 a part14
		df_participant = load_all_data(participant_id)
		# Criar coluna 'Participant ID' e definir como primeira coluna
		df_participant['Participant ID'] = participant_id
		cols = df_participant.columns.tolist()
		cols = ['Participant ID'] + [col for col in cols if col != 'Participant ID']
		df_participant = df_participant[cols]

		df_final = pd.concat([df_final, df_participant], ignore_index=True)
	return df_final

# example
df_all_participants = load_all_participants()
print(f"Total de amostras carregadas de todos os participantes: {len(df_all_participants)}")

# %%
df_all_participants.tail()

# %%
df_all_participants.head()
# Análise e tratamento de Outliers: o objectivo será identificar e tratar outliers no dataset usando diferentes abordagens univariável e multivariável.  Para o efeito iremos os módulos dos vectores aceleração, giroscópio e magnetómetro. Seja 

# o vector aceleração, giroscópio e magnetómetro. O respectivo módulo é determinado recorrendo:

# 🖮 Elabore uma rotina que apresente simultaneamente o boxplot de cada atividade (coluna 12 – eixo horizontal)  relativo a todos os sujeitos e a uma das seguintes variáveis transformadas: módulo do vector de aceleração, módulo do vector de giroscópio e módulo do vector de magnetómetro). Sugere-se o uso da biblioteca matplotlib. 
# 🖎 Analise e comente a densidade de Outliers existentes no dataset transformado, isto é, nos módulos dos vectores aceleração, giroscópio e magnetómetro para cada atividade. Observe que a densidade é determinada recorrendo

# em que no é o número de pontos classificados como outliers e nr é o número total de pontos.
# 🖮 Escreva uma rotina que receba um Array de amostras de uma variável e identifique os outliers usando o teste Z-Score para um k variável (parâmetro de entrada).
# 🖎 Usando o Z-score implementado assinale todos as amostras consideradas outliers nos módulos dos vectores de aceleração, giroscópio e magnetómetro. Apresente plots em que estes pontos surgem a vermelho enquanto os restantes surgem a vermelho. Use k=3, 3.5 e 4.
# 🖎 Compare e discuta os resultados obtidos em 3.1 e 3.4.
# 🖮 Elabore uma rotina que implemente o algoritmo K-means para n (valor de entrada) clusters.
# 🖎 Determine os outliers no dataset transformado usando o  k-means. Experimente k clusters igual ao número de labels e compare com os resultados obtidos em 3.4. Ilustre graficamente os resultados usando plots 3D.  
# Bónus: poderá realizar um estudo análogo usando o algoritmo DBSCAN (sugere-se que recorra à biblioteca sklearn) 

# 🖮 Implemente uma rotina que injete ouliers com uma densidade igual ou superior a x% nas amostras de variável fornecida. Para o efeito deverá:
# A calcular a densidade de outliers existente no Array fornecido com nr pontos; observe que a densidade d é obtida por

# em que


# Se a densidade d for inferior a x, então deverá sortear  (x-d)% dos pontos não outliers de forma aleatória e para cada ponto selecionado deverá transformá-lo tal que
 
# em que μ e σ representam, respectivamente, os valores médio e o desvio padrão da amostra,  k é o limite especificado no ponto 3.3, s∈{-1,1} é uma variável escolhida de forma aleatória usando uma distribuição uniforme e q é uma variável aleatória uniforme no intervalo q ∈ [0,z[ em que z é a amplitude máxima do outlier relativamente a μ±kσ.

# 🖮 Elabore uma rotina que determine o modelo linear de ordem p. Para o efeito, a sua rotina deverá receber n amostras de treino de um vector de dimensão p , isto é, (xi,1, xi,2, xi,2, ... , xi,p) e a respectiva saída yi. A sua rotina deverá determinar o melhor vector de pesos β tal que 

# 🖎 Determine o modelo linear para o módulo aceleração usando uma janela com p valores anteriores. Usando a rotina desenvolvida no ponto 3.9 injete 10% de outliers no módulo da aceleração. Elimine esses outliers e substitua-os pelos valores previstos pelo modelo linear. Analise o erro de predição apresentando i) a distribuição do erro e ii) exemplos de plots contendo o valor previsto e real. Determine o melhor p para o seu modelo.
# 🖎 Repita 3.10 usando uma janela de dimensão p centrada no instante a prever. Deverá usar não só os p/2 valores anteriores e seguintes da variável que pretende prever bem como das restantes variáveis disponíveis (módulos disponíveis). Compare com os resultados obtidos em 3.10.

# Extração de informação característica: o objectivo será comprimir o espaço do problema, extraindo informação característica discriminante que permita implementar soluções eficazes do problema de classificação.

# 🖎 Usando as variáveis aplicadas na alínea 3.1, determine a significância estatísticas dos seus valores médios na diferentes atividades. Observe que poderá aferir a gaussianidade da distribuição usando, por exemplo, o teste Kolmogorov-Smirnov (vide documentação do SciPy). Para rever a escolha de testes estatísticos sugere-se a referência: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2881615/ .   Comente.
# 🖮 Desenvolva as rotinas necessárias à extração do feature set temporal e espectral sugerido no artigo  https://pdfs.semanticscholar.org/8522/ce2bfce1ab65b133e411350478183e79fae7.pdf. Para o efeito deverá:
# Ler o artigo e identificar o conjunto de features temporais e espectrais identificadas por estes autores
# Para cada feature deverá elaborar uma rotina para a respectiva extração 
# Usando as rotinas elaboradas no item anterior, deverá escrever o código necessário para extrair o vetor de features em cada instante.  
# Nota: Poderá usar as bibliotecas NumPy e SciPy. Qualquer outra biblioteca deverá ser identificada. 

# 🖮 Desenvolva o código necessário para implementar o PCA de um feature set.
# 🖎 Determine a importância de cada vetor na explicação da variabilidade do espaço de features. Note que deverá normalizar as features usando o z-score. Quantas variáveis deverá usar para explicar 75% do feature set? 
# Indique como poderia obter as features relativas a esta compressão e exemplifique para um instante à sua escolha.
# Indique as vantagens e as limitações desta abordagem. 
# 🖮 Desenvolva o código necessário para implementar o Fisher feature Score e o ReliefF.
# 🖎 Indentifique as 10 melhores features de acordo com o Fisher Score e o ReliefF.
# Indique como poderia obter as features relativas a esta compressão e exemplifique para um instante à sua escolha.
# Indique as vantagens e as limitações desta abordagem. 


# %%
