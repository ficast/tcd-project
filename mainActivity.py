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

# %%
# Export CSV
df_all_participants.to_csv('all_participants_data.csv', index=False)

# %%
#Load from CSV
df_all_participants = pd.read_csv('all_participants_data.csv')
print(df_all_participants.head())


# %%
def plot_boxplot_by_activity(df, variable):
	plt.figure(figsize=(12, 6))
	df.boxplot(column=variable, by='Activity Label', grid=False)
	plt.title(f'Boxplot of {variable} by Activity Label')
	plt.suptitle('')
	plt.xlabel('Activity Label')
	plt.ylabel(variable)
	plt.show()

# %%
#Seaborn version of plot_boxplot_by_activity
import seaborn as sns
def plot_boxplot_by_activity_seaborn(df, variable):
	plt.figure(figsize=(12, 6))
	sns.boxplot(x='Activity Label', y=variable, data=df)
	plt.title(f'Boxplot of {variable} by Activity Label')
	plt.xlabel('Activity Label')
	plt.ylabel(variable)
	plt.show()

# Exemplo de uso
# Calcular o módulo do vector de aceleração
df_all_participants['accel_magnitude'] = np.sqrt(df_all_participants['accel_x']**2 + df_all_participants['accel_y']**2 + df_all_participants['accel_z']**2)

# Calcular o módulo do vector de giroscópio
df_all_participants['gyro_magnitude'] = np.sqrt(df_all_participants['gyro_x']**2 + df_all_participants['gyro_y']**2 + df_all_participants['gyro_z']**2)

# Calcular o módulo do vector de magnetómetro
df_all_participants['magneto_magnitude'] = np.sqrt(df_all_participants['mag_x']**2 + df_all_participants['mag_y']**2 + df_all_participants['mag_z']**2)

# Análise e tratamento de Outliers: o objectivo será identificar e tratar outliers no dataset usando diferentes abordagens univariável e multivariável.  Para o efeito iremos os módulos dos vectores aceleração, giroscópio e magnetómetro. 
# Seja o vector aceleração, giroscópio e magnetómetro. O respectivo módulo é determinado recorrendo:

# 🖮 Elabore uma rotina que apresente simultaneamente o boxplot de cada atividade (coluna 12 – eixo horizontal)  
# relativo a todos os sujeitos e a uma das seguintes variáveis transformadas: módulo do vector de aceleração, 
# módulo do vector de giroscópio e módulo do vector de magnetómetro). Sugere-se o uso da biblioteca matplotlib. 
plot_boxplot_by_activity_seaborn(df_all_participants, 'accel_magnitude')
plot_boxplot_by_activity_seaborn(df_all_participants, 'gyro_magnitude')
plot_boxplot_by_activity_seaborn(df_all_participants, 'magneto_magnitude')

#Analise e comente a densidade de Outliers existentes no dataset transformado, 
# isto é, nos módulos dos vectores aceleração, giroscópio e magnetómetro 
# para cada atividade. Observe que a densidade é determinada recorrendo
# d = no/nr * 100
# em que no é o número de pontos classificados como outliers e nr é o número total de pontos.

# %%
def calculate_outlier_density(df, variable):
	# Calcular Q1, Q3 e IQR
	Q1 = df[variable].quantile(0.25)
	Q3 = df[variable].quantile(0.75)
	IQR = Q3 - Q1

	# Definir limites para outliers
	lower_bound = Q1 - 1.5 * IQR
	upper_bound = Q3 + 1.5 * IQR

	# Identificar outliers
	outliers = df[(df[variable] < lower_bound) | (df[variable] > upper_bound)]
	no = len(outliers)
	nr = len(df)
	density = (no / nr) * 100 if nr > 0 else 0
	return no, nr, density

# Uso
variables = ['accel_magnitude', 'gyro_magnitude', 'magneto_magnitude']
for var in variables:
	no, nr, density = calculate_outlier_density(df_all_participants, var)
	print(f'Variable: {var}, Outliers: {no}, Total: {nr}, Density: {density:.2f}%')



# Dado que a accel_magnitude tem uma densidade de outliers significativamente maior,
# vamos mudar o método de detecção de outliers para essa variável, dado que para 
# esse tipo específico de sensor pode ser necessário limites mais flexíveis.


# %%
# 🖮 Escreva uma rotina que receba um Array de amostras de uma variável e identifique os outliers 
# usando o teste Z-Score para um k variável (parâmetro de entrada).

def detect_outliers_z_score(data, k):
	mean = np.mean(data)
	std_dev = np.std(data)
	z_scores = (data - mean) / std_dev
	outliers = np.where(np.abs(z_scores) > k)[0]
	return outliers, z_scores


# Exemplo de uso

variables = ['accel_magnitude', 'gyro_magnitude', 'magneto_magnitude']

for var in variables:
	data = df_all_participants[var].values
	k_values = [3, 3.5, 4]
	for k in k_values:
		outliers, z_scores = detect_outliers_z_score(data, k)
		print(f'Variable: {var}, k={k}: Outliers detected: {len(outliers)}')


# 🖎 Usando o Z-score implementado assinale todos as amostras consideradas outliers nos 
# módulos dos vectores de aceleração, giroscópio e magnetómetro. 
# Apresente plots em que estes pontos surgem a vermelho enquanto os restantes surgem a azul.
#  Use k=3, 3.5 e 4.

# %%
# use seaborn for better plots, let's plot all three k for each variable in the same plot
# x axis is the activity label, y axis is the variable value
# blue points are normal, red points are outliers, use pastel colors
# for each variavle plot all three k values in the same plot side by side

#based on this one:

# def plot_boxplot_by_activity_seaborn(df, variable):
# 	plt.figure(figsize=(12, 6))
# 	sns.boxplot(x='Activity Label', y=variable, data=df)
# 	plt.title(f'Boxplot of {variable} by Activity Label')
# 	plt.xlabel('Activity Label')
# 	plt.ylabel(variable)
# 	plt.show()

import seaborn as sns
def plot_outliers_by_activity_seaborn(df, variable, k_values):
	plt.figure(figsize=(18, 6))
	for i, k in enumerate(k_values):
		plt.subplot(1, len(k_values), i + 1)
		data = df[variable].values
		outliers, z_scores = detect_outliers_z_score(data, k)
		df['Outlier'] = 'Normal'
		df.loc[outliers, 'Outlier'] = 'Outlier'
		sns.scatterplot(x='Activity Label', y=variable, hue='Outlier', data=df, palette={'Normal': 'blue', 'Outlier': 'red'}, alpha=0.5)
		plt.title(f'Outliers in {variable} (k={k})')
		plt.xlabel('Activity Label')
		plt.ylabel(variable)
		plt.legend()
	plt.tight_layout()
	plt.show()

variables = ['accel_magnitude', 'gyro_magnitude', 'magneto_magnitude']
k_values = [3, 3.5, 4]
for var in variables:
	plot_outliers_by_activity_seaborn(df_all_participants, var, k_values)


# %%
# Plotar a distribuição de cada variavel
def plot_distribution(df, variable):
	plt.figure(figsize=(10, 6))
	sns.histplot(df[variable], bins=50, kde=True)
	plt.title(f'Distribution of {variable}')
	plt.xlabel(variable)
	plt.ylabel('Frequency')
	plt.show()

for var in variables:
	plot_distribution(df_all_participants, var)


# 🖎 Compare e discuta os resultados obtidos em 3.1 e 3.4.


# Dados:

# Variable: accel_magnitude, Outliers: 1103326, Total: 3930798, Density: 28.07%
# Variable: gyro_magnitude, Outliers: 160794, Total: 3930798, Density: 4.09%
# Variable: magneto_magnitude, Outliers: 174273, Total: 3930798, Density: 4.43%

# Variable: accel_magnitude, k=3: Outliers detected: 85688
# Variable: accel_magnitude, k=3.5: Outliers detected: 57721
# Variable: accel_magnitude, k=4: Outliers detected: 40641
# Variable: gyro_magnitude, k=3: Outliers detected: 72113
# Variable: gyro_magnitude, k=3.5: Outliers detected: 42708
# Variable: gyro_magnitude, k=4: Outliers detected: 23096
# Variable: magneto_magnitude, k=3: Outliers detected: 18698
# Variable: magneto_magnitude, k=3.5: Outliers detected: 6631
# Variable: magneto_magnitude, k=4: Outliers detected: 3063


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
