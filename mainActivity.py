# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
print("Ambiente configurado com sucesso!")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Parte 1 e 2

# %% Load Data
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
df.tail()

# # %%
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

df_all = load_all_data(0)
print(f"Total de amostras carregadas: {len(df_all)}")
# df_all.tail()

# # %%
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

# # example
df_all_participants = load_all_participants()
print(f"Total de amostras carregadas de todos os participantes: {len(df_all_participants)}")

# # %%
# Export CSV
# df_all_participants.to_csv('all_participants_data.csv', index=False)

# %%
# Load from CSV
df_all_participants = pd.read_csv('all_participants_data.csv')
# print(df_all_participants.head())

# Parte 3

# %%
# def plot_boxplot_by_activity(df, variable):
# 	plt.figure(figsize=(12, 6))
# 	df.boxplot(column=variable, by='Activity Label', grid=False)
# 	plt.title(f'Boxplot of {variable} by Activity Label')
# 	plt.suptitle('')
# 	plt.xlabel('Activity Label')
# 	plt.ylabel(variable)
# 	plt.show()

# %% 3.1 
import seaborn as sns
def plot_boxplot_by_activity_seaborn(df, variable):
	plt.figure(figsize=(12, 6))
	sns.boxplot(x='Activity Label', y=variable, data=df)
	plt.title(f'Boxplot of {variable} by Activity Label')
	plt.xlabel('Activity Label')
	plt.ylabel(variable)
	plt.show()

# Calcular o módulo do vector de aceleração
df_all_participants['accel_magnitude'] = np.sqrt(df_all_participants['accel_x']**2 + df_all_participants['accel_y']**2 + df_all_participants['accel_z']**2)

# Calcular o módulo do vector de giroscópio
df_all_participants['gyro_magnitude'] = np.sqrt(df_all_participants['gyro_x']**2 + df_all_participants['gyro_y']**2 + df_all_participants['gyro_z']**2)

# Calcular o módulo do vector de magnetómetro
df_all_participants['magneto_magnitude'] = np.sqrt(df_all_participants['mag_x']**2 + df_all_participants['mag_y']**2 + df_all_participants['mag_z']**2)

plot_boxplot_by_activity_seaborn(df_all_participants, 'accel_magnitude')
plot_boxplot_by_activity_seaborn(df_all_participants, 'gyro_magnitude')
plot_boxplot_by_activity_seaborn(df_all_participants, 'magneto_magnitude')

# %% 3.2
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

# %% 3.3
def detect_outliers_z_score(data, k):
	mean = np.mean(data)
	std_dev = np.std(data)
	z_scores = (data - mean) / std_dev
	outliers = np.where(np.abs(z_scores) > k)[0]
	return outliers, z_scores


# %% 3.4
variables = ['accel_magnitude', 'gyro_magnitude', 'magneto_magnitude']

for var in variables:
	data = df_all_participants[var].values
	k_values = [3, 3.5, 4]
	for k in k_values:
		outliers, z_scores = detect_outliers_z_score(data, k)
		print(f'Variable: {var}, k={k}: Outliers detected: {len(outliers)}')

# %%
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


# 3.5 - Compare e discuta os resultados obtidos em 3.1 e 3.4.

# Em 3.1, observamos que a magnitude do acelerômetro apresenta uma densidade 
# de outliers significativamente alta (28.07%), enquanto as magnitudes do 
# giroscópio e magnetômetro têm densidades mais baixas (4.09% e 4.43%, 
# respectivamente). Isso sugere que os dados do acelerômetro são mais 
# suscetíveis a variações extremas, possivelmente devido a movimentos bruscos
# ou erros de medição.
# Em 3.4, ao aplicar o teste Z-Score com diferentes valores de k,
# observamos que o número de outliers detectados diminui à medida que k aumenta.
# Isso é esperado, pois um k maior torna o critério para classificar um ponto
# como outlier mais rigoroso. A magnitude do acelerômetro continua a mostrar
# o maior número de outliers detectados, reforçando a conclusão de que esses
# dados são mais voláteis.

# Dados encontrados:

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


# %% 3.6
from sklearn.cluster import KMeans

def k_means_clustering(data, n_clusters):
	kmeans = KMeans(n_clusters=n_clusters, random_state=42)
	kmeans.fit(data)
	return kmeans.labels_, kmeans.cluster_centers_

# 3.7
clustering_data = df_all_participants[['accel_magnitude', 'gyro_magnitude', 'magneto_magnitude']].values
n_clusters = 16  # Número de atividades
labels, centers = k_means_clustering(clustering_data, n_clusters)
df_all_participants['Cluster Label'] = labels

# %% Visualizar os clusters
def plot_clusters(df, x, y):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, hue='Cluster Label', data=df, palette='tab10', alpha=0.5)
    plt.title('K-Means Clustering of Sensor Magnitudes')
    plt.xlabel('Acceleration Magnitude')
    plt.ylabel('Gyroscope Magnitude')
    plt.legend(title='Cluster Label')
    plt.show()


plot_clusters(df_all_participants, 'accel_magnitude', 'gyro_magnitude')
plot_clusters(df_all_participants, 'accel_magnitude', 'magneto_magnitude')
plot_clusters(df_all_participants, 'gyro_magnitude', 'magneto_magnitude')

# %% Comparação entre rótulos de atividade e clusters
contingency_table = pd.crosstab(df_all_participants['Activity Label'], df_all_participants['Cluster Label'])
print("Contingency Table between Activity Labels and Cluster Labels:")
print(contingency_table)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 8))
sns.heatmap(contingency_table, annot=False, fmt="d", cmap="YlGnBu", linewidths=0.5)
plt.title("Contingency Table: Activity vs Cluster (Heatmap)")
plt.xlabel("Cluster Label")
plt.ylabel("Activity Label")
plt.show()

# %% Calcular % de acerto dos clusters por atividade
def cluster_accuracy_by_activity(contingency_table):
    total = contingency_table.sum().sum()
    correct = 0
    activity_hits = {}
    for activity in contingency_table.index:
        # Para cada atividade, pega o cluster com mais amostras
        cluster = contingency_table.loc[activity].idxmax()
        hits = contingency_table.loc[activity, cluster]
        activity_hits[activity] = hits / contingency_table.loc[activity].sum() * 100
        correct += hits
    overall_accuracy = correct / total * 100
    print("Percentual de acerto por atividade (cluster majoritário):")
    for activity, pct in activity_hits.items():
        print(f"Atividade {activity}: {pct:.2f}%")
    print(f"\nAcerto global (cluster majoritário por atividade): {overall_accuracy:.2f}%")
    return activity_hits, overall_accuracy

# Uso:
activity_hits, overall_accuracy = cluster_accuracy_by_activity(contingency_table)

# %% 3.8
# ---------- 3.8 (Injeção de Outliers) ----------
def inject_outliers(data, x_percent, k, z, seed=None):
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float).copy()

    mu = data.mean()
    sigma = data.std(ddof=0)

    lower, upper = mu - k*sigma, mu + k*sigma
    mask_out = (data < lower) | (data > upper)
    nr = data.size
    no = int(mask_out.sum())
    d = no / nr * 100.0

    if d >= x_percent:
        return data

    # sortear entre os NÃO-outliers
    non_idx = np.where(~mask_out)[0]
    needed = math.ceil(((x_percent - d) / 100.0) * nr)
    needed = min(needed, non_idx.size)
    sel = rng.choice(non_idx, size=needed, replace=False)

    s = rng.choice([-1, 1], size=needed)
    q = rng.uniform(0.0, z, size=needed)

    # Garante que os novos valores realmente fiquem fora dos limites
    data[sel] = mu + s * k * (sigma + q)
    return data

# %% checkar porcentagem de outliers por variavel
def outlier_percentage(data, k):
    mu = data.mean()
    sigma = data.std(ddof=0)
    lower, upper = mu - k*sigma, mu + k*sigma
    mask_out = (data < lower) | (data > upper)
    no = mask_out.sum()
    nr = data.size
    d = no / nr * 100.0
    return d

variables = ['accel_magnitude', 'gyro_magnitude', 'magneto_magnitude']
k = 4.0
x_percent = 5.0
z = 0.1

for var in variables:
    data = df_all_participants[var].values
    d_before = outlier_percentage(data, k)
    data_with_outliers = inject_outliers(data, x_percent, k, z, seed=42)
    d_after = outlier_percentage(data_with_outliers, k)
    print(f'Variable: {var}, Outlier % before: {d_before:.2f}%, after injection: {d_after:.2f}%')

# %% 3.9 e 3.10
# ---------- 3.9 (OLS) ----------
def linear_model_fit(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1, 1)
    Xd = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.inv(Xd.T @ Xd) @ (Xd.T @ y)
    return beta.ravel()

def linear_model_predict(X, beta):
    X = np.asarray(X, float)
    Xd = np.column_stack([np.ones(len(X)), X])
    return (Xd @ beta).ravel()

# ---------- util: matriz de lags para AR(p) ----------
def make_lag_matrix(y, p):
    """
    Retorna X, y_target alinhados para AR(p): y_t ~ [y_{t-1},...,y_{t-p}]
    """
    y = np.asarray(y, float)
    n = len(y)
    rows = n - p
    X = np.zeros((rows, p), float)
    for j in range(p):
        X[:, j] = y[p-1-j : n-1-j]   # col j é y_{t-1-j}
    y_target = y[p:]
    return X, y_target

# ---------- detecção de outliers por μ ± kσ (do exercício) ----------
def outlier_mask_mu_k_sigma(y, k):
    mu = np.mean(y)
    s = np.std(y, ddof=0)
    lower, upper = mu - k*s, mu + k*s
    return (y < lower) | (y > upper)

# ---------- substituição sequencial por previsão do AR(p) ----------
def replace_outliers_with_ar(y, p, k, beta=None):
    """
    Detecta outliers por μ±kσ e substitui-os por previsões do AR(p).
    Substituição sequencial (passo à frente) para não propagar outliers.
    """
    y = np.asarray(y, float).copy()
    # 1) treina o AR(p) na série "como está" (ou passe beta pronto)
    if beta is None:
        X, yt = make_lag_matrix(y, p)
        beta = linear_model_fit(X, yt)

    # 2) mascara de outliers na série inteira
    mask = outlier_mask_mu_k_sigma(y, k)

    # 3) varre do índice p em diante substituindo quando necessário
    for t in range(p, len(y)):
        if mask[t]:
            # usa os p valores imediatamente anteriores (já possivelmente corrigidos)
            x_t = y[t-1:t-p-1:-1]   # y_{t-1} ... y_{t-p}
            y[t] = linear_model_predict(x_t.reshape(1, -1), beta)[0]
    return y, beta


def make_centered_matrix(target, others=None, p=10):
    """
    target : array (n,)
        Série que você quer prever/imputar.
    others : dict[str, array] | DataFrame | None
        Outras séries disponíveis com o MESMO comprimento.
    p : int (par de preferência)
        Tamanho total da janela centrada. Usa p//2 passados e p//2 futuros.

    Retorna X, y_alvo alinhados.
    """
    target = np.asarray(target, float)
    n = len(target)
    h = p // 2
    start, end = h, n - h
    rows = end - start
    feats = []

    # lags passados da target (1..h)
    for j in range(1, h+1):
        feats.append(target[start-j:end-j])

    # leads futuros da target (1..h)
    for j in range(1, h+1):
        feats.append(target[start+j:end+j])

    # outras variáveis: incluir lags/leads 0 (valor simultâneo) e, se quiser, também lags/leads
    if others is not None:
        if isinstance(others, dict):
            items = others.items()
        elif isinstance(others, pd.DataFrame):
            items = others.items()
        else:
            raise ValueError("others deve ser dict ou DataFrame")
        for name, arr in items:
            arr = np.asarray(arr, float)
            feats.append(arr[start:end])  # valor no tempo t
            # pode opcionalmente incluir lags/leads:
            # for j in range(1, h+1):
            #     feats.append(arr[start-j:end-j])
            #     feats.append(arr[start+j:end+j])

    X = np.column_stack(feats) if feats else np.empty((rows, 0))
    y_center = target[start:end]
    return X, y_center


# %% 3.10

# Parâmetros
p = 10  # número de lags
k = 3.0  # limite para outlier
x_percent = 10.0  # densidade alvo de outliers
z = 0.5  # amplitude extra para garantir que os outliers fiquem fora

# 1. Série original
y_orig = df_all_participants['accel_magnitude'].values

# 2. Injetar outliers
y_out = inject_outliers(y_orig, x_percent, k, z, seed=42)

# 3. Treinar modelo AR(p) na série com outliers
X, y_target = make_lag_matrix(y_out, p)
beta = linear_model_fit(X, y_target)

# 4. Corrigir outliers usando o modelo
y_corr, _ = replace_outliers_with_ar(y_out, p, k, beta)

# 5. Calcular erro de predição (apenas nos pontos corrigidos)
mask_out = outlier_mask_mu_k_sigma(y_out, k)
err = y_corr - y_orig

# 6. Plot distribuição do erro
plt.figure(figsize=(10, 5))
sns.histplot(err[mask_out], bins=50, kde=True)
plt.title(f'Distribuição do erro de predição (p={p}) nos outliers corrigidos')
plt.xlabel('Erro (valor previsto - valor real)')
plt.ylabel('Frequência')
plt.show()

# 7. Plot exemplo de valores previstos vs reais (janela)
idx = np.where(mask_out)[0]
if len(idx) > 0:
    i0 = idx[0]
    plt.figure(figsize=(12, 6))
    plt.plot(range(i0-20, i0+20), y_orig[i0-20:i0+20], label='Valor real')
    plt.plot(range(i0-20, i0+20), y_corr[i0-20:i0+20], label='Valor corrigido')
    plt.scatter(idx, y_corr[idx], color='red', label='Outliers corrigidos')
    plt.title(f'Exemplo de correção de outliers (p={p})')
    plt.xlabel('Índice')
    plt.ylabel('Aceleração')
    plt.legend()
    plt.show()

# 8. Testar diferentes valores de p e mostrar RMSE
from sklearn.metrics import mean_squared_error

ps = [3, 5, 10, 20, 30, 50,]
for p_test in ps:
    X, y_target = make_lag_matrix(y_out, p_test)
    beta = linear_model_fit(X, y_target)
    y_corr, _ = replace_outliers_with_ar(y_out, p_test, k, beta)
    rmse = np.sqrt(mean_squared_error(y_orig[mask_out], y_corr[mask_out]))
    print(f'p={p_test}: RMSE nos outliers corrigidos = {rmse:.4f}')

# Valores encontrados:
# p=3: RMSE nos outliers corrigidos = 11.0489
# p=5: RMSE nos outliers corrigidos = 11.0291
# p=10: RMSE nos outliers corrigidos = 10.9061
# p=20: RMSE nos outliers corrigidos = 10.7447
# p=30: RMSE nos outliers corrigidos = 10.6596
# p=50: RMSE nos outliers corrigidos = 10.5845
# p=100: RMSE nos outliers corrigidos = 10.4509

# %% 3.11

p = 10  # janela total (deve ser par)
k = 3.0
x_percent = 10.0
z = 0.5

# Série original
y_orig = df_all_participants['accel_magnitude'].values
gyro = df_all_participants['gyro_magnitude'].values
magneto = df_all_participants['magneto_magnitude'].values

# Injetar outliers
y_out = inject_outliers(y_orig, x_percent, k, z, seed=42)

# Criar matriz de features centrada
others = {
    'gyro_magnitude': gyro,
    'magneto_magnitude': magneto
}
Xc, y_center = make_centered_matrix(y_out, others=others, p=p)

# Treinar modelo linear centrado
beta_c = linear_model_fit(Xc, y_center)

# Prever valores corrigidos (apenas para pontos onde há features centradas)
y_corr_c = y_out.copy()
h = p // 2
mask_out_c = outlier_mask_mu_k_sigma(y_out, k)[h:-h]  # só nos pontos com janela centrada
idx_c = np.where(mask_out_c)[0] + h  # índices reais no vetor original

for i, t in enumerate(idx_c):
    x_t = Xc[t-h]
    y_corr_c[t] = linear_model_predict(x_t.reshape(1, -1), beta_c)[0]

# Erro de predição centrado
err_c = y_corr_c - y_orig
rmse_c = np.sqrt(mean_squared_error(y_orig[idx_c], y_corr_c[idx_c]))
print(f'Janela centrada (p={p}): RMSE nos outliers corrigidos = {rmse_c:.4f}')

# Distribuição do erro centrado
plt.figure(figsize=(10, 5))
sns.histplot(err_c[idx_c], bins=50, kde=True)
plt.title(f'Distribuição do erro de predição centrado (p={p}) nos outliers corrigidos')
plt.xlabel('Erro (valor previsto - valor real)')
plt.ylabel('Frequência')
plt.show()

# Exemplo de valores previstos vs reais (janela centrada)
if len(idx_c) > 0:
    i0 = idx_c[0]
    plt.figure(figsize=(12, 6))
    plt.plot(range(i0-20, i0+20), y_orig[i0-20:i0+20], label='Valor real')
    plt.plot(range(i0-20, i0+20), y_corr_c[i0-20:i0+20], label='Valor corrigido centrado')
    plt.scatter(idx_c, y_corr_c[idx_c], color='red', label='Outliers corrigidos')
    plt.title(f'Exemplo de correção centrada de outliers (p={p})')
    plt.xlabel('Índice')
    plt.ylabel('Aceleração')
    plt.legend()
    plt.show()

# Janela centrada (p=10): RMSE nos outliers corrigidos = 6.9948

# %% Parte 4
# %% 4.1 - Significância estatística dos valores médios + teste de gaussianidade

from scipy.stats import kstest, f_oneway, kruskal

variables = ['accel_magnitude', 'gyro_magnitude', 'magneto_magnitude']

# 1. Médias por atividade
activity_means = df_all_participants.groupby('Activity Label')[variables].mean()
print("Médias por atividade:")
print(activity_means)

# %%
# 2. Teste de gaussianidade (Kolmogorov-Smirnov)
print("\nTeste de gaussianidade (Kolmogorov-Smirnov):")
for var in variables:
    print(f"\n{var}:")
    for activity in sorted(df_all_participants['Activity Label'].unique()):
        data = df_all_participants[df_all_participants['Activity Label'] == activity][var].dropna()
        normed = (data - data.mean()) / data.std(ddof=0)
        stat, p = kstest(normed, 'norm')
        print(f"  Atividade {activity}: KS={stat:.3f}, p={p:.3g}")

# %%
# 3. Teste de significância dos valores médios (ANOVA e Kruskal-Wallis)
print("\nTeste de significância dos valores médios entre atividades:")
for var in variables:
    groups = [df_all_participants[df_all_participants['Activity Label'] == act][var].dropna()
              for act in sorted(df_all_participants['Activity Label'].unique())]
    anova_stat, anova_p = f_oneway(*groups)
    kw_stat, kw_p = kruskal(*groups)
    print(f"{var}: ANOVA p={anova_p:.3g}, Kruskal-Wallis p={kw_p:.3g}")

# Comentário:
# ============================================================
# Análise estatística das variáveis
# ============================================================
# 
# Teste de normalidade (Kolmogorov–Smirnov):
# ------------------------------------------
# O teste KS foi aplicado às distribuições das variáveis
# accel_magnitude, gyro_magnitude e magneto_magnitude para cada
# uma das 16 atividades.
#
# Os p-valores obtidos foram extremamente baixos (p ≈ 0 em todos
# os casos, variando até ~1e-150 nas atividades 15 e 16), o que
# leva à rejeição da hipótese nula de normalidade.
# 
# Conclui-se, portanto, que as distribuições dessas variáveis
# não seguem uma distribuição normal, o que é esperado em dados
# de sensores de movimento (aceleração, giroscópio, magnetômetro),
# que normalmente apresentam assimetrias e caudas longas.
#
# --------------------------------------------------------------
# Teste de diferenças de médias entre atividades:
# --------------------------------------------------------------
# Devido à ausência de normalidade, o teste não paramétrico de
# Kruskal–Wallis é mais apropriado do que o ANOVA tradicional.
# 
# Ambos os testes (ANOVA e Kruskal–Wallis) retornaram p-valores
# praticamente nulos (p ≈ 0) para todas as variáveis, indicando
# que as médias (ou distribuições) diferem significativamente
# entre as atividades.
#
# --------------------------------------------------------------
# Conclusão:
# --------------------------------------------------------------
# - As variáveis analisadas NÃO são gaussianas (p < 0.05 em todos os casos).
# - Há diferenças estatisticamente significativas entre as atividades.
# - As magnitudes de aceleração, giroscópio e magnetômetro refletem
#   padrões distintos de movimento, podendo ser utilizadas como
#   descritores discriminativos entre atividades.
#
# ============================================================

# %% 4.2 - Rotinas de extração de features temporais e espectrais

from scipy.stats import skew, kurtosis
from scipy.signal import welch

def extract_temporal_features(window):
    feats = {}
    feats['mean'] = np.mean(window)
    feats['std'] = np.std(window)
    feats['median'] = np.median(window)
    feats['min'] = np.min(window)
    feats['max'] = np.max(window)
    feats['skewness'] = skew(window)
    feats['kurtosis'] = kurtosis(window)
    feats['rms'] = np.sqrt(np.mean(window**2))
    # Zero Crossing Rate
    feats['zero_crossings'] = ((np.diff(np.sign(window)) != 0).sum()) / len(window)
    return feats

def extract_spectral_features(window, fs=50):
    # Welch para espectro de potência
    f, Pxx = welch(window, fs=fs, nperseg=len(window))
    feats = {}
    feats['energy'] = np.sum(Pxx)
    # Entropia espectral
    Pxx_norm = Pxx / np.sum(Pxx)
    feats['spectral_entropy'] = -np.sum(Pxx_norm * np.log2(Pxx_norm + 1e-12))
    # Frequência dominante
    feats['dominant_freq'] = f[np.argmax(Pxx)]
    # Magnitude média do espectro
    feats['mean_spectral_mag'] = np.mean(Pxx)
    # Banda de energia (exemplo: 0.5–3 Hz)
    band_mask = (f >= 0.5) & (f <= 3.0)
    feats['energy_band_0.5_3Hz'] = np.sum(Pxx[band_mask])
    return feats

# Exemplo: extrair features de uma janela de 128 amostras
window_size = 128
start = 0
window = df_all_participants['accel_magnitude'].values[start:start+window_size]

temporal_feats = extract_temporal_features(window)
spectral_feats = extract_spectral_features(window, fs=50)

print("Features temporais:", temporal_feats)
print("Features espectrais:", spectral_feats)

# %%
def extract_features_for_all(df, variable, window_size=128, fs=50):
    data = df[variable].values
    feats_list = []
    for start in range(0, len(data) - window_size + 1, window_size):
        window = data[start:start+window_size]
        feats = {}
        feats.update(extract_temporal_features(window))
        feats.update(extract_spectral_features(window, fs=fs))
        feats_list.append(feats)
    return pd.DataFrame(feats_list)

# Exemplo para aceleração
features_accel = extract_features_for_all(df_all_participants, 'accel_magnitude', window_size=128, fs=50)
print(features_accel.head())

# %% 4.3 - Implementação do PCA

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Suponha que você já extraiu o feature set (features_accel, por exemplo)
# Combine features de todas as variáveis se quiser um feature set completo:
features_accel = extract_features_for_all(df_all_participants, 'accel_magnitude', window_size=128, fs=50)
features_gyro = extract_features_for_all(df_all_participants, 'gyro_magnitude', window_size=128, fs=50)
features_mag = extract_features_for_all(df_all_participants, 'magneto_magnitude', window_size=128, fs=50)

# Concatenar todos os features (opcional)
features_all = pd.concat([features_accel, features_gyro, features_mag], axis=1)

# Normalizar as features (z-score)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_all)

# Aplicar PCA
pca = PCA()
pca_features = pca.fit_transform(features_scaled)

# Visualizar variância explicada
explained_var = pca.explained_variance_ratio_
print("Variância explicada por cada componente:", explained_var)
print("Variância acumulada:", np.cumsum(explained_var))

# Exemplo: plotar variância acumulada
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(explained_var), marker='o')
plt.xlabel('Número de componentes')
plt.ylabel('Variância explicada acumulada')
plt.title('PCA - Variância explicada acumulada')
plt.grid(True)
plt.show()

# %% usando features originais diretamente
# Selecionar apenas as features originais (sem IDs, Timestamp, rótulo)
feature_cols = ['accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z',
                'mag_x', 'mag_y', 'mag_z']

features_all = df_all_participants[feature_cols]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_all)

pca = PCA()
pca_features = pca.fit_transform(features_scaled)

# Visualizar variância explicada
explained_var = pca.explained_variance_ratio_
print("Variância explicada por cada componente:", explained_var)
print("Variância acumulada:", np.cumsum(explained_var)) 

plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(explained_var), marker='o')
plt.xlabel('Número de componentes')
plt.ylabel('Variância explicada acumulada')
plt.title('PCA - Variância explicada acumulada (features originais)')
plt.grid(True)
plt.show()

# Agora aplique Fisher Score, ReliefF, PCA, etc. normalmente


# %% 4.4
# Variância acumulada
explained_var = pca.explained_variance_ratio_
cum_var = np.cumsum(explained_var)

# Quantas componentes para 75%?
n_components_75 = np.argmax(cum_var >= 0.75) + 1
print(f"Número de componentes para explicar 75% da variância: {n_components_75}")

# %%
# Features comprimidas (projeção PCA)
features_pca_75 = pca_features[:, :n_components_75]

# Exemplo para o primeiro instante
instante = 0
print(f"Features PCA (75%) para instante {instante}:")
print(features_pca_75[instante])

# %%
# Mostrar os coeficientes das 6 primeiras componentes
feature_names = features_all.columns
for i in range(6):
    print(f"\nComponente {i+1} (PC{i+1}):")
    # Mostra as 5 features originais mais importantes para cada componente
    top_idx = np.argsort(np.abs(pca.components_[i]))[::-1][:5]
    for idx in top_idx:
        print(f"  {feature_names[idx]}: {pca.components_[i][idx]:.3f}")

# %%
# Fisher Score para cada feature
def fisher_score(X, y):
    # X: matriz de features (n_samples, n_features)
    # y: vetor de classes (n_samples,)
    scores = []
    classes = np.unique(y)
    for i in range(X.shape[1]):
        feat = X[:, i]
        mean_total = np.mean(feat)
        num = 0
        denom = 0
        for c in classes:
            idx = (y == c)
            mean_c = np.mean(feat[idx])
            var_c = np.var(feat[idx])
            num += len(feat[idx]) * (mean_c - mean_total) ** 2
            denom += len(feat[idx]) * var_c
        score = num / (denom + 1e-12)
        scores.append(score)
    return np.array(scores)

# Exemplo de uso:
X = features_scaled  # matriz de features normalizadas
y = df_all_participants['Activity Label'].values[:X.shape[0]]
fisher_scores = fisher_score(X, y)
top_fisher_idx = np.argsort(fisher_scores)[::-1][:10]
print("Top 10 features pelo Fisher Score:")
for idx in top_fisher_idx:
    print(f"{features_all.columns[idx]}: {fisher_scores[idx]:.3f}")

# %%  ReliefF
from skrebate import ReliefF

# X: matriz de features normalizadas
X = features_scaled
y = df_all_participants['Activity Label'].values[:X.shape[0]]

relief = ReliefF(n_neighbors=10, n_features_to_select=10)
relief.fit(X, y)
relief_scores = relief.feature_importances_

top_relief_idx = np.argsort(relief_scores)[::-1][:10]
print("Top 10 features pelo ReliefF:")
for idx in top_relief_idx:
    print(f"{features_all.columns[idx]}: {relief_scores[idx]:.3f}")
# %%
