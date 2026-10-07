#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
Script 04: Análisis de Conglomerados K-means
==========================================================================
Descripción:
    Realiza un agrupamiento (clustering) no supervisado de las muestras
    de agrominerales DRX usando el algoritmo K-means.
    
    Variables de clustering (estandarizadas):
      - quartz_pct, micas_pct, plagioclase_pct, mafics_pct, ICM_index
    
    Incluye:
      - Método del codo (Elbow) para determinar k óptimo
      - Análisis de silueta (Silhouette) para evaluar la calidad
      - Ajuste K-means con k=3 (Premium/Estándar/Descartar)
      - Visualización PCA 2D coloreada por conglomerado
      - Tabla de centroides y perfil de cada conglomerado

Autor: Pipeline AgroMineral-PCA
==========================================================================
"""

# ============================================================
# IMPORTACIÓN DE BIBLIOTECAS
# ============================================================
import numpy as np                    # Operaciones numéricas
import pandas as pd                   # Manipulación de datos
import matplotlib                     # Biblioteca base de visualización
matplotlib.use('Agg')                 # Backend no interactivo
import matplotlib.pyplot as plt       # Interfaz de alto nivel
import matplotlib.font_manager as fm  # Gestión de fuentes
import seaborn as sns                 # Visualización estadística
from sklearn.preprocessing import StandardScaler  # Estandarización
from sklearn.cluster import KMeans    # Algoritmo K-means
from sklearn.metrics import silhouette_score, silhouette_samples  # Métricas
from sklearn.decomposition import PCA  # Reducción de dimensionalidad
import os                             # Operaciones del sistema
import random                         # Números aleatorios

# ============================================================
# SEMILLAS PARA REPRODUCIBILIDAD
# ============================================================
np.random.seed(42)
random.seed(42)

# ============================================================
# CONFIGURACIÓN DE FUENTES Y ESTILO VISUAL
# ============================================================
FONT_PATH = '/home/z/my-project/download/AgroMineral-PCA/data/NotoSansSC-Regular.ttf'
fm.fontManager.addfont(FONT_PATH)
plt.rcParams['font.sans-serif'] = ['Sarasa Mono SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# RUTAS DE DIRECTORIOS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FIG_DIR = os.path.join(BASE_DIR, 'results', 'figures')
TAB_DIR = os.path.join(BASE_DIR, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ============================================================
# 1. CARGA DE DATOS
# ============================================================
print("=" * 70)
print("SCRIPT 04: ANÁLISIS DE CONGLOMERADOS K-MEANS")
print("=" * 70)

data_path = os.path.join(DATA_DIR, 'agrominerals_drx_master.csv')
df = pd.read_csv(data_path)
print(f"\n[INFO] Datos cargados: {df.shape}")

# ============================================================
# 2. PREPARACIÓN DE VARIABLES
# ============================================================
cluster_features = [
    'quartz_pct', 'micas_pct', 'plagioclase_pct', 'mafics_pct', 'ICM_index'
]

spanish_names = {
    'quartz_pct': 'Cuarzo (%)',
    'micas_pct': 'Micas (%)',
    'plagioclase_pct': 'Plagioclasa (%)',
    'mafics_pct': 'Máficos (%)',
    'ICM_index': 'Índice ICM'
}

# Eliminar filas con valores faltantes
df_clean = df.dropna(subset=cluster_features).copy()
print(f"[INFO] Muestras completas para clustering: {df_clean.shape[0]}")

X = df_clean[cluster_features].values

# ============================================================
# 3. ESTANDARIZACIÓN
# ============================================================
print("\n[INFO] Estandarizando variables para clustering...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar nombres de clusters en español
cluster_labels_es = {0: 'Descartar', 1: 'Estándar', 2: 'Premium'}

# ============================================================
# 4. MÉTODO DEL CODO Y ANÁLISIS DE SILUETA
# ============================================================
print("\n[INFO] Evaluando número óptimo de clusters (k=2 a k=10)...")

k_range = range(2, 11)
inertias = []      # Inercia (SSE) para el método del codo
silhouettes = []   # Puntuación de silueta promedio

for k in k_range:
    # Ajustar K-means con semilla fija para reproducibilidad
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_score = silhouette_score(X_scaled, km.labels_)
    silhouettes.append(sil_score)
    print(f"  k={k}: Inercia={km.inertia_:.2f}, Silueta={sil_score:.4f}")

# ============================================================
# 5. GRÁFICO DEL MÉTODO DEL CODO Y SILUETA
# ============================================================
print("\n[INFO] Generando gráficos de Elbow y Silueta...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
fig.suptitle('Determinación del Número Óptimo de Conglomerados',
             fontsize=14, fontweight='bold')

# Panel izquierdo: Método del codo (Elbow)
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8, label='Inercia (SSE)')
ax1.set_xlabel('Número de Clusters (k)', fontsize=11)
ax1.set_ylabel('Inercia (Suma de Cuadrados Intra-cluster)', fontsize=11)
ax1.set_title('Método del Codo (Elbow)', fontsize=12)
ax1.set_xticks(list(k_range))
ax1.grid(alpha=0.3, linestyle='--')

# Marcar k=3 como elegido
k3_idx = list(k_range).index(3)
ax1.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='k=3 (elegido)')
ax1.plot(3, inertias[k3_idx], 'r*', markersize=20, zorder=5)
ax1.legend(fontsize=9)

# Panel derecho: Análisis de silueta
ax2.plot(k_range, silhouettes, 'rs-', linewidth=2, markersize=8, label='Silueta Promedio')
ax2.set_xlabel('Número de Clusters (k)', fontsize=11)
ax2.set_ylabel('Puntuación de Silueta', fontsize=11)
ax2.set_title('Análisis de Silueta', fontsize=12)
ax2.set_xticks(list(k_range))
ax2.grid(alpha=0.3, linestyle='--')

# Marcar k=3
ax2.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='k=3 (elegido)')
ax2.plot(3, silhouettes[k3_idx], 'r*', markersize=20, zorder=5)
ax2.legend(fontsize=9)

fig09_path = os.path.join(FIG_DIR, 'fig09_elbow_silhouette.png')
fig.savefig(fig09_path, dpi=150, bbox_inches=None)
plt.close(fig)
print(f"[OK] Figura guardada: {fig09_path}")

# ============================================================
# 6. AJUSTE FINAL K-MEANS CON k=3
# ============================================================
print("\n[INFO] Ajustando K-means final con k=3 (Premium/Estándar/Descartar)...")

k_final = 3
kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10, max_iter=300)
kmeans_final.fit(X_scaled)

# Asignar etiquetas de cluster al DataFrame
df_clean['cluster'] = kmeans_final.labels_
df_clean['cluster_nombre'] = df_clean['cluster'].map(cluster_labels_es)

# Conteo de muestras por cluster
print(f"\n--- Distribución de Muestras por Cluster ---")
cluster_counts = df_clean['cluster_nombre'].value_counts().sort_index()
for name, count in cluster_counts.items():
    pct = count / len(df_clean) * 100
    print(f"  {name}: {count} muestras ({pct:.1f}%)")

# ============================================================
# 7. TABLA DE CENTROIDES
# ============================================================
# Centroides en espacio estandarizado
centroids_scaled = kmeans_final.cluster_centers_

# Convertir centroides a espacio original para interpretación
centroids_original = scaler.inverse_transform(centroids_scaled)

centroides_df = pd.DataFrame(
    centroids_original,
    columns=cluster_features
)
centroides_df.insert(0, 'Cluster', [cluster_labels_es[i] for i in range(k_final)])
centroides_df['N_Muestras'] = [cluster_counts.get(cluster_labels_es[i], 0) for i in range(k_final)]

# Renombrar columnas a español
centroides_df.columns = [
    'Cluster', 'Cuarzo (%)', 'Micas (%)', 'Plagioclasa (%)',
    'Máficos (%)', 'Índice ICM', 'N Muestras'
]

print(f"\n--- Centroides de los Clusters ---")
print(centroides_df.to_string(index=False))

tab08_path = os.path.join(TAB_DIR, 'tab08_centroides_cluster.csv')
centroides_df.to_csv(tab08_path, index=False, encoding='utf-8-sig')
print(f"\n[OK] Tabla guardada: {tab08_path}")

# ============================================================
# 8. PERFIL DE CLUSTERS (ESTADÍSTICAS POR GRUPO)
# ============================================================
print("\n[INFO] Calculando perfil de características por cluster...")

perfil_data = []
for cluster_id in range(k_final):
    cluster_mask = df_clean['cluster'] == cluster_id
    cluster_name = cluster_labels_es[cluster_id]
    
    row = {
        'Cluster': cluster_name,
        'N_Muestras': cluster_mask.sum(),
    }
    
    # Estadísticas para cada variable
    for feat in cluster_features:
        vals = df_clean.loc[cluster_mask, feat]
        row[f'{spanish_names[feat]} (Media)'] = vals.mean().round(2)
        row[f'{spanish_names[feat]} (Desv.)'] = vals.std().round(2)
    
    perfil_data.append(row)

perfil_df = pd.DataFrame(perfil_data)
print(perfil_df.to_string(index=False))

tab09_path = os.path.join(TAB_DIR, 'tab09_perfil_cluster.csv')
perfil_df.to_csv(tab09_path, index=False, encoding='utf-8-sig')
print(f"\n[OK] Tabla guardada: {tab09_path}")

# ============================================================
# 9. VISUALIZACIÓN PCA 2D COLOREADA POR CLUSTER
# ============================================================
print("\n[INFO] Generando visualización PCA 2D coloreada por cluster...")

# Reducir a 2 dimensiones con PCA para visualización
pca_2d = PCA(n_components=2, random_state=42)
X_pca = pca_2d.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
fig.suptitle('Agrupamiento K-means (k=3) - Visualización PCA 2D',
             fontsize=14, fontweight='bold')

# Colores para los clusters
cluster_colors = {'Descartar': '#e74c3c', 'Estándar': '#f39c12', 'Premium': '#2ecc71'}

# Graficar cada cluster
for cluster_id in range(k_final):
    mask = df_clean['cluster'] == cluster_id
    name = cluster_labels_es[cluster_id]
    color = cluster_colors[name]
    
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=color, label=f'{name} (n={mask.sum()})',
               alpha=0.5, s=20, edgecolors='none')

# Graficar centroides en espacio PCA
centroids_pca = pca_2d.transform(centroids_scaled)
for cluster_id in range(k_final):
    name = cluster_labels_es[cluster_id]
    color = cluster_colors[name]
    ax.scatter(centroids_pca[cluster_id, 0], centroids_pca[cluster_id, 1],
               c=color, marker='X', s=300, edgecolors='black',
               linewidths=2, zorder=10, label=f'Centroide {name}')

ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)
ax.set_xlabel(f'CP1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% de varianza)',
              fontsize=11)
ax.set_ylabel(f'CP2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% de varianza)',
              fontsize=11)
ax.legend(title='Conglomerado', fontsize=9, title_fontsize=10,
          markerscale=1.5, loc='best')
ax.grid(alpha=0.3, linestyle='--')

fig10_path = os.path.join(FIG_DIR, 'fig10_cluster_scatter.png')
fig.savefig(fig10_path, dpi=150, bbox_inches=None)
plt.close(fig)
print(f"[OK] Figura guardada: {fig10_path}")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 70)
print("SCRIPT 04 COMPLETADO EXITOSAMENTE")
print("=" * 70)
print(f"\nArchivos generados:")
print(f"  - {fig09_path}")
print(f"  - {fig10_path}")
print(f"  - {tab08_path}")
print(f"  - {tab09_path}")
print()