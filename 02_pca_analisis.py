#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
Script 02: Análisis de Componentes Principales (PCA)
==========================================================================
Descripción:
    Realiza un Análisis de Componentes Principales (ACP/PCA) desde cero
    sobre las variables de fases minerales DRX:
      - quartz_pct (cuarzo)
      - micas_pct (micas)
      - plagioclase_pct (plagioclasa)
      - mafics_pct (máficos)
      - other_minerals_pct (otros minerales)
    
    Incluye: escalamiento, extracción de autovalores, varianza explicada,
    biplot con coloreado por malla, mapa de calor de cargas, gráfico de
    sedimentación (scree plot) y comparación con PC1/PC2 pre-calculados.

Autor: Pipeline AgroMineral-PCA
==========================================================================
"""

# ============================================================
# IMPORTACIÓN DE BIBLIOTECAS
# ============================================================
import numpy as np                    # Álgebra lineal y operaciones numéricas
import pandas as pd                   # Manipulación de datos tabulares
import matplotlib                     # Biblioteca base de visualización
matplotlib.use('Agg')                 # Backend no interactivo
import matplotlib.pyplot as plt       # Interfaz de alto nivel para gráficos
import matplotlib.font_manager as fm  # Gestión de fuentes tipográficas
import seaborn as sns                 # Visualización estadística
from sklearn.preprocessing import StandardScaler  # Estandarización
from numpy.linalg import eig          # Descomposición de valores propios
import os                             # Operaciones del sistema
import random                         # Generación de números aleatorios

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
print("SCRIPT 02: ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)")
print("=" * 70)

data_path = os.path.join(DATA_DIR, 'agrominerals_drx_master.csv')
df = pd.read_csv(data_path)
print(f"\n[INFO] Datos cargados: {df.shape}")

# Variables de fases minerales DRX para el PCA
pca_features = [
    'quartz_pct', 'micas_pct', 'plagioclase_pct',
    'mafics_pct', 'other_minerals_pct'
]

# Nombres en español para visualización
spanish_names = {
    'quartz_pct': 'Cuarzo (%)',
    'micas_pct': 'Micas (%)',
    'plagioclase_pct': 'Plagioclasa (%)',
    'mafics_pct': 'Máficos (%)',
    'other_minerals_pct': 'Otros Minerales (%)'
}

# Extraer subconjunto de datos (sin valores faltantes)
X = df[pca_features].dropna()
print(f"[INFO] Muestras sin valores faltantes: {X.shape[0]}")

# ============================================================
# 2. ESTANDARIZACIÓN DE LOS DATOS
# ============================================================
print("\n[INFO] Estandarizando variables (media=0, desv.=1)...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convertir a DataFrame para facilitar el manejo
X_scaled_df = pd.DataFrame(X_scaled, columns=pca_features, index=X.index)

print(f"[INFO] Medias post-estandarización: {X_scaled_df.mean().round(6).tolist()}")
print(f"[INFO] Desviaciones post-estandarización: {X_scaled_df.std().round(6).tolist()}")

# ============================================================
# 3. CÁLCULO DE PCA DESDE CERO
# ============================================================
print("\n[INFO] Calculando matriz de covarianza y autovalores...")

# Matriz de covarianza (sobre datos estandarizados = matriz de correlación)
cov_matrix = np.cov(X_scaled, rowvar=False)
print(f"[INFO] Dimensión de matriz de covarianza: {cov_matrix.shape}")

# Descomposición en valores propios y vectores propios
eigenvalues, eigenvectors = eig(cov_matrix)

# Ordenar en orden descendente por valor propio
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices].real  # Tomar parte real
eigenvectors = eigenvectors[:, sorted_indices].real

# Convertir autovalores a float
eigenvalues = eigenvalues.astype(float)
eigenvectors = eigenvectors.astype(float)

print(f"\n[INFO] Autovalores: {eigenvalues}")
print(f"[INFO] Suma de autovalores (debe ser ~{len(pca_features)}): {eigenvalues.sum():.4f}")

# ============================================================
# 4. VARIANZA EXPLICADA
# ============================================================
total_variance = eigenvalues.sum()
var_explained = eigenvalues / total_variance * 100  # Porcentaje individual
cumulative_var = np.cumsum(var_explained)           # Porcentaje acumulado

print(f"\n--- Varianza Explicada por Componente ---")
for i, (ev, ve, cv) in enumerate(zip(eigenvalues, var_explained, cumulative_var)):
    print(f"  CP{i+1}: Autovalor={ev:.4f}, Var.Individual={ve:.2f}%, Var.Acumulada={cv:.2f}%")

# ============================================================
# 5. CÁLCULO DE PUNTUACIONES (SCORES) DE LAS COMPONENTES
# ============================================================
# Proyectar datos estandarizados sobre los vectores propios
scores = X_scaled @ eigenvectors

# Crear DataFrame de puntuaciones
scores_df = pd.DataFrame(
    scores,
    columns=[f'CP{i+1}' for i in range(len(pca_features))],
    index=X.index
)

# ============================================================
# 6. TABLA DE AUTOVALORES
# ============================================================
eigenvalue_table = pd.DataFrame({
    'Componente': [f'CP{i+1}' for i in range(len(pca_features))],
    'Autovalor': eigenvalues,
    'Varianza Individual (%)': np.round(var_explained, 4),
    'Varianza Acumulada (%)': np.round(cumulative_var, 4)
})
print(f"\n--- Tabla de Autovalores ---")
print(eigenvalue_table.to_string(index=False))

tab03_path = os.path.join(TAB_DIR, 'tab03_autovalores_pca.csv')
eigenvalue_table.to_csv(tab03_path, index=False, encoding='utf-8-sig')
print(f"\n[OK] Tabla guardada: {tab03_path}")

# ============================================================
# 7. TABLA DE CARGAS (LOADINGS)
# ============================================================
loadings = pd.DataFrame(
    eigenvectors,
    index=pca_features,
    columns=[f'CP{i+1}' for i in range(len(pca_features))]
)
# Renombrar índices a español
loadings.index = [spanish_names.get(c, c) for c in loadings.index]

print(f"\n--- Matriz de Cargas (Loadings) ---")
print(loadings.round(4).to_string())

tab04_path = os.path.join(TAB_DIR, 'tab04_loadings_pca.csv')
loadings.to_csv(tab04_path, encoding='utf-8-sig')
print(f"\n[OK] Tabla guardada: {tab04_path}")

# ============================================================
# 8. TABLA DE VARIANZA EXPLICADA
# ============================================================
var_table = pd.DataFrame({
    'Componente': [f'CP{i+1}' for i in range(len(pca_features))],
    'Autovalor': eigenvalues,
    'Varianza Individual (%)': np.round(var_explained, 4),
    'Varianza Acumulada (%)': np.round(cumulative_var, 4),
    'Proporción de Varianza': np.round(var_explained / 100, 4)
})

tab05_path = os.path.join(TAB_DIR, 'tab05_varianza_explicada.csv')
var_table.to_csv(tab05_path, index=False, encoding='utf-8-sig')
print(f"[OK] Tabla guardada: {tab05_path}")

# ============================================================
# 9. GRÁFICO DE SEDIMENTACIÓN (SCREE PLOT)
# ============================================================
print("\n[INFO] Generando gráfico de sedimentación (Scree Plot)...")

fig, ax1 = plt.subplots(figsize=(10, 7), constrained_layout=True)
fig.suptitle('Gráfico de Sedimentación (Scree Plot) - PCA DRX',
             fontsize=14, fontweight='bold')

components = range(1, len(pca_features) + 1)

# Barras para autovalores
bars = ax1.bar(components, eigenvalues, color='steelblue', alpha=0.8,
               edgecolor='navy', linewidth=0.8, label='Autovalor')

# Línea de varianza acumulada en eje secundario
ax2 = ax1.twinx()
ax2.plot(components, cumulative_var, 'ro-', linewidth=2, markersize=8,
         label='Varianza Acumulada (%)')
ax2.set_ylabel('Varianza Acumulada (%)', fontsize=11, color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, 110)

# Línea de referencia del criterio de Kaiser (autovalor = 1)
ax1.axhline(y=1, color='orange', linestyle='--', linewidth=1.5,
            label='Criterio de Kaiser (λ=1)')

# Etiquetas en las barras
for bar, ev, ve in zip(bars, eigenvalues, var_explained):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{ev:.2f}\n({ve:.1f}%)', ha='center', va='bottom', fontsize=9)

ax1.set_xlabel('Componente Principal', fontsize=11)
ax1.set_ylabel('Autovalor (λ)', fontsize=11)
ax1.set_xticks(list(components))
ax1.set_xticklabels([f'CP{i}' for i in components])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Leyenda combinada
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

fig04_path = os.path.join(FIG_DIR, 'fig04_scree_plot.png')
fig.savefig(fig04_path, dpi=150, bbox_inches=None)
plt.close(fig)
print(f"[OK] Figura guardada: {fig04_path}")

# ============================================================
# 10. BIPLOT (CP1 vs CP2) COLOREADO POR MALLA
# ============================================================
print("\n[INFO] Generando Biplot PCA (CP1 vs CP2)...")

fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
fig.suptitle('Biplot PCA: CP1 vs CP2 (Coloreado por Categoría de Malla)',
             fontsize=14, fontweight='bold')

# Obtener categorías de malla para colorear
mesh_labels = df.loc[X.index, 'mesh']
unique_mesh = sorted(mesh_labels.unique())
colors = sns.color_palette('Set1', n_colors=len(unique_mesh))

# Graficar cada categoría con color diferente
for mesh_cat, color in zip(unique_mesh, colors):
    mask = mesh_labels == mesh_cat
    ax.scatter(scores_df.loc[mask, 'CP1'],
               scores_df.loc[mask, 'CP2'],
               c=[color], label=f'Malla {mesh_cat}',
               alpha=0.5, s=20, edgecolors='none')

# Añadir vectores de carga (flechas)
scale_factor = 3.0  # Factor de escala para las flechas
for i, var in enumerate(pca_features):
    ax.annotate('',
                xy=(eigenvectors[i, 0] * scale_factor,
                    eigenvectors[i, 1] * scale_factor),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='darkred',
                                lw=2, alpha=0.8))
    ax.text(eigenvectors[i, 0] * scale_factor * 1.15,
            eigenvectors[i, 1] * scale_factor * 1.15,
            spanish_names.get(var, var),
            fontsize=9, color='darkred', fontweight='bold',
            ha='center', va='center')

ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)
ax.set_xlabel(f'CP1 ({var_explained[0]:.1f}% de varianza)', fontsize=11)
ax.set_ylabel(f'CP2 ({var_explained[1]:.1f}% de varianza)', fontsize=11)
ax.legend(title='Categoría Malla', fontsize=8, title_fontsize=9,
          markerscale=2, loc='best')
ax.grid(alpha=0.3, linestyle='--')

fig05_path = os.path.join(FIG_DIR, 'fig05_biplot_pca.png')
fig.savefig(fig05_path, dpi=150, bbox_inches=None)
plt.close(fig)
print(f"[OK] Figura guardada: {fig05_path}")

# ============================================================
# 11. MAPA DE CALOR DE LA MATRIZ DE CARGAS
# ============================================================
print("\n[INFO] Generando mapa de calor de cargas...")

fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
fig.suptitle('Matriz de Cargas (Loadings) del PCA',
             fontsize=14, fontweight='bold')

sns.heatmap(loadings, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.8, linecolor='white',
            cbar_kws={'label': 'Carga (Loading)', 'shrink': 0.8},
            ax=ax)
ax.tick_params(axis='y', rotation=0, labelsize=10)
ax.tick_params(axis='x', rotation=0, labelsize=10)

fig06_path = os.path.join(FIG_DIR, 'fig06_loading_matrix.png')
fig.savefig(fig06_path, dpi=150, bbox_inches=None)
plt.close(fig)
print(f"[OK] Figura guardada: {fig06_path}")

# ============================================================
# 12. COMPARACIÓN CON PC1_SCORE Y PC2_SCORE PRE-CALCULADOS
# ============================================================
print("\n" + "-" * 50)
print("COMPARACIÓN: PCA Nuevo vs PC1/PC2 Pre-calculados")
print("-" * 50)

# Comparar con los scores pre-calculados del dataset original
pc1_pre = df.loc[X.index, 'PC1_score']
pc2_pre = df.loc[X.index, 'PC2_score']

# Nota: Los signos pueden invertirse en PCA (eigenvectores no tienen dirección fija)
# Verificamos correlación absoluta
corr_pc1 = np.corrcoef(scores_df['CP1'].values, pc1_pre.values)[0, 1]
corr_pc2 = np.corrcoef(scores_df['CP2'].values, pc2_pre.values)[0, 1]

print(f"\nCorrelación entre nuevo CP1 y PC1_score: {corr_pc1:.4f}")
print(f"Correlación entre nuevo CP2 y PC2_score: {corr_pc2:.4f}")

# Verificar con signos invertidos
corr_pc1_inv = np.corrcoef(-scores_df['CP1'].values, pc1_pre.values)[0, 1]
corr_pc2_inv = np.corrcoef(-scores_df['CP2'].values, pc2_pre.values)[0, 1]
print(f"\nCorrelación (signo invertido) CP1 vs PC1_score: {corr_pc1_inv:.4f}")
print(f"Correlación (signo invertido) CP2 vs PC2_score: {corr_pc2_inv:.4f}")

# Determinar si los signos coinciden
pc1_match = "COINCIDE" if abs(corr_pc1) > 0.99 else "DIFIERE EN SIGNO" if abs(corr_pc1_inv) > 0.99 else "DIFERENTE"
pc2_match = "COINCIDE" if abs(corr_pc2) > 0.99 else "DIFIERE EN SIGNO" if abs(corr_pc2_inv) > 0.99 else "DIFERENTE"

print(f"\nCP1: Signo {pc1_match} con PC1_score pre-calculado")
print(f"CP2: Signo {pc2_match} con PC2_score pre-calculado")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 70)
print("SCRIPT 02 COMPLETADO EXITOSAMENTE")
print("=" * 70)
print(f"\nArchivos generados:")
print(f"  - {tab03_path}")
print(f"  - {tab04_path}")
print(f"  - {tab05_path}")
print(f"  - {fig04_path}")
print(f"  - {fig05_path}")
print(f"  - {fig06_path}")
print()