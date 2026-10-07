#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
Script 01: Análisis Exploratorio de Datos (AED)
==========================================================================
Descripción:
    Este script realiza un análisis exploratorio completo del conjunto de 
    datos maestro de agrominerales DRX (2000 muestras × 18 variables).
    Incluye estadísticas descriptivas, verificación de valores faltantes,
    histogramas de variables numéricas, mapa de calor de correlación y
    diagramas de caja por categoría de malla.

Autor: Pipeline AgroMineral-PCA
==========================================================================
"""

# ============================================================
# IMPORTACIÓN DE BIBLIOTECAS
# ============================================================
import numpy as np                    # Álgebra lineal y operaciones numéricas
import pandas as pd                   # Manipulación de datos tabulares
import matplotlib                     # Biblioteca base de visualización
matplotlib.use('Agg')                 # Backend no interactivo para guardar figuras
import matplotlib.pyplot as plt       # Interfaz de alto nivel para gráficos
import matplotlib.font_manager as fm  # Gestión de fuentes tipográficas
import seaborn as sns                 # Visualización estadística avanzada
import os                             # Operaciones del sistema de archivos
import random                         # Generación de números aleatorios

# ============================================================
# SEMILLAS PARA REPRODUCIBILIDAD
# ============================================================
np.random.seed(42)   # Semilla para NumPy
random.seed(42)      # Semilla para el módulo random

# ============================================================
# CONFIGURACIÓN DE FUENTES Y ESTILO VISUAL
# ============================================================
# Cargar fuente compatible con caracteres chinos/españoles
FONT_PATH = '/home/z/my-project/download/AgroMineral-PCA/data/NotoSansSC-Regular.ttf'
fm.fontManager.addfont(FONT_PATH)
plt.rcParams['font.sans-serif'] = ['Sarasa Mono SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Mostrar signos negativos correctamente

# ============================================================
# RUTAS DE DIRECTORIOS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FIG_DIR = os.path.join(BASE_DIR, 'results', 'figures')
TAB_DIR = os.path.join(BASE_DIR, 'results', 'tables')

# Crear directorios de salida si no existen
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ============================================================
# 1. CARGA DEL CONJUNTO DE DATOS
# ============================================================
print("=" * 70)
print("SCRIPT 01: ANÁLISIS EXPLORATORIO DE DATOS")
print("=" * 70)

# Ruta del archivo de datos
data_path = os.path.join(DATA_DIR, 'agrominerals_drx_master.csv')
df = pd.read_csv(data_path)

print(f"\n[INFO] Datos cargados: {df.shape[0]} muestras × {df.shape[1]} variables")
print(f"[INFO] Columnas: {df.columns.tolist()}")

# ============================================================
# 2. ESTADÍSTICAS DESCRIPTIVAS BÁSICAS
# ============================================================
# Seleccionar solo las variables numéricas para el análisis
numeric_cols = [
    'quartz_pct', 'micas_pct', 'plagioclase_pct', 'mafics_pct',
    'other_minerals_pct', 'SiO2_pct', 'K2O_pct', 'CaO_pct',
    'MgO_pct', 'P2O5_pct', 'K_solubility_index', 'PC1_score',
    'PC2_score', 'ICM_index'
]

# Nombres en español para las etiquetas de los gráficos
spanish_names = {
    'quartz_pct': 'Cuarzo (%)',
    'micas_pct': 'Micas (%)',
    'plagioclase_pct': 'Plagioclasa (%)',
    'mafics_pct': 'Máficos (%)',
    'other_minerals_pct': 'Otros Minerales (%)',
    'SiO2_pct': 'SiO₂ (%)',
    'K2O_pct': 'K₂O (%)',
    'CaO_pct': 'CaO (%)',
    'MgO_pct': 'MgO (%)',
    'P2O5_pct': 'P₂O₅ (%)',
    'K_solubility_index': 'Índice Solubilidad K',
    'PC1_score': 'Puntuación CP1',
    'PC2_score': 'Puntuación CP2',
    'ICM_index': 'Índice ICM'
}

# Calcular estadísticas descriptivas
desc_stats = df[numeric_cols].describe().T
desc_stats['missing'] = df[numeric_cols].isnull().sum()
desc_stats['missing_pct'] = (desc_stats['missing'] / len(df) * 100).round(2)

# Renombrar columnas a español
desc_stats.columns = [
    'Cantidad', 'Media', 'Desv.Estándar', 'Mínimo', 'Q1 (25%)',
    'Mediana (50%)', 'Q3 (75%)', 'Máximo', 'Valores Faltantes',
    '% Faltantes'
]

print("\n--- Estadísticas Descriptivas ---")
print(desc_stats.to_string())

# Guardar tabla de estadísticas descriptivas
tab01_path = os.path.join(TAB_DIR, 'tab01_estadisticas_descriptivas.csv')
desc_stats.to_csv(tab01_path, encoding='utf-8-sig')
print(f"\n[OK] Tabla guardada: {tab01_path}")

# Verificación de valores faltantes
print("\n--- Verificación de Valores Faltantes ---")
missing_total = df[numeric_cols].isnull().sum()
print(missing_total)
if missing_total.sum() == 0:
    print("[INFO] No se encontraron valores faltantes en las variables numéricas.")

# ============================================================
# 3. HISTOGRAMAS DE TODAS LAS VARIABLES NUMÉRICAS (2x3 por figura)
# ============================================================
print("\n[INFO] Generando histogramas...")

# Agrupar las 14 variables en 3 figuras de 2x3 (12 en total) + 1 extra de 2x1
# Figura 1: Primeras 6 variables (fases minerales DRX)
fig1_vars = numeric_cols[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
fig.suptitle('Histogramas de Variables de Fases Minerales DRX', fontsize=16, fontweight='bold')

for idx, var in enumerate(fig1_vars):
    ax = axes[idx // 3, idx % 3]
    ax.hist(df[var].dropna(), bins=40, color='steelblue', edgecolor='white',
            alpha=0.85, density=False)
    ax.set_title(spanish_names.get(var, var), fontsize=11)
    ax.set_xlabel('Valor', fontsize=9)
    ax.set_ylabel('Frecuencia', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

# Ocultar el último subplot si sobra
# (6 variables en 2x3 = exacto, no sobra)

fig01_path = os.path.join(FIG_DIR, 'fig01_histogramas.png')
fig.savefig(fig01_path, dpi=150, bbox_inches=None)
plt.close(fig)
print(f"[OK] Figura guardada: {fig01_path}")

# ============================================================
# 4. MAPA DE CALOR DE CORRELACIÓN (todas las variables numéricas)
# ============================================================
print("\n[INFO] Generando mapa de calor de correlación...")

# Calcular matriz de correlación de Pearson
corr_matrix = df[numeric_cols].corr()

# Renombrar índices y columnas a español para visualización
corr_display = corr_matrix.rename(index=spanish_names, columns=spanish_names)

# Crear figura con mapa de calor
fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
fig.suptitle('Matriz de Correlación de Pearson - Variables Numéricas',
             fontsize=16, fontweight='bold', y=1.02)

# Generar mapa de calor con anotaciones numéricas
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # No se usa, se muestra completa
sns.heatmap(corr_matrix, 
            annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5,
            xticklabels=[spanish_names.get(c, c) for c in corr_matrix.columns],
            yticklabels=[spanish_names.get(c, c) for c in corr_matrix.index],
            ax=ax, cbar_kws={'shrink': 0.8, 'label': 'Coeficiente de Correlación'})
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0, labelsize=8)

fig02_path = os.path.join(FIG_DIR, 'fig02_correlacion.png')
fig.savefig(fig02_path, dpi=150, bbox_inches=None)
plt.close(fig)
print(f"[OK] Figura guardada: {fig02_path}")

# Guardar matriz de correlación como tabla CSV
corr_save = corr_matrix.copy()
corr_save.index = [spanish_names.get(c, c) for c in corr_save.index]
corr_save.columns = [spanish_names.get(c, c) for c in corr_save.columns]
tab02_path = os.path.join(TAB_DIR, 'tab02_matriz_correlacion.csv')
corr_save.to_csv(tab02_path, encoding='utf-8-sig')
print(f"[OK] Tabla guardada: {tab02_path}")

# ============================================================
# 5. DIAGRAMAS DE CAJA POR CATEGORÍA DE MALLA
# ============================================================
print("\n[INFO] Generando diagramas de caja por categoría de malla...")

# Variables clave para boxplots
boxplot_vars = ['quartz_pct', 'micas_pct', 'SiO2_pct', 'K2O_pct', 'ICM_index']

# Verificar que la columna 'mesh' existe
if 'mesh' in df.columns:
    mesh_categories = sorted(df['mesh'].unique())
    n_cats = len(mesh_categories)
    
    fig, axes = plt.subplots(1, len(boxplot_vars), figsize=(20, 7), constrained_layout=True)
    fig.suptitle('Diagramas de Caja por Categoría de Malla (Mesh)',
                 fontsize=16, fontweight='bold')
    
    # Paleta de colores para las categorías de malla
    palette = sns.color_palette('Set2', n_colors=n_cats)
    
    for idx, var in enumerate(boxplot_vars):
        ax = axes[idx]
        # Crear lista de datos por categoría
        data_groups = [df[df['mesh'] == cat][var].dropna().values for cat in mesh_categories]
        
        bp = ax.boxplot(data_groups, tick_labels=[str(c) for c in mesh_categories],
                        patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=1.5),
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))
        
        # Colorear las cajas
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(spanish_names.get(var, var), fontsize=11, fontweight='bold')
        ax.set_xlabel('Malla (Mesh)', fontsize=9)
        ax.set_ylabel('Valor', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig03_path = os.path.join(FIG_DIR, 'fig03_boxplots_mesh.png')
    fig.savefig(fig03_path, dpi=150, bbox_inches=None)
    plt.close(fig)
    print(f"[OK] Figura guardada: {fig03_path}")
else:
    print("[ADVERTENCIA] La columna 'mesh' no existe en el dataset.")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 70)
print("SCRIPT 01 COMPLETADO EXITOSAMENTE")
print("=" * 70)
print(f"\nArchivos generados:")
print(f"  - {tab01_path}")
print(f"  - {tab02_path}")
print(f"  - {fig01_path}")
print(f"  - {fig02_path}")
if 'mesh' in df.columns:
    print(f"  - {fig03_path}")
print()