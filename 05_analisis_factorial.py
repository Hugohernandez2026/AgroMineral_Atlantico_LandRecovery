#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
Script 05: Análisis Factorial
==========================================================================
Descripción:
    Realiza un Análisis Factorial (AF) sobre las variables de fases
    minerales DRX y variables químicas del dataset maestro.
    
    Incluye:
      - Prueba de Kaiser-Meyer-Olkin (KMO) para adecuación muestral
      - Prueba de esfericidad de Bartlett
      - Extracción de factores con rotación Varimax
      - Matriz de cargas factoriales, comunalidades y varianza explicada
      - Cálculo de puntuaciones factoriales ("Índice de Calidad Mineral")
      - Comparación con el índice ICM_index existente

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
from factor_analyzer import FactorAnalyzer  # Análisis factorial
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import os                             # Operaciones del sistema
import random                         # Números aleatorios
import warnings                       # Control de advertencias
warnings.filterwarnings('ignore')     # Suprimir advertencias no críticas

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
print("SCRIPT 05: ANÁLISIS FACTORIAL")
print("=" * 70)

data_path = os.path.join(DATA_DIR, 'agrominerals_drx_master.csv')
df = pd.read_csv(data_path)
print(f"\n[INFO] Datos cargados: {df.shape}")

# ============================================================
# 2. SELECCIÓN DE VARIABLES
# ============================================================
# Variables de fases minerales DRX + variables químicas
fa_features = [
    'quartz_pct', 'micas_pct', 'plagioclase_pct', 'mafics_pct',
    'other_minerals_pct', 'SiO2_pct', 'K2O_pct', 'CaO_pct',
    'MgO_pct', 'P2O5_pct'
]

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
    'P2O5_pct': 'P₂O₅ (%)'
}

# Preparar datos sin valores faltantes
df_clean = df.dropna(subset=fa_features).copy()
X = df_clean[fa_features].values
print(f"[INFO] Muestras completas: {X.shape[0]}")
print(f"[INFO] Variables: {len(fa_features)}")

# ============================================================
# 3. PRUEBA DE KMO (KAISER-MEYER-OLKIN)
# ============================================================
print("\n[INFO] Calculando prueba de Kaiser-Meyer-Olkin (KMO)...")

kmo_per_variable, kmo_total = calculate_kmo(X)

print(f"\n--- Resultados KMO ---")
print(f"  KMO Global: {kmo_total:.4f}", end="")
if kmo_total >= 0.9:
    print(" → Excelente (muy adecuado para AF)")
elif kmo_total >= 0.8:
    print(" → Bueno (adecuado para AF)")
elif kmo_total >= 0.7:
    print(" → Aceptable")
elif kmo_total >= 0.6:
    print(" → Regular (dudoso)")
else:
    print(" → Inadecuado (no se recomienda AF)")

print(f"\n  KMO por Variable:")
for feat, kmo_val in zip(fa_features, kmo_per_variable):
    print(f"    {spanish_names.get(feat, feat)}: {kmo_val:.4f}")

# ============================================================
# 4. PRUEBA DE ESFERICIDAD DE BARTLETT
# ============================================================
print("\n[INFO] Calculando prueba de esfericidad de Bartlett...")

chi_square, p_value_bartlett = calculate_bartlett_sphericity(X)

print(f"\n--- Resultados Bartlett ---")
print(f"  Estadístico Chi-cuadrado: {chi_square:.2f}")
print(f"  Grados de libertad: {len(fa_features) * (len(fa_features) - 1) // 2}")
print(f"  Valor p: {p_value_bartlett:.2e}")
if p_value_bartlett < 0.05:
    print("  → Se rechaza H₀: La matriz de correlación NO es identidad (adecuado para AF)")
else:
    print("  → No se rechaza H₀: Las variables pueden ser independientes (no adecuado)")

# Guardar resultados KMO y Bartlett
kmo_bartlett_df = pd.DataFrame({
    'Prueba': [
        'KMO Global',
        'Bartlett - Chi-cuadrado',
        'Bartlett - Grados de Libertad',
        'Bartlett - Valor p',
        'Conclusión KMO',
        'Conclusión Bartlett'
    ],
    'Valor': [
        f'{kmo_total:.4f}',
        f'{chi_square:.2f}',
        str(len(fa_features) * (len(fa_features) - 1) // 2),
        f'{p_value_bartlett:.2e}',
        'Adecuado' if kmo_total >= 0.6 else 'Inadecuado',
        'Adecuado' if p_value_bartlett < 0.05 else 'No adecuado'
    ]
})

tab10_path = os.path.join(TAB_DIR, 'tab10_kmo_bartlett.csv')
kmo_bartlett_df.to_csv(tab10_path, index=False, encoding='utf-8-sig')
print(f"\n[OK] Tabla guardada: {tab10_path}")

# ============================================================
# 5. DETERMINACIÓN DEL NÚMERO ÓPTIMO DE FACTORES
# ============================================================
print("\n[INFO] Determinando número óptimo de factores...")

# Probar de 1 a (n_variables-1) factores y evaluar varianza explicada
fa_diag = FactorAnalyzer(rotation=None, n_factors=len(fa_features))
fa_diag.fit(X)

# Obtener autovalores
ev, v = fa_diag.get_eigenvalues()

# Criterio de Kaiser: factores con autovalor > 1
n_factors_kaiser = np.sum(ev > 1)
print(f"  Autovalores: {ev.round(4)}")
print(f"  Factores con autovalor > 1 (Kaiser): {n_factors_kaiser}")

# Seleccionar número de factores (usar Kaiser, mínimo 2)
n_factors = max(2, n_factors_kaiser)
print(f"  Número de factores a extraer: {n_factors}")

# ============================================================
# 6. EXTRACCIÓN DE FACTORES CON ROTACIÓN VARIMAX
# ============================================================
print(f"\n[INFO] Extrayendo {n_factors} factores con rotación Varimax...")

fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax', method='principal')
fa.fit(X)

# ============================================================
# 7. CARGAS FACTORIALES
# ============================================================
loadings = pd.DataFrame(
    fa.loadings_,
    index=fa_features,
    columns=[f'Factor {i+1}' for i in range(n_factors)]
)

# Renombrar índices a español
loadings.index = [spanish_names.get(c, c) for c in loadings.index]

# Añadir comunalidades
loadings['Comunalidad (h²)'] = fa.get_communalities().round(4)

print(f"\n--- Cargas Factoriales (Rotación Varimax) ---")
print(loadings.round(4).to_string())

tab11_path = os.path.join(TAB_DIR, 'tab11_factor_loadings.csv')
loadings.to_csv(tab11_path, encoding='utf-8-sig')
print(f"\n[OK] Tabla guardada: {tab11_path}")

# ============================================================
# 8. COMUNALIDADES
# ============================================================
communalities = pd.DataFrame({
    'Variable': fa_features,
    'Nombre Español': [spanish_names.get(f, f) for f in fa_features],
    'Comunalidad (h²)': fa.get_communalities().round(4),
    'Varianza Única (1-h²)': (1 - fa.get_communalities()).round(4)
})

print(f"\n--- Comunalidades ---")
print(communalities.to_string(index=False))

tab12_path = os.path.join(TAB_DIR, 'tab12_comunalidades.csv')
communalities.to_csv(tab12_path, index=False, encoding='utf-8-sig')
print(f"\n[OK] Tabla guardada: {tab12_path}")

# ============================================================
# 9. VARIANZA EXPLICADA POR FACTORES
# ============================================================
variance = fa.get_factor_variance()
var_df = pd.DataFrame({
    'Factor': [f'Factor {i+1}' for i in range(n_factors)],
    'Varianza del Factor (SS Loadings)': np.round(variance[0], 4),
    'Varianza Proporcional (%)': np.round(variance[1] * 100, 4),
    'Varianza Acumulada (%)': np.round(variance[2] * 100, 4)
})
print(f"\n--- Varianza Explicada por Factores ---")
print(var_df.to_string(index=False))

# ============================================================
# 10. GRÁFICO DE CARGAS FACTORIALES
# ============================================================
print("\n[INFO] Generando gráfico de cargas factoriales...")

# Solo las columnas de factores (sin comunalidad)
loadings_only = loadings.iloc[:, :n_factors].copy()

fig, ax = plt.subplots(figsize=(8, max(6, len(fa_features) * 0.7)),
                       constrained_layout=True)
fig.suptitle('Cargas Factoriales (Rotación Varimax)',
             fontsize=14, fontweight='bold')

sns.heatmap(loadings_only, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=False,
            linewidths=0.8, linecolor='white',
            cbar_kws={'label': 'Carga Factorial', 'shrink': 0.8},
            ax=ax)
ax.tick_params(axis='y', rotation=0, labelsize=10)
ax.tick_params(axis='x', rotation=0, labelsize=11)
ax.set_ylabel('Variable', fontsize=11)
ax.set_xlabel('Factor', fontsize=11)

fig11_path = os.path.join(FIG_DIR, 'fig11_factor_loadings.png')
fig.savefig(fig11_path, dpi=150, bbox_inches=None)
plt.close(fig)
print(f"[OK] Figura guardada: {fig11_path}")

# ============================================================
# 11. GRÁFICO DE COMUNALIDADES
# ============================================================
print("\n[INFO] Generando gráfico de comunalidades...")

fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
fig.suptitle('Comunalidades por Variable',
             fontsize=14, fontweight='bold')

vars_es = [spanish_names.get(f, f) for f in fa_features]
comm_vals = fa.get_communalities()

# Barras horizontales coloreadas según nivel de comunalidad
colors = ['#e74c3c' if c < 0.4 else '#f39c12' if c < 0.7 else '#2ecc71'
          for c in comm_vals]

bars = ax.barh(vars_es, comm_vals, color=colors, edgecolor='white',
               alpha=0.85, height=0.6)

# Añadir valores en las barras
for bar, val in zip(bars, comm_vals):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)

ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5,
           label='Umbral alto (0.7)')
ax.axvline(x=0.4, color='orange', linestyle='--', alpha=0.5,
           label='Umbral bajo (0.4)')
ax.set_xlabel('Comunalidad (h²)', fontsize=11)
ax.set_xlim(0, 1.05)
ax.legend(fontsize=9, loc='lower right')
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Invertir eje y para que primera variable esté arriba
ax.invert_yaxis()

fig12_path = os.path.join(FIG_DIR, 'fig12_comunalidades.png')
fig.savefig(fig12_path, dpi=150, bbox_inches=None)
plt.close(fig)
print(f"[OK] Figura guardada: {fig12_path}")

# ============================================================
# 12. PUNTUACIONES FACTORIALES Y COMPARACIÓN CON ICM_INDEX
# ============================================================
print("\n" + "-" * 50)
print("COMPARACIÓN: Puntuación Factorial vs ICM_index")
print("-" * 50)

# Calcular puntuaciones factoriales (regresión)
factor_scores = fa.transform(X)

# Usar el primer factor como "Índice de Calidad Mineral" calculado
icm_calculado = factor_scores[:, 0]

# Comparar con el ICM_index existente
icm_existente = df_clean['ICM_index'].values

# Correlación
from scipy.stats import pearsonr, spearmanr
corr_pearson, p_pearson = pearsonr(icm_calculado, icm_existente)
corr_spearman, p_spearman = spearmanr(icm_calculado, icm_existente)

print(f"\nCorrelación Pearson (Factor 1 vs ICM_index): r={corr_pearson:.4f}, p={p_pearson:.2e}")
print(f"Correlación Spearman (Factor 1 vs ICM_index): ρ={corr_spearman:.4f}, p={p_spearman:.2e}")

# Si la correlación es negativa, probar con signo invertido
if corr_pearson < 0:
    corr_pearson_inv, _ = pearsonr(-icm_calculado, icm_existente)
    print(f"Correlación Pearson (-Factor 1 vs ICM_index): r={corr_pearson_inv:.4f}")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 70)
print("SCRIPT 05 COMPLETADO EXITOSAMENTE")
print("=" * 70)
print(f"\nArchivos generados:")
print(f"  - {tab10_path}")
print(f"  - {tab11_path}")
print(f"  - {tab12_path}")
print(f"  - {fig11_path}")
print(f"  - {fig12_path}")
print()