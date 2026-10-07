#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
Script 03: Regresión Lineal Múltiple
==========================================================================
Descripción:
    Ajusta un modelo de regresión lineal múltiple para predecir la
    Solubilidad_Relativa de rocas remineralizadoras en función de:
      - SiO2_pct (dióxido de silicio)
      - K2O_pct (óxido de potasio)
      - CaO_pct (óxido de calcio)
      - MgO_pct (óxido de magnesio)
      - Granulometria_Mesh (tamaño de partícula)
    
    Incluye: ajuste OLS con statsmodels, gráficos de diagnóstico
    (residuos vs ajustados, Q-Q, escala-localización) y análisis
    de VIF para multicolinealidad.

Autor: Pipeline AgroMineral-PCA
==========================================================================
"""

# ============================================================
# IMPORTACIÓN DE BIBLIOTECAS
# ============================================================
import numpy as np                    # Operaciones numéricas
import pandas as pd                   # Manipulación de datos tabulares
import matplotlib                     # Biblioteca base de visualización
matplotlib.use('Agg')                 # Backend no interactivo
import matplotlib.pyplot as plt       # Interfaz de alto nivel
import matplotlib.font_manager as fm  # Gestión de fuentes
import scipy.stats as stats           # Estadísticas científicas
import statsmodels.api as sm          # Modelos estadísticos (OLS)
from statsmodels.stats.outliers_influence import variance_inflation_factor  # VIF
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
print("SCRIPT 03: REGRESIÓN LINEAL MÚLTIPLE")
print("=" * 70)

data_path = os.path.join(DATA_DIR, 'remineralizacion_roca.csv')
df = pd.read_csv(data_path)
print(f"\n[INFO] Datos cargados: {df.shape}")
print(f"[INFO] Columnas disponibles: {df.columns.tolist()}")

# ============================================================
# 2. PREPARACIÓN DE VARIABLES
# ============================================================
# Variable objetivo (respuesta)
target = 'Solubilidad_Relativa'

# Variables predictoras (características)
features = ['SiO2_pct', 'K2O_pct', 'CaO_pct', 'MgO_pct', 'Granulometria_Mesh']

# Nombres en español para visualización
spanish_features = {
    'SiO2_pct': 'SiO₂ (%)',
    'K2O_pct': 'K₂O (%)',
    'CaO_pct': 'CaO (%)',
    'MgO_pct': 'MgO (%)',
    'Granulometria_Mesh': 'Malla (Mesh)'
}

# Eliminar filas con valores faltantes en las variables de interés
df_clean = df[[target] + features].dropna()
print(f"\n[INFO] Muestras completas: {df_clean.shape[0]} (de {len(df)})")

# Separar variables dependiente e independientes
y = df_clean[target]
X = df_clean[features]

# Añadir término de intersección (β₀) al modelo
X_const = sm.add_constant(X)
print(f"[INFO] Matriz de diseño: {X_const.shape}")

# ============================================================
# 3. AJUSTE DEL MODELO OLS
# ============================================================
print("\n[INFO] Ajustando modelo de regresión lineal múltiple (OLS)...")
print(f"  Modelo: {target} = β₀ + β₁·SiO₂ + β₂·K₂O + β₃·CaO + β₄·MgO + β₅·Mesh")

model = sm.OLS(y, X_const)
results = model.fit()

# Imprimir resumen completo del modelo
print("\n" + "=" * 70)
print(results.summary())
print("=" * 70)

# ============================================================
# 4. GUARDAR TABLA DE RESULTADOS DE REGRESIÓN
# ============================================================
# Extraer coeficientes y estadísticos del modelo
coef_table = pd.DataFrame({
    'Variable': results.params.index,
    'Coeficiente': results.params.values,
    'Error Estándar': results.bse.values,
    'Estadístico t': results.tvalues.values,
    'Valor p': results.pvalues.values,
    'IC Inferior 95%': results.conf_int()[0].values,
    'IC Superior 95%': results.conf_int()[1].values
})

# Añadir significancia
def significancia(p):
    """Clasificar significancia estadística según valor p."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

coef_table['Significancia'] = coef_table['Valor p'].apply(significancia)

# Renombrar variables a español
rename_map = {'const': 'Intersección (β₀)'}
rename_map.update(spanish_features)
coef_table['Variable'] = coef_table['Variable'].map(lambda x: rename_map.get(x, x))

# Añadir estadísticas globales del modelo al final
global_stats = pd.DataFrame({
    'Variable': [
        '--- Estadísticas Globales ---',
        'R² (Ajustado)',
        'Estadístico F',
        'Valor p (modelo)',
        'Log-Verosimilitud',
        'AIC',
        'BIC',
        'N (muestras)'
    ],
    'Coeficiente': [
        '',
        f'{results.rsquared_adj:.4f}',
        f'{results.fvalue:.4f}',
        f'{results.f_pvalue:.6f}',
        f'{results.llf:.4f}',
        f'{results.aic:.4f}',
        f'{results.bic:.4f}',
        f'{int(results.nobs)}'
    ],
    'Error Estándar': [''] * 8,
    'Estadístico t': [''] * 8,
    'Valor p': [''] * 8,
    'IC Inferior 95%': [''] * 8,
    'IC Superior 95%': [''] * 8,
    'Significancia': [''] * 8
})

# Combinar coeficientes y estadísticas globales
full_table = pd.concat([coef_table, global_stats], ignore_index=True)

tab06_path = os.path.join(TAB_DIR, 'tab06_regresion_resultados.csv')
full_table.to_csv(tab06_path, index=False, encoding='utf-8-sig')
print(f"\n[OK] Tabla guardada: {tab06_path}")

# ============================================================
# 5. ANÁLISIS VIF (FACTOR DE INFLACIÓN DE VARIANZA)
# ============================================================
print("\n[INFO] Calculando Factores de Inflación de Varianza (VIF)...")

vif_data = pd.DataFrame({
    'Variable': features,
    'VIF': [variance_inflation_factor(X_const.values, i+1) for i in range(len(features))]
})

# Añadir interpretación del VIF
def interpretar_vif(vif):
    """Interpretar el factor VIF según umbrales estándar."""
    if vif < 2:
        return 'Sin multicolinealidad'
    elif vif < 5:
        return 'Moderada'
    elif vif < 10:
        return 'Alta'
    else:
        return 'Muy Alta (problemática)'

vif_data['Interpretación'] = vif_data['VIF'].apply(interpretar_vif)

# Renombrar variables a español
vif_data['Variable'] = vif_data['Variable'].map(lambda x: spanish_features.get(x, x))

print(vif_data.to_string(index=False))

tab07_path = os.path.join(TAB_DIR, 'tab07_vif.csv')
vif_data.to_csv(tab07_path, index=False, encoding='utf-8-sig')
print(f"\n[OK] Tabla guardada: {tab07_path}")

# ============================================================
# 6. GRÁFICOS DE DIAGNÓSTICO
# ============================================================

# --- 6a. Distribución de Residuos + Residuos vs Valores Ajustados ---
print("\n[INFO] Generando gráficos de diagnóstico...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
fig.suptitle('Diagnóstico de Residuos del Modelo de Regresión',
             fontsize=14, fontweight='bold')

residuals = results.resid
fitted = results.fittedvalues

# Panel izquierdo: Histograma de residuos
ax = axes[0]
ax.hist(residuals, bins=40, color='steelblue', edgecolor='white', alpha=0.8, density=True)
# Superponer curva normal teórica
xmin, xmax = ax.get_xlim()
x_norm = np.linspace(xmin, xmax, 200)
y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
ax.plot(x_norm, y_norm, 'r-', linewidth=2, label='Distribución Normal Teórica')
ax.set_title('Distribución de Residuos', fontsize=12)
ax.set_xlabel('Residuo', fontsize=10)
ax.set_ylabel('Densidad', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linestyle='--')

# Panel derecho: Residuos vs Valores Ajustados
ax = axes[1]
ax.scatter(fitted, residuals, alpha=0.3, s=10, color='steelblue', edgecolors='none')
ax.axhline(y=0, color='red', linewidth=1.5, linestyle='--')
# Línea de tendencia LOESS (usando polinomio simple como aproximación)
z = np.polyfit(fitted, residuals, 2)
p = np.poly1d(z)
x_smooth = np.linspace(fitted.min(), fitted.max(), 100)
ax.plot(x_smooth, p(x_smooth), 'orange', linewidth=2, label='Tendencia (polinomio)')
ax.set_title('Residuos vs Valores Ajustados', fontsize=12)
ax.set_xlabel('Valores Ajustados (Predichos)', fontsize=10)
ax.set_ylabel('Residuos', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linestyle='--')

fig07_path = os.path.join(FIG_DIR, 'fig07_distribucion_residuales.png')
fig.savefig(fig07_path, dpi=150, bbox_inches=None)
plt.close(fig)
print(f"[OK] Figura guardada: {fig07_path}")

# --- 6b. Gráfico Q-Q Normal ---
fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
fig.suptitle('Gráfico Q-Q Normal de Residuos',
             fontsize=14, fontweight='bold')

# Q-Q plot usando scipy
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm')

ax.scatter(osm, osr, alpha=0.4, s=15, color='steelblue', edgecolors='none')
# Línea de referencia teórica
ax.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2,
        label=f'Ajuste lineal (r={r:.4f})')
ax.set_title('Gráfico Cuantil-Cuantil (Q-Q) de Residuos', fontsize=12)
ax.set_xlabel('Cuantiles Teóricos (Normal)', fontsize=11)
ax.set_ylabel('Cuantiles Observados', fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--')
# Línea de identidad
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)

fig08_path = os.path.join(FIG_DIR, 'fig08_qq_plot.png')
fig.savefig(fig08_path, dpi=150, bbox_inches=None)
plt.close(fig)
print(f"[OK] Figura guardada: {fig08_path}")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 70)
print("SCRIPT 03 COMPLETADO EXITOSAMENTE")
print("=" * 70)
print(f"\nArchivos generados:")
print(f"  - {tab06_path}")
print(f"  - {tab07_path}")
print(f"  - {fig07_path}")
print(f"  - {fig08_path}")
print()