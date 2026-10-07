#!/usr/bin/env python3
"""
Tests de reproducibilidad para AgroMineral-PCA.
Verifica la integridad de datos procesados, figuras, tablas y resultados estadísticos.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ruta base del repositorio
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Contadores ──────────────────────────────────────────────────────────
passed = 0
failed = 0


def check(condition, label):
    """Imprime PASSED/FAILED y actualiza contadores."""
    global passed, failed
    if condition:
        print(f"  [PASSED] {label}")
        passed += 1
    else:
        print(f"  [FAILED] {label}")
        failed += 1


# ========================================================================
# 1. CSV procesados: existencia y forma
# ========================================================================
print("=" * 65)
print("1. Archivos CSV procesados — existencia y forma")
print("=" * 65)

csv_checks = {
    "data/processed/agrominerals_drx_master.csv": (2000, 18),
    "data/processed/remineralizacion_roca.csv": (1000, 9),
}

for rel_path, (expected_rows, expected_cols) in csv_checks.items():
    fpath = os.path.join(BASE, rel_path)
    exists = os.path.isfile(fpath)
    check(exists, f"Existe: {rel_path}")
    if exists:
        df = pd.read_csv(fpath)
        check(df.shape == (expected_rows, expected_cols),
              f"  Forma correcta: {df.shape} (esperado ({expected_rows}, {expected_cols}))")

# ========================================================================
# 2. Figuras PNG: existencia y tamaño > 10 KB
# ========================================================================
print()
print("=" * 65)
print("2. Figuras PNG — existencia y tamaño mínimo (10 KB)")
print("=" * 65)

python_figures = [
    "fig01_histogramas.png",
    "fig02_correlacion.png",
    "fig03_boxplots_mesh.png",
    "fig04_scree_plot.png",
    "fig05_biplot_pca.png",
    "fig06_loading_matrix.png",
    "fig07_distribucion_residuales.png",
    "fig08_qq_plot.png",
    "fig09_elbow_silhouette.png",
    "fig10_cluster_scatter.png",
    "fig11_factor_loadings.png",
    "fig12_comunalidades.png",
]

r_figures = [
    "r_fig01_histogramas.png",
    "r_fig02_correlacion.png",
    "r_fig03_scree.png",
    "r_fig04_biplot.png",
    "r_fig05_contribuciones.png",
    "r_fig06_residuales.png",
    "r_fig07_qq.png",
    "r_fig08_cluster.png",
    "r_fig09_silhouette.png",
    "r_fig10_factor_loadings.png",
]

all_figures = python_figures + r_figures
min_size_bytes = 10 * 1024  # 10 KB

for fig in all_figures:
    fpath = os.path.join(BASE, "results", "figures", fig)
    exists = os.path.isfile(fpath)
    check(exists, f"Existe: {fig}")
    if exists:
        size = os.path.getsize(fpath)
        check(size > min_size_bytes,
              f"  Tamaño: {size:,} bytes (> 10 KB)")

# ========================================================================
# 3. Tablas Python: existencia
# ========================================================================
print()
print("=" * 65)
print("3. Tablas de resultados (Python) — existencia")
print("=" * 65)

python_tables = [
    "tab01_estadisticas_descriptivas.csv",
    "tab02_matriz_correlacion.csv",
    "tab03_autovalores_pca.csv",
    "tab04_loadings_pca.csv",
    "tab05_varianza_explicada.csv",
    "tab06_regresion_resultados.csv",
    "tab07_vif.csv",
    "tab08_centroides_cluster.csv",
    "tab09_perfil_cluster.csv",
    "tab10_kmo_bartlett.csv",
    "tab11_factor_loadings.csv",
    "tab12_comunalidades.csv",
    "tab13_resumen_ejecucion.txt",
]

for tbl in python_tables:
    fpath = os.path.join(BASE, "results", "tables", tbl)
    check(os.path.isfile(fpath), f"Existe: {tbl}")

# ========================================================================
# 4. Tablas R: existencia
# ========================================================================
print()
print("=" * 65)
print("4. Tablas de resultados (R) — existencia")
print("=" * 65)

r_tables = [
    "r_tab01_pca_summary.txt",
    "r_tab02_regresion.txt",
    "r_tab03_vif.txt",
    "r_tab04_centroides.txt",
    "r_tab05_factor_loadings.txt",
    "r_tab06_kmo.txt",
    "r_tab07_execution_log.txt",
]

for tbl in r_tables:
    fpath = os.path.join(BASE, "results", "tables", tbl)
    check(os.path.isfile(fpath), f"Existe: {tbl}")

# ========================================================================
# 5. Verificar ausencia de NaN en datasets procesados
# ========================================================================
print()
print("=" * 65)
print("5. Datos procesados — verificación de NaN")
print("=" * 65)

for rel_path in csv_checks:
    fpath = os.path.join(BASE, rel_path)
    df = pd.read_csv(fpath)
    nan_count = df.isna().sum().sum()
    check(nan_count == 0,
          f"Sin NaN: {rel_path} (NaN encontrados: {nan_count})")

# ========================================================================
# 6. Autovalores PCA: la suma debe ser igual al número de variables
# ========================================================================
print()
print("=" * 65)
print("6. PCA — autovalores")
print("=" * 65)

eigenvalues_path = os.path.join(BASE, "results", "tables", "tab03_autovalores_pca.csv")
if os.path.isfile(eigenvalues_path):
    df_eig = pd.read_csv(eigenvalues_path)
    # Detectar columna de autovalores
    eig_col = None
    for col in df_eig.columns:
        if 'eigenvalue' in col.lower() or 'autovalor' in col.lower() or 'valor' in col.lower():
            eig_col = col
            break
    if eig_col is None:
        # Usar la última columna numérica
        for col in df_eig.columns:
            if pd.api.types.is_numeric_dtype(df_eig[col]):
                eig_col = col
                break
    if eig_col is not None:
        eigenvalues = df_eig[eig_col].values
        n_vars = len(eigenvalues)
        eig_sum = eigenvalues.sum()
        check(abs(eig_sum - n_vars) < 0.01,
              f"Suma de autovalores = {eig_sum:.4f} (esperado ≈ {n_vars})")
    else:
        check(False, "No se encontró columna de autovalores en tab03")
else:
    check(False, "No existe tab03_autovalores_pca.csv")

# ========================================================================
# 7. Etiquetas de cluster: deben ser exactamente 3 (Premium/Estándar/Descartar o 0/1/2)
# ========================================================================
print()
print("=" * 65)
print("7. K-Means — etiquetas de cluster")
print("=" * 65)

# Verificar en centroides
centroides_path = os.path.join(BASE, "results", "tables", "tab08_centroides_cluster.csv")
if os.path.isfile(centroides_path):
    df_cent = pd.read_csv(centroides_path)
    # La columna de cluster puede llamarse 'cluster' o ser el índice
    cluster_col = None
    for col in df_cent.columns:
        if 'cluster' in col.lower():
            cluster_col = col
            break
    if cluster_col is not None:
        labels = set(df_cent[cluster_col].unique())
        numeric_ok = labels.issubset({0, 1, 2})
        string_ok = labels == {"Premium", "Estándar", "Descartar"}
        check(numeric_ok or string_ok,
              f"Etiquetas de cluster válidas: {labels}")
    else:
        # Verificar número de filas = 3
        n_clusters = len(df_cent)
        check(n_clusters == 3, f"Número de centroides = {n_clusters} (esperado 3)")
else:
    check(False, "No existe tab08_centroides_cluster.csv")

# ========================================================================
# Resumen final
# ========================================================================
print()
print("=" * 65)
total = passed + failed
print(f"RESUMEN: {passed}/{total} pruebas aprobadas, {failed} fallidas")
print("=" * 65)

sys.exit(0 if failed == 0 else 1)