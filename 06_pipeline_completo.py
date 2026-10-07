#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
Script 06: Pipeline Completo de Análisis AgroMineral-PCA
==========================================================================
Descripción:
    Ejecuta secuencialmente todos los scripts de análisis (01-05),
    mide el tiempo de ejecución de cada paso y genera un resumen
    final consolidado en formato de texto.

Autor: Pipeline AgroMineral-PCA
==========================================================================
"""

# ============================================================
# IMPORTACIÓN DE BIBLIOTECAS
# ============================================================
import subprocess       # Ejecutar scripts como procesos separados
import sys              # Información del sistema y parámetros
import os               # Operaciones del sistema de archivos
import time             # Medición de tiempo de ejecución
import datetime         # Fechas y horas
import glob             # Búsqueda de patrones de archivos
import random           # Números aleatorios
import numpy as np      # Para semilla

# ============================================================
# SEMILLAS PARA REPRODUCIBILIDAD
# ============================================================
np.random.seed(42)
random.seed(42)

# ============================================================
# RUTAS DE DIRECTORIOS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_DIR = os.path.join(BASE_DIR, 'scripts', 'python')
FIG_DIR = os.path.join(BASE_DIR, 'results', 'figures')
TAB_DIR = os.path.join(BASE_DIR, 'results', 'tables')

# Asegurar que los directorios de salida existen
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ============================================================
# DEFINICIÓN DE SCRIPTS A EJECUTAR
# ============================================================
scripts = [
    {
        'numero': '01',
        'nombre': 'Análisis Exploratorio de Datos',
        'archivo': '01_analisis_exploratorio.py',
        'descripcion': 'Estadísticas descriptivas, histogramas, correlación, boxplots'
    },
    {
        'numero': '02',
        'nombre': 'Análisis de Componentes Principales (PCA)',
        'archivo': '02_pca_analisis.py',
        'descripcion': 'PCA desde cero, biplot, scree plot, cargas, comparación'
    },
    {
        'numero': '03',
        'nombre': 'Regresión Lineal Múltiple',
        'archivo': '03_regresion_multiple.py',
        'descripcion': 'OLS, diagnóstico de residuos, Q-Q plot, análisis VIF'
    },
    {
        'numero': '04',
        'nombre': 'Análisis de Conglomerados K-means',
        'archivo': '04_cluster_kmeans.py',
        'descripcion': 'Elbow, silueta, clustering k=3, perfiles de cluster'
    },
    {
        'numero': '05',
        'nombre': 'Análisis Factorial',
        'archivo': '05_analisis_factorial.py',
        'descripcion': 'KMO, Bartlett, factores Varimax, comunalidades, ICM'
    }
]

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def ejecutar_script(script_path, script_nombre):
    """
    Ejecuta un script Python como subproceso y devuelve
    el tiempo de ejecución, el código de salida y la salida estándar.
    
    Parámetros:
        script_path (str): Ruta absoluta al script .py
        script_nombre (str): Nombre descriptivo del script
    
    Retorna:
        tuple: (tiempo_segundos, codigo_salida, salida_estandar, salida_error)
    """
    print(f"\n{'=' * 70}")
    print(f"EJECUTANDO: {script_nombre}")
    print(f"Archivo: {script_path}")
    print(f"{'=' * 70}")
    
    inicio = time.time()
    
    try:
        # Ejecutar el script usando el mismo intérprete de Python
        resultado = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300  # Timeout de 5 minutos por script
        )
        fin = time.time()
        duracion = fin - inicio
        
        # Imprimir la salida del script
        if resultado.stdout:
            print(resultado.stdout)
        
        # Si hay errores, imprimirlos
        if resultado.stderr and resultado.returncode != 0:
            print(f"[ADVERTENCIA] Errores en stderr:\n{resultado.stderr}")
        
        return duracion, resultado.returncode, resultado.stdout, resultado.stderr
        
    except subprocess.TimeoutExpired:
        fin = time.time()
        duracion = fin - inicio
        print(f"[ERROR] El script excedió el tiempo límite (300s)")
        return duracion, -1, '', 'Timeout excedido'
    except Exception as e:
        fin = time.time()
        duracion = fin - inicio
        print(f"[ERROR] Excepción al ejecutar: {e}")
        return duracion, -1, '', str(e)


def listar_archivos_generados():
    """
    Lista todos los archivos generados en los directorios de resultados.
    
    Retorna:
        list: Lista de diccionarios con información de cada archivo
    """
    archivos = []
    
    # Buscar figuras
    for f in sorted(glob.glob(os.path.join(FIG_DIR, '*.png'))):
        size_kb = os.path.getsize(f) / 1024
        archivos.append({
            'tipo': 'Figura',
            'ruta': f,
            'nombre': os.path.basename(f),
            'tamano_kb': round(size_kb, 1)
        })
    
    # Buscar tablas
    for f in sorted(glob.glob(os.path.join(TAB_DIR, '*.csv'))):
        size_kb = os.path.getsize(f) / 1024
        archivos.append({
            'tipo': 'Tabla',
            'ruta': f,
            'nombre': os.path.basename(f),
            'tamano_kb': round(size_kb, 1)
        })
    
    # Buscar archivos de texto
    for f in sorted(glob.glob(os.path.join(TAB_DIR, '*.txt'))):
        size_kb = os.path.getsize(f) / 1024
        archivos.append({
            'tipo': 'Texto',
            'ruta': f,
            'nombre': os.path.basename(f),
            'tamano_kb': round(size_kb, 1)
        })
    
    return archivos


# ============================================================
# EJECUCIÓN SECUENCIAL DE TODOS LOS SCRIPTS
# ============================================================
print("=" * 70)
print("PIPELINE COMPLETO AGROMINERAL-PCA")
print(f"Fecha de inicio: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
print(f"\nScripts a ejecutar: {len(scripts)}")
print(f"Directorio base: {BASE_DIR}")
print(f"Directorio de scripts: {SCRIPT_DIR}")

# Lista para almacenar resultados de cada script
resultados = []
tiempo_total_inicio = time.time()

for script_info in scripts:
    script_path = os.path.join(SCRIPT_DIR, script_info['archivo'])
    
    # Verificar que el script existe
    if not os.path.exists(script_path):
        print(f"\n[ERROR] Script no encontrado: {script_path}")
        resultados.append({
            'numero': script_info['numero'],
            'nombre': script_info['nombre'],
            'duracion': 0,
            'estado': 'NO ENCONTRADO',
            'descripcion': script_info['descripcion']
        })
        continue
    
    # Ejecutar el script
    duracion, codigo_salida, stdout, stderr = ejecutar_script(
        script_path, script_info['nombre']
    )
    
    # Determinar estado
    if codigo_salida == 0:
        estado = 'EXITOSO'
    else:
        estado = f'ERROR (código {codigo_salida})'
    
    resultados.append({
        'numero': script_info['numero'],
        'nombre': script_info['nombre'],
        'duracion': duracion,
        'estado': estado,
        'descripcion': script_info['descripcion']
    })

tiempo_total = time.time() - tiempo_total_inicio

# ============================================================
# GENERACIÓN DEL RESUMEN DE EJECUCIÓN
# ============================================================
print("\n" + "=" * 70)
print("GENERANDO RESUMEN DE EJECUCIÓN")
print("=" * 70)

fecha_fin = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Construir el texto del resumen
lineas = []
lineas.append("=" * 70)
lineas.append("RESUMEN DE EJECUCIÓN - PIPELINE AGROMINERAL-PCA")
lineas.append("=" * 70)
lineas.append(f"")
lineas.append(f"Fecha de inicio: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lineas.append(f"Fecha de finalización: {fecha_fin}")
lineas.append(f"Tiempo total de ejecución: {tiempo_total:.2f} segundos")
lineas.append(f"")
lineas.append("-" * 70)
lineas.append("ESTADO POR SCRIPT:")
lineas.append("-" * 70)

for res in resultados:
    estado_str = res['estado']
    lineas.append(f"")
    lineas.append(f"  Script {res['numero']}: {res['nombre']}")
    lineas.append(f"    Descripción: {res['descripcion']}")
    lineas.append(f"    Estado: {estado_str}")
    lineas.append(f"    Tiempo: {res['duracion']:.2f} segundos")

# Resumen de archivos generados
archivos_gen = listar_archivos_generados()
n_figuras = sum(1 for a in archivos_gen if a['tipo'] == 'Figura')
n_tablas = sum(1 for a in archivos_gen if a['tipo'] == 'Tabla')
n_textos = sum(1 for a in archivos_gen if a['tipo'] == 'Texto')

lineas.append(f"")
lineas.append("-" * 70)
lineas.append("ARCHIVOS GENERADOS:")
lineas.append("-" * 70)
lineas.append(f"")
lineas.append(f"  Total de archivos: {len(archivos_gen)}")
lineas.append(f"    - Figuras (PNG): {n_figuras}")
lineas.append(f"    - Tablas (CSV):  {n_tablas}")
lineas.append(f"    - Textos (TXT):  {n_textos}")
lineas.append(f"")

# Lista detallada de archivos
lineas.append(f"  DETALLE:")
for arch in archivos_gen:
    lineas.append(f"    [{arch['tipo']}] {arch['nombre']} ({arch['tamano_kb']} KB)")

lineas.append(f"")
lineas.append("=" * 70)
lineas.append("FIN DEL RESUMEN")
lineas.append("=" * 70)

# Unir todas las líneas en un solo texto
resumen_texto = '\n'.join(lineas)

# Imprimir el resumen en consola
print(resumen_texto)

# ============================================================
# GUARDAR RESUMEN EN ARCHIVO DE TEXTO
# ============================================================
tab13_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'tables', 'tab13_resumen_ejecucion.txt')
tab13_path = os.path.normpath(tab13_path)

with open(tab13_path, 'w', encoding='utf-8') as f:
    f.write(resumen_texto)

print(f"\n[OK] Resumen guardado en: {tab13_path}")
print(f"\n{'=' * 70}")
print("PIPELINE COMPLETO FINALIZADO")
print(f"{'=' * 70}\n")