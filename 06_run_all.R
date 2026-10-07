#!/usr/bin/env Rscript
# =============================================================================
# Script 06: Ejecutor de Todos los Scripts
# Proyecto: AgroMineral-PCA
# Descripcion: Ejecuta los 5 scripts de analisis en secuencia y registra
#              la salida completa en un archivo de log.
# =============================================================================

# --- Rutas y directorios ---
setwd("/home/z/my-project/download/AgroMineral-PCA")
dir.create("results/figures", showWarnings = FALSE, recursive = TRUE)
dir.create("results/tables", showWarnings = FALSE, recursive = TRUE)

# --- Configuracion del log ---
archivo_log <- "results/tables/r_tab07_execution_log.txt"

# Funcion auxiliar para escribir en log y consola
escribir_log <- function(texto, archivo) {
  cat(texto)
  cat(texto, file = archivo, append = TRUE)
}

# --- Inicializar archivo de log ---
cat("") # Limpiar buffer
writeLines("", archivo_log) # Crear/truncar archivo

escribir_log(paste0("============================================================\n"), archivo_log)
escribir_log(paste0("REGISTRO DE EJECUCION - AGROMINERAL PCA (R)\n"), archivo_log)
escribir_log(paste0("Fecha de inicio: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n"), archivo_log)
escribir_log(paste0("Directorio de trabajo: ", getwd(), "\n"), archivo_log)
escribir_log(paste0("Version de R: ", R.version.string, "\n"), archivo_log)
escribir_log(paste0("============================================================\n\n"), archivo_log)

# --- Lista de scripts a ejecutar ---
scripts <- list(
  list(nombre = "01_analisis_exploratorio.R", descripcion = "Analisis Exploratorio de Datos (AED)"),
  list(nombre = "02_pca.R", descripcion = "Analisis de Componentes Principales (PCA)"),
  list(nombre = "03_regresion.R", descripcion = "Regresion Lineal Multiple"),
  list(nombre = "04_cluster.R", descripcion = "Analisis de Conglomerados (K-means)"),
  list(nombre = "05_analisis_factorial.R", descripcion = "Analisis Factorial")
)

# --- Ejecutar cada script ---
resultados <- data.frame(
  Script = character(),
  Descripcion = character(),
  Estado = character(),
  Duracion_seg = numeric(),
  stringsAsFactors = FALSE
)

for (i in seq_along(scripts)) {
  scr <- scripts[[i]]
  ruta_script <- file.path("scripts/r", scr$nombre)
  
  escribir_log(paste0("------------------------------------------------------------\n"), archivo_log)
  escribir_log(paste0("SCRIPT ", i, ": ", scr$descripcion, "\n"), archivo_log)
  escribir_log(paste0("Archivo: ", ruta_script, "\n"), archivo_log)
  escribir_log(paste0("Inicio: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n"), archivo_log)
  
  # Verificar que el script existe
  if (!file.exists(ruta_script)) {
    escribir_log(paste0("ERROR: El script no existe: ", ruta_script, "\n"), archivo_log)
    escribir_log(paste0("Estado: FALLIDO (archivo no encontrado)\n\n"), archivo_log)
    resultados <- rbind(resultados, data.frame(
      Script = scr$nombre, Descripcion = scr$descripcion,
      Estado = "FALLIDO (no encontrado)", Duracion_seg = NA
    ))
    next
  }
  
  # Ejecutar el script y capturar salida
  inicio <- Sys.time()
  salida <- tryCatch({
    capture.output(
      source(ruta_script, local = TRUE, echo = FALSE, verbose = FALSE),
      type = "output"
    )
  }, error = function(e) {
    paste0("ERROR: ", conditionMessage(e))
  }, warning = function(w) {
    paste0("ADVERTENCIA: ", conditionMessage(w))
  })
  fin <- Sys.time()
  duracion <- as.numeric(difftime(fin, inicio, units = "secs"))
  
  # Determinar estado
  if (any(grepl("ERROR", salida))) {
    estado <- "FALLIDO"
  } else if (any(grepl("COMPLETADO EXITOSAMENTE", salida))) {
    estado <- "EXITOSO"
  } else {
    estado <- "COMPLETADO (con advertencias)"
  }
  
  # Escribir salida capturada al log
  escribir_log(paste0("\n--- Salida del script ---\n"), archivo_log)
  escribir_log(paste0(paste(salida, collapse = "\n"), "\n"), archivo_log)
  escribir_log(paste0("--- Fin de salida ---\n"), archivo_log)
  
  escribir_log(paste0("Fin: ", format(fin, "%Y-%m-%d %H:%M:%S"), "\n"), archivo_log)
  escribir_log(paste0("Duracion: ", round(duracion, 2), " segundos\n"), archivo_log)
  escribir_log(paste0("Estado: ", estado, "\n\n"), archivo_log)
  
  resultados <- rbind(resultados, data.frame(
    Script = scr$nombre, Descripcion = scr$descripcion,
    Estado = estado, Duracion_seg = round(duracion, 2)
  ))
}

# --- Resumen final ---
escribir_log(paste0("============================================================\n"), archivo_log)
escribir_log(paste0("RESUMEN DE EJECUCION\n"), archivo_log)
escribir_log(paste0("Fecha de fin: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n"), archivo_log)
escribir_log(paste0("============================================================\n\n"), archivo_log)

print(resultados, row.names = FALSE)
escribir_log(paste0(capture.output(print(resultados, row.names = FALSE)), collapse = "\n"), archivo_log)
escribir_log("\n", archivo_log)

# --- Verificacion de archivos generados ---
esperados_figuras <- c(
  "results/figures/r_fig01_histogramas.png",
  "results/figures/r_fig02_correlacion.png",
  "results/figures/r_fig03_scree.png",
  "results/figures/r_fig04_biplot.png",
  "results/figures/r_fig05_contribuciones.png",
  "results/figures/r_fig06_residuales.png",
  "results/figures/r_fig07_qq.png",
  "results/figures/r_fig08_cluster.png",
  "results/figures/r_fig09_silhouette.png",
  "results/figures/r_fig10_factor_loadings.png"
)

esperados_tablas <- c(
  "results/tables/r_tab01_pca_summary.txt",
  "results/tables/r_tab02_regresion.txt",
  "results/tables/r_tab03_vif.txt",
  "results/tables/r_tab04_centroides.txt",
  "results/tables/r_tab05_factor_loadings.txt",
  "results/tables/r_tab06_kmo.txt"
)

escribir_log("\n--- VERIFICACION DE ARCHIVOS GENERADOS ---\n\n", archivo_log)
escribir_log("FIGURAS:\n", archivo_log)
for (f in esperados_figuras) {
  existe <- file.exists(f)
  tam <- if (existe) paste0(file.info(f)$size, " bytes") else "NO GENERADO"
  estado_str <- ifelse(existe, "OK", "FALTA")
  escribir_log(paste0("  [", estado_str, "] ", f, " (", tam, ")\n"), archivo_log)
}

escribir_log("\nTABLAS:\n", archivo_log)
for (f in esperados_tablas) {
  existe <- file.exists(f)
  tam <- if (existe) paste0(file.info(f)$size, " bytes") else "NO GENERADO"
  estado_str <- ifelse(existe, "OK", "FALTA")
  escribir_log(paste0("  [", estado_str, "] ", f, " (", tam, ")\n"), archivo_log)
}

# Conteo final
total_figuras <- sum(file.exists(esperados_figuras))
total_tablas <- sum(file.exists(esperados_tablas))
escribir_log(paste0("\nTotal figuras generadas: ", total_figuras, "/", length(esperados_figuras), "\n"), archivo_log)
escribir_log(paste0("Total tablas generadas: ", total_tablas, "/", length(esperados_tablas), "\n"), archivo_log)

if (total_figuras == length(esperados_figuras) && total_tablas == length(esperados_tablas)) {
  escribir_log("\n=== TODOS LOS ARCHIVOS FUERON GENERADOS EXITOSAMENTE ===\n", archivo_log)
} else {
  escribir_log("\n=== ADVERTENCIA: Algunos archivos no fueron generados ===\n", archivo_log)
}

escribir_log(paste0("\n============================================================\n"), archivo_log)
escribir_log(paste0("FIN DEL REGISTRO DE EJECUCION\n"), archivo_log)
escribir_log(paste0("============================================================\n"), archivo_log)

cat("\nLog guardado en:", archivo_log, "\n")
cat("=== SCRIPT 06 COMPLETADO EXITOSAMENTE ===\n")