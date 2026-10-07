#!/usr/bin/env Rscript
# =============================================================================
# Script 01: Analisis Exploratorio de Datos (AED)
# Proyecto: AgroMineral-PCA
# Descripcion: Carga datos DRX, genera histogramas y mapa de correlacion
# =============================================================================

# --- Configuracion de paquetes ---
if (!require("ggplot2")) install.packages("ggplot2", repos = "https://cloud.r-project.org")
if (!require("corrplot")) install.packages("corrplot", repos = "https://cloud.r-project.org")
library(ggplot2)
library(corrplot)

# --- Rutas y directorios ---
setwd("/home/z/my-project/download/AgroMineral-PCA")
dir.create("results/figures", showWarnings = FALSE, recursive = TRUE)
dir.create("results/tables", showWarnings = FALSE, recursive = TRUE)

# --- Carga de datos ---
cat("=== CARGANDO DATOS ===\n")
datos <- read.csv("data/processed/agrominerals_drx_master.csv", stringsAsFactors = FALSE)
cat("Dimensiones del conjunto de datos:", nrow(datos), "muestras x", ncol(datos), "variables\n\n")

# --- Estructura y resumen ---
cat("=== ESTRUCTURA (str) ===\n")
str(datos)
cat("\n=== RESUMEN ESTADISTICO (summary) ===\n")
print(summary(datos))

# --- Valores faltantes ---
cat("\n=== VALORES FALTANTES POR VARIABLE ===\n")
valores_faltantes <- colSums(is.na(datos))
print(valores_faltantes)
cat("Total de valores faltantes:", sum(valores_faltantes), "\n")

# --- Variables numericas para analisis ---
variables_num <- c("SiO2_pct", "K2O_pct", "CaO_pct", "MgO_pct", "ICM_index")

# --- Preparar datos para histogramas (formato largo) ---
datos_largo <- datos[, variables_num]
datos_largo <- data.frame(
  Variable = rep(names(datos_largo), each = nrow(datos_largo)),
  Valor = unlist(datos_largo)
)
# Nombres en espanol para etiquetas
nombres_espanol <- c(
  SiO2_pct = "SiO2 (%)",
  K2O_pct  = "K2O (%)",
  CaO_pct  = "CaO (%)",
  MgO_pct  = "MgO (%)",
  ICM_index = "Indice ICM"
)
datos_largo$Variable <- factor(datos_largo$Variable, levels = variables_num,
                                labels = nombres_espanol[variables_num])

# --- Generar histogramas con ggplot2 ---
cat("\n=== GENERANDO HISTOGRAMAS ===\n")
g_hist <- ggplot(datos_largo, aes(x = Valor)) +
  geom_histogram(bins = 40, fill = "steelblue", color = "white", alpha = 0.8) +
  facet_wrap(~ Variable, scales = "free") +
  theme_minimal(base_size = 12) +
  labs(
    title = "Distribucion de Variables Quimicas Principales y Indice ICM",
    x = "Valor", y = "Frecuencia"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    strip.text = element_text(face = "bold")
  )

ggsave("results/figures/r_fig01_histogramas.png", g_hist,
       width = 14, height = 10, dpi = 150)
cat("Guardado: results/figures/r_fig01_histogramas.png\n")

# --- Matriz de correlacion ---
cat("\n=== CALCULANDO MATRIZ DE CORRELACION ===\n")
datos_numericos <- datos[, sapply(datos, is.numeric)]
mat_cor <- cor(datos_numericos, use = "pairwise.complete.obs")
cat("Dimensiones de la matriz de correlacion:", nrow(mat_cor), "x", ncol(mat_cor), "\n")

# --- Mapa de calor de correlacion con corrplot ---
cat("\n=== GENERANDO MAPA DE CORRELACION ===\n")
png("results/figures/r_fig02_correlacion.png",
    width = 12, height = 10, units = "in", res = 150)
corrplot(mat_cor, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45,
         col = colorRampPalette(c("#B2182B", "white", "#2166AC"))(200),
         title = "Mapa de Calor - Correlacion entre Variables",
         mar = c(0, 0, 2, 0),
         addCoef.col = "black", number.cex = 0.5,
         diag = FALSE)
dev.off()
cat("Guardado: results/figures/r_fig02_correlacion.png\n")

cat("\n=== SCRIPT 01 COMPLETADO EXITOSAMENTE ===\n")