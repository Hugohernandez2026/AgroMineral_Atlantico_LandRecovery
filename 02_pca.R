#!/usr/bin/env Rscript
# =============================================================================
# Script 02: Analisis de Componentes Principales (PCA)
# Proyecto: AgroMineral-PCA
# Descripcion: PCA sobre variables de fases DRX, scree plot, biplot, contribuciones
# =============================================================================

# --- Configuracion de paquetes ---
if (!require("ggplot2")) install.packages("ggplot2", repos = "https://cloud.r-project.org")
if (!require("factoextra")) install.packages("factoextra", repos = "https://cloud.r-project.org")
library(ggplot2)
library(factoextra)

# --- Rutas y directorios ---
setwd("/home/z/my-project/download/AgroMineral-PCA")
dir.create("results/figures", showWarnings = FALSE, recursive = TRUE)
dir.create("results/tables", showWarnings = FALSE, recursive = TRUE)

# --- Carga de datos ---
cat("=== CARGANDO DATOS PARA PCA ===\n")
datos <- read.csv("data/processed/agrominerals_drx_master.csv", stringsAsFactors = FALSE)
cat("Muestras cargadas:", nrow(datos), "\n")

# --- Seleccion de variables de fases DRX ---
variables_drx <- c("quartz_pct", "micas_pct", "plagioclase_pct",
                   "mafics_pct", "other_minerals_pct")
datos_drx <- datos[, variables_drx]
cat("Variables DRX utilizadas:", paste(variables_drx, collapse = ", "), "\n")

# --- Ejecucion de PCA con prcomp ---
cat("\n=== EJECUTANDO PCA (prcomp, scale=TRUE) ===\n")
pca_result <- prcomp(datos_drx, scale. = TRUE, center = TRUE)

# --- Resumen del PCA ---
cat("\n=== RESUMEN DEL PCA ===\n")
summary_pca <- summary(pca_result)
print(summary_pca)

# --- Varianza explicada ---
var_exp <- pca_result$sdev^2 / sum(pca_result$sdev^2) * 100
cat("\nVarianza explicada por componente:\n")
print(round(var_exp, 2))

# --- Cargas de los componentes ---
cat("\n=== CARGAS (LOADINGS) ===\n")
print(round(pca_result$rotation, 4))

# --- Guardar resumen en archivo de texto ---
sink("results/tables/r_tab01_pca_summary.txt")
cat("============================================================\n")
cat("RESUMEN DEL ANALISIS DE COMPONENTES PRINCIPALES (PCA)\n")
cat("Proyecto: AgroMineral-PCA\n")
cat("Fecha:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n\n")
cat("Variables utilizadas:\n")
cat(paste(variables_drx, collapse = ", "), "\n\n")
cat("Muestras:", nrow(datos_drx), "\n\n")
cat("--- Resumen de Importancia de Componentes ---\n")
print(summary_pca)
cat("\n--- Varianza Explicada (%) ---\n")
print(round(var_exp, 2))
cat("\n--- Cargas (Loadings) ---\n")
print(round(pca_result$rotation, 4))
cat("\n--- Valores Propios (Eigenvalues) ---\n")
print(round(pca_result$sdev^2, 4))
cat("\n--- Media de las variables (antes de escalar) ---\n")
print(round(colMeans(datos_drx), 4))
cat("\n--- Desviacion Estandar (antes de escalar) ---\n")
print(round(apply(datos_drx, 2, sd), 4))
sink()
cat("Guardado: results/tables/r_tab01_pca_summary.txt\n")

# --- Figura 03: Scree Plot (diagrama de codo) ---
cat("\n=== GENERANDO SCREE PLOT ===\n")
g_scree <- fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 60),
                    xlab = "Componente Principal",
                    ylab = "Porcentaje de Varianza Explicada",
                    ggtheme = theme_minimal(base_size = 13)) +
  ggtitle("Diagrama de Codo (Scree Plot) - PCA Variables DRX") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
ggsave("results/figures/r_fig03_scree.png", g_scree,
       width = 10, height = 7, dpi = 150)
cat("Guardado: results/figures/r_fig03_scree.png\n")

# --- Figura 04: Biplot coloreado por malla (mesh) ---
cat("\n=== GENERANDO BIPLOT ===\n")
datos$mesh <- factor(datos$mesh)
g_biplot <- fviz_pca_biplot(pca_result,
                            habillage = datos$mesh,
                            addEllipses = TRUE,
                            alpha.ind = 0.3,
                            col.var = "steelblue",
                            label = "var",
                            repel = TRUE,
                            xlab = paste0("PC1 (", round(var_exp[1], 1), "%)"),
                            ylab = paste0("PC2 (", round(var_exp[2], 1), "%)"),
                            legend.title = "Mesh",
                            ggtheme = theme_minimal(base_size = 13)) +
  ggtitle("Biplot PCA - Coloreado por Granulometria (Mesh)") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
ggsave("results/figures/r_fig04_biplot.png", g_biplot,
       width = 12, height = 9, dpi = 150)
cat("Guardado: results/figures/r_fig04_biplot.png\n")

# --- Figura 05: Contribuciones de variables ---
cat("\n=== GENERANDO GRAFICO DE CONTRIBUCIONES ===\n")
g_contrib <- fviz_pca_var(pca_result,
                          col.var = "contrib",
                          gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                          repel = TRUE,
                          ggtheme = theme_minimal(base_size = 13)) +
  ggtitle("Contribuciones de Variables a PC1 y PC2") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
ggsave("results/figures/r_fig05_contribuciones.png", g_contrib,
       width = 10, height = 8, dpi = 150)
cat("Guardado: results/figures/r_fig05_contribuciones.png\n")

cat("\n=== SCRIPT 02 COMPLETADO EXITOSAMENTE ===\n")