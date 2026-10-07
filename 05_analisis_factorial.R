#!/usr/bin/env Rscript
# =============================================================================
# Script 05: Analisis Factorial
# Proyecto: AgroMineral-PCA
# Descripcion: Analisis factorial con rotacion varimax, KMO, cargas factoriales
# Nota: Se usan 4 variables DRX (excluyendo other_minerals_pct) porque
#       las 5 variables suman ~100%, creando dependencia lineal (singularidad).
# =============================================================================

# --- Configuracion de paquetes ---
if (!require("ggplot2")) install.packages("ggplot2", repos = "https://cloud.r-project.org")
if (!require("psych")) install.packages("psych", repos = "https://cloud.r-project.org")
library(ggplot2)
library(psych)

# --- Rutas y directorios ---
setwd("/home/z/my-project/download/AgroMineral-PCA")
dir.create("results/figures", showWarnings = FALSE, recursive = TRUE)
dir.create("results/tables", showWarnings = FALSE, recursive = TRUE)

# --- Carga de datos ---
cat("=== CARGANDO DATOS PARA ANALISIS FACTORIAL ===\n")
datos <- read.csv("data/processed/agrominerals_drx_master.csv", stringsAsFactors = FALSE)
cat("Muestras cargadas:", nrow(datos), "\n")

# --- Variables DRX para analisis factorial ---
# Se usan solo 4 variables porque las 5 originales suman ~100%
# (dependencia lineal), lo que produce matriz singular.
variables_fa <- c("quartz_pct", "micas_pct", "plagioclase_pct", "mafics_pct")
datos_fa <- datos[, variables_fa]
cat("Variables utilizadas (4 de 5, excluyendo other_minerals_pct por colinealidad):\n")
cat(paste(variables_fa, collapse = ", "), "\n")

# --- Test KMO (Kaiser-Meyer-Olkin) ---
cat("\n=== TEST KMO (Kaiser-Meyer-Olkin) ===\n")
kmo_result <- KMO(datos_fa)
cat("KMO global:", round(kmo_result$MSA, 4), "\n\n")
cat("KMO por variable:\n")
print(round(kmo_result$MSAi, 4))

# --- Interpretacion del KMO ---
cat("\nInterpretacion del KMO:\n")
kmo_val <- kmo_result$MSA
if (kmo_val >= 0.9) {
  cat("  Excelente (>= 0.9) - Muy adecuado para analisis factorial\n")
  kmo_interpretacion <- "Excelente - Muy adecuado para analisis factorial"
} else if (kmo_val >= 0.8) {
  cat("  Bueno (>= 0.8) - Adecuado para analisis factorial\n")
  kmo_interpretacion <- "Bueno - Adecuado para analisis factorial"
} else if (kmo_val >= 0.7) {
  cat("  Aceptable (>= 0.7) - Adecuado para analisis factorial\n")
  kmo_interpretacion <- "Aceptable - Adecuado para analisis factorial"
} else if (kmo_val >= 0.6) {
  cat("  Regular (>= 0.6) - Marginalmente aceptable\n")
  kmo_interpretacion <- "Regular - Marginalmente aceptable"
} else if (kmo_val >= 0.5) {
  cat("  Pobre (>= 0.5) - No muy recomendado\n")
  kmo_interpretacion <- "Pobre - No muy recomendado"
} else {
  cat("  Inaceptable (< 0.5) - No recomendado\n")
  cat("  Nota: KMO bajo es esperado cuando las variables son composicionales\n")
  cat("  (porcentajes que reflejan categorias mutuamente excluyentes).\n")
  kmo_interpretacion <- "Inaceptable - No recomendado (esperado en datos composicionales)"
}

# --- Guardar KMO en archivo ---
sink("results/tables/r_tab06_kmo.txt")
cat("============================================================\n")
cat("PRUEBA KMO (KAISER-MEYER-OLKIN)\n")
cat("Proyecto: AgroMineral-PCA\n")
cat("Fecha:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n\n")
cat("Variables utilizadas (4 de 5):\n")
cat(paste(variables_fa, collapse = ", "), "\n")
cat("Nota: other_minerals_pct se excluye porque las 5 variables DRX\n")
cat("      suman ~100%, creando dependencia lineal.\n\n")
cat("KMO global:", round(kmo_result$MSA, 4), "\n\n")
cat("KMO por variable:\n")
print(round(kmo_result$MSAi, 4))
cat("\nInterpretacion:", kmo_interpretacion, "\n")
sink()
cat("Guardado: results/tables/r_tab06_kmo.txt\n")

# --- Analisis Factorial con rotacion varimax ---
# Se usa la matriz de correlacion directamente para evitar problemas
cat("\n=== EJECUTANDO ANALISIS FACTORIAL (rotacion varimax) ===\n")
mat_cor <- cor(datos_fa)
cat("Matriz de correlacion:\n")
print(round(mat_cor, 4))

fa_result <- fa(r = mat_cor, nfactors = 2, rotate = "varimax",
                fm = "pa", n.obs = nrow(datos_fa))

# --- Resumen del analisis factorial ---
cat("\n=== RESUMEN DEL ANALISIS FACTORIAL ===\n")
print(fa_result, sort = TRUE)

# --- Cargas factoriales ---
cat("\n=== CARGAS FACTORIALES ===\n")
cargas <- fa_result$loadings
print(round(cargas, 4))

# --- Comunalidades ---
cat("\n=== COMUNALIDADES ===\n")
comunalidades <- fa_result$communality
print(round(comunalidades, 4))

# --- Varianza explicada por factor ---
cat("\n=== VARIANZA EXPLICADA POR FACTOR ===\n")
print(round(fa_result$Vaccounted, 4))

# --- Guardar cargas factoriales en archivo ---
sink("results/tables/r_tab05_factor_loadings.txt")
cat("============================================================\n")
cat("CARGAS FACTORIALES - ANALISIS FACTORIAL (varimax)\n")
cat("Proyecto: AgroMineral-PCA\n")
cat("Fecha:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n\n")
cat("Variables utilizadas (4 de 5):\n")
cat(paste(variables_fa, collapse = ", "), "\n")
cat("Numero de factores: 2\n")
cat("Rotacion: varimax\n")
cat("Metodo: ejes principales (pa)\n")
cat("Nota: other_minerals_pct se excluyo por dependencia lineal.\n\n")
cat("--- Matriz de Correlacion ---\n")
print(round(mat_cor, 4))
cat("\n--- Cargas Factoriales (rotadas) ---\n")
print(round(cargas, 4))
cat("\n--- Comunalidades (h2) ---\n")
comun_df <- data.frame(
  Variable = names(comunalidades),
  Comunalidad = round(comunalidades, 4),
  row.names = NULL
)
print(comun_df, row.names = FALSE)
cat("\n--- Varianza Explicada ---\n")
print(round(fa_result$Vaccounted, 4))
cat("\n--- Estructura del Modelo ---\n")
print(fa_result, sort = TRUE)
sink()
cat("Guardado: results/tables/r_tab05_factor_loadings.txt\n")

# --- Figura 10: Grafico de cargas factoriales ---
cat("\n=== GENERANDO GRAFICO DE CARGAS FACTORIALES ===\n")
cargas_df <- data.frame(
  Variable = rownames(fa_result$loadings),
  Factor1 = unname(fa_result$loadings[, 1]),
  Factor2 = unname(fa_result$loadings[, 2]),
  row.names = NULL
)

g_fa <- ggplot(cargas_df, aes(x = Factor1, y = Factor2, label = Variable)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey60") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
  geom_point(size = 4, color = "steelblue") +
  geom_text(vjust = -0.8, hjust = 0.5, size = 3.5, fontface = "bold") +
  xlim(-1.2, 1.2) + ylim(-1.2, 1.2) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Diagrama de Cargas Factoriales (Rotacion Varimax)",
    subtitle = paste0("Varianza explicada: F1=",
                       round(fa_result$Vaccounted[2, 1] * 100, 1), "%, F2=",
                       round(fa_result$Vaccounted[2, 2] * 100, 1), "%"),
    x = "Factor 1",
    y = "Factor 2"
  ) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))
ggsave("results/figures/r_fig10_factor_loadings.png", g_fa,
       width = 10, height = 8, dpi = 150)
cat("Guardado: results/figures/r_fig10_factor_loadings.png\n")

# --- Comparacion de puntuaciones factoriales con ICM_index ---
cat("\n=== COMPARACION DE PUNTUACIONES FACTORIALES CON ICM_index ===\n")
# Calcular puntuaciones factoriales usando las 4 variables
punt_fa <- factor.scores(datos_fa, fa_result)
if (is.list(punt_fa)) {
  scores_fa <- punt_fa$scores
} else {
  scores_fa <- punt_fa
}

cor_f1_icm <- cor(scores_fa[, 1], datos$ICM_index, use = "complete.obs")
cor_f2_icm <- cor(scores_fa[, 2], datos$ICM_index, use = "complete.obs")
cat("Correlacion Factor1 - ICM_index:", round(cor_f1_icm, 4), "\n")
cat("Correlacion Factor2 - ICM_index:", round(cor_f2_icm, 4), "\n")

# Modelo de regresion para cuantificar relacion
datos_con_fa <- data.frame(
  ICM_index = datos$ICM_index,
  Factor1 = scores_fa[, 1],
  Factor2 = scores_fa[, 2]
)
modelo_fi <- lm(ICM_index ~ Factor1 + Factor2, data = datos_con_fa)
cat("\nRegresion de ICM_index sobre puntuaciones factoriales:\n")
cat("  R^2 ajustado:", round(summary(modelo_fi)$adj.r.squared, 4), "\n")
cat("  Coef. Factor1:", round(coef(modelo_fi)[2], 4),
    " (p=", round(summary(modelo_fi)$coefficients[2, 4], 4), ")\n")
cat("  Coef. Factor2:", round(coef(modelo_fi)[3], 4),
    " (p=", round(summary(modelo_fi)$coefficients[3, 4], 4), ")\n")

cat("\n=== SCRIPT 05 COMPLETADO EXITOSAMENTE ===\n")