#!/usr/bin/env Rscript
# =============================================================================
# Script 03: Regresion Lineal Multiple
# Proyecto: AgroMineral-PCA
# Descripcion: Modelo de regresion para Solubilidad_Relativa con diagnosticos
# =============================================================================

# --- Configuracion de paquetes ---
if (!require("ggplot2")) install.packages("ggplot2", repos = "https://cloud.r-project.org")
if (!require("car")) install.packages("car", repos = "https://cloud.r-project.org")
library(ggplot2)
library(car)

# --- Rutas y directorios ---
setwd("/home/z/my-project/download/AgroMineral-PCA")
dir.create("results/figures", showWarnings = FALSE, recursive = TRUE)
dir.create("results/tables", showWarnings = FALSE, recursive = TRUE)

# --- Carga de datos ---
cat("=== CARGANDO DATOS DE REMINERALIZACION ===\n")
datos <- read.csv("data/processed/remineralizacion_roca.csv", stringsAsFactors = FALSE)
cat("Dimensiones:", nrow(datos), "muestras x", ncol(datos), "variables\n")
cat("Variables:", paste(colnames(datos), collapse = ", "), "\n\n")

# --- Ajuste del modelo de regresion lineal multiple ---
cat("=== AJUSTANDO MODELO DE REGRESION LINEAL MULTIPLE ===\n")
cat("Formula: Solubilidad_Relativa ~ SiO2_pct + K2O_pct + CaO_pct + MgO_pct + Granulometria_Mesh\n\n")
modelo <- lm(Solubilidad_Relativa ~ SiO2_pct + K2O_pct + CaO_pct + MgO_pct + Granulometria_Mesh,
             data = datos)

# --- Resumen del modelo ---
resumen_modelo <- summary(modelo)
cat("=== RESUMEN DEL MODELO ===\n")
print(resumen_modelo)

# --- Tabla ANOVA ---
cat("\n=== TABLA ANOVA ===\n")
anova_modelo <- anova(modelo)
print(anova_modelo)

# --- Factor de Inflacion de Varianza (VIF) ---
cat("\n=== FACTOR DE INFLACION DE VARIANZA (VIF) ===\n")
vif_valores <- vif(modelo)
print(round(vif_valores, 4))

# --- Guardar resumen de regresion en archivo ---
sink("results/tables/r_tab02_regresion.txt")
cat("============================================================\n")
cat("RESUMEN DEL MODELO DE REGRESION LINEAL MULTIPLE\n")
cat("Proyecto: AgroMineral-PCA\n")
cat("Fecha:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n\n")
cat("Formula: Solubilidad_Relativa ~ SiO2_pct + K2O_pct + CaO_pct + MgO_pct + Granulometria_Mesh\n")
cat("Muestras:", nrow(datos), "\n\n")
cat("--- Resumen del Modelo ---\n")
print(resumen_modelo)
cat("\n--- Tabla ANOVA ---\n")
print(anova_modelo)
sink()
cat("Guardado: results/tables/r_tab02_regresion.txt\n")

# --- Guardar VIF en archivo ---
sink("results/tables/r_tab03_vif.txt")
cat("============================================================\n")
cat("FACTOR DE INFLACION DE VARIANZA (VIF)\n")
cat("Proyecto: AgroMineral-PCA\n")
cat("Fecha:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n\n")
cat("Valores VIF por predictor:\n")
print(round(vif_valores, 4))
cat("\nInterpretacion:\n")
cat("  VIF < 5: Multicolinealidad baja (aceptable)\n")
cat("  5 <= VIF < 10: Multicolinealidad moderada\n")
cat("  VIF >= 10: Multicolinealidad alta (problematica)\n")
cat("\nConclusión: ")
if (all(vif_valores < 5)) {
  cat("No se detecta multicolinealidad problematica.\n")
} else if (all(vif_valores < 10)) {
  cat("Multicolinealidad moderada en algunas variables.\n")
} else {
  cat("ALERTA: Multicolinealidad alta detectada.\n")
}
sink()
cat("Guardado: results/tables/r_tab03_vif.txt\n")

# --- Figura 06: Grafico de residuales ---
cat("\n=== GENERANDO GRAFICO DE RESIDUALES ===\n")
residuales <- resid(modelo)
valores_ajustados <- fitted(modelo)

g_resid <- ggplot(data.frame(Residuales = residuales,
                              Ajustados = valores_ajustados),
                   aes(x = Ajustados, y = Residuales)) +
  geom_point(alpha = 0.4, color = "steelblue", size = 1.2) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 0.8) +
  geom_smooth(method = "loess", se = TRUE, color = "darkred", linewidth = 0.6) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Residuales vs Valores Ajustados",
    subtitle = "Diagnostico de Homocedasticidad",
    x = "Valores Ajustados",
    y = "Residuales"
  ) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))
ggsave("results/figures/r_fig06_residuales.png", g_resid,
       width = 10, height = 7, dpi = 150)
cat("Guardado: results/figures/r_fig06_residuales.png\n")

# --- Figura 07: Grafico QQ ---
cat("\n=== GENERANDO GRAFICO QQ ===\n")
# Calcular cuantiles teoricos y de los datos para el grafico QQ
qq_data <- as.data.frame(qqnorm(residuales, plot.it = FALSE))
colnames(qq_data) <- c("Teoricos", "Muestrales")
qq_data$lim_inf <- qnorm(ppoints(length(residuales))) - 1.96 * sqrt((1 - ppoints(length(residuales))) * ppoints(length(residuales)) / dnorm(qq_data$Teoricos)^2)
qq_data$lim_sup <- qnorm(ppoints(length(residuales))) + 1.96 * sqrt((1 - ppoints(length(residuales))) * ppoints(length(residuales)) / dnorm(qq_data$Teoricos)^2)

# Simplificar: usar bandas de confianza estandar
conf_band <- 1.96 / sqrt(length(residuales))

g_qq <- ggplot(qq_data, aes(x = Teoricos, y = Muestrales)) +
  geom_point(alpha = 0.4, color = "steelblue", size = 1.2) +
  geom_abline(intercept = 0, slope = 1, color = "red", linewidth = 0.8) +
  geom_ribbon(aes(ymin = Teoricos - conf_band, ymax = Teoricos + conf_band),
              alpha = 0.15, fill = "grey50") +
  theme_minimal(base_size = 13) +
  labs(
    title = "Grafico Q-Q de Residuales",
    subtitle = "Diagnostico de Normalidad",
    x = "Cuantiles Teoricos (Normal)",
    y = "Cuantiles Muestrales"
  ) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))
ggsave("results/figures/r_fig07_qq.png", g_qq,
       width = 9, height = 7, dpi = 150)
cat("Guardado: results/figures/r_fig07_qq.png\n")

cat("\n=== SCRIPT 03 COMPLETADO EXITOSAMENTE ===\n")