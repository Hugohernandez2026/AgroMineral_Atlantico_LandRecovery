#!/usr/bin/env Rscript
# =============================================================================
# Script 04: Analisis de Conglomerados (K-means)
# Proyecto: AgroMineral-PCA
# Descripcion: K-means con k=3 sobre variables DRX estandarizadas
# =============================================================================

# --- Configuracion de paquetes ---
if (!require("ggplot2")) install.packages("ggplot2", repos = "https://cloud.r-project.org")
if (!require("factoextra")) install.packages("factoextra", repos = "https://cloud.r-project.org")
if (!require("cluster")) install.packages("cluster", repos = "https://cloud.r-project.org")
library(ggplot2)
library(factoextra)
library(cluster)

# --- Rutas y directorios ---
setwd("/home/z/my-project/download/AgroMineral-PCA")
dir.create("results/figures", showWarnings = FALSE, recursive = TRUE)
dir.create("results/tables", showWarnings = FALSE, recursive = TRUE)

# --- Carga de datos ---
cat("=== CARGANDO DATOS PARA CLUSTERING ===\n")
datos <- read.csv("data/processed/agrominerals_drx_master.csv", stringsAsFactors = FALSE)
cat("Muestras cargadas:", nrow(datos), "\n")

# --- Variables DRX para clustering ---
variables_drx <- c("quartz_pct", "micas_pct", "plagioclase_pct",
                   "mafics_pct", "other_minerals_pct")
datos_drx <- datos[, variables_drx]

# --- Estandarizacion de variables ---
cat("Estandarizando variables (media=0, desv.est.=1)...\n")
datos_escalados <- scale(datos_drx)

# --- Ejecucion de K-means con k=3 ---
cat("\n=== EJECUTANDO K-MEANS (k=3) ===\n")
set.seed(42)  # Reproducibilidad
km_result <- kmeans(datos_escalados, centers = 3, nstart = 25, iter.max = 100)
cat("Tamano de los clusters:\n")
print(table(km_result$cluster))

# --- Coeficiente de silueta ---
cat("\n=== CALCULANDO COEFICIENTE DE SILUETA ===\n")
sil <- silhouette(km_result$cluster, dist(datos_escalados))
sil_promedio <- mean(sil[, "sil_width"])
cat("Silueta promedio:", round(sil_promedio, 4), "\n")
cat("Silueta promedio por cluster:\n")
print(summary(sil)$clus.avg.widths)

# --- Figura 08: Grafico de clusters ---
cat("\n=== GENERANDO GRAFICO DE CLUSTERS ===\n")
# PCA para visualizacion en 2D
pca_viz <- prcomp(datos_escalados)
datos_pca <- data.frame(
  PC1 = pca_viz$x[, 1],
  PC2 = pca_viz$x[, 2],
  Cluster = as.factor(km_result$cluster)
)

g_cluster <- ggplot(datos_pca, aes(x = PC1, y = PC2, color = Cluster, fill = Cluster)) +
  geom_point(alpha = 0.5, size = 1.5) +
  stat_ellipse(level = 0.95, geom = "polygon", alpha = 0.15) +
  scale_color_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1") +
  theme_minimal(base_size = 13) +
  labs(
    title = "Clusters K-means (k=3) Visualizados en Espacio PCA",
    subtitle = paste0("Silueta promedio: ", round(sil_promedio, 3)),
    x = paste0("PC1 (", round(pca_viz$sdev[1]^2 / sum(pca_viz$sdev^2) * 100, 1), "% varianza)"),
    y = paste0("PC2 (", round(pca_viz$sdev[2]^2 / sum(pca_viz$sdev^2) * 100, 1), "% varianza)")
  ) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))
ggsave("results/figures/r_fig08_cluster.png", g_cluster,
       width = 10, height = 8, dpi = 150)
cat("Guardado: results/figures/r_fig08_cluster.png\n")

# --- Figura 09: Silhouette plot ---
cat("\n=== GENERANDO GRAFICO DE SILUETA ===\n")
png("results/figures/r_fig09_silhouette.png",
    width = 10, height = 8, units = "in", res = 150)
plot(sil, col = c("#E41A1C", "#377EB8", "#4DAF4A")[km_result$cluster],
     border = NA,
     main = "Grafico de Silueta - K-means (k=3)",
     xlab = "Ancho de Silueta",
     ylab = "Cluster",
     cex.names = 0.8)
dev.off()
cat("Guardado: results/figures/r_fig09_silhouette.png\n")

# --- Perfiles de clusters (centroides) ---
cat("\n=== CENTROIDES DE LOS CLUSTERS ===\n")
centroides <- data.frame(
  Variable = variables_drx,
  Cluster_1 = unname(round(km_result$centers[1, ], 4)),
  Cluster_2 = unname(round(km_result$centers[2, ], 4)),
  Cluster_3 = unname(round(km_result$centers[3, ], 4)),
  row.names = NULL,
  stringsAsFactors = FALSE
)
print(centroides)

# --- Guardar centroides en archivo ---
sink("results/tables/r_tab04_centroides.txt")
cat("============================================================\n")
cat("CENTROIDES DE CLUSTERS K-MEANS (k=3)\n")
cat("Proyecto: AgroMineral-PCA\n")
cat("Fecha:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n\n")
cat("Variables DRX estandarizadas:\n")
cat(paste(variables_drx, collapse = ", "), "\n\n")
cat("Tamano de los clusters:\n")
print(table(km_result$cluster))
cat("\nSilueta promedio:", round(sil_promedio, 4), "\n\n")
cat("Silueta promedio por cluster:\n")
print(round(summary(sil)$clus.avg.widths, 4))
cat("\n--- Centroides (en escala estandarizada) ---\n")
print(centroides, row.names = FALSE)
cat("\n--- Centroides en escala original ---\n")
for (k in 1:3) {
  cat(paste0("\nCluster ", k, " (n=", sum(km_result$cluster == k), "): Media original:\n"))
  cluster_datos <- datos_drx[km_result$cluster == k, ]
  print(round(colMeans(cluster_datos), 2))
}
sink()
cat("Guardado: results/tables/r_tab04_centroides.txt\n")

cat("\n=== SCRIPT 04 COMPLETADO EXITOSAMENTE ===\n")