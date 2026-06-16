# =========================================================
# SOM analysis for daily JJA 850 hPa uwnd
# Period : 1991-2023
# Domain : 0-60N, 100-180E
# Method : 3x3 SOM using kohonen
# Features:
#   - JJA anomaly
#   - area weighting
#   - annual occurrence frequency CSV
# =========================================================

# ----------------------------
# 0. packages
# ----------------------------
library(terra)
library(kohonen)

# ----------------------------
# 1. user settings
# ----------------------------
repo_root <- "/home/user/EASM"
processed_dir <- file.path(repo_root, "data")
outdir <- file.path(repo_root, "outputs", "som_u850")
infile <- file.path(processed_dir, "uwnd_z850_jja_1982_2025.nc")
varname <- "uwnd"

if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

set.seed(42)  # the answer to life the universe and everything!

# SOM settings
xdim <- 3
ydim <- 3
rlen <- 1000
alpha_vals <- c(0.5, 0.01)

# ----------------------------
# 2. read data
# ----------------------------
r <- rast(infile)

# If the file contains multiple variables or levels, this simple version
# assumes the selected raster already corresponds to daily 850hPa uwnd.
# If not, the file structure must be adjusted upstream.

# ----------------------------
# 3. crop domain
#    domain: 100E-180E, 0-60N
# ----------------------------
r <- crop(r, ext(100, 180, 0, 60))

# ----------------------------
# 4. select JJA 1982-2025
# ----------------------------
tt <- as.Date(time(r))
yy <- as.integer(format(tt, "%Y"))
mm <- as.integer(format(tt, "%m"))
dd <- as.integer(format(tt, "%d"))

idx <- which(yy >= 1982 & yy <= 2025 & mm %in% c(6, 7, 8))

r_jja <- r[[idx]]
t_jja <- tt[idx]
y_jja <- yy[idx]
m_jja <- mm[idx]
d_jja <- dd[idx]

cat("Selected daily samples:", nlyr(r_jja), "\n")

# ----------------------------
# 5. make JJA monthly anomaly
#    anomaly = daily value - monthly climatology
#    climatology based on all JJA days in 1991-2023 for each month separately
# ----------------------------
# month-specific climatology (June, July, August)
clim_list <- vector("list", 3)
names(clim_list) <- c("6", "7", "8")

for (mon in c(6, 7, 8)) {
  ii <- which(m_jja == mon)
  clim_list[[as.character(mon)]] <- mean(r_jja[[ii]], na.rm = TRUE)
}

# anomaly raster stack
anom_layers <- vector("list", length = nlyr(r_jja))

for (i in seq_len(nlyr(r_jja))) {
  mon_i <- m_jja[i]
  anom_layers[[i]] <- r_jja[[i]] - clim_list[[as.character(mon_i)]]
}

r_anom <- rast(anom_layers)
time(r_anom) <- t_jja

# ----------------------------
# 6. area weighting
#    weight = sqrt(cos(lat))
#    often used so that Euclidean distance in SOM is less dominated by
#    high-latitude grid density
# ----------------------------
lat_rast <- init(r_anom[[1]], "y")
w_rast <- sqrt(cos(pi * lat_rast / 180))

# apply weights to each day
r_weighted <- r_anom * w_rast

# ----------------------------
# 7. make SOM input matrix
#    rows = time samples
#    cols = grid cells
# ----------------------------
# values() returns [ncell x nlyr]
x0 <- values(r_weighted)
dim(x0)

# transpose -> [ntime x ncell]
x <- t(x0)

# remove samples with any missing values
keep_row <- complete.cases(x)
x <- x[keep_row, , drop = FALSE]

t_use <- t_jja[keep_row]
y_use <- y_jja[keep_row]
m_use <- m_jja[keep_row]
d_use <- d_jja[keep_row]

# remove cells with any missing values
keep_col <- colSums(is.na(x)) == 0
x <- x[, keep_col, drop = FALSE]

cat("Final SOM matrix dimension:", dim(x)[1], "samples x", dim(x)[2], "grid cells\n")

# standardize each variable (grid point)
x_scaled <- scale(x)

# ----------------------------
# 8. train SOM
# ----------------------------
grid_3x3 <- kohonen::somgrid(xdim = xdim, ydim = ydim, topo = "rectangular")

som_model <- kohonen::som(
  X = x_scaled,
  grid = grid_3x3,
  rlen = rlen,
  alpha = alpha_vals,
  keep.data = TRUE
)

# ----------------------------
# 9. basic diagnostics
# ----------------------------
png(file.path(outdir, "som_diagnostics.png"), width = 1700, height = 1350, res = 170)
par(
  mfrow = c(2, 2),
  mar = c(5.2, 6.0, 3.4, 2.8),
  mgp = c(2.8, 0.9, 0),
  oma = c(0.4, 0.4, 0.4, 0.4),
  xpd = NA
)
plot(som_model, type = "changes", main = "Training Changes")
plot(som_model, type = "counts", main = "Counts")
plot(som_model, type = "quality", main = "Quality")
plot(som_model, type = "dist.neighbours", main = "Neighbour Distances")
dev.off()

# ----------------------------
# 10. BMU assignment
# ----------------------------
bmu <- som_model$unit.classif

assign_df <- data.frame(
  date  = t_use,
  year  = y_use,
  month = m_use,
  day   = d_use,
  node  = bmu
)

write.csv(assign_df, file.path(outdir, "som_daily_assignment.csv"), row.names = FALSE)

# ----------------------------
# 11. node counts
# ----------------------------
n_nodes <- xdim * ydim
node_counts <- as.integer(table(factor(bmu, levels = seq_len(n_nodes))))

# ----------------------------
# 12. occurrence frequency by year
#     frequency within each year's JJA samples
# ----------------------------
year_seq <- sort(unique(assign_df$year))

freq_mat <- matrix(
  0,
  nrow = length(year_seq),
  ncol = n_nodes,
  dimnames = list(year_seq, paste0("Node_", seq_len(n_nodes)))
)

for (i in seq_along(year_seq)) {
  yr <- year_seq[i]
  sub <- assign_df[assign_df$year == yr, ]
  tab <- table(factor(sub$node, levels = seq_len(n_nodes)))
  freq_mat[i, ] <- as.numeric(tab) / sum(tab)
}

freq_df <- data.frame(
  year = year_seq,
  freq_mat,
  check.names = FALSE
)

write.csv(freq_df, file.path(outdir, "som_occurrence_frequency_by_year.csv"), row.names = FALSE)

cat("Render occurrence timeseries with:\n")
cat("  Rscript scripts/render_occurrence_timeseries.R\n")

# ----------------------------
# 13. optional: node-wise mean occurrence over all years
# ----------------------------
node_mean_freq <- colMeans(freq_df[, -1], na.rm = TRUE)
node_mean_freq_df <- data.frame(
  node = seq_len(n_nodes),
  mean_occurrence_frequency = node_mean_freq,
  count = node_counts
)
write.csv(node_mean_freq_df,
  file.path(outdir, "som_node_mean_occurrence_summary.csv"),
  row.names = FALSE
)

# ----------------------------
# 14. save model object
# ----------------------------
saveRDS(som_model, file.path(outdir, "som_model.rds"))

cat("All outputs saved in:", outdir, "\n")
