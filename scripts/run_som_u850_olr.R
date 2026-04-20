# =========================================================
# Multivariate SOM analysis for daily JJA 850 hPa uwnd + OLR
# Period : 1991-2023
# Domain : 0-60N, 100-180E
# Method : 3x3 SOM using kohonen
# Features:
#   - u850 MM-DD anomaly
#   - convection proxy from -OLR anomaly
#   - area weighting
#   - node codebook and composite maps
#   - annual occurrence frequency
# =========================================================

library(terra)
library(kohonen)
library(ncdf4)

find_repo_root <- function(start) {
  current <- normalizePath(start, winslash = "/", mustWork = TRUE)

  repeat {
    has_repo_shape <- dir.exists(file.path(current, "scripts")) &&
      dir.exists(file.path(current, "notebooks")) &&
      dir.exists(file.path(current, "data"))

    if (has_repo_shape) {
      return(current)
    }

    parent <- dirname(current)
    if (identical(parent, current)) {
      break
    }
    current <- parent
  }

  stop("Could not locate the repository root from the current execution context.")
}

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  matches <- grep(file_arg, args, value = TRUE)

  if (length(matches) > 0) {
    script_path <- sub(file_arg, "", matches[1])
    return(dirname(normalizePath(script_path, winslash = "/", mustWork = TRUE)))
  }

  frame_files <- vapply(sys.frames(), function(env) {
    if (exists("ofile", envir = env, inherits = FALSE)) {
      get("ofile", envir = env, inherits = FALSE)
    } else {
      NA_character_
    }
  }, character(1))

  frame_files <- frame_files[!is.na(frame_files)]
  if (length(frame_files) > 0) {
    return(dirname(normalizePath(frame_files[1], winslash = "/", mustWork = TRUE)))
  }

  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

read_nc_dates <- function(ncfile) {
  nc <- nc_open(ncfile)
  on.exit(nc_close(nc))

  if (!("time" %in% names(nc$dim) || "time" %in% names(nc$var))) {
    stop(sprintf("No time variable found in %s", ncfile))
  }

  time_vals <- ncvar_get(nc, "time")
  unit_att <- ncatt_get(nc, "time", "units")$value

  unit_lower <- tolower(unit_att)
  origin_str <- sub(".*since\\s+", "", unit_lower)
  origin_str <- gsub("t", " ", origin_str, fixed = TRUE)
  origin <- as.POSIXct(origin_str, tz = "UTC")

  if (is.na(origin)) {
    stop(sprintf("Failed to parse time origin '%s' in %s", unit_att, ncfile))
  }

  multiplier <- if (grepl("^hours since", unit_lower)) {
    3600
  } else if (grepl("^days since", unit_lower)) {
    86400
  } else if (grepl("^seconds since", unit_lower)) {
    1
  } else {
    stop(sprintf("Unsupported time unit '%s' in %s", unit_att, ncfile))
  }

  as.Date(origin + time_vals * multiplier, tz = "UTC")
}

assign_nc_georef <- function(r, ncfile) {
  nc <- nc_open(ncfile)
  on.exit(nc_close(nc))

  lon <- ncvar_get(nc, "lon")
  lat <- ncvar_get(nc, "lat")

  dlon <- median(abs(diff(lon)))
  dlat <- median(abs(diff(lat)))

  ext(r) <- ext(
    min(lon) - dlon / 2,
    max(lon) + dlon / 2,
    min(lat) - dlat / 2,
    max(lat) + dlat / 2
  )
  crs(r) <- "OGC:CRS84"
  r
}

select_jja_common_dates <- function(r_u, r_o, years = 1991:2023) {
  t_u <- as.Date(time(r_u))
  t_o <- as.Date(time(r_o))

  idx_u <- which(format(t_u, "%Y") %in% years & format(t_u, "%m") %in% c("06", "07", "08"))
  idx_o <- which(format(t_o, "%Y") %in% years & format(t_o, "%m") %in% c("06", "07", "08"))

  r_u <- r_u[[idx_u]]
  r_o <- r_o[[idx_o]]
  t_u <- t_u[idx_u]
  t_o <- t_o[idx_o]

  common_dates <- sort(intersect(t_u, t_o))
  if (length(common_dates) == 0) {
    stop("No common JJA dates found between u850 and OLR.")
  }

  r_u <- r_u[[match(common_dates, t_u)]]
  r_o <- r_o[[match(common_dates, t_o)]]
  time(r_u) <- common_dates
  time(r_o) <- common_dates

  list(u = r_u, o = r_o, dates = common_dates)
}

make_mmdd_anomaly <- function(r, dates) {
  mmdd <- format(dates, "%m-%d")
  mmdd_unique <- unique(mmdd)

  clim_list <- vector("list", length(mmdd_unique))
  names(clim_list) <- mmdd_unique

  for (key in mmdd_unique) {
    ii <- which(mmdd == key)
    clim_list[[key]] <- mean(r[[ii]], na.rm = TRUE)
  }

  anom_layers <- vector("list", nlyr(r))
  for (i in seq_len(nlyr(r))) {
    anom_layers[[i]] <- r[[i]] - clim_list[[mmdd[i]]]
  }

  anom <- rast(anom_layers)
  time(anom) <- dates
  anom
}

safe_scale_block <- function(x) {
  center <- colMeans(x)
  scale_vals <- apply(x, 2, sd)
  keep <- is.finite(scale_vals) & scale_vals > 0

  x_keep <- x[, keep, drop = FALSE]
  center_keep <- center[keep]
  scale_keep <- scale_vals[keep]

  x_scaled <- sweep(x_keep, 2, center_keep, "-")
  x_scaled <- sweep(x_scaled, 2, scale_keep, "/")

  list(
    x_scaled = x_scaled,
    center = center_keep,
    scale = scale_keep,
    keep = keep
  )
}

recover_node_raster <- function(code_scaled, template, used_cells, center, scale, weight_rast) {
  code_weighted <- code_scaled * scale + center
  full_values <- rep(NA_real_, ncell(template))
  full_values[used_cells] <- code_weighted
  weighted_rast <- setValues(template, full_values)
  weighted_rast / weight_rast
}

plot_stack_grid <- function(stack_obj, filename, title_prefix, xdim, ydim) {
  rng <- max(
    abs(global(stack_obj, "max", na.rm = TRUE)[, 1]),
    abs(global(stack_obj, "min", na.rm = TRUE)[, 1]),
    na.rm = TRUE
  )

  png(filename, width = 1600, height = 1600, res = 160)
  par(mfrow = c(ydim, xdim), mar = c(3, 3, 3, 5))
  for (k in seq_len(nlyr(stack_obj))) {
    plot(
      stack_obj[[k]],
      main = paste(title_prefix, k),
      zlim = c(-rng, rng)
    )
  }
  dev.off()
}

stack_to_long_csv <- function(stack_obj, filename, value_name) {
  cell_xy <- as.data.frame(xyFromCell(stack_obj[[1]], seq_len(ncell(stack_obj[[1]]))))
  names(cell_xy) <- c("lon", "lat")

  out <- vector("list", nlyr(stack_obj))
  for (k in seq_len(nlyr(stack_obj))) {
    out[[k]] <- data.frame(
      node = k,
      cell_id = seq_len(ncell(stack_obj[[k]])),
      lat = cell_xy$lat,
      lon = cell_xy$lon,
      value = values(stack_obj[[k]], mat = FALSE)
    )
  }

  out_df <- do.call(rbind, out)
  names(out_df)[names(out_df) == "value"] <- value_name
  write.csv(out_df, filename, row.names = FALSE)
}

repo_root <- find_repo_root(get_script_dir())
processed_dir <- file.path(repo_root, "data", "processed")
outdir <- file.path(repo_root, "outputs", "som_u850_olr")

if (!dir.exists(outdir)) {
  dir.create(outdir, recursive = TRUE)
}

set.seed(123)

xdim <- 3
ydim <- 3
rlen <- 200
alpha_vals <- c(0.05, 0.01)
n_nodes <- xdim * ydim

u_file <- file.path(processed_dir, "uwnd_z850_jja_1991_2023.nc")
olr_file <- file.path(processed_dir, "olr_jja_1991_2023.nc")

u_raw <- rast(u_file)
olr_raw <- rast(olr_file)
u_raw <- assign_nc_georef(u_raw, u_file)
olr_raw <- assign_nc_georef(olr_raw, olr_file)
time(u_raw) <- read_nc_dates(u_file)
time(olr_raw) <- read_nc_dates(olr_file)

u_raw <- crop(u_raw, ext(100, 180, 0, 60), snap = "out")
olr_raw <- crop(olr_raw, ext(100, 180, 0, 60), snap = "out")

selected <- select_jja_common_dates(u_raw, olr_raw)
u_jja <- selected$u
olr_jja <- selected$o
dates <- selected$dates

years <- as.integer(format(dates, "%Y"))
months <- as.integer(format(dates, "%m"))
days <- as.integer(format(dates, "%d"))

cat("Aligned daily JJA samples:", length(dates), "\n")

u_anom <- make_mmdd_anomaly(u_jja, dates)
conv <- -olr_jja
time(conv) <- dates

template <- u_anom[[1]]
lat_rast <- init(template, "y")
weight_rast <- sqrt(cos(pi * lat_rast / 180))

u_weighted <- u_anom * weight_rast
conv_weighted <- conv * weight_rast

x_u0 <- t(values(u_weighted))
x_c0 <- t(values(conv_weighted))

keep_col_u <- colSums(is.na(x_u0)) == 0
keep_col_c <- colSums(is.na(x_c0)) == 0
x_u0 <- x_u0[, keep_col_u, drop = FALSE]
x_c0 <- x_c0[, keep_col_c, drop = FALSE]

keep_row <- complete.cases(x_u0) & complete.cases(x_c0)
x_u <- x_u0[keep_row, , drop = FALSE]
x_c <- x_c0[keep_row, , drop = FALSE]

dates_use <- dates[keep_row]
years_use <- years[keep_row]
months_use <- months[keep_row]
days_use <- days[keep_row]

scale_u <- safe_scale_block(x_u)
scale_c <- safe_scale_block(x_c)
x_all <- cbind(scale_u$x_scaled, scale_c$x_scaled)

cat("Final SOM matrix dimension:", nrow(x_all), "samples x", ncol(x_all), "features\n")

grid_3x3 <- kohonen::somgrid(xdim = xdim, ydim = ydim, topo = "rectangular")
som_model <- kohonen::som(
  X = x_all,
  grid = grid_3x3,
  rlen = rlen,
  alpha = alpha_vals,
  keep.data = TRUE
)

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

bmu <- som_model$unit.classif
assign_df <- data.frame(
  date = dates_use,
  year = years_use,
  month = months_use,
  day = days_use,
  node = bmu
)
write.csv(assign_df, file.path(outdir, "som_daily_assignment.csv"), row.names = FALSE)

used_cells_u <- which(keep_col_u)[scale_u$keep]
used_cells_c <- which(keep_col_c)[scale_c$keep]
n_u <- ncol(scale_u$x_scaled)
n_c <- ncol(scale_c$x_scaled)

u_code_maps <- vector("list", n_nodes)
conv_code_maps <- vector("list", n_nodes)
u_comp_maps <- vector("list", n_nodes)
conv_comp_maps <- vector("list", n_nodes)
node_counts <- integer(n_nodes)

for (k in seq_len(n_nodes)) {
  code_u_scaled <- som_model$codes[[1]][k, seq_len(n_u)]
  code_c_scaled <- som_model$codes[[1]][k, n_u + seq_len(n_c)]

  u_code_maps[[k]] <- recover_node_raster(
    code_scaled = code_u_scaled,
    template = template,
    used_cells = used_cells_u,
    center = scale_u$center,
    scale = scale_u$scale,
    weight_rast = weight_rast
  )

  conv_code_maps[[k]] <- recover_node_raster(
    code_scaled = code_c_scaled,
    template = template,
    used_cells = used_cells_c,
    center = scale_c$center,
    scale = scale_c$scale,
    weight_rast = weight_rast
  )

  ii <- which(bmu == k)
  node_counts[k] <- length(ii)

  if (length(ii) > 0) {
    u_comp_maps[[k]] <- mean(u_anom[[keep_row]][[ii]], na.rm = TRUE)
    conv_comp_maps[[k]] <- mean(conv[[keep_row]][[ii]], na.rm = TRUE)
  } else {
    u_comp_maps[[k]] <- setValues(template, rep(NA_real_, ncell(template)))
    conv_comp_maps[[k]] <- setValues(template, rep(NA_real_, ncell(template)))
  }
}

u_code_stack <- rast(u_code_maps)
conv_code_stack <- rast(conv_code_maps)
u_comp_stack <- rast(u_comp_maps)
conv_comp_stack <- rast(conv_comp_maps)

names(u_code_stack) <- paste0("Node_", seq_len(n_nodes))
names(conv_code_stack) <- paste0("Node_", seq_len(n_nodes))
names(u_comp_stack) <- paste0("Node_", seq_len(n_nodes))
names(conv_comp_stack) <- paste0("Node_", seq_len(n_nodes))

writeCDF(
  u_code_stack,
  file.path(outdir, "som_u850_codebook_maps.nc"),
  overwrite = TRUE,
  varname = "u850_codebook",
  longname = "u850 SOM codebook anomaly",
  unit = "m/s"
)
writeCDF(
  conv_code_stack,
  file.path(outdir, "som_conv_codebook_maps.nc"),
  overwrite = TRUE,
  varname = "conv_codebook",
  longname = "Convection proxy SOM codebook anomaly",
  unit = "W m-2"
)
writeCDF(
  u_comp_stack,
  file.path(outdir, "som_u850_composite_maps.nc"),
  overwrite = TRUE,
  varname = "u850_composite",
  longname = "u850 SOM node composite anomaly",
  unit = "m/s"
)
writeCDF(
  conv_comp_stack,
  file.path(outdir, "som_conv_composite_maps.nc"),
  overwrite = TRUE,
  varname = "conv_composite",
  longname = "Convection proxy SOM node composite anomaly",
  unit = "W m-2"
)

stack_to_long_csv(
  u_code_stack,
  file.path(outdir, "som_u850_codebook_maps.csv"),
  "u850_codebook"
)
stack_to_long_csv(
  conv_code_stack,
  file.path(outdir, "som_conv_codebook_maps.csv"),
  "conv_codebook"
)
stack_to_long_csv(
  u_comp_stack,
  file.path(outdir, "som_u850_composite_maps.csv"),
  "u850_composite"
)
stack_to_long_csv(
  conv_comp_stack,
  file.path(outdir, "som_conv_composite_maps.csv"),
  "conv_composite"
)

plot_stack_grid(
  u_code_stack,
  file.path(outdir, "som_u850_codebook_maps.png"),
  "u850 codebook node",
  xdim,
  ydim
)
plot_stack_grid(
  conv_code_stack,
  file.path(outdir, "som_conv_codebook_maps.png"),
  "conv codebook node",
  xdim,
  ydim
)
plot_stack_grid(
  u_comp_stack,
  file.path(outdir, "som_u850_composite_maps.png"),
  "u850 composite node",
  xdim,
  ydim
)
plot_stack_grid(
  conv_comp_stack,
  file.path(outdir, "som_conv_composite_maps.png"),
  "conv composite node",
  xdim,
  ydim
)

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

running_mean <- function(z, k = 10) {
  as.numeric(stats::filter(z, rep(1 / k, k), sides = 1))
}

freq10_df <- data.frame(year = year_seq)
for (k in seq_len(n_nodes)) {
  nm <- paste0("Node_", k)
  freq10_df[[nm]] <- running_mean(freq_df[[nm]], k = 10)
}
write.csv(
  freq10_df,
  file.path(outdir, "som_occurrence_frequency_10yr_running_mean.csv"),
  row.names = FALSE
)

png(file.path(outdir, "som_occurrence_frequency_timeseries.png"),
    width = 1800, height = 1800, res = 170)
par(mfrow = c(ydim, xdim), mar = c(4, 4, 3, 1))
for (k in seq_len(n_nodes)) {
  nm <- paste0("Node_", k)
  plot(
    freq_df$year,
    freq_df[[nm]],
    type = "h",
    lwd = 2,
    ylim = c(0, max(freq_df[, -1], na.rm = TRUE)),
    xlab = "Year",
    ylab = "Occurrence frequency",
    main = paste("Node", k)
  )
  lines(freq10_df$year, freq10_df[[nm]], lwd = 3)
}
dev.off()

node_mean_freq <- colMeans(freq_df[, -1], na.rm = TRUE)
node_summary <- data.frame(
  node = seq_len(n_nodes),
  count = node_counts,
  mean_occurrence_frequency = node_mean_freq
)
write.csv(node_summary, file.path(outdir, "som_node_mean_occurrence_summary.csv"), row.names = FALSE)

saveRDS(som_model, file.path(outdir, "som_model.rds"))

cat("All outputs saved in:", outdir, "\n")
