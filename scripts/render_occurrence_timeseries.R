# =========================================================
# Render SOM occurrence frequency timeseries
# Input:
#   outputs/som_u850/som_occurrence_frequency_by_year.csv
# Outputs:
#   outputs/som_u850/som_occurrence_frequency_<N>yr_running_mean.csv
#   outputs/som_u850/som_occurrence_frequency_timeseries.png
# =========================================================

# ----------------------------
# 0. user settings
# ----------------------------
default_repo_root <- "/home/user/EASM"
default_outdir <- file.path(default_repo_root, "outputs", "som_u850")
default_window <- 3L
default_xdim <- 3L
default_ydim <- 3L

# ----------------------------
# 1. command line options
# ----------------------------
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default) {
  prefix <- paste0(name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit) == 0) {
    return(default)
  }
  sub(prefix, "", hit[[1]], fixed = TRUE)
}

outdir <- get_arg("--outdir", default_outdir)
running_window <- as.integer(get_arg("--window", default_window))
xdim <- as.integer(get_arg("--xdim", default_xdim))
ydim <- as.integer(get_arg("--ydim", default_ydim))

if (is.na(running_window) || running_window < 1L) {
  stop("--window must be a positive integer")
}

# ----------------------------
# 2. input/output files
# ----------------------------
freq_file <- file.path(outdir, "som_occurrence_frequency_by_year.csv")
running_mean_file <- file.path(
  outdir,
  sprintf("som_occurrence_frequency_%dyr_running_mean.csv", running_window)
)
plot_file <- file.path(outdir, "som_occurrence_frequency_timeseries.png")

if (!file.exists(freq_file)) {
  stop("Occurrence frequency file not found: ", freq_file)
}

# ----------------------------
# 3. read occurrence frequency
# ----------------------------
freq_df <- read.csv(freq_file, check.names = FALSE)

if (!"year" %in% names(freq_df)) {
  stop("Input CSV must contain a 'year' column")
}

node_names <- grep("^Node_[0-9]+$", names(freq_df), value = TRUE)
n_nodes <- length(node_names)

if (n_nodes == 0L) {
  stop("Input CSV must contain Node_<n> columns")
}

if (xdim * ydim != n_nodes) {
  stop("xdim * ydim must match the number of node columns: ", n_nodes)
}

# ----------------------------
# 4. centered running mean
# ----------------------------
running_mean <- function(z, k) {
  as.numeric(stats::filter(z, rep(1 / k, k), sides = 2))
}

freq_running_df <- data.frame(year = freq_df$year)

for (nm in node_names) {
  freq_running_df[[nm]] <- running_mean(freq_df[[nm]], running_window)
}

write.csv(freq_running_df, running_mean_file, row.names = FALSE)

# ----------------------------
# 5. plot annual occurrence frequency
# ----------------------------
png(plot_file, width = 1800, height = 1800, res = 170)

par(mfrow = c(ydim, xdim), mar = c(4, 4, 3, 1))

ymax <- max(freq_df[, node_names], na.rm = TRUE)

for (i in seq_along(node_names)) {
  nm <- node_names[[i]]

  yr <- freq_df$year
  y1 <- freq_df[[nm]]
  y2 <- freq_running_df[[nm]]

  plot(yr, y1,
    type = "h",
    lwd = 2,
    ylim = c(0, ymax),
    xlab = "Year",
    ylab = "Occurrence frequency",
    main = paste("Node", i)
  )
  lines(yr, y2, lwd = 3)
}

dev.off()

cat("Saved:", running_mean_file, "\n")
cat("Saved:", plot_file, "\n")
