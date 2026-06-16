# =========================================================
# Render SOM node lead-lag relationships
# Input:
#   outputs/som_u850/som_daily_assignment.csv
# Outputs:
#   outputs/som_u850/som_node_lead_lag_counts_lag<N>d.csv
#   outputs/som_u850/som_node_lead_lag_probability_lag<N>d.csv
#   outputs/som_u850/som_node_lead_lag_pvalue_lag<N>d.csv
#   outputs/som_u850/som_node_lead_lag_qvalue_lag<N>d.csv
#   outputs/som_u850/som_node_lead_lag_asymmetry_pvalue_lag<N>d.csv
#   outputs/som_u850/som_node_lead_lag_asymmetry_qvalue_lag<N>d.csv
#   outputs/som_u850/som_node_lead_lag_asymmetry_lag<N>d.csv
#   outputs/som_u850/som_node_lead_lag_dominant_successor_lag<N>d.csv
#   outputs/som_u850/som_node_lead_lag_relationships_lag<N>d.png
# =========================================================

# ----------------------------
# 0. user settings
# ----------------------------
default_repo_root <- "/home/user/EASM"
default_outdir <- file.path(default_repo_root, "outputs", "som_u850")
default_lag_days <- 1L
default_xdim <- 3L
default_ydim <- 3L
default_alpha <- 0.05

# ----------------------------
# 1. command line options
# ----------------------------
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default) {
  prefix <- paste0(name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit) == 0L) {
    return(default)
  }
  sub(prefix, "", hit[[1]], fixed = TRUE)
}

outdir <- get_arg("--outdir", default_outdir)
lag_days <- as.integer(get_arg("--lag", default_lag_days))
xdim <- as.integer(get_arg("--xdim", default_xdim))
ydim <- as.integer(get_arg("--ydim", default_ydim))
alpha_sig <- as.numeric(get_arg("--alpha", default_alpha))

if (is.na(lag_days) || lag_days < 1L) {
  stop("--lag must be a positive integer")
}

if (is.na(alpha_sig) || alpha_sig <= 0 || alpha_sig >= 1) {
  stop("--alpha must be between 0 and 1")
}

# ----------------------------
# 2. input/output files
# ----------------------------
assignment_file <- file.path(outdir, "som_daily_assignment.csv")
tag <- sprintf("lag%dd", lag_days)

counts_file <- file.path(outdir, sprintf("som_node_lead_lag_counts_%s.csv", tag))
probability_file <- file.path(outdir, sprintf("som_node_lead_lag_probability_%s.csv", tag))
pvalue_file <- file.path(outdir, sprintf("som_node_lead_lag_pvalue_%s.csv", tag))
qvalue_file <- file.path(outdir, sprintf("som_node_lead_lag_qvalue_%s.csv", tag))
asymmetry_file <- file.path(outdir, sprintf("som_node_lead_lag_asymmetry_%s.csv", tag))
asymmetry_pvalue_file <- file.path(outdir, sprintf("som_node_lead_lag_asymmetry_pvalue_%s.csv", tag))
asymmetry_qvalue_file <- file.path(outdir, sprintf("som_node_lead_lag_asymmetry_qvalue_%s.csv", tag))
dominant_file <- file.path(outdir, sprintf("som_node_lead_lag_dominant_successor_%s.csv", tag))
plot_file <- file.path(outdir, sprintf("som_node_lead_lag_relationships_%s.png", tag))

if (!file.exists(assignment_file)) {
  stop("Daily assignment file not found: ", assignment_file)
}

# ----------------------------
# 3. read daily node sequence
# ----------------------------
assign_df <- read.csv(assignment_file, check.names = FALSE)

required_cols <- c("date", "node")
missing_cols <- setdiff(required_cols, names(assign_df))
if (length(missing_cols) > 0L) {
  stop("Input CSV is missing columns: ", paste(missing_cols, collapse = ", "))
}

assign_df$date <- as.Date(assign_df$date)
assign_df <- assign_df[order(assign_df$date), ]

if (anyDuplicated(assign_df$date)) {
  stop("Input CSV contains duplicated dates")
}

node_levels <- seq_len(max(assign_df$node, na.rm = TRUE))
n_nodes <- length(node_levels)
node_labels <- paste0("Node_", node_levels)

if (xdim * ydim != n_nodes) {
  stop("xdim * ydim must match the number of nodes: ", n_nodes)
}

# ----------------------------
# 4. lagged pairs
#    exact date matching prevents false transitions across JJA gaps
# ----------------------------
node_by_date <- setNames(assign_df$node, as.character(assign_df$date))
target_dates <- assign_df$date + lag_days
to_node <- unname(node_by_date[as.character(target_dates)])

valid <- !is.na(to_node)
from_node <- assign_df$node[valid]
to_node <- as.integer(to_node[valid])

if (length(from_node) == 0L) {
  stop("No valid lagged pairs found for lag_days = ", lag_days)
}

counts_all <- table(
  factor(from_node, levels = node_levels),
  factor(to_node, levels = node_levels)
)
counts <- matrix(as.numeric(counts_all),
  nrow = n_nodes,
  ncol = n_nodes,
  dimnames = list(node_labels, node_labels)
)

# Exclude self-persistence/autocorrelation from all downstream summaries.
diag(counts) <- NA_real_

row_totals <- rowSums(counts, na.rm = TRUE)
probability <- sweep(counts, 1, row_totals, "/")
probability[row_totals == 0, ] <- NA_real_
diag(probability) <- NA_real_

asymmetry <- probability - t(probability)
diag(asymmetry) <- NA_real_

# ----------------------------
# 5. statistical significance
#    transition p-values: one-sided Fisher enrichment of i -> j
#    asymmetry p-values: two-sided Fisher comparison of i -> j vs j -> i
# ----------------------------
pvalue <- matrix(NA_real_, nrow = n_nodes, ncol = n_nodes, dimnames = dimnames(counts))
qvalue <- matrix(NA_real_, nrow = n_nodes, ncol = n_nodes, dimnames = dimnames(counts))

col_totals <- colSums(counts, na.rm = TRUE)
grand_total <- sum(counts, na.rm = TRUE)

for (i in seq_len(n_nodes)) {
  for (j in seq_len(n_nodes)) {
    if (i == j || is.na(counts[i, j])) {
      next
    }

    a <- counts[i, j]
    b <- row_totals[i] - a
    c <- col_totals[j] - a
    d <- grand_total - a - b - c

    test_mat <- matrix(as.integer(c(a, b, c, d)), nrow = 2, byrow = TRUE)
    pvalue[i, j] <- fisher.test(test_mat, alternative = "greater")$p.value
  }
}

qvalue[!is.na(pvalue)] <- p.adjust(pvalue[!is.na(pvalue)], method = "BH")
significant <- qvalue <= alpha_sig
significant[is.na(significant)] <- FALSE

asymmetry_pvalue <- matrix(NA_real_, nrow = n_nodes, ncol = n_nodes, dimnames = dimnames(counts))
asymmetry_qvalue <- matrix(NA_real_, nrow = n_nodes, ncol = n_nodes, dimnames = dimnames(counts))

for (i in seq_len(n_nodes - 1L)) {
  for (j in (i + 1L):n_nodes) {
    a <- counts[i, j]
    b <- row_totals[i] - a
    c <- counts[j, i]
    d <- row_totals[j] - c

    test_mat <- matrix(as.integer(c(a, b, c, d)), nrow = 2, byrow = TRUE)
    p <- fisher.test(test_mat, alternative = "two.sided")$p.value
    asymmetry_pvalue[i, j] <- p
    asymmetry_pvalue[j, i] <- p
  }
}

upper_idx <- upper.tri(asymmetry_pvalue)
asymmetry_qvalue[upper_idx] <- p.adjust(asymmetry_pvalue[upper_idx], method = "BH")
asymmetry_qvalue[lower.tri(asymmetry_qvalue)] <- t(asymmetry_qvalue)[lower.tri(asymmetry_qvalue)]
diag(asymmetry_qvalue) <- NA_real_

asymmetry_significant <- asymmetry_qvalue <= alpha_sig
asymmetry_significant[is.na(asymmetry_significant)] <- FALSE

# ----------------------------
# 6. dominant successor by node
# ----------------------------
dominant_successor <- data.frame(
  from_node = node_labels,
  to_node = NA_character_,
  transition_probability = NA_real_,
  transition_count = NA_real_,
  transition_q_value = NA_real_,
  transition_significant = FALSE,
  row_total_without_self = row_totals
)

for (i in seq_len(n_nodes)) {
  row_prob <- probability[i, ]
  if (all(is.na(row_prob))) {
    next
  }
  j <- which.max(replace(row_prob, is.na(row_prob), -Inf))
  dominant_successor$to_node[i] <- node_labels[[j]]
  dominant_successor$transition_probability[i] <- probability[i, j]
  dominant_successor$transition_count[i] <- counts[i, j]
  dominant_successor$transition_q_value[i] <- qvalue[i, j]
  dominant_successor$transition_significant[i] <- significant[i, j]
}

# ----------------------------
# 7. write CSV outputs
# ----------------------------
matrix_to_df <- function(x) {
  data.frame(from_node = rownames(x), x, check.names = FALSE, row.names = NULL)
}

write.csv(matrix_to_df(counts), counts_file, row.names = FALSE, na = "")
write.csv(matrix_to_df(probability), probability_file, row.names = FALSE, na = "")
write.csv(matrix_to_df(pvalue), pvalue_file, row.names = FALSE, na = "")
write.csv(matrix_to_df(qvalue), qvalue_file, row.names = FALSE, na = "")
write.csv(matrix_to_df(asymmetry), asymmetry_file, row.names = FALSE, na = "")
write.csv(matrix_to_df(asymmetry_pvalue), asymmetry_pvalue_file, row.names = FALSE, na = "")
write.csv(matrix_to_df(asymmetry_qvalue), asymmetry_qvalue_file, row.names = FALSE, na = "")
write.csv(dominant_successor, dominant_file, row.names = FALSE, na = "")

# ----------------------------
# 8. plot heatmaps
# ----------------------------
significance_stars <- function(q) {
  if (is.na(q)) {
    return("")
  }
  if (q <= 0.001) {
    return("***")
  }
  if (q <= 0.01) {
    return("**")
  }
  if (q <= alpha_sig) {
    return("*")
  }
  ""
}

plot_heatmap <- function(mat, main, cols, zlim, value_fmt, q_mat = NULL) {
  n <- nrow(mat)
  image(
    x = seq_len(n),
    y = seq_len(n),
    z = t(mat[n:1, , drop = FALSE]),
    col = cols,
    zlim = zlim,
    axes = FALSE,
    xlab = "",
    ylab = "",
    main = main
  )
  axis(1, at = seq_len(n), labels = colnames(mat), las = 2, cex.axis = 0.8)
  axis(2, at = seq_len(n), labels = rev(rownames(mat)), las = 1, cex.axis = 0.8)
  mtext("Following node", side = 1, line = 4.5)
  mtext("Leading node", side = 2, line = 4.5)
  box()

  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (is.na(mat[i, j])) {
        next
      }
      cell_y <- n - i + 1
      stars <- ""
      is_sig <- FALSE
      if (!is.null(q_mat)) {
        stars <- significance_stars(q_mat[i, j])
        is_sig <- stars != ""
      }

      text(j, cell_y, paste0(sprintf(value_fmt, mat[i, j]), stars), cex = 0.65)
      if (is_sig) {
        rect(j - 0.5, cell_y - 0.5, j + 0.5, cell_y + 0.5, border = "black", lwd = 2)
      }
    }
  }
}

plot_colorbar <- function(cols, zlim, label) {
  z <- matrix(seq_along(cols), nrow = 1)
  image(
    x = 1,
    y = seq_along(cols),
    z = z,
    col = cols,
    axes = FALSE,
    xlab = "",
    ylab = ""
  )
  ticks <- seq(1, length(cols), length.out = 5)
  tick_labels <- seq(zlim[1], zlim[2], length.out = 5)
  axis(4, at = ticks, labels = sprintf("%.2f", tick_labels), las = 1, cex.axis = 0.8)
  mtext(label, side = 4, line = 2.8)
  box()
}

prob_cols <- colorRampPalette(c("#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"))(80)
asym_cols <- colorRampPalette(c("#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac"))(81)

prob_zlim <- c(0, max(probability, na.rm = TRUE))
if (!is.finite(prob_zlim[2]) || prob_zlim[2] == 0) {
  prob_zlim[2] <- 1
}

asym_lim <- max(abs(asymmetry), na.rm = TRUE)
if (!is.finite(asym_lim) || asym_lim == 0) {
  asym_lim <- 1
}
asym_zlim <- c(-asym_lim, asym_lim)

png(plot_file, width = 2400, height = 1200, res = 170)
layout(matrix(c(1, 2, 3, 4), nrow = 1), widths = c(1, 0.16, 1, 0.16))

par(mar = c(7, 7, 4, 1))
plot_heatmap(
  probability,
  sprintf("Node Transition Probability, Lag %d Day (Self Excluded, FDR q <= %.2f)", lag_days, alpha_sig),
  prob_cols,
  prob_zlim,
  "%.2f",
  qvalue
)

par(mar = c(7, 0.5, 4, 5))
plot_colorbar(prob_cols, prob_zlim, "Probability")

par(mar = c(7, 7, 4, 1))
plot_heatmap(
  asymmetry,
  sprintf("Lead-Lag Asymmetry P(i -> j) - P(j -> i), Lag %d Day (FDR q <= %.2f)", lag_days, alpha_sig),
  asym_cols,
  asym_zlim,
  "%+.2f",
  asymmetry_qvalue
)

par(mar = c(7, 0.5, 4, 5))
plot_colorbar(asym_cols, asym_zlim, "Asymmetry")

dev.off()

cat("Valid lagged pairs:", length(from_node), "\n")
cat("FDR alpha:", alpha_sig, "\n")
cat("Significant enriched transitions:", sum(significant), "\n")
cat("Significant asymmetric directed cells:", sum(asymmetry_significant), "\n")
cat("Saved:", counts_file, "\n")
cat("Saved:", probability_file, "\n")
cat("Saved:", pvalue_file, "\n")
cat("Saved:", qvalue_file, "\n")
cat("Saved:", asymmetry_file, "\n")
cat("Saved:", asymmetry_pvalue_file, "\n")
cat("Saved:", asymmetry_qvalue_file, "\n")
cat("Saved:", dominant_file, "\n")
cat("Saved:", plot_file, "\n")
