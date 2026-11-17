# Shared settings for the MYO5B figures (sourced by fig_myo5b_*.R).

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

INK <- "#1f1f1e"; MUTED <- "#6b6b67"; GRID <- "#e6e5e1"; ABSENT <- "#d9d8d3"
CLUSTER_COLOURS <- c("MVID" = "#d9822b", "PFIC" = "#2e8b6a", "MVID+PFIC" = "#7b52b9", "Noise" = "#b9b8b3")
CLUSTER_ORDER <- c("MVID", "MVID+PFIC", "PFIC", "Noise")

# colours for any cluster names (e.g. "MVID a", "Cluster 3" in the SI runs)
cluster_colours <- function(names) {
  names <- unique(names)
  base <- sub(" [a-f]$", "", names)
  col <- CLUSTER_COLOURS[base]
  extra <- c("#3f7fd0", "#c4495a", "#8a8a2e", "#5aa0b0")
  col[is.na(col)] <- extra[seq_len(sum(is.na(col)))]
  setNames(unname(col), names)
}
cluster_levels <- function(names) {
  names <- unique(names)
  base <- match(sub(" [a-f]$", "", names), CLUSTER_ORDER)
  names[order(ifelse(is.na(base), 99, base), names)]
}

theme_fig <- function(base_size = 8) {
  theme_minimal(base_size = base_size) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(colour = GRID, linewidth = 0.3),
          axis.text = element_text(colour = MUTED),
          axis.title = element_text(colour = INK),
          strip.text = element_text(colour = INK),
          plot.title = element_text(colour = INK, size = base_size + 1, face = "bold"),
          plot.tag = element_text(colour = INK, face = "bold", size = base_size + 2),
          legend.text = element_text(colour = INK),
          legend.title = element_text(colour = INK),
          plot.background = element_rect(fill = "white", colour = NA))
}

parse_args <- function(defaults) {
  args <- commandArgs(trailingOnly = TRUE)
  out <- defaults
  i <- 1; pos <- character(0)
  while (i <= length(args)) {
    a <- args[i]
    if (a %in% c("-o", "--output")) { out$output <- args[i + 1]; i <- i + 2 }
    else if (grepl("^--", a)) { out[[sub("^--", "", a)]] <- args[i + 1]; i <- i + 2 }
    else { pos <- c(pos, a); i <- i + 1 }
  }
  if (length(pos) > 0) out$prefix <- pos[1]
  if (is.null(out$prefix)) stop("Give the data prefix, e.g. figdata/ob_nearest")
  if (is.null(out$output)) out$output <- paste0(out$prefix, "_", out$suffix)
  out
}

save_fig <- function(g, output, width_mm, height_mm) {
  ggsave(paste0(output, ".png"), g, width = width_mm, height = height_mm, units = "mm", dpi = 600, bg = "white")
  ggsave(paste0(output, ".pdf"), g, width = width_mm, height = height_mm, units = "mm", bg = "white")
  cat("Wrote", paste0(output, ".png"), "and", paste0(output, ".pdf"), "\n")
}
