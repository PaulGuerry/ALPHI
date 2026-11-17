#!/usr/bin/env Rscript
# TEST 1 figures (repeat with compositions C1-C3): how often pairD_phenome.py recovers the
# true number of subgroups, for each distance (OB, Jaccard, Resnik), as a function of
# cohort size N, signature size s (distinguishing symptoms/signs) and term-substitution
# probability p.
#
# Scenario names: the CSVs (and generate_test1_patients.py) use D1S0 / D2S2 / D3S3; figures,
# output files and the summary use D1S0 / D1S2 / D1S3 (one disease, 0 / 2 / 3 latent subgroups).
#
# One figure per scenario with subgroups (D1S2, D1S3):
#   panels   rows = substitution p (0.5 at the top), columns = signature size s
#   x        cohort size N (log scale)
#   y        runs recovering the true number of subgroups (%)
#   line     mean over the compositions given (C1-C3)
#   band     range (min-max) over the compositions
# The D1S0 (no subgroups) spurious-split counts are given in the caption.
# By default the two layouts (dispersed, concentrated) are pooled with equal weight.
#
# Usage (from ~/pairD_eval/test1/min_dist_0; with no files given, C*/test1_grid_iterations.csv
# are used):
#   cd ~/pairD_eval/test1/min_dist_0 && Rscript ../../scripts/plot_test1_distances.R -o ../../figures/Fig_test1
#   Rscript plot_test1_distances.R C1/test1_grid_iterations.csv C2/test1_grid_iterations.csv \
#           C3/test1_grid_iterations.csv -o Fig_test1
#   options:
#     -o, --output PREFIX   writes PREFIX_D1S2.png, PREFIX_D1S3.png (annotated),
#                           PREFIX_D1S2_article.png/.pdf, PREFIX_D1S3_article.png/.pdf (no title,
#                           subtitle or caption; 180 mm wide, 600 dpi PNG and vector PDF) and
#                           PREFIX_summary.csv; the caption text is printed for the figure legend
#     --layout L            pooled (default), dispersed, concentrated, or both (one set of
#                           figures per layout, suffix _dispersed / _concentrated)
#     --contrast C          contrast to plot (default strong)
#     --reporting R         reporting rate to plot (default 1)
#     --dpi D               default 300
#
# The composition is read from the 'composition' column of each file. Rows for distances
# other than dphenome/jaccard/resnik (e.g. leftover hamming rows) are ignored.
#
# Requires: ggplot2 (>= 3.4)

suppressPackageStartupMessages(library(ggplot2))

# legend labels: the method with each distance (ALPHI-OB etc.), not the distance alone
DIST_LABELS  <- c(dphenome = "ALPHI-OB", jaccard = "ALPHI-Jaccard", resnik = "ALPHI-Resnik")
DIST_ORDER   <- c("ALPHI-OB", "ALPHI-Jaccard", "ALPHI-Resnik")
DIST_COLOURS <- c(`ALPHI-OB` = "#2a78d6", `ALPHI-Jaccard` = "#eb6834", `ALPHI-Resnik` = "#1baf7a")
DIST_SHAPES  <- c(`ALPHI-OB` = 21, `ALPHI-Jaccard` = 22, `ALPHI-Resnik` = 24)      # second encoding besides colour
DISEASE_NAMES  <- c(D1S0 = "D1S0", D2S2 = "D1S2", D3S3 = "D1S3")   # CSV name -> name in the article
DISEASE_TITLES <- c(D1S2 = "One disease, two latent subgroups (D1S2)",
                    D1S3 = "One disease, three latent subgroups (D1S3)")
N_BREAKS <- c(30, 90, 180, 360)
INK <- "#1f1f1e"; MUTED <- "#6b6b67"; GRID <- "#e6e5e1"
KEEP <- c("composition", "disease", "n_patients", "sig_size", "contrast", "reporting", "layout",
          "substitution", "distance", "iteration", "n_clusters", "k_correct")

# ---- command line ------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
prefix <- "Fig_test1"; layout_opt <- "pooled"; contrast <- "strong"; reporting <- 1; dpi <- 300
inputs <- character(0)
i <- 1
while (i <= length(args)) {
  a <- args[i]
  if (a %in% c("-o", "--output")) { prefix <- args[i + 1]; i <- i + 2 }
  else if (a == "--layout")    { layout_opt <- args[i + 1]; i <- i + 2 }
  else if (a == "--contrast")  { contrast <- args[i + 1]; i <- i + 2 }
  else if (a == "--reporting") { reporting <- as.numeric(args[i + 1]); i <- i + 2 }
  else if (a == "--dpi")       { dpi <- as.numeric(args[i + 1]); i <- i + 2 }
  else { inputs <- c(inputs, a); i <- i + 1 }
}
if (length(inputs) == 0) {
  # default: the composition folders next to this script's working directory (~/test1)
  inputs <- Sys.glob(file.path("C*", "test1_grid_iterations.csv"))
  if (length(inputs) == 0)
    stop("No input given and no C*/test1_grid_iterations.csv found here. ",
         "Run from ~/pairD_eval/test1/min_dist_0 or give the files: Rscript plot_test1_distances.R C1/... C2/... C3/...")
  cat("Using", paste(inputs, collapse = ", "), "\n")
}
if (!layout_opt %in% c("pooled", "dispersed", "concentrated", "both"))
  stop("--layout must be pooled, dispersed, concentrated or both.")
prefix <- sub("\\.png$", "", prefix)

# ---- data --------------------------------------------------------------------------
load_one <- function(path) {
  x <- read.csv(path, stringsAsFactors = FALSE)
  missing <- setdiff(KEEP, names(x))
  if (length(missing) > 0) stop(path, " lacks column(s): ", paste(missing, collapse = ", "))
  x[, KEEP]
}
d <- do.call(rbind, lapply(inputs, load_one))
d <- d[d$distance %in% names(DIST_LABELS) & d$contrast == contrast &
         abs(d$reporting - reporting) < 1e-9, ]
if (nrow(d) == 0) stop("No rows left for contrast '", contrast, "' and reporting ", reporting, ".")

# keep the last entry of each (grid point, iteration), in case of --resume reruns
key <- do.call(paste, c(d[, setdiff(KEEP, c("n_clusters", "k_correct"))], sep = "|"))
d <- d[!duplicated(key, fromLast = TRUE), ]

d$dist <- factor(DIST_LABELS[d$distance], levels = DIST_ORDER)
d$disease <- ifelse(d$disease %in% names(DISEASE_NAMES), DISEASE_NAMES[d$disease], d$disease)
d$correct <- as.numeric(d$k_correct == 1)
comps <- sort(unique(d$composition))
cat(sprintf("Loaded %d runs; compositions %s; layouts %s\n", nrow(d),
            paste(comps, collapse = ", "), paste(sort(unique(d$layout)), collapse = ", ")))

# ---- one set of figures ------------------------------------------------------------
make_figures <- function(dd, layout_label, out_prefix) {
  # rate per composition x layout x cell, then layouts pooled with equal weight
  cell <- aggregate(correct ~ composition + disease + layout + sig_size + substitution +
                      n_patients + dist, data = dd, FUN = mean)
  comp <- aggregate(correct ~ composition + disease + sig_size + substitution + n_patients + dist,
                    data = cell, FUN = mean)
  s <- aggregate(correct ~ disease + sig_size + substitution + n_patients + dist, data = comp,
                 FUN = function(v) c(mean = mean(v), lo = min(v), hi = max(v), n = length(v)))
  s <- do.call(data.frame, s)
  names(s) <- sub("^correct\\.", "", names(s))
  s$pct <- 100 * s$mean; s$lo <- 100 * s$lo; s$hi <- 100 * s$hi

  # D1S0 spurious splits per distance
  null <- dd[dd$disease == "D1S0", ]
  if (nrow(null) > 0) {
    null$split <- as.numeric(null$n_clusters > 1)
    null$runs <- 1
    fp <- aggregate(cbind(split, runs) ~ dist, data = null, FUN = sum)
    fp_text <- paste0("One disease, no latent subgroups (D1S0), runs with a spurious split: ",
                      paste(sprintf("%s %d/%d", fp$dist, fp$split, fp$runs), collapse = ", "), ".")
  } else {
    fp_text <- "No D1S0 runs in the input."
  }

  n_comp <- length(unique(dd$composition))
  sub_text <- paste0(
    if (n_comp > 1) sprintf("Line: mean over compositions %s; band: range over compositions. ",
                            paste(sort(unique(dd$composition)), collapse = ", "))
    else sprintf("Composition %s. ", unique(dd$composition)),
    "Layouts: ", layout_label, ".")

  write.csv(s[order(s$disease, s$substitution, s$sig_size, s$n_patients, s$dist),
              c("disease", "sig_size", "substitution", "n_patients", "dist", "pct", "lo", "hi", "n")],
            paste0(out_prefix, "_summary.csv"), row.names = FALSE)

  k_lab <- function(x) paste("s =", x)          # s: number of distinguishing symptoms/signs
  p_lab <- function(x) ifelse(as.numeric(x) == 0, "p = 0 (exact terms)", paste("p =", x))

  for (dis in intersect(c("D1S2", "D1S3"), unique(s$disease))) {
    sd <- s[s$disease == dis, ]
    # rows: highest substitution at the top (p = 0.5, 0.25, 0)
    sd$substitution <- factor(sd$substitution, levels = sort(unique(sd$substitution), decreasing = TRUE))
    g <- ggplot(sd, aes(n_patients, pct, colour = dist, fill = dist, group = dist))
    if (n_comp > 1)
      g <- g + geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.13, colour = NA)
    g <- g +
      geom_line(linewidth = 0.7) +
      geom_point(aes(shape = dist), size = 2.1, colour = "white", stroke = 0.5) +
      facet_grid(substitution ~ sig_size,
                 labeller = labeller(sig_size = as_labeller(k_lab),
                                     substitution = as_labeller(p_lab))) +
      scale_colour_manual(values = DIST_COLOURS, name = NULL, drop = FALSE) +
      scale_fill_manual(values = DIST_COLOURS, name = NULL, drop = FALSE) +
      scale_shape_manual(values = DIST_SHAPES, name = NULL, drop = FALSE) +
      scale_x_continuous(trans = "log2", breaks = N_BREAKS) +
      scale_y_continuous(limits = c(-4, 104), breaks = c(0, 50, 100), expand = c(0, 0)) +
      labs(title = DISEASE_TITLES[[dis]], subtitle = sub_text, caption = fp_text,
           x = "Cohort size N (patients)",
           y = "Runs recovering the true number of subgroups (%)") +
      theme_minimal(base_size = 9) +
      theme(panel.grid.major = element_line(colour = GRID, linewidth = 0.4),
            panel.grid.minor = element_blank(),
            panel.spacing = unit(0.8, "lines"),
            axis.line = element_line(colour = MUTED, linewidth = 0.4),
            axis.ticks = element_line(colour = MUTED, linewidth = 0.4),
            axis.text = element_text(colour = MUTED),
            axis.title = element_text(colour = INK),
            strip.text = element_text(colour = INK, size = 9),
            plot.title = element_text(colour = INK, size = 10.5, face = "bold"),
            plot.subtitle = element_text(colour = MUTED, size = 8),
            plot.caption = element_text(colour = MUTED, size = 7.5, hjust = 0),
            plot.title.position = "plot", plot.caption.position = "plot",
            legend.position = "top", legend.justification = "left",
            legend.text = element_text(size = 8.5, colour = INK),
            legend.margin = margin(0, 0, 0, 0),
            plot.background = element_rect(fill = "white", colour = NA))
    out <- paste0(out_prefix, "_", dis, ".png")
    ggsave(out, g, width = 7.2, height = 6.6, dpi = dpi, bg = "white")
    cat("Wrote", out, "\n")

    # article version: no title, subtitle or caption (they go in the figure legend);
    # 180 mm wide (double column), as PNG (600 dpi) and PDF (vector)
    art <- g + labs(title = NULL, subtitle = NULL, caption = NULL) +
      theme(legend.justification = "center", plot.margin = margin(4, 6, 4, 4))
    art_base <- paste0(out_prefix, "_", dis, "_article")
    ggsave(paste0(art_base, ".png"), art, width = 180, height = 160, units = "mm", dpi = 600, bg = "white")
    ggsave(paste0(art_base, ".pdf"), art, width = 180, height = 160, units = "mm", bg = "white")
    cat("Wrote", paste0(art_base, ".png"), "and", paste0(art_base, ".pdf"), "\n")
  }
  cat("Wrote", paste0(out_prefix, "_summary.csv"), "\n")
  cat("\nFor the figure legend:\n", sub_text, " ", fp_text, "\n\n", sep = "")

  # console check: mean correct-k (%) per disease x p x distance, over N, k and compositions
  chk <- aggregate(pct ~ disease + substitution + dist, data = s, FUN = mean)
  chk$pct <- round(chk$pct)
  print(reshape(chk, idvar = c("disease", "substitution"), timevar = "dist", direction = "wide"),
        row.names = FALSE)
}

if (layout_opt == "pooled") {
  make_figures(d, "dispersed and concentrated pooled", prefix)
} else {
  lays <- if (layout_opt == "both") c("dispersed", "concentrated") else layout_opt
  for (lay in lays) {
    dl <- d[d$layout == lay, ]
    if (nrow(dl) == 0) { cat("No rows for layout", lay, "\n"); next }
    make_figures(dl, lay, paste0(prefix, "_", lay))
  }
}
