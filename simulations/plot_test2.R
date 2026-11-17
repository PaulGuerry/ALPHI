#!/usr/bin/env Rscript
# TEST 2 figure: number of patients pairD_phenome.py needs to separate real diseases
# (smallest N at which the true number of clusters is found in >= 80 % of runs), per
# scenario and version (ALPHI-OB, ALPHI-Jaccard, ALPHI-Resnik).
#
#   rows     reporting rate r (1 = all symptoms reported, 0.5 = half)
#   columns  term-substitution probability p
#   y        scenario (pairs 2:1, both ways round; triple 3:2:1, three rotations)
#   x        N needed (log scale); open symbols at the right edge = not reached on the ladder
#   grey     sizes at which the smallest group is too small to form a cluster (below the
#            lowest N needed anywhere for that kind of scenario)
# The caption gives the spurious splits of the single-disease controls (from the
# iterations file).
#
# Usage (in ~/pairD_eval/test2/min_dist_0/pairs_triple, whose iterations file also holds the
# controls):
#   Rscript ../../../scripts/plot_test2.R -o ../../../figures/Fig_test2_n_needed.png \
#       --note "OB: 5 of the 6 split Wilson cohorts are flagged by the few-feature check."
#   Rscript plot_test2.R test2_n_needed.csv --iterations test2_iterations.csv -o Fig_test2_n_needed.png
#   --note TEXT   appended to the caption (e.g. the few-feature flag result, not in the CSVs)
# Writes the annotated figure (-o) and an article version without title, subtitle or
# caption: <name>_article.png (600 dpi) and <name>_article.pdf, 180 mm wide; the caption
# text is printed for the figure legend.
#
# Requires: ggplot2 (>= 3.4)

suppressPackageStartupMessages(library(ggplot2))

# legend labels: the method with each distance (ALPHI-OB etc.), not the distance alone
DIST_LABELS  <- c(dphenome = "ALPHI-OB", jaccard = "ALPHI-Jaccard", resnik = "ALPHI-Resnik")
DIST_ORDER   <- c("ALPHI-OB", "ALPHI-Jaccard", "ALPHI-Resnik")
DIST_COLOURS <- c(`ALPHI-OB` = "#2a78d6", `ALPHI-Jaccard` = "#eb6834", `ALPHI-Resnik` = "#1baf7a")
DIST_SHAPES  <- c(`ALPHI-OB` = 21, `ALPHI-Jaccard` = 22, `ALPHI-Resnik` = 24)
DIST_OFFSET  <- c(`ALPHI-OB` = 0.25, `ALPHI-Jaccard` = 0, `ALPHI-Resnik` = -0.25)   # vertical position within a scenario row
LADDER <- c(10, 20, 30, 45, 60, 90, 120, 180)
INK <- "#1f1f1e"; MUTED <- "#6b6b67"; GRID <- "#e6e5e1"; FLOOR_FILL <- "#efeeea"

# ---- command line ------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
nn_file <- "test2_n_needed.csv"; it_file <- "test2_iterations.csv"
output <- "Fig_test2_n_needed.png"; dpi <- 300; note <- ""
i <- 1; pos <- character(0)
while (i <= length(args)) {
  a <- args[i]
  if (a %in% c("-o", "--output")) { output <- args[i + 1]; i <- i + 2 }
  else if (a == "--iterations") { it_file <- args[i + 1]; i <- i + 2 }
  else if (a == "--dpi") { dpi <- as.numeric(args[i + 1]); i <- i + 2 }
  else if (a == "--note") { note <- args[i + 1]; i <- i + 2 }
  else { pos <- c(pos, a); i <- i + 1 }
}
if (length(pos) > 0) nn_file <- pos[1]
if (!file.exists(nn_file)) stop(nn_file, " not found (run from the TEST 2 results folder or give the file).")

# ---- data --------------------------------------------------------------------------
s <- read.csv(nn_file, stringsAsFactors = FALSE, colClasses = c(proportions = "character"))
s <- s[s$distance %in% names(DIST_LABELS), ]
s$dist <- factor(DIST_LABELS[s$distance], levels = DIST_ORDER)
s$reached <- as.character(s$reached) %in% c("True", "TRUE", "true")
ladder_max <- max(c(s$ladder_max, LADDER), na.rm = TRUE)
x_out <- ladder_max * 1.3                     # where "not reached" is drawn
x_hi <- if (any(!s$reached)) x_out * 1.12 else ladder_max * 1.2   # right edge of the panels
s$x <- ifelse(s$reached, as.numeric(s$n_needed), x_out)

cap <- function(x) paste0(toupper(substring(x, 1, 1)), substring(x, 2))
scen_label <- function(d, p) {
  dd <- cap(strsplit(d, "+", fixed = TRUE)[[1]])
  pp <- strsplit(p, ":", fixed = TRUE)[[1]]
  paste0(paste(substring(dd, 1, 1), collapse = ":"), ", ", paste(pp, collapse = ":"))   # e.g. M:W, 2:1
}

# scenario rows, in file order (pairs, then triples), top to bottom, with a gap between kinds
scen <- unique(s$scenario)
kind <- s$kind[match(scen, s$scenario)]
idx <- seq_along(scen) + cumsum(c(0, diff(as.integer(factor(kind, levels = unique(kind)))) != 0)) * 0.6
ypos <- setNames(max(idx) + 1 - idx, scen)
labels <- mapply(scen_label, s$diseases[match(scen, s$scenario)], s$proportions[match(scen, s$scenario)])
s$y <- ypos[s$scenario] + DIST_OFFSET[as.character(s$dist)]

r_text <- function(r) ifelse(r == 1, "r = 1 (all symptoms reported)",
                             paste0("r = ", r, " (", 100 * r, " % reported)"))
p_text <- function(p) ifelse(p == 0, "p = 0 (exact terms)", paste("p =", p))
s$r_lab <- factor(r_text(s$reporting), levels = r_text(sort(unique(s$reporting), decreasing = TRUE)))
s$p_lab <- factor(p_text(s$substitution), levels = p_text(sort(unique(s$substitution))))

# group-size floor: below the lowest N needed anywhere for that kind of scenario
floor_of <- tapply(s$x[s$reached], s$kind[s$reached], min)
prev_rung <- function(n) { below <- LADDER[LADDER < n]; if (length(below)) max(below) else n / 1.5 }
fl <- data.frame(scenario = scen, kind = kind)
fl$floor <- floor_of[fl$kind]
fl <- fl[!is.na(fl$floor), ]
fl$xmax <- sqrt(fl$floor * sapply(fl$floor, prev_rung))     # half-way (log) to the rung below
fl$ymin <- ypos[fl$scenario] - 0.45
fl$ymax <- ypos[fl$scenario] + 0.45

# caption: spurious splits in the single-disease controls
abbrev <- sort(unique(cap(unlist(strsplit(s$diseases, "+", fixed = TRUE)))))
abbrev_text <- paste0(paste(sprintf("%s, %s", substring(abbrev, 1, 1), abbrev), collapse = "; "), ".")
caption <- paste0(abbrev_text, " Grey: cohort sizes at which the smallest group (about 7 patients or fewer) ",
                  "could not form a cluster in any condition.")
if (any(!s$reached))
  caption <- paste0(caption, " Open symbols at the right edge: not reached by N = ", ladder_max, ".")
if (file.exists(it_file)) {
  it <- read.csv(it_file, stringsAsFactors = FALSE)
  ctl <- it[it$kind == "control" & it$distance %in% names(DIST_LABELS), ]
  if (nrow(ctl) > 0) {
    ctl$dist <- factor(DIST_LABELS[ctl$distance], levels = DIST_ORDER)
    ctl$split <- as.numeric(ctl$n_clusters > 1)
    ctl$runs <- 1
    fp <- aggregate(cbind(split, runs) ~ dist, data = ctl, FUN = sum)
    caption <- paste0(caption, "\nSingle-disease controls (",
                      paste(cap(unique(ctl$scenario)), collapse = ", "), " alone, N = ",
                      min(ctl$n_patients), "-", max(ctl$n_patients), "): spurious split in ",
                      paste(sprintf("%s %d/%d", fp$dist, fp$split, fp$runs), collapse = ", "), " runs.")
  }
}

if (nzchar(note)) caption <- paste0(caption, " ", note)

# ---- plot --------------------------------------------------------------------------
g <- ggplot() +
  geom_rect(data = fl, aes(xmin = 8, xmax = xmax, ymin = ymin, ymax = ymax),
            fill = FLOOR_FILL, colour = NA) +
  geom_point(data = s[s$reached, ], aes(x, y, fill = dist, shape = dist),
             size = 2.3, colour = "white", stroke = 0.4) +
  facet_grid(r_lab ~ p_lab) +
  scale_fill_manual(values = DIST_COLOURS, name = NULL, drop = FALSE) +
  scale_colour_manual(values = DIST_COLOURS, name = NULL, drop = FALSE, guide = "none") +
  scale_shape_manual(values = DIST_SHAPES, name = NULL, drop = FALSE) +
  # 120 keeps its grid line but no label: on one line it would run into 90
  scale_x_continuous(trans = "log10", breaks = LADDER, labels = ifelse(LADDER == 120, "", LADDER),
                     limits = c(8, x_hi), expand = c(0, 0)) +
  scale_y_continuous(breaks = ypos, labels = labels, expand = c(0.02, 0.02)) +
  labs(title = "Patients needed to separate distinct diseases (TEST 2)",
       subtitle = "Smallest cohort in which the true number of diseases is found in at least 80 % of runs",
       x = "Patients needed (N, log scale)", y = "Disease frequency ratios", caption = caption) +
  theme_minimal(base_size = 9) +
  theme(panel.grid.major.x = element_line(colour = GRID, linewidth = 0.4),
        panel.grid.major.y = element_line(colour = GRID, linewidth = 0.3),
        panel.grid.minor = element_blank(),
        panel.spacing = unit(0.9, "lines"),
        axis.ticks.x = element_line(colour = MUTED, linewidth = 0.4),
        axis.text = element_text(colour = MUTED),
        axis.text.y = element_text(colour = INK, size = 8),
        axis.text.x = element_text(size = 7),
        axis.title = element_text(colour = INK),
        strip.text = element_text(colour = INK, size = 8.5),
        plot.title = element_text(colour = INK, size = 10.5, face = "bold"),
        plot.subtitle = element_text(colour = MUTED, size = 8),
        plot.caption = element_text(colour = MUTED, size = 7.2, hjust = 0, lineheight = 1.1),
        plot.title.position = "plot", plot.caption.position = "plot",
        legend.position = "top", legend.justification = "left",
        legend.text = element_text(size = 8.5, colour = INK),
        legend.margin = margin(0, 0, 0, 0),
        plot.background = element_rect(fill = "white", colour = NA))

if (any(!s$reached))
  g <- g + geom_point(data = s[!s$reached, ], aes(x, y, colour = dist, shape = dist),
                      size = 2.1, fill = "white", stroke = 0.8)

ggsave(output, g, width = 8.2, height = 6.4, dpi = dpi, bg = "white")
cat("Wrote", output, "\n")

# article version: no title, subtitle or caption (they go in the figure legend);
# 180 mm wide (double column), as PNG (600 dpi) and PDF (vector)
art <- g + labs(title = NULL, subtitle = NULL, caption = NULL) +
  theme(legend.justification = "center", plot.margin = margin(4, 6, 4, 4))
art_base <- sub("\\.png$", "", output)
ggsave(paste0(art_base, "_article.png"), art, width = 180, height = 145, units = "mm", dpi = 600, bg = "white")
ggsave(paste0(art_base, "_article.pdf"), art, width = 180, height = 145, units = "mm", bg = "white")
cat("Wrote", paste0(art_base, "_article.png"), "and", paste0(art_base, "_article.pdf"), "\n")
cat("\nFor the figure legend:\n", caption, "\n\n", sep = "")

# console check: geometric mean of the N needed per r, p and distance (reached searches)
r_ok <- s[s$reached, ]
gm <- aggregate(x ~ reporting + substitution + dist, data = r_ok, FUN = function(v) round(exp(mean(log(v)))))
print(reshape(gm, idvar = c("reporting", "substitution"), timevar = "dist", direction = "wide"), row.names = FALSE)
