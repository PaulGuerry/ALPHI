#!/usr/bin/env Rscript
# MYO5B patients by liver and intestinal signature scores (from mixed_diagnostic.py -o),
# coloured by the cluster pairD_phenome.py assigned them, one panel per distance.
#   x  liver score: fraction of the liver signature features present
#   y  gut score: fraction of the intestinal signature features present
#   dashed lines: thresholds defining the mixed group (top right = mixed)
# Points are jittered slightly (the scores take few distinct values).
# Two separate groups with the top-right corner sitting with one of them -> no separate
# mixed phenotype; points spread evenly between the corners -> a continuum.
#
# Usage (in ~/261027_MYO5B/min8):
#   Rscript plot_myo5b_scores.R myo5b_mixed.csv -o Fig_myo5b_scores.png
# Writes the PNG and an article version (_article.png 600 dpi, _article.pdf, 180 mm wide).
# Requires: ggplot2 (>= 3.4)

suppressPackageStartupMessages(library(ggplot2))

CLUSTER_COLOURS <- c("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
NOISE <- "#b5b4ae"; INK <- "#1f1f1e"; MUTED <- "#6b6b67"; GRID <- "#e6e5e1"

args <- commandArgs(trailingOnly = TRUE)
input <- "myo5b_mixed.csv"; output <- "Fig_myo5b_scores.png"
i <- 1; pos <- character(0)
while (i <= length(args)) {
  if (args[i] %in% c("-o", "--output")) { output <- args[i + 1]; i <- i + 2 }
  else { pos <- c(pos, args[i]); i <- i + 1 }
}
if (length(pos) > 0) input <- pos[1]
if (!file.exists(input)) stop(input, " not found (run mixed_diagnostic.py -o first).")

d <- read.csv(input, stringsAsFactors = FALSE)
cl_cols <- grep("^cluster_", names(d), value = TRUE)
if (length(cl_cols) == 0) stop("No cluster_<run> columns in ", input)

long <- do.call(rbind, lapply(cl_cols, function(cc)
  data.frame(id = d$id, liver = d$liver_score, gut = d$gut_score, group = d$group,
             run = sub("^cluster_", "", cc), cluster = d[[cc]], stringsAsFactors = FALSE)))
long$run <- factor(long$run, levels = sub("^cluster_", "", cl_cols))
clusters <- sort(unique(long$cluster[long$cluster != "noise"]))
pal <- setNames(CLUSTER_COLOURS[seq_along(clusters)], clusters)
if (any(long$cluster == "noise")) pal <- c(pal, noise = NOISE)
long$cluster <- factor(long$cluster, levels = names(pal))

# thresholds half-way between the highest score below them and the lowest at or above them
thr <- function(score, inside)
  if (any(inside) && any(!inside)) (max(score[!inside]) + min(score[inside])) / 2 else NA_real_
x_thr <- thr(d$liver_score, d$group %in% c("mixed", "liver only"))
y_thr <- thr(d$gut_score, d$group %in% c("mixed", "gut only"))

n_mixed <- sum(d$group == "mixed")
set.seed(1)
g <- ggplot(long, aes(liver, gut, colour = cluster)) +
  geom_vline(xintercept = na.omit(x_thr), linetype = "dashed", colour = MUTED, linewidth = 0.4) +
  geom_hline(yintercept = na.omit(y_thr), linetype = "dashed", colour = MUTED, linewidth = 0.4) +
  geom_jitter(width = 0.03, height = 0.03, size = 1.6, alpha = 0.8) +
  annotate("text", x = 1.04, y = 1.06, label = paste0("mixed (n = ", n_mixed, ")"),
           hjust = 1, vjust = 0, size = 2.8, colour = MUTED) +
  facet_wrap(~ run, nrow = 1) +
  scale_colour_manual(values = pal, name = "Cluster") +
  scale_x_continuous(breaks = seq(0, 1, 0.2), limits = c(-0.06, 1.06), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq(0, 1, 0.25), limits = c(-0.06, 1.1), expand = c(0, 0)) +
  coord_fixed() +
  labs(title = "MYO5B patients by liver and intestinal signature",
       subtitle = "Each point is a patient, coloured by the cluster pairD_phenome assigned (one panel per distance)",
       x = "Liver signature score (fraction of features present)",
       y = "Intestinal signature score (fraction of features present)") +
  theme_minimal(base_size = 9) +
  theme(panel.grid.major = element_line(colour = GRID, linewidth = 0.4),
        panel.grid.minor = element_blank(),
        panel.spacing = unit(1, "lines"),
        axis.text = element_text(colour = MUTED),
        axis.title = element_text(colour = INK),
        strip.text = element_text(colour = INK, size = 9.5),
        plot.title = element_text(colour = INK, size = 10.5, face = "bold"),
        plot.subtitle = element_text(colour = MUTED, size = 8),
        plot.title.position = "plot",
        legend.position = "top", legend.justification = "left",
        legend.text = element_text(size = 8.5, colour = INK),
        legend.title = element_text(size = 8.5, colour = INK),
        plot.background = element_rect(fill = "white", colour = NA))

w <- 2.7 * nlevels(long$run) + 0.6
ggsave(output, g, width = w, height = 3.9, dpi = 300, bg = "white")
base <- sub("\\.png$", "", output)
art <- g + labs(title = NULL, subtitle = NULL)
ggsave(paste0(base, "_article.png"), art, width = 180, height = 180 * 3.5 / w, units = "mm", dpi = 600, bg = "white")
ggsave(paste0(base, "_article.pdf"), art, width = 180, height = 180 * 3.5 / w, units = "mm", bg = "white")
cat("Wrote", output, "and", paste0(base, "_article.png/.pdf"), "\n")
print(table(group = d$group, cluster = d[[cl_cols[1]]]))
