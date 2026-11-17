#!/usr/bin/env Rscript
# MYO5B figure 1: the patients in UMAP space, coloured by pairD cluster (a), and the same
# embedding coloured by the presence of each signature (b MVID: diarrhoea, microvillus
# inclusions and parenteral nutrition; c PFIC: cholestasis, pruritus, hyperbilirubinaemia and
# elevated AST or ALT). All panels square; cluster labels beside the clusters (--label-side).
#
# Input: PREFIX_patients.csv and PREFIX_clusters.csv from extract_myo5b_figdata.py.
# Usage (in ~/pairD_eval/myo5b):
#   Rscript ../scripts/fig_myo5b_umap.R figdata/ob_nearest -o ../figures/Fig_myo5b_umap_OB
#   Rscript ../scripts/fig_myo5b_umap.R figdata/resnik --distance Resnik -o ../figures/FigS_myo5b_umap_Resnik
# Options: --distance NAME (label in the panel title, default OB)
#          --label-side "MVID=left,PFIC=left,MVID+PFIC=right" (side of each cluster's label)
# Writes OUTPUT.png (600 dpi) and OUTPUT.pdf, 180 mm wide; prints the figure-legend text.
# Requires: ggplot2, patchwork

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(here, "myo5b_fig_common.R"))
opt <- parse_args(list(suffix = "umap", distance = "OB", `label-side` = "MVID=left,PFIC=left,MVID+PFIC=right"))

p <- read.csv(paste0(opt$prefix, "_patients.csv"), stringsAsFactors = FALSE)
cl <- read.csv(paste0(opt$prefix, "_clusters.csv"), stringsAsFactors = FALSE)
lev <- cluster_levels(p$cluster)
p$cluster <- factor(p$cluster, levels = lev)
cols <- cluster_colours(lev)

# ---- a: clusters, labelled at their centre (no legend) ---------------------------------------
cen <- aggregate(cbind(umap_x, umap_y) ~ cluster, data = p[p$cluster != "Noise", ], FUN = median)
cen <- merge(cen, cl[, c("cluster", "n", "stability")], by = "cluster")
cen$label <- sprintf("%s\nn = %d", cen$cluster, cen$n)
# label position: left or right of the cluster, at its vertical centre (--label-side,
# default MVID=left,PFIC=left,MVID+PFIC=right); clusters not listed: away from the map centre
sides <- strsplit(strsplit(opt$`label-side`, ",")[[1]], "=")
sides <- setNames(sapply(sides, `[`, 2), sapply(sides, `[`, 1))
span <- max(diff(range(p$umap_x)), diff(range(p$umap_y)))
gap <- 0.04 * span
mid_x <- mean(range(p$umap_x))
cen$side <- ifelse(cen$cluster %in% names(sides), sides[as.character(cen$cluster)],
                   ifelse(cen$umap_x < mid_x, "left", "right"))
for (i in seq_len(nrow(cen))) {
  m <- p[p$cluster == cen$cluster[i], ]
  cen$lx[i] <- if (cen$side[i] == "left") min(m$umap_x) - gap else max(m$umap_x) + gap
  cen$ly[i] <- median(m$umap_y)
}
cen$hjust <- ifelse(cen$side == "left", 1, 0)
# square plotting window containing the points and the labels (label width estimated)
lab_w <- 0.2 * span
xr <- range(c(p$umap_x, cen$lx - (cen$side == "left") * lab_w, cen$lx + (cen$side == "right") * lab_w))
yr <- range(p$umap_y)
half <- max(diff(xr), diff(yr)) / 2 * 1.08
lim_x <- mean(xr) + c(-half, half); lim_y <- mean(yr) + c(-half, half)

# UMAP coordinates have no units: no axis numbers, but the light grid marks the plot area
blank_axes <- theme(axis.text = element_blank(), axis.ticks = element_blank(),
                    panel.grid.major = element_line(colour = GRID, linewidth = 0.3))

pa <- ggplot(p, aes(umap_x, umap_y)) +
  geom_point(aes(fill = cluster), shape = 21, colour = "white", stroke = 0.3, size = 2.2, alpha = 0.95) +
  geom_text(data = cen, aes(lx, ly, label = label, colour = cluster, hjust = hjust), size = 2.8,
            lineheight = 0.9, fontface = "bold") +
  coord_equal(xlim = lim_x, ylim = lim_y, expand = FALSE) +
  scale_fill_manual(values = cols, guide = "none") +
  scale_colour_manual(values = cols, guide = "none") +
  labs(x = "UMAP 1", y = "UMAP 2", title = paste0("Clusters (ALPHI-", opt$distance, ")")) +
  theme_fig() + blank_axes +
  theme(panel.background = element_rect(fill = NA, colour = NA))

# ---- b-d: signatures -------------------------------------------------------------------
feature_panel <- function(var, title, colour) {
  q <- p
  q$present <- factor(ifelse(as.character(q[[var]]) %in% c("True", "TRUE"), "present", "absent"),
                      levels = c("absent", "present"))
  q <- q[order(q$present), ]                      # present drawn on top
  ggplot(q, aes(umap_x, umap_y, fill = present)) +
    geom_point(shape = 21, colour = "white", stroke = 0.2, size = 1.4) +
    coord_equal(xlim = lim_x, ylim = lim_y, expand = FALSE) +
    scale_fill_manual(values = c(absent = ABSENT, present = colour), guide = "none") +
    labs(x = NULL, y = NULL, title = sprintf("%s (%d / %d)", title, sum(q$present == "present"), nrow(q))) +
    theme_fig(7) + blank_axes +
    theme(plot.title = element_text(size = 7.5, face = "plain"))
}
pb <- feature_panel("intestinal", "MVID signature", CLUSTER_COLOURS[["MVID"]])
pc <- feature_panel("cholestatic", "PFIC signature", CLUSTER_COLOURS[["PFIC"]])

g <- pa + pb + pc +
  plot_layout(design = "AAB\nAAC") +
  plot_annotation(tag_levels = "a")
save_fig(g, opt$output, 180, 130)

cat("\nFor the figure legend:\n",
    "UMAP embedding of the ", nrow(p), " MYO5B patients (ALPHI-", opt$distance, "). ",
    "(a) ALPHI clusters; (b, c) the same embedding, patients with the MVID signature ",
    "(diarrhoea, microvillus inclusions and dependency on parenteral nutrition) or the PFIC ",
    "signature (cholestasis, pruritus, hyperbilirubinaemia and elevated AST or ALT; each term counted with its ",
    "descendants) in colour. The 2-D embedding is for display; clusters were found in ",
    "ALPHI's 10-D embedding of the same distances.\n", sep = "")
