#!/usr/bin/env Rscript
# With --kind absent: the same plot for the symptoms explicitly reported ABSENT (PREFIX_absent.csv),
# for the supplementary figure; output name *_absent.
#
# MYO5B figure 2: phenotypic make-up of the ALPHI clusters. Dot plot of symptom prevalence
# (% of patients, dot area) per cluster, for the reported terms whose prevalence differs most
# between clusters; rows grouped by organ system, dots coloured by organ subgroup.
# Column headers give the cluster size and resampling stability.
#
# Input: PREFIX_prevalence.csv and PREFIX_clusters.csv from extract_myo5b_figdata.py (groups
# and subgroups are defined there, in GROUPS).
# Usage (in ~/pairD_eval/myo5b):
#   Rscript ../scripts/fig_myo5b_phenotype.R figdata/ob_nearest -o ../figures/Fig_myo5b_phenotype_OB
# Options: --distance NAME (default OB)
# Writes OUTPUT.png (600 dpi) and OUTPUT.pdf, 180 mm wide; prints the figure-legend text.
# Requires: ggplot2, patchwork

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(here, "myo5b_fig_common.R"))
opt <- parse_args(list(suffix = "phenotype", distance = "OB", kind = "present"))
if (!opt$kind %in% c("present", "absent")) stop("--kind must be present or absent")
ABSENT_KIND <- opt$kind == "absent"
if (ABSENT_KIND && grepl("_phenotype$", opt$output)) opt$output <- sub("_phenotype$", "_absent", opt$output)

v <- read.csv(paste0(opt$prefix, if (ABSENT_KIND) "_absent.csv" else "_prevalence.csv"), stringsAsFactors = FALSE)
cl <- read.csv(paste0(opt$prefix, "_clusters.csv"), stringsAsFactors = FALSE)
cl <- cl[cl$cluster != "Noise", ]
v <- v[v$group != "Expected signature", ]          # older extractions

lev <- cluster_levels(v$cluster)
cl <- cl[match(lev, cl$cluster), ]
flag_txt <- ifelse(as.character(cl$flag_few_features) %in% c("True", "TRUE"), "FLAGGED", "not flagged")
header <- sprintf("%s\nn = %d\nstability\n%.2f", cl$cluster, cl$n, cl$stability)   # 4 lines: no overlap between columns
v$cluster <- factor(v$cluster, levels = lev, labels = header)

# ---- groups, subgroups and their colours (one hue family per group) ----------------------
GROUP_ORDER <- c("Intestine", "Liver", "Growth", "Metabolism", "Blood", "Kidney", "Prenatal and birth", "Other")
SUBGROUP_COLOURS <- c(
  "Diarrhoea" = "#e3a35b", "Mucosal histology" = "#b5651d", "Nutrition and absorption" = "#d9822b",
  "Other intestinal" = "#c9a27a",
  "Cholestasis" = "#2e8b6a", "Hepatocellular injury" = "#56b08c", "Liver histology" = "#1b5a43",
  "Organomegaly" = "#7fa99a", "Other liver" = "#9cc7b6",
  "Growth" = "#7b52b9", "Fluid, protein and acid-base" = "#3f7fd0", "Blood and coagulation" = "#c4495a",
  "Kidney" = "#b5527f", "Prenatal and birth" = "#8a7440")
groups <- intersect(GROUP_ORDER, unique(v$group))
groups <- c(groups, setdiff(unique(v$group), groups))
v$group <- factor(v$group, levels = groups)
subs <- unique(v$subgroup[order(v$group)])
known <- subs[subs %in% names(SUBGROUP_COLOURS)]
extra <- setdiff(subs, known)                       # e.g. "Other" systems: greys
sub_cols <- c(SUBGROUP_COLOURS[known], setNames(grey.colors(max(1, length(extra)), 0.45, 0.7)[seq_along(extra)], extra))
v$subgroup <- factor(v$subgroup, levels = subs)

# ---- term order: by subgroup, then by the cluster where the term peaks, then prevalence ----
peak <- do.call(rbind, lapply(split(v, v$term), function(d) {
  i <- which.max(d$prevalence)
  data.frame(term = d$term[1], group = d$group[1], subgroup = d$subgroup[1],
             peak_cluster = as.integer(d$cluster[i]), peak = d$prevalence[i])
}))
peak <- peak[order(peak$group, peak$subgroup, peak$peak_cluster, -peak$peak), ]
short <- function(x) sub(" concentration$", "", sub("^Elevated circulating ", "Elevated ", x))
v$term_f <- factor(short(v$term), levels = rev(unique(short(peak$term))))

# white text on dark dots, dark text on light ones
lum <- function(hex) { m <- col2rgb(hex) / 255; 0.2126 * m[1, ] + 0.7152 * m[2, ] + 0.0722 * m[3, ] }
v$txt <- ifelse(lum(sub_cols[as.character(v$subgroup)]) < 0.5, "white", INK)

g <- ggplot(v, aes(cluster, term_f)) +
  geom_point(aes(size = prevalence, fill = subgroup), shape = 21, colour = "white", stroke = 0.3) +
  geom_text(data = v[v$prevalence >= if (ABSENT_KIND) 30 else 50, ], aes(label = round(prevalence), colour = txt), size = 1.9) +
  facet_grid(group ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_size_area(max_size = 5.2, limits = c(0, 100), breaks = c(25, 50, 75, 100),
                  name = if (ABSENT_KIND) "Reported absent (%)" else "Patients (%)") +
  scale_fill_manual(values = sub_cols, name = NULL, drop = TRUE) +
  scale_colour_identity() +
  scale_x_discrete(position = "top") +
  labs(x = NULL, y = NULL) +
  theme_fig(7.5) +
  theme(panel.grid.major = element_line(colour = GRID, linewidth = 0.25),
        axis.text.x.top = element_text(colour = INK, size = 7.5, lineheight = 0.95),
        axis.text.y = element_text(colour = INK, size = 7),
        strip.placement = "outside",
        strip.text.y.left = element_text(angle = 0, hjust = 1, face = "bold", size = 7),
        panel.spacing.y = unit(0.35, "lines"),
        legend.position = "right",
        legend.key.size = unit(0.75, "lines"),
        legend.text = element_text(size = 6.8)) +
  guides(fill = guide_legend(override.aes = list(size = 3), order = 1),
         size = guide_legend(override.aes = list(fill = MUTED), order = 2))

n_rows <- length(unique(v$term))
save_fig(g, opt$output, 180, 30 + 4.3 * n_rows)

if (ABSENT_KIND) {
  cat("\nFor the figure legend:\n",
      "Symptoms explicitly reported as absent in the clusters of MYO5B patients (ALPHI-", opt$distance, "),",
      " not used for clustering. A symptom was counted as absent when listed as absent and not ",
      "as present (intermittent symptoms, coded as both, were excluded). Terms reported absent in at least ",
      "10% of the patients of one cluster are shown, grouped by organ and colored by organ subgroup. ",
      "Dot area and number: percentage of patients of the cluster.\n", sep = "")
  quit(save = "no")
}
cat("\nFor the figure legend:\n",
    "Prevalence of symptoms in the ALPHI clusters of MYO5B patients (ALPHI-", opt$distance, "): ",
    "the reported HPO terms whose prevalence differs most between clusters (at least 20 percentage ",
    "points), grouped by organ; colour: organ subgroup; dot area and number: % of patients. ",
    "Stability: mean Jaccard similarity of each cluster over 50 resamples of 80 % of patients. ",
    "Few-feature check: ", paste(sprintf("%s %s", cl$cluster, flag_txt), collapse = ", "),
    " (closest clusters differ on ", paste(cl$n_discriminating, collapse = " / "), " symptoms).\n", sep = "")
