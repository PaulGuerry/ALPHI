#!/usr/bin/env Rscript
# MYO5B figure 3: genotype and outcome of the ALPHI clusters, data not used for clustering.
#   a  genotype class per cluster (genotyped patients with both alleles classified)
#   b  variant positions on the MYO5B protein, one track per cluster
#   c  survival (Kaplan-Meier, age in years) per cluster, with log-rank test
#
# Input: PREFIX_patients.csv, PREFIX_variants.csv from extract_myo5b_figdata.py, and the
# domain map myo5b_domains.csv (in scripts/; CHECK the boundaries against UniProt Q9ULV0
# before publication: they were entered approximately).
# Usage (in ~/pairD_eval/myo5b):
#   Rscript ../scripts/fig_myo5b_genotype.R figdata/ob_nearest -o ../figures/Fig_myo5b_genotype_OB
# Options: --distance NAME (default OB), --domains FILE, --protein-length L (default 1848)
# Writes OUTPUT.png (600 dpi) and OUTPUT.pdf, 180 mm wide; prints the figure-legend text and
# the tests.
# Requires: ggplot2, patchwork, survival (a recommended package, installed with R)

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(here, "myo5b_fig_common.R"))
suppressPackageStartupMessages(library(survival))
opt <- parse_args(list(suffix = "genotype", distance = "OB",
                       domains = file.path(here, "myo5b_domains.csv"), `protein-length` = "1848"))
plen <- as.numeric(opt$`protein-length`)

p <- read.csv(paste0(opt$prefix, "_patients.csv"), stringsAsFactors = FALSE)
v <- read.csv(paste0(opt$prefix, "_variants.csv"), stringsAsFactors = FALSE)
dom <- read.csv(opt$domains, stringsAsFactors = FALSE, comment.char = "#")
p <- p[p$cluster != "Noise", ]
lev <- cluster_levels(p$cluster)
cols <- cluster_colours(lev)
p$cluster <- factor(p$cluster, levels = lev)

# ---- a: genotype class -----------------------------------------------------------------
CLASS_ORDER <- c("Biallelic null", "Null + missense", "Biallelic missense", "Other")
CLASS_COLOURS <- c("Biallelic null" = "#173f73", "Null + missense" = "#5b8fd0",
                   "Biallelic missense" = "#b9d0ee", "Other" = "#d9d8d3")
g_ok <- p[p$genotype_class %in% CLASS_ORDER, ]
tab <- as.data.frame(table(cluster = g_ok$cluster, class = g_ok$genotype_class))
tab$class <- factor(tab$class, levels = rev(CLASS_ORDER))
tot <- aggregate(Freq ~ cluster, data = tab, FUN = sum)
tab$pct <- 100 * tab$Freq / tot$Freq[match(tab$cluster, tot$cluster)]
untested <- table(factor(p$cluster, levels = lev), p$genotype_class %in% c("Not genotyped", "Incomplete"))[, "TRUE"]
xlab_a <- sprintf("%s\nn = %d", lev, tot$Freq[match(lev, tot$cluster)])

pa <- ggplot(tab, aes(cluster, pct, fill = class)) +
  geom_col(width = 0.7, colour = "white", linewidth = 0.3) +
  geom_text(data = tab[tab$Freq > 0, ], aes(label = Freq), position = position_stack(vjust = 0.5),
            size = 2.2, colour = ifelse(tab$class[tab$Freq > 0] %in% c("Biallelic null", "Null + missense"),
                                        "white", INK)) +
  scale_fill_manual(values = CLASS_COLOURS, breaks = CLASS_ORDER, name = NULL) +
  scale_x_discrete(labels = xlab_a) +
  scale_y_continuous(expand = c(0, 0), breaks = c(0, 50, 100)) +
  labs(x = NULL, y = "Genotyped patients (%)") +
  theme_fig() +
  theme(panel.grid.major.x = element_blank(), legend.position = "bottom",
        legend.key.size = unit(0.7, "lines"), legend.text = element_text(size = 6.5)) +
  guides(fill = guide_legend(nrow = 2))

# ---- b: protein map --------------------------------------------------------------------
v <- v[v$cluster %in% lev, ]
v$cluster <- factor(v$cluster, levels = lev)
v$kind <- ifelse(v$type == "missense", "Missense",
                 ifelse(v$type %in% c("nonsense", "frameshift", "splicing"), "Truncating", "Other"))
v$kind <- factor(v$kind, levels = c("Missense", "Truncating", "Other"))
track <- setNames(rev(seq_along(lev)), lev)
set.seed(1)
v$y <- track[as.character(v$cluster)] + runif(nrow(v), -0.28, 0.28)
dom$ymin <- 0.15; dom$ymax <- 0.55
dom$mid <- (dom$start + dom$end) / 2

pb <- ggplot() +
  annotate("segment", x = 1, xend = plen, y = 0.35, yend = 0.35, colour = MUTED, linewidth = 0.6) +
  geom_rect(data = dom, aes(xmin = start, xmax = end, ymin = ymin, ymax = ymax), fill = "#e9e3d5", colour = MUTED,
            linewidth = 0.25) +
  geom_text(data = dom, aes(x = mid, y = 0.35, label = domain), size = 2, colour = INK) +
  geom_point(data = v, aes(position, y, fill = cluster, shape = kind), colour = "white", stroke = 0.25,
             size = 1.9, alpha = 0.95) +
  scale_fill_manual(values = cols, guide = "none") +
  scale_shape_manual(values = c(Missense = 21, Truncating = 24, Other = 22), name = NULL) +
  scale_x_continuous(limits = c(0, plen + 10), breaks = c(1, 500, 1000, 1500, plen), expand = c(0.01, 0)) +
  scale_y_continuous(breaks = track, labels = names(track), limits = c(0.1, max(track) + 0.45)) +
  labs(x = "MYO5B residue", y = NULL) +
  theme_fig() +
  theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID, linewidth = 0.25),
        axis.text.y = element_text(colour = cols[names(track)], face = "bold"),
        legend.position = "bottom", legend.key.size = unit(0.7, "lines")) +
  guides(shape = guide_legend(override.aes = list(fill = MUTED)))

# ---- c: survival -----------------------------------------------------------------------
s <- p[!is.na(p$age_years) & p$vital %in% c("alive", "dead"), ]
s$event <- as.numeric(s$vital == "dead")
fit <- survfit(Surv(age_years, event) ~ cluster, data = s)
lr <- survdiff(Surv(age_years, event) ~ cluster, data = s)
p_lr <- pchisq(lr$chisq, df = length(lr$n) - 1, lower.tail = FALSE)
km <- data.frame(time = fit$time, surv = fit$surv, n.censor = fit$n.censor,
                 cluster = rep(sub("^cluster=", "", names(fit$strata)), fit$strata))
start <- data.frame(time = 0, surv = 1, n.censor = 0, cluster = unique(km$cluster))
km <- rbind(start, km)
km$cluster <- factor(km$cluster, levels = lev)
cens <- km[km$n.censor > 0, ]
ends <- do.call(rbind, lapply(split(km, km$cluster), function(d) d[nrow(d), ]))
ends$label <- sprintf("%s (%d/%d)", ends$cluster, tapply(s$event, s$cluster, sum)[as.character(ends$cluster)],
                      table(s$cluster)[as.character(ends$cluster)])

pc <- ggplot(km, aes(time, 100 * surv, colour = cluster)) +
  geom_step(linewidth = 0.7) +
  geom_point(data = cens, shape = 3, size = 1, stroke = 0.4) +
  scale_colour_manual(values = cols, guide = "none") +
  annotate("text", x = Inf, y = 4, hjust = 1.05, size = 2.3, colour = MUTED,
           label = sprintf("log-rank p %s", format.pval(p_lr, digits = 2, eps = 1e-4))) +
  geom_text(data = ends, aes(x = time, y = 100 * surv, label = label), hjust = -0.05, size = 2.2,
            show.legend = FALSE) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.35))) +
  scale_y_continuous(limits = c(0, 100), breaks = c(0, 50, 100)) +
  labs(x = "Age (years)", y = "Alive (%)") +
  theme_fig()

g <- (pa | pc) / pb + plot_layout(heights = c(1.15, 1)) + plot_annotation(tag_levels = "a")
save_fig(g, opt$output, 180, 140)

# ---- console: tests and legend -----------------------------------------------------------
null_tab <- table(g_ok$cluster, g_ok$genotype_class == "Biallelic null")
cat("\nBiallelic null vs other complete genotypes, by cluster:\n"); print(null_tab)
ft <- fisher.test(null_tab)
cat(sprintf("Fisher's exact test: p = %s\n", format.pval(ft$p.value, digits = 2)))
cat(sprintf("Log-rank test: chi2 = %.1f, p = %s\n", lr$chisq, format.pval(p_lr, digits = 2, eps = 1e-4)))
pre <- s[!is.na(s$year) & s$year >= 2010, ]
if (nrow(pre) > 0 && length(unique(pre$cluster)) > 1) {
  lr2 <- survdiff(Surv(age_years, event) ~ cluster, data = pre)
  cat(sprintf("Log-rank, articles from 2010 on (n = %d): p = %s\n", nrow(pre),
              format.pval(pchisq(lr2$chisq, length(lr2$n) - 1, lower.tail = FALSE), digits = 2, eps = 1e-4)))
}
gen <- s[s$genotyped %in% c("True", "TRUE"), ]
if (nrow(gen) > 0 && length(unique(gen$cluster)) > 1) {
  lr3 <- survdiff(Surv(age_years, event) ~ cluster, data = gen)
  cat(sprintf("Log-rank, genotyped patients only (n = %d): p = %s\n", nrow(gen),
              format.pval(pchisq(lr3$chisq, length(lr3$n) - 1, lower.tail = FALSE), digits = 2, eps = 1e-4)))
}

cat("\nFor the figure legend:\n",
    "Genotype and outcome of the MYO5B clusters (ALPHI-", opt$distance, "); neither was used for clustering. ",
    "(a) Genotype class of patients with both alleles classified (null: nonsense, frameshift or splicing; ",
    "not genotyped or incompletely classified: ", paste(sprintf("%s %d", lev, untested), collapse = ", "), "). ",
    "Fisher's exact test, biallelic null against other genotypes: p = ", format.pval(ft$p.value, digits = 2), ". ",
    "(b) Variants with a protein position (splice variants without one are not shown); one point per ",
    "distinct variant per patient. (c) Survival from birth (deaths / patients in brackets); ",
    "+ censored at last follow-up.\n", sep = "")
