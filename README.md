# ALPHI — agnostic latent phenotype identification

ALPHI groups patients described by [Human Phenotype Ontology](https://hpo.jax.org) (HPO) terms
into phenotypes, without reference to their diagnostic labels. It tests whether the patients
sharing a diagnosis form one phenotype, several, or none: a bottom-up definition of diseases.

The method combines an ontology-based distance between patients (organ-branch distance, OB),
UMAP embedding, density-based clustering (HDBSCAN) with a grid search over its parameters,
resampling stability, a few-feature check and a latent class check. It is described in:

> Guerry, P., & Fabre, A. Defining diseases from the bottom up with agnostic latent phenotype
> identification (ALPHI). Submitted.

## Two ways to run it

- **In the browser** — <https://paulguerry.github.io/ALPHI/> (see [web/README.md](web/README.md)).
  Nothing is installed and the patient data never leave the browser.
- **From the command line** — `alphi.py`, below.

## Installation

Python 3.11 or later (the article used Python 3.14).

```bash
git clone https://github.com/PaulGuerry/ALPHI.git
cd ALPHI
pip install -r requirements.txt
```

## Usage

```bash
# MYO5B cohort, as in the article
python alphi.py -f 261013_pytheasDB.json --gene MYO5B --min-symptoms 8 --stability 50 --lca \
    --output-json clusters.json
```

Main options (`python alphi.py --help` lists all of them):

| Option | Meaning |
|---|---|
| `-f FILE` | patient file (JSON, format below) |
| `--gene G [G ...]` | genes defining the cohort |
| `--min-symptoms N` | minimum number of distinct HPO terms per patient (default 5) |
| `--lca` | latent class check (ALPHI); without it, ALPHI-OB |
| `--stability B` | B resamples of 80% of the patients for cluster stability (default 0, off) |
| `--excluded-doi DOI` | patients of one article left out (pass `""` to keep all) |
| `--workers N` | processes used to compute the distances |

Each run writes a folder `YYMMDD_GENES_SYSTEM_PAIRS/` with the distance matrix (`cdm.csv`),
the embedding and cluster labels (`pairD_UMAP.csv`; the final ALPHI groups are in
`HDBSCAN_hybrid`), stability (`stability.csv`), the latent class check (`lca_check.json`),
the few-feature check (`few_feature_flags.csv`) and the full log (`pipeline.log`), and the
cluster description in the JSON given to `--output-json`.

## Patient file

A JSON list of patients. Required fields: `id`, `gene` (list of gene symbols) and `symptoms`
(HPO codes, as `"0001508"` strings or `["0001508", …]` lists). Optional: `doi`,
`alivedead` (`alive`/`dead`) and `alivedeadage` (years) for the survival analysis,
`absentsymptoms`.

`261013_pytheasDB.json` is the export of [PytheasDB](https://pytheasdb.com) (13 October 2026)
used in the article (Percheron, Guerry & Fabre, 2024, *Intractable & Rare Diseases Research*,
13, 255–258, https://doi.org/10.5582/irdr.2024.01043).

## HPO files

- `hp_trees.json`: paths of every HPO term to the root, built by `build_hp_trees.py` from the
  HPO graph (`--hpo-json`).
- `id_to_term.json`: HPO identifiers and labels.

These files are derived from the Human Phenotype Ontology (Gargano et al., 2024,
*Nucleic Acids Research*, 52, D1333–D1346), used under its licence; see
<https://hpo.jax.org/license>. The Resnik distance (not used by ALPHI itself) also needs the
HPO annotation file `phenotype.hpoa`, available from the HPO website.

## Licence

MIT (see [LICENSE](LICENSE)).
