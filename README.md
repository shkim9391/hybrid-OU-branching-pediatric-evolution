# hybrid-OU-branching-pediatric-evolution

Code and reproducible analysis for the hybrid Ornstein-Uhlenbeck (OU) / branching framework linking microbial evolution and pediatric tumor evolution.

This repository accompanies the Frontiers in Oncology article:

**Kim S-H (2026). _A hybrid Ornstein-Uhlenbeck-Branching framework unifies microbial and pediatric tumor evolution_. Frontiers in Oncology. 16:1727973. doi: 10.3389/fonc.2026.1727973**

---

## Overview

This repository contains the scripts, processed data tables, and figure-generation workflows used for the Frontiers in Oncology manuscript. The study presents a hybrid OU-branching perspective for evolutionary dynamics, connecting:

- constrained stochastic evolution in microbial systems,
- lineage diversification and transition structure,
- and translational parallels to pediatric tumor evolution.

The code in this repository is organized around manuscript figure generation rather than as a general-purpose software package.

Broadly, the repository includes:

- cleaned mutation-frequency input data,
- grouped bootstrap summaries,
- pairwise trajectory metrics,
- scripts for main figures,
- a supplementary figure script,
- and a table-construction script for manuscript reporting.

---

## Repository contents

Current top-level files include:

```text
README.md
Table1_grouped_bootstrap_samples.csv
Table1_grouped_bootstrap_summary.csv
Table3_pairwise_trajectory_metrics.csv
fig1_schematic_ecoli_pediatric_lineages.py
fig2_line_plot_with_bands_updated.py
fig3_ou_exact_transition_negative_log_likelihoods.py
fig4_ou_negative_likelihoods_wt_priA_recG_vertical.py
fig5_three_phase_planes.py
fig6_lineage_graph_t20.py
fig7_pediatric_precision_showcase_final_priA.py
make_table3_from_grouped_bootstrap.py
mut_freq_data_clean.csv
supp_fig1_lifespan.py

Scientific scope

The repository supports a manuscript that uses a hybrid OU-branching framework to unify two scales of evolutionary interpretation:
	1.	Microbial evolution
Laboratory mutation-frequency trajectories and lineage behavior are used as interpretable model systems for constrained stochastic evolution.
	2.	Pediatric tumor evolution
The framework is extended conceptually and visually to illustrate how branching, transition structure, and constrained dynamics may inform pediatric cancer evolution.

The central modeling idea is that evolutionary systems often exhibit both:
	•	OU-like mean reversion or constrained drift, and
	•	branching or lineage divergence over time.

This repository focuses on figure-ready computational analysis and manuscript-specific outputs.

⸻

Main input data files

mut_freq_data_clean.csv

Primary cleaned dataset used in the manuscript analyses.

This file is the main processed input for trajectory-based and likelihood-based figure scripts. Depending on the exact script, it likely contains combinations of:
	•	strain or condition labels,
	•	mutation-frequency measurements,
	•	timepoint information,
	•	lineage or replicate identifiers,
	•	model-ready summary variables.

This is the core input file for several of the main figures.

Table1_grouped_bootstrap_samples.csv

Grouped bootstrap samples used for manuscript summary statistics.

This file likely contains bootstrap-resampled estimates underlying Table 1 and/or related uncertainty summaries.

Table1_grouped_bootstrap_summary.csv

Grouped bootstrap summary table for manuscript reporting.

This file likely contains summarized bootstrap outputs such as means, intervals, or grouped estimates used in Table 1.

Table3_pairwise_trajectory_metrics.csv

Pairwise trajectory metric table used for manuscript reporting and comparisons.

This file is likely used to summarize distances, transition-related quantities, or pairwise evolutionary trajectory comparisons reported in Table 3 or related downstream figures.

⸻

Script inventory

fig1_schematic_ecoli_pediatric_lineages.py

Generates Figure 1.

This script appears to create the conceptual schematic linking E. coli lineage evolution and pediatric tumor lineage interpretation. It likely serves as the main framing figure for the manuscript.

Typical role:
	•	conceptual figure,
	•	schematic bridging microbial and pediatric systems,
	•	manuscript opening visualization.

fig2_line_plot_with_bands_updated.py

Generates Figure 2.

This script likely produces time-course or trajectory line plots with uncertainty bands, probably from the cleaned mutation-frequency dataset.

Typical role:
	•	longitudinal visualization,
	•	uncertainty bands around trajectories,
	•	comparison across strains, conditions, or grouped lineages.

fig3_ou_exact_transition_negative_log_likelihoods.py

Generates Figure 3.

Based on the filename, this script likely evaluates or visualizes negative log-likelihood quantities under an exact OU transition formulation.

Typical role:
	•	likelihood-based assessment of OU transition behavior,
	•	model-fit comparison,
	•	supporting evidence for constrained evolutionary dynamics.

fig4_ou_negative_likelihoods_wt_priA_recG_vertical.py

Generates Figure 4.

This script likely compares OU-related likelihood summaries across microbial strains or conditions, including WT, priA, and recG, using a vertically stacked layout.

Typical role:
	•	between-strain comparison,
	•	manuscript figure showing differential likelihood structure,
	•	visual support for strain-specific evolutionary behavior.

fig5_three_phase_planes.py

Generates Figure 5.

This script likely creates three phase-plane visualizations showing different evolutionary regimes, state-space behavior, or lineage progression views.

Typical role:
	•	phase-plane visualization,
	•	dynamical interpretation of the hybrid OU-branching framework,
	•	mechanistic comparison across example systems or states.

fig6_lineage_graph_t20.py

Generates Figure 6.

This script likely produces a lineage graph at a selected time index or threshold (t20), highlighting branching structure or lineage connectivity.

Typical role:
	•	lineage-network visualization,
	•	branch structure illustration,
	•	temporal graph representation.

fig7_pediatric_precision_showcase_final_priA.py

Generates Figure 7.

This script appears to be the pediatric precision showcase figure, potentially using the priA framing as a bridge or analogical anchor from microbial evolution to tumor interpretation.

Typical role:
	•	translational showcase figure,
	•	pediatric cancer interpretation panel,
	•	concluding precision-oncology-style visualization.

supp_fig1_lifespan.py

Generates Supplementary Figure 1.

This script likely produces a supplementary lifespan- or duration-related figure supporting the main manuscript narrative.

make_table3_from_grouped_bootstrap.py

Builds Table 3 from grouped bootstrap outputs.

This script likely transforms bootstrap-derived summaries into the pairwise trajectory metric table used in manuscript reporting.

Typical role:
	•	table construction,
	•	aggregation of resampled metrics,
	•	manuscript-ready export for Table 3.

⸻

Suggested workflow

A typical workflow from the repository root may be:

python fig1_schematic_ecoli_pediatric_lineages.py
python fig2_line_plot_with_bands_updated.py
python fig3_ou_exact_transition_negative_log_likelihoods.py
python fig4_ou_negative_likelihoods_wt_priA_recG_vertical.py
python fig5_three_phase_planes.py
python fig6_lineage_graph_t20.py
python fig7_pediatric_precision_showcase_final_priA.py

python supp_fig1_lifespan.py
python make_table3_from_grouped_bootstrap.py

A practical interpretation is:
	1.	load cleaned mutation-frequency and summary tables,
	2.	generate manuscript main figures,
	3.	generate the supplementary figure,
	4.	construct the derived Table 3 output from grouped bootstrap summaries.

⸻

Likely data flow

A simple way to think about the workflow is:
	•	mut_freq_data_clean.csv
	•	used by the main figure scripts for trajectory, likelihood, and phase-plane analysis
	•	Table1_grouped_bootstrap_samples.csv
	•	upstream bootstrap samples used in grouped uncertainty summaries
	•	Table1_grouped_bootstrap_summary.csv
	•	summarized grouped bootstrap statistics for manuscript reporting
	•	make_table3_from_grouped_bootstrap.py
	•	consumes grouped bootstrap outputs
	•	produces or supports Table3_pairwise_trajectory_metrics.csv

⸻

Figure mapping

A simple figure-to-script map is:
	•	fig1_schematic_ecoli_pediatric_lineages.py → Main Figure 1
	•	fig2_line_plot_with_bands_updated.py → Main Figure 2
	•	fig3_ou_exact_transition_negative_log_likelihoods.py → Main Figure 3
	•	fig4_ou_negative_likelihoods_wt_priA_recG_vertical.py → Main Figure 4
	•	fig5_three_phase_planes.py → Main Figure 5
	•	fig6_lineage_graph_t20.py → Main Figure 6
	•	fig7_pediatric_precision_showcase_final_priA.py → Main Figure 7
	•	supp_fig1_lifespan.py → Supplementary Figure S1

⸻

Typical Python loading examples

Load the cleaned mutation-frequency dataset

import pandas as pd

mut = pd.read_csv("mut_freq_data_clean.csv")
print(mut.head())

Load the grouped bootstrap sample table

import pandas as pd

boot_samples = pd.read_csv("Table1_grouped_bootstrap_samples.csv")
print(boot_samples.head())

Load the pairwise trajectory metrics table

import pandas as pd

table3 = pd.read_csv("Table3_pairwise_trajectory_metrics.csv")
print(table3.head())

Software environment

This repository is intended to run in Python 3 with a standard scientific Python environment.

Typical packages likely required:
	•	numpy
	•	pandas
	•	matplotlib
	•	scipy

Depending on the exact plotting scripts, you may also need:
	•	seaborn
	•	networkx
	•	openpyxl

A minimal installation example is:

pip install numpy pandas matplotlib scipy seaborn networkx openpyxl

A conda-based setup is also reasonable:

conda create -n hybrid-ou-branching python=3.11
conda activate hybrid-ou-branching
pip install numpy pandas matplotlib scipy seaborn networkx openpyxl

Reproducibility notes
	•	This repository is manuscript-specific.
	•	The scripts are designed primarily for figure generation and manuscript reporting.
	•	Input datasets are processed and analysis-ready.
	•	Some outputs may depend on local plotting defaults, font availability, or minor package-version differences.
	•	For archival reproducibility, it is recommended to preserve the exact repository commit associated with manuscript submission or publication.

For a stronger public release, consider adding:
	•	requirements.txt
	•	environment.yml
	•	.gitignore
	•	LICENSE
	•	a figures/ directory for exported outputs
	•	a short note in each script header describing its inputs and outputs

⸻

Data availability

All processed data and code required to reproduce the analyses in the Frontiers in Oncology manuscript are included in this repository. No proprietary datasets or access restrictions apply to the materials provided here.

⸻

Citation

If you use this repository, please cite:

Kim S-H (2026). A hybrid Ornstein-Uhlenbeck-Branching framework unifies microbial and pediatric tumor evolution. Frontiers in Oncology. 16:1727973. doi:10.3389/fonc.2026.1727973

⸻

Author

Seung-Hwan Kim
