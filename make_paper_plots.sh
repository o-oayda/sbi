#!/usr/bin/env bash

python -m dipolesbi.scripts.paper_covmap --save_plots
python -m dipolesbi.scripts.catwise_alpha --save
python -m dipolesbi.scripts.evidence_acc --save_dir nside_64_coarse4_funnel_v2 --smoothing_window 2 --skip_delta --save_plot
python -m dipolesbi.scripts.coverage_error_heatmap --mag-min 11 --vline-mag --log-color --save-plot --figure-name cov_mag_error_heatmap.pdf
./run_posterior_plots.sh fiducial_50k --round 14
python -m dipolesbi.scripts.lnB_scatter

python -m dipolesbi.tools.posterior_cli exp_out/nside_64_coarse4_funnel_v2/20251020_124634_SEED21_NLE/ --corner corner_evidence_acc_example.pdf --corner-no-legend --corner-include-true --legend "NLE-derived posterior" --corner-true-legend "True posterior" --corner-simple-titles
cp "corner_evidence_acc_example.pdf" "${HOME}/Documents/papers/catwise_sbi/figures/"
