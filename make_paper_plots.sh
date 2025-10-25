#!/usr/bin/env bash

fig_out_dir="${HOME}/Documents/papers/catwise_sbi/figures/"

python -m dipolesbi.scripts.paper_covmap --save_plots
python -m dipolesbi.scripts.catwise_alpha --save
python -m dipolesbi.scripts.evidence_acc --save_dir nside_64_coarse4_funnel_v2\
  --smoothing_window 2 --skip_delta --save_plot
python -m dipolesbi.scripts.coverage_error_heatmap --mag-min 11 --vline-mag\
  --log-color --save-plot --figure-name cov_mag_error_heatmap.pdf
./run_posterior_plots.sh fiducial_50k --round 14
python -m dipolesbi.scripts.lnB_scatter

# true vs NLE-inferred posterior
python -m dipolesbi.tools.posterior_cli\
  exp_out/nside_64_coarse4_funnel_v2/20251020_124634_SEED21_NLE/\
  --corner corner_evidence_acc_example.pdf\
  --corner-no-legend --corner-include-true --legend "NLE-derived posterior"\
  --corner-true-legend "True posterior" --corner-simple-titles
cp "corner_evidence_acc_example.pdf" ${fig_out_dir}

# NPE vs NLE comparison for free gauss extra err
python -m dipolesbi.tools.posterior_cli fiducial_50k_NPE/20251024_204611_SEED0_NPE\
  fiducial_50k/20251011_155810_SEED0_NLE/\
  --corner "${fig_out_dir}corner_npe_vs_nle.pdf"\
  --corner-columns log10_n_initial_samples w1_extra_error observer_speed\
  dipole_longitude dipole_latitude\
  --legend "CatSIM fiducial model posterior (NPE)" "CatSIM fiducial model posterior (NLE)"\
  --corner-no-credible-lines --corner-stack-credible-titles
