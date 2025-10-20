#!/usr/bin/env bash

python -m dipolesbi.scripts.paper_covmap --save_plots
python -m dipolesbi.scripts.catwise_alpha --save
python -m dipolesbi.scripts.evidence_acc.py --save_dir nside_64_coarse4_funnel_v2 --smoothing_window 2 --skip_delta --save_plot
