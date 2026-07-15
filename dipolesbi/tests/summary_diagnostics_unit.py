import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from dipolesbi.pipelines.based_racs import _round_quantile_diagnostic_specs
from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.summary_diagnostics import (
    QuantileSummaryDiagnosticSpec,
    plot_round_quantile_diagnostics,
)


def test_round_quantile_specs_follow_enabled_summary_order():
    model = SimpleNamespace(
        tile_temperature_by_index=np.asarray([20.0, 30.0]),
        elevation_lookup_values=np.asarray([40.0, 60.0]),
    )

    specs = _round_quantile_diagnostic_specs(
        model,
        summary_features=[
            "flux_elevation_quantiles",
            "log_dispersion",
            "flux_quantiles",
        ],
        flux_temperature_n_bins=2,
        flux_temperature_quantiles=(0.25, 0.5, 0.75),
        flux_elevation_n_bins=2,
        flux_elevation_quantiles=(0.5,),
    )

    assert [spec.name for spec in specs] == [
        "flux_elevation_quantiles",
        "flux_temperature_quantiles",
    ]
    assert specs[0].start == 0
    assert specs[0].stop == 2
    assert specs[1].start == 3
    assert specs[1].stop == 9


def test_round_quantile_plot_uses_flattened_summary_slice(tmp_path):
    spec = QuantileSummaryDiagnosticSpec(
        name="flux_temperature_quantiles",
        start=3,
        bin_edges=(20.0, 25.0, 30.0),
        quantiles=(0.25, 0.5),
        x_label="Temperature [C]",
    )
    reference = np.asarray([0.0, 0.0, 0.0, 2.0, 4.0, 3.0, 6.0], dtype=np.float32)
    simulations = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 3.0, 2.0, 5.0],
            [0.0, 0.0, 0.0, 3.0, 5.0, 4.0, 7.0],
        ],
        dtype=np.float32,
    )
    masks = np.ones_like(simulations, dtype=bool)

    paths = plot_round_quantile_diagnostics(
        2,
        (simulations, masks),
        reference,
        np.ones_like(reference, dtype=bool),
        str(tmp_path),
        specs=(spec,),
    )

    assert len(paths) == 1
    assert paths[0].name == "round_02_flux_temperature_quantiles.png"
    assert paths[0].is_file()


def test_multiround_diagnostic_hook_is_optional_and_receives_round_data(tmp_path):
    inferer = object.__new__(MultiRoundInferer)
    inferer.reference_data = np.asarray([10.0, 1.0], dtype=np.float32)
    inferer.reference_mask = np.asarray([True, True])
    inferer.summary_start = 1
    inferer.summary_ndim = 1
    inferer.mr_config = SimpleNamespace(plot_save_dir=str(tmp_path))
    data = (
        np.asarray([[20.0, 2.0], [30.0, 3.0]], dtype=np.float32),
        np.asarray([[True, True], [True, True]]),
    )

    inferer.round_simulation_diagnostic = None
    assert inferer._prepare_round_simulation_diagnostic(data) is None
    inferer._maybe_run_round_simulation_diagnostic(0, None)

    received = {}

    def diagnostic(round_idx, round_data, reference, reference_mask, output_dir):
        received.update(
            round_idx=round_idx,
            data=round_data,
            reference=reference,
            reference_mask=reference_mask,
            output_dir=output_dir,
        )

    inferer.round_simulation_diagnostic = diagnostic
    diagnostic_data = inferer._prepare_round_simulation_diagnostic(data)
    inferer._maybe_run_round_simulation_diagnostic(3, diagnostic_data)

    assert received["round_idx"] == 3
    np.testing.assert_array_equal(received["data"][0], [[2.0], [3.0]])
    np.testing.assert_array_equal(received["reference"], [1.0])
    np.testing.assert_array_equal(received["reference_mask"], [True])
    assert received["output_dir"] == str(tmp_path)
