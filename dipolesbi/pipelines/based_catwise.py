"""Run the configured S21 CatWISE likelihood inference pipeline."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any, Mapping

import numpy as np
import yaml
from catsim import Catwise, CatwiseJax

from dipolesbi.pipelines.catwise_components import (
    build_catwise_config,
    build_prior,
    build_scenario,
    make_model_sim_wrapper,
)
from dipolesbi.pipelines.catwise_config import (
    CatwiseConfigError,
    load_catwise_run,
)
from dipolesbi.pipelines.prepare_catwise_observation import (
    build_native_retain_mask,
    load_prepared_observation,
    validate_catwise_inputs,
)
from dipolesbi.pipelines.validate_racs_data import DataValidationError
from dipolesbi.pipelines.catwise_diagnostics import (
    save_density_diagnostics,
    save_glass_diagnostics,
)
from dipolesbi.tools.multiround_inferer import MultiRoundInferer


def _copy_if_different(source: Path, destination: Path) -> None:
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def run_inference(
    run: Mapping[str, Any],
    prepared_observation: str | Path,
    output_dir: str | Path | None = None,
    *,
    command: list[str] | None = None,
    repository_root: str | Path | None = None,
    n_workers: int | None = None,
) -> None:
    """Validate the observation and simulator mask, then run NLE.

    The observation's native_mask.npy must be alongside its NPZ file. No
    simulator data or inference state is created until observation validation
    succeeds; no inference is started unless the simulator mask matches.
    """
    prepared = Path(prepared_observation).expanduser().resolve(strict=True)
    x0, mask, metadata = load_prepared_observation(run, prepared)
    report = validate_catwise_inputs(run)
    native_mask = build_native_retain_mask(run, report=report)
    from dipolesbi.pipelines.package_catwise_analysis import (
        capture_software_state,
    )

    software_state = capture_software_state(repository_root or Path.cwd())
    workers = run['execution']['threads'] if n_workers is None else n_workers
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 1
    ):
        raise ValueError(
            'NumPy simulation worker count must be a positive integer.'
        )
    software_state['numpy_simulation_workers'] = (
        workers if run['execution']['backend'] == 'numpy' else None
    )
    output = (
        Path(
            output_dir
            if output_dir is not None
            else run['inference']['multiround']['plot_save_dir']
        )
        .expanduser()
        .resolve()
    )
    resolved = deepcopy(run)
    resolved['inference']['multiround']['plot_save_dir'] = str(output)
    load_simulations = resolved['inference']['multiround'].get(
        'load_simulations'
    )
    resolved['inference']['multiround']['simulation_path'] = (
        str(output / load_simulations / 'data')
        if load_simulations is not None
        else None
    )

    config = build_catwise_config(resolved)
    backend = resolved['execution']['backend']
    if backend not in ('jax', 'numpy'):
        raise ValueError(f'Unsupported CatWISE backend: {backend}')
    model = (CatwiseJax if backend == 'jax' else Catwise)(config)
    model.initialise_data()
    simulator_mask = model.mask_map if backend == 'jax' else model.native_mask
    if simulator_mask is None or not np.array_equal(
        simulator_mask, native_mask
    ):
        raise DataValidationError(
            'Simulator native mask does not match the validated observation mask.'
        )
    prior = build_prior(resolved)
    scenario = build_scenario(resolved, prior, str(output))
    simulator = make_model_sim_wrapper(
        model, resolved, expected_native_mask=native_mask, n_workers=workers
    )
    if backend == 'numpy':
        print(
            f'NumPy simulation workers: {workers}; one BLAS/OpenMP thread per worker.',
            flush=True,
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / 'software-state.yaml').write_text(
        yaml.safe_dump(software_state, sort_keys=False), encoding='utf-8'
    )
    (output / 'experiment.resolved.yaml').write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding='utf-8'
    )
    (output / 'run_command.txt').write_text(
        (
            shlex.join(command)
            if command is not None
            else 'Programmatic invocation of run_inference'
        )
        + '\n',
        encoding='utf-8',
    )
    (output / 'validation-report.yaml').write_text(
        yaml.safe_dump(
            {
                'input_validation': report,
                'observation_metadata': metadata,
                'simulator_native_mask_matches': True,
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )
    _copy_if_different(prepared, output / 'prepared_observation.npz')
    _copy_if_different(
        prepared.parent / 'native_mask.npy', output / 'native_mask.npy'
    )
    native_observation = prepared.parent / 'reference_observation_native.npz'
    if native_observation.is_file():
        _copy_if_different(
            native_observation, output / native_observation.name
        )
    save_density_diagnostics(native_observation, output)
    if config.position_model == 'glass':
        save_glass_diagnostics(model._glass_model, output)
    inferer = MultiRoundInferer(
        'NLE',
        prior,
        simulator,
        (x0, mask),
        multi_round_config=scenario.multiround,
        transform_config=scenario.transforms,
        nflow_config=scenario.flow,
        train_config=scenario.training,
        model_config=config,
        use_ui=not resolved['execution']['no_ui'],
    )
    inferer.run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--reference-observation', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument(
        '--repository-root',
        type=Path,
        default=Path.cwd(),
        help='Source checkout containing pyproject.toml, uv.lock and Git metadata.',
    )
    parser.add_argument('--no-ui', action='store_true')
    parser.add_argument(
        '--n-workers',
        type=int,
        help='NumPy simulation processes; defaults to execution.threads.',
    )
    args = parser.parse_args(argv)
    try:
        run = load_catwise_run(args.experiment, args.site)
        if args.no_ui:
            run['execution']['no_ui'] = True
        command = (
            sys.orig_argv
            if argv is None
            else [sys.executable, '-m', __name__, *argv]
        )
        run_inference(
            run,
            args.reference_observation,
            args.out_dir,
            command=command,
            repository_root=args.repository_root,
            n_workers=args.n_workers,
        )
    except (
        CatwiseConfigError,
        DataValidationError,
        FileNotFoundError,
        KeyError,
        ValueError,
        ImportError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
