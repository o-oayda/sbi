from __future__ import annotations

import math
import warnings

import numpy as np

from getdist import plots


class PaperPlotter(plots.GetDistPlotter):
    """
    Minimal custom style for dipolesbi plots.

    Tweaks over the default style:

    * LaTeX-rendered text using Computer Modern.
    * Thicker 1D density lines.
    * Bolder 2D contour outlines.
    * Titles show the posterior median with 68% credible intervals.
    * Dashed guides mark the median and the 68% credible interval on each 1D marginal.
    """

    _style_rc = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Boost 1D marginal line width slightly over the default of 1.0.
        self.settings.linewidth = 2.0

        # Thicken 2D contour outlines for better visibility.
        self.settings.linewidth_contour = 2.0

        # Show median ±1σ above each 1D panel.
        self.settings.title_limit = 1
        self.settings.title_limit_labels = True
        self.settings.title_limit_fontsize = 1. * self.settings.axes_labelsize

    @staticmethod
    def _weighted_quantiles(values: np.ndarray, weights: np.ndarray | None, quantiles: list[float]) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64)
        qs = np.asarray(quantiles, dtype=np.float64)
        if data.size == 0:
            return np.full_like(qs, np.nan, dtype=np.float64)

        if weights is None:
            return np.quantile(data, qs)

        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != data.size:
            return np.quantile(data, qs)

        mask = np.isfinite(data) & np.isfinite(w) & (w > 0)
        if not np.any(mask):
            return np.quantile(data, qs)

        data = data[mask]
        w = w[mask]
        sorter = np.argsort(data)
        data = data[sorter]
        w = w[sorter]
        cumulative = np.cumsum(w) - 0.5 * w
        total = cumulative[-1] + 0.5 * w[-1]
        if total <= 0:
            return np.quantile(data, qs)
        cumulative /= total
        return np.interp(qs, cumulative, data, left=data[0], right=data[-1])

    @staticmethod
    def _round_to_sig(value: float, sig: int) -> tuple[float, int]:
        """Round ``value`` to ``sig`` significant figures.

        Returns the rounded float and the number of decimal places used so that
        later formatting can align different uncertainties to the same precision.
        """

        if value == 0:
            return 0.0, 0
        exponent = int(math.floor(math.log10(abs(value))))
        round_digits = sig - 1 - exponent
        rounded = round(value, round_digits)
        decimals = max(round_digits, 0)
        return rounded, decimals

    @staticmethod
    def _format_fixed(value: float, decimals: int) -> str:
        """Format ``value`` with a fixed number of decimal places."""

        if decimals <= 0:
            return f"{int(round(value))}"
        return f"{value:.{decimals}f}"

    def _format_interval(self, median: float, lower: float, upper: float) -> str:
        """Return LaTeX string summarising the median and 68% credible interval.

        Rules implemented:
        1. Upper and lower errors are rounded to two significant figures.
        2. The number of decimal places shown for the two errors is matched so
           they share the coarsest precision (e.g. ``+1.1`` and ``-0.72`` → ``±0.7``).
        3. The median is rounded to the same decimal precision as the errors.
        """

        plus = abs(upper - median)
        minus = abs(median - lower)

        plus_val, plus_dec = self._round_to_sig(plus, 2)
        minus_val, minus_dec = self._round_to_sig(minus, 2)

        common_dec = min(plus_dec, minus_dec)
        plus_val = round(plus_val, common_dec)
        minus_val = round(minus_val, common_dec)
        median_val = round(median, common_dec)

        plus_str = self._format_fixed(plus_val, common_dec)
        minus_str = self._format_fixed(minus_val, common_dec)
        median_str = self._format_fixed(median_val, common_dec)

        return rf"{median_str}^{{+{plus_str}}}_{{-{minus_str}}}"

    def add_1d(self, root, param, plotno=0, normalized=None, ax=None, title_limit=None, **kwargs):
        param_info = self._check_param(root, param)
        result = super().add_1d(root, param_info, plotno=plotno, normalized=normalized, ax=ax, title_limit=title_limit, **kwargs)

        try:
            samples = self.sample_analyser.samples_for_root(root)
        except Exception:
            return result

        if samples is None or not hasattr(samples, "getMargeStats"):
            return result

        stats = samples.getMargeStats()
        if stats is None or stats.parWithName(param_info.name) is None:
            return result

        try:
            param_index = samples.paramNames.numberOfName(param_info.name)
        except Exception:
            return result

        if param_index is None or param_index < 0:
            return result

        sample_array = getattr(samples, "samples", None)
        if sample_array is None:
            return result

        try:
            param_samples = np.asarray(sample_array[:, param_index], dtype=np.float64)
        except Exception:
            return result

        weights = getattr(samples, "weights", None)
        q_lower, q_median, q_upper = self._weighted_quantiles(
            param_samples, weights, [0.158655254, 0.5, 0.841344746]
        )

        if not (np.isfinite(q_lower) and np.isfinite(q_median) and np.isfinite(q_upper)):
            warnings.warn(
                f"paperplot: unable to determine median/credible interval for '{param_info.name}', skipping markers."
            )
            return result

        axis = self.get_axes(ax, pars=(param_info,))
        line_kwargs = self.lines_added.get(plotno, {}) if hasattr(self, "lines_added") else {}
        color = line_kwargs.get("color") or self.settings.axis_marker_color
        width = self._scaled_linewidth(self.settings.linewidth_contour)

        axis.axvline(q_median, color=color, ls="--", lw=width, alpha=0.8)
        axis.axvline(q_lower, color=color, ls="--", lw=width, alpha=0.6)
        axis.axvline(q_upper, color=color, ls="--", lw=width, alpha=0.6)

        label = param_info.label or param_info.name
        axis.set_title(
            rf"${label} = {self._format_interval(q_median, q_lower, q_upper)}$",
            fontsize=self._scaled_fontsize(self.settings.title_limit_fontsize, self.settings.axes_fontsize),
        )

        return result


style_name = "paperplot"

# Register the style; callers can activate via plots.set_active_style(style_name).
plots.add_plotter_style(style_name, PaperPlotter)
