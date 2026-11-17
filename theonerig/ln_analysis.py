"""
Linear-nonlinear (LN) analysis of fullfield flicker responses, and the comparison of
paired records (untreated vs treated) built on it.

Python port of Hiroki Asari's MATLAB scripts (folder LNanalysis_example_HIROKI, May 2024):
    - LNanalysis.m         -> `ln_analysis_fff`, `ln_analysis_fff_reM`
    - PairedDataAnalysis.m -> `parse_paired_record_name`, `paired_nonlinearity_gain`,
                              `sta_polarity`, `sta_peak_latency`, `sta_peak_frequency`,
                              `sign_test`, `paired_data_analysis`, `plot_paired_*`
    - GetReMdata.m and GetReMdataIndex.m are replaced by theonerig's `import_record`
      and `Data_Pipe`.

`ln_analysis_fff` returns the fields of the .mat files written by LNanalysis.m, so
`scipy.io.savemat(path, result)` writes a file with the same content.

Choices made to reproduce the MATLAB behaviour:
    - the std of the null STA values uses ddof=1, like MATLAB's std;
    - nonlinearity bins are half-open, [edge_j, edge_j+1), like the MATLAB loop. The
      linear prediction is rounded to 12 decimals before binning, so that a value lying
      exactly on an edge always goes to the upper bin. With a binary stimulus the
      prediction is an integer divided by the spike count and such ties do occur; in
      MATLAB their bin depends on floating-point rounding. On Hiroki's example records
      this port and his .mat files differ only in bins next to an edge with ties;
      everything else (STA, null parameters, p-values, quality) matches to rounding error;
    - percentiles follow MATLAB's prctile definition (`_prctile`);
    - splines are not-a-knot cubic splines, like MATLAB's spline.

Differences from the MATLAB scripts:
    - PairedDataAnalysis.m kept only the last pair of records in its STA and nonlinearity
      arrays; here each pair is analysed by its own call, and results can be stacked.
    - The condition codes of PairedDataAnalysis.m are not built (they were not used in the
      analysis); `parse_paired_record_name` returns the name fields as strings.
    - `sign_test` uses the exact binomial test by default. MATLAB's signtest switches to a
      normal approximation for large samples, available here with method="approximate".
    - The frequency scatter is labelled in Hz (the MATLAB figure said [s]).
"""

__all__ = ['ln_analysis_fff', 'ln_analysis_fff_reM', 'parse_paired_record_name',
           'paired_nonlinearity_gain', 'sta_polarity', 'sta_peak_latency', 'sta_peak_frequency',
           'sign_test', 'paired_data_analysis', 'plot_paired_nonlinearity', 'plot_paired_gain',
           'plot_paired_scatter']

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from scipy.interpolate import CubicSpline

from .core import Data_Pipe


def _default_x_edges():
    """Bin edges of LNanalysis.m: X = -3:.1:3; X = X(1:end-1)+.05;"""
    return np.round(np.arange(-30, 30) * 0.1 + 0.05, 10)


def _bin_centers(x_edges):
    return (np.asarray(x_edges[:-1]) + np.asarray(x_edges[1:])) / 2


def _prctile(a, p, axis=0):
    """
    Percentile p of a along axis, as MATLAB's prctile: the n sorted values sit at the
    percentiles 100*(i-0.5)/n, with linear interpolation in between and the minimum or
    maximum outside. NaNs are ignored.
    """
    a = np.moveaxis(np.asarray(a, dtype=float), axis, 0)
    out = np.full(a.shape[1:], np.nan)
    for idx in np.ndindex(*a.shape[1:]):
        x = np.sort(a[(slice(None),) + idx])
        x = x[~np.isnan(x)]
        if len(x):
            pos = 100 * (np.arange(1, len(x) + 1) - 0.5) / len(x)
            out[idx] = np.interp(p, pos, x)
    return out


def ln_analysis_fff(stim_inten, spike_counts, dt=1/60, n_filter=30, q_threshold=10,
                    null_params=None, x_edges=None, stim_max=255):
    """
    STA, quality check and static nonlinearity of fullfield flicker responses
    (port of LNanalysis.m by Hiroki Asari).

    params:
        - stim_inten: Stimulus intensity of shape (t,), from 0 (black) to stim_max (white)
        - spike_counts: Spike counts of shape (t, n_cell), or (t,) for a single cell
        - dt: Frame length in seconds
        - n_filter: STA length in frames
        - q_threshold: A cell passes the quality check when -log10(p) of any of its STA
                       values exceeds this threshold
        - null_params: (n_repeat, first_shift, shift_step) of the null distribution, made of
                       STA values at large time shifts. Default (100, 5*n_filter, 3*n_filter),
                       as in LNanalysis.m
        - x_edges: Bin edges of the linear prediction for the nonlinearity. Default: the
                   edges of LNanalysis.m
        - stim_max: Value of the white frames; the stimulus is mapped to [-1, 1] with
                    2*stim/stim_max - 1. Use None if stim_inten is already in [-1, 1]

    return: dict with the fields of the .mat files written by LNanalysis.m
        - sta: (n_cell, n_filter) STA; column k is the stimulus k frames before the spike
        - t: (n_filter,) time of each STA column in seconds (0, -dt, -2*dt, ...)
        - q_param: (n_cell, 2) mean and standard deviation of the null STA values
        - pval: (n_cell, n_filter) -log10 of the one-tailed normal p-value of each STA value
        - quality: (n_cell,) True when any pval of the cell exceeds q_threshold
        - nl: (n_cell, n_bin) mean firing rate in Hz in each bin of the linear prediction,
              NaN for empty bins
        - X: the bin edges; dt, Nfilter, Ncell, qth, param, Nx, Ny: the parameters used

    Notes: spike trains are normalised by their total spike count before averaging, and
    the linear prediction uses the STA as it is (not normalised). The comment in
    LNanalysis.m calls q_threshold "10 std", but its code thresholds -log10(p), as here.
    """
    stim = np.asarray(stim_inten, dtype=float).ravel()
    resp = np.asarray(spike_counts, dtype=float)
    if resp.ndim == 1:
        resp = resp[:, None]
    if len(stim) != len(resp):
        raise ValueError("stim_inten and spike_counts must have the same length")
    resp = resp.T  # (n_cell, t), the orientation of the MATLAB code
    n_cell, n_t = resp.shape
    if stim_max is not None:
        stim = 2 * stim / stim_max - 1  # +1 white, -1 black
    if null_params is None:
        null_params = (100, 5 * n_filter, 3 * n_filter)
    n_null, first_shift, shift_step = null_params
    x_edges = _default_x_edges() if x_edges is None else np.asarray(x_edges, dtype=float)

    # Spike-triggered average on zero-padded signals
    stim_pad = np.concatenate([np.zeros(n_filter), stim])
    resp_pad = np.concatenate([np.zeros((n_cell, n_filter)), resp], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = resp_pad / resp_pad.sum(axis=1, keepdims=True)
    sta = np.empty((n_cell, n_filter))
    for k in range(n_filter):
        sta[:, k] = r @ stim_pad
        r = np.roll(r, -1, axis=1)  # shift the response back one frame

    # Null distribution: STA values at large circular time shifts
    null = np.empty((n_cell, n_null))
    r = np.roll(r, -first_shift, axis=1)
    for k in range(n_null):
        null[:, k] = r @ stim_pad
        r = np.roll(r, -shift_step, axis=1)
    q_param = np.column_stack([null.mean(axis=1), null.std(axis=1, ddof=1)])

    # Static nonlinearity: mean response in bins of the linear prediction
    rhat = np.zeros((n_cell, n_t))
    for lag in range(n_filter):
        rhat[:, lag:] += sta[:, lag:lag + 1] * stim[:n_t - lag]
    # Rounding before binning puts values lying exactly on an edge in the upper bin,
    # whatever the floating-point rounding of the sum. Such ties happen with a binary
    # stimulus, where rhat is an integer divided by the cell's spike count.
    rhat = np.round(rhat, 12)
    n_bin = len(x_edges) - 1
    nl = np.full((n_cell, n_bin), np.nan)
    for k in range(n_cell):
        b = np.searchsorted(x_edges, rhat[k], side="right") - 1  # edge_b <= rhat < edge_b+1
        ok = (b >= 0) & (b < n_bin)
        count = np.bincount(b[ok], minlength=n_bin)
        total = np.bincount(b[ok], weights=resp[k, ok], minlength=n_bin)
        filled = count > 0
        nl[k, filled] = total[filled] / count[filled] / dt

    # Quality check: -log10 p-value of each STA value under the null distribution
    with np.errstate(divide="ignore", invalid="ignore"):
        pval = norm.cdf(sta, loc=q_param[:, :1], scale=q_param[:, 1:])
        pval = np.where(pval > .5, 1 - pval, pval)
        pval = -np.log10(pval)
    quality = np.any(pval > q_threshold, axis=1)

    return {"sta": sta, "t": -np.arange(n_filter) * dt, "q_param": q_param, "pval": pval,
            "quality": quality, "nl": nl, "X": x_edges, "dt": dt, "Nfilter": n_filter,
            "Ncell": n_cell, "qth": q_threshold, "param": np.array(null_params), "Nx": 1, "Ny": 1}


def ln_analysis_fff_reM(reM, stim_name="fullfield_flicker", **kwargs):
    """
    Run `ln_analysis_fff` on the frames of a record where both stim_name and the
    S_matrix are present, all such chunks concatenated in time order (as LNanalysis.m).

    params:
        - reM: RecordMaster of the record
        - stim_name: Name of the stimulus DataChunk
        - **kwargs: Parameters of `ln_analysis_fff`

    return:
        - The dictionary of `ln_analysis_fff`, plus "cellidx": the cluster id of each row
    """
    pipe = Data_Pipe(reM, [stim_name, "S_matrix"], ["stim_inten", "spike_counts"])
    pipe += stim_name
    pipe &= "S_matrix"
    if len(pipe) == 0:
        raise ValueError("No frame of the record has both " + stim_name + " and S_matrix")
    chunks = [pipe[i] for i in range(len(pipe))]
    stim = np.concatenate([np.asarray(c["stim_inten"]) for c in chunks])
    spikes = np.concatenate([np.asarray(c["spike_counts"]) for c in chunks])
    result = ln_analysis_fff(stim, spikes, **kwargs)
    cell_map = chunks[0]["spike_counts"].attrs["cell_map"]
    result["cellidx"] = np.array([int(c) for c in sorted(cell_map, key=cell_map.get)])
    return result


def parse_paired_record_name(filename):
    """
    Parse the name of a paired record the way PairedDataAnalysis.m does, e.g.
    "reM_M116_paired(M115)_KILOSORT_2024-03-15_awake_LGN_ciproxifan_day.h5".

    return: dict with
        - idx: id of the record ("M116"); pair_idx: id of the paired record ("M115")
        - date; anesthesia ("awake" or the anesthesia); drug ("untreated" or the drug)
        - circadian ("day" or "night"); tokens: all the parts of the name
    When a token sits between the anesthesia and the drug ("LGN" above), the drug is the
    token after it, as in PairedDataAnalysis.m; the extra token is kept in "tokens".
    """
    stem = Path(filename).name
    if stem.endswith(".h5"):
        stem = stem[:-3]
    tokens = stem.split("_")
    info = {"idx": tokens[1], "pair_idx": tokens[2][len("paired("):-1], "date": tokens[4],
            "anesthesia": tokens[5], "drug": tokens[6], "circadian": "_".join(tokens[7:]),
            "tokens": tokens}
    if "_" in info["circadian"]:
        info["drug"], info["circadian"] = info["circadian"].split("_", 1)
    return info


def paired_nonlinearity_gain(res_untreated, res_treated):
    """
    Response gain between the nonlinearities of two paired records, for the cells that
    pass the quality check in both (PairedDataAnalysis.m).

    params:
        - res_untreated, res_treated: outputs of `ln_analysis_fff` for the two records,
          with the same cells in the same order

    return:
        - gain: (n_good, n_bin) nl_treated / nl_untreated, with inf set to NaN
        - good: (n_cell,) mask of the cells good in both records
    """
    if "cellidx" in res_untreated and "cellidx" in res_treated:
        if not np.array_equal(np.ravel(res_untreated["cellidx"]), np.ravel(res_treated["cellidx"])):
            raise ValueError("The two records do not have the same cells in the same order")
    good = np.asarray(res_untreated["quality"], bool) & np.asarray(res_treated["quality"], bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = res_treated["nl"][good] / res_untreated["nl"][good]
    gain[gain == np.inf] = np.nan
    return gain, good


def sta_polarity(stas, qualities):
    """
    Polarity of each cell from the mean of its good STAs across conditions, as in
    PairedDataAnalysis.m: ON when the maximum exceeds the absolute minimum.

    params:
        - stas: list of (n_cell, n_filter) STAs, one per condition
        - qualities: list of (n_cell,) quality masks, one per condition

    return:
        - (n_cell,) array: 1 for ON, 0 for OFF, NaN when no condition has a good STA
    """
    stas = np.asarray(stas, dtype=float)
    good = np.asarray(qualities, bool)
    polarity = np.full(stas.shape[1], np.nan)
    for k in range(stas.shape[1]):
        if good[:, k].any():
            mean_sta = stas[good[:, k], k].mean(axis=0)
            polarity[k] = float(mean_sta.max() > abs(mean_sta.min()))
    return polarity


def sta_peak_latency(sta, t, polarity, t_min=-0.3, upsample_dt=0.001):
    """
    Latency of the STA peak (ON cells) or trough (OFF cells), found on a cubic spline of
    the STA upsampled to upsample_dt, at times later than t_min (PairedDataAnalysis.m).

    params:
        - sta: (n_cell, n_filter) STA
        - t: (n_filter,) time of each STA column (0, -dt, ...)
        - polarity: (n_cell,) 1 for ON, 0 for OFF, NaN to skip the cell
        - t_min: Earliest time searched, in seconds
        - upsample_dt: Step of the upsampled time axis, in seconds

    return:
        - (n_cell,) latency in seconds (positive), NaN for skipped cells
    """
    sta = np.atleast_2d(np.asarray(sta, dtype=float))
    t = np.asarray(t, dtype=float)
    n_up = int(np.floor((t[0] - t[-1]) / upsample_dt + 1e-9)) + 1
    T = t[0] - upsample_dt * np.arange(n_up)
    order = np.argsort(t)
    window = T > t_min
    latency = np.full(len(sta), np.nan)
    for k, pol in enumerate(np.asarray(polarity, dtype=float)):
        if np.isnan(pol):
            continue
        y = CubicSpline(t[order], sta[k][order])(T)[window]
        latency[k] = abs(T[window][np.argmax(y) if pol else np.argmin(y)])
    return latency


def sta_peak_frequency(sta, dt, df=0.1):
    """
    Peak frequency of the amplitude spectrum of each STA, found on a cubic spline of the
    spectrum upsampled to df (PairedDataAnalysis.m).

    params:
        - sta: (n_cell, n_filter) STA
        - dt: Frame length in seconds
        - df: Step of the upsampled frequency axis, in Hz

    return:
        - (n_cell,) peak frequency in Hz
    """
    sta = np.atleast_2d(np.asarray(sta, dtype=float))
    n = sta.shape[1]
    f = np.arange(n // 2 + 1) / (n * dt)
    n_up = int(np.floor((f[-1] - f[0]) / df + 1e-9)) + 1
    F = f[0] + df * np.arange(n_up)
    peak = np.full(len(sta), np.nan)
    for k, y in enumerate(sta):
        a = np.abs(np.fft.fft(y - y.mean())) / n
        a = a[:len(f)]
        a[1:-1] = 2 * a[1:-1]
        a_up = CubicSpline(f, a)(F)
        peak[k] = F[a_up == a_up.max()].mean()
    return peak


def sign_test(x, y, method="exact"):
    """
    Two-sided sign test of paired samples (MATLAB's signtest). NaN pairs and zero
    differences are ignored.

    params:
        - x, y: Paired samples
        - method: "exact" (binomial) or "approximate" (normal approximation with
          continuity correction, which MATLAB's signtest uses for large samples)

    return:
        - p-value, NaN when no pair differs
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[~np.isnan(d)]
    n_pos, n_neg = int(np.sum(d > 0)), int(np.sum(d < 0))
    n = n_pos + n_neg
    if n == 0:
        return np.nan
    if method == "exact":
        return min(1.0, 2 * binom.cdf(min(n_pos, n_neg), n, 0.5))
    if method == "approximate":
        z = (n_pos - n_neg - np.sign(n_pos - n_neg)) / np.sqrt(n)
        return 2 * norm.cdf(-abs(z))
    raise ValueError("method must be 'exact' or 'approximate'")


def paired_data_analysis(res_untreated, res_treated, t_min=-0.3, upsample_dt=0.001, df=0.1):
    """
    Compare two paired records as PairedDataAnalysis.m: nonlinearity gain, and STA peak
    latency and peak frequency of each cell, with sign tests on the cells that pass the
    quality check in both records.

    params:
        - res_untreated, res_treated: outputs of `ln_analysis_fff` (or `ln_analysis_fff_reM`)
          for the two records, with the same cells in the same order
        - t_min, upsample_dt: see `sta_peak_latency`
        - df: see `sta_peak_frequency`

    return: dict with
        - good: (n_cell,) cells good in both records
        - gain: nonlinearity gain of the good cells (see `paired_nonlinearity_gain`)
        - polarity: (n_cell,) 1 ON, 0 OFF, NaN when neither STA is good (see `sta_polarity`)
        - latency: (2, n_cell) peak latency in s, untreated then treated; NaN where the STA
          fails the quality check
        - frequency: (2, n_cell) peak frequency in Hz, same layout
        - p_latency, p_frequency: sign test p-values over the good cells
    """
    results = (res_untreated, res_treated)
    gain, good = paired_nonlinearity_gain(res_untreated, res_treated)
    polarity = sta_polarity([r["sta"] for r in results], [r["quality"] for r in results])
    latency = np.full((2, len(good)), np.nan)
    frequency = np.full((2, len(good)), np.nan)
    for i, r in enumerate(results):
        q = np.asarray(r["quality"], bool)
        latency[i] = sta_peak_latency(r["sta"], r["t"], np.where(q, polarity, np.nan), t_min, upsample_dt)
        if q.any():
            frequency[i, q] = sta_peak_frequency(np.asarray(r["sta"])[q], float(r["dt"]), df)
    return {"good": good, "gain": gain, "polarity": polarity, "latency": latency,
            "frequency": frequency,
            "p_latency": sign_test(latency[0, good], latency[1, good]),
            "p_frequency": sign_test(frequency[0, good], frequency[1, good])}


def plot_paired_nonlinearity(res_untreated, res_treated, ax=None):
    """
    Median and interquartile range of the nonlinearities of the cells good in both
    records: untreated in blue, treated in red (PairedDataAnalysis.m).

    return:
        - The axis of the plot
    """
    if ax is None:
        fig, ax = plt.subplots()
    _, good = paired_nonlinearity_gain(res_untreated, res_treated)
    x = _bin_centers(res_untreated["X"])
    for res, color in ((res_untreated, "b"), (res_treated, "r")):
        nl = np.asarray(res["nl"])[good]
        ax.fill_between(x, _prctile(nl, 25), _prctile(nl, 75), color=color, alpha=.2, linewidth=0)
        ax.plot(x, _prctile(nl, 50), color=color, linewidth=2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 100)
    ax.set_xlabel("STA output")
    ax.set_ylabel("Firing rate [Hz]")
    ax.set_title("untreated, blue; treated, red")
    return ax


def plot_paired_gain(gain, x_edges=None, ax=None):
    """
    Median and interquartile range of the response gain (`paired_nonlinearity_gain`)
    along the linear filter output (PairedDataAnalysis.m).

    params:
        - gain: (n_good, n_bin) response gain
        - x_edges: Bin edges of the nonlinearity. Default: the edges of LNanalysis.m
        - ax: The axis for the plot. If None, a new plot is created

    return:
        - The axis of the plot
    """
    if ax is None:
        fig, ax = plt.subplots()
    x = _bin_centers(_default_x_edges() if x_edges is None else x_edges)
    ax.plot([-2, 2], [1, 1], "k--", linewidth=1)
    ax.fill_between(x, _prctile(gain, 25), _prctile(gain, 75), color="g", alpha=.2, linewidth=0)
    ax.plot(x, _prctile(gain, 50), color="g", linewidth=3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks(range(-2, 3))
    ax.set_yticks(range(0, 3))
    ax.set_xlabel("linear filter output")
    ax.set_ylabel("response gain")
    return ax


def plot_paired_scatter(values, good, label, unit, lim=None, ax=None):
    """
    Scatter of a per-cell value, untreated against treated, for the cells good in both
    records, with the sign test p-value in the title (PairedDataAnalysis.m).

    params:
        - values: (2, n_cell) values, e.g. the "latency" or "frequency" of `paired_data_analysis`
        - good: (n_cell,) cells to plot
        - label, unit: Name and unit of the value, for the axis labels
        - lim: Upper end of the identity line. Default: the largest value plotted
        - ax: The axis for the plot. If None, a new plot is created

    return:
        - The axis of the plot
    """
    if ax is None:
        fig, ax = plt.subplots()
    x, y = np.asarray(values)[0, good], np.asarray(values)[1, good]
    if lim is None:
        lim = np.nanmax(np.concatenate([x, y])) if len(x) else 1
    ax.plot(x, y, "o", linewidth=2)
    ax.plot([0, lim], [0, lim], "k:", linewidth=1)
    ax.set_aspect("equal")
    ax.set_xlabel(label + ", untreated [" + unit + "]")
    ax.set_ylabel(label + ", drug [" + unit + "]")
    ax.set_title("p=" + format(sign_test(x, y), ".4g"))
    return ax
