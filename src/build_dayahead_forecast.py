"""
Phase 3e - Construction of the day-ahead forecast signal (S2).

The published forecast (S1) carries no stated vintage, so its accuracy may flatter a genuine
day-ahead decision. S2 is a forecast this study builds and controls: every feature is drawn
from information available at least 24 hours before the period being predicted, so a
scheduler using it could have acted on it a day in advance.

Leakage is prevented in three ways:
  - every lag is at least 48 half-hourly periods (24 hours);
  - rolling statistics end at least 24 hours before the target period;
  - the model is fitted on the design period only (months 1 to 24) and never sees the
    held-out period during training.

The published forecast is deliberately excluded from the feature set. Including it would
make S2 a correction of S1 rather than an independent day-ahead signal, and would inherit
whatever vintage advantage S1 enjoys.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs")

DESIGN_END = pd.Timestamp("2025-09-01", tz="UTC")
H = 48          # half-hourly periods in 24 hours; the minimum permissible lag


def build_features(d):
    x = pd.DataFrame(index=d.index)
    a = d["actual"]
    for lag in (H, H + 1, H + 2, 2 * H, 3 * H, 7 * H, 14 * H):
        x["lag_%d" % lag] = a.shift(lag)
    for win in (H, 7 * H):                     # windows that end 24 hours back
        x["roll_mean_%d" % win] = a.shift(H).rolling(win).mean()
        x["roll_std_%d" % win] = a.shift(H).rolling(win).std()
    x["roll_min_48"] = a.shift(H).rolling(H).min()
    x["roll_max_48"] = a.shift(H).rolling(H).max()
    t = d["from"]
    x["hour"] = t.dt.hour + t.dt.minute / 60.0
    x["dow"] = t.dt.dayofweek
    x["month"] = t.dt.month
    x["doy_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365.25)
    x["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
    return x


def main():
    t0 = time.time()
    d = pd.read_csv(os.path.join(PROC, "national_intensity.csv"), parse_dates=["from"])
    d = d.dropna(subset=["actual"]).sort_values("from").reset_index(drop=True)

    x = build_features(d)
    y = d["actual"]
    ok = x.notna().all(axis=1)
    design = ok & (d["from"] < DESIGN_END)

    # Model selection uses a chronological split inside the design period: the last six
    # months are held back for validation, so complexity is chosen without any contact with
    # the held-out period and without the temporal leakage a random split would introduce.
    val_start = DESIGN_END - pd.Timedelta(days=182)
    tr = design & (d["from"] < val_start)
    va = design & (d["from"] >= val_start)
    grid = [dict(max_iter=m, learning_rate=lr, max_leaf_nodes=ln, min_samples_leaf=ms,
                 l2_regularization=l2)
            for m, lr, ln, ms, l2 in [(120, 0.05, 15, 200, 5.0),
                                      (250, 0.05, 31, 100, 1.0),
                                      (400, 0.06, 63, 40, 1.0),
                                      (80, 0.08, 8, 400, 10.0)]]
    best, best_mae = None, np.inf
    for cfg in grid:
        m = HistGradientBoostingRegressor(random_state=20260913, **cfg).fit(x[tr], y[tr])
        mae = mean_absolute_error(y[va], m.predict(x[va]))
        print("  candidate %s -> validation MAE %.2f" % (cfg, mae), flush=True)
        if mae < best_mae:
            best, best_mae = cfg, mae
    print("selected %s (validation MAE %.2f)" % (best, best_mae), flush=True)
    model = HistGradientBoostingRegressor(random_state=20260913, **best)
    model.fit(x[design], y[design])
    pred = pd.Series(np.nan, index=d.index)
    pred[ok] = model.predict(x[ok])

    d["forecast_s2"] = pred
    d = d.rename(columns={"forecast": "forecast_published"})
    d[["from", "actual", "forecast_published", "forecast_s2"]].to_csv(
        os.path.join(PROC, "national_intensity_signals.csv"), index=False)

    rows = []
    for name, col in (("S1 published", "forecast_published"), ("S2 day-ahead", "forecast_s2")):
        for split, mask in (("design", d["from"] < DESIGN_END), ("held-out", d["from"] >= DESIGN_END)):
            m = mask & d[col].notna() & d["actual"].notna()
            e = d.loc[m, col] - d.loc[m, "actual"]
            rows.append({"signal": name, "split": split, "n": int(m.sum()),
                         "MAE": mean_absolute_error(d.loc[m, "actual"], d.loc[m, col]),
                         "RMSE": float(np.sqrt((e ** 2).mean())),
                         "bias_ME": float(e.mean()),
                         "corr": float(np.corrcoef(d.loc[m, col], d.loc[m, "actual"])[0, 1])})
    # Benchmark: 24-hour persistence, the naive day-ahead alternative.
    per = d["actual"].shift(H)
    for split, mask in (("design", d["from"] < DESIGN_END), ("held-out", d["from"] >= DESIGN_END)):
        m = mask & per.notna() & d["actual"].notna()
        e = per[m] - d.loc[m, "actual"]
        rows.append({"signal": "24 h persistence", "split": split, "n": int(m.sum()),
                     "MAE": float(e.abs().mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
                     "bias_ME": float(e.mean()),
                     "corr": float(np.corrcoef(per[m], d.loc[m, "actual"])[0, 1])})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "phase3e_signal_accuracy.csv"), index=False)
    print(res.round(3).to_string(index=False))
    print("\nelapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
