#!/usr/bin/env python
"""
Value of demand response per shifted MWh, and its distribution.

For every DR set (one bus x one device group) in a solved network:
    value   = sum_t  price_t * (L_down_t - L_up_t)      [EUR]
    shifted = sum_t  L_down_t                          [MWh]  (one direction)
    value per MWh = value / shifted                    [EUR/MWh]

price_t is the nodal electricity price at the DR set's bus. L_down is load
moved away from hour t, L_up is load moved into hour t, so the value is the
price arbitrage DR earns at that bus. Summed over all sets it approximates the
system saving (not exactly, since DR also changes the prices themselves).

Usage:
    pixi run python thesis-tools/dr_value_per_mwh.py results/<run>/networks/solved_2025.nc
Writes thesis-tools/dr_value_per_mwh_<run>.csv with one row per DR set.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


def wquantile(x, w, qs):
    """Weighted quantiles (weight = shifted energy)."""
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, cw, x)


fn = sys.argv[1]
run = Path(fn).parents[1].name
n = pypsa.Network(fn)
w = n.snapshot_weightings.generators
price = n.buses_t.marginal_price

links = n.links[n.links.carrier.str.startswith("DR")]
up = links[links.carrier.str.endswith(" up")]
rows = []
for name, l in up.iterrows():
    bus = l.bus0
    group = l.carrier[len("DR "):-len(" up")]
    down_name = name[: -len(" up")] + " down"
    if down_name not in n.links.index:
        continue
    p_up = n.links_t.p0[name].mul(w, axis=0)
    p_down = n.links_t.p0[down_name].mul(w, axis=0)
    shifted = p_down.sum()
    if shifted < 1:  # ignore sets that shift less than 1 MWh per year
        continue
    value = (price[bus] * (p_down - p_up)).sum()
    rows.append({
        "bus": bus, "country": bus[:2], "group": group,
        "shifted_MWh": shifted, "value_EUR": value,
        "value_per_MWh": value / shifted,
    })

df = pd.DataFrame(rows)
out = Path("thesis-tools") / f"dr_value_per_mwh_{run}.csv"
df.to_csv(out, index=False)

qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
labels = ["P5", "P10", "P25", "median", "P75", "P90", "P95"]


def summary(d):
    v, e = d.value_per_MWh, d.shifted_MWh
    s = {
        "sets": len(d),
        "shifted_TWh": e.sum() / 1e6,
        "value_MEUR": d.value_EUR.sum() / 1e6,
        "mean (energy-weighted)": d.value_EUR.sum() / e.sum(),
        "mean (unweighted)": v.mean(),
    }
    s.update(dict(zip(labels, wquantile(v, e, qs))))
    return pd.Series(s)


pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

print(f"\n=== DR value per shifted MWh: {run} ===")
print("Percentiles are energy-weighted (each shifted MWh counts once).\n")
print("ALL DR SETS")
print(summary(df).to_string())
print("\nBY DEVICE GROUP")
print(df.groupby("group").apply(summary).to_string())
print("\nBY COUNTRY (10 largest by shifted energy)")
top = df.groupby("country").shifted_MWh.sum().nlargest(10).index
print(df[df.country.isin(top)].groupby("country").apply(summary).to_string())
print(f"\nPer-set results saved to {out}")