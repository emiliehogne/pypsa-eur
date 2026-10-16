"""DR checks per bus: cost, same-hour overlap, chain lengths.
Usage: python dr_check.py <network.nc>"""
import sys

import numpy as np
import pandas as pd
import pypsa

fn = sys.argv[1] if len(sys.argv) > 1 else "results/dr-test/networks/solved_2025.nc"
n = pypsa.Network(fn)
print(f"\nObjective: {n.objective/1e6:,.3f} M EUR\n")

L = n.links
p = n.links_t.p0
e = n.stores_t.e
rows = {}
for dev in sorted({c.split()[1] for c in n.stores.carrier if c.startswith("DR ")}):
    up_tot = ov_tot = 0.0
    runs_all, active, emax = [], [], 0.0
    for s in n.stores.index[n.stores.carrier == f"DR {dev}"]:
        b = n.stores.at[s, "bus"]  # the DR bus of this device at one location
        up_i = L.index[(L.carrier == f"DR {dev} up") & (L.bus1 == b)]
        dn_i = L.index[(L.carrier == f"DR {dev} down") & (L.bus0 == b)]
        up = p[up_i].sum(axis=1)
        dn = p[dn_i].sum(axis=1)
        up_tot += up.sum()
        ov_tot += np.minimum(up, dn).sum()  # same hour, same bus
        on = e[s].abs() > 1e-3
        runs_all += list((on != on.shift()).cumsum()[on].value_counts())
        active.append(on.sum())
        emax = max(emax, e[s].abs().max())
    r = pd.Series(runs_all, dtype=float)
    rows[dev] = {
        "up GWh": up_tot / 1e3,
        "overlap GWh": ov_tot / 1e3,
        "real GWh": (up_tot - ov_tot) / 1e3,
        "events": len(r),
        "median h": r.median() if len(r) else 0,
        "p90 h": r.quantile(0.9) if len(r) else 0,
        "longest h": r.max() if len(r) else 0,
        "avg h active/bus": float(np.mean(active)) if active else 0,
        "max |e| MWh/bus": emax,
    }
df = pd.DataFrame(rows).T
print(df.round(1).to_string())
print(f"\nTOTAL up {df['up GWh'].sum():.1f} GWh, overlap {df['overlap GWh'].sum():.1f} GWh, "
      f"real {df['real GWh'].sum():.1f} GWh")
