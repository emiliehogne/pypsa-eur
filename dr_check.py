"""DR checks: cost, same-hour overlap, chain lengths. Usage: python dr_check.py <network.nc>"""
import sys
import pandas as pd
import pypsa

fn = sys.argv[1] if len(sys.argv) > 1 else "results/dr-test/networks/solved_2025.nc"
n = pypsa.Network(fn)
print(f"\nObjective: {n.objective/1e6:,.3f} M EUR\n")

c = n.links.carrier
p = n.links_t.p0
e = n.stores_t.e
rows = {}
for dev in sorted({x.split()[1] for x in c[c.str.startswith("DR ")]}):
    up = p.loc[:, c[c == f"DR {dev} up"].index].sum(axis=1)
    dn = p.loc[:, c[c == f"DR {dev} down"].index].sum(axis=1)
    overlap = pd.concat([up, dn], axis=1).min(axis=1).sum()
    st = n.stores.index[n.stores.carrier == f"DR {dev}"]
    s = e[st].sum(axis=1)
    on = s.abs() > 1e-3
    runs = (on != on.shift()).cumsum()[on].value_counts()
    rows[dev] = {
        "up GWh": up.sum() / 1e3,
        "overlap GWh": overlap / 1e3,
        "real GWh": (up.sum() - overlap) / 1e3,
        "chains": len(runs),
        "median h": runs.median() if len(runs) else 0,
        "longest h": runs.max() if len(runs) else 0,
        "hours active": int(on.sum()),
        "max |e| MWh": s.abs().max(),
    }
df = pd.DataFrame(rows).T
print(df.round(1).to_string())
print(f"\nTOTAL up {df['up GWh'].sum():.1f} GWh, overlap {df['overlap GWh'].sum():.1f} GWh, real {df['real GWh'].sum():.1f} GWh")
