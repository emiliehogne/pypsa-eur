"""
Look inside a solved PyPSA-Eur network.

Run from the pypsa-eur folder, inside pixi shell:
    python inspect_run.py
or for another file:
    python inspect_run.py results/dr-test/networks/solved_2025.nc
"""

import sys

import pandas as pd
import pypsa

fn = sys.argv[1] if len(sys.argv) > 1 else "results/dr-test/networks/solved_2025.nc"
n = pypsa.Network(fn)

w = n.snapshot_weightings.generators  # hours each snapshot stands for
hours = w.sum()


def line(title):
    print("\n" + title)
    print("-" * len(title))


line("The network")
print(f"Snapshots     : {len(n.snapshots)} ({n.snapshots[0]} to {n.snapshots[-1]})")
print(f"Hours covered : {hours:.0f}")
print(f"Buses         : {len(n.buses)}  {list(n.buses.index)}")
print(f"Loads         : {len(n.loads)}")
print(f"Generators    : {len(n.generators)}")
print(f"Lines / Links : {len(n.lines)} / {len(n.links)}")
print(f"Stores / StorageUnits : {len(n.stores)} / {len(n.storage_units)}")

line("Demand")
demand = n.loads_t.p_set.multiply(w, axis=0).sum().sum()
peak = n.loads_t.p_set.sum(axis=1).max()
print(f"Total demand  : {demand / 1e6:.3f} TWh")
print(f"Peak demand   : {peak / 1e3:.2f} GW")

line("Generation by carrier (TWh)")
gen = n.generators_t.p.multiply(w, axis=0).sum()
by_carrier = gen.groupby(n.generators.carrier).sum().sort_values(ascending=False)
print((by_carrier / 1e6).round(4).to_string())

line("Capacity by carrier (GW)")
g = n.generators[n.generators.carrier != "load"]
before = g.p_nom.groupby(g.carrier).sum() / 1e3
after = g.p_nom_opt.groupby(g.carrier).sum() / 1e3 if "p_nom_opt" in g else before
cap = pd.DataFrame({"existing": before, "after optimisation": after})
cap["newly built"] = cap["after optimisation"] - cap["existing"]
print(cap.round(2).sort_values("after optimisation", ascending=False).to_string())

line("Load shedding (unmet demand)")
shed_i = n.generators.index[n.generators.carrier == "load"]
if len(shed_i):
    shed = n.generators_t.p[shed_i].multiply(w, axis=0)
    shed_total = shed.sum().sum()
    shed_hours = (n.generators_t.p[shed_i].sum(axis=1) > 1e-3).sum()
    shed_cost = shed_total * n.generators.loc[shed_i, "marginal_cost"].iloc[0]
    print(f"Energy shed   : {shed_total:,.0f} MWh ({100 * shed_total / demand:.3f} % of demand)")
    print(f"Hours with shedding : {shed_hours} of {len(n.snapshots)}")
    print(f"Cost of shedding    : {shed_cost / 1e6:,.1f} M€")
else:
    print("No load shedding generators in the network.")

line("Costs")
print(f"Objective     : {n.objective / 1e6:,.1f} M€")
try:
    opex = n.statistics.opex(groupby="carrier").groupby(level=-1).sum()
    print("Operating cost by carrier (M€):")
    print((opex / 1e6).round(2).sort_values(ascending=False).to_string())
except Exception as e:  # statistics API differs a bit between PyPSA versions
    print(f"(could not split operating cost: {e})")

line("Electricity prices (€/MWh)")
prices = n.buses_t.marginal_price
print(f"Average       : {prices.mean().mean():,.1f}")
print(f"Max           : {prices.max().max():,.1f}")
print(f"Hours at shedding price (>= 10 000) : {(prices.max(axis=1) >= 1e4).sum()}")