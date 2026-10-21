"""
Log results from solved PyPSA-Eur DR runs into one shared CSV.

Usage (from ~/pypsa-eur-thesis, on Solstorm or the Mac):

    # add one or more finished runs to the log
    pixi run python thesis_tools/log_dr_results.py add \
        results/BE5_w13_base_dr/networks/solved_2025.nc \
        results/BE5_w13_base_nodr/networks/solved_2025.nc

    # show DR savings for every dr/nodr pair in the log
    pixi run python thesis_tools/log_dr_results.py compare

Naming convention for run names (run: name in the config):

    <AREA><NODES>_w<WEATHER>_<CASE>_<dr|nodr>

    BE5_w13_base_dr       Belgium, 5 nodes, 2013 weather, base case, with DR
    BE5_w13_base_nodr     same, without DR
    EU100_w13_base_dr     Europe, 100 nodes
    EU100_w13_part50_dr   sensitivity, 50% participation
    EU100_w13_shift2h_dr  sensitivity, shorter shift time
    EU100_w10_base_dr     sensitivity, 2010 weather year

A dr run and a nodr run with the same prefix are treated as a pair.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pypsa

LOG_FILE = Path(__file__).parent / "dr_results.csv"


def get(d, *keys, default=None):
    """Safely read a nested value from n.meta."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def summarise(path):
    n = pypsa.Network(path)
    m = n.meta or {}
    run_name = get(m, "run", "name", default=Path(path).parts[-3])

    gen = n.generators
    gen_twh = n.generators_t.p.mul(n.snapshot_weightings.generators, axis=0)
    gen_twh = gen_twh.sum().groupby(gen.carrier).sum() / 1e6

    # CO2: fuel use = output / efficiency, times carrier emission factor
    co2_factor = gen.carrier.map(n.carriers.co2_emissions).fillna(0)
    fuel = n.generators_t.p.mul(n.snapshot_weightings.generators, axis=0).sum() / gen.efficiency
    co2_mt = (fuel * co2_factor).sum() / 1e6

    opex_meur = n.statistics.opex().sum() / 1e6
    shed_twh = gen_twh.get("load", 0.0)

    row = {
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run_name": run_name,
        "dr_enabled": get(m, "load", "demand_response", "enable"),
        "countries": " ".join(sorted(n.buses.country.dropna().unique())),
        "n_buses": len(n.buses[n.buses.carrier == "AC"]),
        "snapshots": f"{n.snapshots[0]:%Y-%m-%d} to {n.snapshots[-1]:%Y-%m-%d} ({len(n.snapshots)}h)",
        "demand_year": get(m, "load", "fixed_year", default="same as snapshots"),
        "cost_year": get(m, "costs", "year"),
        "co2_price_enabled": get(m, "costs", "emission_prices", "enable"),
        "co2_price": get(m, "costs", "emission_prices", "co2"),
        "irena_capacities": get(m, "electricity", "estimate_renewable_capacities", "from_irenastat"),
        "objective_MEUR": round(n.objective / 1e6, 2) if n.objective is not None else None,
        "opex_MEUR": round(opex_meur, 2),
        "co2_Mt": round(co2_mt, 3),
        "load_shedding_TWh": round(shed_twh, 4),
        "solar_GW": round(gen.p_nom[gen.carrier.str.startswith("solar")].sum() / 1e3, 2),
        "onwind_GW": round(gen.p_nom[gen.carrier == "onwind"].sum() / 1e3, 2),
        "file": str(path),
    }
    for carrier, twh in gen_twh.items():
        if carrier != "load":
            row[f"gen_{carrier}_TWh"] = round(twh, 3)
    return row


def add(paths):
    rows = [summarise(p) for p in paths]
    new = pd.DataFrame(rows)
    if LOG_FILE.exists():
        old = pd.read_csv(LOG_FILE)
        # re-logging a run replaces its old row
        old = old[~old.run_name.isin(new.run_name)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(LOG_FILE, index=False)
    for r in rows:
        print(f"logged {r['run_name']}: objective {r['objective_MEUR']} MEUR, "
              f"CO2 {r['co2_Mt']} Mt, DR {r['dr_enabled']}")
    print(f"-> {LOG_FILE}")


def compare():
    if not LOG_FILE.exists():
        sys.exit("No results logged yet.")
    df = pd.read_csv(LOG_FILE)
    df["pair"] = df.run_name.str.replace(r"_(nodr|dr)$", "", regex=True)

    settings = ["countries", "n_buses", "snapshots", "demand_year", "cost_year",
                "co2_price_enabled", "co2_price", "irena_capacities"]
    out = []
    for pair, g in df.groupby("pair"):
        dr = g[g.run_name.str.endswith("_dr")]
        nodr = g[g.run_name.str.endswith("_nodr")]
        if dr.empty or nodr.empty:
            continue
        dr, nodr = dr.iloc[-1], nodr.iloc[-1]
        mismatch = [s for s in settings if str(dr[s]) != str(nodr[s])]
        saving = nodr.objective_MEUR - dr.objective_MEUR
        out.append({
            "pair": pair,
            "cost_nodr_MEUR": nodr.objective_MEUR,
            "cost_dr_MEUR": dr.objective_MEUR,
            "saving_MEUR": round(saving, 2),
            "saving_pct": round(100 * saving / nodr.objective_MEUR, 3),
            "co2_change_Mt": round(dr.co2_Mt - nodr.co2_Mt, 3),
            "settings_match": "yes" if not mismatch else "NO: " + ", ".join(mismatch),
        })
    if not out:
        sys.exit("No complete dr/nodr pairs yet.")
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("paths", nargs="+")
    sub.add_parser("compare")
    args = p.parse_args()
    add(args.paths) if args.cmd == "add" else compare()