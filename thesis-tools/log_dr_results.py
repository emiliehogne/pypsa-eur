"""
Log results from solved PyPSA-Eur DR runs into one shared CSV, and compare
every DR run with its no-DR reference.

Usage (from ~/pypsa-eur-thesis):

    # add one or more finished runs to the log
    pixi run python thesis-tools/log_dr_results.py add \
        results/EU100_w13_plow_smid_dr/networks/solved_2025.nc \
        results/EU100_w13_nodr/networks/solved_2025.nc

    # compare every DR run with its no-DR reference
    pixi run python thesis-tools/log_dr_results.py compare

Naming convention (run: name in the config)
-------------------------------------------

    DR runs:     <AREA><NODES>_w<WEATHER>_p<PART>_s<SHIFT>_dr
    No-DR runs:  <AREA><NODES>_w<WEATHER>_nodr

    AREA     BE, MC (multi-country), EU
    NODES    number of clusters, e.g. 5, 20, 100
    WEATHER  weather year, last two digits, e.g. 13
    PART     participation: low (achievable), mid, high (theoretical)
    SHIFT    shift time:    low, mid (O'Reilly reference), high

    EU100_w13_plow_smid_dr    base case
    EU100_w13_phigh_smid_dr   theoretical participation
    EU100_w13_plow_shigh_dr   longer shift times
    EU100_w10_plow_smid_dr    2010 weather
    EU100_w13_nodr            no-DR reference for all w13 runs
    EU100_w10_nodr            no-DR reference for all w10 runs

Participation and shift time do not exist without DR, so there is ONE no-DR
run per area, nodes and weather year. Each DR run is compared with the no-DR
run that has the same <AREA><NODES>_w<WEATHER>.

Old names still work: "base" is read as plow_smid, and a no-DR run named
e.g. EU100_w13_base_nodr is used as the reference for EU100_w13.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

LOG_FILE = Path(__file__).parent / "dr_results.csv"
LEVEL = {"low": "low", "mid": "mid", "high": "high"}


# ---------------------------------------------------------------- naming


def parse_name(name):
    """Split a run name into its parts. Unknown tokens go into 'extra'."""
    tokens = str(name).split("_")
    out = {"area_nodes": "", "weather": "", "participation": "", "shift": "",
           "dr": "", "extra": ""}
    if len(tokens) < 3:
        out["extra"] = name
        return out
    out["area_nodes"] = tokens[0]
    out["weather"] = tokens[1]
    out["dr"] = tokens[-1] if tokens[-1] in ("dr", "nodr") else ""
    extra = []
    for t in tokens[2:-1] if out["dr"] else tokens[2:]:
        if t == "base":
            out["participation"] = out["participation"] or "low"
            out["shift"] = out["shift"] or "mid"
        elif t.startswith("p") and t[1:] in LEVEL:
            out["participation"] = t[1:]
        elif t.startswith("s") and t[1:] in LEVEL:
            out["shift"] = t[1:]
        else:
            extra.append(t)
    out["extra"] = "_".join(extra)
    if out["dr"] == "nodr":  # participation/shift don't apply without DR
        out["participation"] = out["shift"] = ""
    return out


def key(name):
    p = parse_name(name)
    return f"{p['area_nodes']}_{p['weather']}"


# ---------------------------------------------------------------- summarise


def get(d, *keys, default=None):
    """Safely read a nested value from n.meta."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def summarise(path):
    import pypsa

    n = pypsa.Network(path)
    m = n.meta or {}
    run_name = get(m, "run", "name", default=Path(path).parts[-3])
    w = n.snapshot_weightings.generators

    gen = n.generators
    p = n.generators_t.p.mul(w, axis=0)
    gen_twh = p.sum().groupby(gen.carrier).sum() / 1e6

    # CO2: fuel use = output / efficiency, times carrier emission factor
    co2_factor = gen.carrier.map(n.carriers.co2_emissions).fillna(0)
    co2_mt = (p.sum() / gen.efficiency * co2_factor).sum() / 1e6

    # DR volumes and same-hour up/down overlap (chain shifting indicator)
    up_cols = n.links.index[n.links.index.str.contains(" DR ") & n.links.index.str.endswith(" up")]
    shifted_twh = overlap_pct = None
    if len(up_cols):
        down_cols = up_cols.str.replace(r" up$", " down", regex=True)
        down_cols = down_cols[down_cols.isin(n.links.index)]
        up = n.links_t.p0[up_cols].mul(w, axis=0)
        down = n.links_t.p0[down_cols].mul(w, axis=0)
        down.columns = down.columns.str.replace(r" down$", " up", regex=True)
        overlap = up.clip(upper=down.reindex(columns=up.columns, fill_value=0)).clip(lower=0)
        shifted_twh = up.values.sum() / 1e6
        overlap_pct = 100 * overlap.values.sum() / up.values.sum() if shifted_twh else 0.0

    row = {
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run_name": run_name,
        "dr_enabled": get(m, "load", "demand_response", "enable"),
        "dr_data_dir": get(m, "load", "demand_response", "data_dir"),
        "countries": " ".join(sorted(n.buses.country.dropna().unique())),
        "n_buses": len(n.buses[n.buses.carrier == "AC"]),
        "snapshots": f"{n.snapshots[0]:%Y-%m-%d} to {n.snapshots[-1]:%Y-%m-%d} ({len(n.snapshots)}h)",
        "demand_year": get(m, "load", "fixed_year", default="same as snapshots"),
        "cost_year": get(m, "costs", "year"),
        "co2_price_enabled": get(m, "costs", "emission_prices", "enable"),
        "co2_price": get(m, "costs", "emission_prices", "co2"),
        "irena_capacities": get(m, "electricity", "estimate_renewable_capacities", "from_irenastat"),
        "objective_MEUR": round(n.objective / 1e6, 2) if n.objective is not None else None,
        "opex_MEUR": round(n.statistics.opex().sum() / 1e6, 2),
        "co2_Mt": round(co2_mt, 3),
        "load_shedding_TWh": round(gen_twh.get("load", 0.0), 4),
        "dr_shifted_TWh": round(shifted_twh, 3) if shifted_twh is not None else None,
        "dr_overlap_pct": round(overlap_pct, 2) if overlap_pct is not None else None,
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
        old = old[~old.run_name.isin(new.run_name)]  # re-logging replaces the old row
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(LOG_FILE, index=False)
    for r in rows:
        extra = (f", shifted {r['dr_shifted_TWh']} TWh, overlap {r['dr_overlap_pct']}%"
                 if r["dr_shifted_TWh"] is not None else "")
        print(f"logged {r['run_name']}: objective {r['objective_MEUR']} MEUR, "
              f"CO2 {r['co2_Mt']} Mt, DR {r['dr_enabled']}{extra}")
    print(f"-> {LOG_FILE}")


# ---------------------------------------------------------------- compare


def compare():
    if not LOG_FILE.exists():
        sys.exit("No results logged yet.")
    df = pd.read_csv(LOG_FILE)
    parsed = df.run_name.apply(parse_name).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    df["key"] = df.area_nodes + "_" + df.weather
    for col in ["dr_shifted_TWh", "dr_overlap_pct"]:
        if col not in df.columns:
            df[col] = None

    settings = ["countries", "n_buses", "snapshots", "demand_year", "cost_year",
                "co2_price_enabled", "co2_price", "irena_capacities"]
    drs = df[df.dr == "dr"]
    nodrs = df[df.dr == "nodr"]

    out, missing = [], []
    for _, dr in drs.iterrows():
        ref = nodrs[nodrs.key == dr.key]
        if ref.empty:
            missing.append(f"{dr.run_name} (needs {dr.key}_nodr)")
            continue
        nodr = ref.iloc[-1]  # latest logged reference
        mismatch = [s for s in settings if s in df.columns and str(dr[s]) != str(nodr[s])]
        saving = nodr.objective_MEUR - dr.objective_MEUR
        out.append({
            "dr_run": dr.run_name,
            "area": dr.area_nodes,
            "weather": dr.weather,
            "part": dr["participation"],
            "shift": dr["shift"],
            "nodr_run": nodr.run_name,
            "cost_nodr_MEUR": nodr.objective_MEUR,
            "cost_dr_MEUR": dr.objective_MEUR,
            "saving_MEUR": round(saving, 2),
            "saving_pct": round(100 * saving / nodr.objective_MEUR, 3),
            "co2_change_Mt": round(dr.co2_Mt - nodr.co2_Mt, 3),
            "shifted_TWh": dr.dr_shifted_TWh,
            "overlap_pct": dr.dr_overlap_pct,
            "settings_match": "yes" if not mismatch else "NO: " + ", ".join(mismatch),
        })

    if not out:
        sys.exit("No DR runs with a matching no-DR reference yet.")
    order = {"low": 0, "mid": 1, "high": 2, "": 3}
    res = pd.DataFrame(out)
    res = res.sort_values(["area", "weather", "part", "shift"],
                          key=lambda c: c.map(order) if c.name in ("part", "shift") else c)
    pd.set_option("display.width", 250)
    print(res.to_string(index=False))
    if missing:
        print("\nNo no-DR reference found for: " + "; ".join(missing))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("paths", nargs="+")
    sub.add_parser("compare")
    args = ap.parse_args()
    add(args.paths) if args.cmd == "add" else compare()