#!/usr/bin/env python
"""
Hent ut små CSV-filer fra løste nettverk, så figurene kan lages på Mac.

Kjøres på Solstorm (der nettverkene ligger), på en compute-node, fra ~/pypsa-eur-thesis:

    pixi run python thesis-tools/extract_results.py EU150_w13_plow_smid_g_dr EU150_w13_base_nodr ...

Hvert argument er et kjøringsnavn (mappen under results/). Leser
results/<run>/networks/solved_2025.nc og skriver til thesis-tools/data/<run>/:

    summary.csv           én rad: total opex, aktiveringskostnad, flyttet energi, CO2
    prices_country.csv    timepris per land (lastvektet snitt av nodene), 8760 x land
    demand_country.csv    årlig etterspørsel per land [MWh]
    dr_country_group.csv  flyttet energi per land og DR-gruppe [MWh] (kun DR-kjøringer)
    dr_daily_profile.csv  gj.snitt netto flytting per time på døgnet og DR-gruppe [MW]

Flyttet energi = sum av L_down (én retning), samme definisjon som dr_value_per_mwh.py.
Filene er små (under 1 MB per kjøring) og kan committes eller hentes med scp.
"""

import sys
from pathlib import Path

import pandas as pd
import pypsa

OUT = Path("thesis-tools/data")


def country_of(n, buses):
    return n.buses.loc[buses, "country"].values


def extract(run: str, fn: str | None = None) -> None:
    fn = fn or f"results/{run}/networks/solved_2025.nc"
    print(f"--- {run}: leser {fn}")
    n = pypsa.Network(fn)
    w = n.snapshot_weightings.generators
    out = OUT / run
    out.mkdir(parents=True, exist_ok=True)

    # Last og priser per land (kun AC-noder med last)
    load = n.loads_t.p_set.T.groupby(n.loads.bus).sum().T  # MW, tid x node
    load = load.loc[:, load.columns.isin(n.buses.index[n.buses.carrier == "AC"])]
    price = n.buses_t.marginal_price[load.columns]
    cty = pd.Series(country_of(n, load.columns), index=load.columns)
    num = (price * load).T.groupby(cty).sum().T
    den = load.T.groupby(cty).sum().T
    prices = num / den
    prices["EU"] = num.sum(axis=1) / den.sum(axis=1)  # lastvektet snitt for hele Europa
    prices.round(3).to_csv(out / "prices_country.csv")
    load.mul(w, axis=0).sum().groupby(cty).sum().rename("demand_MWh").to_csv(
        out / "demand_country.csv")

    # DR
    dr_links = n.links[n.links.carrier.str.startswith("DR ")]
    up = dr_links[dr_links.carrier.str.endswith(" up")]
    down = dr_links[dr_links.carrier.str.endswith(" down")]
    shifted_MWh = 0.0
    activation_MEUR = 0.0
    if len(down):
        p_up = n.links_t.p0[up.index].mul(w, axis=0)
        p_down = n.links_t.p0[down.index].mul(w, axis=0)
        activation_MEUR = ((p_up * up.marginal_cost).sum().sum()
                           + (p_down * down.marginal_cost).sum().sum()) / 1e6
        shifted_MWh = p_down.sum().sum()

        # Per land og gruppe. Elektrisitetsbussen: bus0 for up, bus1 for down
        grp_down = down.carrier.str[len("DR "):-len(" down")]
        cty_down = pd.Series(country_of(n, down.bus1), index=down.index)
        per = p_down.sum().to_frame("shifted_MWh")
        per["country"], per["group"] = cty_down, grp_down
        per.groupby(["country", "group"]).shifted_MWh.sum().to_csv(out / "dr_country_group.csv")

        # Døgnprofil: netto (L_up - L_down) summert over Europa per gruppe
        grp_up = up.carrier.str[len("DR "):-len(" up")]
        net = (n.links_t.p0[up.index].T.groupby(grp_up).sum().T
               - n.links_t.p0[down.index].T.groupby(grp_down).sum().T)
        net.groupby(net.index.hour).mean().round(1).to_csv(out / "dr_daily_profile.csv")

    opex_MEUR = n.statistics.opex().sum() / 1e6
    gen = n.generators_t.p.mul(w, axis=0).sum()
    co2 = (gen / n.generators.efficiency * n.generators.carrier.map(
        n.carriers.co2_emissions).fillna(0)).sum() / 1e6
    pd.DataFrame([{
        "run": run,
        "dr": len(down) > 0,
        "opex_MEUR": round(opex_MEUR, 3),
        "activation_MEUR": round(activation_MEUR, 3),
        "opex_excl_activation_MEUR": round(opex_MEUR - activation_MEUR, 3),
        "shifted_TWh": round(shifted_MWh / 1e6, 4),
        "demand_TWh": round(load.mul(w, axis=0).sum().sum() / 1e6, 3),
        "co2_Mt": round(co2, 3),
    }]).to_csv(out / "summary.csv", index=False)
    print(f"    ferdig: {out}")
    del n


if __name__ == "__main__":
    runs = sys.argv[1:]
    if not runs:
        sys.exit(__doc__)
    for r in runs:
        extract(r)
