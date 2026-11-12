"""
2050 electricity demand per country and sector from ENTSO-E/ENTSOG TYNDP 2026
(National Trends+ scenario, 2050). Replaces the JRC CETO 2025 growth factors.

Two TYNDP files are used:
  NT+_KPI_Dashboard.xlsx          country total electricity demand in the TYNDP market
                                  model (all countries, incl. NO, UK, Balkans)
  TYNDP2026_ETM_DASHBOARD.xlsx    final electricity demand per sector from the Energy
                                  Transition Model (EU-27 + CH only)

Country total (TWh) = Native Demand (prosumer + market) + EV charging (prosumer + market).
  Included:  everything a grid-connected customer uses, incl. heat pumps, EVs, data
             centres, and network losses (the TYNDP market load includes losses, like
             the ENTSO-E load that PyPSA-Eur uses for today).
  Excluded:  electrolysers (hydrogen), pumped-hydro pumping, hybrid heat pumps on
             CH4/H2 boilers (all endogenous in TYNDP, i.e. decided by their model),
             and "Fixed Load (prosumer)", which is negative behind-the-meter PV in
             AT/CH/HU (PyPSA-Eur models rooftop PV as generation instead).
  Non-EU countries (NO, UK, RS, BA, ME, MK, AL) only report "Native Demand (E-Market)",
  which already contains EVs.

Sector split:
  EU-27 + CH: TYNDP ETM sector shares, scaled to the country total above.
  Others:     sector shares of the JRC-based 2050 split (electricity_by_sector_2050_TWh.csv
              from build_2050_country_demand.py), scaled to the TYNDP total.
  GB = UK + NI (Northern Ireland is a separate TYNDP node).
  XK (Kosovo) is not in TYNDP: today's load x Serbia's TYNDP growth (2050 / 2023).

Hydrogen demand per country (TWh H2, TYNDP "H2 Demand") is written to a separate file.
It is NOT added to the electricity demand.

Usage:
  python build_2050_country_demand_tyndp.py KPI.xlsx ETM.xlsx jrc_totals_dir out_dir
jrc_totals_dir holds the output of build_2050_country_demand.py (2023 split, JRC 2050 split,
growth factors). Those files are copied to out_dir so the nodal script finds everything there.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

SCENARIO_KPI = "Weighted WS"   # TYNDP weighted average of its three 2050 weather scenarios
SCENARIO_ETM = "NT"            # NT in the ETM file = NT+ in the market model
YEAR = 2050
COLS = ["residential appliances", "residential heat/cooling/cooking", "services",
        "agriculture", "transport", "industry+other"]

kpi_fn, etm_fn, jrc_dir, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- country totals (KPI)
ROWS = {  # row label in column B -> short name
    "Native Demand (Prosumer)": "native_prosumer",
    "Native Demand (E-Market)": "native_market",
    "Charging Station Load (Prosumer)": "ev_prosumer",
    "Charging Station Load (E-Market)": "ev_market",
    "Fixed Load (Prosumer)": "fixed_prosumer (excl.)",
    "Fixed Load (Emarket)": "fixed_market (excl.)",
    "Electrolyzers E-Market (P2G)": "electrolysers (excl.)",
    "PS Pump Load": "pumping (excl.)",
    "H2 Demand": "H2 demand",
}
xl = pd.ExcelFile(kpi_fn)
kpi = {}
for sheet in xl.sheet_names:
    if sheet in ("Info", "EU27"):
        continue
    df = pd.read_excel(xl, sheet, header=None)
    # column of the 2050 weighted value: header row has the year, next row the weather scenario
    year_col = [c for c in df.columns if str(df.iloc[2, c]).strip() == str(YEAR)][0]
    col = [c for c in df.columns[year_col:] if str(df.iloc[3, c]).strip() == SCENARIO_KPI][0]
    consumption = df.index[df[0].astype(str).str.startswith("Consumption")][0]
    block = df.loc[consumption:consumption + 20]
    vals = {}
    for label, name in ROWS.items():
        hit = block[block[1].astype(str).str.strip() == label]
        vals[name] = pd.to_numeric(hit[col].iloc[0], errors="coerce") if len(hit) else 0.0
    kpi[sheet] = vals
kpi = pd.DataFrame(kpi).T.fillna(0.0)
kpi.index = kpi.index.str.replace("UK", "GB")
kpi.loc["GB"] = kpi.loc["GB"] + kpi.loc["NI"]
kpi = kpi.drop(index="NI")
kpi["total"] = kpi[["native_prosumer", "native_market", "ev_prosumer", "ev_market"]].sum(axis=1)

# ---------------------------------------------------------------- sector split (ETM)
etm = pd.read_excel(etm_fn, "data")
etm = etm[(etm.SCENARIO == SCENARIO_ETM) & (etm.YEAR == YEAR)]
el = etm[etm.ENERGY_CARRIER == "Electricity"]


def pick(sector, subsectors=None):
    s = el[el.SECTOR == sector]
    if subsectors is not None:
        s = s[s.SUBSECTOR.isin(subsectors)]
    return s.groupby("COUNTRY")["values"].sum()


sec = pd.DataFrame({
    # cooling goes with appliances so it keeps today's (summer-including) shape, not the heat shape
    "residential appliances": pick("Households", ["Appliances", "Cooling"]),
    "residential heat/cooling/cooking": pick("Households", ["Space heating", "Hot water"]),
    "services": pick("Buildings"),                       # heat, cooling, appliances, data centres
    "services heat": pick("Buildings", ["Space heating"]),
    "agriculture": pick("Agriculture"),
    "transport": pick("Transport"),                      # domestic only, no int. aviation/shipping
    "industry+other": pick("Industry") + pick("Others_Sector").reindex(pick("Industry").index, fill_value=0),
}).fillna(0.0)
etm_sum = sec[COLS].sum(axis=1)

# ---------------------------------------------------------------- combine
t23 = pd.read_csv(jrc_dir / "electricity_by_sector_2023_TWh.csv", index_col=0)
jrc50 = pd.read_csv(jrc_dir / "electricity_by_sector_2050_TWh.csv", index_col=0)
countries = t23.index

totals = kpi["total"].reindex(countries)
totals["XK"] = t23.loc["XK", "total load (ENTSO-E)"] * kpi.loc["RS", "total"] / t23.loc["RS", "total load (ENTSO-E)"]

out = pd.DataFrame(index=countries, columns=COLS + ["services heat"], dtype=float)
source = pd.Series("", index=countries)
for c in countries:
    if c in sec.index:
        shares = sec.loc[c] / etm_sum[c]
        source[c] = "TYNDP total, TYNDP ETM sector split"
    else:
        shares = (jrc50.loc[c, COLS] / jrc50.loc[c, "sum sectors"]).reindex(out.columns)
        shares["services heat"] = float("nan")   # nodal script falls back to its own estimate
        source[c] = "TYNDP total, JRC-based sector split" if c != "XK" else "today x RS TYNDP growth, JRC-based split"
    out.loc[c] = shares * totals[c]
out["sum sectors"] = out[COLS].sum(axis=1)
out = out.round(2)
out.to_csv(out_dir / "electricity_by_sector_2050_TWh.csv")

# ---------------------------------------------------------------- side outputs
for fn in ["electricity_by_sector_2023_TWh.csv", "ceto2025_growth_factors_vs_2023.csv"]:
    shutil.copy(jrc_dir / fn, out_dir / fn)   # the nodal script reads these too

kpi.round(2).to_csv(out_dir / "tyndp2026_nt_2050_kpi_TWh.csv")
h2 = kpi["H2 demand"].reindex(countries).fillna(0.0).rename("H2 demand 2050 (TWh H2)")
h2.round(2).to_csv(out_dir / "tyndp2026_nt_2050_hydrogen_demand_TWh.csv")

cmp = pd.DataFrame({
    "today_TWh": t23["total load (ENTSO-E)"],
    "JRC_2050_TWh": jrc50["sum sectors"],
    "TYNDP_2050_TWh": out["sum sectors"],
    "TYNDP_growth": out["sum sectors"] / t23["total load (ENTSO-E)"],
    "TYNDP_vs_JRC": out["sum sectors"] / jrc50["sum sectors"],
    "ETM_final_demand_TWh": etm_sum.reindex(countries),
    "H2_TWh": h2,
    "source": source,
}).round(2)
cmp.to_csv(out_dir / "country_totals_jrc_vs_tyndp.csv")

pd.set_option("display.width", 200)
print(cmp.to_string())
print("Sum today / JRC 2050 / TYNDP 2050 (TWh):",
      round(cmp.today_TWh.sum()), "/", round(cmp.JRC_2050_TWh.sum()), "/", round(cmp.TYNDP_2050_TWh.sum()))
print("Hydrogen demand, sum (TWh H2):", round(h2.sum()))
