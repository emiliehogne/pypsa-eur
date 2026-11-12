"""
2050 electricity demand per country and sector (option 1).

Today (2023) per country and sector comes from PyPSA-Eur's energy_totals.csv.
2050 = today x EU-27 growth factor per sector from JRC POTEnCIA CETO 2025.
Transport is handled differently: the EU-27 2050 transport electricity is
split across countries by today's road transport energy.
Industry is not in energy_totals.csv. If PyPSA-Eur's electricity_demand.csv
(hourly ENTSO-E load per country) is given, industry + other is taken as
the rest of today's load and scaled with the industry factor.

Usage:
  python build_2050_country_demand.py energy_totals.csv ceto2025_EU27_summary_yearly.xlsx out_dir [electricity_demand.csv]
"""

import sys
from pathlib import Path

import pandas as pd

BASE_YEAR = 2023          # PyPSA-Eur default energy_totals_year, and last statistical year in CETO 2025
TARGET_YEAR = 2050
KTOE_TO_TWH = 0.01163
# Share of household electricity used for heat (space + water + cooking) where
# PyPSA-Eur's split is not usable. Norway: energy_totals puts ~100 % in heat.
# 0.60 sits between SSB (Larsen & Nesbakken 2005: space ~1/3, water ~10 % in 2001)
# and SINTEF (Mørch et al. 2013: space heating ~60 % in 2010). Change if needed.
RES_HEAT_SHARE_OVERRIDE = {"NO": 0.60}

EU27 = ["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
        "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"]

et_fn, ceto_fn, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
load_fn = sys.argv[4] if len(sys.argv) > 4 else None
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- JRC growth factors
# (sheet, row) in ceto2025_EU27_summary_yearly.xlsx, values in ktoe
ROWS = {
    "total": ("Overview", 94),
    "residential": ("Residential", 36),
    "residential appliances": ("Residential", 18),
    "services": ("Tertiary", 35),
    "transport": ("Transport", 54),
    "industry": ("Industry", 71),
}
ceto = {}
for name, (sheet, row) in ROWS.items():
    df = pd.read_excel(ceto_fn, sheet, header=None)
    years = df.iloc[0]
    col = {int(v): k for k, v in years.items() if isinstance(v, (int, float)) and v == v and v > 1999}
    ceto[name] = {y: df.iloc[row, col[y]] * KTOE_TO_TWH for y in (BASE_YEAR, TARGET_YEAR)}
ceto = pd.DataFrame(ceto).T
ceto.loc["residential heat/cooling/cooking"] = ceto.loc["residential"] - ceto.loc["residential appliances"]
ceto.loc["agriculture+other"] = ceto.loc["total"] - ceto.loc[["residential", "services", "transport", "industry"]].sum()
ceto["factor"] = ceto[TARGET_YEAR] / ceto[BASE_YEAR]
ceto.round(3).to_csv(out_dir / "ceto2025_growth_factors_vs_2023.csv")
f = ceto["factor"]

# ---------------------------------------------------------------- today per country
et = pd.read_csv(et_fn, index_col=[0, 1])
today = et.xs(BASE_YEAR, level="year").fillna(0.0)

res_heat = today[["electricity residential space", "electricity residential water",
                  "electricity residential cooking"]].sum(axis=1)
res_other = (today["electricity residential"] - res_heat).clip(lower=0)  # negative in a few countries
for c, share in RES_HEAT_SHARE_OVERRIDE.items():
    if c in today.index:
        total = today.loc[c, "electricity residential"]
        res_heat[c], res_other[c] = share * total, (1 - share) * total
        print(f"{c}: household heat share set to {share:.0%} of {total:.1f} TWh")

t = pd.DataFrame({
    "residential appliances": res_other,
    "residential heat/cooling/cooking": res_heat,
    "services": today["electricity services"],
    "agriculture": today["total agriculture electricity"],
    "transport": today["electricity road"] + today["electricity rail"],
})

# ---------------------------------------------------------------- 2050
t50 = pd.DataFrame(index=t.index)
t50["residential appliances"] = t["residential appliances"] * f["residential appliances"]
t50["residential heat/cooling/cooking"] = t["residential heat/cooling/cooking"] * f["residential heat/cooling/cooking"]
t50["services"] = t["services"] * f["services"]
t50["agriculture"] = t["agriculture"] * f["agriculture+other"]

# transport: EU-27 2050 transport electricity per unit of today's road energy
eu_road = today.loc[today.index.intersection(EU27), "total road"].sum()
twh_per_road = ceto.loc["transport", TARGET_YEAR] / eu_road
t50["transport"] = today["total road"] * twh_per_road

# industry + other: rest of today's total load (if the load file is given)
if load_fn:
    load = pd.read_csv(load_fn, index_col=0, parse_dates=True)
    years = load.index.year.unique()
    if len(years) > 1:  # file with several years: use the base year
        load = load[load.index.year == (BASE_YEAR if BASE_YEAR in years else years.max())]
    print("Load file year(s) used:", sorted(load.index.year.unique()))
    annual = load.sum() / 1e6  # MWh -> TWh over the year in the file
    annual.index = annual.index.str[:2]
    annual = annual.groupby(level=0).sum()
    t["industry+other"] = (annual.reindex(t.index) - t.sum(axis=1)).clip(lower=0)
    t50["industry+other"] = t["industry+other"] * f["industry"]
    t["total load (ENTSO-E)"] = annual.reindex(t.index)

t["sum sectors"] = t.drop(columns=["total load (ENTSO-E)"], errors="ignore").sum(axis=1)
t50["sum sectors"] = t50.sum(axis=1)
ratio = (t50["sum sectors"] / t["sum sectors"]).rename("2050/2023")

t.round(2).to_csv(out_dir / f"electricity_by_sector_{BASE_YEAR}_TWh.csv")
t50.round(2).to_csv(out_dir / f"electricity_by_sector_{TARGET_YEAR}_TWh.csv")
ratio.round(3).to_csv(out_dir / "country_growth_2050_vs_2023.csv")

pd.set_option("display.width", 220)
print("Growth factors (EU-27, CETO 2025):")
print(ceto.round(2))
print(f"\nTransport: {twh_per_road:.3f} TWh electricity in 2050 per TWh road energy today")
print("\n2050 by sector (TWh):")
print(t50.round(1))
print("\nTotals TWh: 2023 =", round(t['sum sectors'].sum(), 0), " 2050 =", round(t50['sum sectors'].sum(), 0))
