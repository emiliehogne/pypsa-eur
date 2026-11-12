"""
Spread each country's 2050 electricity demand across nodes and hours.

2050 load at node n, hour t:

    L50 = base(n,t) * f_rest(c)                      appliances, services excl. heat,
                                                     agriculture, today's industry
        + heat_res(n,t) + heat_ser(n,t)              household + services heating
        + transport(n,t)                             EVs and rail
        + industry_extra(n)                          industry growth, flat

base(n,t) = today's load - today's electric heating - today's transport electricity
            (the PyPSA-Eur approach, so heating and EVs are not counted twice)

Where (spatial split inside a country):
  base            today's load per node (JRC Energy Atlas, already in electricity_demand.nc)
  heating         population (pop_layout.csv)
  transport       PyPSA-Eur road transport demand per node (transport_demand.csv)
  industry growth industrial sites (industrial_distribution_key.csv)
When (hourly shape):
  base            today's hourly shape (2025 load, with 2013 timestamps to match the
                  2013 weather year used for wind, solar and heat)
  heating         PyPSA-Eur hourly heat demand from 2013 temperatures, mixed with
                  heat pump COP for the share of heat from heat pumps (from JRC CETO 2025)
  transport       PyPSA-Eur road transport profile
  industry growth flat

Annual totals per country and sector come from build_2050_country_demand.py.
Each country is rescaled at the end so its annual 2050 total matches those totals exactly.

Usage:
  python build_2050_nodal_demand.py in_dir country_totals_dir energy_totals.csv out_dir
in_dir holds: electricity_demand.nc, hourly_heat_demand_total.nc, cop_profiles_2025.nc,
              transport_demand.csv, pop_layout.csv, industrial_distribution_key.csv
"""

import sys
from pathlib import Path

import pandas as pd
import xarray as xr

# ASSUMPTIONS (state in the thesis)
BASE_YEAR = 2023
# Share of electric HEAT delivered by heat pumps (rest resistive), by end use.
# Derived from JRC POTEnCIA CETO 2025 (ceto2025_EU27_res_yearly.xlsx, RES_sector):
# useful heat / electricity consumed for space heating = 1.87 in 2023 and 3.85 in 2050,
# for water heating 0.79 (2023) and 1.04 (2050). With a seasonal COP of ~2.8 this gives:
HP_SHARE = {"space": {"today": 0.7, "2050": 1.0},
            "water": {"today": 0.0, "2050": 0.05}}
RES_HEAT_SHARE_OVERRIDE = {"NO": 0.60}   # same as build_2050_country_demand.py
INDUSTRY_COLS = ["Iron and steel", "Cement", "Refineries", "Paper and printing",
                 "Chemical industry", "Glass", "Non-ferrous metals",
                 "Non-metallic mineral products", "Other non-classified"]

in_dir, totals_dir, et_fn, out_dir = (Path(a) for a in sys.argv[1:5])
out_dir.mkdir(parents=True, exist_ok=True)


def country(nodes):
    return pd.Index(nodes).str[:2]


# ---------------------------------------------------------------- inputs
load = xr.open_dataset(in_dir / "electricity_demand.nc")["electricity demand (MW)"].to_pandas()
nodes, ctry = load.columns, country(load.columns)
time = load.index

heat = xr.open_dataset(in_dir / "hourly_heat_demand_total.nc")
def heat_shape(var):
    df = heat[var].to_pandas().reindex(columns=nodes)
    df.index = time
    return df / df.sum()          # each node sums to 1 over the year

space_shape = {s: heat_shape(f"{s} space") for s in ["residential", "services"]}
water_shape = {s: heat_shape(f"{s} water") for s in ["residential", "services"]}

cop = xr.open_dataset(in_dir / "cop_profiles_2025.nc")["temperature"]   # variable holds the COP
cop = cop.sel(heat_source="air", heat_system=["urban decentral", "rural"]).mean("heat_system")
cop = cop.transpose("time", "name").to_pandas().reindex(columns=nodes)
cop.index = time

def elec_shape(shape, hp_share):
    """Electricity for heat: heat pumps use heat/COP, resistive heaters heat/1."""
    s = shape * (hp_share / cop + (1 - hp_share))
    return s / s.sum()

pop = pd.read_csv(in_dir / "pop_layout.csv", index_col=0)["fraction"].reindex(nodes)

tr = pd.read_csv(in_dir / "transport_demand.csv", index_col=0, parse_dates=True).reindex(columns=nodes)
tr.index = time
tr_share = tr.sum() / tr.sum().groupby(ctry).transform("sum")
tr_shape = tr / tr.sum()

ind_key = pd.read_csv(in_dir / "industrial_distribution_key.csv", index_col=0)[INDUSTRY_COLS]
ind_key = ind_key.mean(axis=1).reindex(nodes).fillna(0)
ind_key = ind_key / ind_key.groupby(ctry).transform("sum")

# ---------------------------------------------------------------- country totals (TWh)
t23 = pd.read_csv(totals_dir / f"electricity_by_sector_{BASE_YEAR}_TWh.csv", index_col=0)
t50 = pd.read_csv(totals_dir / "electricity_by_sector_2050_TWh.csv", index_col=0)
fac = pd.read_csv(totals_dir / "ceto2025_growth_factors_vs_2023.csv", index_col=0)["factor"]
et = pd.read_csv(et_fn, index_col=[0, 1]).xs(BASE_YEAR, level="year").fillna(0.0)

# split household heat into space and water(+cooking), and services heat out of services
res_space_share = (et["electricity residential space"] /
                   et[["electricity residential space", "electricity residential water",
                       "electricity residential cooking"]].sum(axis=1)).fillna(0.5)
ser_heat23 = et[["electricity services space", "electricity services water",
                 "electricity services cooking"]].sum(axis=1).clip(upper=et["electricity services"])
ser_space_share = (et["electricity services space"] / ser_heat23).fillna(0.5).clip(0, 1)

C = sorted(set(ctry))
parts23 = pd.DataFrame(index=C)
parts23["res_heat"] = t23["residential heat/cooling/cooking"]
parts23["ser_heat"] = ser_heat23
parts23["transport"] = t23["transport"]
parts50 = pd.DataFrame(index=C)
parts50["res_heat"] = t50["residential heat/cooling/cooking"]
parts50["ser_heat"] = ser_heat23 * fac["services"]
parts50["transport"] = t50["transport"]
parts50["industry_extra"] = t50["industry+other"] - t23["industry+other"]
# everything else (appliances, services excl. heat, agriculture, today's industry)
rest50 = (t50["residential appliances"] + (t50["services"] - parts50["ser_heat"])
          + t50["agriculture"] + t23["industry+other"])
total50 = t50["sum sectors"]


def by_node(series_twh, key):
    """Country TWh -> node MWh using a within-country key."""
    return series_twh.reindex(ctry).values * 1e6 * key


def heat_profile(c_twh_res, c_twh_ser, year):
    out = 0
    for sector, c_twh in [("residential", c_twh_res), ("services", c_twh_ser)]:
        s = res_space_share if sector == "residential" else ser_space_share
        s = s.reindex(ctry).fillna(0.5).values
        annual = by_node(c_twh, pop)
        out = out + elec_shape(space_shape[sector], HP_SHARE["space"][year]) * (annual * s)
        out = out + elec_shape(water_shape[sector], HP_SHARE["water"][year]) * (annual * (1 - s))
    return out


# ---------------------------------------------------------------- today: remove heat and transport
heat23 = heat_profile(parts23["res_heat"], parts23["ser_heat"], "today")
tr23 = tr_shape * by_node(parts23["transport"], tr_share)
base = (load - heat23 - tr23)
neg_share = (base < 0).sum().sum() / base.size
base = base.clip(lower=0)

# ---------------------------------------------------------------- 2050
f_rest = rest50 / (base.sum().groupby(ctry).sum() / 1e6)
L50 = base * f_rest.reindex(ctry).values
heat50 = heat_profile(parts50["res_heat"], parts50["ser_heat"], "2050")
tr50 = tr_shape * by_node(parts50["transport"], tr_share)
ind50 = pd.DataFrame(1 / len(time), index=time, columns=nodes) * by_node(parts50["industry_extra"], ind_key)
L50 = L50 + heat50 + tr50 + ind50

# rescale each country to its annual total from build_2050_country_demand.py
scale = total50 / (L50.sum().groupby(ctry).sum() / 1e6)
L50 = L50 * scale.reindex(ctry).values

# ---------------------------------------------------------------- outputs
da = xr.DataArray(L50.values, coords={"time": time, "bus": nodes}, dims=["time", "bus"],
                  name="electricity demand (MW)")
da.to_dataset().to_netcdf(out_dir / "electricity_demand_2050.nc")

summary = pd.DataFrame({
    "today_TWh": load.sum() / 1e6,
    "2050_TWh": L50.sum() / 1e6,
    "today_peak_GW": load.max() / 1e3,
    "2050_peak_GW": L50.max() / 1e3,
    "heat_2050_TWh": heat50.sum() / 1e6,
    "transport_2050_TWh": tr50.sum() / 1e6,
    "industry_extra_2050_TWh": ind50.sum() / 1e6,
})
summary.round(3).to_csv(out_dir / "nodal_summary_2050.csv")
csum = summary.groupby(ctry).sum()
csum["growth"] = csum["2050_TWh"] / csum["today_TWh"]
csum["peak_growth"] = csum["2050_peak_GW"] / csum["today_peak_GW"]
csum.round(3).to_csv(out_dir / "country_summary_2050.csv")

print(f"Hours where today's load minus heat/transport went negative (set to 0): {neg_share:.2%}")
print("Rescale factors per country (should be close to 1):")
print(scale.round(3).describe())
pd.set_option("display.width", 200)
print(csum[["today_TWh", "2050_TWh", "growth", "today_peak_GW", "2050_peak_GW", "peak_growth"]].round(1))
print("Total TWh today / 2050:", round(load.sum().sum() / 1e6), "/", round(L50.sum().sum() / 1e6))
