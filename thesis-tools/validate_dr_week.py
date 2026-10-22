"""
Validate the DR module for one node: plot one week and check the constraints
over the whole year.

Usage (from ~/pypsa-eur-thesis, login node is fine):

    pixi run python thesis-tools/validate_dr_week.py \
        results/BE5_w13_base_dr/networks/solved_2025.nc

Options:
    --bus "BE0 0AC"        node to check (default: first AC bus)
    --week 2013-07-08      Monday the plotted week starts (default: busiest week)
    --out dr_week.png      output figure

How the DR module is stored in the network (per node and device XX):
    Store "<bus> DR XX"       e     = DR_act (>0 advanced, <0 delayed)
    Link  "<bus> DR XX up"    p0    = L_up   (extra consumption from the grid)
    Link  "<bus> DR XX down"  p0    = L_down (reduced consumption, back to the grid)

Checks, over the full year:
    1. L_up and L_down stay within 0 and p_nom * p_max_pu
    2. DR_act stays within e_nom * e_min_pu and e_nom * e_max_pu
    3. Energy balance: DR_act[t] - DR_act[t-1] = L_up[t] - L_down[t]
    4. Shift window (S = shift_time from the config):
         advanced energy at t is paid back by L_down within the next S hours
         delayed  energy at t is caught up by L_up  within the next S hours
       The last S hours of the year are skipped (window runs past the end).
    5. Info only: hours with L_up and L_down active at the same time
       (a sign of chain shifting)
"""

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

TOL = 1e-3  # MW / MWh tolerance for solver noise


def bound(n, comp, attr, name):
    """Time series of a component attribute, falling back to the static value."""
    t = getattr(n, comp + "_t")[attr]
    if name in t.columns:
        return t[name]
    return pd.Series(getattr(n, comp).at[name, attr], index=n.snapshots)


def check_device(n, bus, dev, shift_time):
    store, up, down = f"{bus} DR {dev}", f"{bus} DR {dev} up", f"{bus} DR {dev} down"
    L_up = n.links_t.p0[up]
    L_down = n.links_t.p0[down]
    e = n.stores_t.e[store]
    eff_up = bound(n, "links", "efficiency", up)

    up_max = n.links.at[up, "p_nom"] * bound(n, "links", "p_max_pu", up)
    down_max = n.links.at[down, "p_nom"] * bound(n, "links", "p_max_pu", down)
    e_nom = n.stores.at[store, "e_nom"]
    e_hi = e_nom * bound(n, "stores", "e_max_pu", store)
    e_lo = e_nom * bound(n, "stores", "e_min_pu", store)

    res = {"device": dev, "S_h": shift_time}

    # 1. power bounds
    res["up_bound_viol_h"] = int(((L_up > up_max + TOL) | (L_up < -TOL)).sum())
    res["down_bound_viol_h"] = int(((L_down > down_max + TOL) | (L_down < -TOL)).sum())

    # 2. energy bounds
    res["DRact_bound_viol_h"] = int(((e > e_hi + TOL) | (e < e_lo - TOL)).sum())

    # 3. energy balance (first hour compared with the last if cyclic, else e_initial)
    e_prev = e.shift(1)
    if n.stores.at[store, "e_cyclic"]:
        e_prev.iloc[0] = e.iloc[-1]
    else:
        e_prev.iloc[0] = n.stores.at[store, "e_initial"]
    resid = (e - e_prev) - (L_up * eff_up - L_down)
    res["balance_max_err"] = round(float(resid.abs().max()), 4)

    # 4. shift window
    S = int(shift_time)
    down_next = L_down[::-1].rolling(S, min_periods=1).sum()[::-1].shift(-1)
    up_next = (L_up * eff_up)[::-1].rolling(S, min_periods=1).sum()[::-1].shift(-1)
    valid = slice(None, len(e) - S)
    adv = e.clip(lower=0).iloc[valid]
    dly = (-e).clip(lower=0).iloc[valid]
    adv_viol = adv > down_next.iloc[valid] + TOL
    dly_viol = dly > up_next.iloc[valid] + TOL
    res["advance_window_viol_h"] = int(adv_viol.sum())
    res["delay_window_viol_h"] = int(dly_viol.sum())

    # 5. info
    res["simultaneous_up_down_h"] = int(((L_up > TOL) & (L_down > TOL)).sum())
    res["shifted_up_GWh"] = round(float(L_up.sum()) / 1e3, 2)
    res["shifted_down_GWh"] = round(float(L_down.sum()) / 1e3, 2)
    res["max_advanced_MWh"] = round(float(e.max()), 2)
    res["max_delayed_MWh"] = round(float(-e.min()), 2)

    series = dict(L_up=L_up, L_down=L_down, e=e, up_max=up_max, down_max=down_max)
    return res, series


def main():
    p = argparse.ArgumentParser()
    p.add_argument("network")
    p.add_argument("--bus", default=None)
    p.add_argument("--week", default=None)
    p.add_argument("--out", default="dr_week.png")
    args = p.parse_args()

    n = pypsa.Network(args.network)
    bus = args.bus or n.buses.index[n.buses.carrier == "AC"][0]
    devices_cfg = (n.meta or {}).get("load", {}).get("demand_response", {}).get("devices", {})

    prefix = f"{bus} DR "
    devices = sorted(s[len(prefix):] for s in n.stores.index if s.startswith(prefix))
    if not devices:
        raise SystemExit(f"No DR stores found at bus '{bus}'.")

    results, data = [], {}
    for dev in devices:
        S = devices_cfg.get(dev, {}).get("shift_time")
        if S is None:
            print(f"WARNING: no shift_time for {dev} in n.meta, window check uses S=1")
            S = 1
        r, s = check_device(n, bus, dev, S)
        results.append(r)
        data[dev] = s

    table = pd.DataFrame(results).set_index("device")
    viol_cols = ["up_bound_viol_h", "down_bound_viol_h", "DRact_bound_viol_h",
                 "advance_window_viol_h", "delay_window_viol_h"]
    print(f"\nDR validation for bus {bus} (full year, {len(n.snapshots)} hours)\n")
    print(table.to_string())

    ok = (table[viol_cols].sum().sum() == 0) and (table.balance_max_err.max() < 0.01)
    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "VIOLATIONS FOUND, see table above")

    out_csv = args.out.rsplit(".", 1)[0] + "_checks.csv"
    table.to_csv(out_csv)

    # pick the week
    if args.week:
        start = pd.Timestamp(args.week)
    else:
        activity = sum(d["L_up"] + d["L_down"] for d in data.values())
        weekly = activity.rolling(7 * 24).sum()
        start = weekly.idxmax() - pd.Timedelta(hours=7 * 24 - 1)
    end = start + pd.Timedelta(hours=7 * 24 - 1)
    sl = slice(start, end)
    print(f"\nPlotted week: {start:%Y-%m-%d} to {end:%Y-%m-%d}")

    # plot: price on top, one panel per device below
    ncols = 3
    nrows = math.ceil(len(devices) / ncols)
    fig = plt.figure(figsize=(18, 3 + 2.6 * nrows))
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[1.2] + [1] * nrows)

    ax_p = fig.add_subplot(gs[0, :])
    if bus in n.buses_t.marginal_price:
        pr = n.buses_t.marginal_price[bus].loc[sl]
        ax_p.plot(pr.index, pr.values, color="black", lw=1)
    ax_p.set_ylabel("€/MWh")
    ax_p.set_title(f"Electricity price at {bus}")

    for i, dev in enumerate(devices):
        d = data[dev]
        ax = fig.add_subplot(gs[1 + i // ncols, i % ncols], sharex=ax_p)
        idx = d["L_up"].loc[sl].index
        ax.bar(idx, d["L_up"].loc[sl], width=1 / 24, color="tab:green", label="L_up")
        ax.bar(idx, -d["L_down"].loc[sl], width=1 / 24, color="tab:red", label="L_down")
        ax.step(idx, d["up_max"].loc[sl], where="mid", color="tab:green", ls="--", lw=0.8)
        ax.step(idx, -d["down_max"].loc[sl], where="mid", color="tab:red", ls="--", lw=0.8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("MW")
        ax2 = ax.twinx()
        ax2.plot(idx, d["e"].loc[sl], color="tab:blue", lw=1.2, label="DR_act")
        ax2.set_ylabel("MWh", color="tab:blue")
        r = table.loc[dev]
        flag = "" if r[viol_cols].sum() == 0 else "  ⚠ violations"
        ax.set_title(f"{dev} (S={int(r.S_h)} h){flag}", fontsize=10)
        if i == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    fig.suptitle(f"DR operation at {bus}, week of {start:%d %b}  "
                 f"(dashed = max/min limits)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Saved figure to {args.out} and checks to {out_csv}")


if __name__ == "__main__":
    main()