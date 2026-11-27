#!/usr/bin/env python
"""
Figurer og tabeller for 8.1.3, 8.1.4, 8.1.5 og 8.1.8 (Emilie).

Kjøres på Mac fra repo-roten, etter at extract_results.py er kjørt på Solstorm
og thesis-tools/data/ er hentet ned:

    python thesis-tools/plots_emilie.py

Leser:   thesis-tools/data/<run>/*.csv
         thesis-tools/dr_value_per_mwh_<base>.csv   (fra dr_value_per_mwh.py)
Skriver: thesis-tools/figures/*.pdf og thesis-tools/tables/*.tex
Kjøringer som mangler blir hoppet over med en melding.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))
import plot_style as ps  # noqa: E402

DATA = Path("thesis-tools/data")
BASE, NODR = ps.BASE, ps.NODR
EXTRA_COUNTRY_CASE = "EU150_w13_phigh_smid_g_dr"  # case nr. 2 i figur 8.1.3
EXTRA_GROUP_CASE = "EU150_w13_plow_shigh_g_dr"    # case nr. 2 i figur 8.1.4
WEEK = ("2013-01-14", "2013-01-21")              # uke for kronologisk prisplott
MIN_DEMAND_TWH = 10                               # små land holdes utenfor prisfigur
# Aktiveringskostnad per MWh flyttet for hver kjøring (base: 0.05 per retning = 0.10)
COST_LEVELS = {
    BASE: 0.10,
    "EU150_w13_plow_smid_g_c6.2_dr": 6.2,
    "EU150_w13_plow_smid_g_c12.4_dr": 12.4,
    "EU150_w13_plow_smid_g_c49.6_dr": 49.6,
}
VALUE_LEVELS = {"0.5×": 6.2, "1× (mean)": 12.4, "4×": 49.6}


def have(run, f="summary.csv"):
    ok = (DATA / run / f).exists()
    if not ok:
        print(f"  mangler {run}/{f}, hopper over")
    return ok


def read(run, f, **kw):
    return pd.read_csv(DATA / run / f, **kw)


def dr_cases():
    return [r for r in ps.CASE_LABELS if (DATA / r / "dr_country_group.csv").exists()]


# ------------------------------------------------------------------ 8.1.3
def dr_per_country():
    print("8.1.3 DR per country")
    rows = {}
    for run in dr_cases():
        dr = read(run, "dr_country_group.csv").groupby("country").shifted_MWh.sum()
        dem = read(run, "demand_country.csv", index_col=0).demand_MWh
        rows[run] = pd.DataFrame({"GWh": dr / 1e3, "pct": dr / dem * 100})
    if BASE not in rows:
        return print("  mangler base, hopper over")

    # Tabell: GWh og % i base, % i de andre casene
    tab = pd.DataFrame({"Base [GWh]": rows[BASE].GWh, "Base [% of demand]": rows[BASE].pct})
    for run, d in rows.items():
        if run != BASE:
            tab[f"{ps.CASE_LABELS[run]} [%]"] = d.pct
    tab = tab.sort_values("Base [GWh]", ascending=False)
    tab.index.name = "Country"
    ps.save_table(tab, "8_1_3_dr_per_country", caption="Shifted energy per country and case.",
                  label="tab:dr_per_country", decimals=1)

    # Figur: % av etterspørsel, base + ett case
    fig_df = pd.DataFrame({BASE: rows[BASE].pct})
    if EXTRA_COUNTRY_CASE in rows:
        fig_df[EXTRA_COUNTRY_CASE] = rows[EXTRA_COUNTRY_CASE].pct
    ax = ps.grouped_bar(fig_df.dropna(), ylabel="Shifted energy [% of demand]")
    ps.save_fig(ax.get_figure(), "8_1_3_dr_per_country", png=True)


# ------------------------------------------------------------------ 8.1.4
def dr_per_group():
    print("8.1.4 DR per DR group")
    rows = {}
    for run in dr_cases():
        d = read(run, "dr_country_group.csv").groupby("group").shifted_MWh.sum()
        rows[run] = d.groupby(ps.dr_group).sum() / 1e6  # TWh
    if BASE not in rows:
        return print("  mangler base, hopper over")
    tab = pd.DataFrame({ps.CASE_LABELS[r]: v for r, v in rows.items()})
    tab = tab.reindex([g for g in ps.DR_GROUP_COLORS if g in tab.index])
    tab.loc["Total"] = tab.sum()
    tab.index.name = "DR group [TWh]"
    ps.save_table(tab.T, "8_1_4_dr_per_group", caption="Shifted energy per DR group and case [TWh].",
                  label="tab:dr_per_group", decimals=2)

    fig_df = pd.DataFrame({BASE: rows[BASE]})
    if EXTRA_GROUP_CASE in rows:
        fig_df[EXTRA_GROUP_CASE] = rows[EXTRA_GROUP_CASE]
    fig_df = fig_df.reindex([g for g in ps.DR_GROUP_COLORS if g in fig_df.index])
    ax = ps.grouped_bar(fig_df, ylabel="Shifted energy [TWh]", sort=False)
    ps.save_fig(ax.get_figure(), "8_1_4_dr_per_group", png=True)

    if have(BASE, "dr_daily_profile.csv"):
        prof = read(BASE, "dr_daily_profile.csv", index_col=0)
        ax = ps.daily_profile(ps.group_dr(prof) / 1e3)
        ps.save_fig(ax.get_figure(), "8_1_4_dr_daily_profile_base", png=True)


# ------------------------------------------------------------------ 8.1.5
def market_prices():
    print("8.1.5 Market prices")
    if not (have(BASE) and have(NODR)):
        return
    p_dr = read(BASE, "prices_country.csv", index_col=0, parse_dates=True)
    p_no = read(NODR, "prices_country.csv", index_col=0, parse_dates=True)

    ax = ps.price_duration_curve(p_dr["EU"], p_no["EU"])
    ps.save_fig(ax.get_figure(), "8_1_5_price_duration_base", png=True)

    # To land: høyest og lavest DR-andel blant land med nok etterspørsel
    dem = read(BASE, "demand_country.csv", index_col=0).demand_MWh
    dr = read(BASE, "dr_country_group.csv").groupby("country").shifted_MWh.sum()
    share = (dr / dem)[dem > MIN_DEMAND_TWH * 1e6].dropna().sort_values()
    pick = [share.index[-1], share.index[0]]
    fig, axes = ps.pair_fig()
    for ax, c in zip(axes, pick):
        sl = slice(*WEEK)
        ps.price_timeseries(p_dr.loc[sl, c], p_no.loc[sl, c], ax=ax)
        ax.get_legend().remove()
        ax.set_title(f"{c} ({share[c] * 100:.1f} % of demand shifted)")
    axes[1].set_ylabel("")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    ps.save_fig(fig, "8_1_5_price_week_" + "_".join(pick), png=True)

    # Prisstatistikk per land
    def stats(p):
        top = p.apply(lambda s: s.nlargest(100).mean())
        return pd.DataFrame({"mean": p.mean(), "std": p.std(), "top100": top})
    a, b = stats(p_no), stats(p_dr)
    tab = pd.DataFrame({
        "Mean w/o DR": a["mean"], "Mean with DR": b["mean"],
        "Std w/o DR": a["std"], "Std with DR": b["std"],
        "Top-100 h w/o DR": a.top100, "Top-100 h with DR": b.top100,
    })
    tab.index.name = "Country [€/MWh]"
    ps.save_table(tab, "8_1_5_price_stats", caption="Electricity price statistics with and "
                  "without DR, base case. Top-100 h is the mean of the 100 most expensive hours.",
                  label="tab:price_stats", decimals=1)


# ------------------------------------------------------------------ 8.1.8
def activation_cost():
    print("8.1.8 Activation cost")
    f = Path(f"thesis-tools/dr_value_per_mwh_{BASE}.csv")
    if f.exists():
        v = pd.read_csv(f)
        ax = ps.histogram_with_levels(v.value_per_MWh, v.shifted_MWh, VALUE_LEVELS)
        ps.save_fig(ax.get_figure(), "8_1_8_value_per_mwh_hist", png=True)
    else:
        print(f"  mangler {f} (kjør dr_value_per_mwh.py på base), hopper over histogram")

    if not have(NODR):
        return
    no = read(NODR, "summary.csv").iloc[0]
    rows = []
    for run, level in COST_LEVELS.items():
        if not have(run):
            continue
        s = read(run, "summary.csv").iloc[0]
        rows.append({
            "Activation cost [€/MWh]": level,
            "Saving excl. activation [M€]": no.opex_MEUR - s.opex_excl_activation_MEUR,
            "Activation payments [M€]": s.activation_MEUR,
            "Net saving [M€]": no.opex_MEUR - s.opex_MEUR,
            "Net saving [%]": (no.opex_MEUR - s.opex_MEUR) / no.opex_MEUR * 100,
            "Shifted [TWh]": s.shifted_TWh,
        })
    if not rows:
        return
    tab = pd.DataFrame(rows).sort_values("Activation cost [€/MWh]")
    ps.save_table(tab, "8_1_8_activation_cost", caption="Results per activation cost level.",
                  label="tab:activation_cost", index=False,
                  decimals={"Activation cost [€/MWh]": 1, "Saving excl. activation [M€]": 1,
                            "Activation payments [M€]": 1, "Net saving [M€]": 1,
                            "Net saving [%]": 2, "Shifted [TWh]": 2})

    fig, (a1, a2) = ps.pair_fig()
    x = range(len(tab))
    lab = [f"{c:g}" for c in tab["Activation cost [€/MWh]"]]
    ps.line_by_level(x, tab["Net saving [M€]"], a1, "Activation cost [€/MWh]",
                     "Net saving [M€]", labels=lab)
    ps.line_by_level(x, tab["Shifted [TWh]"], a2, "Activation cost [€/MWh]",
                     "Shifted energy [TWh]", labels=lab)
    ps.save_fig(fig, "8_1_8_cost_levels", png=True)


if __name__ == "__main__":
    ps.apply_style()
    dr_per_country()
    dr_per_group()
    market_prices()
    activation_cost()
    print("Ferdig. Se thesis-tools/figures og thesis-tools/tables")