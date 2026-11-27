"""
Felles stil for alle figurer og tabeller i thesis-rapporten.

Bruk:
    import sys; sys.path.append("thesis-tools")
    import plot_style as ps
    ps.apply_style()

    fig, ax = ps.new_fig("full")
    ...
    ps.save_fig(fig, "8_1_5_price_duration_base")

    ps.save_table(df, "8_1_2_total_cost", caption="...", label="tab:total_cost")

Regler:
- Farger hentes ALLTID herfra. Aldri skriv hex-koder i egne skript.
- En farge følger en teknologi/apparatgruppe/case, aldri rekkefølgen i plottet.
- Med DR = blå, uten DR = grå. Overalt.
- Sensitivitetsnivåer (lav -> høy) = én farge fra lys til mørk, ikke regnbue.
- Ingen to y-akser. To størrelser med ulik skala = to figurer.
- Figurer lagres som PDF (vektor) i figures/, tabeller som .tex i tables/.

Kjør `python thesis-tools/plot_style.py` for å lage eksempel-figurer med falske data.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Mapper (relativt til der skriptet kjøres fra, vanligvis repo-roten)
# ---------------------------------------------------------------------------
FIG_DIR = Path("thesis-tools/figures")
TABLE_DIR = Path("thesis-tools/tables")

# ---------------------------------------------------------------------------
# Størrelser. TEXTWIDTH må matche \the\textwidth i Overleaf (pt / 72.27 = tommer)
# ---------------------------------------------------------------------------
TEXTWIDTH_IN = 6.3  # TODO: sjekk mot Overleaf og oppdater
SIZES = {
    "full": (TEXTWIDTH_IN, TEXTWIDTH_IN * 0.45),   # standard, hele bredden
    "tall": (TEXTWIDTH_IN, TEXTWIDTH_IN * 0.65),   # dispatch, mange land
    "half": (TEXTWIDTH_IN * 0.49, TEXTWIDTH_IN * 0.40),  # to side om side
}

# ---------------------------------------------------------------------------
# Farger. Validert for fargeblindhet i rekkefølgene under (hvit bakgrunn).
# ---------------------------------------------------------------------------
DR = "#3f6fb0"       # med DR
NO_DR = "#a6a6a6"    # uten DR
TEXT = "#2b2b2b"
TEXT_MUTED = "#666666"
GRID = "#e6e6e6"

# Teknologigrupper. Rekkefølgen er stablerekkefølgen (nederst -> øverst).
# Ikke bytt rekkefølge, da kommer like farger ved siden av hverandre.
TECH_COLORS = {
    "Nuclear": "#8a6bbf",
    "Coal & lignite": "#b4553f",
    "Hydro": "#1c9bb0",
    "Biomass & other": "#5a9a3e",
    "Wind": "#4a7fc1",
    "Gas": "#e0874a",
    "Storage": "#c9679a",
    "Solar": "#dca52a",
}
IMPORT_COLOR = "#c8c8c8"  # nøytral, alltid med skravur
IMPORT_HATCH = "////"

# PyPSA-Eur carrier -> gruppe. Legg til nye carriers her hvis advarsel dukker opp.
CARRIER_GROUPS = {
    "nuclear": "Nuclear",
    "coal": "Coal & lignite", "lignite": "Coal & lignite", "oil": "Coal & lignite",
    "biomass": "Biomass & other", "geothermal": "Biomass & other", "waste": "Biomass & other",
    "ror": "Hydro", "hydro": "Hydro",
    "CCGT": "Gas", "OCGT": "Gas",
    "onwind": "Wind", "offwind-ac": "Wind", "offwind-dc": "Wind", "offwind-float": "Wind",
    "solar": "Solar", "solar-hsat": "Solar", "solar rooftop": "Solar",
    "PHS": "Storage", "battery": "Storage", "H2": "Storage",
}

# Apparatgrupper (5 farger, validert i denne rekkefølgen).
APPLIANCE_GROUP_COLORS = {
    "Heating": "#4a7fc1",
    "Wet appliances": "#e0874a",
    "Cold appliances": "#1c9bb0",
    "EV": "#dca52a",
    "Cooling": "#c9679a",
}
APPLIANCE_GROUPS = {
    "SH": "Heating", "WH": "Heating", "HP": "Heating", "CP": "Heating",
    "WM": "Wet appliances", "DW": "Wet appliances", "TD": "Wet appliances",
    "RF": "Cold appliances", "FR": "Cold appliances",
    "EV": "EV",
    "AC": "Cooling",
}
APPLIANCE_NAMES = {
    "SH": "Space heating", "WH": "Water heater", "HP": "Heat pump", "CP": "Circulation pump",
    "WM": "Washing machine", "DW": "Dishwasher", "TD": "Tumble dryer",
    "RF": "Refrigerator", "FR": "Freezer", "EV": "Electric vehicle", "AC": "Air conditioning",
}

# Faste case-navn i figurer og tabeller. Nøkkel = navnet i resultatloggen.
# TODO: tilpass nøklene til de faktiske kjøringsnavnene i resultatloggen.
CASE_LABELS = {
    "base": "Base",
    "part_mid": "Participation: middle",
    "part_100": "Participation: 100 %",
    "shift_low": "Shift time: low",
    "shift_high": "Shift time: high",
    "weather_low": "Weather: low RES",
    "weather_high": "Weather: high RES",
    "ac_half": "Activation cost: 0.5×",
    "ac_1x": "Activation cost: 1×",
    "ac_4x": "Activation cost: 4×",
    "ac_high": "Activation cost: high",
    "2050": "2050",
}


def sequential(n: int, hue: str = "Blues") -> list:
    """n farger fra lys til mørk, for sensitivitetsnivåer (lav -> høy)."""
    cmap = mpl.colormaps[hue]
    return [mpl.colors.to_hex(cmap(x)) for x in np.linspace(0.35, 0.9, n)]


# ---------------------------------------------------------------------------
# Stil
# ---------------------------------------------------------------------------
def apply_style(serif: bool = False) -> None:
    """serif=True gir samme skrift som LaTeX-teksten. Standard er en ren sans-serif."""
    mpl.rcParams.update({
        "font.family": "serif" if serif else "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "cm" if serif else "dejavusans",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": "#b0b0b0",
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "text.color": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "axes.spines.left": False,
        "axes.axisbelow": True,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def new_fig(size: str = "full", nrows: int = 1, ncols: int = 1, **kwargs):
    return plt.subplots(nrows, ncols, figsize=SIZES[size], **kwargs)


def save_fig(fig, name: str, png: bool = False) -> Path:
    """Lagre som PDF (til Overleaf). png=True lager også en PNG for Notion/Slack."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    if png:
        fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    return path


def save_table(df: pd.DataFrame, name: str, caption: str, label: str,
               decimals: int | dict = 1, index: bool = True) -> Path:
    """Lagre som booktabs-tabell (.tex). Krever \\usepackage{booktabs} i Overleaf.
    Sett enhet i kolonnenavnet, f.eks. 'Total cost [M€]'."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(decimals, dict):
        fmt = {c: f"{{:,.{d}f}}" for c, d in decimals.items()}
    else:
        fmt = {c: f"{{:,.{decimals}f}}" for c in df.select_dtypes("number").columns}
    styler = df.style.format(fmt)
    if not index:
        styler = styler.hide(axis="index")
    tex = styler.to_latex(hrules=True, caption=caption, label=label, position="htbp",
                  position_float="centering")
    tex = tex.replace("%", r"\%").replace("€", r"\euro{}")  # krever \usepackage{eurosym}
    path = TABLE_DIR / f"{name}.tex"
    path.write_text(tex)
    return path


# ---------------------------------------------------------------------------
# Hjelpefunksjoner for data
# ---------------------------------------------------------------------------
def group_carriers(df: pd.DataFrame) -> pd.DataFrame:
    """Kolonner = PyPSA-carriers -> kolonner = teknologigrupper, i stablerekkefølge."""
    unknown = [c for c in df.columns if c not in CARRIER_GROUPS]
    if unknown:
        warnings.warn(f"Ukjente carriers lagt i 'Biomass & other': {unknown}. "
                      "Legg dem til i CARRIER_GROUPS.")
    grouped = df.T.groupby(lambda c: CARRIER_GROUPS.get(c, "Biomass & other")).sum().T
    return grouped.reindex(columns=[g for g in TECH_COLORS if g in grouped.columns])


def group_appliances(df: pd.DataFrame) -> pd.DataFrame:
    """Kolonner = apparatkoder (WM, SH, ...) -> kolonner = apparatgrupper."""
    grouped = df.T.groupby(lambda c: APPLIANCE_GROUPS[c]).sum().T
    return grouped.reindex(columns=[g for g in APPLIANCE_GROUP_COLORS if g in grouped.columns])


# ---------------------------------------------------------------------------
# Standardfigurer
# ---------------------------------------------------------------------------
def price_duration_curve(prices_dr: pd.Series, prices_nodr: pd.Series, ax=None,
                         ylabel: str = "Price [€/MWh]"):
    """Timepriser sortert fra høyest til lavest, med og uten DR."""
    if ax is None:
        _, ax = new_fig("full")
    hours = np.arange(1, len(prices_dr) + 1)
    ax.plot(hours, np.sort(prices_nodr.values)[::-1], color=NO_DR, label="Without DR")
    ax.plot(hours, np.sort(prices_dr.values)[::-1], color=DR, label="With DR")
    ax.set_xlim(1, len(hours))
    ax.set_xlabel("Hours of the year, sorted")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    return ax


def price_timeseries(prices_dr: pd.Series, prices_nodr: pd.Series, ax=None,
                     ylabel: str = "Price [€/MWh]"):
    """Kronologiske priser med og uten DR (indeks = tidspunkt)."""
    if ax is None:
        _, ax = new_fig("full")
    ax.plot(prices_nodr.index, prices_nodr.values, color=NO_DR, label="Without DR")
    ax.plot(prices_dr.index, prices_dr.values, color=DR, label="With DR")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", ncols=2)
    fig = ax.get_figure()
    fig.autofmt_xdate()
    return ax


def grouped_bar(df: pd.DataFrame, ax=None, ylabel: str = "Shifted energy [GWh]",
                colors: dict | None = None, sort: bool = True):
    """Søyler per land/apparat (rader) for 1-2 case (kolonner), f.eks. base + ett case.
    Kolonnenavn skal være nøkler i CASE_LABELS."""
    if ax is None:
        _, ax = new_fig("full")
    if sort:
        df = df.sort_values(df.columns[0], ascending=False)
    n = len(df.columns)
    width = 0.8 / n
    x = np.arange(len(df))
    default = [DR, "#a9c3e6"]  # base = mørk blå, ekstra case = lys blå
    for i, col in enumerate(df.columns):
        c = (colors or {}).get(col, default[i % 2])
        ax.bar(x + (i - (n - 1) / 2) * width, df[col].values, width * 0.92,
               color=c, label=CASE_LABELS.get(col, col))
    ax.set_xticks(x, df.index, rotation=90 if len(df) > 12 else 0)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    if n > 1:
        ax.legend(loc="upper right")
    return ax


def dispatch_plot(gen: pd.DataFrame, demand: pd.Series, demand_nodr: pd.Series | None = None,
                  imports: pd.Series | None = None, exports: pd.Series | None = None,
                  ax=None, ylabel: str = "Power [GW]"):
    """Drift for én node over noen dager.
    gen: kolonner = teknologigrupper (bruk group_carriers), verdier >= 0
    demand: total etterspørsel MED DR (svart linje)
    demand_nodr: etterspørsel uten DR (stiplet), forskjellen = DR-aktivitet
    imports/exports: positive serier; eksport tegnes under null
    """
    if ax is None:
        _, ax = new_fig("tall")
    cols = [g for g in TECH_COLORS if g in gen.columns]
    layers = [gen[c].values for c in cols]
    colors = [TECH_COLORS[c] for c in cols]
    labels = cols
    if imports is not None:
        layers.append(imports.values)
        colors.append(IMPORT_COLOR)
        labels = labels + ["Imports"]
    polys = ax.stackplot(gen.index, *layers, colors=colors, labels=labels,
                         edgecolor="white", linewidth=0.3)
    if imports is not None:
        polys[-1].set_hatch(IMPORT_HATCH)
    if exports is not None:
        ax.fill_between(exports.index, 0, -exports.values, facecolor=IMPORT_COLOR,
                        hatch="xxx", edgecolor="white", linewidth=0.3, label="Exports")
    ax.plot(demand.index, demand.values, color=TEXT, lw=1.4, label="Demand with DR")
    if demand_nodr is not None:
        ax.plot(demand_nodr.index, demand_nodr.values, color=TEXT, lw=1.2, ls="--",
                label="Demand without DR")
    ax.axhline(0, color="#b0b0b0", lw=0.6)
    ax.set_ylabel(ylabel)
    # Legenden i samme rekkefølge som stabelen (øverst i stabelen = øverst i legenden)
    handles, labels_ = ax.get_legend_handles_labels()
    n_stack = len(polys)
    order = list(range(n_stack))[::-1] + list(range(n_stack, len(handles)))
    ax.legend([handles[i] for i in order], [labels_[i] for i in order],
              loc="upper left", bbox_to_anchor=(1.01, 1), ncols=1)
    ax.margins(x=0)
    ax.get_figure().autofmt_xdate()
    return ax


# ---------------------------------------------------------------------------
# Demo med falske data: python thesis-tools/plot_style.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(1)

    # Varighetskurve
    p_nodr = pd.Series(np.clip(rng.gamma(4, 18, 8760), 0, 400))
    p_dr = p_nodr * 0.97 + (p_nodr.mean() - p_nodr) * 0.08
    ax = price_duration_curve(p_dr, p_nodr)
    save_fig(ax.get_figure(), "demo_price_duration", png=True)

    # DR per land, base + ett case
    countries = ["DE", "FR", "ES", "IT", "PL", "NL", "BE", "SE", "NO", "AT", "DK", "FI", "CZ", "PT"]
    df = pd.DataFrame({"base": rng.gamma(2, 400, len(countries))}, index=countries)
    df["part_100"] = df["base"] * rng.uniform(1.6, 2.4, len(countries))
    ax = grouped_bar(df)
    save_fig(ax.get_figure(), "demo_dr_per_country", png=True)

    # Dispatch to dager
    t = pd.date_range("2013-01-15", periods=48, freq="h")
    h = np.arange(48)
    solar = np.clip(np.sin((h % 24 - 6) / 12 * np.pi), 0, None) * 6
    gen = pd.DataFrame({"nuclear": 8 + 0 * h, "lignite": 3 + 0.5 * np.sin(h / 5),
                        "biomass": 1.5 + 0 * h, "ror": 1 + 0 * h,
                        "CCGT": 2 + 1.5 * np.sin(h / 3.8) ** 2, "onwind": 4 + 2 * np.cos(h / 9),
                        "solar": solar, "PHS": np.clip(np.sin((h % 24 - 17) / 3), 0, None) * 1.5},
                       index=t)
    g = group_carriers(gen)
    imp = pd.Series(np.clip(1.5 * np.sin(h / 6), 0, None), index=t)
    exp = pd.Series(np.clip(-1.5 * np.sin(h / 6), 0, None), index=t)
    dem_nodr = g.sum(axis=1) + imp - exp
    shift = 0.8 * np.sin((h % 24 - 9) / 24 * 2 * np.pi)
    dem = dem_nodr + shift
    ax = dispatch_plot(g, dem, dem_nodr, imp, exp)
    save_fig(ax.get_figure(), "demo_dispatch", png=True)

    # Tabell
    tab = pd.DataFrame({"Total cost [M€]": [95000.0, 94700.0, 94300.0],
                        "Change vs base [%]": [0.0, -0.32, -0.74]},
                       index=[CASE_LABELS[c] for c in ["base", "part_mid", "part_100"]])
    save_table(tab, "demo_total_cost", caption="Total system cost per case.",
               label="tab:demo_total_cost", decimals={"Total cost [M€]": 0,
                                                       "Change vs base [%]": 2})
    print("Demo lagret i", FIG_DIR, "og", TABLE_DIR)