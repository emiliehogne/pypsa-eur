# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Extract historic Swiss energy balances in TWh/a from Excel file.
"""

import logging

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def load_sheet(xlsx: pd.ExcelFile, sheet_name: str, index_col: str) -> pd.DataFrame:
    df = xlsx.parse(sheet_name, skiprows=5, index_col=index_col)
    year_columns = [col for col in df.columns if isinstance(col, int)]
    df = df[year_columns].dropna(how="all", axis=0)
    df.index = df.index.astype(str).str.strip()
    return df


def pick(df: pd.DataFrame, *labels: str) -> pd.Series:
    """
    Return the first row label that exists. The BFE file renamed some rows
    between editions (e.g. 'Total Endenergieverbrauch' -> 'Total'), so older
    and newer names are both accepted.
    """
    for label in labels:
        if label in df.index:
            return df.loc[label]
    raise KeyError(f"None of {labels} found. Available rows: {list(df.index)}")


def pick_sum(df: pd.DataFrame, *labels: str) -> pd.Series:
    """Sum of several rows (used where one row was split into two)."""
    missing = [label for label in labels if label not in df.index]
    if missing:
        raise KeyError(f"{missing} not found. Available rows: {list(df.index)}")
    return df.loc[list(labels)].sum()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_swiss_energy_balances")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    xlsx = pd.ExcelFile(snakemake.input.xlsx)

    tab1 = load_sheet(xlsx, "Tabelle1", index_col="Verwendungszweck")

    tab17 = load_sheet(xlsx, "Tabelle17", index_col="Verwendungszweck")

    tab18 = load_sheet(xlsx, "Tabelle18", index_col="Verwendungszweck")

    tab26 = load_sheet(xlsx, "Tabelle26", index_col="Verwendungszweck")

    tab28 = load_sheet(xlsx, "Tabelle28", index_col="Verwendungszweck")

    tab35 = load_sheet(xlsx, "Tabelle35", index_col="Verkehrsträger")

    tab38 = load_sheet(xlsx, "Tabelle38", index_col="Verkehrsträger")

    # residential cooking: old file had one row "Kochen / Geschirrspüler",
    # the 2000-2025 edition splits it into "Kochen" and "Geschirrspüler"
    if "Kochen / Geschirrspüler" in tab17.index:
        residential_cooking = tab17.loc["Kochen / Geschirrspüler"]
    else:
        residential_cooking = pick_sum(tab17, "Kochen", "Geschirrspüler")

    data = {
        "total residential": pick(tab17, "Total Endenergieverbrauch", "Total"),
        "total residential space": pick(tab17, "Raumwärme"),
        "total residential water": pick(tab17, "Warmwasser"),
        "total residential cooking": residential_cooking,
        "electricity residential": pick(tab18, "Total"),
        "electricity residential space": pick(tab18, "Raumwärme"),
        "electricity residential water": pick(tab18, "Warmwasser"),
        "electricity residential cooking": pick(tab18, "Kochherde"),
        "total services": pick(tab26, "Total Endenergie", "Total"),
        "total services space": pick(tab26, "Raumwärme"),
        "total services water": pick(tab26, "Warmwasser"),
        "total services cooking": pick(tab26, "Prozesswärme"),
        "electricity services": pick(tab28, "Total Elektrizität", "Total"),
        "electricity services space": pick(tab28, "Raumwärme"),
        "electricity services water": pick(tab28, "Warmwasser"),
        "electricity services cooking": pick(tab28, "Prozesswärme"),
        "total rail": pick(tab35, "Schiene", "Schiene*"),
        "total road": pick(tab35, "Strasse"),
        "electricity road": pick(tab38, "Strasse"),
        "electricity rail": pick(tab38, "Schiene", "Schiene*"),
        "total domestic aviation": pick(tab35, "Luft (Inland)"),
        "total international aviation": pick(
            tab1, "int. Flugverkehr", "Internationaler Flugverkehr", "int. Flugverkehr*"
        ),
        "total domestic navigation": pick(tab35, "Wasser"),
        "total international navigation": 0,
    }

    df = pd.DataFrame(data)
    df.columns.name = None
    df.index = pd.MultiIndex.from_product([["CH"], df.index], names=["country", "year"])

    # convert PJ/a to TWh/a
    df /= 3.6

    df.to_csv(snakemake.output.csv)