import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.dates as mdates
today = xr.open_dataset(sys.argv[1])["electricity demand (MW)"].to_pandas()
y50 = xr.open_dataset(sys.argv[2])["electricity demand (MW)"].to_pandas()
cc = lambda df, c: df.loc[:, df.columns.str[:2] == c].sum(axis=1) / 1e3
fig, ax = plt.subplots(2, 2, figsize=(12, 6.5), sharey="row")
for i, c in enumerate(["NO", "DE"]):
    for j, (lab, a, b) in enumerate([("Winter week (14–20 Jan)", "2013-01-14", "2013-01-20 23:00"),
                                     ("Summer week (15–21 Jul)", "2013-07-15", "2013-07-21 23:00")]):
        x = ax[i, j]
        x.plot(cc(today, c)[a:b], color="#8a8a8a", lw=1.6, label="Today (2025 load)")
        x.plot(cc(y50, c)[a:b], color="#2a6fdb", lw=1.6, label="2050")
        x.set_title(f"{c} – {lab}", fontsize=11); x.grid(alpha=.3)
        if j == 0: x.set_ylabel("GW")
        x.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        x.tick_params(axis="x", labelsize=8)
ax[0, 0].legend(frameon=False)
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=130)
