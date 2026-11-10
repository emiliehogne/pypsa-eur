# SPDX-FileCopyrightText: : 2023- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

from scripts._helpers import dr_device_groups

def custom_extra_functionality(n, snapshots, snakemake):
    """
    Add custom extra functionality constraints.
    """
    dr = snakemake.config["load"]["demand_response"]
    devices = dr_device_groups(dr["devices"], dr.get("group_by_shift", False))

    dr_active = n.stores.carrier.isin([f"DR {dev}" for dev in devices]).any()
    if dr_active and not (n.snapshot_weightings.objective == 1).all():
        raise ValueError(
            "Demand response is enabled but snapshots are not hourly "
            "(snapshot_weightings.objective != 1 for at least one snapshot). "
            "The rolling shift_time recovery constraint operates on snapshot "
            "indices, not wall-clock hours, so temporal averaging or "
            "segmentation would silently change the physical meaning of every "
            "device's shift time. Keep snapshots hourly when load.demand_response "
            "is enabled."
        )

    for dev, cfg in devices.items():
        S = cfg["shift_time"]
        DR_i = n.stores.index[n.stores.carrier == f"DR {dev}"]
        if DR_i.empty:
            continue

        up_i = n.links.index[n.links.carrier == f"DR {dev} up"]
        down_i = n.links.index[n.links.carrier == f"DR {dev} down"]

        L_up = n.model["Link-p"].sel(name=up_i)
        L_up = L_up.assign_coords(name=n.links.loc[up_i, "bus0"].values)
        L_down = n.model["Link-p"].sel(name=down_i)
        L_down = L_down.assign_coords(name=n.links.loc[down_i, "bus1"].values)

        DR_act = n.model["Store-e"].sel(name=DR_i)
        DR_act = DR_act.assign_coords(
            name=n.stores.loc[DR_i, "bus"].map(n.buses.location).values
        )

        sum_ups = sum(L_up.roll(snapshot=-k) for k in range(1, S + 1))
        n.model.add_constraints(sum_ups >= -DR_act, name=f"DR-{dev}-delayed")

        sum_downs = sum(L_down.roll(snapshot=-k) for k in range(1, S + 1))
        n.model.add_constraints(sum_downs >= DR_act, name=f"DR-{dev}-advanced")
