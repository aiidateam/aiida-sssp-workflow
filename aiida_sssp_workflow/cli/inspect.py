#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI to inspect the results of the workflow"""
import json

import click
import matplotlib.pyplot as plt
import numpy as np
from aiida import orm

from aiida_sssp_workflow.cli import cmd_root


def birch_murnaghan(V, E0, V0, B0, B01):
    """
    Return the energy for given volume (V - it can be a vector) according to
    the Birch Murnaghan function with parameters E0,V0,B0,B01.
    """
    r = (V0 / V) ** (2.0 / 3.0)
    return E0 + 9.0 / 16.0 * B0 * V0 * (
        (r - 1.0) ** 3 * B01 + (r - 1.0) ** 2 * (6.0 - 4.0 * r)
    )


def eos_plot(
    ax, ref_ae_data, line_data, energy0, volumes, energies, nu, title="EOS", fontsize=8
):
    """plot EOS result on ax"""
    dense_volume_max = max(volumes)
    dense_volume_min = min(volumes)

    dense_volumes = np.linspace(dense_volume_min, dense_volume_max, 100)

    ref_V0, ref_B0, ref_B01 = ref_ae_data
    V0, B0, B01 = line_data

    ae_eos_fit_energy = birch_murnaghan(
        V=dense_volumes,
        E0=energy0,  # in future update E0 from referece json, where ACWF has E0 stored.
        V0=ref_V0,
        B0=ref_B0,
        B01=ref_B01,
    )
    psp_eos_fit_energy = birch_murnaghan(
        V=dense_volumes,
        E0=energy0,
        V0=V0,
        B0=B0,
        B01=B01,
    )

    # Plot EOS
    ax.tick_params(axis="y", labelsize=6, rotation=45)

    ax.plot(volumes, energies, "ob", label="RAW equation of state")
    ax.plot(dense_volumes, ae_eos_fit_energy, "--r", label="AE")
    ax.axvline(V0, linestyle="--", color="gray")

    ax.plot(dense_volumes, psp_eos_fit_energy, "-b", label=f"Pseudo")
    ax.fill_between(
        dense_volumes,
        ae_eos_fit_energy,
        psp_eos_fit_energy,
        alpha=0.5,
        color="red",
    )

    center_x = (max(volumes) + min(volumes)) / 2
    center_y = (max(energies) + min(energies)) / 2

    # write text of nu value in close middle
    nu = round(nu, 3)
    ax.text(center_x, center_y, f"$\\nu$={nu}")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=fontsize)
    ax.set_xlabel("Cell volume per formula unit ($\\AA^3$)", fontsize=fontsize)
    ax.set_ylabel("$E - TS$ (eV)", fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=fontsize)


@cmd_root.command("inspect")
@click.argument("node")  # The node to inspect, uuid or pk
@click.option("--output", "-o", default="output", help="The output file name")
def inspect(node, output):
    """Inspect the results of the workflow"""
    wf_node = orm.load_node(node)
    if "measure" in wf_node.outputs:
        if "precision" in wf_node.outputs.measure:
            precision = wf_node.outputs.measure.precision

            # print summary of the precision to a json file
            d_str = json.dumps(precision.output_parameters.get_dict(), indent=4)
            with open(f"{output}_precision_summary.json", "w") as f:
                f.write(d_str)

            # if there are 5 plots, need 3 rows, since the output_parametres is in the dict len(precision) / 2 is the number of rows
            rows = len(precision) // 2

            # create a figure with 2 columns and rows rows on a a4 size paper
            fig, axs = plt.subplots(rows, 2, figsize=(8.27, 11.69), dpi=100)

            # Plot EOS curve and save to a pdf file
            i = 0
            for conf, res in precision.items():
                if conf == "output_parameters":
                    continue

                # Plot EOS curve
                try:
                    ref_data = res.output_parameters.get_dict()["reference_ae_V0_B0_B1"]
                except KeyError:
                    # For backward compatibility, where I used a different key name before (It is changed after v4.2.0)
                    ref_data = res.output_parameters.get_dict()[
                        "reference_wien2k_V0_B0_B1"
                    ]

                data = res.output_parameters.get_dict()["birch_murnaghan_results"]

                energy0 = res.eos.output_birch_murnaghan_fit.get_dict()["energy0"]
                volumes = res.eos.output_volume_energy.get_dict()["volumes"]
                energies = res.eos.output_volume_energy.get_dict()["energies"]

                nu = res.output_parameters.get_dict()["rel_errors_vec_length"]
                title = f"EOS of {conf}"

                ax = axs.flat[i]
                eos_plot(
                    ax, ref_data, data, energy0, volumes, energies, nu, title=title
                )

                i += 1

            # fig to pdf
            fig.tight_layout()
            fig.savefig(f"{output}_precision.pdf", bbox_inches="tight")


if __name__ == "__main__":
    inspect()
