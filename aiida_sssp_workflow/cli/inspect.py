#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI to inspect the results of the workflow"""
import json
import random
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from aiida import orm
from aiida.plugins import WorkflowFactory

from aiida_sssp_workflow.cli import cmd_root
from aiida_sssp_workflow.utils import HIGH_DUAL_ELEMENTS, get_protocol


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    REF from https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def cmap(pseudo_info: dict) -> str:
    """Return RGB string of color for given pseudo info
    Hardcoded at the momment.
    """
    if pseudo_info["family"] == "sg15" and pseudo_info["version"] == "v0":
        return "#000000"

    if pseudo_info["family"] == "gbrv":
        return "#4682B4"

    if (
        pseudo_info["family"] == "psl"
        and pseudo_info["type"] == "us"
        and pseudo_info["version"] == "v1.0.0-high"
    ):
        return "#F50E02"

    if (
        pseudo_info["family"] == "psl"
        and pseudo_info["type"] == "us"
        and pseudo_info["version"] == "v1.0.0-low"
    ):
        return lighten_color("#F50E02")

    if (
        pseudo_info["family"] == "psl"
        and pseudo_info["type"] == "paw"
        and pseudo_info["version"] == "v1.0.0-high"
    ):
        return "#008B00"

    if (
        pseudo_info["family"] == "psl"
        and pseudo_info["type"] == "paw"
        and pseudo_info["version"] == "v1.0.0-low"
    ):
        return lighten_color("#008B00")

    if (
        pseudo_info["family"] == "psl"
        and pseudo_info["type"] == "paw"
        and "v0." in pseudo_info["version"]
    ):
        return "#FF00FF"

    if (
        pseudo_info["family"] == "psl"
        and pseudo_info["type"] == "us"
        and "v0." in pseudo_info["version"]
    ):
        return lighten_color("#FF00FF")

    if pseudo_info["family"] == "dojo" and pseudo_info["version"] == "v4-str":
        return "#F9A501"

    if pseudo_info["family"] == "dojo" and pseudo_info["version"] == "v4-std":
        return lighten_color("#F9A501")

    if pseudo_info["family"] == "jth" and pseudo_info["version"] == "v1.1-str":
        return "#00C5ED"

    if pseudo_info["family"] == "jth" and pseudo_info["version"] == "v1.1-std":
        return lighten_color("#00C5ED")

    # TODO: more mapping
    # if a unknow type generate random color based on ascii sum
    ascn = sum([ord(c) for c in pseudo_info["representive_label"]])
    random.seed(ascn)
    return f"#{random.randint(0, 16777215):06x}"


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


def convergence_plot(
    ax,
    convergence_data,
    converged_xy,
    y_thresholds_range,
    y_label,
    title,
):
    xs = convergence_data["xs"]
    ys = convergence_data["ys"]

    # xlim a bit larger than the range of xs
    xlims = (min(xs) * 0.9, max(xs) * 1.02)

    ax.plot(xs, ys, "-x")
    ax.scatter(
        *converged_xy,
        marker="s",
        s=100,
        label=f"Converge at {converged_xy[0]} Ry",
        facecolors="none",
        edgecolors="red",
    )
    ax.fill_between(
        x=xlims,
        y1=y_thresholds_range[0],
        y2=y_thresholds_range[1],
        alpha=0.3,
        color="green",
    )
    ax.legend(loc="upper right", fontsize=8)

    # twice the range of ylimits if the y_thresholds_range just cover the y range
    if y_thresholds_range[1] > max(ys) or y_thresholds_range[1] < max(ys) * 4:
        y_max = y_thresholds_range[1] * 2
        y_min = -0.05 * y_max
        ax.set_ylim(bottom=y_min, top=y_max)

    # change ticks size
    ax.tick_params(axis="both", labelsize=6)

    ax.set_xlim(*xlims)
    ax.set_ylabel(y_label, fontsize=8)
    ax.set_xlabel("Cutoff (Ry)", fontsize=8)
    ax.set_title(title, fontsize=8)


@cmd_root.command("analyze")
@click.argument("group")  # The group to inspect, name or pk of the group
@click.option("--output", "-o", default="output", help="The output file name")
@click.option(
    "--ylimit", "-y", default=0.5, help="The y limit for the plot", type=float
)
def analyze(group, output, ylimit):
    """Render the plot for the given pseudos and measure type."""
    from aiida_sssp_workflow.utils import ACWF_CONFIGURATIONS, parse_label

    # landscape mode
    fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27), dpi=100)

    group_node = orm.load_group(group)

    # Get all nodes in the output group (EOS workflows)
    group_node_query = (
        orm.QueryBuilder()
        .append(orm.Group, filters={"id": group_node.id}, tag="groups")
        .append(orm.Node, project="*", with_group="groups")
    )

    nodes_lst = group_node_query.all(flat=True)
    width = 0.6 / len(nodes_lst)

    for i, node in enumerate(nodes_lst):
        pseudo_info = parse_label(node.base.extras.all["label"].split(" ")[-1])

        # TODO assert element should not change
        element = node.outputs.pseudo_info.get_dict()["element"]

        y_nu = []
        for configuration in ACWF_CONFIGURATIONS:
            res = node.outputs.measure.precision[configuration]["output_parameters"]

            nu = res["rel_errors_vec_length"]
            y_nu.append(nu)

        x = np.arange(len(ACWF_CONFIGURATIONS))

        ax.bar(
            x + width * i,
            y_nu,
            width,
            color=cmap(pseudo_info),
            edgecolor="black",
            linewidth=1,
            label=pseudo_info["representive_label"],
        )
        ax.set_title(f"X={element}")

    d = 0.01  # offset for the text

    ax.axhline(y=0.1, linestyle="--", color="green")
    ax.text(0 - d, 0.1 + d, "0.1 excellent agreement", color="black")
    ax.axhline(y=0.33, linestyle="--", color="red")
    ax.text(0 - d, 0.33 + d, "0.33 good agreement", color="black")
    ax.legend(loc="upper right", prop={"size": 10})
    ax.set_ylabel("ν -factor")
    ax.set_ylim([0, ylimit])

    xticks_shift = len(nodes_lst) * width / 2
    xticks = [i + xticks_shift for i in range(len(ACWF_CONFIGURATIONS))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(ACWF_CONFIGURATIONS)

    # fig to pdf
    fig.tight_layout()
    # fig.suptitle(
    #    f"Comparison on {element}",
    #    fontsize=10,
    # )
    fig.subplots_adjust(top=0.92)
    fpath = Path.cwd() / f"{output}_{element}_fight.pdf"
    fig.savefig(fpath.name, bbox_inches="tight")


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
            try:
                d_str = json.dumps(precision.output_parameters.get_dict(), indent=4)
                with open(f"{output}_precision_summary.json", "w") as f:
                    f.write(d_str)
            except:
                pass

            # if there are 5 plots, need 3 rows, since the output_parametres is in the dict len(precision) / 2 is the number of rows
            rows = len(precision) // 2

            # create a figure with 2 columns and rows rows on a a4 size paper
            fig, axs = plt.subplots(rows, 2, figsize=(8.27, 11.69), dpi=100)

            # Plot EOS curve and save to a pdf file
            i = 0
            for conf, res in precision.items():
                if conf == "output_parameters":
                    continue

                if res.output_parameters.get_dict() == {}:
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

    if "convergence" in wf_node.outputs:
        convergence_summary = {}

        # A4 canvas for plot
        # landscape mode, shoulder to shoulder for ecutwfc and ecutrho for each property
        # five rows for five properties

        # always plot 5 rows if the result is not accessable, add a text to indicate the property is not calculated
        # rows = len(wf_node.outputs.convergence)
        rows = 5
        fig, axs = plt.subplots(
            rows,
            2,
            figsize=(11.69, 8.27),
            dpi=100,
            gridspec_kw={"width_ratios": [3, 1]},
        )
        subplot_index = 0

        for property in [
            "pressure",
            "delta",
            "phonon_frequencies",
            "cohesive_energy",
            "bands",
        ]:
            # plot to the ax
            # ax1 on the left for ecutwfc
            # ax2 on the right for ecutrho
            # the ratio of the width is 3:1
            ax1 = axs.flat[subplot_index]
            ax2 = axs.flat[subplot_index + 1]

            # print summary of the convergence to a json file
            try:
                convergence = wf_node.outputs.convergence[property]
            except KeyError:
                click.secho(
                    f"Property {property} is not calculated for this workflow",
                    fg="red",
                )
                # add to ax1 and ax2 with a red text to indicate the property is not calculated
                ax1.text(
                    0.5,
                    0.5,
                    f"Convergence test of {property} is not run or failed.",
                    color="red",
                )
                ax1.set_axis_off()
                ax2.set_axis_off()
                # jump to the next row
                subplot_index += 2

                continue

            cutoff_control_protocol = wf_node.inputs.convergence.cutoff_control.value
            cutoff_control = get_protocol("control", name=cutoff_control_protocol)
            wfc_scan = cutoff_control["wfc_scan"]
            # See if all the scans are finished by compare the list with the list of control protocol
            ecutwfc_list = convergence.output_parameters_wfc_test.get_dict().get(
                "ecutwfc"
            )
            wfc_scan_healthy = len(ecutwfc_list) / len(wfc_scan)

            # if only first two scan values are not in the list, it is still regarted as 100% healthy
            # Since it is because the first two ecutwfc values are too small for some elements
            if wfc_scan_healthy != 1 and (
                wfc_scan[0] not in ecutwfc_list or wfc_scan[1] not in ecutwfc_list
            ):
                wfc_scan_healthy = 1

            ecutrho_test_list = convergence.output_parameters_rho_test.get_dict().get(
                "ecutrho"
            )

            element = wf_node.outputs.pseudo_info.get_dict().get("element")
            pp_type = wf_node.outputs.pseudo_info.get_dict().get("pp_type")
            if pp_type in ["nc", "sl"]:
                expected_len_dual_scan = cutoff_control["nc_dual_scan"]
            else:
                if element in HIGH_DUAL_ELEMENTS:
                    expected_len_dual_scan = cutoff_control["nonnc_high_dual_scan"]
                else:
                    expected_len_dual_scan = cutoff_control["nonnc_dual_scan"]

            # minus one for the reference value
            rho_scan_healthy = (len(ecutrho_test_list) - 1) / len(
                expected_len_dual_scan
            )

            color = "red" if wfc_scan_healthy != 1 or rho_scan_healthy != 1 else "green"
            click.secho(
                f"Convergence scan healthy check for {property}: wavefunction scan = {round(wfc_scan_healthy*100, 2)}%, charge density scan = {round(rho_scan_healthy*100, 2)}%",
                fg=color,
            )

            # print summary of the convergence to a json file
            # be careful the key for charge density is "chargedensity_cutoff" instead of "charge_density_cutoff
            property_summary = convergence.output_parameters.get_dict()
            property_summary["wfc_scan_healthy"] = wfc_scan_healthy
            property_summary["rho_scan_healthy"] = rho_scan_healthy

            convergence_summary[property] = property_summary

            # data preparation
            # Will only plot the measured properties e.g. for bands it is the eta_c
            _ConvergenceWorkChain = WorkflowFactory(
                f"sssp_workflow.convergence.{property}"
            )
            measured_key = _ConvergenceWorkChain._MEASURE_OUT_PROPERTY

            used_criteria = convergence.output_parameters.get_dict().get(
                "used_criteria"
            )
            crieria_protocol = get_protocol("criteria", name=used_criteria)
            y_thresholds_range = crieria_protocol[property]["bounds"]
            y_unit = crieria_protocol[property]["unit"]
            # use greek letter delta
            y_label = f"Δ {y_unit}"

            conv_data = {}
            conv_data["xs"] = convergence.output_parameters_wfc_test.get_dict().get(
                "ecutwfc"
            )
            conv_data["ys"] = convergence.output_parameters_wfc_test.get_dict().get(
                measured_key
            )

            _x = convergence.output_parameters.get_dict().get("wavefunction_cutoff")
            _y = dict(zip(conv_data["xs"], conv_data["ys"])).get(_x)
            converged_xy = (_x, _y)

            _max_ecutwfc = conv_data["xs"][-1]
            _max_ecutrho = convergence.output_parameters_rho_test.get_dict().get(
                "ecutrho"
            )[-1]
            dual = round(_max_ecutrho / _max_ecutwfc, 1)

            property_name = property.replace("_", " ").capitalize()

            title = f"{property_name} convergence wrt wavefunction cutoff (at charge density cutoff = wavefunction cutoff * {dual} Ry)"

            convergence_plot(
                ax1,
                conv_data,
                converged_xy,
                y_thresholds_range,
                y_label=y_label,
                title=title,
            )

            # data preparation for ecutrho
            conv_data = {}
            conv_data["xs"] = convergence.output_parameters_rho_test.get_dict().get(
                "ecutrho"
            )
            conv_data["ys"] = convergence.output_parameters_rho_test.get_dict().get(
                measured_key
            )

            ecutwfc = convergence.output_parameters.get_dict().get(
                "wavefunction_cutoff"
            )

            _x = convergence.output_parameters.get_dict().get("chargedensity_cutoff")
            _y = dict(zip(conv_data["xs"], conv_data["ys"])).get(_x)
            converged_xy = (_x, _y)

            title = f"charge density cutoff (at wavefunction cutoff {ecutwfc} Ry)"

            convergence_plot(
                ax2,
                conv_data,
                converged_xy,
                y_thresholds_range,
                y_label=y_label,
                title=title,
            )

            # jump to the next row
            subplot_index += 2

        # calculate the recommended cutoffs from the maximum of all properties scan
        recommended_ecutwfc = 0
        recommended_ecutrho = 0
        for value in convergence_summary.values():
            recommended_ecutwfc = max(recommended_ecutwfc, value["wavefunction_cutoff"])
            recommended_ecutrho = max(
                recommended_ecutrho, value["chargedensity_cutoff"]
            )

        convergence_summary["recommended_cutoffs"] = {
            "wavefunction_cutoff": recommended_ecutwfc,
            "chargedensity_cutoff": recommended_ecutrho,
        }

        try:
            d_str = json.dumps(convergence_summary, indent=4)
            with open(f"{output}_convergence_summary.json", "w") as f:
                f.write(d_str)
        except:
            pass

        # fig to pdf
        psp_label = wf_node.base.extras.all["label"].split(" ")[-1]
        criteria = wf_node.inputs.convergence.criteria.value
        fig.tight_layout()
        fig.suptitle(
            f"Convergence verification for {psp_label} under {criteria} creteria",
            fontsize=10,
        )
        fig.subplots_adjust(top=0.92)
        fpath = Path.cwd() / f"{output}_convergence.pdf"
        fig.savefig(fpath.name, bbox_inches="tight")


if __name__ == "__main__":
    inspect()
