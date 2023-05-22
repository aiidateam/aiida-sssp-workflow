#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract SSSP verification results from AiiDA database.

This script will get verification nodes pk as inputs and export all results into a json file
It is only a template solution for how to store the results for showing in
the aiidalab-sssp app. In the long term, I plan to transfor these json file into mongodb
database and held a API on the dokku server able to be queried for detailed specific field of every pseudopotentials.
For the large data such as bandstructure, they are store on the materials cloud and
for aiidalab-sssp app, the results will be queried and retrieved from there by using restapi.

The json exported from this script will contain the uuid of the verification, and the outputs
from all its output namespaces.

One requirement of this script is when running on new verification node of the same pseudopotential, the outputs need to update with the sub-workflows with latest run but not the whole verification output.

The bands structure of bands data is extracted to a folder the json files with label as basename. The format should be that can directly used for bandsplot.

The bands data and bands parameters for bands distance need to store as one json for distance calculation.

If output json file exist, will abort only if --update flag is set. It will check the ctime and hash and update to the latest results.
Therefore, the logic to run the new verficiation is that the properties only run for necessary workflows but not the already good one.
Since the latest workflow results will always update the old one.
But it is okay and I can still do that.
"""

import json
import os
import tarfile
from pathlib import Path

import aiida
import click
from aiida import orm
from aiida.common import AttributeDict
from monty.json import jsanitize
from tqdm import tqdm

from aiida_sssp_workflow.cli import cmd_root

process_prop_label_mapping = {
    "cohesive_energy": "ConvergenceCohesiveEnergyWorkChain",
    "phonon_frequencies": "ConvergencePhononFrequenciesWorkChain",
    "pressure": "ConvergencePressureWorkChain",
    "bands": "ConvergenceBandsWorkChain",
    "delta": "ConvergenceDeltaWorkChain",
}


def export_bands_structure(band_structure, band_parameters):
    data = json.loads(band_structure._exportcontent("json", comments=False)[0])
    data["fermi_level"] = band_parameters["fermi_energy"]
    data["number_of_electrons"] = band_parameters["number_of_electrons"]
    data["number_of_bands"] = band_parameters["number_of_bands"]

    return jsanitize(data)


def export_bands_data(band_structure: orm.BandsData, band_parameters: orm.Dict):
    bands_arr = band_structure.get_bands()
    kpoints_arr, weights_arr = band_structure.get_kpoints(also_weights=True)

    data = {
        "fermi_level": band_parameters["fermi_energy"],
        "number_of_electrons": band_parameters["number_of_electrons"],
        "number_of_bands": band_parameters["number_of_bands"],
        "bands": bands_arr.tolist(),
        "kpoints": kpoints_arr.tolist(),  # TODO: using override JSON encoder
        "weights": weights_arr.tolist(),
    }

    return data


def _flatten_output(attr_dict, skip: list = []):
    """
    flaten output dict node
    node_collection is a list to accumulate the nodes that not unfolded

    :param skip: is a list of keys (format with parent_key.key) of Dict name that
        will not collected into the json file.

    For output nodes not being expanded, write down the uuid and datatype for future query.
    """
    # do_not_unfold = ["band_parameters", "scf_parameters", "seekpath_parameters"]

    for key, value in attr_dict.items():
        if key in skip:
            continue

        if isinstance(value, AttributeDict):
            # keep on unfold if it is a namespace
            _flatten_output(value, skip)
        elif isinstance(value, orm.Dict):
            attr_dict[key] = value.get_dict()
        elif isinstance(value, orm.Int):
            attr_dict[key] = value.value
        else:
            # node type not handled attach uuid
            attr_dict[key] = {
                "uuid": value.uuid,
                "datatype": type(value),
            }

    # print(archive_uuids)
    return attr_dict


def get_metadata(node):
    return {
        "uuid": node.uuid,
        "ctime": node.ctime,
        "_aiida_hash": node.get_hash(),
    }


@cmd_root.command("extract")
@click.option("element", "-e", help="element")
@click.option(
    "override",
    "--override",
    is_flag=True,
    default=False,
    help="Purge previous results of element, careful to use.",
)
@click.option("--dst", type=click.Path(), help="folder to store the output.")
@click.option(
    "custom_label",
    "--custom-label",
    required=False,
    help="custumrized label as key for pseudos the output json file.",
)
@click.argument("pks", nargs=-1)
def extract(pks, element, dst, override, custom_label):
    """Extract the results from the verification nodes and export to json file."""
    _profile = aiida.load_profile()
    click.echo(f"Profile: {_profile.name}")

    # can't use custom_label with more than one node output
    if custom_label and len(pks) > 1:
        raise ValueError("cant use label with more than one node.")

    if not dst:
        dst = "./sssp_db"

    # if no element provide, element==None, only pack the db to tar file
    if element:
        json_fn = f"{element}.json"
        Path(os.path.join(dst, "bands", element)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dst, "band_structure", element)).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(dst, json_fn)).touch(exist_ok=True)

        # read the results from the <element>.json, if not exist, create the json and set to {}
        try:
            with open(os.path.join(dst, json_fn), "r") as fh:
                curated_result = json.load(fh)
        except json.decoder.JSONDecodeError:
            # newly create empty file
            curated_result = {}

        for pk in tqdm(pks):
            # meta info of such verification
            # uuid, label
            _node = orm.load_node(pk)
            assert _node.get_attribute("process_label") == "VerificationWorkChain"
            label = (
                custom_label or _node.extras.get("label").split()[-1]
            )  # do not contain the extra machine info
            assert element == label.split(".")[0]

            # TODO label check

            # if override or newly run element
            if override or label not in curated_result:
                psp_result = {
                    "_metadata": [get_metadata(_node)],
                }  # the results of one verification
                psp_result["measure"] = {}
                psp_result["convergence"] = {}
            else:
                # append the new _metadata of the output node if it update the previous result
                psp_result = curated_result[label]
                metadata = get_metadata(_node)
                if metadata["_aiida_hash"] in [
                    i.get("_aiida_hash") for i in psp_result["_metadata"]
                ]:
                    click.echo(f"_metadata of {label} is in the list, do nothing.")
                    continue
                else:
                    psp_result["_metadata"] += [get_metadata(_node)]

            for called_wf in _node.called:
                if called_wf.process_label == "parse_pseudo_info":
                    psp_result["pseudo_info"] = {
                        **called_wf.outputs.result.get_dict(),
                    }
                # precision (EOS)
                if called_wf.process_label == "PrecisionMeasureWorkChain":
                    psp_result["measure"]["precision"] = _flatten_output(
                        _node.outputs.measure.precision
                    )
                    psp_result["measure"]["precision"]["_metadata"] = get_metadata(
                        called_wf
                    )
                # bands
                if called_wf.process_label == "BandsMeasureWorkChain":
                    psp_result["measure"]["bands"] = {
                        "bands": f"bands/{element}/{label}.json",
                        "band_structure": f"band_structure/{element}/{label}.json",
                    }
                    psp_result["measure"]["bands"]["_metadata"] = get_metadata(
                        called_wf
                    )

                    try:
                        with open(
                            os.path.join(dst, "bands", element, f"{label}.json"), "w"
                        ) as fh:
                            bands = called_wf.outputs.bands
                            json.dump(
                                export_bands_data(
                                    bands.band_structure, bands.band_parameters
                                ),
                                fh,
                            )
                        with open(
                            os.path.join(
                                dst, "band_structure", element, f"{label}.json"
                            ),
                            "w",
                        ) as fh:
                            bands = called_wf.outputs.band_structure
                            json.dump(
                                export_bands_structure(
                                    bands.band_structure, bands.band_parameters
                                ),
                                fh,
                            )
                    except Exception:
                        # bands not finished okay
                        pass

                # convergence
                for k, v in process_prop_label_mapping.items():
                    if called_wf.process_label == v:
                        try:
                            output_res = _flatten_output(_node.outputs.convergence[k])
                        except KeyError:
                            # run but not finished therefore no output node
                            output_res = {"message": "error"}
                        psp_result["convergence"][k] = output_res
                        psp_result["convergence"][k]["_metadata"] = get_metadata(
                            called_wf
                        )

            curated_result[f"{label}"] = psp_result

        with open(os.path.join(dst, json_fn), "w") as fh:
            json.dump(dict(curated_result), fh, indent=2, sort_keys=True, default=str)

    else:
        # if no element and pks as argument, simply pach for new version
        # # resursive to imply same order and use bz2 since no timestamp that
        # generate unidentical tarball.
        click.echo("Compressing...")
        with tarfile.open(f"{dst}.tar.gz", "w:bz2") as tar:
            tar.add(dst, arcname=os.path.basename(dst), recursive=True)


if __name__ == "__main__":
    run()
