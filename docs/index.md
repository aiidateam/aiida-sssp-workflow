# Introduction

This is the documentation for the `aiida-sssp-workflow` plugin and `aiidalab-sssp` AiiDAlab app.
The solid-state pseudopotential (SSSP) libraries are collections of pseudopotentials for solid-state calculations.

The pseudopotentials of the SSSP libraries are carefully validated and optimized to ensure transferability and precision by using the automated verification workflows through the workflow engine [AiiDA](https://aiida.net/) by `aiida-sssp-workflow` plugin.

The `aiidalab-sssp` app based ond the [AiiDAlab](https://aiidalab.net/) provides a user-friendly interface to the SSSP libraries and the verification workflows.

The workflows contains the verification of transferability (precision) and softness of the pseudopotentials.
The precision of the pseudopotentials are measured by comparing the EOS of the solid-state calculation with the all-electron calculation.

The workflows run verification for pseudopotential convergence are for:

 * Delta factor:
 * Cohesive energy:
 * Phonon frequencies:
 * Bands distance:
 * Residual pressure: reflect the precision and the softness of the given pseudopotential.

## Quick start

The verification can be run and inspected by plugin through the command line interface (CLI) or the AiiDAlab app (Go to [GUI App](gui/) for more information).

To run through CLI we provide a docker image to setup working environment without installing the plugin and AiiDA.

First, you need to [install Docker on your workstation or laptop](https://docs.docker.com/get-docker/).

If you are using Linux or MacOS, you can run the following command to run the workflow:

```bash
docker run -it ghcr.io/unkcpz/aiida-sssp-workflow:edge bash
```

If you are using Windows, open the Docker desktop and search for the `aiida-sssp-workflow` image to start the container, then go to "Exec" tab to open a terminal.

Then you can run the following command to run the workflow:

```bash
wget -c https://raw.githubusercontent.com/unkcpz/sssp-verify-scripts/main/libraries-pbe/PAW-PSL1.0.0-high/Si.paw.z_4.ld1.psl.v1.0.0-high.upf
aiida-sssp-workflow launch --property convergence --pw-code pw-7.2@localhost --ph-code ph-7.2@localhost --protocol test --cutoff-control test --criteria efficiency --withmpi True --num-mpiprocs 2 --npool 1 -- Si.paw.z_4.ld1.psl.v1.0.0-high.upf
```

The workflow is run with the Quantum ESPRESSO codes installed in the docker image.
Since the verfication requires a lot of calculations, it is recommended to setup and configure codes on a remote cluster. See AiiDA documentation on [how to run external codes](https://aiida.readthedocs.io/projects/aiida-core/en/latest/howto/run_codes.html) for more information.

To run the convergence workflow change the `--property` option to `convergence`, the option accept multiple values to run on multiple properties.
For example, to run the convergence workflow for cohesive energy and phonon frequencies:

```bash
aiida-sssp-workflow launch --property convergence.phonon_frequencies --property convergence.cohesive_energy --pw-code pw-7.2@localhost --ph-code ph-7.2@localhost --protocol test --cutoff-control test --criteria efficiency --withmpi True -- Si.paw.z_4.ld1.psl.v1.0.0-high.upf
```

Please run `aiida-sssp-workflow launch --help` for more information about the command line interface.

The verification workflows has its node pk as the label, you can use `verdi process list -a -L VerificationWorkChain` to check the status of the workflow.

The outputs can be inspected by generating the report and summary PDF files by run:

```bash
aiida-sssp-workflow inspect <pk>
```

You can use `-o` option to specify the output name.
See `aiida-sssp-workflow inspect --help` for more information.
