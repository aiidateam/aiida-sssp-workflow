# aiida-sssp-workflow

## Criteria and protocol of pressure convergence

Since the magnitude of pressure (P = 1/3 Tr(sigma)) directly output from DFT calcultion depends on the stiffness of the material strongly.
We convert it into an equivalent volume.
This what we call it residual volume is therefore a stiffness-agnostic value that can be used to set criteria for all elements.

## bands distance compare specitication

### bands distance of magnetic structures

The bands of magnetic structure has one more dimension to distinguish up and down spins.
In bands distance comparing, I simply reduce the array along the last axis e.g. does not distinguish the spins but
merge eigenvalues (sorted) of the same kpoints.

### bands distance protocol specifications

There are three kinds of kpoint distance (`kpoints_distance_scf`, `kpoints_distance_bands` and `kpoints_distance_band_strcuture` respectively) for bands measure workflow and two for bands distance convergence workflow where the band structure is not calculated in convergence verification.

In the production protocol, the `kpoints_distance_bands` is set to `0.25` which is not so dense as scf calculation since in the bands nscf calculation the symmetry is not applied (as discussed following.) which lead to the calculation very time consuming if the number of kpoints are enourmous.

We choose a uniform k-grid for bands distance comparison,
in the full Brillouin zone and with no symmetry reduction.
Because, choosing a high-symmetry path could result in an unsatisfactory arbitrary choice,
as different recipes for the standardisation of paths have been introduced in the recent literature and interesting features of the band structure may occur far from the high-symmetry lines (such as Weyl points).
A uniform mesh is also more appropriate from the point of view of electron’s nearsightedness
if the energy eigenvalues are known on a sufficiently fine uniform k-points mesh,
it is possible to get an exact real-space representation of the Hamiltonian in a Wannier function basis32 and then interpolate to an arbitrary fine mesh.

## Parameters of protocol

### Specific parameters for magnetic elements

For atomic energy calculated for cohesive energy, it is hard to converge for magnetic elements with untreated parameters.
The following extra parameters are added for the calculation:

```
"SYSTEM": {
    "nspin": 2,
    "starting_magnetization": {
        self.ctx.element: 0.5,
    },
},
"ELECTRONS": {
    "diagonalization": 'cg',
    "mixing_beta": 0.5,
    "electron_maxstep": 200,
},
```

## Resource options and parallelzation

### Walltime settings

The max wallclock seconds are set from the `options` input parameters from verification workflow.
This option will then pass to all the inside pw and ph calculation process as the `metadata.options` setting.
The `options` dict has the format of:

```python
{
    "resources": {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 32,
    },
    "max_wallclock_seconds": 1800,  # 30 min
    "withmpi": True,
}

```
where the `max_wallclock_seconds` is exactly used for pw calculation while for ph calculation the value is set to 4 time of value since the ph calculation roughly estimated to elapse 4 times slower that pw calculation of the corresponding pw calculation to finish.


## Configurations

Different verifications use different structure configurations.
For all delta measure verification, the configurations are all compatible with the ACWF.
While for bands measure and convergence workflow, the configurations used are set in file `statics/cif/mapping.json`.
The principles are for bands measure, using the configurations from Cottiner's paper since they are the groud state structures exist in real wolrd.
And for lanthanoids using the Nitrides from Wenzowitch paper for bands measure and convergence otherwise hard to converge in scf calculation.

But the structures used for convergence verfication are varias.
The lanthanides still using the nitrides from Wenzovitch paper.
We keep on using typical nature configuration from Cottiner's paper, but convert them to primitive with pymatgen (for magnetic elements, no primitive convert process but still refine with `pymatgen`).
Maybe the flourine (F) still have thekproblem mentioned in legacy SSSP that hard to convergence (SCF convergence), will rollback to use SiF4 for it.


## For maintainers

To create a new release, clone the repository, install development dependencies with `pip install '.[dev]'`, and then execute `bumpver update`.
This will:

  1. Create a tagged release with bumped version and push it to the repository.
  2. Trigger a GitHub actions workflow that creates a GitHub release.

Additional notes:

  - Use the `--dry` option to preview the release change.
  - The release tag (e.g. a/b/rc) is determined from the last release.
    Use the `--tag` option to switch the release tag.

## License

MIT


## Contact

📧 email: jusong.yu@epfl.ch
