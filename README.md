# aiida-sssp-workflow

## bands distance compare specitication

### bands distance of magnetic structures

The bands of magnetic structure has one more dimension to distinguish up and down spins.
In bands distance comparing, I simply reduce the array along the last axis e.g. does not distinguish the spins but
merge eigenvalues (sorted) of the same kpoints.

## Parameters of protocol

### Specific parameters for magnetic elements

For bulk structure (Diamond configuration in this case) calculation in convergence verification.
The spin polarization is on for magnetic elements Fe, Mn, Cr, Co, Ni where
the start magnetizations are set to [0.5, -0.4] respectively for each sites of diamond cell.

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
And for lanthanides using the Nitrides from Wenzowech paper.
Using the uniaries/diamond configurations for convergence verification.

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
