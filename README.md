# aiida-sssp-workflow

## Initial dual (E_rho/E_wfc) for convergence test

For NC psp we use dual start from 4, PAW/US use dual from 8 for most of elements but 18 for high dual elements which we found usually high charge density cutoff are needed.
They are O,Fe,Hf, Mn, Co, Ni, Cr.

## Logic of convergence test on different criteria protocol

The wavefunction cutoff scan list are same no matter which criteria protocol is used.
Therefore, the running of wavefunction cutoff test write wavefunction cutoff recommendation for all criteria protocol.
When switching to other criteria protocol, the wavefunction cutoff test can be skipped by setting `preset_ecutwfc`. It will then be used and only run charge density cutoff test.
This is not conflict with the caching workflow, since caching workflow will only run on wfc test the results is that for caching workflow when `preset_ecutwfc` is set, only reference calculation is run.

## Lanthanides

For lanthanides the precision measure is run on nitrides as Wentzcovitch paper and on oxides.
The unaries are not run for precision measurement since it is know that the oxidation state of lanthanides pseudopotentials is not zero.
Only for lanthanide nitrides the magnatization is on.
It is mostly because the reference of RE-N (Wentzcovitch) and RE-O (ACWF) is run with/without magnatization.
The kpoint distance of lanthanide nitrides is hard code to `0.2` (both for precision measure, for delta convergence and first EOS reference calculation of pressure convergence test as well. implemented as `+ 0.1` of precision protocol and `+ 0.05` of convergence protocol from acwf value) using the tetrahedron method rather than as acwf protocol where kpoint distance is `0.10` with fermi-dirac smearing, in order to compatible with Wentzcovitch paper results.
Lanthanide nitrides such as ErN, in equation of state calculation, large (0.06) volume change lead to supurious energy.
To mitigate the issue, the `scale_increment` is set to 0.01 (??? probably only for Er?) to make sure the volume change is in the parabolic range.
The nitrogen pseudopotential is the one from first run on pseudopotentials verifications on nitrigen on libraries include Pslibrary 0.1, Pslibrary 1.0.0, pseudo-dojo, ONCVPSP with legacy sg15 inputs... (Run and check)
The lanthanides have convergence issue if mixing is too large or number of bands is not enough.
For bands measure, the `nbnd_factor` is set to `2.0` while for precision measure and convergence the factor is set to `1.5`.
The mixing is set to 0.5 (default is 0.7 for non-lanthanides), and even smaller for atomic calculation of lanthanides elements which set to 0.3.
The `nbnd` set for lanthanides oxides are set to `1.5` times of the default number of bands since it also has issue that highest bands partially occupied and often retrigger the error handler to restart the calculation.

## Criteria and protocol of pressure convergence

Since the magnitude of pressure (P = 1/3 Tr(sigma)) directly output from DFT calcultion depends on the stiffness of the material strongly.
We convert it into an equivalent volume.
This what we call it residual volume is therefore a stiffness-agnostic value that can be used to set criteria for all elements.

## Delta & nu metric in precision and convergence verificaiton

In convergence delta factor calculation, the GS configurations from Cottiner's paper are used.
To have a uniform sence of precision through looking at delta factor, the value is defined as delta factor per atom.
However, since it is hard to define delta per/atom for oxides, ACWF does not use per/atom value to represent results.
It uses the nu factor which was defined in the ACWF paper.

For the oxides, it needs oxygen pseudopotential first, same as nitrides.
The verification needs to run for all know oxygen/nitrigen pseudopotentials filst. However, the cutoffs of them are not known and the cutoffs are input for the precision measure.
So the very first run is the convergence test on all oxygen/nitrigen pseudopotentials.
Once the recommended cutoffs are known, the precision measure can be run for all oxygen/nitrigen elements.

In the convergence verification, there are many options to choose from for what configuration to use. But it is redundant to use all of them, we what the configuration that gives the largest recommended cutoff energy for the pseudopotential.
After tests (show test results, which in `sssp-verify-scripts`), we found that the for non-magnetic elements, the Diamond configuration that gives the largest cutoff energy. For magnetic elements, the GS configuration calculation with magnetization turned on for the bulk (in cohesive energy which usually give the largest recommended cutoff, where the atomic calculation still don't have magnetization turned on and it is not needed because the pseudopotential is generated in the close shell approximiation (??)) that gives the largest cutoff energy.

## Phonon frequencies specification

For some elements, we have neglected the first n frequencies in the summation above, because the frequencies are negative and/or with strong oscillations as function of the cutoff for all the considered pseudos). We have neglected the first 4 frequencies for H and I, 12 for N and Cl, 6 for O and ??SiF4 (which replaces F)??.
## bands distance compare specification

### bands distance of magnetic structures

The bands of magnetic structure has one more dimension to distinguish up and down spins.
In bands distance comparing, I simply reduce the array along the last axis e.g. does not distinguish the spins but
merge eigenvalues (sorted) of the same kpoints.

### bands distance protocol specifications

There are three kinds of kpoint distance (`kpoints_distance_scf`, `kpoints_distance_bands` and `kpoints_distance_band_strcuture` respectively) for bands measure workflow and two for bands distance convergence workflow where the band structure is not calculated in convergence verification.

In the production protocol, `kpoints_distance_scf` and `kpoints_distance_bands` are set to `0.15`.
We choose a uniform k-grid for bands distance comparison,
in the full Brillouin zone and with symmetry reduction which not implemented in previous version.
After applying symmetry reduction, it is able to compute with more dense grids.
Because, choosing a high-symmetry path could result in an unsatisfactory arbitrary choice,
as different recipes for the standardisation of paths have been introduced in the recent literature and interesting features of the band structure may occur far from the high-symmetry lines (such as Weyl points).
A uniform mesh is also more appropriate from the point of view of electron’s nearsightedness
if the energy eigenvalues are known on a sufficiently fine uniform k-points mesh,
it is possible to get an exact real-space representation of the Hamiltonian in a Wannier function basis and then interpolate to an arbitrary fine mesh.

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

## Resource options and parallelization

## parallelization for atomic calculation

In case of atomic calculation running on a machine with too many CPUs,
the `npool` and `ndiag` is explicitly set to 1 for atomic calculation.
Since there is only one kpoint in atomic case, there is no efficient lost of this parallelization setting.

Because using too many `mpiprocs` (128 in EIGER) in atomic calculation would cause spurious issue the calculation terminate and not being handled.
The maximum number of mpiprocs is set to `32`.

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
where the `max_wallclock_seconds` is exactly used for pw calculation while for ph calculation the value is set to 4 times of value since the ph calculation roughly estimated to elapse 4 times slower that pw calculation of the corresponding pw calculation to finish.
For atomic calculation in cohesive energy evaluation of lanthanides, the `max_wallclock_seconds` also set to 4 times otherwise it may not finished after 5 times of restart.


## Configurations

Different verifications use different structure configurations.
For all precision measure verification, the configurations are all compatible with the ACWF.
While for bands measure and convergence workflow, the configurations used are set in file `statics/cif/mapping.json`.
The principles are for bands measure, using the configurations from Cottiner's paper since they are the groud state structures exist in real wolrd.
And for lanthanoids using the Nitrides from Wenzowitch paper for bands measure and convergence otherwise hard to converge in scf calculation.

But the structures used for convergence verfication are varias.
The lanthanides still using the nitrides from Wenzovitch paper.
We keep on using GS nature configuration from Cottiner's paper, but convert them to primitive with pymatgen (for magnetic elements, no primitive convert process but still refine with `pymatgen`).
Maybe the flourine (F) still have thekproblem mentioned in legacy SSSP that hard to convergence (SCF convergence), will rollback to use SiF4 for it.

## Work chains clean policy

It is very delegate of how to set the remote folder clean in pseudopotential verification, since the verification consist of workflows with lots of sub processes and create huge amount of remote files, although the single calculations are small and resource non-stringent.
If the work chains are not cleaned, the number of remote files will increase rapidly and disk quota in remote machine will soon being run out.
On the contrast, if clean too early would make caching machanism not used and therefore waste resource for same calculations.
To trait off above two side of contradiction, the clean policy is designed as:
- set to the precision measures are cleaned right after its workchain are finished.
- set to clean remote workdir right after the evaluate work chains, but only clean calcjobs that are not cached from. Since (see [#150](https://github.com/aiidateam/aiida-sssp-workflow/pull/150) for detail) cached calcjob node share remote path with its nonmenon, the nonmenon nodes by `_caching` should clean itself in the last.
- set calcjob nodes of `_caching` work chain to be cleaned only by calling from verefication work chain in its terminate step rather than do itself.
For other convergence work chain, they used nodes cached from `_caching` workchain and will clean themself right after it is finished.
The nodes create from other cached nodes (which have `_aiida_cached_from` extra attributes) from convergence work chain never used for furthur caching except for testing mode.

This partially solve the caching issue of bands calculation ([issue #138](https://github.com/aiidateam/aiida-sssp-workflow/issues/138).
But same as ph.x calculation, is subsequent step failed and previous step cleaned, it will still failed to continue.
In phonon, the workaround is to check if the remote folder is empty.
This has the expense of even the ph.x calculation finished ok, the previous step clean will force the scf prepare calculation to run again so to sure the ph.x get its parent_folder not empty.
Also implemented in [#150](https://github.com/aiidateam/aiida-sssp-workflow/pull/150), the force running of `_caching` workflow is applied. To make sure the second run of phonon_frequencies and bands convergence workflow will use the `parant_folder` that are not empty. The expense of such strategy is the `_caching` always need to be run for second run of verification, but since in principle not always to run second time and it bring huge advantage to make the second run more robust.

The clean policy of big verification work chain is controlled by `test_mode`.
It will do clean as described above or do not clean anything so it can be checked afterwards.

## For maintainers

To create a new release, clone the repository, install development dependencies with `pip install '.[dev]'`, and then execute `bumpver update`.
This will:

  1. Create a tagged release with bumped version and push it to the repository.
  2. Trigger a GitHub actions workflow that creates a GitHub release.

Additional notes:

  - Use the `--dry` option to preview the release change.
  - The release tag (e.g. a/b/rc) is determined from the last release.
    Use the `--tag` option to switch the release tag.

### Logger level

The aiida-core logger level is recommened to set to `REPORT` as default.
In workflow, process related messages are pop to daemon logger in REPORT level.
If process finished with none-zero exit_code, log a waring message.
While for debug purpose, the INFO level is for showing the parameters infomations when processes are launched.

### Issues of version

- v3.0.1:
    - nowf not turned on for saving inode purpose, fixed in next version
    - the dual value of NC/SL is 8 for precision measure and bands for norm-conserving pseudopotentials, better to be 4. fixed in next version
    - the dual value for high dual elements is 8 for non-NC pseudopotentials, better to be 16. fixed in next version

## License

MIT


## Contact

📧 email: jusong.yu@psi.ch
