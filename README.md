[![Build Status](https://github.com/aiidateam/aiida-sssp-workflow/workflows/ci/badge.svg?branch=master)](https://github.com/aiidateam/aiida-sssp-workflow/actions)
[![Coverage Status](https://coveralls.io/repos/github/aiidateam/aiida-sssp-workflow/badge.svg?branch=master)](https://coveralls.io/github/aiidateam/aiida-sssp-workflow?branch=master)
[![Docs status](https://readthedocs.org/projects/aiida-sssp-workflow/badge)](http://aiida-sssp-workflow.readthedocs.io/)
[![PyPI version](https://badge.fury.io/py/aiida-sssp-workflow.svg)](https://badge.fury.io/py/aiida-sssp-workflow)

# aiida-sssp-workflow

The `aiida-sssp-workflow` is an aiida plugin to run the verification for a given pseudopotential. The plugin contains
workflows to verify the pseudopotential.
It can:

1) evaluate the [delta factor]() of the pseudopotential with respect to WIEN2K all-electrons results.
2) Converge test on varies of properties to give a recommended energy cutoff of the pseudopotential, include properties:
    1) Cohesive energy
    2) phonon frequencies
    3) pressure
    4) bands distance

### The computational details to running the calculation

#### Input Structures:

- In Δ-factor calculation: most stable elemental system from [Cottenier's work](http://molmod.ugent.be/deltacodesdft)
    and rare-earth nitrides from [Topsakal-Wentzkovitch work](https://www.sciencedirect.com/science/article/abs/pii/S0927025614005059);
- Phonon, pressure, cohesive energy: Cottenier's structures
    (except SiF4 has been used for F because of convergence issues) and
    rare-earth nitrides; Use primitive cells.
- Bands: Cottenier's structures reduced to primitive cells
    (except SiF4 has been used for F because of convergence issues) and rare-earth nitrides.
    PwbandWorkflow will make a primitive cell for band calculation (Remember to turn off the relax step).

#### Parameters of Δ calculations

- wave function cutoffs: 200 Ry;
- dual = 8 (PAW/US), 4 (NC); Mn/Fe/Co have larger duals tested as well; 12 and 16.
    We have gone in a mode where we do not use the dual, but we use ECUTRHO and ECUTWFC. However, dual is still used in
    simply setting the ecutwfc/ecutrho pairs.
- k-points: 0.1A^-1;
- smearing (degauss): Marzari-Vanderbilt, 0.01 Ry;
- non spin-polarized calculations except Mn (antiferromagnetic),
    O and Cr (antiferromagnetic),
    Fe, Co, and Ni (ferromagnetic).

> As for calculation of lanthenide, always increase `nbnd` to two times of the default number.

#### Parameters in phonon, pressure, cohesive energy calculations:

- k-points: 0.15A^-1
- smearing: Marzari-Vanderbilt, 0.01 Ry;
- k-points for the isolated atoms: 1x1x1;
- smearing for the isolated atoms: gaussian 0.01 Ry;
- unit cell for the isolated atoms: 12x12x12 Å with atom sit in [6.0, 6.0, 6.0] the middle of the cell;
- q-point: only calculate the phonon frequencies on Brillouin-Zone border q=(0.5, 0.5, 0.5).
- all calculations non-spin-polarized.

> In isolate atom energy calculation of cohesive energy evaluation.
> As for lanthenide, increase `nbnd` to three times of the default number. Moreover, use more RAM(by increase `num_machine` to 4).


> NOTE: PWscf writes in the output something called total energy. This is *NOT* the total energy when you have smearing;
> it’s the total free energy E-TS. PWscf also writes -TS, so one can get back the total energy E.
> In general (for a metal) E-TS should be used. For an atom instead the total energy should be used,
> since the -TS term is not really physical (it comes from the entropy of fractional occupations on the atom).
> Check with Nicola if you have atoms where -TS is different from zero. (http://theossrv1.epfl.ch/Main/ElectronicTemperature)


##### The convergence pattern for the phonons is calculated as:
- circle = (1/N * ∑i=1,N [ωi(cutoff) - ωi(200Ry)]2 / ωi(200Ry)2)1/2 * 100 (in percentage) and half error bar = Max |[ω(cutoff) - ω(200Ry)] / ω(200Ry)| * 100, if the highest frequency is more than 100 cm-1;
- circle = (1/N * ∑i=1,N [ωi(cutoff) - ωi(200Ry)]2)1/2 (absolute value) and half error bar = Max |ωi(cutoff) - ω(200Ry)|, if the highest frequency is less than 100 cm-1;
- N is the total number of frequencies;
- For some elements, we have neglected the first n frequencies in the summation above, because the frequencies are negative and/or with strong oscillations as function of the cutoff for all the considered pseudos). We have neglected the first four frequencies for H and I, 12 for N and Cl, 6 for O and SiF4 (which replaces F).

#### Bands calculations:

- k-points for the self-consistent calculation: 0.1; (can use cache one for the latter calculation)
- k-points for the bands calculation (as in, calculations of the eta and eta10 factors): uniform mesh 0.2 with no symmetry reduction, rather than high-symmetry path which is not determinant;
- smearing: Marzari-Vanderbilt, 0.01 Ry in scf calculation and Fermi-Dirac in bands distance calculation;
- all calculations non spin-polarized.

## Repository contents

## Features

## More meta-info collection

### SiF4 structure and its (V0, B0, B1) reference value
Re-generate the SiF4 structure start from the cif file from [COD database](http://www.crystallography.net/cod/index.php). Detail inputs parameters are list below.

#### Pseudopotentials(SSSP-v1.1 precision)
- Si: Si.pbe-n-rrkjus_psl.1.0.0.UPF
- F: F.oncvpsp.upf

#### Pw relax and eos

##### pwscf parameters
```
'SYSTEM': {
    'degauss': 0.00735,
    'ecutrho': 1600,
    'ecutwfc': 200,
    'occupations': 'smearing',
    'smearing': 'marzari-vanderbilt',
},
'ELECTRONS': {
    'conv_thr': 1e-10,
},
```

##### EOS parameters

- seven points
- 0.02 interval

## Publishing Releases

1. Create a release PR/commit to the `develop` branch, updating version number of `aiida_sssp_workflow/__init__.py`, `setup.json` and update `CHANGELOG.md`.
2. Fast-forward merge `develop` into the `master` branch
3. Create a release on GitHub (<https://github.com/aiidateam/aiida-sssp-workflow/releases/new>), pointing to the release commit on `master`, named `v.X.Y.Z` (identical to version in `setup.json`)
4. This will trigger the `continuous-deployment` GitHub workflow which, if all tests pass, will publish the package to PyPi. Check this has successfully completed in the GitHub Actions tab (<https://github.com/aiidateam/aiida-sssp-workflow/actions>).

(if the release fails, delete the release and tag)

## License

MIT


## Contact

morty.yeu@gmail.com
