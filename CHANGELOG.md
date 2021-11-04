## aiida-sssp-workflow

### 0.2.0-beta.0
the release are ready to be run by command line interface.
Will be the first stable backend of aiidalab-sssp app.

- The legacy convergence workchains are all extended from _base convergence work chain.
- The pseudo_parser library is add for pseudopotential parsing only.
- Add CLI interface for verification workflow and the properties can be chosen.
- Test verification (at `test` protocol) for silicon, fluorine(SiF4 in convergence WF), gold and a lanthenide element Samarium (Sm). Fix some issues.

### 0.2.0-alpha.1
The release to correctly deploy the documentation and install the package in aiidalab

- replace fortran efermi with pure python efermi [#50](https://github.com/aiidateam/aiida-sssp-workflow/pull/50)
- verification workflow based on legacy workflows
- delta factor workflow accept dual as input

### 0.2.0-alpha.0
The release after Jason start working at EPFL

Add Legacy convergence workflows and simplify the interface of all workflows

- use aiida-pseudo for `UpfData`.
- add legacy convergence workflows for comparing the old sssp and for benchmarking purpose
- add `BandsDistanceWorkChain` for future bands chessboard plot

### 0.1.0
The first release

fixbug for the workflows and restructure the repository

- add bandstructure evaluation to verification
- mag_structure generate bugfix for Fe,Co,Ni

### 0.1.0b0
The initial (pre) release contain the most workflows

- workflow delta_factor
- the cli tool for delta factor calculation
- the convergence workflow integrate with aiida-optimize
- workflow convergence cohesive energy
- workflow convergence phonon frequencies
- workflow convergence pressue
- workflow convergence bands distance
