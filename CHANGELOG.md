## aiida-sssp-workflow

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
