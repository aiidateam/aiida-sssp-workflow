========
Tutorial
========

Running verification workflows for a given pseudopotential. The complete verification
workflow contain a delta factor calculation, a band structure calculation and
convergence tests for cohesive energy, phonon frequencies, bands distance and residual
pressure.

The required inputs of the verification workflows are ``pw_code`` for
``quantumespresso.pw`` code node,
``ph_code`` for ``quantumespresso.ph`` code node
and ``pseudo`` for a ``UpfData`` node which can be imported by::

    verdi data upf import <path-to-upf-file>

The delta factor workflow and four convergence workflow are also included in
the ``aiida.workflow`` entry point. Their minimal required inputs are the same
the codes and the pseudopotential upf node. Except for pressure convergence
workflow which need Birch-Murnaghan fit result the V0, B0, B1 as an additional
required input to calculate the residual pressure(TODO: link to the definition).

The others common optional inputs for convergence workflows are 1) the list of energy cutoff of
wavefunction ``ecutwfc_list`` 2) the list energy cutoff of charge density ``ecutrho_list``
3) ``ref_cutoff_pair`` which set the wavefunction cutoff and density charge cutoff used
as the reference quantum-espresso pw.x inputs.

The outputs of every subworkflow are included as specific output namespace of the
verification workflow.
If all subworkflows are running successful and all the convergence
are reached within the pre-setting energy cutoff list the outputs would be the following::

    Outputs                            PK     Type
    ---------------------------------  -----  ---------
    band_structure
        seekpath_band_structure
            scf_parameters             27250  Dict
            band_parameters            27372  Dict
            band_structure             27370  BandsData
        scf_parameters                 26971  Dict
        band_parameters                27039  Dict
        band_structure                 27037  BandsData
        nbands_factor                  27410  Float
    convergence_bands
        output_pseudo_header           26862  Dict
        output_convergence_parameters  27505  Dict
        xy_data_ecutwfc                27506  XyData
        xy_data_ecutrho                27507  XyData
        output_parameters              27508  Dict
    convergence_cohesive_energy
        output_pseudo_header           27813  Dict
        output_convergence_parameters  28112  Dict
        xy_data_ecutwfc                28113  XyData
        xy_data_ecutrho                28114  XyData
        output_parameters              28115  Dict
    convergence_phonon_frequencies
        output_pseudo_header           26860  Dict
        output_convergence_parameters  27805  Dict
        xy_data_ecutwfc                27806  XyData
        xy_data_ecutrho                27807  XyData
        output_parameters              27808  Dict
    convergence_pressure
        output_pseudo_header           27815  Dict
        output_convergence_parameters  28108  Dict
        xy_data_ecutwfc                28109  XyData
        xy_data_ecutrho                28110  XyData
        output_parameters              28111  Dict
    delta_factor
        output_pseudo_header           26858  Dict
        output_eos_parameters          27143  Dict
        output_birch_murnaghan_fit     27242  Dict
        output_parameters              27208  Dict

Otherwise, if some of the convergence workflow not converged, the ``output_convergence_parameters`` will not been given. Or if some subworkflow
not finished ok with exit code 0, you will not see its outputs in the output namespace
of verification workflow.
