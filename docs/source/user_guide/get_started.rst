===============
Getting started
===============

``aiida-sssp-workflow`` is an AiiDA plugin contains workflows to running pseudopotential verification.
It runs delta factor calculation and convergence testes on cohesive energy, phonon frequencies, bands distance and
residual pressure, which reflect the precision and the softness of the given pseudopotential.

Installation
++++++++++++

Use the following commands to install the plugin for developing::

    git clone https://github.com/aiidateam/aiida-sssp-workflow .
    cd aiida-sssp-workflow
    pip install -e .  # also installs aiida, if missing (but not postgres)
    #pip install -e .[pre-commit,testing] # install extras for more features
    verdi quicksetup  # better to set up a new profile
    verdi calculation plugins  # should now show your calclulation plugins

Or you can install it from pypi by::

    pip install aiida-sssp-workflow

To use this plugin you show have ``aiida-quantumespresso`` plugin installed and configure
codes for ``quantumespresso.pw`` and ``quantumespresso.ph``.

Add the following line into activate script file to enable the autocompletiom of
cammand line interface::

    eval "$(_AIIDA_SSSP_WORKFLOW_COMPLETE=source aiida-sssp-workflow)"

Moreover, enable the caching_ of ``quantumespresso.pw`` and ``quantumespresso.ph`` can
save time and computational resource in running the verification.

.. _caching: https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/provenance/caching.html

Usage
+++++

A quick demo of how to submit a verification through command line::

    verdi daemon start         # make sure the daemon is running
    aiida-sssp-workflow workflow launch verification -w 5400 -X qe-6.5-pw@daint-mc -Y qe-6.5-ph@daint-mc -D 4 ../psp/sg15/Si_ONCV_PBE-1.2.upf --daemon

The options of ``aiida-sssp-workflow workflow launch verification`` are::

    Usage: aiida-sssp-workflow workflow launch verification [OPTIONS] PSEUDO

      Run the workflow to calculate delta factor

    Options:
      -X, --pw-code CODE              A single code identified by its ID, UUID or
                                      label.  [required]

      -Y, --ph-code CODE              A single code identified by its ID, UUID or
                                      label.  [required]

      -P, --protocol TEXT             The protocol used in verification.
                                      [default: efficiency]

      -D, --dual INTEGER              The dual between ecutwfc and ecutrho.
                                      [default: 8]

      -x, --clean-workdir             Clean the remote folder of all the launched
                                      calculations after completion of the
                                      workchain.  [default: False]

      -m, --max-num-machines INTEGER  The maximum number of machines (nodes) to
                                      use for the calculations.  [default: 1]

      -w, --max-wallclock-seconds INTEGER
                                      the maximum wallclock time in seconds to set
                                      for the calculations.  [default: 1800]

      -i, --with-mpi                  Run the calculations with MPI enabled.
                                      [default: True]

      -d, --daemon                    Submit the process to the daemon instead of
                                      running it locally.  [default: False]

      -h, --help                      Show this message and exit.


