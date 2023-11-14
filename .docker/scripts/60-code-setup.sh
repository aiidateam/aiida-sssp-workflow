#!/bin/bash

# If lock file exists, then we have already run this script
if [ -f ${HOME}/.lock_code_setup ]; then
    exit 0
else
    touch ${HOME}/.lock_code_setup
fi

# Loop over executables to set up
for code_name in pw ph; do
    # Set up caching
    verdi config set -a caching.enabled_for aiida.calculations:quantumespresso.${code_name}
    # Set up code
    verdi code create core.code.installed \
        --non-interactive \
        --label ${code_name}-${QE_VERSION} \
        --description "${code_name}.x code on localhost" \
        --default-calc-job-plugin quantumespresso.${code_name} \
        --computer localhost --prepend-text 'eval "$(conda shell.posix hook)"\nconda activate base\nexport OMP_NUM_THREADS=1' \
        --filepath-executable ${code_name}.x
done
