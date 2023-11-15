FROM ghcr.io/aiidateam/aiida-core-with-services:edge

USER root

ARG QE_VERSION
ENV QE_VERSION ${QE_VERSION}

# Install aiida-quantumespresso from source code
COPY --from=src . /tmp/aiida-sssp-workflow
RUN pip install /tmp/aiida-sssp-workflow --no-cache-dir && \
    rm -rf /tmp/aiida-sssp-workflow

# Install quantum espresso from conda (the latest version)
RUN mamba install -y -c conda-forge qe=${QE_VERSION} && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${SYSTEM_USER}"

COPY scripts/60-code-setup.sh /etc/init/run-before-daemon-start/

# Static example upf files
RUN mkdir -p /opt/examples
COPY _static/ /opt/examples

USER ${SYSTEM_UID}
