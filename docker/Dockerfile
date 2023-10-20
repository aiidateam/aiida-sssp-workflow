FROM aiidateam/aiida-core-with-services:edge

USER root

# Install aiida-sssp-workflow
# XXX:to have the latest code copy the src folder into the container
RUN pip install aiida-sssp-workflow

COPY s6-assets/s6-rc.d /etc/s6-overlay/s6-rc.d
COPY s6-assets/init /etc/init

USER ${SYSTEM_UID}
