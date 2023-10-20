#!/bin/bash

# XXX for testing
verdi config set warnings.rabbitmq_version True

# XXX for testing
verdi config set -a caching.enabled_for aiida.calculations:quantumespresso.pw

# computer/code setup
# verdi computer setup --config $COMPUTER_CONFIG_NAME
# ...
