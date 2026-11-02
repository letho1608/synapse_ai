import os as _os

# OpenMP threads that spin after every parallel region steal the cores of
# whatever runs next. Torch ops and the numba quantized kernels alternate
# inside every decoder layer; with the default (spinning) policy a torch +
# numba pair took 3.5x as long as the two run apart. Must be set before torch
# loads its OpenMP runtime, hence here; an explicit user setting wins.
_os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
_os.environ.setdefault("KMP_BLOCKTIME", "0")

from synapse.helpers import DEBUG as DEBUG, DEBUG_DISCOVERY as DEBUG_DISCOVERY, VERSION as VERSION
