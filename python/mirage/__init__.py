import os

try:
    from .core import *
except ImportError:
    import z3

    _z3_lib = os.path.join(os.path.dirname(z3.__file__), "lib")
    os.environ["LD_LIBRARY_PATH"] = (
        f"{_z3_lib}:{os.environ.get('LD_LIBRARY_PATH','LD_LIBRARY_PATH')}"
    )

    from .core import *

from .kernel import *
from .persistent_kernel import PersistentKernel
from .threadblock import *


class InputNotFoundError(Exception):
    """Raised when cannot find input tensors"""

    pass


def set_gpu_device_id(device_id: int):
    global_config.gpu_device_id = device_id
    core.set_gpu_device_id(device_id)


def bypass_compile_errors(value: bool = True):
    global_config.bypass_compile_errors = value


def new_kernel_graph():
    kgraph = core.CyKNGraph()
    return KNGraph(kgraph)


def new_threadblock_graph(
    grid_dim: tuple, block_dim: tuple, forloop_range: int, reduction_dimx: int
):
    bgraph = core.CyTBGraph(grid_dim, block_dim, forloop_range, reduction_dimx)
    return TBGraph(bgraph)


# Other Configurations
from .global_config import global_config

# Graph Datasets
from .graph_dataset import graph_dataset
from .version import __version__
