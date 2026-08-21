"""
ProteinCIL baseline methods for class-incremental learning on frozen protein encoders.

Each baseline is implemented in its own module. This package re-exports the
shared utilities and the METHODS registry so that run.py / run_protocols.py
can import from ``methods`` exactly as before.
"""

from methods._common import (
    set_seed,
    make_clf,
    train_loop,
    evaluate,
    compute_metrics,
    print_results,
    ReplayBuffer,
)

from methods.naive import run_naive
from methods.joint import run_joint
from methods.ewc import run_ewc
from methods.replay import run_replay
from methods.icarl import run_icarl
from methods.derpp import run_derpp
from methods.fetril import run_fetril
from methods.rer import run_rer
from methods.fecam import run_fecam_common
from methods.ranpac import run_ranpac
from methods.ease import run_ease

METHODS = {
    "naive": run_naive,
    "joint": run_joint,
    "ewc": run_ewc,
    "replay": run_replay,
    "fecam_common": run_fecam_common,
    "icarl": run_icarl,
    "derpp": run_derpp,
    "fetril": run_fetril,
    "rer": run_rer,
    "ranpac": run_ranpac,
    "ease": run_ease,
}
