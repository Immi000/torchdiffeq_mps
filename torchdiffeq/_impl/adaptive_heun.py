import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver
from .common import DTYPE


_ADAPTIVE_HEUN_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1.], dtype=DTYPE),
    beta=[
        torch.tensor([1.], dtype=DTYPE),
    ],
    c_sol=torch.tensor([0.5, 0.5], dtype=DTYPE),
    c_error=torch.tensor([
        0.5,
        -0.5,
    ], dtype=DTYPE),
)

_AH_C_MID = torch.tensor([
    0.5, 0.
], dtype=DTYPE)


class AdaptiveHeunSolver(RKAdaptiveStepsizeODESolver):
    order = 2
    tableau = _ADAPTIVE_HEUN_TABLEAU
    mid = _AH_C_MID
