import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver
from .common import DTYPE


_BOGACKI_SHAMPINE_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2, 3 / 4, 1.], dtype=DTYPE),
    beta=[
        torch.tensor([1 / 2], dtype=DTYPE),
        torch.tensor([0., 3 / 4], dtype=DTYPE),
        torch.tensor([2 / 9, 1 / 3, 4 / 9], dtype=DTYPE)
    ],
    c_sol=torch.tensor([2 / 9, 1 / 3, 4 / 9, 0.], dtype=DTYPE),
    c_error=torch.tensor([2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, -1 / 8], dtype=DTYPE),
)

_BS_C_MID = torch.tensor([0., 0.5, 0., 0.], dtype=DTYPE)


class Bosh3Solver(RKAdaptiveStepsizeODESolver):
    order = 3
    tableau = _BOGACKI_SHAMPINE_TABLEAU
    mid = _BS_C_MID
