import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver
from common import DTYPE


_FEHLBERG2_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2, 1.0], dtype=DTYPE),
    beta=[
        torch.tensor([1 / 2], dtype=DTYPE),
        torch.tensor([1 / 256, 255 / 256], dtype=DTYPE),
    ],
    c_sol=torch.tensor([1 / 512, 255 / 256, 1 / 512], dtype=DTYPE),
    c_error=torch.tensor(
        [-1 / 512, 0, 1 / 512], dtype=DTYPE
    ),
)

_FE_C_MID = torch.tensor([0.0, 0.5, 0.0], dtype=DTYPE)


class Fehlberg2(RKAdaptiveStepsizeODESolver):
    order = 2
    tableau = _FEHLBERG2_TABLEAU
    mid = _FE_C_MID
