from enum import Enum
from typing import Protocol

import numpy as np


class RiskState(Enum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"
    GREY = "grey"


class Scorer(Protocol):
    def score(self, audio: np.ndarray) -> float: ...
