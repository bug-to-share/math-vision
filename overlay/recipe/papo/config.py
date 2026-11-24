"""Recipe-local configuration; shared actor and engine defaults are unchanged."""

import math
from dataclasses import dataclass, field

from verl.base_config import BaseConfig
from verl.workers.config.engine import FSDPEngineConfig


@dataclass
class PAPOConfig(BaseConfig):
    enabled: bool = True
    mask_probability: float = 0.6
    seed: int = 42
    perception_coef: float = 0.01
    clean_entropy_coef: float = 0.03
    masked_entropy_coef: float = 0.03
    # Keep the original default for historical experiments; cached_old is opt-in.
    gradient_mode: str = "recompute_grad"

    def __post_init__(self):
        if not 0 <= self.mask_probability <= 1:
            raise ValueError("PAPO mask_probability must be in [0, 1]")
        if self.gradient_mode not in {"recompute_grad", "stop_gradient", "cached_old"}:
            raise ValueError("PAPO gradient_mode must be recompute_grad, stop_gradient or cached_old")
        for name in ("perception_coef", "clean_entropy_coef", "masked_entropy_coef"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"PAPO {name} must be finite and nonnegative")

    @property
    def active(self):
        return self.enabled and any((self.perception_coef, self.clean_entropy_coef, self.masked_entropy_coef))

    @property
    def needs_masked_forward(self):
        return self.active and bool(self.perception_coef or self.masked_entropy_coef)


@dataclass
class PAPOFSDPEngineConfig(FSDPEngineConfig):
    papo: PAPOConfig = field(default_factory=PAPOConfig)
