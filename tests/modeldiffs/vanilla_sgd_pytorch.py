import torch

from algoperf import spec
from algorithms.target_setting_algorithms.data_selection import (  # noqa: F401
  data_selection,
)
from algorithms.target_setting_algorithms.pytorch_submission_base import (  # noqa: F401
  update_params,
)


def init_optimizer_state(
  workload: spec.Workload,
  model_params: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  rng: spec.RandomState,
) -> spec.OptimizerState:
  """Creates a Vanilla SGD Optimizer."""
  del model_state
  del rng

  optimizer_state = {
    'optimizer': torch.optim.SGD(
      model_params.parameters(), lr=0.001, weight_decay=0
    ),
  }

  return optimizer_state
