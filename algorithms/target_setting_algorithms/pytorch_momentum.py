"""Submission file for a SGD with HeavyBall momentum optimizer in PyTorch."""

import torch
from torch.optim.lr_scheduler import LambdaLR

from algoperf import spec
from algorithms.target_setting_algorithms.data_selection import (  # noqa: F401
  data_selection,
)
from algorithms.target_setting_algorithms.get_batch_size import (  # noqa: F401
  get_batch_size,
)
from algorithms.target_setting_algorithms.jax_momentum import (
  create_lr_schedule_fn,
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
  """Creates a Nesterov optimizer and a learning rate schedule."""
  del model_state
  del rng

  # Create optimizer.
  optimizer_state = {
    'optimizer': torch.optim.SGD(
      model_params.parameters(),
      lr=hyperparameters.learning_rate,
      momentum=hyperparameters.beta1,
      weight_decay=hyperparameters.weight_decay,
      nesterov=False,
    ),
  }

  # Create learning rate schedule.
  target_setting_step_hint = int(0.75 * workload.step_hint)
  lr_schedule_fn = create_lr_schedule_fn(
    target_setting_step_hint, hyperparameters
  )

  # PyTorch's LambdaLR expects the lr_lambda fn to return a factor which will
  # be multiplied with the base lr, so we have to divide by it here.
  def _lr_lambda(step: int) -> float:
    return lr_schedule_fn(step).item() / hyperparameters.learning_rate

  optimizer_state['scheduler'] = LambdaLR(
    optimizer_state['optimizer'], lr_lambda=_lr_lambda
  )

  return optimizer_state
