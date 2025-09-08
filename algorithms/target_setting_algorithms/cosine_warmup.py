"""Implementions of a linear warmup then cosine decay LR schedule."""

import optax
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def jax_cosine_warmup(step_hint: int, hyperparameters):
  # Create learning rate schedule.
  warmup_fn = optax.linear_schedule(
    init_value=0.0,
    end_value=hyperparameters.learning_rate,
    transition_steps=hyperparameters.warmup_steps,
  )
  cosine_steps = max(step_hint - hyperparameters.warmup_steps, 1)
  cosine_fn = optax.cosine_decay_schedule(
    init_value=hyperparameters.learning_rate, decay_steps=cosine_steps
  )
  schedule_fn = optax.join_schedules(
    schedules=[warmup_fn, cosine_fn], boundaries=[hyperparameters.warmup_steps]
  )
  return schedule_fn


def pytorch_cosine_warmup(step_hint: int, hyperparameters, optimizer):
  warmup = LinearLR(
    optimizer,
    start_factor=1e-10,
    end_factor=1.0,
    total_iters=hyperparameters.warmup_steps,
  )
  cosine_steps = max(step_hint - hyperparameters.warmup_steps, 1)
  cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps)
  return SequentialLR(
    optimizer,
    schedulers=[warmup, cosine_decay],
    milestones=[hyperparameters.warmup_steps],
  )
