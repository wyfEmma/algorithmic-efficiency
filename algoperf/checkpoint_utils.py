"""Utilities for checkpointing.

Note: Code adapted from
https://github.com/google/init2winit/blob/master/init2winit/checkpoint.py.
"""

import os
from typing import Optional, Sequence, Tuple

import jax
import numpy as np
import orbax.checkpoint as ocp
import torch
from absl import logging
from flax import serialization
from flax.training import checkpoints as flax_checkpoints
from flax.training.checkpoints import latest_checkpoint
from orbax.checkpoint.type_handlers import NumpyHandler
from tensorflow.io import gfile  # pytype: disable=import-error

from algoperf import jax_sharding_utils, spec
from algoperf.pytorch_utils import pytorch_setup

_, _, DEVICE, _ = pytorch_setup()
CheckpointReturn = Tuple[
  spec.OptimizerState,
  spec.ParameterContainer,
  spec.ModelAuxiliaryState,
  dict,
  list,
  int,
  int,
]


class BoolHandler(NumpyHandler):
  """
  An implementation of TypeHandler for np.bool_ that inherits from NumpyHandler.
  It works by treating the scalar as a 0-dimensional array.
  """

  def typestr(self) -> str:
    """Unique string identifier for this handler."""
    return 'np.bool_'

  async def serialize(
    self,
    values: Sequence[np.bool_],
    infos: Sequence,
    args: Optional[Sequence[ocp.SaveArgs]] = None,
  ):
    """
    Serializes a sequence of np.bool_ scalars by first converting them
    to 0-dim numpy arrays and then calling the parent NumpyHandler.
    """
    # Convert each scalar np.bool_ to a 0-dimensional np.ndarray
    array_values = [np.asarray(v, dtype=np.bool_) for v in values]
    # Use the parent class's robust serialization logic
    return await super().serialize(array_values, infos, args)

  async def deserialize(
    self,
    infos: Sequence,
    args: Optional[Sequence[ocp.RestoreArgs]] = None,
  ) -> Sequence[np.bool_]:
    """
    Deserializes into a sequence of np.bool_ scalars by calling the
    parent handler and then converting the resulting 0-dim arrays.
    """
    # Parent deserialize will return a sequence of 0-dimensional np.ndarray
    results = await super().deserialize(infos, args)

    # Convert each 0-d array back to an np.bool_ scalar using .item()
    scalar_results = [np.bool_(r.item()) for r in results]
    return scalar_results


ocp.type_handlers.register_type_handler(np.bool_, BoolHandler(), override=True)


def _restore_jax_eval_results(raw_eval_results) -> list:
  """Restores eval_results list of (step, metrics_dict) tuples from Flax state dict."""
  if not raw_eval_results:
    return []
  if isinstance(raw_eval_results, (list, tuple)):
    restored = []
    for item in raw_eval_results:
      if isinstance(item, dict) and '0' in item and '1' in item:
        restored.append((int(item['0']), item['1']))
      else:
        restored.append(tuple(item))
    return restored
  if isinstance(raw_eval_results, dict):
    restored = []
    sorted_items = sorted(
      raw_eval_results.items(),
      key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0],
    )
    for key, value in sorted_items:
      if isinstance(value, dict) and '0' in value and '1' in value:
        restored.append((int(value['0']), value['1']))
      elif isinstance(value, (list, tuple)) and len(value) == 2:
        restored.append((int(value[0]), value[1]))
      else:
        restored.append((value, key))
    return restored
  return list(raw_eval_results)


def maybe_restore_checkpoint(
  framework: str,
  optimizer_state: spec.OptimizerState,
  model_params: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  train_state: dict,
  eval_results: list,
  global_step: int,
  preemption_count: int,
  checkpoint_dir: str,
) -> CheckpointReturn:
  """Optionally restores from a checkpoint.

  The checkpoint logic is as follows: if there is a checkpoint in
  `checkpoint_dir`, restore it. Else, don't restore any checkpoint, and
  just return the passed-in optimizer_state, model_params,
  model_state, and train_state.

  Args:
    framework: Current framework (e.g., `jax` or `pytorch`).
    optimizer_state: Optimizer state.
    model_params: Model parameters.
    model_state: Model state such as batch statistics when batch
      normalization is used.
    train_state: Training state such as `last_eval_time`.
    eval_results: Previous evaluation results.
    global_step: Global step.
    preemption_count: Number of preemptions.
    checkpoint_dir: The training directory where we will look for a checkpoint.

  Returns:
    A tuple of (optimizer_state, model_params, model_state,
    train_state, eval_results, global_step, preemption_count).
  """
  checkpoint_dir = os.path.abspath(checkpoint_dir)
  if framework == 'jax':
    opt_state, opt_update_fn = optimizer_state
  else:
    opt_state, opt_update_fn = optimizer_state, None

  uninitialized_global_step = -1
  uninitialized_preemption_count = -1
  checkpoint_state = {
    'model_params': model_params,
    'optimizer_state': opt_state,
    'model_state': model_state,
    'train_state': train_state,
    'eval_results': None,
    'global_step': uninitialized_global_step,
    'preemption_count': uninitialized_preemption_count,
  }

  if framework == 'jax':
    raw_ckpt = flax_checkpoints.restore_checkpoint(
      checkpoint_dir, target=None
    )
    if (
      raw_ckpt is None
      or raw_ckpt.get('global_step', uninitialized_global_step)
      == uninitialized_global_step
    ):
      found_checkpoint = False
      save_path = None
    else:
      found_checkpoint = True
      save_path = os.path.join(
        checkpoint_dir, 'checkpoint_' + str(raw_ckpt['global_step'])
      )
  else:
    latest_ckpt = checkpoint_state
    save_path = latest_checkpoint(checkpoint_dir)
    if save_path is not None:
      latest_ckpt = torch.load(
        save_path, map_location=DEVICE, weights_only=False
      )
    found_checkpoint = latest_ckpt['global_step'] != uninitialized_global_step

  # If no checkpoint is found, return the passed-in initial state.
  if not found_checkpoint:
    return (
      optimizer_state,
      model_params,
      model_state,
      train_state,
      eval_results,
      global_step,
      preemption_count,
    )

  # If there's the latest checkpoint in the checkpoint_dir, restore from that.
  if framework == 'jax':
    restored_params = serialization.from_state_dict(
      jax.device_get(model_params), raw_ckpt['model_params']
    )
    restored_opt_state = serialization.from_state_dict(
      jax.device_get(opt_state), raw_ckpt['optimizer_state']
    )
    if (
      model_state is not None
      and len(jax.tree.leaves(model_state)) > 0
      and raw_ckpt.get('model_state') is not None
    ):
      restored_model_state = serialization.from_state_dict(
        jax.device_get(model_state), raw_ckpt['model_state']
      )
    else:
      restored_model_state = model_state

    latest_ckpt = {
      'model_params': restored_params,
      'optimizer_state': restored_opt_state,
      'model_state': restored_model_state,
      'train_state': raw_ckpt['train_state'],
      'eval_results': _restore_jax_eval_results(raw_ckpt.get('eval_results')),
      'global_step': int(raw_ckpt['global_step']),
      'preemption_count': int(raw_ckpt['preemption_count']),
    }
    checkpoint_state = replicate_checkpoint(
      latest_ckpt,
      pytree_keys=[
        'optimizer_state',
        'model_params',
        'model_state',
      ],
    )
    checkpoint_state['optimizer_state'] = (
      checkpoint_state['optimizer_state'],
      opt_update_fn,
    )
    logging.info(f'Loaded checkpoint from {save_path}.')

  else:
    checkpoint_state = latest_ckpt
    if isinstance(
      model_params,
      (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
    ):
      model_params = model_params.module
    model_params.load_state_dict(checkpoint_state['model_params'])
    checkpoint_state['model_params'] = model_params
    for key in optimizer_state.keys():
      optimizer_state[key].load_state_dict(
        checkpoint_state['optimizer_state'][key]
      )
      checkpoint_state['optimizer_state'][key] = optimizer_state[key]

    logging.info(f'Loaded checkpoint from {save_path}.')
  return (
    checkpoint_state['optimizer_state'],
    checkpoint_state['model_params'],
    checkpoint_state['model_state'],
    checkpoint_state['train_state'],
    list(checkpoint_state['eval_results']),
    checkpoint_state['global_step'],
    checkpoint_state['preemption_count'] + 1,
  )


def replicate_checkpoint(
  latest: dict, pytree_keys: Sequence[str], replicate: bool = True
) -> dict:
  """Restores from the provided checkpoint.

  Args:
    latest: A dict representing the state of the
      checkpoint we want to restore.
    pytree_keys: A sequence of keys into `latest` that are pytrees, which will
      be replicated if replicate=True.
    replicate: If set, replicate the state across devices.

  Returns:
    A JAX pytree holding the arrays that need to be replicated/unreplicated.
  """
  pytree = {k: latest[k] for k in pytree_keys}
  if replicate:
    pytree = jax_sharding_utils.replicate(pytree)
  extra_dict = {k: latest[k] for k in latest.keys() if k not in pytree_keys}
  pytree.update(extra_dict)
  return pytree


def save_checkpoint(
  framework: str,
  optimizer_state: spec.OptimizerState,
  model_params: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  train_state: dict,
  eval_results: list,
  global_step: int,
  preemption_count: int,
  checkpoint_dir: str,
  save_intermediate_checkpoints: bool,
) -> None:
  """Save the checkpoint in `checkpoint_dir`.

  Args:
    framework: Current framework (e.g., `jax` or `pytorch`).
    optimizer_state: Optimizer state.
    model_params: Model parameters.
    model_state: Model state such as batch statistics when batch
      normalization is used.
    train_state: Training state such as `last_eval_time`.
    eval_results: Previous evaluation results.
    global_step: Global step.
    preemption_count: Number of preemptions.
    checkpoint_dir: The training directory where we will look for a checkpoint.
    save_intermediate_checkpoints: Whether to save intermediate checkpoints.

  Returns:
    A tuple of (optimizer_state, model_params, model_state,
    train_state, eval_results, global_step, preemption_count).
  """
  checkpoint_dir = os.path.abspath(checkpoint_dir)
  if framework == 'jax':
    opt_state, _ = optimizer_state
    model_params = jax.device_get(model_params)
    opt_state = jax.device_get(opt_state)
    model_state = jax.device_get(model_state)
    train_state = jax.device_get(train_state)
    eval_results = jax.device_get(eval_results)
  else:
    if isinstance(
      model_params,
      (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
    ):
      model_params = model_params.module
    model_params = model_params.state_dict()
    optimizer_state_dict = {}
    for key in optimizer_state.keys():
      if hasattr(optimizer_state[key], 'state_dict'):
        optimizer_state_dict[key] = optimizer_state[key].state_dict()
      else:
        logging.warning(
          f'The optimizer state for key {key} is not saved, because '
          f'{type(optimizer_state[key])} has not implemented a state_dict() '
          'method.'
        )
    opt_state = optimizer_state_dict

  checkpoint_state = {
    'model_params': model_params,
    'optimizer_state': opt_state,
    'model_state': model_state,
    'train_state': train_state,
    'eval_results': tuple(eval_results),
    'global_step': global_step,
    'preemption_count': preemption_count,
  }

  save_path = os.path.join(checkpoint_dir, f'checkpoint_{global_step}')
  if framework == 'jax':
    flax_checkpoints.save_checkpoint(
      checkpoint_dir,
      target=checkpoint_state,
      step=global_step,
      overwrite=True,
      keep=np.inf if save_intermediate_checkpoints else 1,
    )
  else:
    if not save_intermediate_checkpoints:
      checkpoint_files = gfile.glob(
        os.path.join(checkpoint_dir, 'checkpoint_*')
      )
      for path in checkpoint_files:
        logging.info('Removing checkpoint at %s', path)
        if gfile.isdir(path):
          gfile.rmtree(path)
        else:
          gfile.remove(path)
    torch.save(checkpoint_state, save_path)

  logging.info(f'Saved checkpoint to {save_path}.')
