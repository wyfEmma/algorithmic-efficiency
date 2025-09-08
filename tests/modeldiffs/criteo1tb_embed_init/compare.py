import os

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import numpy as np
import torch

from algoperf import spec
from algoperf.workloads.criteo1tb.criteo1tb_jax.workload import (
  Criteo1TbDlrmSmallEmbedInitWorkload as JaxWorkload,
)
from algoperf.workloads.criteo1tb.criteo1tb_pytorch.workload import (
  Criteo1TbDlrmSmallEmbedInitWorkload as PyTorchWorkload,
)
from tests.modeldiffs.diff import ModelDiffRunner


def key_transform(k):
  new_key = []
  s_count = None
  for i in k:
    if 'Sequential' in i:
      s_count = int(i.split('_')[1])
      continue
    if 'Embedding' in i:
      return ('embedding_table',)
    if 'Linear' in i:
      i = i.replace('Linear', 'Dense')
      name, count = i.split('_')
      i = name + '_' + str(s_count * 3 + int(count))
    elif 'weight' in i:
      i = i.replace('weight', 'kernel')
    new_key.append(i)
  return tuple(new_key)


def sd_transform(sd):
  out = {}
  chunks = []
  for k in sd:
    if 'embedding_chunk' in ''.join(k):
      chunks.append(sd[k].cpu())
    else:
      out[k] = sd[k]
  out[('embedding_table',)] = torch.cat(chunks, dim=0)
  return out


if __name__ == '__main__':
  # pylint: disable=locally-disabled, not-callable

  jax_workload = JaxWorkload()
  pytorch_workload = PyTorchWorkload()

  pytorch_batch = {
    'inputs': torch.ones((2, 13 + 26)),
    'targets': torch.randint(low=0, high=1, size=(2,)),
    'weights': torch.ones(2),
  }
  jax_batch = {k: np.array(v) for k, v in pytorch_batch.items()}

  # Test outputs for identical weights and inputs.
  pytorch_model_kwargs = dict(
    augmented_and_preprocessed_input_batch=pytorch_batch,
    model_state=None,
    mode=spec.ForwardPassMode.EVAL,
    rng=None,
    update_batch_norm=False,
    dropout_rate=0.0,
  )

  jax_model_kwargs = dict(
    augmented_and_preprocessed_input_batch=jax_batch,
    mode=spec.ForwardPassMode.EVAL,
    rng=jax.random.PRNGKey(0),
    update_batch_norm=False,
    dropout_rate=0.0,
  )

  ModelDiffRunner(
    jax_workload=jax_workload,
    pytorch_workload=pytorch_workload,
    jax_model_kwargs=jax_model_kwargs,
    pytorch_model_kwargs=pytorch_model_kwargs,
    key_transform=key_transform,
    sd_transform=sd_transform,
    out_transform=None,
  ).run()
