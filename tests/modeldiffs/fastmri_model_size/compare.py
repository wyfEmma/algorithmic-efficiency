import os

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import torch

from algoperf import spec
from algoperf.workloads.fastmri.fastmri_jax.workload import (
  FastMRIModelSizeWorkload as JaxWorkload,
)
from algoperf.workloads.fastmri.fastmri_pytorch.workload import (
  FastMRIModelSizeWorkload as PyTorchWorkload,
)
from tests.modeldiffs.diff import ModelDiffRunner


def sd_transform(sd):
  def sort_key(k):
    if k[0] == 'ModuleList_0':
      return (0, *k)
    if k[0] == 'ConvBlock_0':
      return (1, *k)
    if k[0] == 'ModuleList_1':
      return (2, *k)
    if k[0] == 'ModuleList_2':
      return (3, *k)

  keys = sorted(sd.keys(), key=sort_key)
  c = 0
  for idx, k in enumerate(keys):
    new_key = []
    for idx2, i in enumerate(k):
      if 'ModuleList' in i or 'Sequential' in i:
        continue
      if i.startswith('ConvBlock'):
        if idx != 0 and keys[idx - 1][: idx2 + 1] != k[: idx2 + 1]:
          c += 1
        i = f'ConvBlock_{c}'
      if 'Conv2d' in i:
        i = i.replace('Conv2d', 'Conv')
      if 'ConvTranspose2d' in i:
        i = i.replace('ConvTranspose2d', 'ConvTranspose')
      if 'weight' in i:
        i = i.replace('weight', 'kernel')
      new_key.append(i)
    new_key = tuple(new_key)
    sd[new_key] = sd[k]
    del sd[k]
  return sd


key_transform = None
if __name__ == '__main__':
  # pylint: disable=locally-disabled, not-callable

  jax_workload = JaxWorkload()
  pytorch_workload = PyTorchWorkload()

  # Test outputs for identical weights and inputs.
  image = torch.randn(2, 320, 320)

  jax_batch = {'inputs': image.detach().numpy()}
  pytorch_batch = {'inputs': image}

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
    key_transform=None,
    sd_transform=sd_transform,
  ).run()
