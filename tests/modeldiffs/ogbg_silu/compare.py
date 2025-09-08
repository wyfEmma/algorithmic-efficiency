import os

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import jraph
import numpy as np
import torch

from algoperf import spec
from algoperf.workloads.ogbg.ogbg_jax.workload import (
  OgbgSiluWorkload as JaxWorkload,
)
from algoperf.workloads.ogbg.ogbg_pytorch.workload import (
  OgbgSiluWorkload as PyTorchWorkload,
)
from tests.modeldiffs.diff import ModelDiffRunner

# Todo: refactor tests to use workload properties in cleaner way
hidden_dims = len(JaxWorkload().hidden_dims)
num_graphs = JaxWorkload().num_message_passing_steps


def key_transform(k):
  new_key = []
  bn = False
  ln = False
  graph_network = False
  graph_index = 0
  seq_index = 0
  for i in k:
    bn = bn or 'BatchNorm' in i
    ln = ln or 'LayerNorm' in i
    graph_network = graph_network or 'GraphNetwork' in i
    if 'Sequential' in i:
      seq_index = int(i.split('_')[1])
      continue
    if 'GraphNetwork' in i:
      graph_index = int(i.split('_')[1])
      continue
    if 'Linear' in i:
      layer_index = int(i.split('_')[1])
      if graph_network:
        count = (
          graph_index * 3 * hidden_dims + seq_index * hidden_dims + layer_index
        )
        i = 'Dense_' + str(count)
      elif layer_index == 0:
        i = 'node_embedding'
      elif layer_index == 1:
        i = 'edge_embedding'
      elif layer_index == 2:
        count = num_graphs * 3 * hidden_dims
        i = 'Dense_' + str(count)
    elif 'LayerNorm' in i:
      layer_index = int(i.split('_')[1])
      count = (
        graph_index * 3 * hidden_dims + seq_index * hidden_dims + layer_index
      )
      i = 'LayerNorm_' + str(count)
    elif 'weight' in i:
      if bn or ln:
        i = i.replace('weight', 'scale')
      else:
        i = i.replace('weight', 'kernel')
    new_key.append(i)
  return tuple(new_key)


def sd_transform(sd):
  # pylint: disable=locally-disabled, modified-iterating-dict, consider-using-dict-items
  out = {}
  for k in sd:
    out[k] = sd[k]
  return out


if __name__ == '__main__':
  # pylint: disable=locally-disabled, not-callable

  jax_workload = JaxWorkload()
  pytorch_workload = PyTorchWorkload()

  pytorch_batch = dict(
    n_node=torch.LongTensor([5]),
    n_edge=torch.LongTensor([5]),
    nodes=torch.randn(5, 9),
    edges=torch.randn(5, 3),
    globals=torch.randn(1, 128),
    senders=torch.LongTensor(list(range(5))),
    receivers=torch.LongTensor([(i + 1) % 5 for i in range(5)]),
  )

  jax_batch = {k: np.array(v) for k, v in pytorch_batch.items()}

  # Test outputs for identical weights and inputs.
  graph_j = jraph.GraphsTuple(**jax_batch)
  graph_p = jraph.GraphsTuple(**pytorch_batch)

  jax_batch = {'inputs': graph_j}
  pytorch_batch = {'inputs': graph_p}

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
