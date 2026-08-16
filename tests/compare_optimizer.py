import os

# Disable GPU access to ensure a consistent CPU-based comparison.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.optim as optim

from algoperf import spec
# Note: You may need to adjust these import paths based on your file structure.
from algoperf.workloads.criteo1tb.criteo1tb_jax.workload import (
    Criteo1TbDlrmSmallWorkload as JaxWorkload,
)
from algoperf.workloads.criteo1tb.criteo1tb_pytorch.workload import (
    Criteo1TbDlrmSmallWorkload as PyTorchWorkload,
)
# Assuming the schedule-free optimizers are available in these locations.
# Please adjust the paths if they are located elsewhere.
from algoperf.optimizers.schedule_free_adamw_jax import AdamW as JaxAdamW
from algoperf.optimizers.schedule_free_adamw_pytorch import AdamW as PytorchAdamW

# --- Helper functions from your provided code for model state transformation ---

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

# --- Main comparison script ---

if __name__ == '__main__':
  # 1. Initialize workloads
  jax_workload = JaxWorkload()
  pytorch_workload = PyTorchWorkload()

  # 2. Create identical data batches for PyTorch and JAX
  pytorch_batch = {
      'inputs': torch.randn(2, 13 + 26),
      'targets': torch.randint(low=0, high=2, size=(2,)),
  }
  jax_batch = {k: np.array(v) for k, v in pytorch_batch.items()}

  # 3. Initialize models with the same weights
  
  # Initialize PyTorch model and convert its weights for JAX
  pytorch_model = pytorch_workload._build_model(
      dropout_rate=0.0, aux_dropout_rate=0.0
  )
  pyt_params = sd_transform(
      {key_transform(k): v for k, v in pytorch_model.state_dict().items()}
  )
  
  # Initialize JAX model with the translated PyTorch weights
  jax_params, jax_model_state = jax_workload.init_model_fn(
      jax.random.PRNGKey(0), to_copy=pyt_params
  )

  # 4. Initialize optimizers
  # PyTorch Optimizer
  pytorch_optimizer = PytorchAdamW(pytorch_model.parameters(), lr=1e-3)
  
  # JAX Optimizer
  # In JAX, the optimizer is often stateless; its state is managed explicitly.
  jax_optimizer = JaxAdamW(learning_rate=1e-3)
  jax_opt_state = jax_optimizer.init(jax_params)

  # 5. Simple Training Loop for Comparison
  num_steps = 3
  print(f"--- Running comparison for {num_steps} steps ---\n")

  for step in range(num_steps):
    print(f"--- Step {step + 1} ---")

    # --- PyTorch Training Step ---
    pytorch_optimizer.zero_grad()
    pyt_logits, _ = pytorch_workload.model_fn(
        params=pytorch_model,
        augmented_and_preprocessed_input_batch=pytorch_batch,
        model_state=None,
        mode=spec.ForwardPassMode.TRAIN,
        rng=None,
        update_batch_norm=True,
    )
    pyt_loss_dict = pytorch_workload.loss_fn(
        label_batch=pytorch_batch['targets'],
        logits_batch=pyt_logits
    )
    pyt_loss = pyt_loss_dict['summed'] / pyt_loss_dict['n_valid_examples']
    pyt_loss.backward()
    pytorch_optimizer.step()

    # --- JAX Training Step ---
    def jax_update_step(params, model_state, opt_state, batch):
      def loss_fn(params):
        logits, new_model_state = jax_workload.model_fn(
            params=params,
            augmented_and_preprocessed_input_batch=batch,
            model_state=model_state,
            mode=spec.ForwardPassMode.TRAIN,
            rng=jax.random.PRNGKey(step),
            update_batch_norm=True,
        )
        loss_dict = jax_workload.loss_fn(
            label_batch=batch['targets'], logits_batch=logits
        )
        loss = loss_dict['summed'] / loss_dict['n_valid_examples']
        return loss, new_model_state

      (jax_loss, new_model_state), grads = jax.value_and_grad(
          loss_fn, has_aux=True
      )(params)
      
      updates, new_opt_state = jax_optimizer.update(grads, opt_state, params)
      new_params = jax.tree_map(lambda p, u: p + u, params, updates)

      return new_params, new_model_state, new_opt_state, jax_loss

    jax_params, jax_model_state, jax_opt_state, jax_loss = jax_update_step(
        jax_params, jax_model_state, jax_opt_state, jax_batch
    )

    # --- 6. Print and Compare Internal States ---
    print(f"Loss -> JAX: {jax_loss:.6f}, PyTorch: {pyt_loss.item():.6f}")

    # Compare a checksum of the first momentum vector (m)
    # For PyTorch, the state is in optimizer.state
    pyt_m_buffer = next(iter(pytorch_optimizer.state.values()))['exp_avg']
    # For JAX, the state is in the explicit opt_state object
    jax_m_buffer = jax.tree_util.tree_leaves(jax_opt_state.m)[0]

    print(f"Optimizer 'm' buffer norm -> JAX: {jnp.linalg.norm(jax_m_buffer):.6f}, PyTorch: {torch.linalg.norm(pyt_m_buffer):.6f}")
    
    # Compare a checksum of the model parameters
    pyt_param_norm = torch.linalg.norm(next(iter(pytorch_model.parameters())))
    jax_param_norm = jnp.linalg.norm(jax.tree_util.tree_leaves(jax_params)[0])

    print(f"First layer param norm -> JAX: {jax_param_norm:.6f}, PyTorch: {pyt_param_norm:.6f}\n")

