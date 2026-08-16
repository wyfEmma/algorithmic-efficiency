import importlib
import types
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=3'

import jax
import jax.numpy as jnp
import numpy as np
import torch

from algoperf.pytorch_utils import pytorch_setup
from algoperf import random_utils as prng
from absl import app, flags
from tests.modeldiffs import diff as diff_utils
from tests.schedule_free.jax import submission as jax_submission
from tests.schedule_free.pytorch import submission as pytorch_submission
from tests.optimizer_diffs import handlers

FLAGS = flags.FLAGS
flags.DEFINE_string('framework', 'jax', 'Framework for PRNG generation.')
flags.DEFINE_string('data_dir', '~/data/fineweb_edu_10B', 'Dataset location.')

WORKLOAD_SUBFOLDER = {
    "librispeech_conformer":"librispeech",
    "librispeech_deepspeech":"librispeech",
    "imagenet_resnet":"imagenet",
    "imagenet_vit":"imagenet",
}

WORKLOAD_CLASSES = {
    'mnist': 'MnistWorkload',
    'cifar': 'CifarWorkload',
    'criteo1tb': 'Criteo1TbDlrmSmallWorkload',
    'fastmri': 'FastMRIWorkload',
    'imagenet_resnet': 'ImagenetResNetWorkload',
    'imagenet_vit': 'ImagenetVitWorkload',
    'librispeech_conformer': 'LibriSpeechConformerWorkload',
    'librispeech_deepspeech': 'LibriSpeechDeepSpeechWorkload',
    'ogbg': 'OgbgWorkload',
    'wmt': 'WmtWorkload',
    'finewebedu_lm': 'LmWorkload',
}

def get_fake_batch(workload_name, batch_size=16):
    """Translate the data: Create identical fake data batches for both frameworks."""
    
    if workload_name in ['librispeech_deepspeech', 'librispeech_conformer']:
        # LibriSpeech inputs: tuple of (audio_samples, paddings)
        # Audio signals are typically up to 320,000 samples long
        inputs = np.random.normal(size=(batch_size, 320000)).astype(np.float32)
        input_paddings = np.zeros((batch_size, 320000), dtype=np.float32)
        
        # Targets: tuple of (token_sequences, paddings)
        targets = np.random.randint(1, 1024, size=(batch_size, 256)).astype(np.int32)
        target_paddings = np.zeros((batch_size, 256), dtype=np.float32)
        
        jax_batch = {
            'inputs': (jnp.array(inputs), jnp.array(input_paddings)),
            'targets': (jnp.array(targets), jnp.array(target_paddings)),
        }
        
        pt_batch = {
            'inputs': (torch.tensor(inputs, dtype=torch.float32), 
                       torch.tensor(input_paddings, dtype=torch.float32)),
            'targets': (torch.tensor(targets, dtype=torch.long), 
                        torch.tensor(target_paddings, dtype=torch.float32)),
        }
    elif workload_name == 'finewebedu_lm':
        seq_len = 1024
        vocab_size = 50257
        inputs = np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
        targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
        weights = np.ones((batch_size, seq_len), dtype=np.float32)
        
        jax_batch = {
            'inputs': jnp.array(inputs),
            'targets': jnp.array(targets),
            'weights': jnp.array(weights),
        }
        
        pt_batch = {
            'inputs': torch.tensor(inputs, dtype=torch.int32),
            'targets': torch.tensor(targets, dtype=torch.int64),
            'weights': torch.tensor(weights, dtype=torch.float32),
        }
    else:
        raise NotImplementedError(f"Fake batch generation for {workload_name} not implemented.")
        
    return jax_batch, pt_batch

def key_transform(k):
    new_key = []
    bn = False
    for i in k:
        bn = bn or 'BatchNorm' in i
        if 'ModuleList' in i: continue
        if 'CustomBatchNorm' in i: continue
        
        if 'Linear' in i:
            i = 'out' if 'NonDynamicallyQuantizableLinear' in i else i.replace('Linear', 'Dense')
        elif 'Conv1d' in i:
            i = i.replace('Conv1d', 'Conv')
        elif 'CudnnLSTM' in i:
            i = i.replace('CudnnLSTM', 'LSTM')
            
        if 'weight' in i:
            if '_ih_' in i or '_hh_' in i:
                pass 
            else:
                i = 'scale' if bn else 'kernel'
        new_key.append(i)
    return tuple(new_key)

def sd_transform(sd):
    out = {}
    # Track which original keys we've processed so we can remove them at the end
    keys_to_del = []

    for k, v in sd.items():
        if 'LSTM' in ''.join(k):
            # Example k: ('BatchRNN_0', 'CudnnLSTM_0', 'weight_ih_l0')
            layer_root = k[:-1]  # ('BatchRNN_0', 'CudnnLSTM_0')
            param_name = k[-1]   # 'weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0'

            # Identify Direction
            if 'reverse' in param_name:
                enc = 'LSTMSequenceEncoder_1'
            else:
                enc = 'LSTMSequenceEncoder_0'

            # Identify Source
            is_input_source = '_ih_' in param_name
            source_char = 'i' if is_input_source else 'h'

            # Identify Parameter Type
            is_bias = 'bias' in param_name
            suffix = 'bias' if is_bias else 'kernel'

            # Split into 4 Gates
            # PyTorch gate order: Input, Forget, Cell Gate, Output
            chunks = torch.chunk(v, 4, dim=0)
            gate_chars = ['i', 'f', 'g', 'o']

            for gate_char, chunk in zip(gate_chars, chunks):
                # If we see 'ih_bias' and 'hh_bias' in PyTorch, we sum them into JAX 'h{gate}_bias'
                if is_bias:
                    jax_gate_key = 'h' + gate_char  # Always map to hi, hf, hg, ho
                else:
                    jax_gate_key = source_char + gate_char # ii, if, ig, io OR hi, hf, hg, ho

                # Construct the full JAX Path
                new_path = layer_root + ('LSTM', enc, 'cell', jax_gate_key, suffix)

                # Merge Biases, Jax only have bias on hi, hf, hg, ho gates.
                if new_path in out and is_bias:
                    out[new_path] = out[new_path] + chunk
                else:
                    out[new_path] = chunk
            
            keys_to_del.append(k)
        else:
            # For non-LSTM layers, keep them as they are
            out[k] = v

    # Clean up the original PyTorch keys
    for k in keys_to_del:
        if k in out:
            del out[k]

    return out



def print_schedule_free_stats_table(jax_params, pt_model_params, jax_opt_state, pt_opt_state, jax_workload, pt_workload, pt_model_state, step):
    # --- Row y (Parameters) ---
    # JAX stats
    jax_leaves = jax.tree_util.tree_leaves(jax_params)
    jax_all_params = jnp.concatenate([p.flatten() for p in jax_leaves])
    jax_y_mean = jnp.mean(jax_all_params).item()
    jax_y_std = jnp.std(jax_all_params).item()
    jax_y_norm = jnp.sqrt(jnp.sum(jax_all_params**2)).item()
    
    # PyTorch stats
    pt_leaves = [p.flatten() for p in pt_model_params.parameters()]
    pt_all_params = torch.cat(pt_leaves)
    pt_y_mean = torch.mean(pt_all_params).item()
    pt_y_std = torch.std(pt_all_params).item()
    pt_y_norm = torch.sqrt(torch.sum(pt_all_params**2)).item()
    
    # Max Diff for y
    max_y_diff = float('nan')
    try:
        if jax_workload.__class__.__name__ == 'LmWorkload':
            from flax.core import unfreeze, freeze
            from algoperf import jax_sharding_utils
            
            translated_jax_params = jax.tree_util.tree_map(lambda x: x, jax_params)
            translated_jax_params = unfreeze(translated_jax_params)
            
            pytorch_model = pt_model_params
            if hasattr(pytorch_model, 'module'):
                pytorch_model = pytorch_model.module
                
            config = pytorch_model.cfg
            n_layers = config.num_layers
            n_heads = config.num_heads
            dim = config.model_dim
            head_dim = dim // n_heads
            
            def reshape_for_flax(w, n_heads, head_dim):
                return w.reshape(n_heads, head_dim, -1).transpose(2, 0, 1)
                
            # Copy embedding
            translated_jax_params['embed']['embedding'] = pytorch_model.embed_tokens.weight.detach().cpu().numpy()
            
            # Copy blocks
            for i in range(n_layers):
                pytorch_block = pytorch_model.layers[i]
                w_qkv = pytorch_block.attn.w_qkv.weight
                q_weight, k_weight, v_weight = [
                    u.detach().cpu().numpy() for u in w_qkv.split(dim, dim=0)
                ]
                
                attn_params = {
                    'query': {'kernel': reshape_for_flax(q_weight, n_heads, head_dim)},
                    'key': {'kernel': reshape_for_flax(k_weight, n_heads, head_dim)},
                    'value': {'kernel': reshape_for_flax(v_weight, n_heads, head_dim)},
                    'attn_out_proj': {
                        'kernel': pytorch_block.attn.w_out.weight.detach().cpu().numpy().T
                    },
                    'attn_scale': pytorch_block.attn.attn_scale.detach().cpu().numpy(),
                }
                
                mlp_params = {
                    'Dense_0': {'kernel': pytorch_block.mlp.fc1.weight.detach().cpu().numpy().T},
                    'Dense_1': {'kernel': pytorch_block.mlp.fc2.weight.detach().cpu().numpy().T},
                }
                
                attn_norm = {'scale': pytorch_block.attn_norm.weight.detach().cpu().numpy()}
                mlp_norm = {'scale': pytorch_block.mlp_norm.weight.detach().cpu().numpy()}
                
                block_key = f'blocks_{i}'
                translated_jax_params[block_key] = {
                    'CausalAttn_0': attn_params,
                    'Mlp_0': mlp_params,
                    'RMSNorm_0': attn_norm,
                    'RMSNorm_1': mlp_norm,
                }
                
            # Output norm
            translated_jax_params['out_ln'] = {
                'scale': pytorch_model.out_norm.weight.detach().cpu().numpy()
            }
            
            # Output projection (tied or untied)
            if not config.tie_embeddings:
                translated_jax_params['output_proj'] = {
                    'kernel': pytorch_model.lm_head.weight.detach().cpu().numpy().T
                }
                
            translated_jax_params = freeze(translated_jax_params)
            translated_jax_params = jax_sharding_utils.replicate(translated_jax_params)
        else:
            translated_jax_params, _, _ = diff_utils.torch2jax(
                jax_workload=jax_workload,
                pytorch_workload=pt_workload,
                key_transform=key_transform,
                sd_transform=sd_transform,
                unfreeze_jax_param=True,
                pytorch_model=pt_model_params,
                jax_params=jax_params,
            )
        trans_leaves = jax.tree_util.tree_leaves(translated_jax_params)
        if len(jax_leaves) == len(trans_leaves):
            max_y_diff = 0.0
            for p1, p2 in zip(jax_leaves, trans_leaves):
                diff = jnp.max(jnp.abs(p1 - p2)).item()
                if diff > max_y_diff:
                    max_y_diff = diff
        else:
            print(f"Warning: Number of JAX leaves ({len(jax_leaves)}) does not match translated leaves ({len(trans_leaves)}).")
    except Exception as e:
        print(f"Warning: Could not calculate max diff for y: {e}")

    # --- Row z (Optimizer State) ---
    # JAX stats
    jax_opt_state_core = jax_opt_state[0][0]
    jax_z_leaves = jax.tree_util.tree_leaves(jax_opt_state_core.z)
    jax_all_z = jnp.concatenate([p.flatten() for p in jax_z_leaves])
    jax_z_mean = jnp.mean(jax_all_z).item()
    jax_z_std = jnp.std(jax_all_z).item()
    jax_z_norm = jnp.sqrt(jnp.sum(jax_all_z**2)).item()
    
    # PyTorch stats
    pt_optimizer = pt_opt_state['optimizer']
    pt_z_dict = {}
    for name, p in pt_model_params.named_parameters():
        if p in pt_optimizer.state:
            param_state = pt_optimizer.state[p]
            if 'z' in param_state:
                pt_z_dict[name] = param_state['z']

    # Build pt_z_leaves in order of parameters() for global stats
    pt_z_leaves = []
    for p in pt_model_params.parameters():
        if p in pt_optimizer.state and 'z' in pt_optimizer.state[p]:
            pt_z_leaves.append(pt_optimizer.state[p]['z'].detach().cpu().numpy())

    pt_all_z = np.concatenate([p.flatten() for p in pt_z_leaves])
    pt_z_mean = np.mean(pt_all_z)
    pt_z_std = np.std(pt_all_z)
    pt_z_norm = np.sqrt(np.sum(pt_all_z**2))
    
    # Max Diff for z
    max_z_diff = float('nan')
    try:
        if jax_workload.__class__.__name__ == 'LmWorkload':
            from flax.core import unfreeze, freeze
            from algoperf import jax_sharding_utils
            
            # Use jax_params as template to get structure
            translated_jax_z = jax.tree_util.tree_map(lambda x: x, jax_params)
            translated_jax_z = unfreeze(translated_jax_z)
            
            pytorch_model = pt_model_params
            if hasattr(pytorch_model, 'module'):
                pytorch_model = pytorch_model.module
            config = pytorch_model.cfg
            
            def get_z(name):
                t = pt_z_dict.get(name)
                if t is None:
                     p = dict(pytorch_model.named_parameters())[name]
                     return torch.zeros_like(p)
                return t
                
            # Copy embedding
            translated_jax_z['embed']['embedding'] = get_z('embed_tokens.weight').detach().cpu().numpy()
            
            # Copy blocks
            for i in range(config.num_layers):
                w_qkv = get_z(f'layers.{i}.attn.w_qkv.weight')
                w_out = get_z(f'layers.{i}.attn.w_out.weight')
                attn_scale = get_z(f'layers.{i}.attn.attn_scale')
                
                # Use handler!
                attn_params = handlers.translate_attention_finewebedu(w_qkv, w_out, attn_scale, config)
                
                fc1 = get_z(f'layers.{i}.mlp.fc1.weight')
                fc2 = get_z(f'layers.{i}.mlp.fc2.weight')
                
                mlp_params = {
                    'Dense_0': {'kernel': fc1.detach().cpu().numpy().T},
                    'Dense_1': {'kernel': fc2.detach().cpu().numpy().T},
                }
                
                attn_norm = {'scale': get_z(f'layers.{i}.attn_norm.weight').detach().cpu().numpy()}
                mlp_norm = {'scale': get_z(f'layers.{i}.mlp_norm.weight').detach().cpu().numpy()}
                
                block_key = f'blocks_{i}'
                translated_jax_z[block_key] = {
                    'CausalAttn_0': attn_params,
                    'Mlp_0': mlp_params,
                    'RMSNorm_0': attn_norm,
                    'RMSNorm_1': mlp_norm,
                }
                
            # Output norm
            translated_jax_z['out_ln'] = {
                'scale': get_z('out_norm.weight').detach().cpu().numpy()
            }
            
            # Output projection
            if not config.tie_embeddings:
                translated_jax_z['output_proj'] = {
                    'kernel': get_z('lm_head.weight').detach().cpu().numpy().T
                }
                
            translated_jax_z = freeze(translated_jax_z)
            translated_jax_z = jax_sharding_utils.replicate(translated_jax_z)
            
            # Now compare
            trans_z_leaves = jax.tree_util.tree_leaves(translated_jax_z)
            if len(jax_z_leaves) == len(trans_z_leaves):
                max_z_diff = 0.0
                for p1, p2 in zip(jax_z_leaves, trans_z_leaves):
                    diff = jnp.max(jnp.abs(p1 - p2)).item()
                    if diff > max_z_diff:
                        max_z_diff = diff
            else:
                print(f"Warning: Number of JAX z leaves ({len(jax_z_leaves)}) does not match translated z leaves ({len(trans_z_leaves)}).")
                
        else:
            # Fallback to simple shape-based comparison if lengths match
            if len(jax_z_leaves) == len(pt_z_leaves):
                max_z_diff = 0.0
                for p_jax, p_pt in zip(jax_z_leaves, pt_z_leaves):
                    p_pt_processed = p_pt.T if p_jax.shape != p_pt.shape and p_jax.shape == p_pt.T.shape else p_pt
                    diff = jnp.max(jnp.abs(p_jax - p_pt_processed)).item()
                    if diff > max_z_diff:
                        max_z_diff = diff
            else:
                print(f"Warning: Number of JAX z leaves ({len(jax_z_leaves)}) does not match PyTorch z leaves ({len(pt_z_leaves)}).")
    except Exception as e:
        print(f"Warning: Could not calculate max diff for z: {e}")

    # --- Print Table ---
    print(f"\nSTEP {step:02d} Statistics Table:")
    print(f"{'Row':<5} | {'jax_mean':<15} | {'jax_std':<15} | {'pytorch_mean':<15} | {'pytorch_std':<15} | {'max_diff':<15} | {'jax_norm':<15} | {'pytorch_norm':<15}")
    print(f"{'-'*5}-|-{'-'*15}-|-{'-'*15}-|-{'-'*15}-|-{'-'*15}-|-{'-'*15}-|-{'-'*15}-|-{'-'*15}")
    print(f"{'y':<5} | {jax_y_mean:<15.8f} | {jax_y_std:<15.8f} | {pt_y_mean:<15.8f} | {pt_y_std:<15.8f} | {max_y_diff:<15.8f} | {jax_y_norm:<15.8f} | {pt_y_norm:<15.8f}")
    print(f"{'z':<5} | {jax_z_mean:<15.8f} | {jax_z_std:<15.8f} | {pt_z_mean:<15.8f} | {pt_z_std:<15.8f} | {max_z_diff:<15.8f} | {jax_z_norm:<15.8f} | {pt_z_norm:<15.8f}")
    print(f"{'='*120}\n")


def main(argv):
    workload_name = 'finewebedu_lm'
    batch_size = 2
    num_steps = 20
    USE_PYTORCH_DDP, RANK, PYTORCH_DEVICE, N_GPUS = pytorch_setup()
    # Initialize a unified PRNG key for both frameworks
    rng = prng.PRNGKey(0)

    # 1. Translate the model
    print(f"Loading workload: {workload_name}")
    get_subfolder = lambda w: WORKLOAD_SUBFOLDER.get(w, w)
        
    jax_workload_module = importlib.import_module(f'algoperf.workloads.{workload_name}.{get_subfolder(workload_name)}_jax.workload')
    pt_workload_module = importlib.import_module(f'algoperf.workloads.{workload_name}.{get_subfolder(workload_name)}_pytorch.workload')
    workload_class_name = WORKLOAD_CLASSES[workload_name]
    jax_workload = getattr(jax_workload_module, workload_class_name)()
    pt_workload = getattr(pt_workload_module, workload_class_name)()
    
    if workload_name == 'finewebedu_lm':
        from algoperf.workloads.finewebedu_lm.input_pipeline import get_data_iter
        data_rng, rng = prng.split(rng, 2)
        data_iter = get_data_iter(
            data_rng=data_rng,
            split='train',
            data_dir=os.path.expanduser(FLAGS.data_dir),
            batch_size=batch_size,
        )
    
    print("Translating PyTorch and JAX models using diff_utils.torch2jax...")
    pt_model_init_rng, opt_init_rng, rng = prng.split(rng, 3)

    # Initialize PyTorch model params
    pt_model_params, pt_model_state = pt_workload.init_model_fn(pt_model_init_rng)
    if USE_PYTORCH_DDP:
        pt_model_params = pt_model_params.to(PYTORCH_DEVICE)
    
    # Translate to JAX
    if workload_name == 'finewebedu_lm':
        from flax.core import unfreeze, freeze
        from algoperf import jax_sharding_utils
        
        # Initialize JAX params to get the structure
        jax_params, jax_model_state = jax_workload.init_model_fn(rng)
        
        pytorch_model = pt_workload._model
        if hasattr(pytorch_model, 'module'):
            pytorch_model = pytorch_model.module
            
        config = pytorch_model.cfg
        n_layers = config.num_layers
        n_heads = config.num_heads
        dim = config.model_dim
        head_dim = dim // n_heads
        
        def reshape_for_flax(w, n_heads, head_dim):
            return w.reshape(n_heads, head_dim, -1).transpose(2, 0, 1)
            
        jax_params = unfreeze(jax_params)
        
        # Copy embedding
        jax_params['embed']['embedding'] = pytorch_model.embed_tokens.weight.detach().cpu().numpy()
        
        # Copy blocks
        for i in range(n_layers):
            pytorch_block = pytorch_model.layers[i]
            w_qkv = pytorch_block.attn.w_qkv.weight
            q_weight, k_weight, v_weight = [
                u.detach().cpu().numpy() for u in w_qkv.split(dim, dim=0)
            ]
            
            attn_params = {
                'query': {'kernel': reshape_for_flax(q_weight, n_heads, head_dim)},
                'key': {'kernel': reshape_for_flax(k_weight, n_heads, head_dim)},
                'value': {'kernel': reshape_for_flax(v_weight, n_heads, head_dim)},
                'attn_out_proj': {
                    'kernel': pytorch_block.attn.w_out.weight.detach().cpu().numpy().T
                },
                'attn_scale': pytorch_block.attn.attn_scale.detach().cpu().numpy(),
            }
            
            mlp_params = {
                'Dense_0': {'kernel': pytorch_block.mlp.fc1.weight.detach().cpu().numpy().T},
                'Dense_1': {'kernel': pytorch_block.mlp.fc2.weight.detach().cpu().numpy().T},
            }
            
            attn_norm = {'scale': pytorch_block.attn_norm.weight.detach().cpu().numpy()}
            mlp_norm = {'scale': pytorch_block.mlp_norm.weight.detach().cpu().numpy()}
            
            block_key = f'blocks_{i}'
            jax_params[block_key] = {
                'CausalAttn_0': attn_params,
                'Mlp_0': mlp_params,
                'RMSNorm_0': attn_norm,
                'RMSNorm_1': mlp_norm,
            }
            
        # Output norm
        jax_params['out_ln'] = {
            'scale': pytorch_model.out_norm.weight.detach().cpu().numpy()
        }
        
        # Output projection (tied or untied)
        if not config.tie_embeddings:
            jax_params['output_proj'] = {
                'kernel': pytorch_model.lm_head.weight.detach().cpu().numpy().T
            }
            
        jax_params = freeze(jax_params)
        jax_params = jax_sharding_utils.replicate(jax_params)
    else:
        jax_params, jax_model_state, _ = diff_utils.torch2jax(
            jax_workload=jax_workload,
            pytorch_workload=pt_workload,
            key_transform=key_transform,
            sd_transform=sd_transform,
            unfreeze_jax_param=True,
        )
    
    # mock hyperparameters
    hyperparameters = types.SimpleNamespace(label_smoothing=0.0, grad_clip=None)
    
    print("Initializing optimizers...")
    
    jax_opt_state = jax_submission.init_optimizer_state(
        workload=jax_workload,
        model_params=jax_params,
        model_state=jax_model_state,
        hyperparameters=hyperparameters,
        rng=opt_init_rng
    )
    
    pt_opt_state = pytorch_submission.init_optimizer_state(
        workload=pt_workload,
        model_params=pt_model_params,
        model_state=pt_model_state,
        hyperparameters=hyperparameters,
        rng=opt_init_rng
    )
    
    print(f"Starting {num_steps} iterations of training...")
    for step in range(num_steps):
        print(f"\n--- STEP {step} ---")
        if workload_name == 'finewebedu_lm':
            try:
                batch = next(data_iter)
            except StopIteration:
                print("Data iterator exhausted.")
                break
            
            jax_batch = {
                'inputs': jnp.array(batch['inputs']),
                'targets': jnp.array(batch['targets']),
                'weights': jnp.array(batch['weights']) if batch['weights'] is not None else None,
            }
            
            pt_batch = {
                'inputs': torch.tensor(batch['inputs'], dtype=torch.int32),
                'targets': torch.tensor(batch['targets'], dtype=torch.int64),
                'weights': torch.tensor(batch['weights'], dtype=torch.float32) if batch['weights'] is not None else None,
            }
        else:
            jax_batch, pt_batch = get_fake_batch(workload_name, batch_size)
        pt_batch = {k: (tuple(x.to(PYTORCH_DEVICE) for x in v) if isinstance(v, tuple) else (v.to(PYTORCH_DEVICE) if v is not None else None)) 
                    for k, v in pt_batch.items()}
        
        update_rng, rng = prng.split(rng, 2)
        torch.cuda.empty_cache()

        # 3. JAX Training Step
        jax_opt_state, jax_params, jax_model_state = jax_submission.update_params(
            workload=jax_workload, current_param_container=jax_params, current_params_types=jax_workload.model_params_types,
            model_state=jax_model_state, hyperparameters=hyperparameters, batch=jax_batch, loss_type=jax_workload.loss_type,
            optimizer_state=jax_opt_state, eval_results=[], global_step=step, rng=update_rng)
        
        # 4. PyTorch Training Step
        pt_opt_state, pt_model_params, pt_model_state = pytorch_submission.update_params(
            workload=pt_workload, current_param_container=pt_model_params, current_params_types=pt_workload.model_params_types,
            model_state=pt_model_state, hyperparameters=hyperparameters, batch=pt_batch, loss_type=pt_workload.loss_type,
            optimizer_state=pt_opt_state, eval_results=[], global_step=step, rng=update_rng)
        
        # 5. Print data and compare
        print_schedule_free_stats_table(
            jax_params, pt_model_params, jax_opt_state, pt_opt_state, jax_workload, pt_workload, pt_model_state, step)

if __name__ == '__main__':
    app.run(main)
