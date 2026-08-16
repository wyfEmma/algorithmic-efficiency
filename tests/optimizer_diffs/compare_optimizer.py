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
flags.DEFINE_string('workload', 'finewebedu_lm', 'Workload to run.')
flags.DEFINE_string('data_dir', '', 'Dataset location. If empty, uses default for workload.')

DEFAULT_DATA_DIRS = {
    'finewebedu_lm': '~/data/fineweb_edu_10B',
    'imagenet_resnet': '~/data/imagenet',
    'imagenet_vit': '~/data/imagenet',
    'librispeech_conformer': '~/data/librispeech',
    'librispeech_deepspeech': '~/data/librispeech',
    'ogbg': '~/data/ogbg',
    'fastmri': '~/data/fastmri',
    'wmt': '~/data/wmt',
}

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

def convert_batch(batch, framework, device):
    def to_framework(x):
        if x is None: return None
        if isinstance(x, (int, float, bool)): return x
        
        if hasattr(x, 'detach'):
             x_np = x.detach().cpu().numpy()
        elif hasattr(x, 'numpy'):
             x_np = x.numpy()
        elif hasattr(x, '__array__'):
             x_np = np.array(x)
        else:
             x_np = x
             
        if framework == 'jax':
             return jnp.array(x_np)
        elif framework == 'pytorch':
             return torch.tensor(x_np).to(device)
        else:
             return x_np
             
    try:
        import jax
        return jax.tree.map(to_framework, batch)
    except Exception:
        if isinstance(batch, dict):
             return {k: convert_batch(v, framework, device) for k, v in batch.items()}
        return to_framework(batch)

def key_transform(k, workload_name=None):
    k_str = ''.join(str(x) for x in k)
    
    # 1. Detect WMT specific keys (if not handled by SelfAttentionHandler)
    if any(x in k_str for x in ['TransformerDecoder_', 'TransformerEncoder_', 'MultiheadAttention']):
        new_key = []
        for i in k:
            if 'ModuleList' in i or 'TransformerDecoder_' in i or 'TransformerEncoder_' in i:
                continue
            if 'Linear' in i:
                i = 'out' if 'NonDynamicallyQuantizableLinear' in i else i.replace('Linear', 'Dense')
            elif i == 'Decoder_0': i = 'decoder'
            elif i == 'Encoder_0': i = 'encoder'
            elif 'TransformerEncoderLayer' in i: i = i.replace('TransformerEncoderLayer', 'encoderblock')
            elif 'TransformerDecoderLayer' in i: i = i.replace('TransformerDecoderLayer', 'encoderdecoderblock')
            elif 'MultiheadAttention' in i: i = i.replace('MultiheadAttention', 'SelfAttention')
            elif 'weight' in i: i = i.replace('weight', 'kernel')
            new_key.append(i)
        return tuple(new_key)
        
    # 2. Detect OGBG
    elif 'GraphNetwork' in k_str:
        from algoperf.workloads.ogbg.ogbg_jax.workload import OgbgWorkload
        wl = OgbgWorkload()
        hidden_dims = len(wl.hidden_dims)
        num_graphs = wl.num_message_passing_steps
        
        new_key = []
        bn = False
        ln = False
        graph_network = False
        graph_index = 0
        seq_index = 0
        for i in k:
            if 'GraphNetwork' in i:
                graph_network = True
                graph_index = int(i.split('_')[1])
            elif 'Sequential' in i:
                seq_index = int(i.split('_')[1])
            elif 'BatchNorm' in i: bn = True
            elif 'LayerNorm' in i: ln = True
            elif 'Linear' in i:
                layer_index = int(i.split('_')[1])
                if graph_network:
                    count = graph_index * 3 * hidden_dims + seq_index * hidden_dims + layer_index
                    i = 'Dense_' + str(count)
                elif layer_index == 0: i = 'node_embedding'
                elif layer_index == 1: i = 'edge_embedding'
                elif layer_index == 2:
                    count = num_graphs * 3 * hidden_dims
                    i = 'Dense_' + str(count)
            elif 'LayerNorm' in i:
                layer_index = int(i.split('_')[1])
                count = graph_index * 3 * hidden_dims + seq_index * hidden_dims + layer_index
                i = 'LayerNorm_' + str(count)
            elif 'weight' in i:
                i = i.replace('weight', 'scale' if (bn or ln) else 'kernel')
            new_key.append(i)
        return tuple(new_key)
        
    # 3. Detect ResNet
    elif 'Bottleneck' in k_str or 'ResNet' in k_str:
        new_key = []
        bn = False
        for i in k:
            bn = bn or 'BatchNorm' in i
            if 'ModuleList' in i: continue
            if 'Linear' in i: i = 'out' if 'NonDynamicallyQuantizableLinear' in i else i.replace('Linear', 'Dense')
            elif 'Conv2d' in i: i = i.replace('Conv2d', 'Conv')
            elif 'BatchNorm2d' in i: i = i.replace('BatchNorm2d', 'BatchNorm')
            elif 'MHSAwithQS' in i: i = i.replace('MHSAwithQS', 'SelfAttention')
            elif 'weight' in i: i = i.replace('weight', 'scale' if bn else 'kernel')
            new_key.append(i)
        return tuple(new_key)
        
    # 4. Detect FineWebEdu
    elif workload_name == 'finewebedu_lm' or 'embed_tokens' in k_str or 'lm_head' in k_str:
        new_key = []
        for i in k:
            if isinstance(i, str) and i.startswith('layers.'):
                layer_idx = i.split('.')[1]
                new_key.append(f'blocks_{layer_idx}')
            elif i == 'embed_tokens': new_key.append('embed')
            elif i == 'lm_head': new_key.append('output_proj')
            elif i == 'out_norm': new_key.append('out_ln')
            elif i == 'attn_norm': new_key.append('RMSNorm_0')
            elif i == 'mlp_norm': new_key.append('RMSNorm_1')
            elif i == 'mlp': new_key.append('Mlp_0')
            elif i == 'fc1': new_key.append('Dense_0')
            elif i == 'fc2': new_key.append('Dense_1')
            elif i == 'weight':
                if any(x in k_str for x in ['norm', 'ln', 'RMSNorm']):
                    new_key.append('scale')
                elif 'embed_tokens' in k_str:
                    new_key.append('embedding')
                else:
                    new_key.append('kernel')
            else:
                new_key.append(i)
        return tuple(new_key)
        
    # Fallback for others
    new_key = []
    bn = False
    for i in k:
        bn = bn or 'BatchNorm' in i
        if 'ModuleList' in i: continue
        if 'CustomBatchNorm' in i: continue
        
        if 'Linear' in i: i = 'out' if 'NonDynamicallyQuantizableLinear' in i else i.replace('Linear', 'Dense')
        elif 'Conv1d' in i: i = i.replace('Conv1d', 'Conv')
        elif 'Conv2d' in i: i = i.replace('Conv2d', 'Conv')
        elif 'CudnnLSTM' in i: i = i.replace('CudnnLSTM', 'LSTM')
            
        if 'weight' in i:
            if '_ih_' in i or '_hh_' in i: pass 
            else: i = 'scale' if bn else 'kernel'
        new_key.append(i)
    return tuple(new_key)

def sd_transform(sd, workload_name=None):
    from tests.optimizer_diffs.handlers import COMPONENT_HANDLERS
    
    out = {}
    # Convert string keys to tuples if they are strings
    pool = {}
    for k, v in sd.items():
        if isinstance(k, str):
            pool[tuple(k.split('.'))] = v
        else:
            pool[k] = v
    
    for handler in COMPONENT_HANDLERS:
        handler.apply(pool, out)
        
    # Fallback for remaining keys
    for k, v in pool.items():
        new_key = key_transform(k, workload_name)
        out[new_key] = v.detach().cpu().numpy() if hasattr(v, 'detach') else v
        
    return out



def print_schedule_free_stats_table(jax_params, pt_model_params, jax_opt_state, pt_opt_state, jax_workload, pt_workload, pt_model_state, step, workload_name):
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
    max_y_diff_key = None
    try:
        import flax
        # Flatten JAX params
        jax_flat = flax.traverse_util.flatten_dict(jax_params)
        
        # Translate PyTorch params using sd_transform
        pt_sd = pt_model_params.state_dict()
        translated_pt_sd = sd_transform(pt_sd, workload_name)
        
        # Compare keys
        max_y_diff = 0.0
        for k, v_jax in jax_flat.items():
             v_pt = translated_pt_sd.get(k)
             if v_pt is not None:
                  diff = jnp.max(jnp.abs(v_jax - v_pt)).item()
                  if diff > max_y_diff:
                       max_y_diff = diff
                       max_y_diff_key = k
                  if k == ('embed', 'embedding'):
                       print(f"Step {step} Embed Diff (y): {diff}")
             else:
                  # Some keys might not be in state dict or named differently
                  pass
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

    pt_z_leaves = []
    for p in pt_model_params.parameters():
        if p in pt_optimizer.state and 'z' in pt_optimizer.state[p]:
            pt_z_leaves.append(pt_optimizer.state[p]['z'].detach().cpu().numpy())

    pt_leaves_all = list(pt_model_params.parameters())
    print(f"pt_leaves len: {len(pt_leaves_all)}, pt_z_leaves len: {len(pt_z_leaves)}")
    pt_all_z = np.concatenate([p.flatten() for p in pt_z_leaves])
    pt_z_mean = np.mean(pt_all_z)
    pt_z_std = np.std(pt_all_z)
    pt_z_norm = np.sqrt(np.sum(pt_all_z**2))
    
    # Max Diff for z
    max_z_diff = float('nan')
    max_z_diff_key = None
    try:
        import flax
        # Flatten JAX z state
        jax_z_flat = flax.traverse_util.flatten_dict(jax_opt_state_core.z)
        
        # Translate PyTorch z state using sd_transform
        translated_pt_z = sd_transform(pt_z_dict, workload_name)
        
        # Compare keys
        max_z_diff = 0.0
        for k, v_jax in jax_z_flat.items():
             v_pt = translated_pt_z.get(k)
             if v_pt is not None:
                  diff = jnp.max(jnp.abs(v_jax - v_pt)).item()
                  if diff > max_z_diff:
                       max_z_diff = diff
                       max_z_diff_key = k
                  if k == ('embed', 'embedding'):
                       print(f"Step {step} Embed Diff (z): {diff}")
    except Exception as e:
        print(f"Warning: Could not calculate max diff for z: {e}")

    # --- Print Table ---
    print(f"\nSTEP {step:02d} Statistics Table:")
    print(f"{'Framework':<10} | {'y_mean':<15} | {'y_std':<15} | {'y_norm':<15} | {'z_mean':<15} | {'z_std':<15} | {'z_norm':<15}")
    print(f"{'-'*10}-|-{'-'*15}-|-{'-'*15}-|-{'-'*15}-|-{'-'*15}-|-{'-'*15}-|-{'-'*15}")
    print(f"{'JAX':<10} | {jax_y_mean:<15.8f} | {jax_y_std:<15.8f} | {jax_y_norm:<15.8f} | {jax_z_mean:<15.8f} | {jax_z_std:<15.8f} | {jax_z_norm:<15.8f}")
    print(f"{'PyTorch':<10} | {pt_y_mean:<15.8f} | {pt_y_std:<15.8f} | {pt_y_norm:<15.8f} | {pt_z_mean:<15.8f} | {pt_z_std:<15.8f} | {pt_z_norm:<15.8f}")
    print(f"{'Max Diff':<10} | {max_y_diff:<15.8f} | {'N/A':<15} | {'N/A':<15} | {max_z_diff:<15.8f} | {'N/A':<15} | {'N/A':<15}")
    print(f"{'='*120}")
    print(f"Max Diff Key for y: {max_y_diff_key}")
    print(f"Max Diff Key for z: {max_z_diff_key}\n")


def main(argv):
    workload_name = FLAGS.workload
    batch_size = 2
    num_steps = 20
    USE_PYTORCH_DDP, RANK, PYTORCH_DEVICE, N_GPUS = pytorch_setup()
    # Initialize a unified PRNG key for both frameworks
    rng = prng.PRNGKey(0)

    # Determine data_dir
    data_dir = FLAGS.data_dir
    if not data_dir:
        data_dir = DEFAULT_DATA_DIRS.get(workload_name, '~/data')
    data_dir = os.path.expanduser(data_dir)

    # 1. Translate the model
    print(f"Loading workload: {workload_name}")
    get_subfolder = lambda w: WORKLOAD_SUBFOLDER.get(w, w)
        
    jax_workload_module = importlib.import_module(f'algoperf.workloads.{workload_name}.{get_subfolder(workload_name)}_jax.workload')
    pt_workload_module = importlib.import_module(f'algoperf.workloads.{workload_name}.{get_subfolder(workload_name)}_pytorch.workload')
    workload_class_name = WORKLOAD_CLASSES[workload_name]
    jax_workload = getattr(jax_workload_module, workload_class_name)()
    pt_workload = getattr(pt_workload_module, workload_class_name)()
    
    print(f"Building input queue for split 'train' and data_dir {data_dir}...")
    data_rng, rng = prng.split(rng, 2)
    try:
        data_iter = pt_workload._build_input_queue(
            data_rng=data_rng,
            split='train',
            data_dir=data_dir,
            global_batch_size=batch_size,
        )
    except Exception as e:
        print(f"Warning: Could not build input queue using pt_workload: {e}")
        print("Falling back to JAX workload for input queue...")
        data_iter = jax_workload._build_input_queue(
            data_rng=data_rng,
            split='train',
            data_dir=data_dir,
            global_batch_size=batch_size,
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
            key_transform=lambda k: key_transform(k, workload_name),
            sd_transform=lambda sd: sd_transform(sd, workload_name),
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
        try:
            batch = next(data_iter)
            print(f"Step {step} Inputs Sum: {batch['inputs'].sum()}")
            print(f"Step {step} Targets Sum: {batch['targets'].sum()}")
        except StopIteration:
            print("Data iterator exhausted.")
            break
            
        jax_batch = convert_batch(batch, 'jax', None)
        pt_batch = convert_batch(batch, 'pytorch', PYTORCH_DEVICE)
        
        if step == 0:
            from algoperf import spec
            from tests.schedule_free.jax.submission import HPARAMS as JAX_HPARAMS
            from tests.schedule_free.pytorch.submission import HPARAMS as PT_HPARAMS
            print(f"JAX Warmup Steps: {int(JAX_HPARAMS['warmup_factor'] * jax_workload.step_hint * 0.75)}")
            print(f"PyTorch Warmup Steps: {int(PT_HPARAMS['warmup_factor'] * pt_workload.step_hint * 0.75)}")
            print(f"Step 0 Targets Min: {batch['targets'].min()}")
            print(f"Step 0 Targets Max: {batch['targets'].max()}")
            print("Computing Step 0 logits for comparison...")
            jax_logits, _ = jax_workload.model_fn(
                params=jax_params,
                batch=jax_batch,
                model_state=jax_model_state,
                mode=spec.ForwardPassMode.TRAIN,
                rng=rng,
                update_batch_norm=False,
            )
            pt_logits, _ = pt_workload.model_fn(
                params=pt_model_params,
                augmented_and_preprocessed_input_batch=pt_batch,
                model_state=pt_model_state,
                mode=spec.ForwardPassMode.TRAIN,
                rng=rng,
                update_batch_norm=False,
            )
            print(f"Step 0 JAX Logits Mean: {jnp.mean(jax_logits)}")
            print(f"Step 0 PyTorch Logits Mean: {pt_logits.mean().item()}")
            print(f"Step 0 Logits Max Diff: {np.max(np.abs(jax_logits - pt_logits.detach().cpu().numpy()))}")
            
            # Loss comparison
            import optax
            import torch.nn.functional as F
            
            jax_per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
                jax_logits.reshape(-1, jax_logits.shape[-1]),
                jax_batch['targets'].reshape(-1)
            )
            
            pt_per_token_loss = F.cross_entropy(
                pt_logits.view(-1, pt_logits.size(-1)),
                pt_batch['targets'].view(-1),
                reduction='none'
            )
            
            print(f"Step 0 JAX Loss Mean (Computed): {jax_per_token_loss.mean()}")
            print(f"Step 0 PyTorch Loss Mean (Computed): {pt_per_token_loss.mean().item()}")
            
            # Isolate loss function difference
            jax_logits_pt = torch.from_numpy(np.array(jax_logits)).to(PYTORCH_DEVICE)
            pt_loss_using_jax_logits = F.cross_entropy(
                jax_logits_pt.view(-1, jax_logits_pt.size(-1)),
                pt_batch['targets'].view(-1),
                reduction='none'
            )
            
            print(f"Step 0 PyTorch Loss using JAX Logits Mean: {pt_loss_using_jax_logits.mean().item()}")
            print(f"Step 0 Loss Func Diff: {np.abs(jax_per_token_loss.mean() - pt_loss_using_jax_logits.mean().item())}")
            
            # Gradient comparison (Embedding)
            pt_loss_dict = pt_workload.loss_fn(pt_batch['targets'], pt_logits, None)
            pt_loss = pt_loss_dict['summed'] / pt_loss_dict['n_valid_examples']
            pt_loss.backward()
            
            def jax_loss_fn(p):
                l, _ = jax_workload.model_fn(params=p, batch=jax_batch, model_state=jax_model_state, mode=spec.ForwardPassMode.TRAIN, rng=rng, update_batch_norm=False)
                loss_dict = jax_workload.loss_fn(jax_batch['targets'], l)
                return loss_dict['summed'] / loss_dict['n_valid_examples']
            
            jax_grads = jax.grad(jax_loss_fn)(jax_params)
            
            jax_embed_grad = jax_grads['embed']['embedding']
            pt_embed_grad = pt_workload._model.embed_tokens.weight.grad.detach().cpu().numpy()
            
            print(f"Step 0 Embed Grad Max Diff: {np.max(np.abs(jax_embed_grad - pt_embed_grad))}")
            
            # Zero gradients to avoid affecting subsequent steps
            pt_workload._model.zero_grad()
        
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
            jax_params, pt_model_params, jax_opt_state, pt_opt_state, jax_workload, pt_workload, pt_model_state, step, workload_name)

if __name__ == '__main__':
    app.run(main)
