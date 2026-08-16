import numpy as np
import torch
import jax
import jax.numpy as jnp

def translate_linear(weight, bias=None):
    """Translate PyTorch Linear weights to JAX Dense weights."""
    translated = {'kernel': weight.detach().cpu().numpy().T}
    if bias is not None:
        translated['bias'] = bias.detach().cpu().numpy()
    return translated

def translate_conv1d(weight, bias=None):
    """Translate PyTorch Conv1d weights to JAX Conv weights.
    
    PyTorch Conv1d: [out_channels, in_channels, kernel_size]
    JAX Conv: [kernel_size, in_channels, out_channels]
    """
    w_np = weight.detach().cpu().numpy()
    translated = {'kernel': w_np.transpose(2, 1, 0)}
    if bias is not None:
        translated['bias'] = bias.detach().cpu().numpy()
    return translated

def translate_conv2d(weight, bias=None):
    """Translate PyTorch Conv2d weights to JAX Conv weights.
    
    PyTorch Conv2d: [out_channels, in_channels, k_h, k_w]
    JAX Conv: [k_h, k_w, in_channels, out_channels]
    """
    w_np = weight.detach().cpu().numpy()
    translated = {'kernel': w_np.transpose(2, 3, 1, 0)}
    if bias is not None:
        translated['bias'] = bias.detach().cpu().numpy()
    return translated

def translate_batchnorm2d(weight, bias, running_mean=None, running_var=None):
    """Translate PyTorch BatchNorm2d parameters to JAX BatchNorm.
    
    PyTorch uses weight/bias for affine transform, and running stats.
    JAX uses scale/bias, and mean/var in model_state.
    This handler returns a dict with all of them, caller can separate if needed.
    """
    translated = {
        'scale': weight.detach().cpu().numpy(),
        'bias': bias.detach().cpu().numpy(),
    }
    if running_mean is not None:
        translated['mean'] = running_mean.detach().cpu().numpy()
    if running_var is not None:
        translated['var'] = running_var.detach().cpu().numpy()
    return translated

def translate_lstm(weight_ih, weight_hh, bias_ih=None, bias_hh=None):
    """Translate PyTorch LSTM weights to JAX LSTM weights.
    
    PyTorch combines gates: Input, Forget, Cell, Output (IFGO).
    JAX expects separate gates: i, f, g, o.
    """
    dim = weight_ih.shape[0] // 4
    
    chunks_ih = torch.chunk(weight_ih, 4, dim=0)
    chunks_hh = torch.chunk(weight_hh, 4, dim=0)
    
    gate_names = ['i', 'f', 'g', 'o']
    translated = {}
    
    for name, c_ih, c_hh in zip(gate_names, chunks_ih, chunks_hh):
        translated[f'i{name}'] = {'kernel': c_ih.detach().cpu().numpy()}
        translated[f'h{name}'] = {'kernel': c_hh.detach().cpu().numpy()}
        
    if bias_ih is not None and bias_hh is not None:
        chunks_b_ih = torch.chunk(bias_ih, 4, dim=0)
        chunks_b_hh = torch.chunk(bias_hh, 4, dim=0)
        for name, c_b_ih, c_b_hh in zip(gate_names, chunks_b_ih, chunks_b_hh):
            translated[f'h{name}']['bias'] = (c_b_ih + c_b_hh).detach().cpu().numpy()
            
    return translated

def translate_attention_finewebedu(w_qkv, w_out, attn_scale, config):
    """Translate FineWebEdu Attention weights."""
    dim = config.model_dim
    n_heads = config.num_heads
    head_dim = dim // n_heads
    
    def reshape_for_flax(w, n_heads, head_dim):
        return w.reshape(n_heads, head_dim, -1).transpose(2, 0, 1)
        
    q_weight, k_weight, v_weight = [
        u.detach().cpu().numpy() for u in w_qkv.split(dim, dim=0)
    ]
    
    translated = {
        'query': {'kernel': reshape_for_flax(q_weight, n_heads, head_dim)},
        'key': {'kernel': reshape_for_flax(k_weight, n_heads, head_dim)},
        'value': {'kernel': reshape_for_flax(v_weight, n_heads, head_dim)},
        'attn_out_proj': {
            'kernel': w_out.detach().cpu().numpy().T
        },
        'attn_scale': attn_scale.detach().cpu().numpy(),
    }
    return translated

def translate_attention_separate_qkv(q_weight, k_weight, v_weight, out_weight,
                                     q_bias=None, k_bias=None, v_bias=None, out_bias=None):
    """Translate Attention where Q, K, V are separate modules."""
    translated = {
        'query': translate_linear(q_weight, q_bias),
        'key': translate_linear(k_weight, k_bias),
        'value': translate_linear(v_weight, v_bias),
        'out': translate_linear(out_weight, out_bias),
    }
    return translated

class LstmHandler:
    @staticmethod
    def apply(pool, out):
        lstm_groups = {}
        keys_to_del = []
        for k, v in pool.items():
            key_str = ''.join(str(x) for x in k)
            if 'LSTM' in key_str:
                layer_root = k[:-1]
                param_name = k[-1]
                if layer_root not in lstm_groups: lstm_groups[layer_root] = {}
                lstm_groups[layer_root][param_name] = v
                keys_to_del.append(k)
        
        for k in keys_to_del: del pool[k]
        
        for layer_root, params in lstm_groups.items():
            weight_ih = params.get('weight_ih_l0')
            weight_hh = params.get('weight_hh_l0')
            bias_ih = params.get('bias_ih_l0')
            bias_hh = params.get('bias_hh_l0')
            if weight_ih is not None and weight_hh is not None:
                translated = translate_lstm(weight_ih, weight_hh, bias_ih, bias_hh)
                param_names = list(params.keys())
                enc = 'LSTMSequenceEncoder_1' if any('reverse' in p for p in param_names) else 'LSTMSequenceEncoder_0'
                for jax_gate_key, gate_params in translated.items():
                     for suffix, chunk in gate_params.items():
                          new_path = layer_root + ('LSTM', enc, 'cell', jax_gate_key, suffix)
                          out[new_path] = chunk

class QkvAttentionHandler:
    @staticmethod
    def apply(pool, out):
        keys_to_del = []
        for k, v in pool.items():
            key_str = '.'.join(str(x) for x in k)
            if 'w_qkv.weight' in key_str:
                try:
                    attn_idx = k.index('attn')
                    layer_root = k[:attn_idx+1]
                    w_qkv = v
                    w_out = None
                    attn_scale = None
                    
                    for k2, v2 in pool.items():
                        if k2[:attn_idx+1] == layer_root and 'w_out' in str(k2):
                             w_out = v2
                             break
                    for k2, v2 in pool.items():
                         if k2[:attn_idx+1] == layer_root and 'attn_scale' in str(k2):
                             attn_scale = v2
                             break
                             
                    if w_out is not None:
                         class FakeConfig:
                              def __init__(self):
                                   self.model_dim = w_qkv.shape[1]
                                   self.num_heads = 8
                                   self.tie_embeddings = True
                         config = FakeConfig()
                         attn_params = translate_attention_finewebedu(w_qkv, w_out, attn_scale, config)
                         
                         layer_idx = k[attn_idx-1]
                         block_key = f'blocks_{layer_idx}'
                         for k1, v1 in attn_params.items():
                              if k1 == 'attn_scale': out[(block_key, 'CausalAttn_0', k1)] = v1
                              else:
                                   for k2, v2 in v1.items(): out[(block_key, 'CausalAttn_0', k1, k2)] = v2
                                   
                         keys_to_del.append(k)
                         for k2 in pool.keys():
                              if k2[:attn_idx+1] == layer_root and ('w_out' in str(k2) or 'attn_scale' in str(k2)):
                                   keys_to_del.append(k2)
                except ValueError:
                    continue
        for k in keys_to_del: del pool[k]

class SelfAttentionHandler:
    @staticmethod
    def apply(pool, out):
        keys_to_del = []
        for k, v in pool.items():
          k_str = ''.join(str(x) for x in k)
          if 'SelfAttention' in k_str:
            new_key = list(k)
            base_path = tuple(str(x).replace('SelfAttention', 'MultiHeadDotProductAttention') for x in k[:-2])
            if 'SelfAttention_0' in k_str:
              if new_key[-2] == 'Dense_0':
                for name, value in zip(('query', 'key', 'value'), v.chunk(3)): 
                    out[base_path + (name, new_key[-1])] = value.detach().cpu().numpy()
              elif new_key[-2] == 'Dense_1': 
                    out[base_path + ('out', new_key[-1])] = v.detach().cpu().numpy()
            else:
              if new_key[-2] == 'Dense_0': 
                    out[base_path + ('query', new_key[-1])] = v.detach().cpu().numpy()
              elif new_key[-2] == 'Dense_1':
                for name, value in zip(('key', 'value'), v.chunk(2)): 
                    out[base_path + (name, new_key[-1])] = value.detach().cpu().numpy()
              elif new_key[-2] == 'Dense_2': 
                    out[base_path + ('out', new_key[-1])] = v.detach().cpu().numpy()
            keys_to_del.append(k)
            
        for k in keys_to_del: del pool[k]

class ConvBlockHandler:
    @staticmethod
    def apply(pool, out):
        def sort_key(k):
          if k[0] == 'ModuleList_0': return (0, *k)
          if k[0] == 'ConvBlock_0': return (1, *k)
          if k[0] == 'ModuleList_1': return (2, *k)
          if k[0] == 'ModuleList_2': return (3, *k)
          return (4, *k)
          
        keys = sorted([k for k in pool.keys() if 'ConvBlock' in str(k) or 'Conv2d' in str(k) or 'ConvTranspose2d' in str(k) or 'ModuleList' in str(k)], key=sort_key)
        if not keys: return
        
        c = 0
        keys_to_del = []
        for idx, k in enumerate(keys):
          new_key = []
          for idx2, i in enumerate(k):
            if 'ModuleList' in i or 'Sequential' in i: continue
            if i.startswith('ConvBlock'):
              if idx != 0 and keys[idx - 1][: idx2 + 1] != k[: idx2 + 1]: c += 1
              i = f'ConvBlock_{c}'
            if 'Conv2d' in i: i = i.replace('Conv2d', 'Conv')
            if 'ConvTranspose2d' in i: i = i.replace('ConvTranspose2d', 'ConvTranspose')
            if 'weight' in i: i = i.replace('weight', 'kernel')
            new_key.append(i)
          out[tuple(new_key)] = pool[k].detach().cpu().numpy()
          keys_to_del.append(k)
          
        for k in keys_to_del: del pool[k]

class BottleneckHandler:
    @staticmethod
    def apply(pool, out):
        keys = sorted([k for k in pool.keys() if 'Bottleneck' in ''.join(str(x) for x in k)])
        if not keys: return
        
        c = -1
        prev = None
        keys_to_del = []
        for k in keys:
          if prev is None or prev != k[:2]:
            prev = k[:2]
            c += 1
          new_key = (f'BottleneckResNetBlock_{c}',) + k[2:]
          if 'Sequential' in ''.join(str(x) for x in new_key):
            new_key = tuple([(i.replace('_0', '_proj') if 'BatchNorm' in i or 'Conv' in i else i) for i in new_key if 'Sequential' not in i])
          out[new_key] = pool[k].detach().cpu().numpy()
          keys_to_del.append(k)
          
        for k in keys_to_del: del pool[k]

COMPONENT_HANDLERS = [LstmHandler, QkvAttentionHandler, SelfAttentionHandler, ConvBlockHandler, BottleneckHandler]

# Registry of handlers
HANDLERS = {
    'linear': translate_linear,
    'conv1d': translate_conv1d,
    'conv2d': translate_conv2d,
    'batchnorm2d': translate_batchnorm2d,
    'lstm': translate_lstm,
    'attention_finewebedu': translate_attention_finewebedu,
    'attention_separate_qkv': translate_attention_separate_qkv,
}

def get_handler(name):
    """Get a handler by name (case-insensitive)."""
    return HANDLERS.get(name.lower())
