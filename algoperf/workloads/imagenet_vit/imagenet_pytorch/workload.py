"""ImageNet ViT workload implemented in PyTorch."""

import contextlib
from typing import Dict, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from algoperf import param_utils, pytorch_utils, spec
from algoperf.workloads.imagenet_resnet.imagenet_pytorch.workload import (
  ImagenetResNetWorkload,
)
from algoperf.workloads.imagenet_vit.imagenet_pytorch import models
from algoperf.workloads.imagenet_vit.workload import (
  BaseImagenetVitWorkload,
  decode_variant,
)

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()


# Make sure we inherit from the ViT base workload first.
class ImagenetVitWorkload(BaseImagenetVitWorkload, ImagenetResNetWorkload):
  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = models.ViT(
      num_classes=self._num_classes,
      use_glu=self.use_glu,
      use_post_layer_norm=self.use_post_layer_norm,
      use_map=self.use_map,
      **decode_variant('S/16'),
    )
    self._param_shapes = param_utils.pytorch_param_shapes(model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = torch.nn.DataParallel(model)
    return model, None

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key in ['head.weight', 'head.bias']

  def model_fn(
    self,
    params: spec.ParameterContainer,
    augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
    model_state: spec.ModelAuxiliaryState,
    mode: spec.ForwardPassMode,
    rng: spec.RandomState,
    update_batch_norm: bool,
    dropout_rate: float = models.DROPOUT_RATE,
  ) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del rng
    del update_batch_norm

    model = params

    if mode == spec.ForwardPassMode.EVAL:
      model.eval()

    if mode == spec.ForwardPassMode.TRAIN:
      model.train()

    contexts = {
      spec.ForwardPassMode.EVAL: torch.no_grad,
      spec.ForwardPassMode.TRAIN: contextlib.nullcontext,
    }

    with contexts[mode]():
      logits_batch = model(
        augmented_and_preprocessed_input_batch['inputs'],
        dropout_rate=dropout_rate,
      )

    return logits_batch, None


class ImagenetVitGluWorkload(ImagenetVitWorkload):
  @property
  def use_glu(self) -> bool:
    return True

  @property
  def validation_target_value(self) -> float:
    return 0.75738

  @property
  def test_target_value(self) -> float:
    return 0.6359


class ImagenetVitPostLNWorkload(ImagenetVitWorkload):
  @property
  def use_post_layer_norm(self) -> bool:
    return True

  @property
  def validation_target_value(self) -> float:
    return 0.75312

  @property
  def test_target_value(self) -> float:
    return 0.6286


class ImagenetVitMapWorkload(ImagenetVitWorkload):
  @property
  def use_map(self) -> bool:
    return True

  @property
  def validation_target_value(self) -> float:
    return 0.77113

  @property
  def test_target_value(self) -> float:
    return 0.6523
