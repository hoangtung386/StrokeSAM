"""Image-only HPMA network wrapping a SAM 3.1 image model."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory.prototype_bank import PrototypeMemoryBank
from models.adapters.global_adapter import GlobalVisualAdapter, VisualTokenGenerator
from models.adapters.local_adapter import LocalProjectionAdapter
from models.adapters.structural_adapter import StructuralQueryAdapter
from models.sam3.features import extract_image_fpn_features
from models.sam3.model_builder import build_sam3_image_model


class InputAdapter(nn.Module):
    """Project arbitrary image channel counts to the three SAM input channels."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.projection: nn.Module
        if in_channels == out_channels:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, out_channels),
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} image channels, got {images.shape[1]}.")
        return self.projection(images)


class HPMAModel(nn.Module):
    """Image-only hierarchical prototype-memory adaptation model."""

    def __init__(
        self,
        sam3_checkpoint_path: Optional[str] = None,
        feature_dim: int = 256,
        num_classes: int = 1,
        num_prototypes: int = 4,
        input_channels: int = 3,
        visual_token_grid: Tuple[int, int] = (4, 4),
        prototype_bank: Optional[PrototypeMemoryBank] = None,
        unfreeze_vision_blocks: int = 4,
        unfreeze_fusion_layers: int = 2,
        use_mock_sam3: bool = False,
    ) -> None:
        super().__init__()
        if prototype_bank is None and not use_mock_sam3:
            raise ValueError(
                "A prebuilt prototype_bank is required for the production image-only model."
            )
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.input_adapter = InputAdapter(in_channels=input_channels)

        sam_model = build_sam3_image_model(
            checkpoint_path=sam3_checkpoint_path,
            feature_dim=feature_dim,
            use_mock=use_mock_sam3,
        )
        self.vision_encoder = sam_model.backbone
        self.fusion_encoder = getattr(sam_model, "fusion_encoder", None)
        if not hasattr(sam_model, "mask_decoder"):
            raise AttributeError("SAM model is missing the 'mask_decoder' sub-module.")
        self.transformer_decoder = sam_model.mask_decoder
        self.segmentation_head = getattr(
            sam_model,
            "segmentation_head",
            getattr(self.transformer_decoder, "conv_out", None),
        )

        # Keep only image-path modules from the full SAM container.
        del sam_model
        self._apply_selective_freezing(
            unfreeze_vision_blocks=unfreeze_vision_blocks,
            unfreeze_fusion_layers=unfreeze_fusion_layers,
        )

        self.prototype_bank = (
            prototype_bank
            if prototype_bank is not None
            else PrototypeMemoryBank(
                num_classes=num_classes,
                num_prototypes=num_prototypes,
                dim=feature_dim,
            )
        )
        expected_bank_config = (3, num_classes, num_prototypes, feature_dim)
        actual_bank_config = (
            self.prototype_bank.num_scales,
            self.prototype_bank.num_classes,
            self.prototype_bank.num_prototypes,
            self.prototype_bank.dim,
        )
        if actual_bank_config != expected_bank_config:
            raise ValueError(
                f"Prototype bank config {actual_bank_config} does not match "
                f"model config {expected_bank_config}."
            )
        self.visual_token_generator = VisualTokenGenerator(
            dim=feature_dim,
            token_grid=visual_token_grid,
        )
        self.global_visual_adapter = GlobalVisualAdapter(dim=feature_dim)
        self.structural_adapter = StructuralQueryAdapter(dim=feature_dim)
        self.local_adapter = LocalProjectionAdapter(dim=feature_dim)
        self.train(self.training)

    def _apply_selective_freezing(
        self,
        unfreeze_vision_blocks: int,
        unfreeze_fusion_layers: int,
    ) -> None:
        """Freeze SAM image modules, then unfreeze configured final blocks."""
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False
        if self.fusion_encoder is not None:
            for parameter in self.fusion_encoder.parameters():
                parameter.requires_grad = False
        if self.segmentation_head is not None:
            for parameter in self.segmentation_head.parameters():
                parameter.requires_grad = False

        trunk = getattr(self.vision_encoder, "trunk", None)
        blocks = getattr(trunk, "blocks", None)
        self._trainable_vision_blocks = []
        if blocks is not None and unfreeze_vision_blocks > 0:
            self._trainable_vision_blocks = list(blocks[-unfreeze_vision_blocks:])
            for block in self._trainable_vision_blocks:
                for parameter in block.parameters():
                    parameter.requires_grad = True

        fusion_layers = getattr(self.fusion_encoder, "layers", None)
        self._trainable_fusion_layers = []
        if fusion_layers is not None and unfreeze_fusion_layers > 0:
            self._trainable_fusion_layers = list(fusion_layers[-unfreeze_fusion_layers:])
            for layer in self._trainable_fusion_layers:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        for parameter in self.transformer_decoder.parameters():
            parameter.requires_grad = True
        if self.segmentation_head is not None:
            for parameter in self.segmentation_head.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "HPMAModel":
        """Keep frozen SAM modules in eval mode while training selected blocks."""
        super().train(mode)
        if mode:
            self.vision_encoder.eval()
            for block in self._trainable_vision_blocks:
                block.train()

            if self.fusion_encoder is not None:
                self.fusion_encoder.eval()
                for layer in self._trainable_fusion_layers:
                    layer.train()

            if self.segmentation_head is not None:
                self.segmentation_head.eval()
        return self

    @staticmethod
    def _extract_fpn_features(
        images: torch.Tensor,
        encoder: nn.Module,
    ) -> List[torch.Tensor]:
        return extract_image_fpn_features(encoder, images)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict a mask using image pixels and the model's frozen memory bank."""
        images = self.input_adapter(images)
        f_local, f_struct, f_global = self._extract_fpn_features(
            images,
            self.vision_encoder,
        )

        visual_tokens = self.visual_token_generator(f_global)
        visual_tokens = self.global_visual_adapter(
            visual_tokens,
            self.prototype_bank.global_all(),
        )
        image_features = [f_struct, f_global]
        if self.fusion_encoder is not None:
            fused_features = self.fusion_encoder(image_features, visual_tokens)
        else:
            fused_features = f_struct

        batch_size = images.shape[0]
        if hasattr(self.transformer_decoder, "get_object_queries"):
            queries = self.transformer_decoder.get_object_queries(batch_size=batch_size)
        else:
            queries = images.new_zeros(batch_size, 1, self.feature_dim)
        queries = self.structural_adapter(
            queries,
            f_struct,
            self.prototype_bank.structural_all(),
        )
        mask_queries = self.transformer_decoder(fused_features, queries)

        if self.segmentation_head is not None:
            pred_masks = self.segmentation_head(mask_queries, f_local)
        else:
            logits = f_local.mean(dim=1, keepdim=True)
            logits = F.interpolate(
                logits,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pred_masks = torch.sigmoid(logits)

        return {"pred_masks": pred_masks, "high_res_features": f_local}
