from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers.vision_utils import (
    get_vision_bilinear_indices_and_weights,
    get_vision_position_ids,
)


@dataclass
class MAEOutput:
    loss: torch.Tensor
    predictions: torch.Tensor
    mask: torch.Tensor


def random_patch_mask(
    lengths: list[int],
    mask_ratio: float,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create an exact-ratio random mask independently for every packed image."""
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")
    masks = []
    for length in lengths:
        if length < 2:
            raise ValueError("Each image must contain at least two patches")
        masked = min(length - 1, max(1, round(length * mask_ratio)))
        order = torch.rand(length, device=device, generator=generator).argsort()
        mask = torch.zeros(length, dtype=torch.bool, device=device)
        mask[order[:masked]] = True
        masks.append(mask)
    return torch.cat(masks)


class ChandraMAE(nn.Module):
    """Asymmetric MAE around the native Chandra 2/Qwen3.5 vision encoder.

    Masked tokens never enter the vision encoder. The decoder restores the full
    sequence with a learned mask token and predicts unnormalized RGB patches.
    Chandra's multimodal merger is retained for checkpoint compatibility but is
    neither called nor trained during MAE.
    """

    def __init__(
        self,
        vision: nn.Module,
        decoder_hidden_size: int = 384,
        decoder_layers: int = 4,
        decoder_heads: int = 6,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")
        if decoder_hidden_size % decoder_heads:
            raise ValueError("decoder_hidden_size must be divisible by decoder_heads")
        self.vision = vision
        self.mask_ratio = mask_ratio
        config = vision.config
        self.patch_dim = (
            int(config.in_channels)
            * int(config.temporal_patch_size)
            * int(config.patch_size) ** 2
        )
        self.encoder_to_decoder = nn.Linear(config.hidden_size, decoder_hidden_size)
        self.position_to_decoder = nn.Linear(
            config.hidden_size, decoder_hidden_size, bias=False
        )
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_hidden_size,
            nhead=decoder_heads,
            dim_feedforward=decoder_hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(decoder_hidden_size),
            enable_nested_tensor=False,
        )
        self.pixel_head = nn.Linear(decoder_hidden_size, self.patch_dim)
        nn.init.normal_(self.mask_token, std=0.02)
        self._reset_decoder_parameters()

        # The merger/projector is intentionally absent from SSL, per the plan.
        if hasattr(self.vision, "merger"):
            self.vision.merger.requires_grad_(False)

    def _reset_decoder_parameters(self) -> None:
        for module in (
            self.encoder_to_decoder,
            self.position_to_decoder,
            self.pixel_head,
        ):
            nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    @property
    def decoder_parameters(self):
        vision_ids = {id(parameter) for parameter in self.vision.parameters()}
        return (
            parameter
            for parameter in self.parameters()
            if id(parameter) not in vision_ids
        )

    def _random_mask(self, lengths: list[int], device: torch.device) -> torch.Tensor:
        return random_patch_mask(lengths, self.mask_ratio, device=device)

    def _position_embeddings(
        self, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.vision.num_grid_per_side,
            spatial_merge_size=self.vision.config.spatial_merge_size,
            kwargs={},
        )
        learned = (
            self.vision.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]
        ).sum(0)
        position_ids = get_vision_position_ids(
            grid_thw, self.vision.spatial_merge_size, kwargs={}
        )
        rotary = self.vision.rotary_pos_emb(position_ids)
        rotary = rotary.reshape(position_ids.shape[0], -1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        return learned, (rotary.cos(), rotary.sin())

    def _encode_visible(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        mask: torch.Tensor,
        lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.vision.patch_embed(pixel_values)
        learned_position, rotary_position = self._position_embeddings(grid_thw)
        hidden = hidden + learned_position.to(hidden.dtype)
        visible = ~mask
        hidden = hidden[visible]
        cos, sin = rotary_position
        position_embeddings = (cos[visible], sin[visible])
        visible_lengths = torch.tensor(
            [int((~part).sum()) for part in mask.split(lengths)],
            device=hidden.device,
            dtype=torch.int32,
        )
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, device=hidden.device, dtype=torch.int32),
                visible_lengths.cumsum(0),
            ]
        )
        for block in self.vision.blocks:
            hidden = block(
                hidden,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        return hidden, learned_position

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        target_patches: torch.Tensor,
    ) -> MAEOutput:
        lengths = [int(value) for value in grid_thw.prod(dim=1).tolist()]
        if sum(lengths) != pixel_values.shape[0]:
            raise ValueError("grid_thw and pixel_values contain different patch counts")
        mask = self._random_mask(lengths, pixel_values.device)
        encoded, encoder_positions = self._encode_visible(
            pixel_values, grid_thw, mask, lengths
        )

        decoded_sequences = []
        encoded_offset = 0
        token_offset = 0
        for length in lengths:
            sample_mask = mask[token_offset : token_offset + length]
            visible_count = int((~sample_mask).sum())
            visible_tokens = self.encoder_to_decoder(
                encoded[encoded_offset : encoded_offset + visible_count]
            )
            restored = (
                self.mask_token.to(visible_tokens.dtype).expand(length, -1).clone()
            )
            restored = restored.index_copy(
                0, (~sample_mask).nonzero(as_tuple=False).flatten(), visible_tokens
            )
            restored = restored + self.position_to_decoder(
                encoder_positions[token_offset : token_offset + length].to(
                    restored.dtype
                )
            )
            decoded_sequences.append(restored)
            encoded_offset += visible_count
            token_offset += length

        padded = pad_sequence(decoded_sequences, batch_first=True)
        max_length = padded.shape[1]
        padding_mask = (
            torch.arange(max_length, device=padded.device)[None, :]
            >= torch.tensor(lengths, device=padded.device)[:, None]
        )
        decoded = self.decoder(padded, src_key_padding_mask=padding_mask)
        predictions = torch.cat(
            [
                self.pixel_head(decoded[index, :length])
                for index, length in enumerate(lengths)
            ]
        )
        target_patches = target_patches.to(predictions.dtype)
        loss = (predictions[mask] - target_patches[mask]).square().mean()
        return MAEOutput(loss=loss, predictions=predictions, mask=mask)
