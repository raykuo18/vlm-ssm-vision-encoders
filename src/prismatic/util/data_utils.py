"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = self._stack_padded_tensors([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            tensors = [
                pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                for idx in range(len(input_ids))
            ]
            pixel_values = self._stack_padded_tensors(tensors)
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: self._stack_padded_tensors(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )

    def _stack_padded_tensors(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        """Pad tensors in H/W to the max size in the batch so they can be stacked."""
        if not tensors:
            return torch.empty(0)
        max_height = max(t.shape[-2] for t in tensors)
        max_width = max(t.shape[-1] for t in tensors)
        padded = [self._pad_tensor_to_size(t, max_height, max_width) for t in tensors]
        return torch.stack(padded)

    @staticmethod
    def _pad_tensor_to_size(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        _, height, width = tensor.shape
        pad_h = target_h - height
        pad_w = target_w - width
        if pad_h == 0 and pad_w == 0:
            return tensor
        return F.pad(tensor, (0, pad_w, 0, pad_h))
