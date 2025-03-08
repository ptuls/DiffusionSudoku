import torch
import torch.nn as nn
from einops import rearrange

from net import Transformer
from sampling import cosine_schedule
from decoder import MaskGITDecoder


class DiscreteDiffusion(nn.Module):
    def __init__(
        self,
        outer_grid_size=9,
        head_dim=64,
        heads=8,
        depth=12,
        full_mask_token_prob=0.025,
        noise_schedule=cosine_schedule,
    ):
        super().__init__()
        # includes an additional MASK token
        num_classes = outer_grid_size + 1
        self.outer_grid_size = outer_grid_size
        self.model = Transformer(
            head_dim, heads, num_classes, depth, seq_len=outer_grid_size * outer_grid_size
        )
        self.full_mask_token_prob = full_mask_token_prob
        self.noise_schedule = noise_schedule
        self.mask_id = 0
        
        # Initialize the MaskGIT decoder
        self.decoder = MaskGITDecoder(
            model=self,
            outer_grid_size=outer_grid_size,
            mask_id=self.mask_id,
            noise_schedule=noise_schedule,
        )

    def forward(self, board_bl, labels=None):
        """
        forward and compute loss
        """
        preds_bld = self.model(board_bl)
        if labels is not None:
            loss = nn.functional.cross_entropy(
                preds_bld.reshape(-1, preds_bld.shape[-1]), labels.flatten(0), ignore_index=-1
            )
        else:
            loss = 0

        return preds_bld, loss

    def forward_loss(self, board_bhw, ignore_index=-1):
        mask_id = self.mask_id
        b, h, w = board_bhw.shape
        board_bl = board_bhw.flatten(1)
        _, l = board_bl.shape

        rand_time = torch.rand(board_bhw.shape[0], device=board_bl.device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (l * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((b, l), device=board_bhw.device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")

        labels = torch.where(mask, board_bl, ignore_index)

        if self.full_mask_token_prob > 0.0:
            full_mask_mask = torch.full_like(mask, True)
            indices = torch.arange(b, device=mask.device)
            indices_mask = torch.bernoulli(
                torch.full((b,), self.full_mask_token_prob, device=mask.device)
            ).long()
            indices = indices[indices_mask]
            mask[indices] = full_mask_mask[indices]

        board_bl = torch.where(mask, mask_id, board_bl)
        preds_bld, loss = self.forward(board_bl, labels=labels)

        return preds_bld, loss

    def generate(
        self,
        batch_size=32,
        timesteps=128,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,
    ):
        return self.decoder.generate(
            batch_size=batch_size,
            timesteps=timesteps,
            temperature=temperature,
            topk_filter_thres=topk_filter_thres,
            can_remask_prev_masked=can_remask_prev_masked,
            no_mask_token_prob=self.full_mask_token_prob,
        )
