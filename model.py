import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from net import Transformer
from sampling import cosine_schedule, gumbel_sample, top_k


class DiscreteDiffusion(nn.Module):
    def __init__(
        self, outer_grid_size=9, head_dim=64, heads=8, depth=12, full_mask_token_prob=0.025
    ):
        super().__init__()
        # includes an additional MASK token
        num_classes = outer_grid_size + 1
        self.outer_grid_size = outer_grid_size
        self.model = Transformer(
            head_dim, heads, num_classes, depth, seq_len=outer_grid_size * outer_grid_size
        )
        self.full_mask_token_prob = full_mask_token_prob

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
        mask_id = 0
        b, h, w = board_bhw.shape
        board_bl = board_bhw.flatten(1)
        _, l = board_bl.shape

        rand_time = torch.rand(board_bhw.shape[0], device=board_bl.device)
        rand_mask_probs = cosine_schedule(rand_time)
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
        device = next(self.parameters()).device
        shape = (batch_size, self.outer_grid_size, self.outer_grid_size)
        board_bhw = torch.full(shape, 0, dtype=torch.long, device=device)
        scores_bhw = torch.zeros(shape, dtype=torch.float32, device=device)
        board_bl = board_bhw.flatten(1)
        scores_bl = scores_bhw.flatten(1)
        seq_len = board_bl.shape[1]

        starting_temperature = temperature
        mask_id = 0

        for timestep, steps_until_x0 in tqdm(
            zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))),
            total=timesteps,
        ):
            rand_mask_prob = cosine_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores_bl.topk(num_token_masked, dim=-1).indices

            board_bl = board_bl.scatter(1, masked_indices, mask_id)

            logits, _ = self.forward(board_bl)

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (
                steps_until_x0 / timesteps
            )  # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = board_bl == mask_id

            board_bl = torch.where(is_mask, pred_ids, board_bl)

            probs_without_temperature = logits.softmax(dim=-1)

            scores_bl = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
            scores_bl = rearrange(scores_bl, "... 1 -> ...")

            if not can_remask_prev_masked:
                scores_bl = scores_bl.masked_fill(~is_mask, -1e5)
            else:
                assert (
                    self.no_mask_token_prob > 0.0
                ), "without training with some of the non-masked tokens forced to predict, not sure if the logits will be meaningful for these token"

        board_bhw = board_bl.reshape(board_bhw.shape)

        return board_bhw
