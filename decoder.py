import torch
from einops import rearrange
from tqdm import tqdm

from util import cosine_schedule, gumbel_sample, top_k


class MaskGITDecoder:
    def __init__(
        self,
        outer_grid_size,
        mask_id=0,
        noise_schedule=cosine_schedule,
    ):
        self.outer_grid_size = outer_grid_size
        self.mask_id = mask_id
        self.noise_schedule = noise_schedule

    def generate(
        self,
        model,
        batch_size=32,
        timesteps=128,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,
        no_mask_token_prob=0.0,
    ):
        device = next(model.parameters()).device
        shape = (batch_size, self.outer_grid_size, self.outer_grid_size)
        board_bhw = torch.full(shape, self.mask_id, dtype=torch.long, device=device)
        scores_bhw = torch.zeros(shape, dtype=torch.float32, device=device)
        board_bl = board_bhw.flatten(1)
        scores_bl = scores_bhw.flatten(1)
        seq_len = board_bl.shape[1]

        starting_temperature = temperature

        for timestep, steps_until_x0 in tqdm(
            zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))),
            total=timesteps,
        ):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores_bl.topk(num_token_masked, dim=-1).indices

            board_bl = board_bl.scatter(1, masked_indices, self.mask_id)

            logits, _ = model(board_bl)

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (
                steps_until_x0 / timesteps
            )  # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = board_bl == self.mask_id

            board_bl = torch.where(is_mask, pred_ids, board_bl)

            probs_without_temperature = logits.softmax(dim=-1)

            scores_bl = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
            scores_bl = rearrange(scores_bl, "... 1 -> ...")

            if not can_remask_prev_masked:
                scores_bl = scores_bl.masked_fill(~is_mask, -1e5)
            else:
                assert no_mask_token_prob > 0.0, (
                    "without training with some of the non-masked tokens forced to predict, "
                    "not sure if the logits will be meaningful for these token"
                )

        board_bhw = board_bl.reshape(board_bhw.shape)

        return board_bhw
