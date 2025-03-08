import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from dataset import SudokuDataset
from model import DiscreteDiffusion
from sudoku_mrv import verify_board
from util import visualize_sudoku


def sample_board(model, generated_dir, total_step):
    model.eval()
    # sample
    with torch.no_grad():
        generated_boards = model.generate(batch_size=10)
        boards = generated_boards.chunk(generated_boards.shape[0], dim=0)
        boards = [b.squeeze(0).tolist() for b in boards]
        results = []
        board_figs = []
        for board in boards:
            results.append(verify_board(board, outer_grid_size=model.outer_grid_size))
            board_figs.append(visualize_sudoku(board, outer_grid_size=model.outer_grid_size))

        num_valid_boards = sum(results)
        num_boards = len(results)
        proportion_valid_boards = float(num_valid_boards) / num_boards

        logger.info(f"proportion valid boards: {proportion_valid_boards:.2f}")
        canvas_width = board_figs[0].width * len(boards)
        canvas_height = board_figs[0].height

        # create canvas
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        for i, board_fig in enumerate(board_figs):
            canvas.paste(board_fig, (board_fig.width * i, 0))
        canvas.save(generated_dir / f"generated_boards_{total_step}.png")
        canvas.close()

        # Clean up
        for fig in board_figs:
            fig.close()


def train(
    model: DiscreteDiffusion,
    outer_grid_size=9,
    num_epochs=10,
    batch_size=32,
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "mps",
    eval_every_n_step=100,
    warmup_steps=200,
    compiled=False,
):
    # Create dataset and dataloader
    logger.info("set up dataset and data loading")
    dataset = SudokuDataset(board_size=outer_grid_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    # Setup directory to save generated boards
    generated_dir = Path("generated")
    generated_dir.mkdir(parents=True, exist_ok=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.997), weight_decay=0.01)

    # warm up lr scheduler
    # Calculate total steps for the entire training
    total_steps = len(dataloader) * num_epochs

    # Create a learning rate scheduler with linear warmup and linear decay
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup phase
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Linear decay phase
            return max(
                0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Move model to device
    model = model.to(device)
    if compiled:
        model = torch.compile(model)

    total_step = 0

    # Training loop
    logger.info("start training")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, boards in enumerate(progress_bar):
            model.train()
            boards = boards.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds_bld, loss = model.forward_loss(boards)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

            total_step += 1
            if total_step % eval_every_n_step == 0:
                sample_board(model, generated_dir, total_step)

    return model


if __name__ == "__main__":
    # Example usage
    outer_grid_size = 9
    model = DiscreteDiffusion(outer_grid_size=outer_grid_size)
    trained_model = train(model, outer_grid_size=outer_grid_size)
