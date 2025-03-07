import torch
from torch.utils.data import Dataset

from sudoku_mrv import generate_board


# Custom dataset for generating Sudoku boards
class SudokuDataset(Dataset):
    def __init__(self, num_samples=10000, board_size=9):
        self.num_samples = num_samples
        self.board_size = board_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # For now, we'll use a simple approach to generate valid Sudoku boards
        # In a real implementation, you might want to use a more sophisticated generator
        # completeness = int(torch.rand(1).item() * 100)
        board = generate_board(completeness=100, outer_grid_size=self.board_size)
        board = torch.tensor(board)
        return board
