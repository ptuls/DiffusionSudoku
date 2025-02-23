import random
import copy

def find_empty(board):
    """Find an empty cell (with 0) in the board."""
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return None

def is_valid(board, row, col, num):
    """Check if it's valid to place num at board[row][col]."""
    # Check row
    if num in board[row]:
        return False
    # Check column
    for i in range(9):
        if board[i][col] == num:
            return False
    # Check the 3x3 subgrid
    start_row, start_col = (row // 3) * 3, (col // 3) * 3
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False
    return True

def get_possible_numbers(board, row, col):
    """Return the set of valid numbers for the cell at (row, col)."""
    possibilities = set(range(1, 10))
    # Remove numbers from the row and column
    possibilities -= set(board[row])
    possibilities -= {board[i][col] for i in range(9)}
    # Remove numbers from the 3x3 subgrid
    start_row, start_col = (row // 3) * 3, (col // 3) * 3
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            possibilities.discard(board[i][j])
    return possibilities

def find_empty_mrv(board):
    """
    Find the empty cell with the fewest legal candidates.
    Returns (row, col, candidates) or None if board is full.
    """
    best = None
    min_options = 10  # More than maximum candidates (9)
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                candidates = get_possible_numbers(board, i, j)
                if len(candidates) < min_options:
                    min_options = len(candidates)
                    best = (i, j, candidates)
                    # If we find a cell with 0 possibilities, we can stop early.
                    if min_options == 0:
                        return best
    return best

def solve_mrv(board, filled, target):
    """
    Backtracking solver using the MRV heuristic.
    Stops after filling 'target' cells.
    """
    if filled >= target:
        return True
    
    cell = find_empty_mrv(board)
    if cell is None:
        return True  # Board is full (target should be 81 in this case)
    row, col, candidates = cell

    # Convert candidates to a list and shuffle them to add randomness.
    candidate_list = list(candidates)
    random.shuffle(candidate_list)
    
    for num in candidate_list:
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_mrv(board, filled + 1, target):
                return True
            board[row][col] = 0  # backtrack

    return False

def plant_random_seeds(board, seed_count):
    """
    Plant a given number of random seed values into the board.
    For each seed, a random empty cell is chosen and a valid number is randomly placed.
    Returns the number of seeds successfully planted.
    """
    seeds_planted = 0
    attempts = 0
    max_attempts = seed_count * 10  # Prevent an infinite loop if board fills up.
    
    while seeds_planted < seed_count and attempts < max_attempts:
        empty_cell = find_empty(board)
        if empty_cell is None:
            break
        row, col = empty_cell
        candidates = list(get_possible_numbers(board, row, col))
        if candidates:
            chosen = random.choice(candidates)
            board[row][col] = chosen
            seeds_planted += 1
        attempts += 1
    return seeds_planted

def generate_board(completeness=100, seed_count=0, num_retries=100):
    """
    Generate a Sudoku board that is partially filled based on the completeness percentage.
    completeness: percentage of cells to fill (1-100). For example, 20 means 20% filled.
    seed_count: number of random seed values to plant before solving.
    When completeness is 100, the board is fully solved.
    """
    target = int(81 * completeness / 100)
    target = max(1, target)
    
    for _ in range(num_retries):
        board = [[0 for _ in range(9)] for _ in range(9)]
        # Plant random seeds to add more variance.
        seeds_planted = plant_random_seeds(board, seed_count)
        # Start the solver with the seeds already placed.
        if solve_mrv(board, filled=seeds_planted, target=target):
            return board
    raise Exception("Failed to generate a board with the specified completeness.")

def verify_board(board):
    """
    Verify that the provided complete board is correct.
    Each row, column, and 3x3 subgrid must contain all numbers from 1 to 9 exactly once.
    Returns True if the board is valid, False otherwise.
    """
    # Check rows
    for i in range(9):
        row = board[i]
        if sorted(row) != list(range(1, 10)):
            return False
    # Check columns
    for j in range(9):
        col = [board[i][j] for i in range(9)]
        if sorted(col) != list(range(1, 10)):
            return False
    # Check 3x3 subgrids
    for start_row in range(0, 9, 3):
        for start_col in range(0, 9, 3):
            block = []
            for i in range(start_row, start_row + 3):
                for j in range(start_col, start_col + 3):
                    block.append(board[i][j])
            if sorted(block) != list(range(1, 10)):
                return False
    return True

def print_board(board):
    """Nicely print the Sudoku board."""
    for i in range(9):
        line = ""
        for j in range(9):
            if j % 3 == 0 and j != 0:
                line += " | "
            cell = str(board[i][j]) if board[i][j] != 0 else "."
            line += cell + " "
        print(line)
        if (i + 1) % 3 == 0 and i != 8:
            print("-" * 21)

if __name__ == "__main__":
    # Example: Generate a board with 20% completeness and 3 random seeds
    completeness_percentage = 20
    seed_count = 3
    board = generate_board(completeness=completeness_percentage, seed_count=seed_count)
    print("Generated Board ({}% filled, {} seeds):".format(completeness_percentage, seed_count))
    print_board(board)

    # For a fully solved board:
    complete_board = generate_board(completeness=100, seed_count=3)
    print("\nComplete Board:")
    print_board(complete_board)
    if verify_board(complete_board):
        print("\nThe complete board is verified as correct.")
    else:
        print("\nThe complete board is NOT valid!")
