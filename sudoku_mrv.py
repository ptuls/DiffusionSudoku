import random
import copy

def find_empty(board, length=9):
    """Find an empty cell (with 0) in the board."""
    for i in range(length):
        for j in range(length):
            if board[i][j] == 0:
                return i, j
    return None

def is_valid(board, row, col, num, outer_grid_size=9):
    """Check if it's valid to place num at board[row][col]."""
    # Check row
    if num in board[row]:
        return False
    # Check column
    for i in range(outer_grid_size):
        if board[i][col] == num:
            return False
    # Check the 3x3 subgrid
    inner_grid_size = int(outer_grid_size ** 0.5)
    start_row, start_col = (row // inner_grid_size) * inner_grid_size, (col // inner_grid_size) * inner_grid_size
    for i in range(start_row, start_row + inner_grid_size):
        for j in range(start_col, start_col + inner_grid_size):
            if board[i][j] == num:
                return False
    return True

def get_possible_numbers(board, row, col, outer_grid_size=9):
    """Return the set of valid numbers for the cell at (row, col)."""
    possibilities = set(range(1, outer_grid_size + 1))
    # Remove numbers from the row and column
    possibilities -= set(board[row])
    possibilities -= {board[i][col] for i in range(outer_grid_size)}
    # Remove numbers from the 3x3 subgrid
    inner_grid_size = int(outer_grid_size ** 0.5)
    start_row, start_col = (row // inner_grid_size) * inner_grid_size, (col // inner_grid_size) * inner_grid_size
    for i in range(start_row, start_row + inner_grid_size):
        for j in range(start_col, start_col + inner_grid_size):
            possibilities.discard(board[i][j])
    return possibilities

def find_empty_mrv(board, outer_grid_size=9):
    """
    Find the empty cell with the fewest legal candidates.
    Returns (row, col, candidates) or None if board is full.
    """
    best = None
    min_options = outer_grid_size+1  # More than maximum candidates (9)
    for i in range(outer_grid_size):
        for j in range(outer_grid_size):
            if board[i][j] == 0:
                candidates = get_possible_numbers(board, i, j, outer_grid_size)
                if len(candidates) < min_options:
                    min_options = len(candidates)
                    best = (i, j, candidates)
                    # If we find a cell with 0 possibilities, we can stop early.
                    if min_options == 0:
                        return best
    return best

# def solve_mrv(board, filled, target, outer_grid_size):
#     """
#     Backtracking solver using the MRV heuristic.
#     Stops after filling 'target' cells.
#     """
#     if filled >= target:
#         return True
    
#     cell = find_empty_mrv(board, outer_grid_size)
#     if cell is None:
#         return True  # Board is full (target should be 81 in this case)
#     row, col, candidates = cell

#     # Convert candidates to a list and shuffle them to add randomness.
#     candidate_list = list(candidates)
#     random.shuffle(candidate_list)
    
#     for num in candidate_list:
#         if is_valid(board, row, col, num, outer_grid_size):
#             board[row][col] = num
#             if solve_mrv(board, filled + 1, target, outer_grid_size):
#                 return True
#             board[row][col] = 0  # backtrack

#     return False


def solve_mrv(board, filled, target, outer_grid_size, start_time=None, timeout=2):
    """
    Backtracking solver using the MRV heuristic.
    Stops after filling 'target' cells or if timeout is reached.
    
    Args:
        start_time: Time when solving started (set automatically on first call)
        timeout: Maximum seconds to attempt solving before giving up
    """
    import time
    if start_time is None:
        start_time = time.time()
    elif time.time() - start_time > timeout:
        return False  # Timeout reached
        
    if filled >= target:
        return True
    
    cell = find_empty_mrv(board, outer_grid_size)
    if cell is None:
        return True
    row, col, candidates = cell
    
    # If no candidates available, fail fast
    if not candidates:
        return False

    # Convert candidates to a list and shuffle them to add randomness
    candidate_list = list(candidates)
    random.shuffle(candidate_list)
    
    for num in candidate_list:
        if is_valid(board, row, col, num, outer_grid_size):
            board[row][col] = num
            if solve_mrv(board, filled + 1, target, outer_grid_size, start_time, timeout):
                return True
            board[row][col] = 0  # backtrack

    return False


def plant_random_seeds(board, seed_count, outer_grid_size=9):
    """
    Plant a given number of random seed values into the board.
    For each seed, a random empty cell is chosen and a valid number is randomly placed.
    Returns the number of seeds successfully planted.
    """
    seeds_planted = 0
    attempts = 0
    max_attempts = seed_count * 10  # Prevent an infinite loop if board fills up.
    
    while seeds_planted < seed_count and attempts < max_attempts:
        empty_cell = find_empty(board, outer_grid_size)
        if empty_cell is None:
            break
        row, col = empty_cell
        candidates = list(get_possible_numbers(board, row, col, outer_grid_size))
        if candidates:
            chosen = random.choice(candidates)
            board[row][col] = chosen
            seeds_planted += 1
        attempts += 1
    return seeds_planted


def generate_board(completeness=100, seed_count=0, num_retries=100, outer_grid_size=9):
    """
    Generate a Sudoku board that is partially filled based on the completeness percentage.
    Will retry with new random seeds if solving takes too long or fails.
    """
    target = int(outer_grid_size * outer_grid_size * completeness / 100)
    target = max(1, target)
    
    for attempt in range(num_retries):
        board = [[0 for _ in range(outer_grid_size)] for _ in range(outer_grid_size)]
        seeds_planted = plant_random_seeds(board, seed_count, outer_grid_size)
        if solve_mrv(board, filled=seeds_planted, target=target, outer_grid_size=outer_grid_size):
            return board
    raise Exception(f"Failed to generate a board with {completeness}% completeness after {num_retries} attempts.")


def verify_board(board, outer_grid_size=9):
    """
    Verify that the provided complete board is correct.
    Each row, column, and 3x3 subgrid must contain all numbers from 1 to 9 exactly once.
    Returns True if the board is valid, False otherwise.
    """
    # Check rows
    for i in range(outer_grid_size):
        row = board[i]
        if sorted(row) != list(range(1, outer_grid_size + 1)):
            return False
    # Check columns
    for j in range(outer_grid_size):
        col = [board[i][j] for i in range(outer_grid_size)]
        if sorted(col) != list(range(1, outer_grid_size + 1)):
            return False
    # Check 3x3 subgrids
    inner_grid_size = int(outer_grid_size ** 0.5)
    for start_row in range(0, outer_grid_size, inner_grid_size):
        for start_col in range(0, outer_grid_size, inner_grid_size):
            block = []
            for i in range(start_row, start_row + inner_grid_size):
                for j in range(start_col, start_col + inner_grid_size):
                    block.append(board[i][j])
            if sorted(block) != list(range(1, outer_grid_size + 1)):
                return False
    return True

def print_board(board, outer_grid_size=9):
    """Nicely print the Sudoku board."""
    inner_grid_size = int(outer_grid_size ** 0.5)
    for i in range(outer_grid_size):
        line = ""
        for j in range(outer_grid_size):
            if j % inner_grid_size == 0 and j != 0:
                line += " | "
            cell = str(board[i][j]) if board[i][j] != 0 else "."
            line += cell + " "
        print(line)
        if (i + 1) % inner_grid_size == 0 and i != outer_grid_size - 1:
            print("-" * (outer_grid_size * 2 - 1))

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
