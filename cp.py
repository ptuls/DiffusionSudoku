from ortools.sat.python import cp_model

def solve_cp(board):
    """
    Solve a Sudoku board using OR-Tools' CP-SAT solver.
    The input board is a 9x9 list of lists; 0 indicates empty.
    Returns a solved board (if a solution exists) or None.
    """
    model = cp_model.CpModel()
    # Create a 9x9 matrix of integer variables.
    cells = {}
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                cells[(i, j)] = model.NewIntVar(1, 9, f'cell_{i}_{j}')
            else:
                # For pre-filled cells, create a constant.
                cells[(i, j)] = board[i][j]
    
    # Add row constraints.
    for i in range(9):
        row_vars = [cells[(i, j)] for j in range(9)]
        model.AddAllDifferent(row_vars)
    
    # Add column constraints.
    for j in range(9):
        col_vars = [cells[(i, j)] for i in range(9)]
        model.AddAllDifferent(col_vars)
    
    # Add 3x3 subgrid constraints.
    for bi in range(3):
        for bj in range(3):
            block_vars = [cells[(i, j)] for i in range(bi*3, bi*3+3)
                                      for j in range(bj*3, bj*3+3)]
            model.AddAllDifferent(block_vars)
    
    # Solve the model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        solved = [[0 for _ in range(9)] for _ in range(9)]
        for i in range(9):
            for j in range(9):
                if isinstance(cells[(i, j)], int):
                    solved[i][j] = cells[(i, j)]
                else:
                    solved[i][j] = solver.Value(cells[(i, j)])
        return solved
    return None
