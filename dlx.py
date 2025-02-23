class DLXNode:
    def __init__(self):
        self.left = self.right = self.up = self.down = self
        self.column = None
        self.row_id = -1
        self.col_id = -1

class ColumnNode(DLXNode):
    def __init__(self, name):
        super().__init__()
        self.size = 0
        self.name = name

def build_exact_cover_board():
    """
    Build the exact cover matrix for Sudoku.
    There are 9*9*9 = 729 rows (each candidate assignment)
    and 9*9*4 = 324 columns (constraints: cell, row, col, box).
    Returns a tuple (header, nodes) where header is the root of the DLX structure.
    """
    header = ColumnNode("header")
    columns = []
    for i in range(324):
        col = ColumnNode(i)
        columns.append(col)
        # Link columns in a row (circularly)
        col.right = header
        col.left = header.left
        header.left.right = col
        header.left = col

    # For each candidate row (cell, value)
    nodes = []
    for r in range(9):
        for c in range(9):
            for n in range(1, 10):
                row_id = (r, c, n)
                # Calculate column indices for the 4 constraints:
                # 1. Cell constraint: each cell gets one number.
                col1 = r * 9 + c
                # 2. Row constraint: each row gets each number exactly once.
                col2 = 81 + r * 9 + (n - 1)
                # 3. Column constraint: each column gets each number exactly once.
                col3 = 162 + c * 9 + (n - 1)
                # 4. Box constraint: each 3x3 box gets each number exactly once.
                box = (r // 3) * 3 + (c // 3)
                col4 = 243 + box * 9 + (n - 1)
                cols = [col1, col2, col3, col4]
                row_nodes = []
                for col_index in cols:
                    node = DLXNode()
                    node.row_id = row_id
                    node.col_id = col_index
                    node.column = columns[col_index]
                    row_nodes.append(node)
                    # Insert node into the bottom of its column.
                    node.down = columns[col_index]
                    node.up = columns[col_index].up
                    columns[col_index].up.down = node
                    columns[col_index].up = node
                    columns[col_index].size += 1
                # Link row_nodes together horizontally.
                for i in range(4):
                    row_nodes[i].right = row_nodes[(i + 1) % 4]
                    row_nodes[i].left = row_nodes[(i - 1) % 4]
                nodes.append(row_nodes)
    return header, nodes

def cover(column):
    column.right.left = column.left
    column.left.right = column.right
    i = column.down
    while i != column:
        j = i.right
        while j != i:
            j.down.up = j.up
            j.up.down = j.down
            j.column.size -= 1
            j = j.right
        i = i.down

def uncover(column):
    i = column.up
    while i != column:
        j = i.left
        while j != i:
            j.column.size += 1
            j.down.up = j
            j.up.down = j
            j = j.left
        i = i.up
    column.right.left = column
    column.left.right = column

def search(header, solution, results):
    if header.right == header:
        # Found a complete solution.
        results.append(solution.copy())
        return True  # Optionally, return after first solution.
    # Choose the column with the smallest size.
    c = header.right
    min_size = c.size
    col = c
    while c != header:
        if c.size < min_size:
            min_size = c.size
            col = c
        c = c.right
    cover(col)
    r = col.down
    while r != col:
        solution.append(r)
        j = r.right
        while j != r:
            cover(j.column)
            j = j.right
        if search(header, solution, results):
            return True
        # Backtrack:
        solution.pop()
        j = r.left
        while j != r:
            uncover(j.column)
            j = j.left
        r = r.down
    uncover(col)
    return False

def solve_dlx(board):
    """
    Solve a Sudoku board using Dancing Links (Algorithm X).
    The input board is a 9x9 list of lists; 0 indicates empty.
    This function returns a solved board (if a solution exists) or None.
    """
    header, nodes = build_exact_cover_board()
    # Pre-cover constraints for prefilled cells.
    for r in range(9):
        for c in range(9):
            n = board[r][c]
            if n != 0:
                # Find the row in the DLX matrix corresponding to (r, c, n)
                # (Since rows in nodes list are in order, we can calculate the index)
                index = (r * 9 + c) * 9 + (n - 1)
                row_nodes = nodes[index]
                # Cover all columns for this candidate.
                for node in row_nodes:
                    cover(node.column)
    solution = []
    results = []
    search(header, solution, results)
    if not results:
        return None
    # Build solved board from the solution.
    solved = [[0]*9 for _ in range(9)]
    for node in results[0]:
        r, c, n = node.row_id
        solved[r][c] = n
    return solved
