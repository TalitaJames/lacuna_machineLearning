from lacunaBoard import *

if __name__ == "__main__":
    # Example usage of the LacunaBoard class
    tokens = new_random_lacuna_tokens(size=7, radius=radius)

    board = LacunaBoard(tokens)
    board.viewBoard()