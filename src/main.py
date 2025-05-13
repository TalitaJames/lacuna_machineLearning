from lacunaBoard import *
from player import Player, HumanPlayer


def play_game(gameEnv, playerA, playerB):
    '''
    reset enviroment
    get state from gameEnv
    give state to player A & B

    # presumably player A & B want the observation
    # information before they take a first turn

    while game isn't done:
        (x,y) = playerA.take_turn()
        response = gameEnv.take_turn(x,y)
        playerA.turn_results(response)

        and for playerB
    '''
    pass


def train_models(episodesCount, playerA, playerB):
    '''
    args:
        - episodesCount: number of games to play
    '''

    '''
    for _ in range(episodesCount):
        gameEnv = new game enviroment
        play_game(gameEnv, playerA, playerB)


    # all training is done
    playerA.save(filepath)
    playerB.save(filepath)
    '''
    pass


if __name__ == "__main__":
    # Example usage of the LacunaBoard class
    radius = 1
    tokens = new_random_lacuna_tokens(flowerCount=7, radius=radius)

    board = LacunaBoard(tokens)
    board.view_board().show()
    board.view_board_with_voranoi().show()