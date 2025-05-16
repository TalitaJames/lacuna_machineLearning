import time
from lacunaBoard import *
from player import Player, HumanPlayer


def play_game(gameEnv, playerA, playerB, viewGame=False):
    '''Play a game of Lacuna with the given players and environment.'''
    observation = gameEnv.get_observation()
    playerA.receive_observation(observation, 0, False, {})
    playerB.receive_observation(observation, 0, False, {})

    if viewGame:
        gameEnv.view_board().show()

    players = [playerA, playerB]
    while not gameEnv.is_game_finished():
        for player in players:
            print(f"\nPlayer {player.getName()}'s turn")
            x, y = player.select_action()
            print(f"\t Entered ({x}, {y})")
            gameState = gameEnv.take_turn(x, y)
            player.receive_observation(*gameState)

            if viewGame:
                gameEnv.view_board().show()


def train_models(episodesCount, playerA, playerB, viewGame=False):
    ''' args:
        - episodesCount: number of games to play
        - player(A, B): two players to play against each other
        - viewGame: if True, will show the game board after each turn
    '''
    gameArgs = {"flowerCount": 7, "radius": 1}

    print(f"Training {playerA.getName()} vs {playerB.getName()} for {episodesCount} episodes with {gameArgs} and {viewGame=}")
    for _ in range(episodesCount):
        tokens = new_random_lacuna_tokens(**gameArgs)
        gameEnv = LacunaBoard(tokens, **gameArgs)
        play_game(gameEnv, playerA, playerB, viewGame)

    # all training is done
    timestamp = time.strftime("%Y%m%d-%H%M%S",time.localtime())
    playerA.save(f"out/{timestamp}_{playerA.getName()}_A")
    playerB.save(f"out/{timestamp}_{playerA.getName()}_B")


if __name__ == "__main__":
    # Example usage of the LacunaBoard class

    humanFoo = HumanPlayer()
    humanBaz = HumanPlayer()

    train_models(1, humanFoo, humanBaz, viewGame=True)