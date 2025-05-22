import time
from lacunaBoard import *
from player import HumanPlayer, RandomPlayer
from ppoModel import PPOAgent
from sac_model import SACAgent
import utils
import json



def play_game(gameEnv, playerA, playerB, viewGame=False, verbose=False):
    '''Play a game of Lacuna with the given players and environment.'''
    observation = gameEnv.get_observation()
    playerA.receive_observation(observation, 0, False, {})
    playerB.receive_observation(observation, 0, False, {})

    if viewGame:
        fig, ax = gameEnv.view_board()
        plt.show()

    players = [playerA, playerB]
    while not gameEnv.is_game_finished():
        for i, player in enumerate(players):
            x, y = player.select_action()
            observation, reward, done, info = gameEnv.take_turn(x, y)
            player.receive_observation(observation, reward, done, info)

            if verbose:
                print(f"Player {i} - {player} entered ({x:0.2f}, {x:0.2f}) for a reward of {reward:0.3f}")

            if viewGame:
                fig, ax = gameEnv.view_board()
                plt.show()

    if verbose:
        print(f"Game finished! {gameEnv.calculate_winner()}, {playerA if gameEnv.calculate_winner() else playerB} wins!")


def train_models(episodesCount, playerA, playerB, gameArgs = {"flowerCount": 7, "radius": 1}, viewGame=False, verbose=False):
    ''' args:
        - episodesCount: number of games to play
        - player(A, B): two players to play against each other
        - viewGame: if True, will show the game board after each turn
    '''

    print(f"Training {playerA} vs {playerB} for {episodesCount} episodes with {gameArgs} and {viewGame=}")
    for i in range(episodesCount):
        # generate new random game env, and have both competitors play
        tokens = new_random_lacuna_tokens(**gameArgs)
        gameEnv = LacunaBoard(tokens, **gameArgs)
        play_game(gameEnv, playerA, playerB, viewGame, verbose)

        if i % 5_000 == 0 and i > 0: # periodicly backup models
            print(f"Episode {i} of {episodesCount}")
            utils.backup_models([playerA, playerB], f"models/episode_{i}")

    #TODO properly save models



def evaluate_models():

    pass


if __name__ == "__main__":

    # Init the config and players
    sacParams = utils.load_config('config/sac.json')
    sacPlayerFoo = SACAgent(**sacParams)
    randomPlayer = RandomPlayer()
    ppoFoo = PPOAgent()
    ppoBaz = PPOAgent()

    print(f"Training {randomPlayer} vs {randomPlayer}")

    start_time = time.time()
    train_models(10_000, randomPlayer, randomPlayer)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {(execution_time)/60:.4f} mins")
