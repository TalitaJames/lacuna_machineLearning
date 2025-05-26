import time
from lacunaBoard import *
from player import HumanPlayer, RandomPlayer
import matplotlib.pyplot as plt
from ppoModel import PPOAgent
from sac_model import SACAgent
import utils
import json
import argparse



def play_game(gameEnv, playerA, playerB, viewGame=False, verbose=False):
    '''Play a game of Lacuna with the given players and environment.'''
    observation = gameEnv.get_observation()
    #print(f"the observation recived is: {observation}")
    #print(f"the observation size is:{observation.size}")
    playerA.receive_observation(observation, 0, False, {})
    playerB.receive_observation(observation, 0, False, {})

    if viewGame:
        fig, ax = gameEnv.view_board()
        plt.show()

    players = [playerA, playerB]
    total_rewards = [0.0, 0.0] # player A, B total rewards

    while not gameEnv.is_game_finished():
        for i, player in enumerate(players):
            x, y = player.select_action()
            observation, reward, done, info = gameEnv.take_turn(x, y)
            player.receive_observation(observation, reward, done, info)
            total_rewards[i] += reward

            if verbose:
                print(f"Player {i} - {player} entered ({x:0.2f}, {x:0.2f}) for a reward of {reward:0.3f}")
                pass

            if viewGame:
                fig, ax = gameEnv.view_board()
                plt.show()

    if verbose:
        print(f"Game finished! {gameEnv.calculate_winner()}, {playerA if gameEnv.calculate_winner() else playerB} wins!")

    return total_rewards


def plot_reward_history(playerA, playerB, rewards_A, rewards_B):
    # Plot rewards at the end of training
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_A, label=f'{playerA} A')
    plt.plot(rewards_B, label=f'{playerB} B')

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt


def train_models(episodesCount, playerA, playerB, gameArgs = {"flowerCount": 7, "radius": 1}, viewGame=False, verbose=False):
    ''' args:
        - episodesCount: number of games to play
        - player(A, B): two players to play against each other
        - viewGame: if True, will show the game board after each turn
    '''

    print(f"Training {playerA} vs {playerB} for {episodesCount} episodes with {gameArgs} and {viewGame=}")

    rewards_A = []
    rewards_B = []

    for i in range(episodesCount):
        # generate new random game env, and have both competitors play
        tokens = new_random_lacuna_tokens(**gameArgs)
        #print(f"Game  {i} of {episodesCount}")
        gameEnv = LacunaBoard(tokens, **gameArgs)
        total_rewards = play_game(gameEnv, playerA, playerB, viewGame, verbose)

        rewards_A.append(total_rewards[0])
        rewards_B.append(total_rewards[1])

        if i % 5_000 == 0 and i > 0: # periodicly backup models
            print(f"Episode {i} of {episodesCount}")
            utils.backup_models([playerA, playerB], f"models/episode_{i}")

    # Save final trained models
    playerA.save(f"models/{playerA}")
    playerB.save(f"models/{playerB}")

    plot_reward_history(playerA, playerB, rewards_A, rewards_B).show()


if __name__ == "__main__":
    # Init CLI arguments
    parser = argparse.ArgumentParser(description="Settings to change the running of AI lacuna code")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output")
    parser.add_argument('-s', '--show', action='store_true', help="See the output each turn")
    parser.add_argument('-e', '--episodes', type=int, help="Number of games to repeat", default=10_000)
    args = parser.parse_args()
    print(f"{args=}")
    #end command line arguments

    # Init the config and players
    sacKwargs = utils.load_config("config/sac.json")
    ppoKwargs = utils.load_config("config/ppo.json")

    ppoFoo = PPOAgent(**ppoKwargs)

    print(f"Training ppoFoo vs ppoBaz")

    start_time = time.time()
    train_models(args.episodes, rndAgent, sacFoo, viewGame=args.show, verbose=args.verbose)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {(execution_time)/60:.4f} mins")
