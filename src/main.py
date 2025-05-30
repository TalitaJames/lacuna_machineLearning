import time
from lacunaBoard import *
from player import HumanPlayer, RandomPlayer
import matplotlib.pyplot as plt
from ppoModel import PPOAgent
from sac_model import SACAgent
import utils
import json
import argparse

from PIL import Image
import io


def play_game(gameEnv, playerA, playerB, viewGame=False, verbose=False, gifGameFilename = None):
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

    gameFrames = []

    while not gameEnv.is_game_finished():
        for i, player in enumerate(players):
            badAction = True
            while badAction:
                x, y = player.select_action()
                if gameEnv.is_valid_action(x, y):
                    badAction = False
                else:
                    if verbose:
                        print(f"Player {i} - {player} selected ({x:0.2f}, {y:0.2f}), which isn't valid")
                    player.receive_observation(observation, -50, False, {'status': 'invalid_action'})

            observation, reward, done, info = gameEnv.take_turn(x, y)
            player.receive_observation(observation, reward, done, info)
            total_rewards[i] += reward

            if verbose:
                print(f"Player {i} - {player} entered ({x:0.2f}, {x:0.2f}) for a reward of {reward:0.3f}")
                pass

            fig, ax = gameEnv.view_board()
            if viewGame:
                plt.show()

            # save the game frame as an image
            if gifGameFilename is not None:
                buf = io.BytesIO()  # Create an in-memory buffer
                fig.savefig(buf, format='png', bbox_inches='tight')  # Save the figure to the buffer
                buf.seek(0)  # Rewind the buffer to the beginning
                img = Image.open(buf)  # Open the buffer as a Pillow image
                plt.close(fig)  # Close the figure to free memory
                gameFrames.append(img)  # Append the image to the list

    winner = 1 if gameEnv.calculate_winner() else 0

    if verbose:
        print(f"Game finished! {gameEnv.calculate_winner()}, {playerA if gameEnv.calculate_winner() else playerB} wins!")
        print(f"Players have {gameEnv.userFlowers} flowers\n")

    # stich all images together into one GIF
    if gifGameFilename is not None:
        # append the last image a few times so it stays on screen longer
        for _ in range(5):
            gameFrames.append(gameFrames[-1])

        # Save the frames as a GIF
        gameFrames[0].save(
            f"{gifGameFilename}.gif",
            format="GIF",
            append_images=gameFrames[1:],
            save_all=True,
            duration=450, # time per frame [ms]
            loop=0, # loop forever
            optimize=True
        )

    return total_rewards, winner


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


def train_models(episodesCount, playerA, playerB, gameArgs = {"flowerCount": 7, "radius": 1}, viewGame=False, verbose=False, saveGame=True):
    ''' args:
        - episodesCount: number of games to play
        - player(A, B): two players to play against each other
        - viewGame: if True, will show the game board after each turn
    '''

    print(f"Training {playerA} vs {playerB} for {episodesCount} episodes with {gameArgs} and {viewGame=}")

    rewards_A = []
    rewards_B = []
    win_playerA = 0 # count of games won by player A

    for i in range(episodesCount):
        # generate new random game env, and have both competitors play
        tokens = new_random_lacuna_tokens(**gameArgs)
        #print(f"Game  {i} of {episodesCount}")
        gameEnv = LacunaBoard(tokens, **gameArgs)
        total_rewards, winner = play_game(gameEnv, playerA, playerB, viewGame, verbose)
        win_playerA += winner
        rewards_A.append(total_rewards[0])
        rewards_B.append(total_rewards[1])

        if i % 250 == 0 and i > 0: # periodicly update user
            print(f"Episode {i} of {episodesCount}")
            if i % 5_000 == 0 and saveGame: # and backup models
                utils.backup_models([playerA, playerB])

    # Save final trained models
    if saveGame:
        playerA.save(f"models/{playerA}")
        playerB.save(f"models/{playerB}")
    print(f"Training finished, {playerA} vs {playerB} for {episodesCount} episodes, Player A won {win_playerA/episodesCount:.2%}")

    plot_reward_history(playerA, playerB, rewards_A, rewards_B).show()


if __name__ == "__main__":
    # Init CLI arguments
    parser = argparse.ArgumentParser(description="Settings to change the running of AI lacuna code")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output")
    parser.add_argument('-s', '--show', action='store_true', help="See the output each turn")
    parser.add_argument('-l', '--load', action='store_true', help="Load existing models instead of training new ones")
    parser.add_argument('-e', '--episodes', type=int, help="Number of games to repeat", default=10_000)
    parser.add_argument('-r', '--record', type=bool,
                        help="If you don't want to save the models after training",
                        default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    #end command line arguments

    # Init the config and players
    sacKwargs = utils.load_config("config/sac.json")
    ppoKwargs = utils.load_config("config/ppo.json")

    ppoAgent = PPOAgent(**ppoKwargs)
    sacAgent = SACAgent(**sacKwargs)
    rndAgent = RandomPlayer()

    if args.load:
        print("Loading existing models...")
        ppoAgent.load(f"models/{ppoAgent}")
        sacAgent.load(f"models/{sacAgent}")

    for i in range(2):
        play_game(LacunaBoard(new_random_lacuna_tokens()), rndAgent, sacAgent, viewGame=False,
              verbose=args.verbose, gifGameFilename=f"out/game{rndAgent}_{sacAgent}_{i:03}")
    exit() # early exit for just exporting gifs

    start_time = time.time()
    train_models(args.episodes, sacAgent, sacAgent, viewGame=args.show,
                 verbose=args.verbose, saveGame=args.record)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {(execution_time)/60:.4f} mins")
