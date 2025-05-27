import time
from lacunaBoard import *
from player import HumanPlayer, RandomPlayer
import matplotlib.pyplot as plt
from ppoModel import PPOAgent
from sac_model import SACAgent
import utils
import json



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
    total_rewards = [0.0, 0.0] 
    
    while not gameEnv.is_game_finished():
        for i, player in enumerate(players):
            #print("Lets select an ation")
            x, y = player.select_action()
            #print(f"the selected action is x:{x} and y:{y} what a very poor decision")
            observation, reward, done, info = gameEnv.take_turn(x, y)
            player.receive_observation(observation, reward, done, info)
            total_rewards[i] += reward

            if verbose:
                print(f"Player {i} - {player} entered ({x:0.2f}, {y:0.2f}) for a reward of {reward:0.3f}")
                pass

            if viewGame:
                fig, ax = gameEnv.view_board()
                plt.show()

    if verbose:
        print(f"Game finished! {gameEnv.calculate_winner()}, {playerA if gameEnv.calculate_winner() else playerB} wins!")

    return total_rewards

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
        tokens = new_random_lacuna_tokens()
        #print(f"Game  {i} of {episodesCount}")
        gameEnv = LacunaBoard(tokens, **gameArgs)
        total_rewards = play_game(gameEnv, playerA, playerB, viewGame, verbose)

        rewards_A.append(total_rewards[0])
        rewards_B.append(total_rewards[1])

        if i % 1_000 == 0 and i > 0: # periodicly backup models
            print(f"Episode {i} of {episodesCount}")
            #utils.backup_models([playerA, playerB], f"models/episode_{i}")

    #TODO properly save models

    # Plot rewards at the end of training
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_A, label=f'{playerA} 1 Rewards')
    plt.plot(rewards_B, label=f'{playerB} 0 Rewards')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def evaluate_models():

    pass


if __name__ == "__main__":
    # Init the config and players
    with open("config/ppo.json", "r") as f:
        ppoKwargs = json.load(f)

    with open("config/sac.json", "r") as f:
        sacKwargs = json.load(f)

    ppoFoo = PPOAgent(**ppoKwargs)
    ppoBaz = SACAgent(**sacKwargs)

    print(f"Training ppoFoo vs ppoBaz")

    start_time = time.time()
    train_models(3_000, ppoBaz, ppoFoo, viewGame=False, verbose=False)
    end_time = time.time()


    execution_time = end_time - start_time
    print(f"Execution time: {(execution_time)/60:.4f} mins")
