from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np


''' An abstract base class to ensure all agents have required playing models '''
class Player(ABC):
    @abstractmethod
    def select_action(self):
        '''decide and return an (x,y) action'''
        pass

    @abstractmethod
    def receive_observation(self, observation, reward, done, info):
        ''' get input from step, given back to the model
        returns nothing
        '''
        pass

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass

    def __str__(self):
        ''' returns the name of the player '''
        return self.__class__.__name__


''' Human player interacts with the CLI'''
class HumanPlayer(Player):
    def __init__(self, gameEnv=None):
        self.gameEnv = gameEnv

    def select_action_cli(self):
        '''input from the command line'''
        badData = True
        while badData:
            try:
                x = float(input("Enter an x position: "))
                y = float(input("Enter an y position: "))

                badData = False
            except ValueError:
                print("Bad input! please enter a float x & y")

        return x, y

    def select_action_gui(self):
        if self.gameEnv is None:
            raise ValueError("Game environment is not set for HumanPlayer")

        coords = []

        def onclick(event):
            if event.inaxes:
                coords[:] = [event.xdata, event.ydata]
                plt.close()

        fig, ax = self.gameEnv.view_board()  # or however you plot the board
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        x, y = coords
        return x, y

    def select_action(self):
        ''' get input'''
        if self.gameEnv is not None:
            x, y = self.select_action_gui()
        else:
            x, y = self.select_action_cli()
        return x, y

    def receive_observation(self, observation, reward, done, info):
        print(f"You got a reward of {reward:0.2f}, is the game done? {done}")

    # Nothing to save/load for a human player
    def save(self, filepath):
        pass

    def load(self, filepath):
        pass


class RandomPlayer(Player):
    def __init__(self, radius=1.0):
        self.radius = radius

    def select_action(self):
        # Random angle and radius for uniform sampling in a circle
        theta = np.random.uniform(0, 2 * np.pi)
        r = self.radius * np.sqrt(np.random.uniform(0, 1))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def receive_observation(self, observation, reward, done, info):
        # Random player does not use observations
        pass

    # Nothing to save/load for a random player
    def save(self, filepath):
        pass

    def load(self, filepath):
        pass