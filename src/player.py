from abc import ABC, abstractmethod


''' An abstract base class to ensure all agents have required playing models '''
class Player(ABC):
    @abstractmethod
    def take_turn(self):
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

    def getName(self):
        ''' returns the name of the player '''
        return self.__class__.__name__


''' Human player interacts with the CLI'''
class HumanPlayer(Player):
    def __init__(self):
        pass

    def take_turn(self):
        badData = True
        while badData:
            try:
                x = float(input("Enter an x position: "))
                y = float(input("Enter an y position: "))

                badData = False
            except ValueError:
                print("Bad input! please enter a float x & y")

        return x, y

    def receive_observation(self, observation, reward, done, info):
        print(f"You got a reward of {reward:0.2f}, is the game done? {done}")

    def save(self, filename):
        print(f"You are a human! You can't save your brain")
