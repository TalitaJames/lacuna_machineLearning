from abc import ABC, abstractmethod


''' An abstract base class to ensure all agents have required playing models '''
class Player(ABC):
    @abstractmethod
    def take_turn(self):
        '''decide and return an (x,y) action'''
        pass

    @abstractmethod
    def turn_results(self, observation, reward, done, info):
        ''' get input from step, given back to the model
        returns nothing
        '''
        pass

    @abstractmethod
    def save_model(self, filepath):
        pass


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
            except:
                print("Bad input! please enter a float x &y")

    def turn_results(self, observation, reward, done, info):
        print(f"You got a reward of {reward:0.2f}, the game is{'n\'t' if not done else ''} over")

    def save(self, filename):
        print(f"You are a human! You can't save your brain")
