import random
import math
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d


class LacunaBoard:
    def __init__(self, flowerList, flowerCount = 7, radius=1.0, interactive=False):
        # Game Constants
        self.flowerCount = flowerCount # there are x of each token, in x types ()
        self.radius = radius # Circle that the tokens are all in. For plotting only
        self.precision = 0.1 # Precision for the game (how exact tokens should be, or how close they should be to the flowers to count as "on" the flower)

        # Flower tokens (the game pieces)
        self.flowerPositions_initial = flowerList # Where all the game pices were at the start

        # Flower graph
        self.flowerGraph = nx.Graph() # Graph of the flower positions
        self.flowerGraph.add_nodes_from(self.flowerPositions_initial) # Add the flower positions to the graph
        self.find_potential_moves() # Find the potential moves for the game

        # User data
        self.isPlayerATurn = True # True if player A's turn (False for player B)
        self.userFlowers = np.zeros((2, self.flowerCount)) # a count of the tokens the user has claimed
        self.userTokenPositions = np.full((2, 6, 2), np.nan)  # Shape: (2 players, 6 tokens each, 2 coordinates)
        #TODO should there be flowerCount - 1 tokens? (or 6) for each player?

    # Interact with the game
    def _place_user_token(self, player, x, y):
        '''Add one of the users pieces,
        remove the two colinear flower and give them to the user.
        Place a token for the specified player at (x, y)

        returns boolean (was the token was collected?)
        '''

        # Find the first available slot for the player's token
        for i in range(self.userTokenPositions.shape[1]): # for each token
            if np.isnan(self.userTokenPositions[player, i, 0]):  # Check if the slot is empty
                self.userTokenPositions[player, i] = [x, y] # place the token there

                # check if the token intersects with a colinear flower pair
                for edge in self.flowerGraph.edges():
                    f1_id, f2_id = edge # Get the two flower IDs attached to the edge
                    pos1 = self.flowerGraph.nodes[f1_id]['pos']
                    pos2 = self.flowerGraph.nodes[f2_id]['pos']

                    # If the token intersects, remove the flower from the graph
                    if self._does_token_intersect(pos1, pos2, (x, y)):
                        # Add the flower to the user's collection
                        colorID = self.flowerGraph.nodes[f1_id]['colorID']
                        self.userFlowers[player][colorID] += 2

                        # Remove flower from graph
                        self.flowerGraph.remove_node(f1_id)
                        self.flowerGraph.remove_node(f2_id)

                        # print(f"removed nodes {f1_id} and {f2_id}")

                        self.find_potential_moves() # update graph with unblocked edges
                        return True

                return False

    def get_observation(self): #TODO
        ''' Returns the state observations '''

        # Flowers on the board (flattened)
        flower_positions = []
        for node_id, data in self.flowerPositions_initial:
            flowerPos = [-999, -999, -999]
            if node_id in self.flowerGraph.nodes():
                flowerPos=[*data['pos'], data['colorID']]
            flower_positions.extend(flowerPos)

        # Player token positions (flattened, NaNs replaced with zeros or a mask)
        player_tokens = np.nan_to_num(self.userTokenPositions, nan=0.0).flatten()

        # Flowers collected by each player
        collected = self.userFlowers.flatten()

        # Whose turn (0 or 1)
        turn = [1.0 if self.isPlayerATurn else 0.0]

        # Concatenate all parts into a single observation vector
        observation = np.concatenate([flower_positions, player_tokens, collected])

        if(np.isnan(observation).any()): # panic if there are NaNs
            raise ValueError(f"Observation contains NaN values\n{observation=}")

        return observation

    def take_turn(self, x, y):
        '''Take a turn for the current player
        Place a token for the current player at (x, y)
        returns:
        - observations: information about game state
        - reward: for collecting flowers, proximity to flowers
        - done: boolean
        - info: dict for debuging
        '''


        # do the action
        player = 0 if self.isPlayerATurn else 1
        # print(f"Player {player} is taking a turn at ({x}, {y})")
        colectedFlowers = self._place_user_token(player, x, y) # take the turn
        self.isPlayerATurn = not self.isPlayerATurn # Switch turns


        ''' Calculate reward
          + a reward for gaining flowers
          + a smaller reward based on some function f(d), where d is euclidian distance to each flower (reward being close to many flowers)
          + big reward if game is over and the player is the winner
        '''
        # Reward for gaining flowers (number of flowers collected this turn)
        flowerGainedReward = 7 * (1 if colectedFlowers else 0)

        # Reward for proximity to flowers
        flowerProximityReward = 0
        distToFlowerRewardFn = lambda d: math.exp(-8*d) # Reward function for distance to flower

        for flower in self.flowerGraph.nodes(data=True): # distance user to each flower
            flowerPos = flower[1]['pos']
            distance = np.linalg.norm(np.array([x,y]) - np.array(flowerPos))
            flowerProximityReward += distToFlowerRewardFn(distance)

        gameOverReward = 50 if self.is_game_finished() and self.current_winner() == player else 0

        reward = flowerGainedReward + flowerProximityReward + gameOverReward
        # print(f"reward is {flowerGainedReward} + {flowerProximityReward:0.4f} + {gameOverReward} = {reward}")


        done =  self.is_game_finished() # game is done when it is finished
        info = {} # debuging info
        return self.get_observation(), reward, done, info

    def _does_token_intersect(self, start, end, check):
        '''Check if the line between start and end intersects with check position
        All values are tuples of (x, y) coordinates
        '''
        x1, y1 = start
        x2, y2 = end
        px, py = check

        # Check if the point lies within the segment bounds
        if min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2):
            area = abs((x2 - x1) * (py - y1) - (px - x1) * (y2 - y1))
            if area < self.precision: # Close to zero, meaning the point is collinear
                return True
        return False  # Check isn't collinear with start and end


    def is_line_blocked(self, p1, p2):
        '''Check if the line between points 1 & 2 is blocked
        by any other token (flower or user
        '''

        #TODO update to check blockages for user too
        for flower in self.flowerGraph.nodes(data=True):
            if flower == p1 or flower == p2:
                continue # Skip the two flowers being compared

            # Check if the line intersects with the flower
            if self._does_token_intersect(p1[1]['pos'], p2[1]['pos'], flower[1]['pos']):
                return True
        return False  # No flower blocks the line


    # Calculate game features
    def find_potential_moves(self):
        '''Analyse the active game and calculate all possible edges a piece could be placed on

        In the graph, check all the flowers with each other flower of the same color
        and if a line can be drawn between them, without intersecting another flower
        (i.e. if the line between them is not blocked by another flower)
        add an edge between them
        '''

        # Iterate over all pairs of flowers of the same color
        totalColourMatch = 0
        edges = 0
        for flower1 in self.flowerGraph.nodes(data=True):
            id1, node1 = flower1
            for flower2 in self.flowerGraph.nodes(data=True):
                id2, node2 = flower2
                if id1 >= id2: # Don't compare edges prior (duplicates) or self
                    continue

                if node1['color'] == node2['color']:  # Same color
                    totalColourMatch +=1
                    flowerBlocked = self.is_line_blocked(flower1, flower2)
                    if not flowerBlocked:
                        # Add an edge between the two flowers in the graph
                        self.flowerGraph.add_edge(id1, id2)
                        edges += 1

        # print(f"of {totalColourMatch} potential edges, there are {edges}")


    def current_winner(self):
        '''Calculate the winner of the game at its current state
        The winner is the person who collects more flowers in more colors
        0 = player A, 1 = player B
        '''

        pA = self.userFlowers[0] # Player 1's tokens
        pB = self.userFlowers[1]

        winnerPerColor = pA > pB # Calculate the winner for each color

        # pA wins if there are more True values than False (sum > half)
        pAIsWinner = sum(winnerPerColor) > len(winnerPerColor) / 2

        return 0 if pAIsWinner else 1

    def is_game_finished(self):
        '''Check if the game is finished.
        The game is finished when both players have placed all their tokens'''

        return bool(~np.isnan(self.userTokenPositions[:, :, :]).any())

    def _allocate_remaining_flowers(self):
        '''When the tokens are all placed,
        Each remaining flower is given to the player with the closest token
        '''
        if not self.is_game_finished():
            return # Game isn't finished

        # Iterate over all flowers in the graph and find the closest (euluclidean) user
        for flower in self.flowerGraph.nodes(data=True):
            flowerPos = flower[1]['pos']

            # Calculate the distance to each player's tokens
            closestA = np.min(np.linalg.norm(self.userTokenPositions[0, :, :] - flowerPos, axis=1))
            closestB = np.min(np.linalg.norm(self.userTokenPositions[1, :, :] - flowerPos, axis=1))
            closestPlayer = 0 if closestA < closestB else 1 # player A is closer if A < B

            # Add the flower to the player's collection
            colorID = flower[1]['colorID']
            self.userFlowers[closestPlayer][colorID] += 1


    def calculate_winner(self) -> int:
        '''return the game winner, either player 1 or 2.
            Negative value if not finished'''
        if self.is_game_finished():
            self._allocate_remaining_flowers()
            return self.current_winner()
        else:
            return None # Game isn't finished -> no winner yet


    # Display, visualisation methods
    def view_board(self):
        fig, ax = plt.subplots()

        # plot the flowers
        pos = nx.get_node_attributes(self.flowerGraph, 'pos')
        colors = [data.get('color', 'black') for _, data in self.flowerGraph.nodes(data=True)]
        nx.draw(self.flowerGraph, with_labels=True, pos=pos, node_color=colors, ax=ax)

        # Draw the circle representing the board
        boardCircle = plt.Circle((0, 0), self.radius, color='k', fill=False, linewidth=1, linestyle='-' )
        ax.add_patch(boardCircle)

        # Add the user tokens to the plot
        for player in range(2):
            placed_tokens = self.userTokenPositions[player][~np.isnan(self.userTokenPositions[player, :, 0])]
            # print(f"Player {player}'s tokens:\n{placed_tokens}")
            playerColor = ['red', 'blue'] # Color of player token
            ax.scatter(placed_tokens[:, 0], placed_tokens[:, 1], color=playerColor[player], marker='*', label=f"Player {player}")

        ax.set_title("Lacuna Board")
        ax.set_aspect('equal')

        ax.set_xlim([-(self.radius+0.05), (self.radius+0.05)])
        ax.set_ylim([-(self.radius+0.05), (self.radius+0.05)])

        return fig, ax

    def view_board_with_voranoi(self):
        # Draw voranoi diagram and user moves

        # all tokens placed by all players in a single array
        tokenCount = self.userTokenPositions.shape[0] * self.userTokenPositions.shape[1] # users * tokens each
        flatUserTokens = self.userTokenPositions.reshape((tokenCount,2))
        flatUserTokens =  flatUserTokens[~np.isnan(flatUserTokens[:, 0])]

        print(f"{flatUserTokens}")
        v = Voronoi(flatUserTokens)
        print(f"{v}")
        voronoi_plot_2d(v)

        # add the rest on top
        self.view_board()

        return plt


#given an int, return a string color
def get_color(color):
    colorString = ""
    match color:
        case 0: # Red
            colorString = "#eb4034"
        case 1: # Blue
            colorString = "#6886e8"
        case 2: # Cyan
            colorString = "#5cedce"
        case 3: # Yellow
            colorString = "#c99100"
        case 4: # Brown/green
            colorString = "#3cc900"
        case 5: # Pink
            colorString = "#ff00ef"
        case 6: # Purple
            colorString = "#c50cba"
        case _: #Unknown (Black)
            colorString = "#000000"

    return colorString


# new Board with random data
def new_random_lacuna_tokens(flowerCount = 7, radius=1, minDistanceApart = 0.075, seed=None):
    '''Create a new random list of tokens with size tokens of each color

    Args:
    - flowerCount: int, number of colors (types) of tokens
    - radius: float, radius of the circle
    - minDistanceApart: float, minimum distance between tokens
    - seed: int, random seed for reproducibility
    '''

    if seed is not None: # set the seed for reproducibility
        random.seed(seed)

    flowerPositions = []
    flowerNodes = []

    for i in range(flowerCount): # how many colors
        for _ in range(flowerCount): # how many of each color

            inCircle = False # is the token in the circle?
            nearOther = True # are there other tokens nearby?

            while not (inCircle and not nearOther):
                x = random.uniform(-radius, radius)
                y = random.uniform(-radius, radius)

                inCircle = radius > math.sqrt(math.pow(x,2) + math.pow(y,2))
                nearOther = any(math.sqrt((x - fx)**2 + (y - fy)**2) < minDistanceApart for fx, fy, _ in flowerPositions)

            flowerPositions.append([x,y,i])
            nodeDetails = {
                'pos': (x, y), # [float, float] Position of the flower (x, y)
                'color': get_color(i), # [String] color of the flower (hex code)
                'colorID': i, # [int] ID of the flower color
            }
            flowerNodes.append((len(flowerPositions) - 1, nodeDetails))

    return flowerNodes


if __name__ == "__main__":
    radius = 1
    flowerCount = 7
    nodes = new_random_lacuna_tokens(flowerCount=flowerCount, radius=radius, seed = 70, minDistanceApart=0.09)

    board = LacunaBoard(nodes, flowerCount=flowerCount, radius=radius)
    board.view_board()
    # board.take_turn( 0.0, -0.5)
    # board.take_turn( 0.5,  0.7)
    # board.take_turn(-0.4,  0.2)
    # board.take_turn(-0.4, -0.3)
    # print(f"{board.userTokenPositions}")
    # print(f"{board.is_game_finished()}")

    # board.view_board()