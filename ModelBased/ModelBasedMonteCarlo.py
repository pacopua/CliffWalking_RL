import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

class Connect2Game:
    """
    A very, very simple game of ConnectX in which we have:
        rows: 1
        columns: 4
        winNumber: 2
    """
    def __init__(self):
        self.columns = 4
        self.win = 2

    def get_init_board(self):
        b = np.zeros((self.columns,), dtype=int)
        return b

    def get_board_size(self):
        return self.columns

    def get_action_size(self):
        return self.columns

    def get_next_state(self, board, player, action):
        b = np.copy(board)
        b[action] = player

        # Return the new game, but
        # change the perspective of the game with negative
        return (b, -player)

    def has_legal_moves(self, board):
        for index in range(self.columns):
            if board[index] == 0:
                return True
        return False

    def get_valid_moves(self, board):
        # All moves are invalid by default
        valid_moves = [0] * self.get_action_size()

        for index in range(self.columns):
            if board[index] == 0:
                valid_moves[index] = 1

        return valid_moves

    def is_win(self, board, player):
        count = 0
        for index in range(self.columns):
            if board[index] == player:
                count = count + 1
            else:
                count = 0

            if count == self.win:
                return True

        return False

    def get_reward_for_player(self, board, player):
        # return None if not ended, 1 if player 1 wins, -1 if player 1 lost

        if self.is_win(board, player):
            return 1
        if self.is_win(board, -player):
            return -1
        if self.has_legal_moves(board):
            return None

        return 0

    def get_canonical_board(self, board, player):
        return player * board

def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score

class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, to_play, action_probs):
        """
        We expand a node and keep track of the prior policy probability
        """
        self.to_play = to_play
        self.state = state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(prior=prob, to_play=self.to_play * -1)

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())
    
class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def select(self, search_path):
        node = search_path[-1]
        action = None
        while node.expanded():
            action, node = node.select_child()
            if node is None:
                break
            search_path.append(node)
        return search_path, action

    def expand(self, parent, action, node):
        state = parent.state
        next_state, _ = self.game.get_next_state(state, player=1, action=action)
        next_state = self.game.get_canonical_board(next_state, player=-1)
        action_probs = self.game.get_valid_moves(next_state)
        if np.sum(action_probs) > 0:
            action_probs /= np.sum(action_probs)
            node.expand(next_state, parent.to_play * -1, action_probs)
        else:
            node.expand(next_state, parent.to_play * -1, action_probs)
        return node

    def simulate(self, node):
        state = node.state
        reward = self.game.get_reward_for_player(state, player=1)
        while reward is None:  # Not finished yet
            action_probs = self.game.get_valid_moves(state)
            action_probs /= np.sum(action_probs)
            next_action = np.random.choice(range(len(action_probs)), p=action_probs)
            state, _ = self.game.get_next_state(state, player=1, action=next_action)
            state = self.game.get_canonical_board(state, player=-1)
            reward = self.game.get_reward_for_player(state, player=1)
        return reward

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

    def run(self, state, to_play):
        root = Node(0, to_play)
        root_values = []

        action_probs = self.game.get_valid_moves(state)
        action_probs /= np.sum(action_probs)
        root.expand(state, to_play, action_probs)

        for _ in range(self.args['num_simulations']):
            node = root
            search_path = [node]

            search_path, last_action = self.select(search_path)
            parent = search_path[-2]
            node = search_path[-1]
            node = self.expand(parent, last_action, node)
            value = self.simulate(node)
            self.backpropagate(search_path, value, parent.to_play * -1)
            root_values.append(root.value_sum/root.visit_count)
        return root, root_values
    
class Trainer:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.mcts = MCTS(self.game, self.args)

    def learn(self):
        current_player = 1
        state = self.game.get_init_board()

        self.mcts = MCTS(self.game, self.args)
        root, root_values = self.mcts.run(state, to_play=1)
        return root, root_values
    
args = {
    'num_simulations': 5  # Total number of MCTS simulations to run when deciding on a move to play
}

game = Connect2Game()
trainer = Trainer(game, args)
root, root_values = trainer.learn()

print(root)
print(root.children)

plt.figure(figsize=(10, 6))
sns.lineplot(x=np.arange(len(root_values)), y=root_values)
plt.title('Valor del nodo raíz para el jugador 1')
plt.xlabel('Iteraciones')
plt.ylabel('Valor')
plt.show()