import numpy as np
import math
import seaborn as sns
from matplotlib import pyplot as plt

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
    
def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)

class LinearModel:
    def __init__(self, value_weights, policy_weights):
        self.value_weights = value_weights
        self.policy_weights = policy_weights

    def predict(self, features):
        value = np.dot(features, self.value_weights)  # Value prediction
        policy_scores = np.dot(features, self.policy_weights)  # Policy score
        policy_probs = softmax(policy_scores)  # Convert scores to probabilities
        return policy_probs, value

def extract_features(state):
  features = state
  return features

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
        We expand a node and keep track of the prior policy probability given by neural network
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
    def __init__(self, game, model, args):
        self.game = game
        self.model = model  # An instance of LinearModel
        self.args = args

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
        features = extract_features(state)
        action_probs, value = self.model.predict(features)

        valid_moves = self.game.get_valid_moves(state)
        action_probs = action_probs * valid_moves  # Mask invalid moves
        action_probs /= np.sum(action_probs)  # Normalize probabilities
        root.expand(state, to_play, action_probs)

        for _ in range(self.args['num_simulations']):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            next_state, _ = self.game.get_next_state(state, player=1, action=action)
            # Get the board from the perspective of the other player
            next_state = self.game.get_canonical_board(next_state, player=-1)

            # The value of the new state from the perspective of the other player
            value = self.game.get_reward_for_player(next_state, player=1)
            if value is None:
                # If the game has not ended:
                # EXPAND
                leaf_features = extract_features(next_state)
                action_probs, value = self.model.predict(leaf_features)
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = action_probs * valid_moves  # mask invalid moves
                action_probs /= np.sum(action_probs)
                node.expand(next_state, parent.to_play * -1, action_probs)

            # Backpropagate the value estimate
            self.backpropagate(search_path, value, parent.to_play * -1)

        return root
    
class Trainer:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.mcts = MCTS(self.game, self.model, self.args)
        self.learning_rate = args['learning_rate']

    def execute_episode(self):
        train_examples = []
        current_player = 1
        state = self.game.get_init_board()

        while True:
            canonical_board = self.game.get_canonical_board(state, current_player)
            self.mcts = MCTS(self.game, self.model, self.args)
            root = self.mcts.run(canonical_board, current_player)

            action_probs = np.zeros(self.game.get_action_size())
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            action_probs /= np.sum(action_probs)
            train_examples.append((canonical_board, current_player, action_probs, root.value()))

            action = root.select_action(temperature=self.args['temperature'])
            state, current_player = self.game.get_next_state(state, current_player, action)
            reward = self.game.get_reward_for_player(state, current_player)

            if reward is not None:
                ret = []
                for hist_state, hist_current_player, hist_action_probs, hist_value in train_examples:
                    # Adjust the reward based on the player's perspective
                    adjusted_reward = reward if hist_current_player == current_player else -reward
                    ret.append((hist_state, hist_action_probs, adjusted_reward))
                return ret

    def evaluate(self):
        state = game.get_init_board()
        player = 1
        while game.get_reward_for_player(state, player) is None:
            prediction = model.predict(extract_features(state))
            valid_moves = game.get_valid_moves(state)
            action_probs = prediction[0] * valid_moves
            action = np.argmax(softmax(action_probs))
            state, player = game.get_next_state(state, player, action)
        last_move_player = player * -1
        return 1 if last_move_player * game.get_reward_for_player(state, last_move_player) == 1 else 0

    def learn(self):
        successes = []
        for i in range(1, self.args['numIters'] + 1):
            train_examples = []
            for eps in range(self.args['numEps']):
                iteration_train_examples = self.execute_episode()
                train_examples.extend(iteration_train_examples)

            # Implement simple stochastic gradient descent for updating weights
            for example in train_examples:
                features, action_probs, reward = example
                features = extract_features(features)
                predicted_probs, predicted_value = self.model.predict(features)

                # Calculate gradients for policy and value weights
                # We assume that the policy provided by the model is correct,
                # and that the MCTS algorithm has explored balancing exploration
                # and explotation, so the action probabilities should be a good
                # guidance. Therefore we use these probabilities to compute the error
                policy_error = predicted_probs - action_probs
                # For the value, we just compute the error based on the final (real) reward
                value_error = predicted_value - reward

                # Gradient of loss function, in this case derived from Mean Square Error
                grad_policy = np.outer(features, policy_error)  # np.outer: matrix to matrix
                grad_value = value_error * features  # scaling of weights by the error

                # Update weights. We want to minimize the error, so this is
                # gradient descent: we need to substract in this case
                self.model.policy_weights -= self.learning_rate * grad_policy
                self.model.value_weights -= self.learning_rate * grad_value

            # print(f"Iteration {i}/{self.args['numIters']} completed.")
            successes.append(self.evaluate())
        return successes

args = {
    'numIters': 100,  # Total number of training iterations
    'num_simulations': 200,  # Total number of MCTS simulations to populate the tree
    'numEps': 1,  # Number of full games (episodes) to run during each iteration
    'seeds': 5,  # Number of models to train (for producing statistics)
    'temperature': 0,  # Degree of randomness for select_action, to boost exploration
    'learning_rate': 0.01
}

game = Connect2Game()
action_size = game.get_action_size()

# Initialize the trainer with the game, model, and args
all_results = []
for i in range(args['seeds']):
    # Initialize the LinearModel with randomly generated weights
    value_weights = np.random.rand(4)  # Simple feature vector dimension example
    policy_weights = np.random.rand(action_size, 4)  # Feature vector for each possible action
    model = LinearModel(value_weights, policy_weights)
    trainer = Trainer(game, model, args)
    all_results.append(trainer.learn())
counts_per_iteration = [sum(x) for x in zip(*all_results)]

print(trainer.model.predict(extract_features([0, 0, 0, 0])))
print(trainer.model.predict(extract_features([1, 0, 0, 0])))
print(trainer.model.predict(extract_features([0, 1, 0, 0])))
print(trainer.model.predict(extract_features([0, 0, 1, 0])))
print(trainer.model.predict(extract_features([0, 0, 0, 1])))
print(trainer.model.value_weights)
print(trainer.model.policy_weights)

plt.figure(figsize=(10, 6))
ax = sns.lineplot(x=np.arange(len(counts_per_iteration)), y=[x*100/args['seeds'] for x in counts_per_iteration])
ax.set_ylim([0, 100])
plt.title('Rendimiento de los modelos de política')
plt.xlabel('Iteraciones')
plt.ylabel('Éxito (%)')
plt.show()