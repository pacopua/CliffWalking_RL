import math
import random

class Node:
    def __init__(self, state, parent=None, action_taken=None, prior_p=0.0, env_actions_n=4):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken  # Action that led from parent to this node
        self.children = {}  # map action to Node
        self.visit_count = 0
        self.value_sum = 0.0  # Accumulated reward from rollouts
        self.prior_p = prior_p # Prior probability (can be uniform if no policy network)
        self.env_actions_n = env_actions_n # Number of possible actions from this state

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_fully_expanded(self):
        # Assuming discrete actions 0 to N-1
        return len(self.children) == self.env_actions_n

    def select_child_ucb(self, exploration_weight):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            # UCB1 formula
            exploitation_term = child.value()
            exploration_term = exploration_weight * child.prior_p * \
                               math.sqrt(self.visit_count) / (1 + child.visit_count)
            
            # In single-player, we maximize. If it were 2-player, opponent's value is -child.value().
            # Here, child.value() is directly the value from that child's perspective.
            score = exploitation_term + exploration_term
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child
    
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections
import csv
import math # Make sure math is imported
import random # Make sure random is imported

# Declaración de constantes (keep these as they are)
T_MAX = 150
NUM_EPISODES = 100
GAMMA = 0.95
REWARD_THRESHOLD = -200 # Adjusted for potentially slower initial learning with MCTS

# --- MCTS Node Class (from above) ---
class Node:
    def __init__(self, state, parent=None, action_taken=None, env_actions_n=4): # Removed prior_p for simplicity now
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.env_actions_n = env_actions_n
        # For CliffWalking, we need to know if a state in the *model* is terminal
        # This is a simplification; actual terminal states are defined by the environment
        # We'll use this to stop rollouts if we hit a known cliff or goal in our model
        self.is_model_terminal = False # Will be set based on learned rewards/transitions

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_fully_expanded(self, known_actions_for_state):
        # A node is fully expanded if all known actions from this state in the model have children
        # or if there are no known actions from this state (effectively a terminal leaf in the model)
        if not known_actions_for_state: # No known actions from this state in our model
            return True
        return len(self.children) == len(known_actions_for_state)


    def select_child_ucb(self, exploration_weight):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        # Ensure parent visit count is positive before taking log
        log_parent_visit_count = math.log(self.visit_count) if self.visit_count > 0 else 0

        for action, child in self.children.items():
            exploitation_term = child.value() # Value from child's perspective
            exploration_term = exploration_weight * \
                               math.sqrt(log_parent_visit_count / (child.visit_count + 1e-6)) # Add epsilon for stability
            
            score = exploitation_term + exploration_term
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

class ModelBasedMCTSAgent:
    def __init__(self, env, gamma, num_trajectories_for_model_update, 
                 mcts_simulations_per_action=50, mcts_rollout_depth=15, 
                 mcts_exploration_weight=1.0):
        self.env = env
        self.state, _ = self.env.reset() # Initial state
        
        # Learned Model
        self.rewards = collections.defaultdict(float)  # R(s, a, s')
        self.transits = collections.defaultdict(collections.Counter)  # T(s, a)[s'] = count
        self.model_known_states = set() # States for which we have model information

        self.gamma = gamma
        self.num_trajectories_for_model_update = num_trajectories_for_model_update # For play_n_random_steps

        # MCTS parameters
        self.mcts_simulations_per_action = mcts_simulations_per_action
        self.mcts_rollout_depth = mcts_rollout_depth
        self.mcts_exploration_weight = mcts_exploration_weight

        # CliffWalking specific (can be generalized later)
        # These are actual terminal states in the CliffWalking environment
        self.cliff_states = set(range(37, 47)) 
        self.goal_state = 47 # State 47 (3*12 + 11) is the goal for 4x12 grid (0-indexed)

    def _is_env_terminal(self, state):
        """Checks if a state is a terminal state in the actual environment."""
        return state == self.goal_state or state in self.cliff_states

    def play_n_random_steps(self, count):
        """Explores environment to build/update the model."""
        print(f"Playing {count} random steps to update model...")
        initial_model_size = len(self.transits)
        for i in range(count):
            if (i + 1) % (count // 10 if count >=10 else 1) == 0:
                print(f"  Random step {i+1}/{count}")

            action = self.env.action_space.sample()
            current_s = self.state # Use the agent's current tracked state
            
            new_state, reward, is_done, truncated, _ = self.env.step(action)
            
            self.rewards[(current_s, action, new_state)] = reward
            self.transits[(current_s, action)][new_state] += 1
            self.model_known_states.add(current_s)
            self.model_known_states.add(new_state)

            if is_done or truncated:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state
        print(f"Model updated. Transit dictionary size: {initial_model_size} -> {len(self.transits)}")


    def _get_known_actions_from_model(self, state):
        """Returns a list of actions for which we have model data from the given state."""
        known_actions = []
        for action in range(self.env.action_space.n):
            if (state, action) in self.transits:
                known_actions.append(action)
        return known_actions

    def _sample_from_model(self, state, action):
        """Samples a next_state and reward from the learned model."""
        if (state, action) not in self.transits or not self.transits[(state, action)]:
            # This action from this state is unknown to the model or leads nowhere
            # Treat as absorbing state with 0 reward for rollout purposes
            # Or, if it's a known cliff/goal, use appropriate reward
            if self._is_env_terminal(state): # If it's an actual terminal state
                 # This case should ideally be handled before calling _sample_from_model
                 # by checking node.is_model_terminal in the rollout
                 # For cliff, reward is typically -100. For goal, often 0. env step gives this.
                 # Let's assume the reward for reaching this terminal state was already accounted for.
                 # For rollout, if we land here, the rollout ends.
                 return state, 0, True # (next_state is same, reward 0, is_done true)
            return state, -1, True # Fallback: unknown transition, penalize slightly & terminate rollout path


        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        
        # Weighted random choice for next state
        possible_next_states = list(target_counts.keys())
        probabilities = [count / total for count in target_counts.values()]
        
        sampled_next_state = random.choices(possible_next_states, weights=probabilities, k=1)[0]
        
        # Get reward for this specific transition (s, a, s')
        reward = self.rewards.get((state, action, sampled_next_state), 0) 
        
        # Determine if this sampled_next_state is terminal according to our knowledge
        is_done = self._is_env_terminal(sampled_next_state)
        
        return sampled_next_state, reward, is_done

    # --- MCTS Core Logic ---
    def _select_node(self, root_node):
        node = root_node
        while True:
            known_actions = self._get_known_actions_from_model(node.state)
            if node.is_model_terminal or not known_actions: # If model thinks it's terminal or no way out
                break
            
            if not node.is_fully_expanded(known_actions):
                return self._expand(node, known_actions) # Expand and return new child

            # If fully expanded and not terminal, select best child
            _, node = node.select_child_ucb(self.mcts_exploration_weight)
            if node is None: # Should not happen if fully_expanded but no children. Safety.
                 # This can happen if select_child_ucb has an issue or all children have -inf score
                 # Or if self.visit_count was 0 in UCB.
                 # Let's treat this as a leaf for now.
                 break
        return node # This is a leaf node for rollout

    def _expand(self, parent_node, known_actions):
        # Find an action that hasn't been expanded yet
        untried_actions = [a for a in known_actions if a not in parent_node.children]
        
        if not untried_actions: # Should not happen if not is_fully_expanded
            parent_node.is_model_terminal = True # Mark as model terminal if no path out
            return parent_node

        action_to_expand = random.choice(untried_actions)
        
        # Simulate one step using the *model* to get a potential next state
        # Note: MCTS expansion typically doesn't involve rewards, just creates the node.
        # The reward comes from the rollout or value estimator.
        # However, for model-based MCTS, we can use the immediate reward from the model.
        
        # For creating the child, we just need *a* state. The true stochasticity is handled during rollouts.
        # Let's pick the most likely next state for the child node's state attribute.
        # The actual rollout from this child will still use the full stochastic model.
        
        if (parent_node.state, action_to_expand) in self.transits:
            target_counts = self.transits[(parent_node.state, action_to_expand)]
            if not target_counts: # No transitions for this action
                parent_node.is_model_terminal = True
                return parent_node # Can't expand this action

            most_likely_next_state = max(target_counts, key=target_counts.get)
            child_state = most_likely_next_state
        else: # Should not happen if action_to_expand is from known_actions
            parent_node.is_model_terminal = True
            return parent_node


        child_node = Node(child_state, parent=parent_node, action_taken=action_to_expand, env_actions_n=self.env.action_space.n)
        child_node.is_model_terminal = self._is_env_terminal(child_state) # Check if this state is known to be terminal
        
        parent_node.children[action_to_expand] = child_node
        return child_node

    def _simulate_rollout(self, start_node):
        # A rollout uses the learned model to simulate a trajectory.
        current_state = start_node.state
        total_rollout_reward = 0.0
        current_gamma = 1.0
        
        if start_node.is_model_terminal: # If the node itself is terminal, rollout gives its inherent value (often 0 if reward is for action)
            # Or, if it's a cliff, the reward for reaching it was already given.
            # For MCTS, the value of a terminal state is often its immediate utility.
            # Here, we check if it's a cliff or goal based on environment definition.
            if current_state in self.cliff_states:
                 return -100 # Standard cliff penalty
            elif current_state == self.goal_state:
                 return 0 # Standard goal reward (often episode ends, no further reward)
            return 0 # Default for other model-terminal states

        for _ in range(self.mcts_rollout_depth):
            if self._is_env_terminal(current_state): # Check against actual env terminal states
                # If it's a cliff, the reward for *landing* there is what matters
                if current_state in self.cliff_states:
                    total_rollout_reward += current_gamma * (-100) # Reward for falling
                # If it's the goal, typically the episode ends, reward may be 0 or positive.
                # The step *into* the goal gives the reward.
                break 
            
            # Default policy: random action among known actions, or any action if state is new to model
            known_actions_from_current = self._get_known_actions_from_model(current_state)
            if not known_actions_from_current: # State not in model or no outgoing transitions
                # Consider this an absorbing state in the model's view for the rollout
                # Or, if it's an actual terminal state, give its reward
                if self._is_env_terminal(current_state):
                    if current_state in self.cliff_states: total_rollout_reward += current_gamma * (-100)
                # else, it's a dead-end in the model, maybe give a small penalty or 0.
                break 

            action = random.choice(known_actions_from_current)
            
            next_state, reward, is_done = self._sample_from_model(current_state, action)
            total_rollout_reward += current_gamma * reward
            current_state = next_state
            current_gamma *= self.gamma
            
            if is_done: # is_done from model perspective (e.g. hit cliff/goal)
                break
                
        return total_rollout_reward

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.value_sum += reward # Reward is from perspective of player at 'node'
            # For single agent, reward is direct. For 2-player, might flip sign for parent.
            node = node.parent

    def select_action(self, current_env_state):
        """Selects an action from the current_env_state using MCTS."""
        if not self.model_known_states: # If model is empty, take random action
            print("Warning: MCTS called with an empty model. Taking random action.")
            return self.env.action_space.sample()
        
        # Check if current_env_state is known to the model at all.
        # If not, MCTS can't do much. Could do a random step or try to learn about it.
        if current_env_state not in self.model_known_states:
            # This implies we should have played more random steps or the state is truly new
            # print(f"Warning: State {current_env_state} not in model. Taking random action.")
            return self.env.action_space.sample()

        root_node = Node(current_env_state, env_actions_n=self.env.action_space.n)
        root_node.is_model_terminal = self._is_env_terminal(current_env_state) # Is the root itself terminal?

        if root_node.is_model_terminal: # If starting in a terminal state, no real action to take
            # print(f"Info: MCTS called on a terminal state {current_env_state}. No sensible action.")
            # Fallback: if code expects an action, give a random one. Or handle this case upstream.
            return self.env.action_space.sample()


        for _ in range(self.mcts_simulations_per_action):
            leaf_node = self._select_node(root_node) # Phase 1: Selection (and possibly expansion)
            rollout_reward = self._simulate_rollout(leaf_node) # Phase 2: Simulation
            self._backpropagate(leaf_node, rollout_reward) # Phase 3: Backpropagation

        if not root_node.children:
            # This can happen if root_node is terminal, or no actions are known from it,
            # or all simulations led to discovering it's a dead-end.
            # print(f"Warning: MCTS root for state {current_env_state} has no children after simulations. Random action.")
            return self.env.action_space.sample()

        # Select action with the highest visit count (robust) or highest value
        best_action = -1
        # max_visits = -1
        # for action, child in root_node.children.items():
        #     if child.visit_count > max_visits:
        #         max_visits = child.visit_count
        #         best_action = action
        
        # Or, choose by max value (more greedy)
        max_value = -float('inf')
        for action, child in root_node.children.items():
            if child.value() > max_value: # Use the E(Q) value
                max_value = child.value()
                best_action = action

        if best_action == -1: # Fallback if no children were promising or visited
            # print(f"Warning: MCTS couldn't determine best action for state {current_env_state}. Random action.")
            return self.env.action_space.sample()
            
        return best_action

    def update_model_and_plan(self): # Replaces value_iteration conceptually
        """Call this to update the model, MCTS doesn't have an explicit V-table to iterate."""
        self.play_n_random_steps(self.num_trajectories_for_model_update)
        # MCTS plans on-demand in select_action. No explicit value table like V to return difference for.
        # So, we can return a dummy max_diff or remove it from the training loop.
        return 0 # Dummy max_diff

    def get_policy_for_visualization(self):
        """
        Generates a policy table by running MCTS for all known non-terminal states.
        This is computationally expensive and only for visualization.
        """
        print("Generating policy for visualization (can be slow)...")
        policy = np.full(self.env.observation_space.n, -1, dtype=int) # -1 for unknown/terminal
        
        # States that are part of the cliff or the goal state
        terminal_env_states = self.cliff_states.union({self.goal_state})

        for s_idx in range(self.env.observation_space.n):
            if s_idx in self.model_known_states and s_idx not in terminal_env_states:
                # Temporarily run MCTS for this state to get the best action
                # This is distinct from the MCTS run during an actual game step
                # print(f"  Calculating policy for state {s_idx}...")
                root = Node(s_idx, env_actions_n=self.env.action_space.n)
                if self._is_env_terminal(s_idx): # Should be caught by outer if, but safety
                    policy[s_idx] = -1 # Or a default like 'stay' if applicable
                    continue

                for _ in range(self.mcts_simulations_per_action // 4 or 1): # Fewer sims for policy viz
                    leaf = self._select_node(root)
                    reward = self._simulate_rollout(leaf)
                    self._backpropagate(leaf, reward)
                
                if root.children:
                    best_action = -1
                    max_val = -float('inf')
                    # To get policy, usually choose action with max visits or max value
                    for act, child_node in root.children.items():
                        if child_node.value() > max_val:
                            max_val = child_node.value()
                            best_action = act
                    policy[s_idx] = best_action
                else: # No children could be formed or explored
                    policy[s_idx] = -1 # Could also be random if preferred for unknown
            elif s_idx in terminal_env_states:
                policy[s_idx] = 4 # Use a different marker for actual terminal states, e.g., 'T'
            else:
                policy[s_idx] = -1 # Default for states not in model or terminal

        print("Policy generation for visualization complete.")
        return policy
    
env = gym.make('CliffWalking-v0') #, is_slippery = True -> This param doesn't exist for CliffWalking
                                     # Slippery is inherent to FrozenLake, not CliffWalking.
                                     # CliffWalking is deterministic by default.
                                     # If you want stochasticity, you'd need to wrap the env or use a diff version.
                                     # For now, assuming standard CliffWalking.
                                     # If you have a custom slippery CliffWalking, the MCTS will learn its model.

def test_episode(agent, env_to_test): # Pass env explicitly
    state, _ = env_to_test.reset()
    is_done = False
    t = 0
    # print("episode: ", t) # This t is step, not episode
    total_ep_reward = 0
    while not is_done and t < T_MAX : # Add T_MAX here too
        action = agent.select_action(state)
        next_state, reward, is_done, truncated, info = env_to_test.step(action)
        total_ep_reward += reward
        state = next_state
        t += 1
        if truncated: is_done = True # Treat truncated as done for episode end
    return total_ep_reward # Return total reward of the episode

def draw_rewards(rewards_list): # Renamed for clarity
    data = pd.DataFrame({'Episode': range(1, len(rewards_list) + 1), 'Reward': rewards_list})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Reward', data=data)
    plt.title('Rewards Over Training Iterations')
    plt.xlabel('Training Iteration')
    plt.ylabel('Average Reward over Test Episodes')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def check_improvements(agent, eval_env): # Pass env
    reward_test_sum = 0.0
    for i in range(NUM_EPISODES):
        episode_reward = test_episode(agent, eval_env) # Use test_episode
        reward_test_sum += episode_reward
    reward_avg = reward_test_sum / NUM_EPISODES
    return reward_avg

def train_mcts_agent(agent, train_env, eval_env): # Pass envs
    avg_rewards_over_iterations = []
    #max_diffs = [] # Not directly applicable to MCTS model updates like VI
    iteration_count = 0
    best_reward_so_far = -float('inf') # Initialize correctly

    # Initial model building
    print("Performing initial model building...")
    agent.play_n_random_steps(agent.num_trajectories_for_model_update * 2) # More steps initially

    # MCTS doesn't have a direct "convergence" like VI's max_diff.
    # We train for a fixed number of iterations or until performance plateaus.
    # Let's use a max number of training iterations for MCTS.
    MAX_TRAINING_ITERATIONS = 50 # Example, adjust as needed

    for iteration_count in range(MAX_TRAINING_ITERATIONS):
        print(f"\n--- Training Iteration {iteration_count + 1}/{MAX_TRAINING_ITERATIONS} ---")
        
        # Update the model (MCTS uses this model for planning)
        # This is like the "value_iteration" step in the old code, but just for model update
        agent.play_n_random_steps(agent.num_trajectories_for_model_update)
        # max_diff = agent.update_model_and_plan() # This now just updates model
        # max_diffs.append(max_diff)
        # print(f"Model updated. (No max_diff for MCTS like VI)")

        current_avg_reward = check_improvements(agent, eval_env)
        avg_rewards_over_iterations.append(current_avg_reward)
        print(f"Iteration {iteration_count + 1}: Avg Test Reward = {current_avg_reward:.2f}")
               
        if current_avg_reward > best_reward_so_far:
            print(f"Best reward updated {current_avg_reward:.2f} at iteration {iteration_count + 1}")
            best_reward_so_far = current_avg_reward
        
        if best_reward_so_far > REWARD_THRESHOLD : # Using the same threshold
            print(f"Reward threshold {REWARD_THRESHOLD} reached. Stopping training.")
            break
            
    #return avg_rewards_over_iterations, max_diffs
    return avg_rewards_over_iterations


def print_policy_mcts(policy_array): # Modified to handle MCTS policy output
    rows, cols = 4, 12 # CliffWalking grid dimensions
    if len(policy_array) != rows * cols:
        print(f"Warning: Policy length ({len(policy_array)}) doesn't match grid dimensions ({rows*cols}).")
    
    # Action mapping: 0:^, 1:>, 2:v, 3:<. Let's add -1 for unknown, 4 for terminal
    visual_help = {0:'^', 1:'>', 2:'v', 3:'<', -1:'?', 4:'T'} 
    
    policy_arrows = []
    for action_val in policy_array:
        action_int = int(action_val) # Ensure it's an int for dict key
        policy_arrows.append(visual_help.get(action_int, '?')) # Use .get for safety

    try:
        print("Learned Policy (via MCTS evaluation):")
        print(np.array(policy_arrows).reshape(rows, cols))
    except ValueError as e:
        print(f"Error reshaping policy: {e}")
        print("Policy array:", policy_arrows)


def inspect_policy_after_training(agent, eval_env): # Renamed and pass env
    episode_rewards = []
    print(f"\nInspecting policy over {NUM_EPISODES} episodes...")
    for n_ep in range(NUM_EPISODES):
        # print(f"Episode: {n_ep + 1}/{NUM_EPISODES}")
        ep_rew = test_episode(agent, eval_env) # Use the test_episode function
        episode_rewards.append(ep_rew)
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"Average reward over {NUM_EPISODES} inspection episodes: {avg_reward:.2f}")
    # draw_rewards(episode_rewards) # This would show rewards per episode, not per training iteration
                                 # The main training loop's reward plot is more informative for learning progress.

def print_learned_model_stats(agent): # Simplified from full print
    print("\n=== LEARNED ENVIRONMENT MODEL STATS ===")
    print(f"  Number of known states in model: {len(agent.model_known_states)}")
    print(f"  Number of (s,a) transition entries: {len(agent.transits)}")
    num_sa_s_reward_entries = sum(len(s_prime_counts) for s_prime_counts in agent.transits.values())
    print(f"  Number of (s,a,s') reward entries: {len(agent.rewards)} (unique s,a,s')")
    print(f"  Total (s,a) -> s' transitions learned: {num_sa_s_reward_entries}")


def main():
    # Create two separate environment instances: one for training, one for evaluation/rendering
    # This is good practice to avoid interference.
    train_env = gym.make('CliffWalking-v0') 
    eval_env = gym.make('CliffWalking-v0')
    
    # Slippery CliffWalking: If your gym version supports it or you have a custom one.
    # If not, CliffWalking is deterministic. MCTS can handle either.
    # Example for a hypothetical slippery version:
    # train_env = gym.make('CliffWalking-v0', is_slippery=True)
    # eval_env = gym.make('CliffWalking-v0', is_slippery=True)
    # Since 'is_slippery' is not standard for CliffWalking, I'll assume standard deterministic for now.
    # The MCTS agent will learn the environment dynamics as they are.

    # Hyperparameters for MCTS Agent
    # These might need tuning!
    # Number of random steps taken per call to play_n_random_steps during training loop
    MODEL_UPDATE_TRAJECTORIES = 2000 # Was 1,000,000, too high for iterative updates.
                                     # For initial fill, can be higher. For updates, smaller.
    MCTS_SIMULATIONS = 30            # Number of MCTS simulations per action selection.
    MCTS_ROLLOUT_DEPTH = 10          # Max depth of a single MCTS rollout.
    MCTS_EXPLORATION = 1.41          # UCB exploration constant (sqrt(2) is common).

    agent = ModelBasedMCTSAgent(
        env=train_env, # Agent primarily interacts with train_env to build model
        gamma=GAMMA,
        num_trajectories_for_model_update=MODEL_UPDATE_TRAJECTORIES,
        mcts_simulations_per_action=MCTS_SIMULATIONS,
        mcts_rollout_depth=MCTS_ROLLOUT_DEPTH,
        mcts_exploration_weight=MCTS_EXPLORATION
    )
    
    rewards_history = train_mcts_agent(agent, train_env, eval_env) # Pass both envs
    draw_rewards(rewards_history) # Plot average rewards over training iterations

    print_learned_model_stats(agent)

    print("\nAttempting to visualize policy (running MCTS for many states)...")
    policy_viz = agent.get_policy_for_visualization()
    print_policy_mcts(policy_viz)

    print("\nRunning inspection episodes with the trained MCTS agent...")
    inspect_policy_after_training(agent, eval_env) # Use eval_env for inspection

    print("\nVisualizing agent behavior in one episode (human mode)...")
    # For rendering, create a new env with render_mode="human"
    render_env = gym.make('CliffWalking-v0', render_mode="human")
    # If using a slippery variant, ensure render_env is also slippery:
    # render_env = gym.make('CliffWalking-v0', render_mode="human", is_slippery=True)

    state, _ = render_env.reset()
    render_env.render()
    is_done = False
    t = 0
    total_render_reward = 0
    while not is_done and t < T_MAX:
        action = agent.select_action(state) # Agent uses its learned model
        state, reward, is_done, truncated, _ = render_env.step(action)
        total_render_reward += reward
        render_env.render()
        t += 1
        if truncated: is_done = True
    print(f"Rendered episode finished. Total reward: {total_render_reward}, Steps: {t}")

    train_env.close()
    eval_env.close()
    render_env.close()

if __name__ == "__main__":
    main()