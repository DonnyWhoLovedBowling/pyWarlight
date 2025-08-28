import math
import random
from collections import defaultdict
import torch

from src.game.Game import Game


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self.state.get_legal_actions() if hasattr(self.state, 'get_legal_actions') else []

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / (child.visits + 1e-8)) + c_param * math.sqrt(2 * math.log(self.visits + 1) / (child.visits + 1e-8))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

class MCTS:
    def __init__(self, model, n_simulations=50, c_param=1.4, rollout_depth=5):
        self.model = model  # PPO model for value/policy estimation
        self.n_simulations = n_simulations
        self.c_param = c_param
        self.rollout_depth = rollout_depth

    def search(self, root_state):
        root = MCTSNode(root_state)
        for _ in range(self.n_simulations):
            node = root
            state = root_state.clone() if hasattr(root_state, 'clone') else root_state
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c_param)
                state = state.step(node.action)
            # Expansion
            if node.untried_actions:
                action = random.choice(node.untried_actions)
                next_state = state.step(action)
                child_node = MCTSNode(next_state, parent=node, action=action)
                node.children.append(child_node)
                node.untried_actions.remove(action)
                node = child_node
                state = next_state
            # Simulation
            reward = self.rollout(state)
            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent
        # Return the action with the highest visit count
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def rollout(self, state: Game):
        current_state = state.clone() if hasattr(state, 'clone') else state
        for _ in range(self.rollout_depth):
            legal = current_state.get_legal_actions() if hasattr(current_state, 'get_legal_actions') else []
            if not legal:
                break
            action = random.choice(legal)
            current_state = current_state.step(action)
        # Use PPO model to estimate value at leaf
        node_features = current_state.create_node_features() if hasattr(current_state, 'create_node_features') else None
        edge_features = current_state.create_edge_features() if hasattr(current_state, 'create_edge_features') else None
        if node_features is not None:
            with torch.no_grad():
                value = self.model.get_value(
                    torch.tensor(node_features, dtype=torch.float),
                    torch.tensor(edge_features, dtype=torch.float) if edge_features is not None else None,
                    None
                ).item()
            return value
        return 0.0

