"""
DQN Agent for Pathway Recommendations.

This is the unified agent that includes all improvements:
- Double DQN (reduces Q-value overestimation)
- Dueling Architecture (separate value/advantage streams)
- Prioritized Experience Replay (efficient sampling)
- Reward Shaping (dense learning signal)
- Batch Normalization (stable gradients)

Based on:
- Rainbow DQN (Hessel et al., 2017)
- IRT-based latent competency model
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    PATHWAYS, COMPETENCIES, RL_CONFIG, MODELS_DIR,
    get_pathway_from_index
)
from config.competencies import IDEAL_PATHWAY_PROFILES

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# REPLAY BUFFER WITH PRIORITIZATION
# =============================================================================

class SumTree:
    """Sum Tree for efficient priority-based sampling O(log n)."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.data_pointer = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        return self.tree[0]
    
    def add(self, priority: float, data):
        idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, any]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class ReplayBuffer:
    """Prioritized Experience Replay Buffer."""
    
    EPSILON = 1e-6
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4,
                 beta_frames: int = 2000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames  # anneal to 1.0 over beta_frames steps
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int):
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            if data is None:
                # Fallback: scan from root with uniform sample
                s = np.random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
            if data is None:
                continue   # skip — buffer genuinely empty
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-self.beta)
        weights = weights / weights.max()
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.EPSILON) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.tree.n_entries


# =============================================================================
# NEURAL NETWORKS
# =============================================================================

if TORCH_AVAILABLE:
    class DuelingDQNetwork(nn.Module):
        """Dueling DQN with batch normalization."""
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
            super().__init__()
            
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            
            stream_dim = hidden_dim // 2
            self.value_fc1 = nn.Linear(hidden_dim, stream_dim)
            self.value_fc2 = nn.Linear(stream_dim, 1)
            self.advantage_fc1 = nn.Linear(hidden_dim, stream_dim)
            self.advantage_fc2 = nn.Linear(stream_dim, action_dim)
            
            self.dropout = nn.Dropout(0.1)
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.fc2(x)))
            
            value = F.relu(self.value_fc1(x))
            value = self.value_fc2(value)
            
            advantage = F.relu(self.advantage_fc1(x))
            advantage = self.advantage_fc2(advantage)
            
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values


class NumpyDuelingNetwork:
    """NumPy-based Dueling DQN for when PyTorch is unavailable."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Feature layers
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2 / state_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        # Value stream
        stream_dim = hidden_dim // 2
        self.W_v1 = np.random.randn(hidden_dim, stream_dim) * np.sqrt(2 / hidden_dim)
        self.b_v1 = np.zeros(stream_dim)
        self.W_v2 = np.random.randn(stream_dim, 1) * np.sqrt(2 / stream_dim)
        self.b_v2 = np.zeros(1)
        
        # Advantage stream
        self.W_a1 = np.random.randn(hidden_dim, stream_dim) * np.sqrt(2 / hidden_dim)
        self.b_a1 = np.zeros(stream_dim)
        self.W_a2 = np.random.randn(stream_dim, action_dim) * np.sqrt(2 / stream_dim)
        self.b_a2 = np.zeros(action_dim)
        
        # Batch norm running stats
        self.bn1_mean = np.zeros(hidden_dim)
        self.bn1_var = np.ones(hidden_dim)
        self.bn2_mean = np.zeros(hidden_dim)
        self.bn2_var = np.ones(hidden_dim)
        self.training = True
    
    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Layer 1
        z1 = np.dot(x, self.W1) + self.b1
        if self.training:
            mean, var = z1.mean(0), z1.var(0) + 1e-5
            self.bn1_mean = 0.9 * self.bn1_mean + 0.1 * mean
            self.bn1_var = 0.9 * self.bn1_var + 0.1 * var
            z1 = (z1 - mean) / np.sqrt(var)
        else:
            z1 = (z1 - self.bn1_mean) / np.sqrt(self.bn1_var + 1e-5)
        a1 = np.maximum(0, z1)
        
        # Layer 2
        z2 = np.dot(a1, self.W2) + self.b2
        if self.training:
            mean, var = z2.mean(0), z2.var(0) + 1e-5
            self.bn2_mean = 0.9 * self.bn2_mean + 0.1 * mean
            self.bn2_var = 0.9 * self.bn2_var + 0.1 * var
            z2 = (z2 - mean) / np.sqrt(var)
        else:
            z2 = (z2 - self.bn2_mean) / np.sqrt(self.bn2_var + 1e-5)
        a2 = np.maximum(0, z2)
        
        # Value stream
        v1 = np.maximum(0, np.dot(a2, self.W_v1) + self.b_v1)
        value = np.dot(v1, self.W_v2) + self.b_v2
        
        # Advantage stream
        a1_adv = np.maximum(0, np.dot(a2, self.W_a1) + self.b_a1)
        advantage = np.dot(a1_adv, self.W_a2) + self.b_a2
        
        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(axis=1, keepdims=True))
        return q_values
    
    def predict(self, x):
        self.training = False
        result = self.forward(x)
        self.training = True
        return result
    
    def backward(self, x, target, learning_rate=0.001):
        """Backward pass with gradient descent."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            target = target.reshape(1, -1)
        
        batch_size = x.shape[0]
        
        # Forward pass (storing intermediate values)
        # Layer 1
        z1 = np.dot(x, self.W1) + self.b1
        z1_norm = (z1 - z1.mean(0)) / np.sqrt(z1.var(0) + 1e-5)
        a1 = np.maximum(0, z1_norm)
        
        # Layer 2
        z2 = np.dot(a1, self.W2) + self.b2
        z2_norm = (z2 - z2.mean(0)) / np.sqrt(z2.var(0) + 1e-5)
        a2 = np.maximum(0, z2_norm)
        
        # Value stream
        z_v1 = np.dot(a2, self.W_v1) + self.b_v1
        a_v1 = np.maximum(0, z_v1)
        value = np.dot(a_v1, self.W_v2) + self.b_v2
        
        # Advantage stream
        z_a1 = np.dot(a2, self.W_a1) + self.b_a1
        a_a1 = np.maximum(0, z_a1)
        advantage = np.dot(a_a1, self.W_a2) + self.b_a2
        
        # Output
        output = value + (advantage - advantage.mean(axis=1, keepdims=True))
        
        # Compute loss
        loss = np.mean((output - target) ** 2)
        
        # Backward pass
        dL_dout = 2 * (output - target) / batch_size
        
        # Gradient for value and advantage
        # dL/dV_i = sum_a(dL/dQ_a)  — V shifts ALL Q values equally
        # dL/dA_j = dL/dQ_j - mean_a(dL/dQ_a)  — mean-centering removes the baseline
        dL_dV = dL_dout.sum(axis=1, keepdims=True)
        dL_dA = dL_dout - dL_dout.mean(axis=1, keepdims=True)
        
        # Value stream gradients
        dL_dW_v2 = np.dot(a_v1.T, dL_dV)
        dL_db_v2 = dL_dV.sum(axis=0)
        dL_da_v1 = np.dot(dL_dV, self.W_v2.T)
        dL_dz_v1 = dL_da_v1 * (z_v1 > 0).astype(float)
        dL_dW_v1 = np.dot(a2.T, dL_dz_v1)
        dL_db_v1 = dL_dz_v1.sum(axis=0)
        
        # Advantage stream gradients
        dL_dW_a2 = np.dot(a_a1.T, dL_dA)
        dL_db_a2 = dL_dA.sum(axis=0)
        dL_da_a1 = np.dot(dL_dA, self.W_a2.T)
        dL_dz_a1 = dL_da_a1 * (z_a1 > 0).astype(float)
        dL_dW_a1 = np.dot(a2.T, dL_dz_a1)
        dL_db_a1 = dL_dz_a1.sum(axis=0)
        
        # Backprop to shared layers
        dL_da2 = np.dot(dL_dz_v1, self.W_v1.T) + np.dot(dL_dz_a1, self.W_a1.T)
        dL_dz2 = dL_da2 * (z2_norm > 0).astype(float)
        dL_dW2 = np.dot(a1.T, dL_dz2)
        dL_db2 = dL_dz2.sum(axis=0)
        
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * (z1_norm > 0).astype(float)
        dL_dW1 = np.dot(x.T, dL_dz1)
        dL_db1 = dL_dz1.sum(axis=0)
        
        # Gradient clipping — both weights AND biases
        clip = 1.0
        for grad in [dL_dW1, dL_db1, dL_dW2, dL_db2,
                     dL_dW_v1, dL_db_v1, dL_dW_v2, dL_db_v2,
                     dL_dW_a1, dL_db_a1, dL_dW_a2, dL_db_a2]:
            np.clip(grad, -clip, clip, out=grad)
        
        # Update weights
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        
        self.W_v1 -= learning_rate * dL_dW_v1
        self.b_v1 -= learning_rate * dL_db_v1
        self.W_v2 -= learning_rate * dL_dW_v2
        self.b_v2 -= learning_rate * dL_db_v2.flatten()
        
        self.W_a1 -= learning_rate * dL_dW_a1
        self.b_a1 -= learning_rate * dL_db_a1
        self.W_a2 -= learning_rate * dL_dW_a2
        self.b_a2 -= learning_rate * dL_db_a2
        
        return loss
    
    def copy_weights_from(self, other):
        for attr in ['W1', 'b1', 'W2', 'b2', 'W_v1', 'b_v1', 'W_v2', 'b_v2',
                     'W_a1', 'b_a1', 'W_a2', 'b_a2', 'bn1_mean', 'bn1_var',
                     'bn2_mean', 'bn2_var']:
            setattr(self, attr, getattr(other, attr).copy())
    
    def get_weights(self):
        return {attr: getattr(self, attr).copy() for attr in 
                ['W1', 'b1', 'W2', 'b2', 'W_v1', 'b_v1', 'W_v2', 'b_v2',
                 'W_a1', 'b_a1', 'W_a2', 'b_a2', 'bn1_mean', 'bn1_var',
                 'bn2_mean', 'bn2_var']}
    
    def set_weights(self, weights):
        for k, v in weights.items():
            setattr(self, k, v.copy())


# =============================================================================
# PATHWAY RECOMMENDATION AGENT
# =============================================================================

class PathwayRecommendationAgent:
    """
    DQN Agent for recommending academic pathways.
    
    Includes all improvements:
    - Double DQN for stable Q-learning
    - Dueling architecture for better value estimation
    - Prioritized replay for efficient learning
    - Reward shaping for dense signal
    """
    
    def __init__(self,
                 state_dim: int = None,
                 action_dim: int = None,
                 hidden_dim: int = None,
                 hidden_sizes: List[int] = None,
                 learning_rate: float = None,
                 gamma: float = None,
                 epsilon_start: float = None,
                 epsilon_end: float = None,
                 epsilon_decay: float = None,
                 batch_size: int = None,
                 cql_alpha: float = None,
                 verbose: bool = True):
        
        self.state_dim = state_dim or RL_CONFIG['state_dim']
        self.action_dim = action_dim or RL_CONFIG['action_dim']
        
        # Support both hidden_dim (single) and hidden_sizes (list)
        if hidden_sizes:
            self.hidden_sizes = hidden_sizes
            self.hidden_dim = hidden_sizes[0]
        else:
            self.hidden_dim = hidden_dim or RL_CONFIG.get('hidden_dim', 128)
            self.hidden_sizes = [self.hidden_dim, self.hidden_dim // 2]
        
        self.learning_rate = learning_rate or RL_CONFIG['learning_rate']
        self.gamma = gamma or RL_CONFIG['gamma']
        self.epsilon = epsilon_start or RL_CONFIG['epsilon_start']
        self.epsilon_end = epsilon_end or RL_CONFIG['epsilon_end']
        self.epsilon_decay = epsilon_decay or RL_CONFIG['epsilon_decay']
        self.batch_size = batch_size or RL_CONFIG.get('batch_size', 32)
        
        # CQL-lite: conservative offline RL regularization.
        # Penalises high Q-values for actions absent in the replay buffer,
        # preventing OOD overestimation on novel student states at inference.
        self.cql_alpha = cql_alpha if cql_alpha is not None else RL_CONFIG.get('cql_alpha', 0.1)

        self.use_torch = TORCH_AVAILABLE
        self.competency_names = list(COMPETENCIES.keys())
        
        # Initialize networks
        if self.use_torch:
            self._init_torch_networks()
        else:
            self._init_numpy_networks()
        
        # Prioritized replay buffer — wire all PER hyperparameters from config
        self.memory = ReplayBuffer(
            RL_CONFIG.get('memory_size', 20000),
            alpha=RL_CONFIG.get('per_alpha', 0.6),
            beta_start=RL_CONFIG.get('per_beta_start', 0.4),
            beta_frames=RL_CONFIG.get('per_beta_frames', 2000),
        )
        self.train_step = 0
        self._verbose = verbose

        if verbose:
            print(f"  Agent initialized: {'PyTorch' if self.use_torch else 'NumPy'} backend")
            print(f"  Features: Double DQN, Dueling, PER, Reward Shaping")
    
    def _init_torch_networks(self):
        self.device = torch.device('cpu')  # extend to 'cuda' if GPU available
        self.q_network = DuelingDQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_network = DuelingDQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
    
    def _init_numpy_networks(self):
        self.q_network = NumpyDuelingNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.target_network = NumpyDuelingNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.target_network.copy_weights_from(self.q_network)
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        if self.use_torch:
            with torch.no_grad():
                self.q_network.eval()
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                self.q_network.train()
                return q_values.argmax().item()
        else:
            q_values = self.q_network.predict(state)
            return np.argmax(q_values[0])
    
    def shape_reward(self, state: np.ndarray, action: int, info: dict) -> float:
        """Apply reward shaping for dense learning signal (9-action space)."""
        if info.get('match', False):
            reward = 1.0
        elif info.get('pathway_match', False):
            reward = 0.3   # right pathway, wrong specific action
        else:
            reward = -0.5

        # Alignment bonus: cosine sim for the action's target pathway (indices 66-68 in 78-dim state)
        alignment = self._compute_alignment(state, action)
        reward += alignment * 0.3

        return np.clip(reward, -1.0, 1.5)

    def _compute_pathway_affinities(self, state: np.ndarray) -> Dict[str, float]:
        """Read preference one-hot already encoded in state (indices 66-68).
        Returns 1.0 for the student's preferred pathway, 0.0 otherwise.
        Used as a proxy for pathway affinity in reward shaping."""
        pw_names = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
        if len(state) >= 69:
            cos_sims = state[66:69]
            return {pw: float(cos_sims[i]) for i, pw in enumerate(pw_names)}
        # Fallback for short states
        affinities = {}
        for pathway, profile in IDEAL_PATHWAY_PROFILES.items():
            ideal = np.array([profile[c] for c in self.competency_names])
            s = state[:len(ideal)]
            sim = np.dot(s, ideal) / (np.linalg.norm(s) * np.linalg.norm(ideal) + 1e-8)
            affinities[pathway] = sim
        return affinities

    def _compute_alignment(self, state: np.ndarray, action: int) -> float:
        """Return preference alignment: 1.0 if action targets the student's preferred pathway, 0.0 otherwise."""
        from src.rl.dqn_coaching import ACTIONS
        action_obj = ACTIONS[action]
        if action_obj.target_pathway is None:
            return 0.3   # maintain/general actions are pathway-neutral
        pw_idx = {'STEM': 0, 'SOCIAL_SCIENCES': 1, 'ARTS_SPORTS': 2}
        idx = pw_idx.get(action_obj.target_pathway, -1)
        if idx < 0 or len(state) < 69:
            return 0.0
        return float(state[66 + idx])
    
    def update(self, batch_size: int = 64) -> float:
        if len(self.memory) < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size)
        
        if self.use_torch:
            loss, td_errors = self._torch_update(states, actions, rewards, next_states, dones, weights)
        else:
            loss, td_errors = self._numpy_update(states, actions, rewards, next_states, dones, weights)
        
        self.memory.update_priorities(indices, td_errors)
        self.train_step += 1
        return loss
    
    def _torch_update(self, states, actions, rewards, next_states, dones, weights):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Full Q-value matrix needed for CQL logsumexp term
        current_q_all = self.q_network(states)
        current_q = current_q_all.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            # Double DQN: online selects, target evaluates
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        td_errors = (current_q.squeeze() - target_q).detach().cpu().numpy()
        bellman_loss = (weights * F.smooth_l1_loss(current_q.squeeze(), target_q, reduction='none')).mean()

        # CQL-lite conservative regularization:
        # L_CQL = E_s[ log Σ_a exp(Q(s,a)) - Q(s, a_buffer) ]
        # Pushes down Q-values for actions not present in the buffer (OOD actions)
        # while keeping Q-values for observed actions grounded.
        cql_loss = (torch.logsumexp(current_q_all, dim=1) - current_q.squeeze()).mean()
        loss = bellman_loss + self.cql_alpha * cql_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), td_errors
    
    def _numpy_update(self, states, actions, rewards, next_states, dones, weights):
        current_q = self.q_network.forward(states)

        # Double DQN
        online_next = self.q_network.predict(next_states)
        next_actions = np.argmax(online_next, axis=1)
        target_next = self.target_network.predict(next_states)
        next_q = target_next[np.arange(len(next_actions)), next_actions]
        target_q = rewards + self.gamma * next_q * (1 - dones)

        td_errors = current_q[np.arange(len(actions)), actions] - target_q

        # Bellman target
        target = current_q.copy()
        for i, a in enumerate(actions):
            target[i, a] = target_q[i]

        # CQL-lite conservative regularization via effective-target modification.
        # The CQL gradient is: dL_CQL/dQ(s,a) = softmax(Q(s,a)) - 1[a = a_buffer]
        # For MSE loss where dL/dQ = 2*(Q - target)/N, we can absorb the CQL gradient
        # by shifting the effective target:
        #   target_eff = target - (cql_alpha / 2) * cql_grad
        # This pushes down Q-values for actions not in the buffer and pushes up
        # Q-values for the observed buffer action — exactly CQL's intent.
        if self.cql_alpha > 0:
            # Numerically stable softmax
            q_shifted = current_q - current_q.max(axis=1, keepdims=True)
            softmax_q = np.exp(q_shifted)
            softmax_q /= softmax_q.sum(axis=1, keepdims=True)

            cql_grad = softmax_q.copy()  # d/dQ [log Σ exp(Q)] = softmax(Q)
            for i, a in enumerate(actions):
                cql_grad[i, a] -= 1.0    # subtract indicator for buffer action

            target = target - (self.cql_alpha / 2.0) * cql_grad

        loss = self.q_network.backward(states, target, self.learning_rate)

        return loss, td_errors
    
    def update_target_network(self, soft: bool = True, tau: float = 0.01):
        if self.use_torch:
            if soft:
                for target_param, param in zip(self.target_network.parameters(), 
                                               self.q_network.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            else:
                self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            self.target_network.copy_weights_from(self.q_network)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, env, episodes: int = 500, verbose: bool = True,
              batch_per_episode: int = 64, early_stopping: bool = False,
              patience: int = 50, min_accuracy: float = 0.95) -> Dict:
        """
        Train the agent.
        
        Args:
            env: Training environment
            episodes: Number of training episodes (will train ALL episodes unless early_stopping=True)
            verbose: Print progress
            batch_per_episode: Students to sample per episode
            early_stopping: If True, stop when accuracy plateaus. Default: False (train full episodes)
            patience: Episodes without improvement before stopping (only if early_stopping=True)
            min_accuracy: Minimum accuracy before early stopping can trigger
        
        Returns:
            Training history dict
        """
        import time
        
        history = {'rewards': [], 'accuracy': [], 'epsilon': [], 'loss': []}
        best_accuracy = 0
        no_improve_count = 0
        start_time = time.time()
        
        n_students = len(env.students)
        batch_per_episode = min(batch_per_episode, n_students)
        
        if verbose:
            print(f"\n  Training DQN Agent")
            print(f"  Students: {n_students}, Batch/episode: {batch_per_episode}")
            print(f"  Episodes: {episodes}, Early stopping: {'ON' if early_stopping else 'OFF'}")
        
        for episode in range(episodes):
            sampled_students = np.random.choice(env.students, size=batch_per_episode, replace=False)
            
            total_reward = 0
            correct = 0
            episode_loss = 0
            
            for student_id in sampled_students:
                state = env.reset(student_id)
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                shaped_reward = self.shape_reward(state, action, info)
                self.memory.push(state, action, shaped_reward, next_state, done)
                
                total_reward += shaped_reward
                if info.get('match', False):
                    correct += 1
            
            # Multiple gradient updates per episode.
            # 8 updates × larger batch (128) = ~4× more learning signal per episode
            # compared to the previous 4 × 64 configuration.
            n_updates = RL_CONFIG.get('updates_per_episode', 8)
            for _ in range(n_updates):
                if len(self.memory) >= RL_CONFIG['batch_size']:
                    episode_loss += self.update(RL_CONFIG['batch_size'])
            
            self.decay_epsilon()
            
            if episode % 10 == 0:
                self.update_target_network(soft=True)
            
            accuracy = correct / batch_per_episode
            history['rewards'].append(total_reward / batch_per_episode)
            history['accuracy'].append(accuracy)
            history['epsilon'].append(self.epsilon)
            history['loss'].append(episode_loss / 4)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if verbose and (episode % 50 == 0 or episode == episodes - 1):
                elapsed = time.time() - start_time
                print(f"  Ep {episode:4d}/{episodes}: Acc={accuracy:.1%}, "
                      f"ε={self.epsilon:.3f} | {(episode+1)/elapsed:.1f} ep/s")
            
            if early_stopping and best_accuracy >= min_accuracy and no_improve_count >= patience:
                if verbose:
                    print(f"\n  ✓ Early stopping: {best_accuracy:.1%} accuracy")
                break
        
        if verbose:
            print(f"\n  Training complete in {time.time()-start_time:.1f}s")
            print(f"  Best training accuracy: {best_accuracy:.1%} (on batch samples)")
            print(f"  Note: Final accuracy on ALL students may differ")
        
        return history
    
    def recommend(self, state: np.ndarray) -> Dict:
        """Generate improvement recommendation with explanation (9-action space)."""
        from src.rl.dqn_coaching import ACTIONS
        if isinstance(state, list):
            state = np.array(state)

        q_values = self._get_q_values(state)

        q_exp = np.exp(q_values - np.max(q_values))
        confidence = q_exp / q_exp.sum()

        ranking_indices = np.argsort(q_values)[::-1]
        best_idx = int(ranking_indices[0])
        best_action = ACTIONS[best_idx]

        # Derive recommended pathway: from action target, or from G9 suitability in state
        if best_action.target_pathway:
            recommended_pathway = best_action.target_pathway
        else:
            # Actions 5 (maintain) and 8 (general): read G9 suitability at indices 51-53
            pw_names = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
            cw = state[51:54] if len(state) >= 54 else np.ones(3) / 3
            recommended_pathway = pw_names[int(np.argmax(cw))]

        return {
            'recommended_pathway': recommended_pathway,
            'recommended_action': best_action.name,
            'action_description': best_action.description,
            'target_subjects': best_action.target_subjects,
            'pathway_ranking': [ACTIONS[i].target_pathway or '' for i in ranking_indices],
            'action_ranking': [
                {'rank': r + 1, 'action': ACTIONS[i].description, 'q_value': float(q_values[i])}
                for r, i in enumerate(ranking_indices[:3])
            ],
            'q_values': {ACTIONS[i].name: float(q_values[i]) for i in range(len(q_values))},
            'confidence': float(confidence[best_idx]),
            'confidence_scores': {ACTIONS[i].name: float(confidence[i]) for i in range(len(confidence))},
            'reasoning': self._generate_reasoning(state, best_idx),
        }
    
    # =========================================================================
    # EXPLAINABLE AI (XAI) METHODS
    # =========================================================================
    
    def explain_recommendation(self, state: np.ndarray) -> dict:
        """
        Generate comprehensive XAI explanation for a recommendation.
        
        Includes:
        - Feature importance (perturbation-based)
        - Counterfactual analysis
        - Decision boundary analysis
        
        This provides TRUE model interpretation, not just template strings.
        """
        state = np.array(state, dtype=np.float32).flatten()
        
        # Get base recommendation
        rec = self.recommend(state)
        q_values = self._get_q_values(state)
        best_action = np.argmax(q_values)
        
        # 1. Perturbation-based Feature Importance
        feature_importance = self._compute_feature_importance(state, best_action)
        
        # 2. Counterfactual Analysis
        counterfactuals = self._compute_counterfactuals(state, best_action)
        
        # 3. Decision Margin (how close was the decision?)
        sorted_q = np.sort(q_values)[::-1]
        decision_margin = sorted_q[0] - sorted_q[1] if len(sorted_q) > 1 else 0
        
        # 4. Sensitivity Analysis (which competencies would flip the decision?)
        sensitivity = self._compute_sensitivity(state, best_action)
        
        return {
            'recommendation': rec,
            'feature_importance': feature_importance,
            'counterfactuals': counterfactuals,
            'decision_margin': float(decision_margin),
            'decision_confidence': 'high' if decision_margin > 0.5 else 'medium' if decision_margin > 0.2 else 'low',
            'sensitivity': sensitivity,
            'explanation_text': self._generate_xai_explanation(
                rec['recommended_pathway'], feature_importance, counterfactuals, decision_margin
            )
        }
    
    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state (always in eval mode — BatchNorm requires it for batch_size=1)."""
        state = np.array(state, dtype=np.float32).flatten()
        if self.use_torch:
            self.q_network.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
            self.q_network.train()
        else:
            q_values = self.q_network.predict(state)[0]
        return q_values
    
    def _compute_feature_importance(self, state: np.ndarray, action: int,
                                     perturbation: float = 0.1) -> dict:
        """
        Permutation-based feature importance over the 9 current subject scores
        (state[0:9]) and pathway metrics (state[63:69]).
        These are the most interpretable features for teachers.
        """
        from src.rl.dqn_coaching import STATE_SUBJECTS
        base_q = self._get_q_values(state)[action]
        importance = {}

        # Current subject scores (indices 0-8)
        for i, subj in enumerate(STATE_SUBJECTS):
            perturbed = state.copy()
            perturbed[i] = max(0.0, perturbed[i] - perturbation)
            new_q = self._get_q_values(perturbed)[action]
            importance[f'score_{subj}'] = float(base_q - new_q)

        # Cosine similarities (indices 66-68)
        for i, pw in enumerate(['STEM', 'SS', 'Arts']):
            idx = 66 + i
            if idx < len(state):
                perturbed = state.copy()
                perturbed[idx] = max(0.0, perturbed[idx] - perturbation)
                new_q = self._get_q_values(perturbed)[action]
                importance[f'cosine_{pw}'] = float(base_q - new_q)

        # Normalize to sum to 1
        total = sum(abs(v) for v in importance.values())
        if total > 0:
            importance = {k: abs(v) / total for k, v in importance.items()}

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def _compute_counterfactuals(self, state: np.ndarray, current_action: int) -> list:
        """
        Counterfactual explanations: what current-grade subject score changes
        would flip the recommendation to an alternative action?
        """
        from src.rl.dqn_coaching import ACTIONS, STATE_SUBJECTS
        q_values = self._get_q_values(state)
        counterfactuals = []

        for alt_action in range(self.action_dim):
            if alt_action == current_action:
                continue
            alt_action_obj = ACTIONS[alt_action]
            q_diff = float(q_values[current_action] - q_values[alt_action])
            changes_needed = {}

            # Perturb current subject scores (indices 0-8) to find what flips the decision
            for i, subj in enumerate(STATE_SUBJECTS):
                if state[i] < 0.95:
                    test_state = state.copy()
                    for increase in [0.1, 0.2, 0.3, 0.5]:
                        test_state[i] = min(1.0, state[i] + increase)
                        if np.argmax(self._get_q_values(test_state)) == alt_action:
                            changes_needed[subj] = f"+{increase * 100:.0f}%"
                            break

            counterfactuals.append({
                'alternative_action': alt_action_obj.name,
                'alternative_pathway': alt_action_obj.target_pathway or '(none)',
                'q_value_gap': q_diff,
                'changes_needed': changes_needed if changes_needed else "Significant changes required",
                'feasibility': 'easy' if q_diff < 0.3 else 'moderate' if q_diff < 0.7 else 'difficult',
            })

        return counterfactuals

    def _compute_sensitivity(self, state: np.ndarray, action: int) -> dict:
        """Which current subject scores, if reduced, would flip the recommendation?"""
        from src.rl.dqn_coaching import ACTIONS, STATE_SUBJECTS
        sensitivity = {}

        for i, subj in enumerate(STATE_SUBJECTS):
            for reduction in [0.1, 0.2, 0.3]:
                test_state = state.copy()
                test_state[i] = max(0.0, state[i] - reduction)
                new_action = int(np.argmax(self._get_q_values(test_state)))
                if new_action != action:
                    sensitivity[subj] = {
                        'threshold': f"-{reduction * 100:.0f}%",
                        'would_switch_to': ACTIONS[new_action].name,
                    }
                    break

        return sensitivity
    
    def _generate_xai_explanation(self, pathway: str, importance: dict,
                                   counterfactuals: list, margin: float) -> str:
        """Generate human-readable XAI explanation for 78-dim state."""
        top_features = list(importance.keys())[:3]

        # Feature names are like 'score_MATH', 'cosine_STEM' — make them readable
        def _readable(f):
            return f.replace('score_', '').replace('cosine_', 'affinity with ').replace('_', ' ')

        explanation = f"The {PATHWAYS.get(pathway, {}).get('name', pathway)} pathway was recommended "
        explanation += f"primarily based on {_readable(top_features[0])} "
        explanation += f"({importance[top_features[0]]:.0%} importance)"
        if len(top_features) > 1:
            explanation += f" and {_readable(top_features[1])} ({importance[top_features[1]]:.0%} importance)"
        explanation += ". "

        if margin > 0.5:
            explanation += "This is a confident recommendation. "
        elif margin > 0.2:
            explanation += "This recommendation has moderate confidence. "
        else:
            explanation += "This is a close decision; alternative actions are also suitable. "

        easiest_alt = next((c for c in counterfactuals if c['feasibility'] == 'easy'), None)
        if easiest_alt and isinstance(easiest_alt['changes_needed'], dict):
            changes = list(easiest_alt['changes_needed'].items())
            if changes:
                subj, change = changes[0]
                explanation += (f"To switch to '{easiest_alt['alternative_action']}', "
                                f"{subj} score would need to increase by {change}.")

        return explanation

    def _generate_reasoning(self, state: np.ndarray, action_idx: int) -> str:
        """Brief reasoning based on the recommended action and top current subject scores."""
        from src.rl.dqn_coaching import ACTIONS, STATE_SUBJECTS
        action = ACTIONS[action_idx]
        # Top 3 current subject scores (indices 0-8)
        curr_scores = state[:len(STATE_SUBJECTS)]
        top_idxs = np.argsort(curr_scores)[::-1][:2]
        top_subjects = [STATE_SUBJECTS[i] for i in top_idxs]
        return (f"{action.description}. "
                f"Student shows strongest performance in: {', '.join(top_subjects)}.")
    
    def validate_backends(self, n_states: int = 100, seed: int = 42) -> Dict:
        """
        Quantify the output divergence between the PyTorch and NumPy backends
        using identical network weights and identical random input states.

        Only meaningful when use_torch=True; if PyTorch is unavailable the
        NumPy backend is the sole code path and there is nothing to compare.

        The primary divergence source is BatchNorm: PyTorch's BN has learned
        scale (gamma) and shift (beta) parameters that the NumPy BN does not
        apply.  This function copies all non-BN weights exactly and transfers
        BN running-mean/var so that normalisation statistics match; the
        remaining divergence therefore isolates the gamma/beta contribution.

        Returns:
            Dict with mean/max L2 divergence, action-level agreement rate,
            and a 'status' key ('ok' / 'warning' / 'skipped').
        """
        if not TORCH_AVAILABLE or not self.use_torch:
            return {'status': 'skipped', 'reason': 'PyTorch backend not active'}

        np.random.seed(seed)
        test_states = np.random.rand(n_states, self.state_dim).astype(np.float32)

        # --- PyTorch outputs ---
        torch_q = np.zeros((n_states, self.action_dim), dtype=np.float32)
        self.q_network.eval()
        with torch.no_grad():
            batch = torch.FloatTensor(test_states).to(self.device)
            torch_q = self.q_network(batch).cpu().numpy()
        self.q_network.train()

        # --- Build a NumPy network with matched weights ---
        numpy_net = NumpyDuelingNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        sd = {k: v.cpu().numpy() for k, v in self.q_network.state_dict().items()}

        # Linear weights: PyTorch stores (out_features, in_features) → transpose for NumPy (in, out)
        numpy_net.W1   = sd['fc1.weight'].T
        numpy_net.b1   = sd['fc1.bias']
        numpy_net.W2   = sd['fc2.weight'].T
        numpy_net.b2   = sd['fc2.bias']
        numpy_net.W_v1 = sd['value_fc1.weight'].T
        numpy_net.b_v1 = sd['value_fc1.bias']
        numpy_net.W_v2 = sd['value_fc2.weight'].T
        numpy_net.b_v2 = sd['value_fc2.bias']
        numpy_net.W_a1 = sd['advantage_fc1.weight'].T
        numpy_net.b_a1 = sd['advantage_fc1.bias']
        numpy_net.W_a2 = sd['advantage_fc2.weight'].T
        numpy_net.b_a2 = sd['advantage_fc2.bias']

        # BN running statistics (matched; learned gamma/beta are NOT applied in NumPy)
        numpy_net.bn1_mean = sd['bn1.running_mean']
        numpy_net.bn1_var  = sd['bn1.running_var']
        numpy_net.bn2_mean = sd['bn2.running_mean']
        numpy_net.bn2_var  = sd['bn2.running_var']

        # --- NumPy outputs ---
        numpy_q = np.array([numpy_net.predict(s)[0] for s in test_states])

        # --- Divergence metrics ---
        l2 = np.linalg.norm(torch_q - numpy_q, axis=1)
        torch_actions  = np.argmax(torch_q,  axis=1)
        numpy_actions  = np.argmax(numpy_q,  axis=1)
        action_agree   = float(np.mean(torch_actions == numpy_actions))

        status = 'ok' if l2.mean() < 0.05 else 'warning'

        return {
            'status': status,
            'mean_l2_divergence':  float(l2.mean()),
            'max_l2_divergence':   float(l2.max()),
            'std_l2_divergence':   float(l2.std()),
            'action_agreement':    action_agree,
            'n_states':            n_states,
            'note': (
                'Residual divergence after weight-matching is caused by PyTorch BN '
                'learned scale/shift (gamma/beta) parameters absent in the NumPy backend. '
                'Action agreement is the practically important metric.'
            ),
        }

    def save(self, path: Path = None):
        if path is None:
            path = MODELS_DIR / 'dqn_pathway_model'
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.use_torch:
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'epsilon': self.epsilon
            }, str(path) + '.pt')
        else:
            np.save(str(path) + '.npy', {
                'q_network': self.q_network.get_weights(),
                'epsilon': self.epsilon
            })
        print(f"  Model saved to {path}")
    
    def load(self, path: Path = None):
        if path is None:
            path = MODELS_DIR / 'dqn_pathway_model'
        path = Path(path)
        
        if self.use_torch:
            checkpoint = torch.load(str(path) + '.pt', weights_only=False)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
        else:
            data = np.load(str(path) + '.npy', allow_pickle=True).item()
            self.q_network.set_weights(data['q_network'])
            self.target_network.copy_weights_from(self.q_network)
            self.epsilon = data.get('epsilon', self.epsilon)
        print(f"  Model loaded from {path}")
