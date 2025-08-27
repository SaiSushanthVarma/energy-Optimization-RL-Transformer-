
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        
    def forward(self, x):
        return self.network(x)

class DQN:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=100000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.batch_size = 1024  # Store batch size as class attribute
        
        # Networks
        self.model = DQNetwork(state_size, action_size).to(device)
        self.target_model = DQNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize target network
        self.update_target_model()

    def remember(self, state, action, reward, next_state, done):
        # Ensure state is 1D and contains exactly state_size elements
        if isinstance(state, np.ndarray):
            state = state.flatten()[:self.state_size].astype(np.float32)
        if next_state is not None and isinstance(next_state, np.ndarray):
            next_state = next_state.flatten()[:self.state_size].astype(np.float32)
        
        # Ensure action is within bounds
        action = int(action) % self.action_size
        
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Ensure state is proper shape
        if isinstance(state, np.ndarray):
            state = state.flatten()[:self.state_size]
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def replay(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return None
            
        try:
            # Sample batch
            minibatch = random.sample(self.memory, batch_size)
            
            # Pre-allocate arrays
            states = np.zeros((batch_size, self.state_size), dtype=np.float32)
            next_states = np.zeros((batch_size, self.state_size), dtype=np.float32)
            actions = np.zeros(batch_size, dtype=np.int64)
            rewards = np.zeros(batch_size, dtype=np.float32)
            dones = np.zeros(batch_size, dtype=np.float32)
            
            # Fill arrays
            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                states[i] = state
                actions[i] = action
                rewards[i] = reward
                if next_state is not None:
                    next_states[i] = next_state
                dones[i] = float(done)
            
            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            # Current Q values
            current_q_values = self.model(states)
            current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Target Q values
            with torch.no_grad():
                next_q_values = self.target_model(next_states)
                max_next_q = next_q_values.max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
            # Compute loss and optimize
            loss = nn.MSELoss()(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            return loss.item()
            
        except Exception as e:
            print(f"Error in replay: {str(e)}")
            print(f"States shape: {states.shape if 'states' in locals() else 'N/A'}")
            print(f"Actions shape: {actions.shape if 'actions' in locals() else 'N/A'}")
            print(f"Actions unique values: {torch.unique(actions) if 'actions' in locals() else 'N/A'}")
            return None

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']