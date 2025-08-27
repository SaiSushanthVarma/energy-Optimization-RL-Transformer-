import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class RewardWeights:
    temperature_deviation: float = 1.0
    pt_change_penalty: float = 0.3
    comfort_zone_reward: float = 2.0
    extreme_pt_penalty: float = 0.5

class PTEnvironment:
    def __init__(self, model, device, prediction_length=6, desired_avg_value=21, comfort_range=1.0):
        self.model = model
        self.device = device
        self.prediction_length = prediction_length
        self.desired_avg_value = desired_avg_value
        self.comfort_range = comfort_range
        self.weights = RewardWeights()
        
        # Define PT constraints
        self.min_pt = 20
        self.max_pt = 70
        self.optimal_pt_range = (35, 55)  # Define an optimal operating range
        
        # Action space: More granular changes
        self.actions = np.linspace(-2, 2, 9)  # [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
    def calculate_reward(self, predicted_temps, current_pt, new_pt, previous_pt=None):
        """
        Multi-component reward function
        """
        rewards = []
        
        # Temperature deviation component
        temp_deviations = [-abs(temp - self.desired_avg_value) for temp in predicted_temps]
        temp_reward = np.mean(temp_deviations) * self.weights.temperature_deviation
        rewards.append(temp_reward)
        
        # PT change penalty (encourage smoother control)
        if previous_pt is not None:
            pt_change = abs(new_pt - previous_pt)
            pt_change_reward = -pt_change * self.weights.pt_change_penalty
            rewards.append(pt_change_reward)
        
        # Comfort zone reward
        temps_in_comfort = sum(abs(temp - self.desired_avg_value) <= self.comfort_range 
                             for temp in predicted_temps)
        comfort_reward = (temps_in_comfort / len(predicted_temps)) * self.weights.comfort_zone_reward
        rewards.append(comfort_reward)
        
        # Extreme PT penalty
        if new_pt <= self.min_pt + 5 or new_pt >= self.max_pt - 5:
            rewards.append(-self.weights.extreme_pt_penalty)
        
        # Optimal PT range reward
        if self.optimal_pt_range[0] <= new_pt <= self.optimal_pt_range[1]:
            rewards.append(0.5)
            
        # Combine all rewards
        total_reward = sum(rewards)
        
        # Clip reward to prevent extreme values
        return np.clip(total_reward, -10, 10)

    def step(self, action_idx, state, historical_pts=None):
        """
        Takes an action and returns next state, reward, and done flag.
        """
        action = self.actions[action_idx]
        current_pt = state[:, -1, -1].item()  # Get current PT value
        
        # Apply action with constraints
        new_pt = np.clip(current_pt + action, self.min_pt, self.max_pt)
        
        # Create copy of state and update PT
        next_state = state.clone()
        next_state[:, -1, -1] = new_pt
        
        # Predict next temperatures using the transformer model
        with torch.no_grad():
            seq_len = next_state.size(1)
            look_ahead_mask = torch.ones((1, 1, seq_len, seq_len), device=self.device)
            predictions, _ = self.model(next_state, look_ahead_mask, None)
            
        predicted_temps = predictions[:, :, 0].cpu().numpy().squeeze()
        
        # Calculate reward using the enhanced reward function
        previous_pt = historical_pts[-1] if historical_pts is not None else None
        reward = self.calculate_reward(predicted_temps, current_pt, new_pt, previous_pt)
        
        done = False
        
        return next_state, reward, done, {"predicted_temps": predicted_temps, "new_pt": new_pt}

    def reset(self, initial_state):
        """Reset the environment with initial state."""
        self.state = initial_state.to(self.device)
        return self.state