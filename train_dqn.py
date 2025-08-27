import torch
from dqn_agent import DQN
from rl_environment import PTEnvironment
from transformer_env import extract_hyperparams_and_epochs, load_transformer_model, create_dataset_for_rl, test_in_env
import json
import numpy as np
import os
import pickle
import datetime
import warnings
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


warnings.filterwarnings("ignore", category=UserWarning)



# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_data_path = 'Sweden_Finland_test_cuda_1'
norm_params_path = 'Sweden_Finland_test_cuda_1'  # Base path for normalization parameters
output_path = 'test_sweden_finland_results_on_test_data'  
agent_save_path = 'saved_agent_new_oct' 
os.makedirs(agent_save_path, exist_ok=True) 
episodes_output_path = 'episodes_data_new_oct'
os.makedirs(episodes_output_path, exist_ok=True)

model_params_path = 'model_training_ecoguard_fidelix_27th_June_cuda_1/model_params/model_params_iteration_77.json'
model_state_path = 'model_training_ecoguard_fidelix_27th_June_cuda_1/saved_models/best_model_iteration_77.pth'

# Load the transformer model parameters and model
with open(model_params_path, 'r') as file:
    model_params = json.load(file)

with open('model_training_ecoguard_fidelix_27th_June_cuda_1/iteration_details/iteration_details_77.txt', 'r') as file:
    iteration_details_text = file.read()

model_params_emb, model_epochs = extract_hyperparams_and_epochs(iteration_details_text)
trained_model = load_transformer_model(model_params, model_params_emb, model_state_path)

# Initialize environment
env = PTEnvironment(model=trained_model, device=device, prediction_length=6, desired_avg_value=21)

# Initialize the DQN agent
state_size = 9  # 'Average Value', 'OT', 'PT'
action_size = 5  # PT range from -10 to +10
agent = DQN(state_size=state_size, action_size=action_size, device=device)

# Prepare data for RL
test_data_embedded, test_data, building_id = create_dataset_for_rl(test_data_path)

# Load normalization parameters
norm_params_filename = f'norm_params_building_{building_id}.0.pkl'
with open(os.path.join(norm_params_path, norm_params_filename), 'rb') as file:
    mean, std = pickle.load(file)
mean_avg_value = mean['Average Value']
std_avg_value = std['Average Value']
mean_ot_value = mean['OT']
std_ot_value = std['OT']
mean_PT_value = mean['PT']
std_PT_value = std['PT']





# Denormalize initial dataset
test_data_embedded.loc[test_data_embedded.index, 'Average Value'] = test_data_embedded['Average Value'] * std_avg_value + mean_avg_value
test_data_embedded.loc[test_data_embedded.index, 'OT'] = test_data_embedded['OT'] * std_ot_value + mean_ot_value
test_data_embedded.loc[test_data_embedded.index, 'PT'] = test_data_embedded['PT'] * std_PT_value + mean_PT_value

# Training loop
n_episodes = 1000
batch_size = 64

def normalize_columns(df, mean_avg_value, std_avg_value, mean_ot_value, std_ot_value, mean_pt_value, std_pt_value):
    df.loc[df.index, 'Average Value'] = (df['Average Value'] - mean_avg_value) / std_avg_value
    df.loc[df.index, 'OT'] = (df['OT'] - mean_ot_value) / std_ot_value
    df.loc[df.index, 'PT'] = (df['PT'] - mean_pt_value) / std_pt_value
    return df

def calculate_reward(predicted_avg_value, desired_avg_value=21):
    reward = -abs(predicted_avg_value - desired_avg_value)
    return reward

def extract_relevant_columns(test_data_embedded):
    relevant_columns = ['Average Value', 'OT', 'PT']
    return test_data_embedded[relevant_columns]

def replace_pt_values(test_data_embedded, agent, initial_state, device, start_index):
    # Get 48 steps as input for the model
    state = extract_relevant_columns(test_data_embedded).iloc[start_index:start_index + 48].values
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    last_pt_value = state_tensor[0, -1, 2].item()  # Last PT value from 48th step
    predicted_pts = []  # Predicted PT values for the next 6 steps
    actions_taken = []

    # Predict PT for the next 6 steps
    for i in range(6):
        # Extract current state
        current_state = state_tensor[:, -1, :].cpu().numpy()

        # Get future OT values (make sure to reshape to match dimensions)
        future_ot_values = extract_relevant_columns(test_data_embedded).iloc[start_index + 48: start_index + 48 + 6]['OT'].values
        future_ot_values = np.expand_dims(future_ot_values, axis=0)  # Reshape to 2D array

        # Combine current state and future OT values into a 9-dimensional state
        augmented_state = np.concatenate([current_state.flatten(), future_ot_values.flatten()])

        # Let the agent act based on the augmented state
        action_index = agent.act(augmented_state)
        action = action_index - 2  # Convert action index to PT range [-4, 4]
        last_pt_value += action  # Update PT based on action
        last_pt_value = np.clip(last_pt_value, 20, 70)
        predicted_pts.append(last_pt_value)  # Store the new PT value
        actions_taken.append(action)

        # Update state with new PT value
        state_tensor[:, -1, 2] = last_pt_value
        state_tensor = state_tensor.roll(shifts=-1, dims=1)

    test_data_embedded.iloc[start_index + 48:start_index + 54, test_data_embedded.columns.get_loc('PT')] = np.array(predicted_pts)
    return test_data_embedded, actions_taken, predicted_pts




def denormalize_values(predicted_values, mean_value, std_value):
    return (predicted_values * std_value) + mean_value

def save_episode_data(test_data_embedded, episode, avg_values, pt_values, actions ,episode_number, output_path):
    episode_data = {
        'episode': episode,
        'data' : test_data_embedded,
        'denormalized_avg_values': avg_values,
        'predicted_pt_values': pt_values,
        'actions_taken': actions
    }
    # Save to pickle file
    episode_filename = os.path.join(output_path, f'episode_{episode_number}_data.pkl')
    with open(episode_filename, 'wb') as f:
        pickle.dump(episode_data, f)

n_steps = len(test_data_embedded)


for e in range(n_episodes):
    print(f"Starting episode {e + 1}/{n_episodes}")
    all_denormalized_avg_values = []  # Accumulate all predicted Average Values in the episode
    all_predicted_pts = [] 
    all_actions_taken = []  # Accumulate all actions taken by the agent in the episode
    episode_data = []
    initial_index = 0  # Reset for each episode

    total_windows = (n_steps - 48) // 6  # Total number of windows
    progress_bar = tqdm(total=total_windows, desc=f'Episode {e}', unit='window')

    while initial_index + 48 + 6 <= n_steps:
        initial_state = test_data_embedded.iloc[initial_index:initial_index + 48]

        # Get timestamps
        timestamps = test_data_embedded.index[initial_index:initial_index + 48].tolist()

        # Replace PT values using agent
        test_data_embedded_new, actions_taken, predicted_pts = replace_pt_values(test_data_embedded, agent, initial_state, device, initial_index)

        # Extract the next 48 steps, including predicted PT
        env_input_data = test_data_embedded_new.iloc[initial_index + 6:initial_index + 54]

        # Normalize the input data for the transformer model
        env_input_data = normalize_columns(env_input_data, mean_avg_value, std_avg_value, mean_ot_value, std_ot_value, mean_PT_value, std_PT_value)

        # Predict average values using transformer model
        predicted_avg_value = test_in_env(env_input_data, test_data, building_id, trained_model)

        # Denormalize predicted average values
        denormalized_avg_value = denormalize_values(predicted_avg_value, mean_avg_value, std_avg_value)

        all_denormalized_avg_values.append(denormalized_avg_value.tolist())
        all_predicted_pts.append(predicted_pts)
        all_actions_taken.append(actions_taken)

        # Update the predicted average values back into the dataset for future windows
        test_data_embedded.iloc[initial_index + 48:initial_index + 54, test_data_embedded_new.columns.get_loc('Average Value')] = denormalized_avg_value

        # Calculate reward
        reward = calculate_reward(denormalized_avg_value)

        done = initial_index + 48 + 6 >= n_steps 

        # Combine current state and future OT for the agent's state
        current_state = test_data_embedded.iloc[initial_index + 47][['Average Value', 'OT', 'PT']]
        future_ot_values = test_data_embedded.iloc[initial_index + 48: initial_index + 48 + 6]['OT'].values

        # If fewer than 6 OT values are available, pad with the last known OT value
        if len(future_ot_values) < 6:
            padding = np.full((6 - len(future_ot_values),), future_ot_values[-1])  # Pad with last known OT value
            future_ot_values = np.concatenate([future_ot_values, padding])

        # Combine the current state and future OT values
        combined_state = np.concatenate([current_state.values.flatten(), future_ot_values.flatten()])

        # Calculate next state (next 48-step window)
        if initial_index + 54 < n_steps:
            next_state_current = test_data_embedded.iloc[initial_index + 53][['Average Value', 'OT', 'PT']]
            next_state_future_ot = test_data_embedded.iloc[initial_index + 54:initial_index + 54 + 6]['OT'].values

            # If fewer than 6 future OT values are available, pad with the last known OT value
            if len(next_state_future_ot) < 6:
                padding = np.full((6 - len(next_state_future_ot),), next_state_future_ot[-1])  # Pad with last known OT value
                next_state_future_ot = np.concatenate([next_state_future_ot, padding])

            next_combined_state = np.concatenate([next_state_current.values.flatten(), next_state_future_ot.flatten()])
        else:
            next_combined_state = None  # Episode done, no next state

        # Store experience for the agent (state, action, reward, next_state, done)
        agent.remember(combined_state, actions_taken, reward, next_combined_state, done)

        # Train the agent with experience replay
        agent.replay(batch_size)

        # Save episode data
        episode_data.append({
            'episode': e,
            'initial_state': initial_state.to_dict(),
            'predicted_avg_value': denormalized_avg_value.tolist(),
            'predicted_pt_actions': actions_taken,
            'predicted_pt_values': predicted_pts,
            'reward': reward
        })

        episode_filename = os.path.join(episodes_output_path, f'episode_{e}_data.pkl')
        with open(episode_filename, 'wb') as f:
            pickle.dump(episode_data, f)

        output_filename = os.path.join(episodes_output_path, f'test_data_embedded_episode_{e}.csv')
        test_data_embedded_new.to_csv(output_filename, index=False)

        progress_bar.update(1)

        # Shift window by 6 steps
        initial_index += 6

    progress_bar.close()

    # Save episode data at the end of each episode
    save_episode_data(test_data_embedded, test_data_embedded_new, all_denormalized_avg_values, all_predicted_pts, all_actions_taken, e, episodes_output_path)

    # Update the target model at the end of each episode
    agent.update_target_model()
    agent_filename = os.path.join(agent_save_path, f'dqn_agent_episode_{e}.pth') 
    torch.save(agent.model.state_dict(), agent_filename)

    print(f"Episode {e} completed")
