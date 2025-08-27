import torch
import pandas as pd
import numpy as np
import json
import os
from pyexpat import model
import torch.nn as nn
import torch 
import math
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from numpy import array, sqrt, array
#from preprocessing import df
import pickle
#import preprocessing
import evaluation_metrics
import os 
import optuna
from torch.utils.data import DataLoader, TensorDataset       
from tqdm import tqdm
import wandb
import multiprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from test_imports_sweden_finland import TransformerWithEmbeddings
from test_imports_sweden_finland import create_temporal_positional_encoding
import re

# Define the device for model execution
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


categories = ['City', 'Region', 'Rise', 'ADI', 'Floors', 'Building Type']

input_length = 24

# Load the trained model
#model_params_path = 'model_training_artifacts_6th_jan/model_params/model_params_iteration_21.json'
#model_state_path = 'model_training_artifacts_6th_jan/saved_models/best_model_iteration_21.pth'
#categorical_mappings_path = 'model_training_artifacts_6th_jan/categorical_mappings/'
#embeddings_path = 'model_training_artifacts_6th_jan/final_embeddings/'
#iteration_details = 'model_training_artifacts_6th_jan/iteration_details/iteration_details_21.txt'


model_params_path = 'model_training_ecoguard_fidelix_27th_June_cuda_1/model_params/model_params_iteration_77.json'
model_state_path = 'model_training_ecoguard_fidelix_27th_June_cuda_1/saved_models/best_model_iteration_77.pth'
categorical_mappings_path = 'model_training_ecoguard_fidelix_27th_June_cuda_1/categorical_mappings/'
embeddings_path = 'model_training_ecoguard_fidelix_27th_June_cuda_1/final_embeddings/'
iteration_details = 'model_training_ecoguard_fidelix_27th_June_cuda_1/iteration_details/iteration_details_77.txt'


with open(model_params_path, 'r') as file:
    model_params = json.load(file)

with open(iteration_details, 'r') as file:
    iteration_details_text = file.read()

def extract_hyperparams_and_epochs(iteration_details_text):
    # Extract hyperparameters
    hyperparameters_match = re.search(r'Hyperparameters: ({.*?})', iteration_details_text)
    hyperparameters = json.loads(hyperparameters_match.group(1)) if hyperparameters_match else None
    
    # Count epochs
    epochs = iteration_details_text.count('Training Loss:')  # Each 'Training Loss:' entry corresponds to one epoch

    return hyperparameters, epochs

model_params_emb, model_epochs = extract_hyperparams_and_epochs(iteration_details_text)

def load_transformer_model(model_params, model_params_emb, model_state_path):
    model = TransformerWithEmbeddings(
        num_layers=model_params["num_layers"],
        D=model_params["D"],
        H=model_params["H"],
        hidden_mlp_dim=model_params["hidden_mlp_dim"],
        inp_features=model_params["inp_features"],
        out_features=model_params["out_features"],
        dropout_rate=model_params["dropout_rate"],
        prediction_length=model_params["prediction_length"],
        input_columns=model_params["input_columns"],
        embedding_size_factor=model_params_emb["embedding_size_factor"]
    ).to(device)


    model.load_state_dict(torch.load(model_state_path, map_location=device))
    model.eval()
    return model
# Load categorical mappings

mappings = {category: json.load(open(os.path.join(categorical_mappings_path, f'{category}_mapping.json'), 'r')) for category in categories}

# Load embeddings

embeddings = {cat: torch.load(os.path.join(embeddings_path, f'final_embeddings_{cat}_iteration_77.pt')) for cat in categories}

def replace_with_embeddings(data, category_embeddings, mappings):
    original_shape = data.shape
    #print(f"Original data shape: {original_shape}")

    # Create a copy of the data to avoid modifying the original dataframe
    embedded_data = data.copy()

    # Iterate over each category and append its embeddings
    for category in category_embeddings:
        if category in data.columns:
            # Ensure the column is a Series
            category_series = data[category]

            # Check if the column is a Series or still a DataFrame, then convert accordingly
            if isinstance(category_series, pd.DataFrame):
                category_series = category_series.iloc[:, 0]

            # Ensure the series contains string values for mapping
            category_series = category_series.astype(str)

            # Map the category values to their indices
            category_indices = category_series.map(lambda x: mappings[category].get(x, 0)).astype(int)

            # Retrieve embeddings using the indices
            embedded_values = category_embeddings[category][category_indices]

            # Convert embeddings to DataFrame and concatenate
            embedded_df = pd.DataFrame(embedded_values, index=embedded_data.index)
            embedded_data = pd.concat([embedded_data, embedded_df], axis=1)

            #print(f"Data shape after embedding '{category}': {embedded_data.shape}")

    return embedded_data


def reshape_data(data, input_length, prediction_length):
    # Ensure the data is on the CPU before conversion to numpy array
    if data.is_cuda:
        data = data.cpu()
    
    sequences = []
    data = data.numpy()  # Convert tensor to numpy array
    for i in range(len(data) - input_length - prediction_length + 1):
        sequence = data[i:i + input_length]
        sequences.append(sequence)
    return np.array(sequences)

def create_look_ahead_mask(size, prediction_length, device=None):
    mask = torch.ones((size, size), device=device)
    mask = torch.triu(mask, diagonal=prediction_length)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

# Load and prepare test data

test_data_path = 'Sweden_Finland_test_cuda_1'
norm_params_path = 'Sweden_Finland_test_cuda_1'  # Base path for normalization parameters
output_path = 'test_sweden_finland_results_on_test_data'  # Replace with your actual output path

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
#test_building_id = 33692.0
#test_data = pd.read_pickle(os.path.join(test_data_path, f'test_data_building_{test_building_id}.pkl'))
#
#test_data['Building_id_duplicate'] = test_data['Building_id']


#print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R-squared: {r2}")

def create_dataset_for_rl_new(test_data_path):
    """Load and prepare dataset for multiple buildings for RL training"""
    building_data = {}
    processed_count = 0
    max_buildings = 5  # Limit number of buildings to process

    try:
        print("Starting to load building data...")
        
        for test_file in os.listdir(test_data_path):
            if test_file.endswith('_rl_data.pkl'):
                # Extract the building ID from the file name
                building_id = float(test_file.split('_')[1])
                print(f"\nProcessing building {building_id}")

                # Load test data
                file_path = os.path.join(test_data_path, test_file)
                test_data = pd.read_pickle(file_path)

                if test_data.empty:
                    print(f"Skipping building {building_id} due to no data.")
                    continue

                # Add Building_id_duplicate column
                test_data['Building_id_duplicate'] = test_data['Building_id']

                try:
                    # Select required columns
                    input_columns_with_duplicate = model_params["input_columns"] + ['Building_id_duplicate']
                    test_data = test_data[input_columns_with_duplicate]

                    # Create embeddings
                    test_data_embedded = replace_with_embeddings(test_data, embeddings, mappings)
                    
                    # Store the processed data
                    building_data[building_id] = {
                        'embedded_data': test_data_embedded,
                        'original_data': test_data,
                        'building_idx': processed_count
                    }
                    
                    print(f"Successfully processed building {building_id}")
                    print(f"Data shape: {test_data_embedded.shape}")
                    
                    processed_count += 1
                    if processed_count >= max_buildings:
                        print(f"\nReached maximum number of buildings ({max_buildings})")
                        break
                        
                except Exception as e:
                    print(f"Error processing building {building_id}: {str(e)}")
                    continue

        if not building_data:
            print("No valid buildings were processed.")
            return None

        print(f"\nSuccessfully loaded {len(building_data)} buildings")
        return building_data

    except Exception as e:
        print(f"Error in create_dataset_for_rl: {str(e)}")
        return None


def create_dataset_for_rl(test_data_path):

    for test_file in os.listdir(test_data_path):
        if test_file.startswith('test_data_building_') and test_file.endswith('.pkl'):
            # Extract the building ID from the file name
            building_id = int(test_file.split('_')[-1].split('.')[0])

            #test_data = pd.read_pickle(os.path.join(test_data_path, f'test_data_building_{test_building_id}.pkl'))
            test_data = pd.read_pickle(os.path.join(test_data_path, test_file))

            if test_data.empty:
                print(f"Skipping building {building_id} due to no data.")
                continue  # Skip the rest of the code and move to the next file
            


            test_data['Building_id_duplicate'] = test_data['Building_id']


            input_columns_with_duplicate = model_params["input_columns"] + ['Building_id_duplicate']
            test_data = test_data[input_columns_with_duplicate]

            test_data_embedded = replace_with_embeddings(test_data, embeddings, mappings)

    return test_data_embedded, test_data, building_id

def test_in_env(test_data_embedded, test_data, building_id, model):
    """
    Predicts 6 steps based on the input sequence.
    Returns normalized predictions.
    """
    input_length = 48
    prediction_length = 6

    # Ensure data is in correct format
    if isinstance(test_data_embedded, pd.DataFrame):
        input_sequence = test_data_embedded.values
    else:
        input_sequence = test_data_embedded
    
    input_sequence = input_sequence.astype(np.float32)

    # Reshape if needed
    if len(input_sequence.shape) == 2:
        input_sequence = input_sequence.reshape(1, input_sequence.shape[0], input_sequence.shape[1])

    # Convert to tensor
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).to(device)

    # Set model to evaluation mode
    model.eval()
    model.set_apply_embeddings(False)

    # Create positional encoding
    D = model_params["D"]
    pos_encoding_matrix, _ = create_temporal_positional_encoding(test_data, D)
    pos_encoding_tensor = torch.tensor(pos_encoding_matrix, dtype=torch.float32).to(device)

    # Create look-ahead mask
    seq_len = input_tensor.size(1)
    look_ahead_mask = torch.ones((1, 1, seq_len, seq_len), device=device)

    # Get predictions
    with torch.no_grad():
        prediction, _ = model(input_tensor, look_ahead_mask, pos_encoding_tensor)
        predicted_steps = prediction[:, -prediction_length:, 0].cpu().numpy().squeeze()  # Take only the Average Value predictions

    return predicted_steps


    #print(f"tested building:{building_id}")
    ##redicted_values = np.concatenate([pred.reshape(-1, pred.shape[-1]) for pred in predictions])
    #norm_params_filename = f'norm_params_building_{building_id}.0.pkl'
    #with open(os.path.join(norm_params_path, norm_params_filename), 'rb') as file:
    #    mean, std = pickle.load(file)
    #mean_avg_value = mean['Average Value']
    #std_avg_value = std['Average Value']
    #flattened_predictions = all_predictions_np.flatten()
    #predictions_original = flattened_predictions * std_avg_value + mean_avg_value
    ## Extract actual values for comparison
    #Y_test_actual = test_data['Average Value'][-len(flattened_predictions):]
    #Y_test_actual_original = Y_test_actual * std_avg_value + mean_avg_value
    ## Evaluation metrics
    #mse = mean_squared_error(Y_test_actual_original, predictions_original)
    #rmse = sqrt(mse)
    #mae = mean_absolute_error(Y_test_actual_original, predictions_original)
    #r2 = r2_score(Y_test_actual_original, predictions_original)
    #building_type = test_data['Building Type'].iloc[0]
    ## Save the actual and predicted values for each building to separate pickle files
    #actual_pred_filename = os.path.join(output_path, f'building_{building_id}_{building_type}_predictions.pkl')
    #metrics_filename = os.path.join(output_path, f'building_{building_id}_{building_type}_metrics.pkl')
    #
    #
    #with open(actual_pred_filename, 'wb') as f:
    #    pickle.dump({'Y_test_actual_original': Y_test_actual_original.tolist(),
    #                 'predictions_original': predictions_original.tolist()}, f)
    #
    #with open(metrics_filename, 'wb') as f:
    #    pickle.dump({'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}, f)





#test_data_embedded, test_data, building_id = create_dataset_for_rl(test_data_path)