from chessdotcom import *
from random import shuffle
from sklearn.model_selection import train_test_split

import os
import pickle
import requests
import urllib.request
import json

def titled_player_names(title):
    """
    Retrieve a list of player names with a specified title from an external API.

    Parameters:
    - title (str): The title of players to retrieve (e.g., "GM" for Grandmaster, "IM" for International Master).

    Returns:
    - list of str: A list of player names who hold the specified title, obtained from an external API.
      If no players with the specified title are found, an empty list is returned.
    """
    data = get_titled_players(title)
    list = data.json["players"]
    return list

def build_df_player_profiles(username_list):
    """
    Build a list of player profiles based on the given list of usernames.

    Parameters:
    - username_list (list): A list of chess.com usernames.

    Returns:
    - final_list: A list of player profiles, each represented as a list of information (name, username, title, followers, 
            country_code, country, status, is_streamer, verified, league).
    - fail_list (list): A list of usernames for which profile retrieval failed after 15 retries.
    """
    # List to store lists of data records, username and player profile information in each list.
    final_list = []

    # List to store any usernames for which player profiles cannot be collected.
    fail_list = []

    # If a player profile cannot be obtained, this counter is used to stop the while loop
    retry = 0

    
    # For each username in the username_list
    for username in username_list:
        while True:
            try:        
                # Get player profile response from chessdotcom.get_player_profile
                player_profile_response = get_player_profile(username)
                
                # Extract all relevant information, if not available, append None.
                player_info = player_profile_response.player
                name = getattr(player_info, 'name', None)
                username = getattr(player_info, 'username', None)
                title = getattr(player_info, 'title', None)
                followers = getattr(player_info, 'followers', None)
                status = getattr(player_info, 'status', None)
                is_streamer = getattr(player_info, 'is_streamer', None)
                verified = getattr(player_info, 'verified', None)
                league = getattr(player_info, 'league', None)
        
                country_code = ''
                country = ''
                with urllib.request.urlopen(player_info.country) as url:
                    data = json.load(url)
                    country_code = data['code']
                    country = data['name']

                # Create a list with all the relevant player profile information and append this list to the final_list.
                list_data = [name, username, title, followers, country_code, country, status, is_streamer, verified, league]
                final_list.append(list_data)
                break

            # Except a ChessDotComError as e, in the case a usernames get_player_profile does not work.
            # This allows the function to retry the call a specified number of times, in this case 15.
            # If the retry variable hits 15, then this username is appened to the fail_list and the function moves on.
            except ChessDotComError as e:
                print(f'\rError for {username}: {e}')
                if retry < 15:
                    retry += 1
                    print(f'\rRetrying for {username} attempt {retry}...')
                else:
                    print(f'\rMax retries reached for {username}. Moving on.')
                    fail_list.append(username)
                    retry = 0
                    break
            
    return final_list, fail_list


def preprocess_data(data):
    """
    Preprocess a list of feature vectors and corresponding labels.

    Parameters:
    - data (list): A list of samples, where each sample is represented as a list of features followed by a label.

    Returns:
    - tuple: A tuple containing two lists:
      - features (list): A list of feature vectors, where each element is a list representing features for one sample.
      - labels (list): A list of labels corresponding to the feature vectors.

    Note:
    - The function shuffles the input `data` to randomize the order of samples.
    - Each sample in `data` is expected to be a list where the last element is the label,
      and the preceding elements are the feature values.
    - The returned `features` list contains all feature vectors, and the `labels` list contains
      corresponding labels extracted from the input `data`.
    """
    shuffle(data)
    return [i[:-1] for i in data], [i[-1:][0] for i in data]

def train_validation_test_data_sets(features, label, stratify=False, seed=15):
    """
    Split features and labels into training, validation, and test datasets using train-test split.

    Parameters:
    - features (list): A list of feature vectors where each element is a list representing features for one sample.
    - labels (list): A list of labels corresponding to the feature vectors.
    - stratify (bool, optional): If True, perform stratified splitting based on the labels to maintain class distribution.
      Default is False.
    - seed (int, optional): Random seed for reproducibility. Default is 15.

    Returns:
    - tuple: A tuple containing three lists:
      - X_train (list): List of feature vectors for the training set.
      - X_val (list): List of feature vectors for the validation set.
      - X_test (list): List of feature vectors for the test set.

    Note:
    - The function uses `train_test_split` from scikit-learn to split the data into training, validation, and test sets.
    - By default, it splits the data into 70% training, 10% validation, and 20% test sets.
    - If `stratify=True`, the data is stratified based on the labels to ensure proportional class distribution in each split.
    - The returned feature lists (X_train, X_val, X_test) are augmented with corresponding labels for convenience.
    """
    if stratify:
        X_train, X_temp, y_train, y_temp = train_test_split(features, label, test_size=0.3, stratify=label, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, stratify=y_temp, random_state=seed)
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(features, label, test_size=0.3, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=seed)

    counter_1 = len(X_val)
    counter_2 = len(X_test)
    for i in range(len(X_train)):
        if i < counter_1:
            X_val[i].append(y_val[i])

        if i < counter_2:
            X_test[i].append(y_test[i])

        X_train[i].append(y_train[i])
    
    return X_train, X_val, X_test

def read_pkl(file_path):
    """
    Read and load data from a pickle (.pkl) file.

    Parameters:
    - file_path (str): The file path to the pickle file containing the data to be loaded.

    Returns:
    - object: The loaded data object from the pickle file.

    Example:
    >>> loaded_data = read_pkl('data.pkl')
    """
    # Load the required dataset and store within a variable.
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    return loaded_data

def create_dict_pkl_files(pkl_file_loc, pkl_file_names_list):
    """
    Create a dictionary of loaded data from pickle (.pkl) files.

    This function reads and loads data from a list of specified pickle files and
    stores the loaded data in a dictionary, where each key-value pair corresponds
    to a file name (key) and its corresponding loaded data object (value).

    Parameters:
    - pkl_file_loc (str): The directory location where the pickle files are stored.
    - pkl_file_names_list (list): A list of file names (strings) of the pickle files to load.

    Returns:
    - dict: A dictionary containing loaded data objects from the specified pickle files,
      where each key is a file name and the corresponding value is the loaded data object.
      
    Example:
    >>> pkl_files_directory = '/path/to/pickle_files'
    >>> file_names_list = ['file1.pkl', 'file2.pkl', 'file3.pkl']
    >>> data_dict = create_dict_pkl_files(pkl_files_directory, file_names_list)
    >>> print(data_dict)
    """
    data_dict = {}

    # iterate over every file name given.
    for file_name in pkl_file_names_list:

        # read in the current file.
        data = read_pkl(f"{pkl_file_loc}/{file_name}")

        # store the data in the pkl_files values with the key being the file name.
        data_dict[file_name] = data

    return data_dict

def list_files_in_directory(directory):
    """
    Return a list of file names (excluding directories) from a specified directory.

    Parameters:
    - directory (str): The directory path from which to list files.

    Returns:
    - list: A list of file names (excluding directories) in the specified directory.

    Raises:
    - FileNotFoundError: If the specified `directory` does not exist.
    """
    try:
        # Check if the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Get a list of all files (excluding directories) in the specified directory
        files = [filename for filename in os.listdir(directory) if os.path.isfile(os.path.join(directory, filename))]

        return files
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error listing files: {str(e)}")

    
def balance_datasets(chess_data_set, num_classes):
    """
    Balances the input chess dataset by limiting each class to the size of the smallest class.

    Parameters:
    - chess_data_set (list): The input dataset containing data points with class labels.
    - num_classes (int): The total number of distinct class labels in the dataset.

    Returns:
    - list: The balanced test dataset containing an equal number of data points for each class label.

    This function splits the input dataset based on class labels, calculates the size of the smallest class,
    and then limits each class dataset to the size of the smallest class. The resulting test dataset is
    concatenated from the balanced class datasets. It also prints the lcount of each class in the balanced test dataset 
    before returning it.

    Example usage:
    >>> chess_data_set = [
    ...     [1, 2, 3, 0],
    ...     [4, 5, 6, 1],
    ...     [7, 8, 9, 2],
    ...     [10, 11, 12, 0],
    ...     [13, 14, 15, 1],
    ...     [16, 17, 18, 2],
    ...     # Add more data points as needed
    ... ]
    >>> num_classes = 3
    >>> balanced_test_dataset = balance_datasets(chess_data_set, num_classes)

    In the returned balanced test dataset, each class label will have an equal number of data points,
    limited to the size of the smallest class.
    """
    # Initialize empty lists for each class label
    label_datasets = [[] for _ in range(num_classes)]

    # Split the dataset based on class labels
    for data in chess_data_set:
        label = data[-1]  # Assuming class label is the last element in each data point
        label_datasets[label].append(data)

    # Find the size of the smallest class
    min_class_size = min(len(label_dataset) for label_dataset in label_datasets if label_dataset)

    # Balance the datasets by limiting each class to the size of the smallest class
    balanced_datasets = [label_dataset[:min_class_size] for label_dataset in label_datasets if label_dataset]

    # Concatenate the balanced datasets to create the final test dataset
    test_dataset = sum(balanced_datasets, [])

    # Print the counts of each class in the test dataset
    class_counts = [0] * num_classes
    for data in test_dataset:
        label = data[-1]  # Assuming class label is the last element in each data point
        class_counts[label] += 1

    print(f"\nNew size of the overall dataset = {len(test_dataset)}")
    for i in range(num_classes):
        print(f"Count of class {i} in balanced dataset: {class_counts[i]}")

    return test_dataset
    

def display_number_of_each_class(data_set, num_classes):
    print(f"\nLength of overall current chess data = {len(data_set)}")
    
        # Print the counts of each class in the test dataset
    class_counts = [0] * num_classes
    for data in data_set:
        label = data[-1]  # Assuming class label is the last element in each data point
        class_counts[label] += 1

    for i in range(num_classes):
        print(f"Count of class {i} in test dataset: {class_counts[i]}")
