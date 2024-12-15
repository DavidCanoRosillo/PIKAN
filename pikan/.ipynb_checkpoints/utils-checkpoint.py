import pickle 
import os

def save_dict_to_file(dictionary, filepath, filename):
    """Saves a dictionary to a file using pickle."""
    
    # Ensure the directory exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    # Combine the directory path and filename to get the full file path
    full_path = os.path.join(filepath, filename)
    
    # Open the file and dump the dictionary using pickle
    with open(full_path, 'wb') as file:
        pickle.dump(dictionary, file)

def load_dict_from_file(filename):
    """Loads a dictionary from a file using pickle."""
    with open(filename, 'rb') as file:
        return pickle.load(file)