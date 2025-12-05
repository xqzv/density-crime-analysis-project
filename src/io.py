import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame | None:
    """
    Load data from CSV files with robust error handling.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame | None: Loaded DataFrame or None if loading failed.
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Data successfully loaded from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse CSV file at {filepath}. Check file format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}")
        return None

def save_dataframe_to_csv(df: pd.DataFrame, filename: str) -> None:
    """
    Save a pandas DataFrame to a CSV file with error handling.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        filename (str): Destination file path.
    """
    try:
        df.to_csv(filename, index=False)
        print(f"\n--- Saving Progress ---")
        print(f"Done! The progress saved here: '{filename}'.")
    except PermissionError:
        print(f"PermissionError: Unable to save '{filename}'. Please close the file if it's open and try again.")
    except FileNotFoundError:
        print(f"FileNotFoundError: Directory for '{filename}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred while saving '{filename}': {e}")
