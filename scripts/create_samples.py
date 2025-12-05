import pandas as pd
import os

def create_sample(input_path, output_path, n=5000):
    try:
        # Read only the first n rows
        df = pd.read_csv(input_path, nrows=n, low_memory=False)
        df.to_csv(output_path, index=False)
        print(f"Successfully created {output_path} with {len(df)} rows.")
    except Exception as e:
        print(f"Error creating sample for {input_path}: {e}")

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    lapd_path = os.path.join(data_dir, 'lapd_aligned.csv')
    nypd_path = os.path.join(data_dir, 'nypd_aligned.csv')
    
    sample_lapd_path = os.path.join(data_dir, 'sample_lapd.csv')
    sample_nypd_path = os.path.join(data_dir, 'sample_nypd.csv')
    
    print("Creating sample datasets...")
    create_sample(lapd_path, sample_lapd_path)
    create_sample(nypd_path, sample_nypd_path)

if __name__ == "__main__":
    main()
