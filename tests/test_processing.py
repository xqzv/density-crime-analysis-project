import pytest
import pandas as pd
import numpy as np
from src.standardize import clean_nypd_data, clean_lapd_data, convert_numeric_age_to_category, rename_nypd_columns, rename_lapd_columns
from src.data_processing import standardize_age_categories, create_aligned_datasets

@pytest.fixture
def sample_nypd_df():
    return pd.DataFrame({
        'ARREST_KEY': [1, 2],
        'ARREST_DATE': ['2020-01-01', '2020-01-02'],
        'PD_CD': [100, np.nan],
        'KY_CD': [101, np.nan],
        'LAW_CAT_CD': ['F', np.nan],
        'LAW_CODE': ['PL 123', np.nan],
        'OFNS_DESC': ['ASSAULT', np.nan],
        'PD_DESC': ['ASSAULT 3', np.nan],
        'ARREST_BORO': ['M', np.nan],
        'AGE_GROUP': ['25-44', np.nan],
        'PERP_SEX': ['M', 'F'],
        'PERP_RACE': ['BLACK', 'WHITE'],
        'X_COORD_CD': [1000, np.nan],
        'Y_COORD_CD': [1000, np.nan],
        'Latitude': [40.7, np.nan],
        'Longitude': [-73.9, np.nan]
    })

@pytest.fixture
def sample_lapd_df():
    return pd.DataFrame({
        'Report ID': [1, 2],
        'Arrest Date': ['2020-01-01', '2020-01-02'],
        'Age': [30, -1],
        'Sex Code': ['M', 'F'],
        'Descent Code': ['B', 'W'],
        'Charge Group Description': ['Aggravated Assault', np.nan],
        'Charge Group Code': [1, np.nan],
        'Charge': ['A', np.nan],
        'Charge Description': ['Desc', np.nan],
        'Arrest Type Code': ['F', np.nan],
        'Disposition Description': ['Disp', np.nan],
        'Time': [1200, np.nan],
        'Cross Street': ['Main St', np.nan],
        'Booking Date': ['2020-01-01', np.nan],
        'Booking Time': [1300, np.nan],
        'Booking Location': ['Loc', np.nan],
        'Booking Location Code': [1, np.nan]
    })

def test_clean_nypd_data(sample_nypd_df):
    renamed = rename_nypd_columns(sample_nypd_df)
    cleaned = clean_nypd_data(renamed)
    assert cleaned['PD_Code'].isna().sum() == 0
    assert cleaned['Age_Group'].iloc[1] == 'UNKNOWN'
    assert 'Arrest_Year' in cleaned.columns

def test_clean_lapd_data(sample_lapd_df):
    renamed = rename_lapd_columns(sample_lapd_df)
    cleaned = clean_lapd_data(renamed)
    assert cleaned['Charge_Group_Code'].isna().sum() == 0
    assert pd.isna(cleaned['Age'].iloc[1])  # Negative age should be NaN
    assert 'Arrest_Year' in cleaned.columns

def test_convert_numeric_age_to_category():
    assert convert_numeric_age_to_category(20) == '18-24'
    assert convert_numeric_age_to_category(10) == '<18'
    assert convert_numeric_age_to_category(-5) == 'UNKNOWN'
    assert convert_numeric_age_to_category(np.nan) == 'UNKNOWN'

def test_create_aligned_datasets():
    # Create minimal dataframes for alignment
    nypd = pd.DataFrame({
        'Arrest_Year': [2020], 'Arrest_Month': [1], 'Arrest_Day': [1],
        'Gender_Std': ['Male'], 'Race_Std': ['Black'], 
        'Age_Category_Std': ['25-44'], 'Offense_Std': ['Violent Crime'],
        'Latitude': [40.0], 'Longitude': [-73.0]
    })
    lapd = pd.DataFrame({
        'Arrest_Year': [2020], 'Arrest_Month': [1], 'Arrest_Day': [1],
        'Gender_Std': ['Male'], 'Race_Std': ['Black'], 
        'Age_Category_Std': ['25-44'], 'Offense_Std': ['Violent Crime'],
        'Latitude': [34.0], 'Longitude': [-118.0]
    })
    
    nypd_aligned, lapd_aligned, common = create_aligned_datasets(nypd, lapd)
    
    assert nypd_aligned.shape[1] == lapd_aligned.shape[1]
    assert 'Data_Source' in nypd_aligned.columns
    assert nypd_aligned['Data_Source'].iloc[0] == 'NYPD'
    assert lapd_aligned['Data_Source'].iloc[0] == 'LAPD'
