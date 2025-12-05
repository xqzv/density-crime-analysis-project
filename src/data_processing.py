import pandas as pd
import numpy as np
from src.standardize import convert_numeric_age_to_category

def standardize_age_categories(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize age categories across both datasets using NYPD age categories as standard.
    """
    nypd = nypd_df.copy() if nypd_df is not None else None
    lapd = lapd_df.copy() if lapd_df is not None else None
    
    standard_categories = ['<18', '18-24', '25-44', '45-64', '65+', 'UNKNOWN']

    if nypd is not None:
        valid_age_mask = nypd['Age_Group'].isin(standard_categories)
        nypd.loc[~valid_age_mask, 'Age_Group'] = 'UNKNOWN'
        nypd['Age_Category_Std'] = nypd['Age_Group']

    if lapd is not None:
        lapd['Age_Category_Std'] = lapd['Age'].apply(convert_numeric_age_to_category)

    return nypd, lapd

def standardize_gender(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize gender/sex columns across both datasets.
    """
    nypd = nypd_df.copy() if nypd_df is not None else None
    lapd = lapd_df.copy() if lapd_df is not None else None

    if nypd is not None:
        gender_map = {'M': 'Male', 'F': 'Female', 'U': 'Unknown'}
        nypd['Gender_Std'] = nypd['Perp_Sex'].map(gender_map).fillna('Unknown')

    if lapd is not None:
        gender_map = {'M': 'Male', 'F': 'Female', 'X': 'Unknown'}
        lapd['Gender_Std'] = lapd['Perp_Sex_Code'].map(gender_map).fillna('Unknown')

    return nypd, lapd

def standardize_race_ethnicity(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize race and ethnicity columns across both datasets.
    """
    nypd = nypd_df.copy() if nypd_df is not None else None
    lapd = lapd_df.copy() if lapd_df is not None else None

    if nypd is not None:
        race_map = {
            'BLACK': 'Black',
            'WHITE': 'White',
            'WHITE HISPANIC': 'Hispanic',
            'BLACK HISPANIC': 'Hispanic',
            'ASIAN / PACIFIC ISLANDER': 'Asian/Pacific Islander',
            'AMERICAN INDIAN/ALASKAN NATIVE': 'Native American',
            'UNKNOWN': 'Unknown'
        }
        nypd['Race_Std'] = nypd['Perp_Race'].map(race_map).fillna('Unknown')

    if lapd is not None:
        race_map = {
            'H': 'Hispanic', 'B': 'Black', 'W': 'White',
            'A': 'Asian/Pacific Islander', 'C': 'Asian/Pacific Islander',
            'J': 'Asian/Pacific Islander', 'K': 'Asian/Pacific Islander',
            'L': 'Asian/Pacific Islander', 'V': 'Asian/Pacific Islander',
            'F': 'Asian/Pacific Islander', 'D': 'Asian/Pacific Islander',
            'I': 'Native American', 'O': 'Other', 'X': 'Unknown', 'Z': 'Unknown'
        }
        lapd['Race_Std'] = lapd['Perp_Descent_Code'].map(race_map).fillna('Unknown')

    return nypd, lapd

def standardize_offense_categories(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create standardized offense categories across both datasets.
    """
    nypd = nypd_df.copy() if nypd_df is not None else None
    lapd = lapd_df.copy() if lapd_df is not None else None

    offense_map_nypd = {
        'ROBBERY': 'Violent Crime', 'ASSAULT': 'Violent Crime',
        'BURGLARY': 'Property Crime', 'GRAND LARCENY': 'Property Crime',
        'DANGEROUS DRUGS': 'Drug Offense', 'DANGEROUS WEAPONS': 'Weapon Offense',
        'FELONY ASSAULT': 'Violent Crime', 'PETIT LARCENY': 'Property Crime',
    }
    
    offense_map_lapd = {
        'ROBBERY': 'Violent Crime', 'ASSAULT': 'Violent Crime',
        'BURGLARY': 'Property Crime', 'THEFT': 'Property Crime',
        'NARCOTIC': 'Drug Offense', 'WEAPON': 'Weapon Offense',
        'TRAFFIC': 'Traffic Violation',
    }

    if nypd is not None:
        nypd['Offense_Std'] = nypd['Offense_Category'].map(lambda x: 
            next((v for k, v in offense_map_nypd.items() if k in str(x).upper()), 'Other'))

    if lapd is not None:
        lapd['Offense_Std'] = lapd['Charge_Group_Description'].map(lambda x: 
            next((v for k, v in offense_map_lapd.items() if k in str(x).upper()), 'Other'))

    return nypd, lapd

def create_aligned_datasets(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Create aligned datasets with common columns for comparison.
    """
    if nypd_df is None or lapd_df is None:
        return None, None, []

    nypd_df['Data_Source'] = 'NYPD'
    lapd_df['Data_Source'] = 'LAPD'

    common_columns = [
        'Data_Source', 'Arrest_Year', 'Arrest_Month', 'Arrest_Day',
        'Gender_Std', 'Race_Std', 'Age_Category_Std', 'Offense_Std',
        'Latitude', 'Longitude'
    ]

    def get_aligned_df(df, cols):
        data = {col: df[col] if col in df.columns else np.nan for col in cols}
        return pd.DataFrame(data)

    nypd_aligned = get_aligned_df(nypd_df, common_columns)
    lapd_aligned = get_aligned_df(lapd_df, common_columns)

    return nypd_aligned, lapd_aligned, common_columns

def filter_datasets_by_year_range(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter both datasets to include only records from years that overlap in both datasets.
    """
    if nypd_df is None or lapd_df is None:
        return None, None

    if 'Arrest_Year' not in nypd_df.columns or 'Arrest_Year' not in lapd_df.columns:
        return nypd_df, lapd_df

    nypd_years = {y for y in nypd_df['Arrest_Year'].unique() if pd.notna(y) and y > 1900}
    lapd_years = {y for y in lapd_df['Arrest_Year'].unique() if pd.notna(y) and y > 1900}
    
    overlapping_years = nypd_years.intersection(lapd_years)

    if not overlapping_years:
        return nypd_df, lapd_df

    nypd_filtered = nypd_df[nypd_df['Arrest_Year'].isin(overlapping_years)].copy()
    lapd_filtered = lapd_df[lapd_df['Arrest_Year'].isin(overlapping_years)].copy()

    return nypd_filtered, lapd_filtered
