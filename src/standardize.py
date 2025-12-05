import pandas as pd
import numpy as np

def rename_nypd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename NYPD dataset columns to standardized names.
    """
    column_mapping = {
        'ARREST_KEY': 'ID',
        'ARREST_DATE': 'Arrest_Date',
        'PD_CD': 'PD_Code',
        'PD_DESC': 'Offense_Description',
        'KY_CD': 'KY_Code',
        'OFNS_DESC': 'Offense_Category',
        'LAW_CODE': 'Law_Code',
        'LAW_CAT_CD': 'Law_Category',
        'ARREST_BORO': 'Arrest_Borough',
        'ARREST_PRECINCT': 'Arrest_Precinct',
        'JURISDICTION_CODE': 'Jurisdiction_Code',
        'AGE_GROUP': 'Age_Group',
        'PERP_SEX': 'Perp_Sex',
        'PERP_RACE': 'Perp_Race',
        'X_COORD_CD': 'X_Coordinate',
        'Y_COORD_CD': 'Y_Coordinate',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude',
        'Lon_Lat': 'Location_Point'
    }
    return df.rename(columns=column_mapping)

def rename_lapd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename LAPD dataset columns to standardized names.
    """
    column_mapping = {
        'Report ID': 'ID',
        'Report Type': 'Report_Type',
        'Arrest Date': 'Arrest_Date',
        'Time': 'Arrest_Time',
        'Area ID': 'Area_ID',
        'Area Name': 'Area_Name',
        'Reporting District': 'Reporting_District',
        'Age': 'Age',
        'Sex Code': 'Perp_Sex_Code',
        'Descent Code': 'Perp_Descent_Code',
        'Charge Group Code': 'Charge_Group_Code',
        'Charge Group Description': 'Charge_Group_Description',
        'Arrest Type Code': 'Arrest_Type_Code',
        'Charge': 'Charge_Code',
        'Charge Description': 'Charge_Description',
        'Disposition Description': 'Disposition_Description',
        'Address': 'Arrest_Address',
        'Cross Street': 'Cross_Street',
        'LAT': 'Latitude',
        'LON': 'Longitude',
        'Location': 'Location_Point',
        'Booking Date': 'Booking_Date',
        'Booking Time': 'Booking_Time',
        'Booking Location': 'Booking_Location',
        'Booking Location Code': 'Booking_Location_Code'
    }
    return df.rename(columns=column_mapping)

def clean_nypd_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NYPD dataset by handling missing values and converting types.
    """
    df = df.copy()
    
    # Fill missing values
    df['PD_Code'] = df['PD_Code'].fillna(-1)
    df['KY_Code'] = df['KY_Code'].fillna(-1)
    df['Law_Category'] = df['Law_Category'].fillna('Unknown')
    df['Law_Code'] = df['Law_Code'].fillna('Unknown')
    df['Offense_Description'] = df['Offense_Description'].fillna('Not Specified')
    df['Offense_Category'] = df['Offense_Category'].fillna('Not Specified')
    df['Arrest_Borough'] = df['Arrest_Borough'].fillna('Unknown')
    df['Age_Group'] = df['Age_Group'].fillna('UNKNOWN')

    # Convert dates
    try:
        df['Arrest_Date'] = pd.to_datetime(df['Arrest_Date'])
        df['Arrest_Year'] = df['Arrest_Date'].dt.year
        df['Arrest_Month'] = df['Arrest_Date'].dt.month
        df['Arrest_Day'] = df['Arrest_Date'].dt.day
    except Exception as e:
        print(f"Error converting NYPD arrest date: {e}")

    # Handle coordinates
    for coord in ['X_Coordinate', 'Y_Coordinate', 'Latitude', 'Longitude']:
        if coord in df.columns:
            df[coord] = df[coord].fillna(0)

    return df

def clean_lapd_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean LAPD dataset by handling missing values and converting types.
    """
    df = df.copy()

    # Fill missing values
    df['Charge_Group_Code'] = df['Charge_Group_Code'].fillna(-1)
    df['Charge_Group_Description'] = df['Charge_Group_Description'].fillna('Not Specified')
    df['Charge_Code'] = df['Charge_Code'].fillna('Unknown')
    df['Charge_Description'] = df['Charge_Description'].fillna('Not Specified')
    df['Arrest_Type_Code'] = df['Arrest_Type_Code'].fillna('Unknown')
    df['Disposition_Description'] = df['Disposition_Description'].fillna('Unknown')
    df['Arrest_Time'] = df['Arrest_Time'].fillna(-1)
    df['Cross_Street'] = df['Cross_Street'].fillna('Not Specified')
    df['Booking_Date'] = df['Booking_Date'].fillna('Unknown')
    df['Booking_Time'] = df['Booking_Time'].fillna(-1)
    df['Booking_Location'] = df['Booking_Location'].fillna('Unknown')
    df['Booking_Location_Code'] = df['Booking_Location_Code'].fillna(-1)

    # Clean Age
    if 'Age' in df.columns:
        invalid_age_mask = (df['Age'] < 0) | (df['Age'] > 100)
        df.loc[invalid_age_mask, 'Age'] = np.nan

    # Convert dates
    try:
        df['Arrest_Date'] = pd.to_datetime(df['Arrest_Date'])
        df['Arrest_Year'] = df['Arrest_Date'].dt.year
        df['Arrest_Month'] = df['Arrest_Date'].dt.month
        df['Arrest_Day'] = df['Arrest_Date'].dt.day
    except Exception as e:
        print(f"Error converting LAPD arrest date: {e}")

    # Remove duplicates
    df = df.drop_duplicates()

    return df

def convert_numeric_age_to_category(age: float) -> str:
    """
    Convert a numeric age to the corresponding NYPD age group category.
    """
    if pd.isna(age) or age < 0:
        return 'UNKNOWN'

    if age < 18:
        return '<18'
    elif 18 <= age <= 24:
        return '18-24'
    elif 25 <= age <= 44:
        return '25-44'
    elif 45 <= age <= 64:
        return '45-64'
    elif age >= 65:
        return '65+'
    else:
        return 'UNKNOWN'
