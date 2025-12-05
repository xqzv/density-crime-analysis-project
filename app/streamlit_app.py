import streamlit as st
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io import load_data
from src.standardize import rename_nypd_columns, rename_lapd_columns, clean_nypd_data, clean_lapd_data
from src.data_processing import (
    standardize_age_categories, 
    standardize_gender, 
    standardize_race_ethnicity, 
    standardize_offense_categories,
    create_aligned_datasets,
    filter_datasets_by_year_range
)
from src.visualization import (
    plot_crime_by_year, 
    plot_crime_by_month,
    plot_crime_by_weekday,
    plot_crime_by_day_of_month,
    create_crime_density_comparison,
    create_demographic_dashboard
)

st.set_page_config(page_title="Density Crime Analysis", layout="wide")

st.title("Density Crime Analysis: NYPD vs LAPD")
st.markdown("Comparative analysis of arrest patterns in New York and Los Angeles.")

with st.expander("About this Project"):
    st.markdown("""
    ### Research Topic and Background
    This study examines arrest patterns across two major American metropolitan police departments—the New York Police Department (NYPD) and the Los Angeles Police Department (LAPD)—from 2010 to 2019. Understanding temporal and spatial crime patterns is crucial for effective law enforcement resource allocation, community safety initiatives, and evidence-based policy development.

    ### Key Criminological Concepts
    1. **Temporal crime patterns**: Cyclical variations in criminal activity based on time of day, day of week, month, or year
    2. **Crime hotspots**: Geographic areas with disproportionately high concentrations of criminal activity
    3. **Enforcement discretion**: The latitude officers have in deciding whether to make arrests for certain offenses
    4. **Broken windows policing**: Enforcement strategy targeting minor offenses to prevent more serious crime
    5. **Density-crime relationship**: Theoretical frameworks linking population density to crime rates and patterns
    6. **Enforcement density**: The concentration of police resources relative to population and geography

    ### Data Sources
    The datasets used in this analysis come from publicly available police arrest records from the NYPD and LAPD, standardized to allow for direct comparison.
    """)

@st.cache_data
def load_and_process_data():
    # Load data (using samples for demo speed if full data is large, but here we try full)
    # In a real app, we might want to load pre-processed parquet files
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    nypd_path = os.path.join(data_dir, 'nypd_aligned.csv')
    lapd_path = os.path.join(data_dir, 'lapd_aligned.csv')
    
    # Fallback to samples if full data not found
    if not os.path.exists(nypd_path):
        nypd_path = os.path.join(data_dir, 'sample_nypd.csv')
    if not os.path.exists(lapd_path):
        lapd_path = os.path.join(data_dir, 'sample_lapd.csv')

    nypd_df = load_data(nypd_path)
    lapd_df = load_data(lapd_path)

    if nypd_df is None or lapd_df is None:
        return None, None

    # Check if data is already aligned (has standardized columns)
    # The aligned data contains 'Data_Source', 'Arrest_Year', 'Offense_Std', etc.
    required_aligned_cols = {'Data_Source', 'Arrest_Year', 'Offense_Std'}
    is_nypd_aligned = required_aligned_cols.issubset(nypd_df.columns)
    is_lapd_aligned = required_aligned_cols.issubset(lapd_df.columns)

    if is_nypd_aligned and is_lapd_aligned:
        # Data is already processed, ensure year overlap
        nypd_final, lapd_final = filter_datasets_by_year_range(nypd_df, lapd_df)
        return nypd_final, lapd_final

    # Pipeline for raw data
    nypd_df = rename_nypd_columns(nypd_df)
    lapd_df = rename_lapd_columns(lapd_df)
    
    nypd_df = clean_nypd_data(nypd_df)
    lapd_df = clean_lapd_data(lapd_df)
    
    nypd_df, lapd_df = standardize_age_categories(nypd_df, lapd_df)
    nypd_df, lapd_df = standardize_gender(nypd_df, lapd_df)
    nypd_df, lapd_df = standardize_race_ethnicity(nypd_df, lapd_df)
    nypd_df, lapd_df = standardize_offense_categories(nypd_df, lapd_df)
    
    nypd_aligned, lapd_aligned, _ = create_aligned_datasets(nypd_df, lapd_df)
    nypd_final, lapd_final = filter_datasets_by_year_range(nypd_aligned, lapd_aligned)
    
    return nypd_final, lapd_final

with st.spinner('Loading and processing data...'):
    nypd_df, lapd_df = load_and_process_data()

if nypd_df is None or lapd_df is None:
    st.error("Failed to load data.")
else:
    st.sidebar.header("Filters")
    
    # Year Range Slider
    min_year = int(min(nypd_df['Arrest_Year'].min(), lapd_df['Arrest_Year'].min()))
    max_year = int(max(nypd_df['Arrest_Year'].max(), lapd_df['Arrest_Year'].max()))
    
    selected_years = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))
    
    # Filter data
    nypd_filtered = nypd_df[(nypd_df['Arrest_Year'] >= selected_years[0]) & (nypd_df['Arrest_Year'] <= selected_years[1])]
    lapd_filtered = lapd_df[(lapd_df['Arrest_Year'] >= selected_years[0]) & (lapd_df['Arrest_Year'] <= selected_years[1])]
    
    st.metric("NYPD Arrests", f"{len(nypd_filtered):,}")
    st.metric("LAPD Arrests", f"{len(lapd_filtered):,}")

    tab1, tab2, tab3, tab4 = st.tabs(["Temporal Analysis", "Geospatial Analysis", "Demographic Analysis", "Data Preview"])

    with tab1:
        st.subheader("Temporal Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Yearly Trend")
            fig_year, ax_year = plt.subplots(figsize=(10, 6))
            plot_crime_by_year(nypd_filtered, lapd_filtered, ax_year, '#1f77b4', '#ff7f0e')
            st.pyplot(fig_year)
            
        with col2:
            st.markdown("#### Monthly Seasonality")
            fig_month, ax_month = plt.subplots(figsize=(10, 6))
            plot_crime_by_month(nypd_filtered, lapd_filtered, ax_month, '#1f77b4', '#ff7f0e')
            st.pyplot(fig_month)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### Day of Week")
            fig_week, ax_week = plt.subplots(figsize=(10, 6))
            plot_crime_by_weekday(nypd_filtered, lapd_filtered, ax_week, '#1f77b4', '#ff7f0e')
            st.pyplot(fig_week)

        with col4:
            st.markdown("#### Day of Month")
            fig_dom, ax_dom = plt.subplots(figsize=(10, 6))
            plot_crime_by_day_of_month(nypd_filtered, lapd_filtered, ax_dom, '#1f77b4', '#ff7f0e')
            st.pyplot(fig_dom)

    with tab2:
        st.subheader("Geospatial Analysis")
        st.markdown("Comparison of crime density between NYC and LA.")
        
        # Sampling slider for map performance
        sample_frac = st.slider("Map Sample Fraction", 0.001, 0.1, 0.01, format="%.3f")
        
        with st.spinner("Generating maps..."):
            fig_map = create_crime_density_comparison(nypd_filtered, lapd_filtered, sample_frac=sample_frac)
            st.pyplot(fig_map)

    with tab3:
        st.subheader("Demographic Analysis")
        st.markdown("Comparison of arrest demographics (Race, Gender, Age, Offense).")
        
        with st.spinner("Generating demographic dashboard..."):
            fig_demo = create_demographic_dashboard(nypd_filtered, lapd_filtered)
            st.pyplot(fig_demo)

    with tab4:
        st.subheader("Data Preview")
        st.write("NYPD Data", nypd_filtered.head())
        st.write("LAPD Data", lapd_filtered.head())
