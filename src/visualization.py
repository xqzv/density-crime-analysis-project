import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator
from datetime import datetime

def plot_crime_by_weekday(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame, ax: plt.Axes, nypd_color: str, lapd_color: str) -> None:
    """
    Plot crime frequency by day of week.
    """
    def get_weekday(year, month, day):
        try:
            return datetime(year, month, day).weekday()
        except (ValueError, TypeError):
            return None

    # Helper to process dataframe
    def process_weekdays(df, dept_name):
        weekdays = []
        # Process in chunks if needed, but for now simple apply is fine for readability
        # Optimization: Vectorized approach
        dates = pd.to_datetime(df[['Arrest_Year', 'Arrest_Month', 'Arrest_Day']].rename(
            columns={'Arrest_Year': 'year', 'Arrest_Month': 'month', 'Arrest_Day': 'day'}), errors='coerce')
        weekdays = dates.dt.dayofweek.dropna().astype(int)
        
        days_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                    4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        
        day_names = weekdays.map(days_map)
        counts = day_names.value_counts()
        
        data = []
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days_order:
            data.append({'day_name': day, 'count': counts.get(day, 0), 'department': dept_name})
            
        return pd.DataFrame(data)

    nypd_counts = process_weekdays(nypd_df, 'NYPD')
    lapd_counts = process_weekdays(lapd_df, 'LAPD')
    day_counts = pd.concat([nypd_counts, lapd_counts])

    # Add day index for sorting
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts['day_index'] = day_counts['day_name'].map({day: i for i, day in enumerate(days_order)})
    day_counts = day_counts.sort_values('day_index')

    sns.barplot(x='day_name', y='count', hue='department', data=day_counts,
                palette=[nypd_color, lapd_color], ax=ax)
    ax.set_title('Crime Frequency by Day of Week', pad=15)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Number of Crimes')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Department')

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + height*0.02,
                    f'{int(height):,}', ha="center", fontsize=9)

def plot_crime_by_month(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame, ax: plt.Axes, nypd_color: str, lapd_color: str) -> None:
    """
    Create a line plot showing crime frequency by month.
    """
    try:
        def get_month_counts(df, dept_name):
            counts = df['Arrest_Month'].value_counts().reset_index()
            counts.columns = ['month', 'count']
            counts['department'] = dept_name
            counts = counts[counts['month'].between(1, 12)]
            
            # Ensure all months present
            all_months = pd.DataFrame({'month': range(1, 13)})
            all_months['department'] = dept_name
            return pd.merge(all_months, counts, on=['month', 'department'], how='left').fillna(0)

        nypd_counts = get_month_counts(nypd_df, 'NYPD')
        lapd_counts = get_month_counts(lapd_df, 'LAPD')

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        ax.set_title('Crime Frequency by Month', pad=15)
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Crimes')
        ax.set_xticks(range(len(month_names)))
        ax.set_xticklabels(month_names)
        ax.set_xlim(-0.5, 11.5)

        # Seasonal background
        ax.axvspan(-0.5, 1.5, alpha=0.1, color='lightblue', label='Winter')
        ax.axvspan(10.5, 11.5, alpha=0.1, color='lightblue')
        ax.axvspan(1.5, 4.5, alpha=0.1, color='lightgreen', label='Spring')
        ax.axvspan(4.5, 7.5, alpha=0.1, color='yellow', label='Summer')
        ax.axvspan(7.5, 10.5, alpha=0.1, color='orange', label='Fall')

        for dept, color, data in [('NYPD', nypd_color, nypd_counts), ('LAPD', lapd_color, lapd_counts)]:
            data = data.sort_values('month')
            data['month_idx'] = data['month'] - 1
            ax.plot(data['month_idx'], data['count'], color=color, marker='o',
                    label=dept, linewidth=2, markersize=8)
            
            if not data.empty:
                peak_idx = data['count'].idxmax()
                peak_month = data.loc[peak_idx]
                ax.annotate(f"{dept} Peak\n{int(peak_month['count']):,}",
                           xy=(peak_month['month_idx'], peak_month['count']),
                           xytext=(peak_month['month_idx'], peak_month['count'] * 1.05),
                           ha='center', va='bottom', fontsize=9, color=color,
                           bbox=dict(facecolor='white', edgecolor=color, alpha=0.7,
                                    boxstyle='round,pad=0.3'))

        ax.legend(title='Department', loc='upper right')

    except Exception as e:
        ax.text(0.5, 0.5, f"Error in monthly visualization: {str(e)}",
               ha='center', va='center', fontsize=10, transform=ax.transAxes,
               bbox=dict(facecolor='white', edgecolor='red', alpha=0.8))

def plot_crime_by_year(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame, ax: plt.Axes, nypd_color: str, lapd_color: str) -> None:
    """
    Create a bar plot showing crime frequency by year.
    """
    def get_year_counts(df, dept_name):
        counts = df['Arrest_Year'].value_counts().reset_index()
        counts.columns = ['year', 'count']
        counts['department'] = dept_name
        return counts

    nypd_counts = get_year_counts(nypd_df, 'NYPD')
    lapd_counts = get_year_counts(lapd_df, 'LAPD')
    year_counts = pd.concat([nypd_counts, lapd_counts])

    year_pivot = year_counts.pivot(index='year', columns='department', values='count').fillna(0)

    year_pivot[['NYPD', 'LAPD']].plot(kind='bar', stacked=False, ax=ax,
                                      color=[nypd_color, lapd_color], width=0.7,
                                      edgecolor='black', linewidth=0.5)

    # Trend lines
    for i, dept in enumerate(['NYPD', 'LAPD']):
        if len(year_pivot) >= 3:
            x = np.arange(len(year_pivot.index))
            y = year_pivot[dept].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), '--', color=['darkblue', 'darkred'][i],
                    linewidth=1.5, alpha=0.8)

    ax.set_title('Crime Frequency by Year', pad=15)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Crimes')
    ax.legend(title='Department')

    for i, (year, row) in enumerate(year_pivot.iterrows()):
        for j, dept in enumerate(['NYPD', 'LAPD']):
            height = row[dept]
            ax.text(i + (j-0.5)*0.3, height * 1.02,
                   f'{int(height):,}', ha='center', va='bottom',
                   fontsize=9, color=['darkblue', 'darkred'][j])

    ax.set_xticklabels([str(year) for year in year_pivot.index], rotation=0)

def plot_crime_by_day_of_month(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame, ax: plt.Axes, nypd_color: str, lapd_color: str) -> None:
    """
    Create a plot showing crime frequency by day of month.
    """
    if 'Arrest_Day' not in nypd_df.columns or 'Arrest_Day' not in lapd_df.columns:
        ax.text(0.5, 0.5, "Error: Arrest_Day column not found",
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        return

    try:
        all_days = pd.Series(range(1, 32))

        def get_day_counts(df, dept_name):
            counts = df['Arrest_Day'].value_counts().reset_index()
            counts.columns = ['day', 'count']
            day_counts = pd.DataFrame({'day': all_days})
            day_counts = day_counts.merge(counts, on='day', how='left').fillna(0)
            day_counts['department'] = dept_name
            day_counts = day_counts.sort_values('day')
            day_counts['rolling_avg'] = day_counts['count'].rolling(window=3, center=True).mean()
            return day_counts

        nypd_day_counts = get_day_counts(nypd_df, 'NYPD')
        lapd_day_counts = get_day_counts(lapd_df, 'LAPD')

        ax.scatter(nypd_day_counts['day'], nypd_day_counts['count'],
                  color=nypd_color, alpha=0.3, s=30, label='NYPD (Daily)')
        ax.plot(nypd_day_counts['day'], nypd_day_counts['rolling_avg'],
               color=nypd_color, linewidth=2.5, label='NYPD (3-day avg)')

        ax.scatter(lapd_day_counts['day'], lapd_day_counts['count'],
                  color=lapd_color, alpha=0.3, s=30, label='LAPD (Daily)')
        ax.plot(lapd_day_counts['day'], lapd_day_counts['rolling_avg'],
               color=lapd_color, linewidth=2.5, label='LAPD (3-day avg)')

        for day in [1, 15, 28]:
            ax.axvline(x=day, color='gray', linestyle='--', alpha=0.5)

        ax.axvspan(1, 10, alpha=0.1, color='green', label='Early Month')
        ax.axvspan(11, 20, alpha=0.1, color='blue', label='Mid Month')
        ax.axvspan(21, 31, alpha=0.1, color='red', label='Late Month')

        ax.set_title('Crime Frequency by Day of Month', pad=15)
        ax.set_xlabel('Day of Month')
        ax.set_ylabel('Number of Crimes')
        ax.set_xlim(0.5, 31.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        handles, labels = ax.get_legend_handles_labels()
        if len(handles) >= 3:
            # Filter to show only lines in legend, not scatter points if desired, or just show all
            # Simplified for robustness:
            ax.legend(title='Department', loc='upper right')

    except Exception as e:
        ax.text(0.5, 0.5, f"Error processing day of month data: {str(e)}",
                ha='center', va='center', fontsize=10, transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='red', alpha=0.8))

def create_temporal_analysis_plot(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame) -> plt.Figure:
    """
    Create and return the figure for temporal analysis visualizations.
    """
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    
    nypd_color = '#1f77b4'
    lapd_color = '#ff7f0e'

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    fig.suptitle('Temporal Analysis of Crime Patterns: NYPD vs. LAPD', fontsize=18, y=0.98)

    plot_crime_by_weekday(nypd_df, lapd_df, axes[0], nypd_color, lapd_color)
    plot_crime_by_month(nypd_df, lapd_df, axes[1], nypd_color, lapd_color)
    plot_crime_by_year(nypd_df, lapd_df, axes[2], nypd_color, lapd_color)
    plot_crime_by_day_of_month(nypd_df, lapd_df, axes[3], nypd_color, lapd_color)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# City configurations for map visualizations
CITY_CONFIGS = {
    'NYC': {
        'boundaries': {
            'lat': (40.4, 40.95),
            'lon': (-74.30, -73.65)
        },
        'zoom': 11,
        'title': "New York City Crime Density",
        'scale': {
            'length': 0.05,
            'text': "≈ 4 km"
        }
    },
    'LA': {
        'boundaries': {
            'lat': (33.65, 34.35),
            'lon': (-118.95, -118.12)
        },
        'zoom': 10,
        'title': "Los Angeles Crime Density",
        'scale': {
            'length': 0.05,
            'text': "≈ 5 km"
        }
    }
}

def prepare_crime_data(df: pd.DataFrame, city_name: str, sample_frac: float = 0.05) -> pd.DataFrame:
    """
    Filter and prepare crime data for visualization.
    """
    if city_name not in CITY_CONFIGS:
        raise ValueError(f"city_name must be one of {list(CITY_CONFIGS.keys())}")

    boundaries = CITY_CONFIGS[city_name]['boundaries']
    lat_min, lat_max = boundaries['lat']
    lon_min, lon_max = boundaries['lon']

    # Filter data
    filter_query = (
        f"Latitude >= {lat_min} & "
        f"Latitude <= {lat_max} & "
        f"Longitude >= {lon_min} & "
        f"Longitude <= {lon_max}"
    )
    filtered_df = df.query(filter_query)

    # Sample data if necessary
    if len(filtered_df) > 10000:
        if 'Offense_Std' in filtered_df.columns:
            # Stratified sampling
            offense_groups = filtered_df.groupby('Offense_Std')
            sampled_dfs = []

            for name, group in offense_groups:
                group_size = len(group)
                sample_size = min(int(group_size * sample_frac) + 1, group_size)
                sampled_dfs.append(group.sample(sample_size))

            sampled_df = pd.concat(sampled_dfs)
        else:
            # Random sampling
            sampled_df = filtered_df.sample(frac=sample_frac, random_state=42)
    else:
        sampled_df = filtered_df

    return sampled_df

def calculate_density(x: np.ndarray, y: np.ndarray):
    """
    Calculate kernel density estimate for points.
    """
    if len(x) <= 10:
        return None, None

    try:
        k = gaussian_kde(np.vstack([x, y]), bw_method='scott')
        densities = k(np.vstack([x, y]))

        if densities.max() > densities.min():
            densities_norm = (densities - densities.min()) / (densities.max() - densities.min())
        else:
            densities_norm = np.zeros_like(densities)

        return densities, densities_norm
    except Exception as e:
        print(f"KDE calculation failed: {e}")
        return None, None

def plot_crime_density(df: pd.DataFrame, ax: plt.Axes, city_name: str, alpha: float = 0.6, 
                      cmap: str = 'hot_r', point_size: int = 10, zoom_level: int = None) -> None:
    """
    Plot crime density for a given city.
    """
    if city_name not in CITY_CONFIGS:
        raise ValueError(f"city_name must be one of {list(CITY_CONFIGS.keys())}")

    config = CITY_CONFIGS[city_name]
    lat_min, lat_max = config['boundaries']['lat']
    lon_min, lon_max = config['boundaries']['lon']
    zoom = zoom_level if zoom_level is not None else config['zoom']
    title = config['title']

    x = df['Longitude'].values
    y = df['Latitude'].values

    densities, densities_norm = calculate_density(x, y)

    if densities is not None:
        scatter = ax.scatter(
            x, y,
            s=point_size,
            c=densities,
            cmap=cmap,
            alpha=alpha
        )
        cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.01, shrink=0.5)
        cbar.set_label('Crime Density', fontsize=10)
    else:
        ax.scatter(
            x, y,
            s=point_size,
            c=plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(x))),
            alpha=alpha
        )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron, zoom=zoom)
    except Exception as e:
        print(f"Could not add basemap: {e}")
        ax.set_facecolor('#F2F2F2')
        ax.grid(True, linestyle='--', alpha=0.7)

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()

    ax.text(
        0.01, 0.01,
        f"Data Source: {city_name}\nTotal Crimes Plotted: {len(df)}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='bottom',
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
    )

    scale_info = config['scale']
    ax.plot(
        [lon_max - 0.03 - scale_info['length'], lon_max - 0.03],
        [lat_min + 0.03, lat_min + 0.03],
        'k-',
        lw=2
    )
    ax.text(
        lon_max - 0.03 - scale_info['length']/2,
        lat_min + 0.045,
        scale_info['text'],
        fontsize=8,
        ha='center',
        va='bottom'
    )

def create_crime_density_comparison(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame, 
                                  sample_frac: float = 0.01, fig_size: tuple = (20, 10),
                                  cmap: str = 'hot_r', point_size: int = 8) -> plt.Figure:
    """
    Create a side-by-side comparison of crime density maps.
    """
    required_cols = ['Latitude', 'Longitude']
    for df, city in [(nypd_df, 'NYC'), (lapd_df, 'LA')]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Return empty figure or handle error gracefully for dashboard
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Missing coordinates for {city}: {missing_cols}", 
                   ha='center', va='center')
            return fig

    nyc_data = prepare_crime_data(nypd_df, 'NYC', sample_frac=sample_frac)
    la_data = prepare_crime_data(lapd_df, 'LA', sample_frac=sample_frac)

    fig, axes = plt.subplots(1, 2, figsize=fig_size, constrained_layout=True)

    plot_crime_density(nyc_data, axes[0], 'NYC', cmap=cmap, point_size=point_size)
    plot_crime_density(la_data, axes[1], 'LA', cmap=cmap, point_size=point_size)

    fig.suptitle("Crime Density Comparison: NYC vs. LA", fontsize=16)
    return fig

def create_demographic_dashboard(df1: pd.DataFrame, df2: pd.DataFrame, 
                               df1_name: str = "NYPD", df2_name: str = "LAPD") -> plt.Figure:
    """
    Creates a comprehensive demographic comparison dashboard.
    """
    # Use a dark style for this specific plot if desired, or stick to whitegrid
    # The original code used dark_background, but let's stick to the current style or use a context manager
    with plt.style.context('seaborn-v0_8-dark'): # Using a built-in style that is dark-ish or just custom
        
        # Define colors
        NYPD_color = '#5e9cd3'
        LAPD_color = '#f49c3b'
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')

        axes = axes.flatten()

        for ax in axes:
            ax.set_facecolor('#1e1e1e')
            ax.grid(True, color='#333333', linestyle='-', linewidth=0.5, alpha=0.7)
            ax.spines['bottom'].set_color('#333333')
            ax.spines['top'].set_color('#333333')
            ax.spines['right'].set_color('#333333')
            ax.spines['left'].set_color('#333333')
            ax.tick_params(colors='#cccccc')

        # 1. Race Distribution
        ax_race = axes[0]
        race_categories = ['Black', 'Hispanic', 'White', 'Asian/Pacific Islander',
                          'Other', 'Native American', 'Unknown']
        
        race_pct1 = (df1['Race_Std'].value_counts() / len(df1)) * 100
        race_pct2 = (df2['Race_Std'].value_counts() / len(df2)) * 100
        
        race_pct1 = race_pct1.reindex(race_categories, fill_value=0)
        race_pct2 = race_pct2.reindex(race_categories, fill_value=0)
        
        sorted_cats = ['Black', 'Hispanic', 'White', 'Asian/Pacific Islander',
                      'Other', 'Unknown', 'Native American']
        
        race_df = pd.DataFrame({
            df1_name: race_pct1.reindex(sorted_cats),
            df2_name: race_pct2.reindex(sorted_cats)
        })
        
        race_df.plot(kind='barh', ax=ax_race, color=[NYPD_color, LAPD_color])
        
        ax_race.set_title("Race Distribution", fontsize=14, color='white')
        ax_race.set_xlabel("Percentage (%)", fontsize=12, color='white')
        ax_race.legend(title="Department", title_fontsize=12)
        
        leg = ax_race.get_legend()
        for text in leg.get_texts():
            text.set_color('white')
        leg.get_title().set_color('white')

        # 2. Gender Distribution
        ax_gender = axes[1]
        gender_pct1 = (df1['Gender_Std'].value_counts() / len(df1)) * 100
        gender_pct2 = (df2['Gender_Std'].value_counts() / len(df2)) * 100
        
        male1 = gender_pct1.get('Male', 0)
        female1 = gender_pct1.get('Female', 0)
        male2 = gender_pct2.get('Male', 0)
        female2 = gender_pct2.get('Female', 0)
        
        ax_gender.pie([male2, female2], radius=0.7, wedgeprops=dict(width=0.3, edgecolor='#1e1e1e'),
                     startangle=90, colors=[LAPD_color, '#ffc681'])
        ax_gender.pie([male1, female1], radius=1.0, wedgeprops=dict(width=0.3, edgecolor='#1e1e1e'),
                     startangle=90, colors=[NYPD_color, '#85c2f0'])
                     
        ax_gender.text(1.2, 0.15, f"{df1_name} - Male: {male1:.1f}%", ha='left', va='center', fontsize=11, color=NYPD_color)
        ax_gender.text(1.2, 0.0, f"{df1_name} - Female: {female1:.1f}%", ha='left', va='center', fontsize=11, color='#85c2f0')
        ax_gender.text(1.2, -0.15, f"{df2_name} - Male: {male2:.1f}%", ha='left', va='center', fontsize=11, color=LAPD_color)
        ax_gender.text(1.2, -0.3, f"{df2_name} - Female: {female2:.1f}%", ha='left', va='center', fontsize=11, color='#ffc681')
        
        ax_gender.set_title("Gender Distribution", fontsize=14, color='white')
        ax_gender.axis('equal')

        # 3. Age Distribution
        ax_age = axes[2]
        age_categories = ['<18', '18-24', '25-44', '45-64', '65+']
        
        age_pct1 = (df1['Age_Category_Std'].value_counts() / len(df1)) * 100
        age_pct2 = (df2['Age_Category_Std'].value_counts() / len(df2)) * 100
        
        age_pct1 = age_pct1.reindex(age_categories, fill_value=0)
        age_pct2 = age_pct2.reindex(age_categories, fill_value=0)
        
        age_df = pd.DataFrame({df1_name: age_pct1, df2_name: age_pct2})
        age_df.plot(kind='line', marker='o', markersize=8, linewidth=2, ax=ax_age, color=[NYPD_color, LAPD_color])
        
        ax_age.set_title("Age Distribution", fontsize=14, color='white')
        ax_age.set_xlabel("Age Group", fontsize=12, color='white')
        ax_age.set_ylabel("Percentage (%)", fontsize=12, color='white')
        
        leg = ax_age.legend(title="Department", title_fontsize=12)
        for text in leg.get_texts():
            text.set_color('white')
        leg.get_title().set_color('white')

        # 4. Offense Type Distribution
        ax_offense = axes[3]
        offense_categories = ['Violent Crime', 'Property Crime', 'Drug Offense',
                            'Weapon Offense', 'Traffic Violation', 'Other']
                            
        offense_pct1 = (df1['Offense_Std'].value_counts() / len(df1)) * 100
        offense_pct2 = (df2['Offense_Std'].value_counts() / len(df2)) * 100
        
        offense_pct1 = offense_pct1.reindex(offense_categories, fill_value=0)
        offense_pct2 = offense_pct2.reindex(offense_categories, fill_value=0)
        
        offense_diff = abs(offense_pct1 - offense_pct2)
        sorted_offense = offense_diff.sort_values(ascending=False).index
        
        offense_df = pd.DataFrame({
            df1_name: offense_pct1.reindex(sorted_offense),
            df2_name: offense_pct2.reindex(sorted_offense)
        })
        
        offense_df.plot(kind='barh', ax=ax_offense, color=[NYPD_color, LAPD_color])
        
        ax_offense.set_title("Offense Type Distribution", fontsize=14, color='white')
        ax_offense.set_xlabel("Percentage (%)", fontsize=12, color='white')
        
        leg = ax_offense.legend(title="Department", title_fontsize=12)
        for text in leg.get_texts():
            text.set_color('white')
        leg.get_title().set_color('white')

        fig.suptitle("Crime Demographic Analysis:\nNYPD vs LAPD", fontsize=18, color='white', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig
