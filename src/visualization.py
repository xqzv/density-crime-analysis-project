import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import contextily as ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from datetime import datetime

# --- Theme Configuration ---
THEME_CONFIG = {
    'font_family': "Arial, sans-serif",
    'title_font_size': 18,
    'axis_title_font_size': 14,
    'tick_font_size': 12,
    'margin': dict(l=40, r=40, t=60, b=80),
    'height': 450,
    'template': 'plotly_white',
    'colors': {
        'NYPD': '#1f77b4',
        'LAPD': '#ff7f0e',
        'NYPD_light': '#85c2f0',
        'LAPD_light': '#ffc681'
    }
}

def apply_chart_theme(fig: go.Figure, title: str, xaxis_title: str = None, yaxis_title: str = None) -> go.Figure:
    """Apply consistent theme to Plotly figures."""
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': THEME_CONFIG['title_font_size']}
        },
        font=dict(family=THEME_CONFIG['font_family']),
        template=THEME_CONFIG['template'],
        margin=THEME_CONFIG['margin'],
        height=THEME_CONFIG['height'],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    if xaxis_title:
        fig.update_xaxes(title_text=xaxis_title, title_font=dict(size=THEME_CONFIG['axis_title_font_size']))
    if yaxis_title:
        fig.update_yaxes(title_text=yaxis_title, title_font=dict(size=THEME_CONFIG['axis_title_font_size']))
        
    return fig

def plot_crime_by_weekday(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame, 
                        nypd_color: str = THEME_CONFIG['colors']['NYPD'], 
                        lapd_color: str = THEME_CONFIG['colors']['LAPD']) -> go.Figure:
    """Plot crime frequency by day of week using Plotly."""
    
    def process_weekdays(df, dept_name):
        dates = pd.to_datetime(df[['Arrest_Year', 'Arrest_Month', 'Arrest_Day']].rename(
            columns={'Arrest_Year': 'year', 'Arrest_Month': 'month', 'Arrest_Day': 'day'}), errors='coerce')
        weekdays = dates.dt.dayofweek.dropna().astype(int)
        
        days_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                    4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        
        day_names = weekdays.map(days_map)
        counts = day_names.value_counts()
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return [counts.get(day, 0) for day in days_order], days_order

    nypd_counts, days_order = process_weekdays(nypd_df, 'NYPD')
    lapd_counts, _ = process_weekdays(lapd_df, 'LAPD')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=days_order, y=nypd_counts, name='NYPD', marker_color=nypd_color
    ))
    fig.add_trace(go.Bar(
        x=days_order, y=lapd_counts, name='LAPD', marker_color=lapd_color
    ))

    fig.update_layout(barmode='group')
    return apply_chart_theme(fig, 'Crime Frequency by Day of Week', 'Day of Week', 'Number of Crimes')

def plot_crime_by_month(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame,
                      nypd_color: str = THEME_CONFIG['colors']['NYPD'], 
                      lapd_color: str = THEME_CONFIG['colors']['LAPD']) -> go.Figure:
    """Plot crime frequency by month using Plotly."""
    
    def get_month_counts(df):
        counts = df['Arrest_Month'].value_counts().sort_index()
        # Ensure all months 1-12 exist
        return counts.reindex(range(1, 13), fill_value=0)

    nypd_counts = get_month_counts(nypd_df)
    lapd_counts = get_month_counts(lapd_df)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure()
    
    # Add seasonal background shapes
    shapes = [
        dict(type="rect", x0=-0.5, x1=1.5, y0=0, y1=1, xref="x", yref="paper", fillcolor="lightblue", opacity=0.1, layer="below", line_width=0),
        dict(type="rect", x0=10.5, x1=11.5, y0=0, y1=1, xref="x", yref="paper", fillcolor="lightblue", opacity=0.1, layer="below", line_width=0),
        dict(type="rect", x0=1.5, x1=4.5, y0=0, y1=1, xref="x", yref="paper", fillcolor="lightgreen", opacity=0.1, layer="below", line_width=0),
        dict(type="rect", x0=4.5, x1=7.5, y0=0, y1=1, xref="x", yref="paper", fillcolor="yellow", opacity=0.1, layer="below", line_width=0),
        dict(type="rect", x0=7.5, x1=10.5, y0=0, y1=1, xref="x", yref="paper", fillcolor="orange", opacity=0.1, layer="below", line_width=0),
    ]
    
    fig.add_trace(go.Scatter(
        x=month_names, y=nypd_counts, name='NYPD',
        mode='lines+markers', line=dict(color=nypd_color, width=3), marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=month_names, y=lapd_counts, name='LAPD',
        mode='lines+markers', line=dict(color=lapd_color, width=3), marker=dict(size=8)
    ))

    fig.update_layout(shapes=shapes)
    return apply_chart_theme(fig, 'Crime Frequency by Month', 'Month', 'Number of Crimes')

def plot_crime_by_year(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame,
                     nypd_color: str = THEME_CONFIG['colors']['NYPD'], 
                     lapd_color: str = THEME_CONFIG['colors']['LAPD']) -> go.Figure:
    """Plot crime frequency by year using Plotly."""
    
    def get_year_counts(df):
        return df['Arrest_Year'].value_counts().sort_index()

    nypd_counts = get_year_counts(nypd_df)
    lapd_counts = get_year_counts(lapd_df)
    
    years = sorted(list(set(nypd_counts.index) | set(lapd_counts.index)))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=nypd_counts.reindex(years, fill_value=0),
        name='NYPD', marker_color=nypd_color
    ))
    fig.add_trace(go.Bar(
        x=years, y=lapd_counts.reindex(years, fill_value=0),
        name='LAPD', marker_color=lapd_color
    ))

    # Add trend lines (simple linear regression)
    for name, counts, color in [('NYPD', nypd_counts, 'darkblue'), ('LAPD', lapd_counts, 'darkred')]:
        if len(counts) > 1:
            x = counts.index.values
            y = counts.values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=x, y=p(x), mode='lines', name=f'{name} Trend',
                line=dict(color=color, dash='dash', width=2),
                showlegend=False
            ))

    fig.update_layout(barmode='group')
    fig.update_xaxes(type='category') # Ensure years are treated as categories
    return apply_chart_theme(fig, 'Crime Frequency by Year', 'Year', 'Number of Crimes')

def plot_crime_by_day_of_month(nypd_df: pd.DataFrame, lapd_df: pd.DataFrame,
                             nypd_color: str = THEME_CONFIG['colors']['NYPD'], 
                             lapd_color: str = THEME_CONFIG['colors']['LAPD']) -> go.Figure:
    """Plot crime frequency by day of month using Plotly."""
    
    def get_day_counts(df):
        counts = df['Arrest_Day'].value_counts().sort_index()
        full_counts = counts.reindex(range(1, 32), fill_value=0)
        rolling = full_counts.rolling(window=3, center=True).mean()
        return full_counts, rolling

    nypd_counts, nypd_rolling = get_day_counts(nypd_df)
    lapd_counts, lapd_rolling = get_day_counts(lapd_df)
    
    days = list(range(1, 32))

    fig = go.Figure()
    
    # Scatter points for daily counts
    fig.add_trace(go.Scatter(
        x=days, y=nypd_counts, mode='markers', name='NYPD (Daily)',
        marker=dict(color=nypd_color, opacity=0.3, size=6), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=days, y=lapd_counts, mode='markers', name='LAPD (Daily)',
        marker=dict(color=lapd_color, opacity=0.3, size=6), showlegend=False
    ))
    
    # Lines for rolling average
    fig.add_trace(go.Scatter(
        x=days, y=nypd_rolling, mode='lines', name='NYPD (3-day avg)',
        line=dict(color=nypd_color, width=3)
    ))
    fig.add_trace(go.Scatter(
        x=days, y=lapd_rolling, mode='lines', name='LAPD (3-day avg)',
        line=dict(color=lapd_color, width=3)
    ))

    # Background regions
    shapes = [
        dict(type="rect", x0=1, x1=10, y0=0, y1=1, xref="x", yref="paper", fillcolor="green", opacity=0.1, layer="below", line_width=0),
        dict(type="rect", x0=11, x1=20, y0=0, y1=1, xref="x", yref="paper", fillcolor="blue", opacity=0.1, layer="below", line_width=0),
        dict(type="rect", x0=21, x1=31, y0=0, y1=1, xref="x", yref="paper", fillcolor="red", opacity=0.1, layer="below", line_width=0),
    ]
    
    fig.update_layout(shapes=shapes)
    return apply_chart_theme(fig, 'Crime Frequency by Day of Month', 'Day of Month', 'Number of Crimes')



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



def plot_race_distribution(df1: pd.DataFrame, df2: pd.DataFrame, 
                         df1_name: str = "NYPD", df2_name: str = "LAPD",
                         color1: str = THEME_CONFIG['colors']['NYPD'], 
                         color2: str = THEME_CONFIG['colors']['LAPD']) -> go.Figure:
    """Plot race distribution comparison using Plotly."""
    race_categories = ['Black', 'Hispanic', 'White', 'Asian/Pacific Islander',
                      'Other', 'Native American', 'Unknown']
    
    race_pct1 = (df1['Race_Std'].value_counts() / len(df1)) * 100
    race_pct2 = (df2['Race_Std'].value_counts() / len(df2)) * 100
    
    race_pct1 = race_pct1.reindex(race_categories, fill_value=0)
    race_pct2 = race_pct2.reindex(race_categories, fill_value=0)
    
    # Sort by total percentage for better visualization
    total_pct = race_pct1 + race_pct2
    sorted_cats = total_pct.sort_values(ascending=True).index.tolist()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_cats,
        x=race_pct1[sorted_cats],
        name=df1_name,
        orientation='h',
        marker_color=color1
    ))
    fig.add_trace(go.Bar(
        y=sorted_cats,
        x=race_pct2[sorted_cats],
        name=df2_name,
        orientation='h',
        marker_color=color2
    ))
    
    fig.update_layout(barmode='group')
    return apply_chart_theme(fig, "Race Distribution", "Percentage (%)")

def plot_gender_distribution(df1: pd.DataFrame, df2: pd.DataFrame,
                           df1_name: str = "NYPD", df2_name: str = "LAPD",
                           color1: str = THEME_CONFIG['colors']['NYPD'], 
                           color2: str = THEME_CONFIG['colors']['LAPD']) -> go.Figure:
    """Plot gender distribution comparison using Plotly."""
    gender_pct1 = (df1['Gender_Std'].value_counts() / len(df1)) * 100
    gender_pct2 = (df2['Gender_Std'].value_counts() / len(df2)) * 100
    
    # Prepare data
    labels = ['Male', 'Female']
    values1 = [gender_pct1.get('Male', 0), gender_pct1.get('Female', 0)]
    values2 = [gender_pct2.get('Male', 0), gender_pct2.get('Female', 0)]
    
    # Lighter shades for female
    color1_light = THEME_CONFIG['colors']['NYPD_light']
    color2_light = THEME_CONFIG['colors']['LAPD_light']
    
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                       subplot_titles=[df1_name, df2_name])
    
    fig.add_trace(go.Pie(
        labels=labels, values=values1, name=df1_name,
        marker_colors=[color1, color1_light],
        hole=.6, hoverinfo="label+percent+name"
    ), 1, 1)
    
    fig.add_trace(go.Pie(
        labels=labels, values=values2, name=df2_name,
        marker_colors=[color2, color2_light],
        hole=.6, hoverinfo="label+percent+name"
    ), 1, 2)
    
    # Apply theme manually since subplots are tricky with generic update_layout
    fig = apply_chart_theme(fig, "Gender Distribution")
    # Adjust legend for pie charts specifically if needed, but theme default is okay
    return fig

def plot_age_distribution(df1: pd.DataFrame, df2: pd.DataFrame,
                        df1_name: str = "NYPD", df2_name: str = "LAPD",
                        color1: str = THEME_CONFIG['colors']['NYPD'], 
                        color2: str = THEME_CONFIG['colors']['LAPD']) -> go.Figure:
    """Plot age distribution comparison using Plotly."""
    age_categories = ['<18', '18-24', '25-44', '45-64', '65+']
    
    age_pct1 = (df1['Age_Category_Std'].value_counts() / len(df1)) * 100
    age_pct2 = (df2['Age_Category_Std'].value_counts() / len(df2)) * 100
    
    age_pct1 = age_pct1.reindex(age_categories, fill_value=0)
    age_pct2 = age_pct2.reindex(age_categories, fill_value=0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=age_categories, y=age_pct1,
        mode='lines+markers',
        name=df1_name,
        line=dict(color=color1, width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=age_categories, y=age_pct2,
        mode='lines+markers',
        name=df2_name,
        line=dict(color=color2, width=3),
        marker=dict(size=8)
    ))
    
    return apply_chart_theme(fig, "Age Distribution", "Age Group", "Percentage (%)")

def plot_offense_distribution(df1: pd.DataFrame, df2: pd.DataFrame,
                            df1_name: str = "NYPD", df2_name: str = "LAPD",
                            color1: str = THEME_CONFIG['colors']['NYPD'], 
                            color2: str = THEME_CONFIG['colors']['LAPD']) -> go.Figure:
    """Plot offense type distribution comparison using Plotly."""
    offense_categories = ['Violent Crime', 'Property Crime', 'Drug Offense',
                        'Weapon Offense', 'Traffic Violation', 'Other']
                        
    offense_pct1 = (df1['Offense_Std'].value_counts() / len(df1)) * 100
    offense_pct2 = (df2['Offense_Std'].value_counts() / len(df2)) * 100
    
    offense_pct1 = offense_pct1.reindex(offense_categories, fill_value=0)
    offense_pct2 = offense_pct2.reindex(offense_categories, fill_value=0)
    
    # Sort by difference to highlight contrasts
    offense_diff = abs(offense_pct1 - offense_pct2)
    sorted_offense = offense_diff.sort_values(ascending=True).index.tolist()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_offense,
        x=offense_pct1[sorted_offense],
        name=df1_name,
        orientation='h',
        marker_color=color1
    ))
    fig.add_trace(go.Bar(
        y=sorted_offense,
        x=offense_pct2[sorted_offense],
        name=df2_name,
        orientation='h',
        marker_color=color2
    ))
    
    fig.update_layout(barmode='group')
    return apply_chart_theme(fig, "Offense Type Distribution", "Percentage (%)")
