import dash
from dash import Dash, html, dcc, Input, Output, State, no_update, callback_context
import plotly.express as px
import pandas as pd
import os
import json
import plotly.graph_objects as go
import warnings
import asyncio
import functools
import nest_asyncio
from shapely.geometry import shape
import time

warnings.filterwarnings("ignore")

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# # Initialize the Dash app
# service_prefix = os.getenv("JUPYTERHUB_SERVICE_PREFIX")
# port = 50003
# server_url = "https://jupyter.nersc.gov"
# app = Dash(__name__, requests_pathname_prefix=f"{service_prefix}proxy/{port}/")

# Mapbox access token
mapbox_access_token = "pk.eyJ1IjoibGFwaGF0cmFkMDIiLCJhIjoiY2x5ZjFkdTZqMDM4cjJxcHh4dW9oNHd6dSJ9.cbUxYCzotjR4xeDr0fD-Iw"
app = dash.Dash(__name__)
# Folder paths and file names
demographic_file='data/demo/2020-demographic-info.csv'
shapefile_name = 'data/shapefile/county.json'
years = list(range(2000, 2021))

# Read demographic data
demographic_data = pd.read_csv(demographic_file, encoding='latin1')

# Load the GeoJSON file
with open(shapefile_name) as f:
    geojson_data = json.load(f)

# Simplify the GeoJSON data
def simplify_geojson(geojson, tolerance=0.01):
    from shapely.geometry import shape
    from shapely.geometry import mapping
    for feature in geojson['features']:
        geom = shape(feature['geometry'])
        simplified_geom = geom.simplify(tolerance, preserve_topology=True)
        feature['geometry'] = mapping(simplified_geom)
    return geojson

geojson_data = simplify_geojson(geojson_data)

# Add a new column in geojson_data for the last 5 digits of 'GEO_ID'
for feature in geojson_data['features']:
    geo_id = feature['properties']['GEO_ID']
    county_id = geo_id[-5:]
    feature['properties']['county_id'] = county_id
    geom = shape(feature['geometry'])
    centroid = geom.centroid
    feature['properties']['Latitude'] = centroid.y
    feature['properties']['Longitude'] = centroid.x

# Create FIPS code in demographic data
demographic_data['FIPS'] = demographic_data['STATE'].astype(str).str.zfill(2) + demographic_data['COUNTY'].astype(str).str.zfill(3)
demographic_data['county_id'] = demographic_data['FIPS'].astype(str)

# Cache for data loading
cache = {}

# Function to load data for the selected year
@functools.lru_cache(maxsize=128)
def load_data(year):
    if year in cache:
        return cache[year]
    
    if year in [2016, 2017, 2018]:
        sdoh_file = (f'data/SDOH_demo_Dash/{year}.csv')
    else:
        sdoh_file = (f'data/SDOH_demo_Dash/2018.csv')  # Use 2018 as default for other years

    suicide_file = (f'data/suicide_n/{year}.csv')
    
    sdoh_data = pd.read_csv(sdoh_file)
    suicide_data = pd.read_csv(suicide_file)

    # Create new columns in suicide_data and sdoh_data
    suicide_data['county_id'] = suicide_data['fips'].astype(str).str.zfill(5)
    sdoh_data['county_id'] = sdoh_data['FIPSCODE'].astype(str).str.zfill(5)
    
    # Merge suicide_data with sdoh_data on 'county_id'
    merged_data = pd.merge(suicide_data, sdoh_data, on='county_id')
    
    # Merge the demographic data with the already merged data
    merged_data = pd.merge(merged_data, demographic_data, on='county_id', how='left')
    
    # Convert geojson_data to a DataFrame for merging
    geojson_properties_df = pd.DataFrame([feature['properties'] for feature in geojson_data['features']])
    
    # Merge the GeoJSON data with the merged suicide and SDOH data
    final_merged_data = pd.merge(merged_data, geojson_properties_df, on='county_id', how='left')
    
    # Add Year column
    final_merged_data['Year'] = year

    cache[year] = final_merged_data
    return final_merged_data

# Function to get top 5 counties by average suicide rate per 100,000
def get_top_5_counties(data):
    filtered_data = data[data['SuicideDeathRate'] <= 300]
    top_5_counties = filtered_data.nlargest(5, 'SuicideDeathRate')['county_id'].tolist()
    return top_5_counties

def create_choropleth_map(data):
    # Ensure the 'SuicideDeathRate' column exists and normalize it to per 100,000
    if 'SuicideDeathRate' not in data.columns:
        data['SuicideDeathRate'] = (data['Deaths'] / data['POPESTIMATE2020']) * 100000

    # Add a column indicating missing data
    data['missing_data'] = data['Deaths'].isna()

    # Create hover text
    data['hover_text'] = (
        'County: ' + data['CTYNAME'].astype(str) + '<br>' +
        'State: ' + data['STNAME'].astype(str) + '<br>' +
        'Population: ' + data['POPESTIMATE2020'].map('{:,.0f}'.format) + '<br>' +
        'Suicide Rate: ' + data['SuicideDeathRate'].map(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')
    )

    # Define the custom color scale
    custom_color_scale = [
        [0.0, 'lightgray'],   # 0 values to light gray
        [0.00001, 'lightyellow'],  # Smallest positive value to start the color scale
        [0.25, 'lightgoldenrodyellow'],
        [0.5, 'orange'],
        [0.75, 'red'],
        [1.0, 'maroon']
    ]

    # Create the choropleth map using Plotly's express module
    fig = px.choropleth_mapbox(
        data,
        geojson=geojson_data,
        locations='county_id',
        color='SuicideDeathRate',
        color_continuous_scale=custom_color_scale,
        range_color=(5, 30),
        hover_name=None,  # Disable default hover info
        hover_data={'county_id': False, 'SuicideDeathRate': False, 'hover_text': True, 'Year': False},  # Ensure only hover_text is shown
        custom_data=['hover_text'],  
        mapbox_style="light",
        center={"lat": 37.0902, "lon": -95.7129},
        zoom=3.4
    )

    # Update traces to set the marker_line_width to 0 and customize hover template
    fig.update_traces(
        marker_line_width=0,
        hovertemplate='%{customdata[0]}<extra></extra>'
    )

    # Update layout for play/pause buttons and enable selection tools
    fig.update_layout(
        mapbox_accesstoken=mapbox_access_token,
        height=800,
        width=1350,
        margin={"r": 0, "t": 60, "l": 40, "b": 10},
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_color="black"
        ),
        coloraxis_colorbar=dict(
            title="Suicide Rate (per 100,000)" 
        ),
        dragmode='lasso',  # Allow lasso selection on the map
    )

    return fig

def update_bar_graph(selected_year, selected_counties, selected_category, selected_data):
    data = load_data(selected_year)
    
    # Ensure selected_counties is a list
    if selected_counties is None:
        selected_counties = []

    # Process selected_data to get selected counties
    if selected_data and 'points' in selected_data:
        selected_counties = [point['location'] for point in selected_data['points']]

    # If no counties selected, use top 5 counties
    if not selected_counties:
        selected_counties = get_top_5_counties(data)

    # Limit the number of selected counties to 5
    displayed_counties = selected_counties[:5]

    # Ensure we only process the selected counties that exist in the data
    county_data = data[data['county_id'].isin(displayed_counties)]

    # Define the columns for each category
    categories = {
        'Geospatial': ['elevation', 'duration_mins_decimal', 'AHRF_USDA_RUCC_2013'],
        'Demographic': ['ACS_PCT_NON_CITIZEN', 'ACS_PCT_DIVORCE_SEPARAT', 'ACS_PCT_VA_DISABLE'],
        'Poverty': ['ACS_PCT_MOBILE_HOME', 'ACS_PCT_VA_POOR', 'ACS_PCT_NONVA_POOR'],
        'Food Insecurity': ['ACS_PCT_FOOD_STAMP', 'ACS_PCT_HH_PUB_ASSIST'],
        'Crime': ['Violent crime raw value'],
        'Mental Health': ['Frequent mental distress raw value'],
        'Environment': ['mean_temp', 'PM2.5', 'temperature_humid_index']
    }
    
    columns = categories[selected_category]

    # Prepare data for the bar graph with individual columns
    bar_data = []
    for county in displayed_counties:
        county_info = county_data[county_data['county_id'] == county]
        if not county_info.empty:
            for col in columns:
                bar_data.append({
                    'Metric': col,
                    'Value': county_info[col].values[0] if col in county_info and len(county_info[col].values) > 0 and county_info[col].values[0] > 0 else 0,
                    'County': county_info['CTYNAME'].values[0],
                    'State': county_info['STNAME'].values[0],
                    'Suicides': county_info['Deaths'].values[0],
                    'Suicide Rate': f"{county_info['SuicideDeathRate'].values[0]:.0f}" if not pd.isna(county_info['SuicideDeathRate'].values[0]) else 'N/A'
                })

    # Create the DataFrame
    bar_df = pd.DataFrame(bar_data)
    
    # Ensure 'Metric' is in the DataFrame
    if 'Metric' not in bar_df.columns:
        return go.Figure()

    # Define custom color sequence
    custom_colors = ["#1c0c84", "#bc3484", "#8012a9", "#f38b44", "#f3cd38"]

    # Create the bar graph
    fig = px.bar(
        bar_df,
        x='Metric',
        y='Value',
        color='County',
        barmode='group',
        title=f"Social Determinants for {', '.join(bar_df['County'].unique())}",
        labels={'Value': 'Average Value'},
        template='plotly_white',
        color_discrete_sequence=custom_colors  # Use custom color sequence
    )

    # Customize hover labels to show information based on its county name
    hover_template = (
        "%{x}: %{y:.2f}"
    )

    fig.update_traces(
        hovertemplate=hover_template,
        customdata=bar_df[['County', 'State', 'Suicides', 'Suicide Rate']].values,
        hoverlabel=dict(
            bgcolor='white',  # Set the background color of the hover label to white
            font_size=12,
            font_family='Open Sans, sans-serif'
        )
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title='Social Determinants',
        yaxis_title='Value',
        template='plotly_white',  # Use a clean white template
        title={
            'text': f"Social Determinants for {', '.join(bar_df['County'].unique())}",
            'x': 0.5,  # Center the title horizontally
            'xanchor': 'center',
            'font': {'size': 14, 'color': 'black', 'family': 'Open Sans, sans-serif'}
        },
        xaxis=dict(
            tickangle=-45,  # Rotate x-axis labels for better readability
            title_standoff=10,  # Distance between axis title and axis
        ),
        yaxis=dict(
            title_standoff=10,
            gridcolor='lightgrey',  # Add gridlines for y-axis
            zerolinecolor='grey'  # Line for y=0
        ),
        font=dict(
            family="Open Sans, sans-serif",
            size=14,
            color="black"
        ),
        margin=dict(
            l=50, r=50, t=80, b=100  # Increase top margin for better spacing
        ),
        bargap=0.5,  # Gap between bars
        paper_bgcolor='white',  # Background color outside the plot
        plot_bgcolor='white',  # Background color of the plot
        showlegend=True  # Show legend to differentiate counties
    )

    return fig

def trend_bar_graph(selected_year, selected_counties, color_map):
    custom_colors = px.colors.qualitative.Vivid

    # Collect data for all years for the selected counties
    all_years_data = []
    for year in years:
        data = load_data(year)
        data['Year'] = year
        all_years_data.append(data)

    # Concatenate data for all years
    trend_data = pd.concat(all_years_data, ignore_index=True)

    # If no counties selected, use top 5 counties
    if not selected_counties:
        selected_counties = get_top_5_counties(trend_data)

    # Filter data for selected counties
    trend_data = trend_data[trend_data['county_id'].isin(selected_counties)]

    # Group by year and county, then calculate the average suicide rate for each year
    trend_data = trend_data.groupby(['Year', 'county_id']).agg({
        'SuicideDeathRate': 'mean'
    }).reset_index()

    # Replace county_id with county name for better readability
    trend_data = trend_data.merge(demographic_data[['county_id', 'CTYNAME']], on='county_id', how='left')

    # Create the trend bar graph
    fig = px.bar(
        trend_data,
        x='Year',
        y='SuicideDeathRate',
        color='CTYNAME',
        barmode='group',
        title="Suicide Rate Trends Across Years",
        labels={'SuicideDeathRate': 'Suicide Rate', 'Year': 'Year', 'CTYNAME': 'County'},
        template='plotly_white',
        color_discrete_map=color_map  # Use consistent color map
    )

    # Customize hover labels
    hover_template = (
        "Year: %{x}<br>"
        "Suicide Rate: %{y:.0f}"
    )

    fig.update_traces(
        hovertemplate=hover_template,
        customdata=trend_data[['CTYNAME']].values if not trend_data.empty else [],
        hoverlabel=dict(
            bgcolor='white',  # Set the background color of the hover label to white
            font_size=12,
            font_family='Open Sans, sans-serif'
        )
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Suicide Rate (per 100,000)',
        template='plotly_white',  # Use a clean white template
        title={
            'text': "Suicide Rate Trends Across Years",
            'x': 0.5,  # Center the title horizontally
            'xanchor': 'center',
            'font': {'size': 24, 'color': 'black', 'family': 'Open Sans, sans-serif'}
        },
        xaxis=dict(
            tickmode='linear',  # Show every year on the x-axis
            tick0=min(years),
            dtick=1,
            title_standoff=10,  # Distance between axis title and axis
        ),
        yaxis=dict(
            title_standoff=10,
            gridcolor='lightgrey',  # Add gridlines for y-axis
            zerolinecolor='grey'  # Line for y=0
        
        ),
        font=dict(
            family="Open Sans, sans-serif",
            size=14,
            color="black"
        ),
        margin=dict(
            l=50, r=50, t=100, b=100  # Increase margins for better spacing
        ),
        bargap=0.5,  # Gap between bars
        paper_bgcolor='white',  # Background color outside the plot
        plot_bgcolor='white',  # Background color of the plot
        showlegend=True  # Show legend to differentiate counties
    )

    return fig

def summary_sunburst(data):
    # Define the values for each category
    category_values = {
        'Geospatial': 18.75,
        'Demographic': 18.75,
        'Poverty': 18.75,
        'Food Insecurity': 12.5,
        'Crime': 6.25,
        'Mental Health': 6.25,
        'Environment': 18.75
    }

    # Define the values for each metric
    metric_values = {
        'Geospatial': {'elevation': 6.25, 'duration_mins_decimal': 6.25, 'AHRF_USDA_RUCC_2013': 6.25},
        'Demographic': {'ACS_PCT_NON_CITIZEN': 6.25, 'ACS_PCT_DIVORCE_SEPARAT': 6.25, 'ACS_PCT_VA_DISABLE': 6.25},
        'Poverty': {'ACS_PCT_MOBILE_HOME': 6.25, 'ACS_PCT_VA_POOR': 6.25, 'ACS_PCT_NONVA_POOR': 6.25},
        'Food Insecurity': {'ACS_PCT_FOOD_STAMP': 6.25, 'ACS_PCT_HH_PUB_ASSIST': 6.25},
        'Crime': {'Violent crime raw value': 6.25},
        'Mental Health': {'Frequent mental distress raw value': 6.25},
        'Environment': {'mean_temp': 6.25, 'PM2.5': 6.25, 'temperature_humid_index': 6.25}
    }

    # Prepare sunburst data including all counties
    sunburst_data = []
    for category, cols in metric_values.items():
        for col, value in cols.items():
            sunburst_data.append({
                'Category': category,
                'Metric': col,
                'Value': value
            })

    sunburst_df = pd.DataFrame(sunburst_data)

    # Define new custom color scale for the categories
    custom_colors = {
        'Geospatial': '#FF6347',  # Tomato
        'Demographic': '#4682B4',  # SteelBlue
        'Poverty': '#32CD32',  # LimeGreen
        'Food Insecurity': '#FFD700',  # Gold
        'Crime': '#8B0000',  # DarkRed
        'Mental Health': '#9400D3',  # DarkViolet
        'Environment': '#FF8C00'  # DarkOrange
    }

    # Create the sunburst chart
    fig = px.sunburst(
        sunburst_df,
        path=['Category', 'Metric'],
        values='Value',
        title="Average Social Determinants of Health Across All US Counties (2000-2020)",
        color='Category',
        color_discrete_map=custom_colors  # Use custom color sequence
    )

    # Customize the hovertemplate
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Category: %{parent}<br>Value: %{value:.2f}%<extra></extra>'
    )

    # Custom legend items
    legend_items = [
        {'name': 'Geospatial', 'color': custom_colors['Geospatial']},
        {'name': 'Demographic', 'color': custom_colors['Demographic']},
        {'name': 'Poverty', 'color': custom_colors['Poverty']},
        {'name': 'Food Insecurity', 'color': custom_colors['Food Insecurity']},
        {'name': 'Crime', 'color': custom_colors['Crime']},
        {'name': 'Mental Health', 'color': custom_colors['Mental Health']},
        {'name': 'Environment', 'color': custom_colors['Environment']}
    ]

    # Adding custom legend to the figure
    annotations = []
    for i, item in enumerate(legend_items):
        annotations.append(
            dict(
                x=1.05,  # Adjusted x position to fit within the figure
                y=1 - (i * 0.06),  # Adjusted y position to fit all legend items
                xref='paper',
                yref='paper',
                showarrow=False,
                text=item["name"],
                xanchor='left',
                align='left',
                font=dict(size=12, color='white'),  # Set font color to white
                bgcolor=item["color"],  # Set background color to match category color
                bordercolor=item["color"]  # Set border color to match category color
            )
        )

    fig.update_layout(
        height=800,  # Increase height
        margin=dict(l=50, r=150, t=100, b=100),  # Increase right margin to fit legend
        annotations=annotations  # Add custom legend
    )

    return fig

def scatter_graph(selected_year, selected_counties, color_map):
    custom_colors = px.colors.qualitative.Vivid

    # Collect data for all years for the selected counties
    all_years_data = []
    for year in years:
        data = load_data(year)
        data['Year'] = year
        all_years_data.append(data)

    # Concatenate data for all years
    full_data = pd.concat(all_years_data, ignore_index=True)

    # If no counties selected, use top 5 counties
    if not selected_counties:
        selected_counties = get_top_5_counties(full_data)

    # Filter data for selected counties
    filtered_data = full_data[full_data['county_id'].isin(selected_counties)]

    # Group by year and county, then calculate the average suicide rate for each year
    grouped_data = filtered_data.groupby(['Year', 'county_id']).agg({
        'SuicideDeathRate': 'mean'
    }).reset_index()

    # Replace county_id with county name for better readability
    grouped_data = grouped_data.merge(demographic_data[['county_id', 'CTYNAME']], on='county_id', how='left')

    # Create a figure
    fig = go.Figure()

    # Add a trace for each selected county
    for county in selected_counties:
        county_data = grouped_data[grouped_data['county_id'] == county]
        fig.add_trace(go.Scatter(
            x=county_data['Year'],
            y=county_data['SuicideDeathRate'],
            mode='lines+markers',
            name=county_data['CTYNAME'].values[0],  # Use county name for legend
            line=dict(color=color_map.get(county)),  # Set color from color_map
            hovertemplate="Year: %{x}<br>Suicide Rate: %{y:.0f}<extra></extra>"  # Format y-values without decimal points
        ))

    # Update layout to show all years from 2000 to 2020 on the x-axis
    fig.update_layout(
        title="Suicide Rates Over Years",
        xaxis_title="Year",
        yaxis_title="Suicide Rate (per 100,000)",
        xaxis=dict(
            tickmode='linear',
            tick0=2000,
            dtick=1,
            range=[2000, 2020],  # Set the range for the x-axis
            tickvals=list(range(2000, 2021)),  # Ensure ticks for each year
            ticktext=[str(year) for year in range(2000, 2021)],  # Set tick text for each year
            gridcolor='lightgrey',
            gridwidth=1,  # Set grid width
            ticklen=5  # Set tick length
        ),
        yaxis=dict(
            title_standoff=10,
            gridcolor='lightgrey',  # Add gridlines for y-axis
            gridwidth=1,  # Set grid width
            zerolinecolor='grey',  # Line for y=0
            ticklen=5  # Set tick length
        ),
        template='plotly_white',
        font=dict(
            family="Open Sans, sans-serif",
            size=14,
            color="black"
        ),
        margin=dict(
            l=50, r=50, t=100, b=100  # Increase margins for better spacing
        ),
        paper_bgcolor='white',  # Background color outside the plot
        plot_bgcolor='white',  # Background color of the plot
        showlegend=True  # Show legend to differentiate counties
    )

    return fig

def update_highest_suicide_rates_bar(data, selected_counties, selected_data):
    # Process selected_data to get selected counties
    if selected_data and 'points' in selected_data:
        selected_counties = [point['location'] for point in selected_data['points']]

    # Filter data for the selected counties
    filtered_df = data[data['county_id'].isin(selected_counties)]

    # Aggregate data
    if filtered_df.empty:
        # If no counties are selected, show aggregated data for all counties
        agg_df = data.groupby('CTYNAME')['SuicideDeathRate'].mean().reset_index()
    else:
        agg_df = filtered_df.groupby('CTYNAME')['SuicideDeathRate'].mean().reset_index()

    agg_df.columns = ['County', 'SuicideDeathRate']  # Ensure the column names are correct

    # Sort values from highest to lowest and select top 50
    top_agg_df = agg_df.sort_values(by='SuicideDeathRate', ascending=False).head(50)
    
    # Create the bar chart
    fig = px.bar(
        top_agg_df,
        x='County',  # Use the correct column name
        y='SuicideDeathRate',
        title='Counties with Highest Suicide Death Rates',
        labels={'SuicideDeathRate': 'Suicide Death Rate (per 100,000)', 'County': 'County'},  # Update labels
        template='plotly_white',
        color_discrete_sequence=['#8B0000']
    )
    
    # Customize hover text to show values with 2 decimal points
    fig.update_traces(
        hovertemplate='County: %{x}<br>Suicide Death Rate: %{y:.2f}<extra></extra>'
    )
    
    # Update layout for better style
    fig.update_layout(
        xaxis=dict(
            title='County',
            tickangle=-45,
            title_standoff=10
        ),
        yaxis=dict(
            title='Suicide Death Rate (per 100,000)',
            title_standoff=10
        ),
        margin=dict(l=50, r=50, t=50, b=100)
    )

    return fig
# Function to get explanatory notes based on category
def get_category_notes(category):
    notes = {
        'Geospatial': """

        - elevation: Elevation data from USGS.
        - duration_mins_decimal: Driving hours from centroid county to nearest VA facility.
        - AHRF_USDA_RUCC_2013: USDA Rural-Urban Continuum Codes.
        """,
        'Demographic': """

        - ACS_PCT_NON_CITIZEN: Percentage of the population that are non-citizens.
        - ACS_PCT_DIVORCE_SEPARAT: Percentage of the population that is divorced or separated.
        - ACS_PCT_VA_DISABLE: Percentage of the veteran population with a disability.
        """,
        'Poverty': """
   
        - ACS_PCT_MOBILE_HOME: Percentage of housing units that are mobile homes.
        - ACS_PCT_VA_POOR: Percentage of the population living in poverty that are veterans.
        - ACS_PCT_NONVA_POOR: Percentage of the population living in poverty that are non-veterans.
        """,
        'Food Insecurity': """
    
        - ACS_PCT_FOOD_STAMP: Percentage of households receiving food stamps.
        - ACS_PCT_HH_PUB_ASSIST: Percentage of households receiving public assistance.
        """,
        'Crime': """

        - Violent crime raw value: Violent crimes.
        """,
        'Mental Health': """

        - Frequent mental distress raw value: Individuals reporting frequent mental distress.
        """,
        'Environment': """

        - mean_temp: The average temperature.
        - PM2.5: Particulate Matter 2.5.
        - temperature_humid_index: Temperature Humidity Index.
        """
    }
    return notes.get(category, "")

# Layout of the app
app.layout = html.Div([
    html.H1("Suicide Rates with Social Determinants", style={'textAlign': 'center'}),

    dcc.Slider(
        id='year-slider',
        min=min(years),
        max=max(years),
        value=2020,  # Default value
        marks={year: str(year) for year in years},
        step=None,
        updatemode='drag',
        tooltip={"placement": "bottom", "always_visible": True},
        className="thicker-slider"
    ),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=dcc.Graph(id='choropleth-map', style={'position': 'relative', 'height': '80vh'}),
        fullscreen=True
    ),
    html.Div("Click-Drag on the map to select counties", style={'textAlign': 'bottom', 'marginBottom': '10px'}),
    html.Br(),
    html.Div([
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        dcc.Store(id='memory-output'),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        dcc.Dropdown(
            id='county-dropdown',
            options=[
                {'label': f"{row['CTYNAME']}, {row['STNAME']}", 'value': row['county_id']}
                for _, row in demographic_data.iterrows()
            ],
            multi=True,
            placeholder="Please Select",
            maxHeight=200,  # Limit the dropdown height
            className="dropdown",
            style={
                'width': '70%',
                'height': '40px',  # Adjust the height to make the dropdown menu box bigger
                'fontSize': '15px'
            }
        ),
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '100vh'}),
    html.Div([
        html.Div([
            html.Button('Confirm Your Selection', id='confirm-button', n_clicks=0, style={
                'backgroundColor': '#007bff',
                'color': 'white',  # White text
                'border': 'none',  # Blue border
                'padding': '10px 20px',  # Padding
                'textAlign': 'center',  # Center text
                'textDecoration': 'none',  # Remove underline
                'display': 'inline-block',  # Inline block
                'fontSize': '14px',  # Font size
                'borderRadius': '5px',  # Rounded corners
                'cursor': 'pointer'  # Pointer cursor
            })  # Customized confirm button text
        ], style={'flex': '1', 'marginRight': '10px'}),
        html.Div([
            html.Button('Clear All Selections', id='clear-button', n_clicks=0, style={
                'backgroundColor': '#dc3545',
                'color': 'white',  # White text
                'border': 'none',  # Red border
                'padding': '10px 20px',  # Padding
                'textAlign': 'center',  # Center text
                'textDecoration': 'none',  # Remove underline
                'display': 'inline-block',  # Inline block
                'fontSize': '14px',  # Font size
                'borderRadius': '5px',  # Rounded corners
                'cursor': 'pointer'  # Pointer cursor
            })  # Customized reset button text
        ], style={'flex': '1'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'width': '400px', 'margin': '0 auto'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div([
        html.Div([
            dcc.RadioItems(
                id='category-radioitems',
                options=[
                    {'label': 'Geospatial', 'value': 'Geospatial'},
                    {'label': 'Demographic', 'value': 'Demographic'},
                    {'label': 'Poverty', 'value': 'Poverty'},
                    {'label': 'Food Insecurity', 'value': 'Food Insecurity'},
                    {'label': 'Crime', 'value': 'Crime'},
                    {'label': 'Mental Health', 'value': 'Mental Health'},
                    {'label': 'Environment', 'value': 'Environment'},
                ],
                value='Demographic',  # Default value
                labelStyle={'display': 'block', 'margin-bottom': '10px'}
            )
        ], style={'width': '20%', 'float': 'left'}),
        html.Div([
            html.Div([
                dcc.Graph(id='bar-graph'),
                html.Div(
                    id='category-notes',
                    style={
                        'padding': '10px',
                        'border': '1px solid #ccc',
                        'border-radius': '5px',
                        'position': 'absolute',
                        'bottom': '15px',
                        'right': '10px',
                        'width': '250px',
                        'background-color': 'white',
                        'font-size': '12px',
                        'box-shadow': '0 0 10px rgba(0, 0, 0, 0.1)',
                        'color': 'darkgray'
                    }
                )
            ], style={'position': 'relative'})  # Ensure this container is relatively positioned
        ], style={'width': '75%', 'float': 'right'}),
    ], style={'display': 'flex'}),
    html.Br(),
    dcc.Graph(id='summary-sunburst'),  # Add the new summary sunburst
    dcc.Graph(id='trend-bar-graph'),  # Add the new trend bar graph
    dcc.Graph(id='scatter-graph'),  # Add the new scatter graph
    dcc.Graph(id='highest-suicide-rates-bar')  # Add the new bar graph for highest suicide rates
], className="container")

# Callback to clear the county dropdown and clickData when the clear button is clicked
@app.callback(
    Output('memory-output', 'data'),
    Output('county-dropdown', 'value'),
    Output('choropleth-map', 'selectedData'),
    Input('clear-button', 'n_clicks')
)
def clear_counties(n_clicks):
    if n_clicks > 0:
        return [], None, None
    return no_update, no_update, no_update

# Callback to update the map when the year slider is used
@app.callback(
    Output('choropleth-map', 'figure'),
    Input('year-slider', 'value')
)
def update_map(selected_year):
    data = load_data(selected_year)
    return create_choropleth_map(data)

# Callback to update the graphs when the dropdown menu, radio items, and map selection are used
@app.callback(
    Output('bar-graph', 'figure'),
    Output('trend-bar-graph', 'figure'),
    Output('summary-sunburst', 'figure'),
    Output('scatter-graph', 'figure'),
    Output('highest-suicide-rates-bar', 'figure'),
    Output('category-notes', 'children'),
    Input('confirm-button', 'n_clicks'),
    Input('category-radioitems', 'value'),
    Input('choropleth-map', 'selectedData'),
    State('county-dropdown', 'value'),
    State('year-slider', 'value'),
    State('clear-button', 'n_clicks')
)
def update_graphs(n_clicks_confirm, selected_category, selected_data, selected_counties, selected_year, n_clicks_clear):
    ctx = callback_context

    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'clear-button':
            selected_data = None
            selected_counties = []

    # Initialize selected_counties to an empty list if it's None
    if selected_counties is None:
        selected_counties = []

    # Load data for the selected year
    data = load_data(selected_year)

    # Process selected_data to add selected counties
    if selected_data and 'points' in selected_data:
        selected_counties = [point['location'] for point in selected_data['points']]

    selected_counties = selected_counties[:5]

    # Create a color map for consistent coloring
    color_map = {county: color for county, color in zip(selected_counties, px.colors.qualitative.Vivid)}

    # Create the updated figures
    bar_graph_fig = update_bar_graph(selected_year, selected_counties, selected_category, selected_data)
    trend_bar_graph_fig = trend_bar_graph(selected_year, selected_counties, color_map)
    summary_sunburst_fig = summary_sunburst(data)
    scatter_graph_fig = scatter_graph(selected_year, selected_counties, color_map)
    highest_suicide_rates_bar_fig = update_highest_suicide_rates_bar(data, selected_counties, selected_data)
    category_notes = get_category_notes(selected_category)

    return bar_graph_fig, trend_bar_graph_fig, summary_sunburst_fig, scatter_graph_fig, highest_suicide_rates_bar_fig,category_notes

# Run the Dash app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=True, host='0.0.0.0', port=port)


