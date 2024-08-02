# Animated map

import dash
from dash import Dash, html, dcc, Input, Output, State, no_update
import plotly.express as px
import pandas as pd
import os
import json
import plotly.graph_objects as go
import warnings

app = dash.Dash(__name__)

# Mapbox access token
mapbox_access_token = "pk.eyJ1IjoibGFwaGF0cmFkMDIiLCJhIjoiY2x5ZjFkdTZqMDM4cjJxcHh4dW9oNHd6dSJ9.cbUxYCzotjR4xeDr0fD-Iw"

# Folder paths and file names
folder_path = '~/Desktop/website'
home_dir = os.path.expanduser('~')
input_folder = os.path.join(folder_path, 'data')
demographic_file = os.path.join(input_folder, 'demo/2020-demographic-info.csv')
shapefile_name = os.path.join(home_dir, 'Desktop', 'website', 'data', 'shapefile', 'county.json')
years = list(range(2000, 2021))

# Read demographic data
demographic_data = pd.read_csv(demographic_file, encoding='latin1')

# Load the GeoJSON file
with open(shapefile_name) as f:
    geojson_data = json.load(f)

# Add a new column in geojson_data for the last 5 digits of 'GEO_ID'
for feature in geojson_data['features']:
    geo_id = feature['properties']['GEO_ID']
    county_id = geo_id[-5:]
    feature['properties']['county_id'] = county_id

# Create FIPS code in demographic data
demographic_data['FIPS'] = demographic_data['STATE'].astype(str).str.zfill(2) + demographic_data['COUNTY'].astype(str).str.zfill(3)
demographic_data['county_id'] = demographic_data['FIPS'].astype(str)

# Function to load data for the selected year
def load_data(year):
    if year in [2016, 2017, 2018]:
        sdoh_file = os.path.join(input_folder, f'SDOH_demo_Dash/{year}.csv')
    else:
        sdoh_file = os.path.join(input_folder, f'SDOH_demo_Dash/2018.csv')  # Use 2018 as default for other years

    suicide_file = os.path.join(input_folder, f'suicide_n/{year}.csv')
    
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

    return final_merged_data

# Function to get top 5 counties by average suicide rate per 100,000
def get_top_5_counties(data):
    top_5_counties = data.nlargest(5, 'SuicideDeathRate')['CTYNAME'].tolist()
    return top_5_counties

def create_animated_choropleth_map(data):
    # Ensure the 'SuicideDeathRate' column exists and normalize it to per 100,000
    if 'SuicideDeathRate' not in data.columns:
        data['SuicideDeathRate'] = (data['Deaths'] / data['POPESTIMATE2020']) * 100000

    # Add a column indicating missing data
    data['missing_data'] = data['Deaths'].isna()

    # Create hover text
    data['hover_text'] = (
        'Year: ' + data['Year'].astype(str) + '<br>' +
        'County: ' + data['CTYNAME'].astype(str) + '<br>' +
        'State: ' + data['STNAME'].astype(str) + '<br>' +
        'Population: ' + data['POPESTIMATE2020'].map('{:,.0f}'.format) + '<br>' +
        'Suicide Rate: ' + data['SuicideDeathRate'].map(lambda x: f'{x:.0f}' if pd.notna(x) else 'N/A')
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

    # Create the animated choropleth map using Plotly's express module
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
        animation_frame='Year',
        mapbox_style="light",
        center={"lat": 37.0902, "lon": -95.7129},
        zoom=3.4
    )

    # Update traces to set the marker_line_width to 0 and customize hover template
    fig.update_traces(
        marker_line_width=0,
        hovertemplate='%{customdata[0]}<extra></extra>'
    )

    # Update layout for play/pause buttons
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
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True, "mode": "immediate"}],
                        
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "showactive": True,
                "type": "buttons"
         
            }
        ]
    )

    return fig

def update_bar_graph(selected_year, selected_counties, selected_category, clickData):
    data = load_data(selected_year)
    
    # Ensure selected_counties is a list
    if selected_counties is None:
        selected_counties = []

    # Get the top 5 counties by suicide rate
    top_5_counties = get_top_5_counties(data)
    
    # Ensure we only process the selected counties that exist in the data
    county_data = data[data['CTYNAME'].isin(selected_counties)]

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
    for county in selected_counties:
        county_info = county_data[county_data['CTYNAME'] == county]
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

    # Limit the number of counties displayed in the title
    max_counties_in_title = 5  # Updated to allow up to 5 counties
    displayed_counties = selected_counties[:max_counties_in_title]
    if len(selected_counties) > max_counties_in_title:
        title = f"Social Determinants for {', '.join(displayed_counties)} and {len(selected_counties) - max_counties_in_title} more counties"
    else:
        title = f"Social Determinants for {', '.join(displayed_counties)}"

    # Define custom color sequence
    custom_colors = ["#1c0c84", "#bc3484", "#8012a9", "#f38b44", "#f3cd38"]

    # Create the bar graph
    fig = px.bar(
        bar_df,
        x='Metric',
        y='Value',
        color='County',
        barmode='group',
        title=title,
        labels={'Value': 'Average Value'},
        template='plotly_white',
        color_discrete_sequence=custom_colors  # Use custom color sequence
    )

    # Customize hover labels to show information based on its county name
    hover_template = (
        "%{x}: %{y:.0f}"
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
            'text': title, 
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

    # Get the top 5 counties by suicide rate
    if not selected_counties:
        selected_counties = get_top_5_counties(trend_data)

    # Filter data for selected counties
    trend_data = trend_data[trend_data['CTYNAME'].isin(selected_counties)]

    # Group by year and county, then calculate the average suicide rate for each year
    trend_data = trend_data.groupby(['Year', 'CTYNAME']).agg({
        'SuicideDeathRate': 'mean'
    }).reset_index()

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

    # Get the top 5 counties by suicide rate
    if not selected_counties:
        selected_counties = get_top_5_counties(full_data)

    # Filter data for selected counties
    filtered_data = full_data[full_data['CTYNAME'].isin(selected_counties)]

    # Group by year and county, then calculate the average suicide rate for each year
    grouped_data = filtered_data.groupby(['year', 'CTYNAME']).agg({
        'SuicideDeathRate': 'mean'
    }).reset_index()

    # Create a figure
    fig = go.Figure()

    # Add a trace for each selected county
    for county in selected_counties:
        county_data = grouped_data[grouped_data['CTYNAME'] == county]
        fig.add_trace(go.Scatter(
            x=county_data['year'],
            y=county_data['SuicideDeathRate'],
            mode='lines+markers',
            name=county,
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
layout = html.Div([
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
    dcc.Graph(id='animated-choropleth-map', style={'position': 'relative', 'height': '80vh'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div([
        html.Div([
            dcc.Store(id='memory-output'),
            dcc.Dropdown(
                id='county-dropdown',
                options=[
                    {'label': f"{row['CTYNAME']}, {row['STNAME']}", 'value': row['CTYNAME']}
                    for _, row in demographic_data.iterrows()
                ],
                multi=True,
                placeholder="Please Select",
                maxHeight=200,  # Limit the dropdown height
                className="dropdown"
            ),
            html.Small('* Select up to 5 Counties', style={'display': 'block', 'marginTop': '5px', 'color': 'gray'})
        ], style={'flex': '1'}),
        html.Button('Reset All', id='clear-button', n_clicks=0, style={'marginLeft': '10px'})
    ], style={'display': 'flex', 'alignItems': 'center', 'width': '400px', 'margin': '15px auto'}),
    html.Br(),
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
    dcc.Graph(id='scatter-graph')  # Add the new scatter graph
], className="container")

# Callback to clear the county dropdown and clickData when the clear button is clicked
@app.callback(
    Output('memory-output', 'data'),
    Output('county-dropdown', 'value'),
    Output('animated-choropleth-map', 'clickData'),
    Input('clear-button', 'n_clicks')
)
def clear_counties(n_clicks):
    if n_clicks > 0:
        return [], None, None
    return no_update, no_update, no_update

# Update callback
@app.callback(
    Output('animated-choropleth-map', 'figure'),
    Output('bar-graph', 'figure'),
    Output('trend-bar-graph', 'figure'),
    Output('summary-sunburst', 'figure'),
    Output('scatter-graph', 'figure'),
    Output('category-notes', 'children'),
    Input('year-slider', 'value'),
    Input('county-dropdown', 'value'),
    Input('category-radioitems', 'value'),
    Input('animated-choropleth-map', 'clickData'),
    Input('clear-button', 'n_clicks'),
    State('animated-choropleth-map', 'relayoutData')
)
def update_figures(selected_year, selected_counties, selected_category, clickData, n_clicks, relayoutData):
    # Ensure clickData is None if the clear button is clicked
    if n_clicks and clickData:
        clickData = None

    # Load data for the selected year
    data = load_data(selected_year)

    # Get the top 5 counties by suicide rate if no counties are selected
    if not selected_counties:
        selected_counties = get_top_5_counties(data)

    # Process clickData to add selected county
    if clickData:
        location = clickData['points'][0]['location']
        clicked_county = data[data['county_id'] == location]['CTYNAME'].values[0]
        if clicked_county not in selected_counties:
            selected_counties.append(clicked_county)

    # Limit the number of selected counties to 5
    selected_counties = selected_counties[:5]

    # Create a color map for consistent coloring
    color_map = {county: color for county, color in zip(selected_counties, px.colors.qualitative.Vivid)}

    # Load data for all years to create the animated choropleth map
    all_years_data = []
    for year in years:
        year_data = load_data(year)
        year_data['Year'] = year
        all_years_data.append(year_data)
    combined_data = pd.concat(all_years_data, ignore_index=True)

    # Create the updated figures
    animated_choropleth_map_fig = create_animated_choropleth_map(combined_data)
    bar_graph_fig = update_bar_graph(selected_year, selected_counties, selected_category, clickData)
    trend_bar_graph_fig = trend_bar_graph(selected_year, selected_counties, color_map)
    summary_sunburst_fig = summary_sunburst(data)
    scatter_graph_fig = scatter_graph(selected_year, selected_counties, color_map)
    category_notes = get_category_notes(selected_category)

    return animated_choropleth_map_fig, bar_graph_fig, trend_bar_graph_fig, summary_sunburst_fig, scatter_graph_fig, category_notes


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)


    