import os
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import warnings
from shapely.errors import ShapelyDeprecationWarning
import geopandas as gpd
from plotly.subplots import make_subplots
import numpy as np


# Define file paths
folder_path = '~/Desktop/website'
input_folder = os.path.join(folder_path, 'data')
home_dir = os.path.expanduser('~')
csv_file_name = 'demo/2020-demographic-info.csv'
shapefile_name = 'shapefile_2/cb_2018_us_county_500k.shp'
GeoJsonfile_path = os.path.join(home_dir, 'Desktop', 'website', 'data', 'shapefile', 'county.json')


csv_file_path = os.path.join(input_folder, csv_file_name)
shapefile_path = os.path.join(input_folder, shapefile_name)


app = dash.Dash(__name__)

Mapbox_token = "pk.eyJ1IjoibGFwaGF0cmFkMDIiLCJhIjoiY2x5ZjFkdTZqMDM4cjJxcHh4dW9oNHd6dSJ9.cbUxYCzotjR4xeDr0fD-Iw"

# Load and preprocess data
df = pd.read_csv(csv_file_path, encoding='latin1')

# Convert POPESTIMATE2020 to numeric, coercing errors to NaN
df['POPESTIMATE2020'] = pd.to_numeric(df['POPESTIMATE2020'], errors='coerce')
df = df.dropna(subset=['POPESTIMATE2020'])
df = df[df['POPESTIMATE2020'] != 0]

# Map full state names to abbreviations
state_abbreviations = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
    'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
    'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}
df['StateCode'] = df['STNAME'].map(state_abbreviations)

# Load shapefile data
try:
    gdf = gpd.read_file(shapefile_path)
    gdf['COUNTYFP'] = gdf['COUNTYFP'].astype(str)
    gdf['STATEFP'] = gdf['STATEFP'].astype(str)
    gdf['FIPS'] = gdf['STATEFP'] + gdf['COUNTYFP']
except FileNotFoundError:
    raise FileNotFoundError(f"Shapefile does not exist: {shapefile_path}")
except Exception as e:
    raise RuntimeError(f"Error loading shapefile: {e}")

# Merge with demographic data
df['FIPS'] = df['STATE'].astype(str).str.zfill(2) + df['COUNTY'].astype(str).str.zfill(3)
merged_df = gdf.merge(df, left_on='FIPS', right_on='FIPS')

# Layout of the app
layout = html.Div([
    html.H1("US Population by State and County"),
    html.Div([
        dcc.RadioItems(
            id='map-type',
            options=[
                {'label': 'State Map', 'value': 'state'},
                {'label': 'County Map', 'value': 'county'}
            ],
            value='state',
            labelStyle={'display': 'inline-block', 'margin-right': '20px'}
        )
    ]),
    html.Div([
        html.Label("Select Year"),
        dcc.Slider(
            id='year-slider',
            min=2010,
            max=2020,
            step=1,
            value=2020,
            marks={str(year): str(year) for year in range(2010, 2021)}
        ),
    ]),
    html.Div([
        dcc.Graph(id='state-choropleth', style={'width': '100%', 'height': '400px'}),
        html.P("Note: Click on any state to see a detailed county-level map.", style={'fontSize': 12, 'color': 'gray'}),
        dcc.Graph(id='county-choropleth', style={'width': '100%', 'height': '600px'}),
    ]),
    html.Div([
        dcc.Store(id='memory-output'),
        dcc.Dropdown(
            id='state-dropdown',
            options=[{'label': 'United States', 'value': 'US'}] + [{'label': name, 'value': code} for name, code in state_abbreviations.items()],
            placeholder="Select a state",
            value='US',
        ),
    ], style={'width': '50%', 'margin': 'auto'}),
    html.Br(),
    html.Br(),
    html.Div([
        dcc.Graph(id='county-bar-graph', style={'width': '100%', 'height': '800px'})
    ])
])

# State lat/lon dictionary
state_lat_lon = {
    'AL': {'lat': 32.806671, 'lon': -86.791130},
    'AK': {'lat': 61.370716, 'lon': -152.404419},
    'AZ': {'lat': 33.729759, 'lon': -111.431221},
    'AR': {'lat': 34.969704, 'lon': -92.373123},
    'CA': {'lat': 36.116203, 'lon': -119.681564},
    'CO': {'lat': 39.059811, 'lon': -105.311104},
    'CT': {'lat': 41.597782, 'lon': -72.755371},
    'DE': {'lat': 39.318523, 'lon': -75.507141},
    'FL': {'lat': 27.766279, 'lon': -81.686783},
    'GA': {'lat': 33.040619, 'lon': -83.643074},
    'HI': {'lat': 21.094318, 'lon': -157.498337},
    'ID': {'lat': 44.240459, 'lon': -114.478828},
    'IL': {'lat': 40.349457, 'lon': -88.986137},
    'IN': {'lat': 39.849426, 'lon': -86.258278},
    'IA': {'lat': 42.011539, 'lon': -93.210526},
    'KS': {'lat': 38.526600, 'lon': -96.726486},
    'KY': {'lat': 37.668140, 'lon': -84.670067},
    'LA': {'lat': 31.169546, 'lon': -91.867805},
    'ME': {'lat': 44.693947, 'lon': -69.381927},
    'MD': {'lat': 39.063946, 'lon': -76.802101},
    'MA': {'lat': 42.230171, 'lon': -71.530106},
    'MI': {'lat': 43.326618, 'lon': -84.536095},
    'MN': {'lat': 45.694454, 'lon': -93.900192},
    'MS': {'lat': 32.741646, 'lon': -89.678696},
    'MO': {'lat': 38.456085, 'lon': -92.288368},
    'MT': {'lat': 46.921925, 'lon': -110.454353},
    'NE': {'lat': 41.125370, 'lon': -98.268082},
    'NV': {'lat': 38.313515, 'lon': -117.055374},
    'NH': {'lat': 43.452492, 'lon': -71.563896},
    'NJ': {'lat': 40.298904, 'lon': -74.521011},
    'NM': {'lat': 34.840515, 'lon': -106.248482},
    'NY': {'lat': 42.165726, 'lon': -74.948051},
    'NC': {'lat': 35.630066, 'lon': -79.806419},
    'ND': {'lat': 47.528912, 'lon': -99.784012},
    'OH': {'lat': 40.388783, 'lon': -82.764915},
    'OK': {'lat': 35.565342, 'lon': -96.928917},
    'OR': {'lat': 44.572021, 'lon': -122.070938},
    'PA': {'lat': 40.590752, 'lon': -77.209755},
    'RI': {'lat': 41.680893, 'lon': -71.511780},
    'SC': {'lat': 33.856892, 'lon': -80.945007},
    'SD': {'lat': 44.299782, 'lon': -99.438828},
    'TN': {'lat': 35.747845, 'lon': -86.692345},
    'TX': {'lat': 31.054487, 'lon': -97.563461},
    'UT': {'lat': 40.150032, 'lon': -111.862434},
    'VT': {'lat': 44.045876, 'lon': -72.710686},
    'VA': {'lat': 37.769337, 'lon': -78.169968},
    'WA': {'lat': 47.400902, 'lon': -121.490494},
    'WV': {'lat': 38.491226, 'lon': -80.954456},
    'WI': {'lat': 44.268543, 'lon': -89.616508},
    'WY': {'lat': 42.755966, 'lon': -107.302490},
}

# Callback to update maps based on the selected map type, year, and state
@app.callback(
    [Output('state-choropleth', 'style'),
     Output('county-choropleth', 'style'),
     Output('state-choropleth', 'figure'),
     Output('county-choropleth', 'figure'),
     Output('state-dropdown', 'style')],
    [Input('map-type', 'value'), 
     Input('year-slider', 'value'), 
     Input('state-choropleth', 'clickData'),
     Input('state-dropdown', 'value')]
)
def update_maps(map_type, selected_year, state_clickData, selected_state):
    population_column = f'POPESTIMATE{selected_year}'

    # State map
    df_agg = df.groupby('StateCode')[population_column].sum().reset_index()
    df_agg['hover_text'] = (
        'State: ' + df_agg['StateCode'].astype(str) + '<br>' +
        'Population: ' + df_agg[population_column].map('{:,.0f}'.format)
    )
    state_fig = go.Figure(data=go.Choroplethmapbox(
        geojson="https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json",
        locations=df_agg['StateCode'],
        z=df_agg[population_column].astype(float),
        colorscale='YlGnBu',
        colorbar_title="Population",
        text=df_agg['hover_text'],
        hoverinfo='text',
        marker_line_color='white',
        marker_line_width=0
    ))

    state_fig.update_layout(
        mapbox=dict(
            accesstoken=Mapbox_token,
            center=dict(lat=37.0902, lon=-95.7129),  # Center of the United States
            zoom=3.5,  # Adjust the zoom level as needed
            style="light"
        ),
        title=f"US Population Estimate by State in {selected_year}",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=600,
        width=1200
    )

    # County map
    if map_type == 'county':
        if selected_state == 'US':
            filtered_df = df
            state_center = {'lat': 37.0902, 'lon': -95.7129}  # Center of the United States
            zoom_level = 3.5  # Zoom out for the entire US
        else:
            filtered_df = df[df['StateCode'] == selected_state]
            state_center = state_lat_lon[selected_state]
            zoom_level = 5  # Zoom in for the state
    elif map_type == 'state' and state_clickData:
        selected_state = state_clickData['points'][0]['location']
        filtered_df = df[df['StateCode'] == selected_state]
        state_center = state_lat_lon[selected_state]
        zoom_level = 5  # Zoom in for the state
    else:
        filtered_df = df
        state_center = {'lat': 37.0902, 'lon': -95.7129}
        zoom_level = 3  # Default zoom level for the entire US

    if 'COUNTY' in filtered_df.columns and 'STATE' in filtered_df.columns:
        filtered_df['FIPS'] = filtered_df['STATE'].astype(str).str.zfill(2) + filtered_df['COUNTY'].astype(str).str.zfill(3)
    else:
        return {'display': 'block'}, {'display': 'none'}, state_fig, go.Figure(), {'display': 'none'}

    merged_df = gdf.merge(filtered_df, left_on='FIPS', right_on='FIPS')

    if merged_df.empty:
        return {'display': 'block'}, {'display': 'none'}, state_fig, go.Figure(), {'display': 'none'}

    merged_df['log_population'] = np.log1p(merged_df[population_column])

    merged_df['text'] = merged_df['CTYNAME'] + ', ' + merged_df['STNAME'] + '<br>' + \
                        'Population: ' + merged_df[population_column].map('{:,.0f}'.format)

    tickvals = [np.log1p(val) for val in [10, 100, 1000, 10000, 100000, 1000000]]
    ticktext = ['10', '100', '1K', '10K', '100K', '1M']

    county_fig = go.Figure(data=go.Choroplethmapbox(
        geojson=merged_df.__geo_interface__,
        locations=merged_df.index,
        z=merged_df['log_population'].astype(float),
        colorscale='YlGnBu',
        text=merged_df['text'],
        hoverinfo='text',
        marker_line_color='white',
        marker_line_width=0,
        zmin=merged_df['log_population'].min(),
        zmax=merged_df['log_population'].max(),
        showscale=True,
        colorbar=dict(
            title="Population",
            tickvals=tickvals,
            ticktext=ticktext
        )
    ))

    county_fig.update_layout(
        mapbox=dict(
            accesstoken=Mapbox_token,
            center=state_center,
            zoom=zoom_level,  # Adjust zoom level for county map
            style="light"
        ),
        title=f"US Population Estimate by County in {selected_year}",
        margin={"r": 0, "t": 50, "l": 0, "b": 10},
        height=600,
        width=1200
    )

    if map_type == 'state':
        return {'display': 'block'}, {'display': 'block' if state_clickData else 'none'}, state_fig, county_fig if state_clickData else go.Figure(), {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}, go.Figure(), county_fig, {'display': 'block'}

# Helper function to format numbers with K or M
def format_population(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:.2f}"

# Callback to update county bar graph based on selected state, county, and year
@app.callback(
    Output('county-bar-graph', 'figure'),
    [Input('state-choropleth', 'clickData'),
     Input('county-choropleth', 'clickData'),
     Input('year-slider', 'value'),
     Input('state-dropdown', 'value')]
)
def update_county_bar_graph(state_clickData, county_clickData, selected_year, selected_state):
    population_column = f'POPESTIMATE{selected_year}'
    
    if county_clickData:
        selected_fips = county_clickData['points'][0]['location']
        filtered_df = df[df['FIPS'] == selected_fips]
        selected_state = filtered_df.iloc[0]['StateCode'] if not filtered_df.empty else ''
    elif state_clickData:
        selected_state = state_clickData['points'][0]['location']
        filtered_df = df[(df['StateCode'] == selected_state) & (df['CTYNAME'].str.endswith('County'))]
    elif selected_state and selected_state != 'US':
        filtered_df = df[(df['StateCode'] == selected_state) & (df['CTYNAME'].str.endswith('County'))]
    else:
        # Default to whole US
        filtered_df = df[df['CTYNAME'].str.endswith('County')]

    if not filtered_df.empty:
        filtered_df_top = filtered_df.sort_values(by=population_column, ascending=False).head(10)
        filtered_df_bottom = filtered_df.sort_values(by=population_column, ascending=True).head(10)
    else:
        filtered_df_top = pd.DataFrame()
        filtered_df_bottom = pd.DataFrame()

    fig = make_subplots(
        rows=2, 
        cols=1, 
        subplot_titles=("Highest Population Counties", "Lowest Population Counties"),
        vertical_spacing=0.3  # Increase the vertical spacing between the subplots
    )

    # Apply formatting to the population values
    filtered_df_top['formatted_population'] = filtered_df_top[population_column].apply(format_population)
    filtered_df_bottom['formatted_population'] = filtered_df_bottom[population_column].apply(format_population)

    hover_template = 'County: %{x}<br>Population: %{customdata}'

    if not filtered_df_top.empty:
        fig.add_trace(
            go.Bar(
                x=filtered_df_top['CTYNAME'], 
                y=filtered_df_top[population_column], 
                name='Highest Population Counties', 
                marker=dict(color='royalblue'), 
                width=0.5,
                hovertemplate=hover_template,
                customdata=filtered_df_top['formatted_population']  # Pass the formatted data
            ),
            row=1, col=1
        )

    if not filtered_df_bottom.empty:
        fig.add_trace(
            go.Bar(
                x=filtered_df_bottom['CTYNAME'], 
                y=filtered_df_bottom[population_column], 
                name='Lowest Population Counties', 
                marker=dict(color='tomato'), 
                width=0.5,
                hovertemplate=hover_template,
                customdata=filtered_df_bottom['formatted_population']  # Pass the formatted data
            ),
            row=2, col=1
        )

    fig.update_layout(
        title=f'County Population Estimates in {selected_state if selected_state and selected_state != "US" else "the US"}',
        height=700,
        width=800,
        barmode='group',
        bargap=0.05,  # Gap between bars of adjacent location coordinates.
        bargroupgap=0.1,  # Gap between bars of the same location coordinates.
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        ),
        xaxis=dict(
            title='County',
            titlefont=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            zeroline=True,
            showline=True,
            gridcolor='lightgrey',
            gridwidth=1,
            zerolinecolor='black',
            zerolinewidth=2,
            linewidth=1,
            linecolor='black'
        ),
        yaxis=dict(
            title='Population',
            titlefont=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            zeroline=True,
            showline=True,
            gridcolor='lightgrey',
            gridwidth=1,
            zerolinecolor='black',
            zerolinewidth=2,
            linewidth=1,
            linecolor='black'
        ),
        xaxis2=dict(  # Ensure x-axis settings apply to the second subplot
            title='County',
            titlefont=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            zeroline=True,
            showline=True,
            gridcolor='lightgrey',
            gridwidth=1,
            zerolinecolor='black',
            zerolinewidth=2,
            linewidth=1,
            linecolor='black'
        ),
        yaxis2=dict(  # Ensure y-axis settings apply to the second subplot
            title='Population',
            titlefont=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            zeroline=True,
            showline=True,
            gridcolor='lightgrey',
            gridwidth=1,
            zerolinecolor='black',
            zerolinewidth=2,
            linewidth=1,
            linecolor='black',
            title_standoff=25
        ),
        plot_bgcolor='white',
        legend=dict(
            x=0.8,
            y=1.2,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='black',
            borderwidth=1,
            xanchor='left',
            yanchor='middle'
        ),
        margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins to ensure no overlap
    )

    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run(debug=True)
