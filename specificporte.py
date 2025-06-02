# Import Libraries
import dash                                     # Dash Framework     
import pandas as pd                             # DataFrame Engineering                
import plotly.express as px                     # Plotly Express Visualizations
from dash import dcc, html, dash_table          # Basic Dash Components
import dash_bootstrap_components as dbc         # Bootstrap Components
from dash.dependencies import Input, Output     # Input and outputs for Callbacks

# Load the World Port Data
df = pd.read_csv('NGA-World-Ports-2019.csv')

# Initialize the Dash App
app = dash.Dash(title='Specific Port Analysis',               # Dashboard Title
                suppress_callback_exceptions=True,            # Suppress Callback Exceptions
                external_stylesheets=[dbc.themes.DARKLY])     # Set Dark Theme

# Create Dashboard Layout
app.layout = dbc.Container([
    # Set Dashboard Title
    dbc.Row([
        dbc.Col([
            html.H1("Specific Port Analysis", 
                    style={'padding': '20px', 'textAlign': 'left'})
        ]),
        dbc.Col([
            html.H1(id='port-name', style={'padding': '20px', 'textAlign': 'right' })
        ], width=4),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='Port-DropDown',
                options=[{'label': port, 'value': port} for port in df['Main Port Name'].unique()],
                placeholder="Select a Port Name",
                style={'color': 'black'}
            ),
            dbc.Col([
                html.Div(id='port-index-number', style={'padding': '2em', 'textAlign': 'left','fontsize': '1em'}),
                html.Div(id='port-anchor-depth', style={'padding': '2em', 'textAlign': 'left','fontsize': '1em'}),
                html.Div("Anchor:", style={'padding': '2em', 'textAlign': 'left','fontsize': '1em'}),
                html.Div("Anchor:", style={'padding': '2em', 'textAlign': 'left','fontsize': '1em'}),
                html.Div("Anchor:", style={'padding': '2em', 'textAlign': 'left','fontsize': '1em'}),
                html.Div("Anchor:", style={'padding': '2em', 'textAlign': 'left','fontsize': '1em'}),
                html.Div("Anchor:", style={'padding': '2em', 'textAlign': 'left','fontsize': '1em'}),
            ],width=6),
        ], width=4),
        dbc.Col([
            dcc.Graph(id='sat-map', style={'position': 'relative', 'height': '50rem'}),
        ], width=8),
        dbc.Col([
            dcc.Graph(id='globe-map', style={'height': '7rem'}),
        ],width=4),
    ]),
], fluid=True)

# Set CallBacks to Update the Dashboard
@app.callback(
    [Output('sat-map', 'figure'), 
     Output('globe-map', 'figure'), 
     Output('port-name', 'children'),
     Output('port-anchor-depth', 'children'),
     Output('port-index-number', 'children')],
    
     Input('Port-DropDown', 'value')
)
def update_graph(selected_port):
    if selected_port:
        port_df = df[df['Main Port Name'] == selected_port]
        port_anchor_depth = port_df['Anchorage Depth (m)'].values[0]
        map_zoom = 14
        map_center = {"lat": port_df['Latitude'].values[0], "lon": port_df['Longitude'].values[0]}
        port_name = selected_port

        # Port Attributes
        port_index_number = port_df['World Port Index Number'].values[0]
        port_anchor_depth = f"Anchor Depth: {port_anchor_depth} m"
        port_index_number = f"World Port Index Number: {port_index_number}"
        
    else:
        port_df = df[df['Main Port Name'] == 'Maurer']
        map_zoom = 1
        map_center = {"lat": 15, "lon": 0}
        port_name = "Select a Port"
        port_anchor_depth = "N/A"
        port_index_number = port_df['World Port Index Number'].values[0]
    # Create Satellite Map of the Port
    satellite_map = px.scatter_map(port_df,
                                      zoom=map_zoom,
                                      lat='Latitude',
                                      lon='Longitude',
                                      center=map_center,
                                      hover_name="Main Port Name",
                                      hover_data=["Country Code", "Region Name"],
                                      color_discrete_sequence=["red"])

    satellite_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                                map_style="white-bg",
                                map_layers=[{
                                    "below": 'traces',
                                    "sourcetype": "raster",
                                    "source": ["https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"]
                                }])

    # Create Globe Map of the Port
    globe_map = px.scatter_geo(port_df,
                               lat='Latitude',
                               lon='Longitude',
                               hover_name="Main Port Name",
                               hover_data=["Country Code", "Region Name"],
                               color_discrete_sequence=["red"],
                               projection="natural earth",
                               template="plotly_dark")
    globe_map.update_layout(
        margin={"r": 7, "t": 0, "l": 7, "b": 0},
        paper_bgcolor='rgba(0,0,0,0)',  # Set the background color to transparent
        plot_bgcolor='rgba(0,0,0,9)'    # Set the plot area background color to transparent
    )

    return satellite_map, globe_map, port_name, port_anchor_depth, port_index_number

# Run the app with Flask
if __name__ == '__main__':
    app.run_server(debug=True)