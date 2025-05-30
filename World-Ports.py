import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import dash

# Load the CSV data
df = pd.read_csv('NGA-World-Ports-2019.csv')

# Initialize the Dash app
app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True,title='NGA World Ports')

# Create the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("World Ports Dashboard")
        ],style={'padding': '20px', 'color': 'white', 'font-size': '15em', 'justify': 'center'})
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='port-dropdown',
                options=[{'label': port, 'value': port} for port in df['Main Port Name'].unique()],
                placeholder="Select a Main Port Name"
            )
        ], width=4, style={'padding': '0 20px','color': 'black'}),
        dbc.Col([
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': country, 'value': country} for country in df['Country Code'].unique()],
                placeholder="Select a Country Code"
            )
        ], width=4, style={'padding': '0 20px','color': 'black'}),
        dbc.Col([
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': region, 'value': region} for region in df['Region Name'].unique()],
                placeholder="Select a Region Name"
            )
        ], width=4, style={'padding': '0 20px','color': 'black'}),
    ], style={'margin-bottom': '20px'}),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='map-graph', style={'position': 'relative'}),
            html.Div(id='port-details', style={
                'position': 'absolute',
                'top': '150px',
                'left': '20px',
                'padding': '20px',
                'background-color': '#343a40',  # Dark background color
                'border-radius': '5px',
                'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                'z-index': '1000',
                'color': 'white'  # White text color
            }),
            html.Div(id='port-link', style={
                'position': 'absolute',
                'top': '300px',
                'left': '20px',
                'padding': '10px',
                'background-color': '#343a40',  # Dark background color
                'border-radius': '5px',
                'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                'z-index': '1000',
                'color': 'white'  # White text color
            })
        ], width=6),
        dbc.Col([
            dcc.Graph(id='bar-graph'),
            dash_table.DataTable(
                id='port-table',
                columns=[
                    {"name": "Main Port Name", "id": "Main Port Name"},
                    {"name": "Country Code", "id": "Country Code"},
                    {"name": "Tidal Range (m)", "id": "Tidal Range (m)"},
                    {"name": "Entrance Width (m)", "id": "Entrance Width (m)"},
                    {"name": "Channel Depth (m)", "id": "Channel Depth (m)"},
                    {"name": "Anchorage Depth (m)", "id": "Anchorage Depth (m)"}
                ],
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': 'rgb(30, 30, 30)',
                    'color': 'white'
                },
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                }
            )
        ], width=6)
    ])
], fluid=True)

# Callback to update the map, bar graph, and table based on dropdown selection
@app.callback(
    [Output('map-graph', 'figure'),
     Output('port-details', 'children'),
     Output('bar-graph', 'figure'),
     Output('port-table', 'data'),
     Output('port-link', 'children')],
    [Input('port-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('region-dropdown', 'value')]
)
def update_graphs(selected_port, selected_country, selected_region):
    if selected_port:
        filtered_df = df[df['Main Port Name'] == selected_port]
        zoom = 15
    elif selected_country:
        filtered_df = df[df['Country Code'] == selected_country]
        zoom = 4
    elif selected_region:
        filtered_df = df[df['Region Name'] == selected_region]
        zoom = 5
    else:
        filtered_df = df
        zoom = 1.5

    # Create the map figure
    fig = px.scatter_map(
        filtered_df,
        lat="Latitude",
        lon="Longitude",
        hover_name="Main Port Name",
        hover_data=["Country Code", "Region Name"],
        color="Anchorage Depth (m)",  # Color by Anchorage Depth (m)
        color_continuous_scale=px.colors.sequential.Viridis,  # Choose a color scale
        zoom=zoom,
        height=900,
        center={"lat": filtered_df['Latitude'].mean(), "lon": filtered_df['Longitude'].mean()},
        #mapbox_style="carto-darkmatter"
    )
    fig.update_layout(
        map_style="white-bg",
        map_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "sourceattribution": "United States Geological Survey",
            "source": [
                "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
            ]
        }
        ])
    
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        coloraxis_colorbar=dict(
            x=0.9,  # Position the color bar inside the map area
            y=0.5,
            len=0.5,
            thickness=20
        ),
        template='plotly_dark'  # Apply the dark template
    )

    # Create the port details box only if a port is selected
    if selected_port:
        port_details = filtered_df.iloc[0]
        details = [
            html.H4(f"Details for {selected_port}"),
            html.P(f"Tidal Range (m): {port_details['Tidal Range (m)']}"),
            html.P(f"Entrance Width (m): {port_details['Entrance Width (m)']}"),
            html.P(f"Channel Depth (m): {port_details['Channel Depth (m)']}"),
            html.P(f"Anchorage Depth (m): {port_details['Anchorage Depth (m)']}")
        ]
        port_link = html.A("More details", href=f"/port/{selected_port}", style={'color': 'white'})
    else:
        details = []
        port_link = ""

    # Sort the filtered DataFrame by Anchorage Depth (m)
    sorted_df = filtered_df.sort_values(by="Anchorage Depth (m)")

    # Create the bar graph figure
    bar_fig = px.bar(
        sorted_df,
        x="Anchorage Depth (m)",
        y="Main Port Name",
        title="Anchorage Depth (m) by Port",
        labels={"Main Port Name": "Port", "Anchorage Depth (m)": "Anchorage Depth (m)"},
        color="Anchorage Depth (m)",  # Color by Anchorage Depth (m)
        color_continuous_scale=px.colors.sequential.Viridis,  # Choose a color scale
        template='plotly_dark'  # Apply the dark template
    )

    # Create the table data
    table_data = sorted_df.to_dict('records')

    return fig, details, bar_fig, table_data, port_link

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
