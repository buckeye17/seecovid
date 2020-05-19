#############################################################
# Begin defining Dash app layout
# code sections
# 1 Environment setup
# 2 Setup Dataframes
# 3 Define Useful Functions
# 4 Heatmap UI controls
# 5 Curves plot UI controls
# 6 Navbar definition
# 7 Blank figure to display during initial app loading
# 8 Overall app layout
# 9 Dynamic UI callbacks
# 10 Callback for Updating Heat Map Figure
# 11 Callback for Adding Rows to curve_plot_df (dataframe define curves to plot)
# 12 Callback for Updating Curves Plot Figure
# 13 Callback for Updating the first Epidemiology Sandbox Figure
# 14 Callbacks to Update UI of the Second Epidemiology Sandbox
# 15 Callback for Updating the Second Epidemiology Sandbox Figure

import os
import json
import pickle
import base64
from urllib.request import urlopen
#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib, matplotlib.cm as cm
import datetime as dt
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from plotly import subplots
from plotly import graph_objects as go

#################################################################
# 1 Environment Setup
# setup global variables
proj_path = ""
if os.name == "nt":
    # running on my local Windows machine
    ENV = "local"
else:
    # running on heroku server
    ENV = "heroku"

if ENV == "local":
    import os
    os.chdir("C:/Users/adiad/Anaconda3/envs/CovidApp36/covidapp/")

# set graphic elements & color palette
invis = "rgba(0,0,0,0)"

update_jh_data = True # controls whether Johns Hopkins data will be updated
data_path = "data_clean/"
secrets_path = "secret_credentials/"

# setting up images for rendering
image_path = "images/"
cross_icon_image = base64.b64encode(open(image_path + "icon.png", "rb").read())
herd_immunity_image = base64.b64encode(open(image_path + "Herd_Immunity_Fig.png", "rb").read())

# get mapbox token
token = open(secrets_path + ".mapbox_token").read()

# read US county geojson file
# from: https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json
with open(data_path + "us_county_geo.json") as f:
    us_counties_json = json.load(f)

# read US county geojson file
# from: https://eric.clst.org/tech/usgeojson/ (States with 5m resolution)#
with open(data_path + "us_states_geo.json") as f:
    us_states_json = json.load(f)

# read China province geojson file
# from: https://github.com/secsilm/plotly-choropleth-mapbox-demo/blob/master/china_province.geojson
with open(data_path + "china_province_geo2.json") as f:
    china_json = json.load(f)

# read Australia state geojson file
# from: https://github.com/rowanhogan/australian-states/blob/master/states.geojson
with open(data_path + "australia_state_geo2.json") as f:
    australia_json = json.load(f)

# read Canadian geojson file
# from: https://download2.exploratory.io/maps/canada_provinces.zip
with open(data_path + "canada_provinces_geo.json") as f:
    canada_json = json.load(f)

# read world geojson file
# from: https://github.com/datasets/geo-countries/blob/master/data/countries.geojson
with open(data_path + "all_countries_geo.json") as f:
    world_json = json.load(f)

# read initial heatmap figure file
with open(data_path + "init_heatmap.pkl", "rb") as f:
    init_heatmap = pickle.load(f)

# The following code has been omitted because the authentication process with Google
# cannot be completed on Heroku, it requires the user to use a web browser.
# Update Johns Hopkins data from my Google Drive
#if (ENV == "heroku") & (update_jh_data):
#    gauth = GoogleAuth()
#    gauth.LoadClientConfigFile(secrets_path + "client_secrets.json")
#
#    drive = GoogleDrive(gauth)
#
#    # define file to get from Google Drive
#    gd_file = drive.CreateFile({'id': '1KuubeQzOHAzh_TuNyK2w1XO_L8zXHTRF'})
#
#    # download the file and save it as defined
#    gd_file.GetContentFile(data_path + "Johns_Hopkins_Clean.pkl")

#################################################################
# 2 Setup Dataframes

# read dataframes from pickle files
df = pd.read_pickle(data_path + "Johns_Hopkins_Clean.pkl")

# add Active variables
def add_active_col(var_suffix, df):
    confirmed = df["Confirmed" + var_suffix].values
    recovered = np.clip(df["Recovered" + var_suffix].values, 0, None)
    deaths = np.clip(df["Deaths" + var_suffix].values, 0, None)
    df["Active" + var_suffix] = confirmed - recovered - deaths
    
    # correct occurrences where Recovered + Deaths > Confirmed
    # (where negative value rolls back to an enormous positive value)
    mask = ((recovered + deaths) > confirmed)
    df.loc[mask, "Active" + var_suffix] = 0
    return df

df = add_active_col("", df)
df = add_active_col("PerDate", df)
df = add_active_col("PerCapita", df)
df = add_active_col("PerDatePerCapita", df)

# define a dataframe that defines which geographic areas to plot infection curves
curve_plot_data = [[0, "United States of America", "New York", "nan"],
                   [1, "United States of America", "Massachusetts", "nan"],
                   [2, "United States of America", "Indiana", "nan"]]
curve_plot_cols = ["Row ID", "Country/Region", "Province/State", "County"]
curve_plot_df = pd.DataFrame(curve_plot_data, columns=curve_plot_cols)

# define a dataframe that defines the dynamic parameter values for the simulation
# in sandbox 2
sandbox2_df = pd.DataFrame([[0, 14, 3.0, True, True], \
                            [50, 14, 1.5, False, True]], \
                           columns=["t", "d", "r", "In Base", "In Alt"])

#################################################################
# 3 Define Useful Functions

# converts numpy's datetime64 dtype (used by pandas) to a string
def numpy_dt64_to_str(dt64):
    day_timestamp_dt = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    day_dt = dt.datetime.utcfromtimestamp(day_timestamp_dt)
    return day_dt.strftime("%b %d")

# Define function for predicting epidemic, used in sandboxes
# assuming 1 person is infected in the whole population of size N
# and the params d & r0 are providef in a listed arrange as:
# [[t0, d0, r0], [t1, d1, r1], ...]
# where t1, t2, etc. reprsent the beginning of new values for d & r
# dur defines the time point to terminate the simulation
def predict_sir(N, params_t, dur):

    # define a function which 
    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta*S*I/N, beta*S*I/N-gamma*I, gamma*I]
    
    # define a function which extras individual parameters given the time index
    def get_params(t_ind):

        # get basic parameters
        t = params_t[t_ind][0]
        d = params_t[t_ind][1]
        r = params_t[t_ind][2]

        # derive exponential function parameters
        gamma = 1 / d
        beta = r * gamma

        return t, gamma, beta

    # simulatd population sub-group sizes
    sir_init_pop = [N - 1, 1, 0] # [S, I, R]

    # set initial values for loop variables
    epidemic_stopped = False
    n_days = 0
    continue_calc = True
    removed = 0
    n_periods = len(params_t)
    period_ind = 0
    t_period_loop = params_t[0][1] # sim will pause to check termination criterion
    t_start, gamma, beta = get_params(period_ind)
    if n_periods == 1:
        t_period_end = t_period_loop
    else:
        period_ind_max = n_periods - 1
        t_end, ignore1, ignore2 = get_params(period_ind + 1)
        t_period_end = t_end

    while continue_calc:

        # predict SIR for loop period days
        predict_period_sir = solve_ivp(SIR, [0, t_period_end], sir_init_pop, \
                                        t_eval=np.arange(0, t_period_end, 1))

        # append loop results to previous results
        if removed == 0:
            t = predict_period_sir["t"]
            s = predict_period_sir["y"][0]
            i = predict_period_sir["y"][1]
            r = predict_period_sir["y"][2]
        else:
            # segmenting the sim into periods causes the first day's prediction
            # to be a repeat of the results from the last loop's last day, so
            # drop the first day
            t = np.concatenate((t, t_start - 1 + predict_period_sir["t"][1:]))
            s = np.concatenate((s, predict_period_sir["y"][0][1:]))
            i = np.concatenate((i, predict_period_sir["y"][1][1:]))
            r = np.concatenate((r, predict_period_sir["y"][2][1:]))
        
        # update loop variables with new period results
        n_days = len(t)
        removed = r[-1]
        sir_init_pop = [s[-1], i[-1], r[-1]]

        # look for epidemic burnout
        period_i = predict_period_sir["y"][1]
        if period_i[-1] < period_i[0]:

            # infected population is shrinking
            if (period_i[0] - period_i[-1]) < 1:

                # change in the size of the infected population 
                # over the loop period is < 1
                epidemic_stopped = True

        if n_periods > 1:
            if period_ind_max > period_ind + 1:
                # simulate the next period until its end
                period_ind += 1
                t_start, gamma, beta = get_params(period_ind)
                t_end, ignore1, ignore2 = get_params(period_ind + 1)
                t_period_end = t_end - t_start + 1
            elif period_ind_max > period_ind:
                # simulate the last period until termination criteria are met
                period_ind += 1
                t_start, gamma, beta = get_params(period_ind)
                t_period_end = params_t[period_ind][1]
            else:
                # continue simulating the last period until termination criteria are met
                t_start = t[-1] + 1
        else:
            # continue simulating the only period until termination criteria are met
            t_start = t[-1] + 1
        
        # determine whether to continue looping
        if np.isinf(dur):
            continue_calc = not epidemic_stopped
        else:
                continue_calc = (dur > n_days)

    # trim results to desired duration
    if len(t) > dur:
        t = t[:dur + 1]
        s = s[:dur + 1]
        i = i[:dur + 1]
        r = r[:dur + 1]

    return np.column_stack((t, s, i, r))

# Basic setup of Dash app
external_stylesheets = [dbc.themes.COSMO]
btn_color = "primary"
navbar_color = "primary"
navbar_is_dark = True

# dash instantiation
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, assets_folder='assets')
server = app.server

# adding Google Analytics
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=UA-44205806-2"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());

            gtag('config', 'UA-44205806-2');
        </script>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

#################################################################
# 4 Heatmap UI controls
heat_ctrls_row1 = \
    dbc.Row([
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("Map Scope", addon_type="prepend"),
                dbc.Select(
                    id="map-scope",
                    options=[
                        {"label": "Australian States", "value": "Australia"},
                        {"label": "Canadian Provinces", "value": "Canada"},
                        {"label": "Chinese Provinces", "value": "China"},
                        {"label": "US Counties", "value": "UScounties"},
                        {"label": "US States", "value": "USstates"},
                        {"label": "Whole World", "value": "World"}
                    ],
                    value="UScounties"
                )
            ])
        ]), md=4, xs=12, style={"padding": "5px 10px"}),
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("Heat Variable", addon_type="prepend"),
                dbc.Select(
                    id="map-var",
                    options=[
                        {"label": "Confirmed", "value": "Confirmed"},
                        {"label": "Active", "value": "Active"},
                        {"label": "Recovered", "value": "Recovered"},
                        {"label": "Deaths", "value": "Deaths"}
                    ],
                    value="Confirmed"
                )
            ])
        ]), md=4, xs=12, style={"padding": "5px 10px"}),
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("New or Total Cases", addon_type="prepend"),
                dbc.Select(
                    id="map-calc",
                    options=[
                        {"label": "Total Cases to Date", "value": "Total"},
                        {"label": "New Cases on Date", "value": "PerDate"}
                    ],
                    value="Total"
                )
            ])
        ]), md=4, xs=12, style={"padding": "5px 10px"}),
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("Heat & Bar Scales", addon_type="prepend"),
                dbc.Select(
                    id="map-scale",
                    options=[
                        {"label": "Linear", "value": "Linear"},
                        {"label": "Logarithmic", "value": "Logarithmic"}
                    ],
                    value="Logarithmic"
                )
            ])
        ]), md=4, xs=12, style={"padding": "5px 10px"}),
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("Normalize Heat & Bar", addon_type="prepend"),
                dbc.Select(
                    id="map-norm-type",
                    options=[
                        {"label": "None", "value": "None"},
                        {"label": "Per Capita", "value": "PerCapita"}
                    ],
                    value="None"
                )
            ])
        ]), md=4, xs=12, style={"padding": "5px 10px"}),
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("Normalize Per", addon_type="prepend"),
                dbc.Select(
                    id="map-norm-val",
                    options=[
                        {"label": "1 Capita", "value": 1},
                        {"label": "10 Capita", "value": 10},
                        {"label": "100 Capita", "value": 100},
                        {"label": "1k Capita", "value": 1000},
                        {"label": "10k Capita", "value": 10000},
                        {"label": "100k Capita", "value": 100000},
                        {"label": "1M Capita", "value": 1000000}
                    ],
                    value=100000
                )
            ])
        ]), md=4, xs=12, style={"padding": "5px 10px"}),
    ], style={'padding-left': 20, 'padding-right': 20, 'margin-top': 5})

heat_cntrls_accordion = html.Div([
    dbc.Card([
        dbc.CardHeader(
            html.H1(
                dbc.Button(
                    "Plot Controls",
                    color=btn_color,
                    id="heat-edit-toggle",
                ), style={"padding-bottom": 6}
            ), style={"padding-bottom": 0, "padding-top": 0}
        ),
        dbc.Collapse([
            heat_ctrls_row1],
            id="collapse-heat-edit",
        ),
    ]),
], className="accordion")

#################################################################
# 5 Curves plot UI controls
curve_ctrls_row1 = \
    dbc.Row([
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("Country", addon_type="prepend"),
                dbc.Select(
                    id="curve-country",
                    options=[{"label": country, "value": country} for country in \
                             np.sort(df["Country/Region"].unique())],
                    value="United States of America"
                )
            ])
        ]), md=4, xs=12, style={"padding": "5px 10px"}),
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("State", addon_type="prepend"),
                dbc.Select(
                    id="curve-state",
                    options=[],
                    disabled=True,
                    value=""
                )
            ])
        ]), md=4, xs=12, style={"padding": "5px 10px"}),
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("County", addon_type="prepend"),
                dbc.Select(
                    id="curve-county",
                    options=[],
                    disabled=True,
                    value=""
                )
            ])
        ]), md=4, xs=12, style={"padding": "5px 10px"}),
    ], style={'padding-left': 20, 'padding-right': 20, 'margin-top': 5})

curve_ctrls_row2 = \
    dbc.Row([
        dbc.Col(html.Div(""), md=3, xs=1, style={'textAlign': 'right', 'margin-top': 0}),
        dbc.Col(html.Div([
            dbc.Button("Add", id="curve-add", n_clicks=0, color=btn_color)
        ]), md=1, xs=2, style={"textAlign": "center", "margin-top": 0, "padding-left": 0}),

        # Hidden div inside the app that tracks how many times the Add button has been clicked
        # This enables a determination for whether Add button triggered the edit_plotted_curves callback
        html.Div(0, id='curve-add-click-count', style={'display': 'none'}),
        
        dbc.Col(html.Div([
            dbc.Button("Clear All", id="curve-clear", n_clicks=0, color=btn_color)
        ]), md=2, xs=2, style={"textAlign": "center", "margin-top": 0, "padding-left": 0}),

        # Hidden div inside the app that tracks how many times the Clear All button has been clicked
        # This enables a determination for whether Clear All button triggered the edit_plotted_curves callback
        html.Div(0, id='curve-clear-click-count', style={'display': 'none'}),

        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("Drop Row by ID", addon_type="prepend"),
                dbc.Select(
                    id="curve-drop",
                    options=[{"label": val, "value": val} for  val in curve_plot_df["Row ID"].values],
                    value=""
                )
            ])
        ]), md=3, xs=6, style={"padding": "5px 10px"}),
        dbc.Col(html.Div(""), md=3, xs=1, style={'textAlign': 'right', 'margin-top': 0})
    ], style={'padding-left': 20, 'padding-right': 20, 'margin-top': 5})

curve_ctrls_row3 = \
    dbc.Row([
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("New or Total Case", addon_type="prepend"),
                dbc.Select(
                    id="curve-calc",
                    options=[
                        {"label": "Total Cases to Date", "value": "Total"},
                        {"label": "New Cases on Date", "value": "PerDate"}
                    ],
                    value="PerDate"
                )
            ])
        ]), md=4, xs=12, style={"padding": "5px 10px"}),
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("Normalize", addon_type="prepend"),
                dbc.Select(
                    id="curve-norm-type",
                    options=[
                        {"label": "None", "value": "None"},
                        {"label": "Per Capita", "value": "PerCapita"}
                    ],
                    value="None"
                )
            ])
        ]), md=4, xs=12, style={"padding": "5px 10px"}),
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("Normalize Per", addon_type="prepend"),
                dbc.Select(
                    id="curve-norm-val",
                    options=[
                        {"label": "1 Capita", "value": 1},
                        {"label": "10 Capita", "value": 10},
                        {"label": "100 Capita", "value": 100},
                        {"label": "1k Capita", "value": 1000},
                        {"label": "10k Capita", "value": 10000},
                        {"label": "100k Capita", "value": 100000},
                        {"label": "1M Capita", "value": 1000000}
                    ],
                    value=100000
                )
            ])
        ]), md=4, xs=12, style={"padding": "5px 10px"}),
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("Zero Date", addon_type="prepend"),
                dbc.Select(
                    id="curve-zero",
                    options=[
                        {"label": "None (just use dates)", "value": "None"},
                        {"label": "When 1 case is reported", "value": "Total"},
                        {"label": "When 1 case per 10k capita", "value": "PerCapita"},
                    ],
                    value="Total"
                )
            ])
        ]), md=6, xs=12, style={"padding": "5px 10px"}),
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("Case Scale", addon_type="prepend"),
                dbc.Select(
                    id="curve-scale",
                    options=[
                        {"label": "Linear", "value": "linear"},
                        {"label": "Logarithmic", "value": "log"},
                    ],
                    value="log"
                )
            ])
        ]), md=6, xs=12, style={"padding": "5px 10px"}),
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon("Case Types", addon_type="prepend"),
                dbc.Checklist(
                          id="curve-type",
                          options=[
                                {"label": "Confirmed", "value": "Confirmed"},
                                {"label": "Active", "value": "Active"},
                                {"label": "Recovered", "value": "Recovered"},
                                {"label": "Deaths", "value": "Deaths"}
                            ],
                          value=["Confirmed", "Deaths"],
                          inline=True,
                          custom=True,
                          style={"display": "inline-block", "margin-left": 10, "margin-top": 8}
                )
            ])
        ]), xl=6, lg=7, md=12, xs=12, style={"padding": "5px 10px"}),
        dbc.Col(html.Div([
            dbc.InputGroup([
                dbc.InputGroupAddon(
                    "Tune Curve Fit",
                    addon_type="prepend",
                    id="curve-avg-label",
                    style={"width": 140, "padding-right": 0}
                ),
                html.Span([
                    html.Span([
                        dcc.Slider(
                            id="curve-avg-period",
                            marks={1: "1", 7: "7", 14: "14", 21: "21", 28: "28"},
                            min=1,
                            max=28,
                            step=1,
                            value=14,
                            included=False
                        ),
                        dbc.Tooltip(
                            "Curve fitting is calculated with a moving average.  This parameter " + \
                            "determines the max number of days to use in averaging each point.",
                            target="curve-avg-period",
                        ),
                    ], style={"width": "100%", "display": "inline-block"})
                ], style={"width": "60%", "text-align": "left", "padding": "10px 0 0 0"})
            ]),
            dbc.Tooltip(
                "Curve fitting is calculated with a moving average.  This parameter " + \
                "determines the max number of days to use in averaging each point.",
                target="curve-avg-label",
            )
        ]), xl=6, lg=5, md=8, xs=12, style={"padding": "5px 10px"}),
    ], style={'padding-left': 20, 'padding-right': 20, 'margin-top': 5, 'margin-bottom': 10})

data_tbl = \
    dbc.Row([
        dbc.Col(
            html.Div(""), md=3, xs=1, style={'textAlign': 'right', 'margin-top': 0}
        ),
        dbc.Col(html.Div([
            dash_table.DataTable(
                id="data-table",
                data=curve_plot_df.to_dict('records'),
                columns=[{"id": c, "name": c}
                         for c in curve_plot_df.columns],
                editable=False,
                style_cell={
                    'textAlign': 'center'
                },
                style_cell_conditional=[
                    {
                        'if': {'column_id': c},
                        'textAlign': 'center'
                    } for c in ['Date', 'Region']
                ],
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            )]), md=6, xs=10, style={'textAlign': 'right', 'margin-top': 0}
        ),
        dbc.Col(
            html.Div(""), md=3, xs=1, style={'textAlign': 'right', 'margin-top': 0}
        ),
    ], style={'margin-bottom': 10, 'margin-top': 10})

curve_cntrls_accordion = html.Div([
    dbc.Card([
        dbc.CardHeader(
            html.H1(
                dbc.Button(
                    "Curve Picker",
                    color=btn_color,
                    id="curve-edit-toggle",
                ), style={"padding-bottom": 6}
            ), style={'padding-bottom': 0, 'padding-top': 0}
        ),
        dbc.Collapse([
            curve_ctrls_row1,
            curve_ctrls_row2,

            # visualize the curve_plot_df
            data_tbl,

            # Hidden div inside the app that allows the curve_plot_df tp be shared among callbacks
            html.Div([curve_plot_df.to_json(date_format='iso', orient='split')], 
                     id='curve-plot-df', style={'display': 'none'})],
            id="collapse-curve-edit",
        ),
    ]),
    dbc.Card([
        dbc.CardHeader(
            html.H1(
                dbc.Button(
                    "Plot Settings",
                    color=btn_color,
                    id="curve-setting-toggle",
                ), style={"padding-bottom": 6}
            ), style={'padding-bottom': 0, 'padding-top': 0}
        ),
        dbc.Collapse([
            curve_ctrls_row3],
            id="collapse-curve-setting",
        ),
    ]),
], className="accordion")

#################################################################
# 6 Navbar definition
dropdown_menu_items = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Discussion of this App", href="https://buckeye17.github.io/COVID-Dashboard/"),
        dbc.DropdownMenuItem("About the Author", href="https://buckeye17.github.io/about/"),
        dbc.DropdownMenuItem("LinkedIn Profile", href="https://www.linkedin.com/in/chris-raper/"),
        dbc.DropdownMenuItem("Github Repo", href="https://github.com/buckeye17/seecovid"),
        dbc.DropdownMenuItem("Powered by plotly|Dash", href="https://plotly.com/dash/")
    ],
    nav=True,
    in_navbar=True,
    label="Menu",
)

#################################################################
# 7 Blank figure to display during initial app loading
axopts = dict(showticklabels=False)
blank_fig = go.Figure()
blank_fig.update_layout(
    paper_bgcolor=invis,
    plot_bgcolor=invis,
    xaxis=axopts,
    yaxis=axopts,
    annotations=[dict(x=2.5,
        y=4,
        xref="x1", 
        yref="y1",
        text="Please wait while the heatmap is initialized",
        showarrow=False,
        font=dict(size=16)
    )]
)

# define sandbox2 dynamic table
sandbox2_tbl = \
    html.Div([
        dash_table.DataTable(
            id="sandbox2-data-table",
            data=sandbox2_df.to_dict('records'),
            columns=[{"id": c, "name": c} for c in sandbox2_df.columns],
            editable=False,
            style_cell={
                'fontSize': '14px',
                'textAlign': 'center',
                'width': '100px',
                'minWidth': '100px',
                'maxWidth': '100px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            }
        )
    ], style={"margin": 10, "width": "40%", "padding-left": 15})

#################################################################
# 8 Overall app layout
app.layout = html.Div([

    # Banner/header block
    dbc.Navbar(
        dbc.Container([

            # left side of navbar: logo & app name
            html.A(
                # Use row and col to control vertical alignment of logo / brand-
                dbc.Row(
                    [
                        dbc.Col(html.Img(
                            src='data:image/png;base64,{}'.format(cross_icon_image.decode()),
                            height="40px"
                        )),
                        dbc.Col(dbc.NavbarBrand([
                            "COVID-19 Dashboard", 
                            html.Br(), 
                            html.Div("Developed by Chris Raper", style={"fontSize": "small"})
                        ], className="ml-2")),
                    ],
                    align="center", no_gutters=True, className="ml-2",
                ),
                href="https://seecovid.herokuapp.com/",
            ),

            # right side of navbar: nav links & menu
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("My Portfolio", href="https://buckeye17.github.io/")),
                    dropdown_menu_items
                ], className="ml-auto", navbar=True, style={"margin-right": 100}),
                id="navbar-collapse", navbar=True,
            ),
        ]),
        color=navbar_color,
        dark=navbar_is_dark
    ),

    # define tabs which provide different perspectives on data
    dbc.Tabs([

        # heatmap tab
        dbc.Tab([
            heat_cntrls_accordion,
            dbc.Row(dbc.Spinner(type="grow", color="primary", fullscreen=True), id="initial-spinner"),
            dbc.Row(dbc.Col(html.Div([dcc.Loading(dcc.Graph(id="heatmap", figure=blank_fig),
                                                  type="cube")]))),
            html.Div([dcc.Markdown('''
            *Playing the animation may cause iratic behavior if your browser isn't powerful 
            enough.  In this case, reload the page and jump to desired dates with the slider.
            ''', style={"textAlign": "center", "fontSize": "small"}
            )]),
        ], label="Heatmaps"),

        # curves tab
        dbc.Tab([
            curve_cntrls_accordion,
            dbc.Row(dbc.Col(html.Div([dcc.Loading(dcc.Graph(id="curves",
                                                            responsive=True,
                                                            style={"height": 400}), type="cube")]))),
        ], label="The Curves"),

        # epidemiology tab
        dbc.Tab([

            # this tab will consist of a single row, which contains a single column
            # that is centered horizontally in the page
            dbc.Row([
                dbc.Col([

                    # section header for Sandbox #1
                    html.Div([dcc.Markdown('''
                        #### Sandbox #1: Varying Basic Parameters of Infectious Disease
                        ''', style={"margin": 20, "textAlign": "center"}
                    )]),

                    # intro for Sandbox #1
                    html.Div([dcc.Markdown('''
                        The following sandbox allows you to simulate two scenarios of a generic epidemic, 
                        assuming a population of 10,000 people and that 1 of them is infected
                        on day zero.  The sandbox allows you to adjust the underlying parameters of 
                        the epidemic.  These parameters are:
                        * **d**: the average number of days someone is infectious 
                        * **r**: AKA the basic reproduction number, the number of people an infectious 
                        person would infect if everyone they contact is infectious.

                        With these parameters, the sandbox will predict how the fixed population 
                        will move through the 3 stages of infection: susceptible, infected, 
                        removed.  For further discussion of the underlying modeling method, it has
                        been provided further below.
                        ''', style={"margin": 20, "textAlign": "justify"}
                    )]),

                    # Sandbox #1
                    html.Div([

                        # Sandbox #1 title
                        html.Div([dcc.Markdown('''
                            ##### Epidemic Simulation Sandbox #1
                            ''', style={"margin": 20, "textAlign": "center"}
                        )]),

                        # UI for Scenario #1 of Sanbox #1
                        dbc.Row([
                            dbc.Col(["Scenario #1"], md=2, sm=12, \
                                style={"text-align": "center", "margin": "10px 0"}),
                            dbc.Col([
                                html.B("d0"), ": ", html.Span("28", id="sandbox1-scenario1-d-text"),
                            ], md=1, sm=2, style={"text-align": "right", "margin": "10px 0", \
                                                  "padding-right": 0}),
                            dbc.Col([
                                dcc.Slider(
                                    id="sandbox1-scenario1-d",
                                    marks={1: "1", 7: "7", 14: "14", 21: "21", 28: "28"},
                                    min=1,
                                    max=28,
                                    step=1,
                                    value=14,
                                    included=False
                                )
                            ], md=4, sm=10, style={"margin": "10px 0", "padding-left": 0}),
                            dbc.Col([
                                html.B("r0"), ": ", html.Span("8", id="sandbox1-scenario1-r-text"),
                            ], md=1, sm=2, style={"text-align": "right", "margin": "10px 0", \
                                                  "padding-right": 0}),
                            dbc.Col([
                                dcc.Slider(
                                    id="sandbox1-scenario1-r",
                                    marks={0: "0", 1: "1", 2: "2", 4: "4", 6: "6", 8: "8"},
                                    min=0,
                                    max=8,
                                    step=0.1,
                                    value=1.5,
                                    included=False
                                )
                            ], md=4, sm=10, style={"margin": "10px 0", "padding-left": 0}),
                        ]),

                        # UI for Scenario #2 of Sanbox #1
                        dbc.Row([
                            dbc.Col(["Scenario #2"], md=2, sm=12, \
                                style={"text-align": "center", "margin": "10px 0"}),
                            dbc.Col([
                                html.B("d0"), ": ", html.Span("28", id="sandbox1-scenario2-d-text"),
                            ], md=1, sm=2, style={"text-align": "right", "margin": "10px 0", \
                                                  "padding-right": 0}),
                            dbc.Col([
                                dcc.Slider(
                                    id="sandbox1-scenario2-d",
                                    marks={1: "1", 7: "7", 14: "14", 21: "21", 28: "28"},
                                    min=1,
                                    max=28,
                                    step=1,
                                    value=14,
                                    included=False
                                )
                            ], md=4, sm=10, style={"margin": "10px 0", "padding-left": 0}),
                            dbc.Col([
                                html.B("r0"), ": ", html.Span("8", id="sandbox1-scenario2-r-text"),
                            ], md=1, sm=2, style={"text-align": "right", "margin": "10px 0", \
                                                  "padding-right": 0}),
                            dbc.Col([
                                dcc.Slider(
                                    id="sandbox1-scenario2-r",
                                    marks={0: "0", 1: "1", 2: "2", 4: "4", 6: "6", 8: "8"},
                                    min=0,
                                    max=8,
                                    step=0.1,
                                    value=3,
                                    included=False
                                )
                            ], md=4, sm=10, style={"margin": "10px 0", "padding-left": 0}),
                        ]),

                        # Area Plot for Sandbox #1
                        dcc.Loading(dcc.Graph(id="sandbox1-area-fig",
                                              responsive=True,
                                              style={"height": 400}), type="dot"),
                        
                        # Lines Plot for Sandbox #1
                        dcc.Loading(dcc.Graph(id="sandbox1-line-fig",
                                              responsive=True,
                                              style={"height": 200}), type="dot"),
                    ], style={"border-style": "solid", "border-color": "#aaaaaa", "padding": 10}),

                    # section header for Sandbox #2
                    html.Div([dcc.Markdown('''
                        #### Sandbox #2: Time-Dependence of Basic Parameters of Infectious Disease
                        ''', style={"margin-top": 40, "margin-bottom": 20, "textAlign": "center"}
                    )]),

                    # intro for Sandbox #2
                    html.Div([dcc.Markdown('''
                            This second sandbox is similar to the first, but it allows you 
                            to vary the parameters of the epidemic in time, whereas the first 
                            sandbox simulated constant parameters values.  With COVID-19, 
                            social distancing was implemented to reduce the **r** parameter 
                            of the disease (AKA "slowing the spread") and would reduce the 
                            total number of infected if sustained.
                            
                            In this sandbox you can chose the initial parameter values, then 
                            add time points when the parameters will change values.  You can add 
                            as many time points as you want.  The Baseline scenario will 
                            consist of all the parmeter value changes except for the final 
                            change.  The Alternate scenario will consist of all the parameter value 
                            changes.  The "In Base" and "In Alt" table columns should clarify 
                            this point.
                        ''', style={"margin": 20, "textAlign": "justify"}
                    )]),

                    # Sandbox #2
                    html.Div([

                        # Title for Sandbox #2
                        html.Div([dcc.Markdown('''
                            ##### Epidemic Simulation Sandbox #2
                            ''', style={"padding": "20px 0", "textAlign": "center", \
                                        "border-bottom": "solid", "border-width": "thin"}
                        )]),

                        # UI for initial conditions of Sandbox #2
                        dbc.Row([
                            dbc.Col(["Initial (t0) Values"], md=2, sm=12, \
                                style={"text-align": "center", "margin": "10px 0"}),
                            dbc.Col([
                                html.B("d0"), ": ", html.Span("28", id="sandbox2-baseline-d-text"),
                            ], md=1, sm=2, style={"text-align": "right", "margin": "10px 0", \
                                                  "padding-right": 0}),
                            dbc.Col([
                                dcc.Slider(
                                    id="sandbox2-baseline-d",
                                    marks={1: "1", 7: "7", 14: "14", 21: "21", 28: "28"},
                                    min=1,
                                    max=28,
                                    step=1,
                                    value=14,
                                    included=False
                                )
                            ], md=4, sm=10, style={"margin": "10px 0", "padding-left": 0}),
                            dbc.Col([
                                html.B("r0"), ": ", html.Span("8", id="sandbox2-baseline-r-text"),
                            ], md=1, sm=2, style={"text-align": "right", "margin": "10px 0", \
                                                  "padding-right": 0}),
                            dbc.Col([
                                dcc.Slider(
                                    id="sandbox2-baseline-r",
                                    marks={0: "0", 1: "1", 2: "2", 4: "4", 6: "6", 8: "8"},
                                    min=0,
                                    max=8,
                                    step=0.1,
                                    value=3,
                                    included=False
                                )
                            ], md=4, sm=10, style={"margin": "10px 0", "padding-left": 0}),
                        ]),

                        # UI for adding or editing dynamic values of Sandbox #2
                        # these UI element have a light blue background to distinguish
                        # their function from the row above, which pertains to 
                        # initial values, not dnamic values
                        html.Div([

                            # UI for defining new dynamic value of d & r in Sandbox #2
                            dbc.Row([
                                dbc.Col(["New Values at time t"], md=2, sm=12, \
                                    style={"text-align": "center", "margin": "10px 0"}),
                                dbc.Col([
                                    html.B("d"), html.Span(": "), html.Span("28", id="sandbox2-new-d-text"),
                                ], md=1, sm=2, style={"text-align": "right", "margin": "10px 0", \
                                                      "padding-right": 0}),
                                dbc.Col([
                                    dcc.Slider(
                                        id="sandbox2-new-d",
                                        marks={1: "1", 7: "7", 14: "14", 21: "21", 28: "28"},
                                        min=1,
                                        max=28,
                                        step=1,
                                        value=14,
                                        included=False
                                    )
                                ], md=4, sm=10, style={"margin": "10px 0", "padding-left": 0}),
                                dbc.Col([
                                    html.B("r"), html.Span(": "), html.Span("8", id="sandbox2-new-r-text"),
                                ], md=1, sm=2, style={"text-align": "right", "margin": "10px 0", \
                                                      "padding-right": 0}),
                                dbc.Col([
                                    dcc.Slider(
                                        id="sandbox2-new-r",
                                        marks={0: "0", 1: "1", 2: "2", 4: "4", 6: "6", 8: "8"},
                                        min=0,
                                        max=8,
                                        step=0.1,
                                        value=1.5,
                                        included=False
                                    )
                                ], md=4, sm=10, style={"margin": "10px 0", "padding-left": 0}),
                            ]),

                            # UI for defining the time point when the new dynamic values
                            # of d & r will take effect, as well to add, clear & edit 
                            # these dynamic values for Sandbox #2
                            dbc.Row([
                                dbc.Col([dbc.InputGroup([
                                    dbc.InputGroupAddon("t=", addon_type="prepend"),
                                    dbc.Input(id="sandbox2-new-t", placeholder="", type="number", min=0),
                                    dbc.Tooltip(dcc.Markdown('''
                                        Enter the time (in days) when when **d** & **r** values 
                                        should change.  The value must be positive.
                                        '''
                                        ), target="sandbox2-new-t"
                                    ),
                                ])], md=3, sm=4, style={"margin": "10px 0"}),
                                dbc.Col([
                                    dbc.Button("Add", id="sandbox2-add", n_clicks=0, color=btn_color)
                                ], md=2, sm=4, style={"margin": "10px 0"}),

                                # Hidden span inside the app that tracks how many times the Add button has been clicked
                                # This enables a determination for whether Add button triggered the edit_plotted_curves callback
                                html.Span(0, id='sandbox2-add-click-count', style={'display': 'none'}),

                                dbc.Col([
                                    dbc.Button("Clear All", id="sandbox2-clear", n_clicks=0, color=btn_color)
                                ], md=2, sm=4, style={"margin": "10px 0"}),

                                # Hidden span inside the app that tracks how many times the Clear All button has been clicked
                                # This enables a determination for whether Clear All button triggered the edit_plotted_curves callback
                                html.Span(0, id='sandbox2-clear-click-count', style={'display': 'none'}),

                                dbc.Col([
                                    dbc.InputGroup([
                                        dbc.InputGroupAddon("Drop Row @ t=", addon_type="prepend"),
                                        dbc.Select(
                                            id="sandbox2-drop",
                                            options=[{"label": val, "value": val} for  val in sandbox2_df.t.values],
                                            value=""
                                        )
                                    ]),
                                ], md=5, sm=12, style={"margin": "10px 0"})
                            ]),
                        ], style={"background-color": "#e8f6fc", "padding": 10}),

                        # UI to specify the current dynamic values in table form for.
                        # both the baseline and alternate scenarios for Sandbox #2
                        dbc.Row([
                            dbc.Col([
                                html.Div("All Dynamic Values", \
                                    style={"padding-top": 10, "text-align": "center"}),

                                # visualize the sandbox2_df
                                sandbox2_tbl,

                                # Hidden span inside the app that allows the sandbox2_df tp be shared among callbacks
                                html.Span([
                                    sandbox2_df.to_json(date_format='iso', orient='split')
                                ], id='sandbox2-df', style={'display': 'none'})
                            ], width=9),
                        ], justify="center"),
                        dcc.Loading(dcc.Graph(id="sandbox2-area-fig",
                                              responsive=True,
                                              style={"height": 400}
                        ), type="dot"),
                        dcc.Loading(dcc.Graph(id="sandbox2-lines-fig", 
                                              responsive=True,
                                              style={"height": 200}
                        ), type="dot"),
                    ], style={"border-style": "solid", "border-color": "#aaaaaa"}),

                    # section header for discussing modeling methodology
                    html.Div([dcc.Markdown('''
                        #### Examining the Fundamentals of Epidemics
                        ''', style={"margin": 20, "textAlign": "center"}
                    )]),

                    # body of section discussing modeling methodology
                    html.Div([dcc.Markdown('''
                        ##### Introduction
                        The sandboxes above use a simple class of models for epidemics called 
                        compartmental models.  Specifically they use an 
                        [SIR compartmental model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology), 
                        which segments a population into 3 stages of infection.  These stages are 
                        named **Susceptible**, **Infected** and **Removed**.  The meaning of these 
                        segments should be self-explanatory, with the clarification that the Removed 
                        segment includes those who have survived the infection (then becoming immune) 
                        as well as those who have died from it.

                        SIR models have two parameters which govern how quickly infection spreads through
                        the population.  The first is **d**, which represents the average time 
                        someone is infectious.  The second is **r0** (pronounced r naught), representing 
                        the average number of people a single infected person would infect 
                        if everyone they contact is susceptible.  In some simulations, this value may 
                        change with time, in which case **r0** is the initial value of **r**.

                        All that remains before simulating an epidemic is to make some assumptions 
                        about the initial condition of the population.  The sandboxes above assumed the 
                        population has 10,000 people, with 1 infected person and the remainder are 
                        susceptible.

                        ##### Examining Simulation Results
                        An epidemic is technically defined as a disease which has **r0** > 1.  If a 
                        disease yields **r0** < 1, then each chain of infections will die out.  But it 
                        should be noted that **r0** is not solely dependent on the nature of the 
                        disease.  It also depends on the behavior of disease's host.  So the occurrence 
                        of an epidemic depends on the combination of a disease's efficiency to infect 
                        the susceptible as well as the host's social behavior.

                        The following figure depicts two scenarios.  Scenario 1 assumes **d** = 10 days and 
                        **r0** = 1.5 people while Scenario 2 assumes **d** = 10 and **r0** = 3.  Notice 
                        both of these epidemic scenarios end without the disease infecting the whole 
                        population.  If **r0** > 1, this occurs because the number of infected people 
                        will peak when the number of susceptible people makes up 50% of the total 
                        population.  After this point, the number of infected will decline until it 
                        reaches zero.  The combination of the declining susceptible sub-population along 
                        with the recovery or death of infected people ensures that epidemic will die out
                        before it infects everyone.  This phenomenon is called **herd immunity**.
                        ''', style={"margin": 20, "textAlign": "justify"}
                    )]),
                    dbc.Row(dbc.Col(
                        html.Div(
                            html.Img(src="data:image/png;base64,{}".format(herd_immunity_image.decode()),
                                height=300,
                                style={"display": "inline-block"}
                            ),
                            style={"text-align": "center"}
                        ),
                        width=12,
                    )),
                    html.Div([dcc.Markdown('''
                        Also note that in the two scenarios above, herd immunity was reached with 
                        different sizes of the population never being infected (i.e. those who are still 
                        susceptible).  Scenario #1 ends the epidemic with 4,175 never being infected, 
                        while Scenario #2 ends with 595.  This illustrates that herd immunity is not only 
                        dependent on the parameters of the epidemic, but is also very sensitive to those 
                        values.  The difference was solely due to **r0** being 1.5 or 3.

                        ##### Examining Weaknesses of SIR Models
                        One manner in which SIR models over-simplify reality is that they assume that 
                        there is no variation in the parameter models.  Even if the the parmeters 
                        are allowed to change with time, they still assume that at each time point the 
                        infected will be sick for X days on average and will infect Y people.  But in 
                        reality, some people will recover quicker than others, some will shed more 
                        virus and some will be more social.
                        
                        This leads to the concept of so called "super-spreaders".  These 
                        people account for a drastic number of infections.  One example is a South Korean 
                        woman referred to as patient 31.  Through contact tracing the government had 
                        identified 3,700 COVID-19 patients who could be traced back to her, representing 
                        60% of all known cases by March 1 in South Korea.  This was reported by the 
                        [Wall Street Journal](https://www.wsj.com/articles/why-a-south-korean-church-was-the-perfect-petri-dish-for-coronavirus-11583082110). 
                        Compartmental models do not account for any variation in parameters as exhibited 
                        with super-spreaders.

                        Another shortcoming of this modeling method is that the parameter **d** is a little 
                        misleading.  If **d** = 14, then the precise calculation to determine how many 
                        infected people have been removed (by recovery or death) is to divide the number 
                        of infected by 14.  This implies that when the number of infected is peaking, 
                        the rate of removal will be greatest at this time.  Conversely, when the number of 
                        infected is small, the rate of recovery will be much slower.  In reality, the 
                        number of infected should not affect the rate of recovery.  This is why **d** 
                        is referred to as an "average" number of infectious days, because this 
                        characteristic actually varies with time in the simulation, even when **d**
                        is constant.

                        If you'd like more information on this subject, I would recommend the following
                        YouTube video.
                        ''', style={"margin": 20, "textAlign": "justify"}
                    )]),
                    html.Div(
                        html.Iframe(width="560", height="315", src="https://www.youtube.com/embed/gxAaO2rsdIs"),
                    style={"padding": 20})
                ], sm=12, md=10, xl=8),
            ], justify="center")
        ], label="Epidemiology Sandbox"),

        # links tab
        dbc.Tab([

            # this tab will consist of a single row, which contains a single column
            # that is centered horizontally in the page
            dbc.Row([
                dbc.Col([
                    html.Div([dcc.Markdown('''
                        ##### Useful Dashboards & Visualizations
                        * [Johns Hopkins COVID-19 Dashboard](https://www.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6)
                        * [CDC's COVID-NET](https://gis.cdc.gov/grasp/COVIDNet/COVID19_5.html): Provides US Demographics for COVID-19
                        * [University of Washington IHME COVID-19 Predictions for US](https://covid19.healthdata.org/united-states-of-america)
                        * [University of Minnesota Center for Infectious Disease Research and Policy](https://www.cidrap.umn.edu/): Provides latest research news on many infectious diseases, including COVID-19
                        * [Covidly.com](https://covidly.com/): Another COVID-19 dashboard
                        * Many US state health agencies have produced great COVID-19 dashboards for their state.  Just search for them.

                        ##### References
                        SIR Model Help:
                        * [Wikipedia](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)
                        * [Oregon State University presentation](http://sites.science.oregonstate.edu/~gibsonn/Teaching/MTH323-010S18/Supplements/Infectious.pdf)
                        * [A blog post](https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html)

                        All of the data sources used for this dashboard:
                        * [Johns Hopkins COVID data CSV files](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports)
                        * [Australian State Populations](https://en.wikipedia.org/wiki/States_and_territories_of_Australia)
                        * [Australian State Geo JSON File](https://github.com/rowanhogan/australian-states/blob/master/states.geojson)
                        * [Canadian Province Populations](https://en.wikipedia.org/wiki/Population_of_Canada_by_province_and_territory)
                        * [Canadian Province Geo JSON File](https://download2.exploratory.io/maps/canada_provinces.zip)
                        * [Chinese Province Populations](https://en.wikipedia.org/wiki/Provinces_of_China#List_of_province-level_divisions)
                        * [Chinese Province Geo JSON File](https://github.com/secsilm/plotly-choropleth-mapbox-demo/blob/master/china_province.geojson)
                        * [US County Populations (2019 Census Estimate)](https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv)
                        * [US County Geo JSON File](https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json)
                        * [US State Geo JSON File](https://eric.clst.org/tech/usgeojson/)
                        * [All other Country Populations](https://en.wikipedia.org/wiki/List_of_countries_by_population_%28United_Nations%29)
                        * [All Countries Geo JSON File](https://github.com/datasets/geo-countries/blob/master/data/countries.geojson)
                    ''', style={"margin": 20}
                    )])
                ], sm=12, md=10, xl=8, style={"border": "solid", "border-width": "thin", "margin-top": 40}),
            ], justify="center")
        ], label="Links & References")
    ])
])

#################################################################
# 9 Dynamic UI callbacks
# add callback for toggling the right nav menu collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# callback for curve plot accordion containing plot controls
@app.callback(
    [Output("collapse-heat-edit", "is_open"),
     Output("collapse-curve-edit", "is_open"),
     Output("collapse-curve-setting", "is_open")],
    [Input("heat-edit-toggle", "n_clicks"),
     Input("curve-edit-toggle", "n_clicks"),
     Input("curve-setting-toggle", "n_clicks")],
    [State("collapse-heat-edit", "is_open"),
     State("collapse-curve-edit", "is_open"),
     State("collapse-curve-setting", "is_open")])
def toggle_accordion(n_heat, n_curve_edit, n_curve_set, is_open_heat, \
                     is_open_curve_edit, is_open_curve_set):
    ctx = dash.callback_context

    if ctx.triggered:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    else:
        return False, False, False
    
    if button_id == "heat-edit-toggle" and n_heat:
        return not is_open_heat, is_open_curve_edit, is_open_curve_set
    elif button_id == "curve-edit-toggle" and n_curve_edit:
        return is_open_heat, not is_open_curve_edit, is_open_curve_set
    elif button_id == "curve-setting-toggle" and n_curve_set:
        return is_open_heat, is_open_curve_edit, not is_open_curve_set

# define curves tab control callbacks
# add callback for defining the contents of the State dropdown
@app.callback(
    [Output("curve-state", "options"),
     Output("curve-state", "value"),
     Output("curve-state", "disabled")],
    [Input("curve-country", "value")]
)
def country_selected(country):
    default_val = "All of " + country
    options = [{"label": default_val, "value": "nan"}]
    states_ls = np.sort(df.loc[df["Country/Region"] == country, "Province/State"].unique().tolist())
    states_ls = states_ls[states_ls != "nan"]
    state_options = [{"label": state, "value": state} for state in states_ls]
    if len(states_ls) > 0:
        options.extend(state_options)
    return options, default_val, False

# add callback for defining the contents of the County dropdown
@app.callback(
    [Output("curve-county", "options"),
     Output("curve-county", "value"),
     Output("curve-county", "disabled")],
    [Input("curve-state", "value")]
)
def state_selected(state):
    
    if state == "":
        # no state has been selected, so don't give county options
        options = []
        default_value = ""
        county_disabled = True
    elif state.startswith("All of ") | (state == "nan"):
        # whole state has been selected, so don't give county options
        options = []
        default_value = ""
        county_disabled = True
    else:
        # a state was selected, determine county options
        county_disabled = False
        default_value = "All of " + state
        options = [{"label": default_value, "value": "nan"}]
        county_ls = np.sort(df.loc[df["Province/State"] == state, "County"].unique().tolist())
        county_ls = county_ls[county_ls != "nan"]
        county_options = [{"label": county, "value": county} for county in county_ls]
        if len(county_ls) > 0:
            options.extend(county_options)
    return options, default_value, county_disabled

#################################################################
# 10 Callback for Updating Heat Map Figure
@app.callback(
    [Output("heatmap", "figure"),
     Output("initial-spinner", "style")],
    [Input("map-scope", "value"),
     Input("map-var", "value"),
     Input("map-calc", "value"),
     Input("map-scale", "value"),
     Input("map-norm-type", "value"),
     Input("map-norm-val", "value")],
    [State("initial-spinner", "style")]
)
def update_heatmap(map_scope, map_var, map_calc, map_scale, map_norm_type, map_norm_val,
                   init_spinner_style):
    
    # test if this is the initial execution of this callback
    is_init = (init_spinner_style is None)

    # only generate a new heatmap if the user initialized this callback
    if is_init:
        fig = init_heatmap

    else:

        # set null values of map parameters
        if map_calc == "Total":
            map_calc = ""
        if map_norm_type == "None":
            map_norm_type = ""
        plot_var = map_var + map_calc + map_norm_type

        frame_dur = 1000 # milliseconds, controls animation speed

        # set variables conditioned on the map scope
        if map_scope == "UScounties":
            geo_json = us_counties_json
            plot_df = df[df["MapScope"] == "US Counties"]
            plot_df["AreaLabel"] = plot_df.County.astype(str) + ", " + plot_df["Province/State"].astype(str)
            location_var = "FIPS"
            geo_json_name_field = None
            map_center = {"lat": 37.0902, "lon": -95.7129}
            title = "US counties"
            init_zoom = 3
        
        elif map_scope == "USstates":
            geo_json = us_states_json
            plot_df = df[df["MapScope"] == "US States"]
            plot_df["AreaLabel"] = plot_df["Province/State"].astype(str)
            location_var = "Province/State"
            geo_json_name_field = "properties.NAME"
            map_center = {"lat": 37.0902, "lon": -95.7129}
            title = "US states"
            init_zoom = 3
        
        elif map_scope == "China":
            geo_json = china_json
            plot_df = df[df["MapScope"] == "China Provinces"]
            plot_df["AreaLabel"] = plot_df["Province/State"].astype(str)
            location_var = "Province/State"
            geo_json_name_field = "properties.NL_NAME_1"
            map_center = {"lat": 37.110573, "lon": 106.493924}
            title = "Chinese provinces"
            init_zoom = 2
        
        elif map_scope == "Australia":
            geo_json = australia_json
            plot_df = df[df["MapScope"] == "Australia States"]
            plot_df["AreaLabel"] = plot_df["Province/State"].astype(str)
            location_var = "Province/State"
            geo_json_name_field = None
            map_center = {"lat": -26, "lon": 133 + 25/60}
            title = "Australian states"
            init_zoom = 3
        
        elif map_scope == "Canada":
            geo_json = canada_json
            plot_df = df[df["MapScope"] == "Canada Provinces"]
            plot_df["AreaLabel"] = plot_df["Province/State"].astype(str)
            location_var = "Province/State"
            geo_json_name_field = "properties.PRENAME"
            map_center = {"lat": 58, "lon": -96 - 48/60}
            title = "Canadian Provinces"
            init_zoom = 2
        
        elif map_scope == "World":
            geo_json = world_json
            plot_df = df[df["MapScope"] == "Countries"]
            plot_df["AreaLabel"] = plot_df["Country/Region"].astype(str)
            location_var = "Country/Region"
            geo_json_name_field = "properties.ADMIN"
            map_center = {"lat": 0, "lon": 0}
            title = "Countries"
            init_zoom = 0

        # set axis variables conditioned on scale settings
        var_finite = plot_df[plot_var].values
        var_finite = var_finite[(var_finite != 0) & (var_finite != -np.inf) & (var_finite != np.inf)]
        if len(var_finite) > 0:
            var_min = min(var_finite)
            var_max = max(var_finite)
        else:
            var_min = 0
            var_max = 0
        
        log_txt = ["1e-6", "1e-5", "1e-4", ".001", ".01", ".1", \
                "1", "10", "100", "1K", "10K", "100K", "1M"]
        map_log_hvr_txt = "Cases per " + log_txt[int(np.log10(map_norm_val)) + 6] + " Capita: "
        if map_scale == "Logarithmic":
            bar_scale_type = "log"
            map_tick_vals = np.arange(-6, 7)
            map_tick_txt = log_txt
            
            if map_norm_type == "PerCapita":
                plot_df["CaseVar"] = np.log10(plot_df[plot_var]*map_norm_val)
                bar_range = np.log10(np.array([var_min, var_max])*map_norm_val)
            else:
                plot_df["CaseVar"] = np.log10(plot_df[plot_var])
                bar_range = np.log10(np.array([var_min, var_max]))
        
        else:
            bar_scale_type = "linear"
            map_tick_vals = None
            map_tick_txt = None

            if map_norm_type == "PerCapita":
                plot_df["CaseVar"] = plot_df[plot_var]*map_norm_val
                bar_range = np.array([0, var_max])*map_norm_val
            else:
                plot_df["CaseVar"] = plot_df[plot_var]
                bar_range = np.array([0, var_max])
        
        if map_var == "Recovered":
            heat_color_scale = "ylgn"
            bar_color = "rgb(69, 161, 69)"
        else:
            heat_color_scale = "ylorrd"
            bar_color = "rgb(236, 62, 19)"
        
        days = np.sort(plot_df.Date.unique())

        # when the figure first loads, show the most recent date which has some data to plot
        if len(var_finite) > 0:
            date_has_data_df = plot_df.groupby(["Date"]).sum().reset_index()
            init_date = date_has_data_df.loc[date_has_data_df[plot_var] > 0, "Date"].max()
            init_date_ind = np.where(days == init_date.to_datetime64())[0][0]
        else:
            init_date = days[-1]
            init_date_ind = len(days) - 1
        plot_day_df = plot_df[plot_df.Date == init_date]

        # define custom hover data
        cust_data = np.dstack((plot_day_df.loc[:, map_var + map_calc].values, \
                            plot_day_df.loc[:, map_var + map_calc + "PerCapita"]. \
                                        values*map_norm_val))[0]
        location_series = plot_day_df[location_var]
        if map_norm_type == "PerCapita":
            bar_txt_format = "{:.2e}"
        else:
            bar_txt_format = "{:,.0f}"
        
        # define the left bar plot
        bar_df = plot_day_df.nlargest(10, plot_var, keep="all").reset_index()
        bar_df = bar_df.head(10) # nlargest may return more than 10 rows if there are duplicate values
        bar_df = bar_df[bar_df.CaseVar > -np.inf]
        nrows = bar_df.shape[0]
        bar_df = bar_df.iloc[np.arange(nrows - 1, -1, -1),:] # reverse order of top 10 rows

        # plotly does not tolerate changing the number of bars in 
        # a bar graph during animation define a function to pad 
        # data arrays with blank elements so the bar graph always 
        # has 10 elements
        def pad_10_arr(x, pad_val, unique_fill_bool):
            xlen = len(x)
            if xlen == 10:
                result = x
            else:
                npad = 10 - xlen
                fill_arr = np.array([pad_val for i in range(npad)])

                # shorten each string fill element in array to make the elements unique
                if unique_fill_bool:
                    fill_arr = [item[i:] for i, item in enumerate(fill_arr)]
                
                result = np.append(fill_arr, x)
            return result

        # only build the bar plot if there is data to plot
        if plot_df[plot_var].max() > 0:
            no_data = False

            max_width_label = 25
            if map_scope == "UScounties":

                # some of the county, state labels are too long, taking up too much space
                # in the figure.  Long labels will have the county label trimmed with an ellipsis appended.
                labels_to_trim = bar_df["AreaLabel"].astype(str).str.len() > max_width_label
                county_len_arr = max_width_label - 5 - bar_df.loc[labels_to_trim, "Province/State"].astype(str).str.len().values
                county_abbr = [bar_df.loc[labels_to_trim, "County"].astype(str).values[i][:county_len_arr[i]] \
                            for i in range(len(county_len_arr))]
                state_abbr = bar_df.loc[labels_to_trim, "Province/State"].astype(str).values.tolist()
                county_state_abbr = [county_abbr[i] + "..., " + state_abbr[i] for i in range(len(county_abbr))]
                bar_df.loc[labels_to_trim, "AreaLabel"] = county_state_abbr

            # bar labels must be padded so all labels have the same length
            # as some labels disappear and others are introduced,
            # varied-length label cause bad animation behavior
            area_labels = [label.rjust(max_width_label) for label in bar_df.AreaLabel.values]

            if map_norm_type == "PerCapita":
                bar_df[plot_var] = bar_df[plot_var] * map_norm_val
            
            bar_df["ValLabels"] = bar_df[plot_var].astype("float")
            bar_fig_data = go.Bar(x=pad_10_arr(bar_df[plot_var].values, 0, False),
                                y=pad_10_arr(area_labels, " " * max_width_label, True),
                                text=pad_10_arr(bar_df.ValLabels.map(bar_txt_format.format).values, "", False),
                                textposition="auto",
                                hoverinfo="none",
                                orientation="h",
                                marker_color=bar_color,
                                name="")
        else:
            no_data = True
            bar_fig_data = go.Bar(x=[],
                                y=[],
                                orientation="h",
                                name="")
        
        # build the heatmap
        heat_fig_data =go.Choroplethmapbox(geojson=geo_json,
                                        locations=location_series,
                                        featureidkey=geo_json_name_field,
                                        z=plot_day_df.CaseVar,
                                        zmin=0,
                                        zmax=plot_df.CaseVar.max(),
                                        customdata=cust_data,
                                        name="",
                                        text=plot_day_df.AreaLabel,
                                        hovertemplate="<b>%{text}</b><br>" + \
                                                        "<b>Cases</b>: %{customdata[0]:,}<br>" + \
                                                        "<b>" + map_log_hvr_txt + "</b>: %{customdata[1]:.2e}",
                                        colorbar=dict(outlinewidth=1,
                                                        outlinecolor="#333333",
                                                        len=0.9,
                                                        lenmode="fraction",
                                                        xpad=30,
                                                        xanchor="right",
                                                        bgcolor=None,
                                                        title=dict(text="Cases",
                                                                    font=dict(size=14)),
                                                        tickvals=map_tick_vals,
                                                        ticktext=map_tick_txt,
                                                        tickcolor="#333333",
                                                        tickwidth=2,
                                                        tickfont=dict(color="#333333",
                                                                    size=12)),
                                        colorscale=heat_color_scale,
                                        marker_opacity=0.7,
                                        marker_line_width=0)

        # define animation controls
        fig_ctrls = []
        sliders_dict = dict()

        # only define the animation controls of there is data to plot
        if plot_df[plot_var].max() > 0:
            fig_ctrls = [dict(type="buttons",
                            buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None,
                                                dict(frame=dict(duration=frame_dur,
                                                                redraw=True),
                                                    fromcurrent=True)]),
                                    dict(label="Pause",
                                            method="animate",
                                            args=[[None],
                                                dict(frame=dict(duration=0,
                                                                redraw=True),
                                                    mode="immediate")])],
                                direction="left",
                                pad={"r": 10, "t": 35},
                                showactive=False,
                                x=0.1,
                                xanchor="right",
                                y=0,
                                yanchor="top")]

            if (not is_init):
                sliders_dict = dict(active=init_date_ind,
                                    visible=True,
                                    yanchor="top",
                                    xanchor="left",
                                    currentvalue=dict(font=dict(size=14),
                                                    prefix="Plotted Date: ",
                                                    visible=True,
                                                    xanchor="center"),
                                    pad=dict(b=10,
                                            t=10),
                                    len=0.875,
                                    x=0.125,
                                    y=0,
                                    steps=[])

        # define the animation frames
        fig_frames = []
        if is_init:
            fig_frames = init_fig_frames
            sliders_dict = init_slider_steps

        # only define the animation frames if there is data to plot
        elif plot_df[plot_var].max() > 0:
            for day in days:

                # this code repeating what was done to build the initial bar plot above
                plot_day_df = plot_df[plot_df.Date == day]
                bar_df = plot_day_df.nlargest(10, plot_var, keep="all").reset_index()
                bar_df = bar_df.head(10) # nlargest may return more than 10 rows if there are duplicate values
                bar_df = bar_df[bar_df.CaseVar > -np.inf]
                nrows = bar_df.shape[0]
                bar_df = bar_df.iloc[np.arange(nrows - 1, -1, -1),:] # reverse order of top 10 rows
                area_labels = [label.rjust(max_width_label) for label in bar_df.AreaLabel.values]
                if map_norm_type == "PerCapita":
                    bar_df[plot_var] = bar_df[plot_var] * map_norm_val
                bar_df["ValLabels"] = bar_df[plot_var].astype("float")

                # this code repeats what was done to build the initial heatmap above
                cust_data = np.dstack((plot_day_df.loc[:, map_var + map_calc].values, \
                                    plot_day_df.loc[:, map_var + map_calc + "PerCapita"]. \
                                                values*map_norm_val))[0]
                location_series = plot_day_df[location_var]
                
                # define the frame, repeatinf what was done for the initial plots above
                frame = go.Frame(data=[go.Bar(x=pad_10_arr(bar_df[plot_var].values, 0, False),
                                            y=pad_10_arr(area_labels, " " * max_width_label, True),
                                            text=pad_10_arr(bar_df.ValLabels.map(bar_txt_format.format). \
                                                                    values, "", False),
                                            textposition="auto",
                                            hoverinfo="none",
                                            name=""),
                                    go.Choroplethmapbox(locations=location_series,
                                                        featureidkey=geo_json_name_field,
                                                        z=plot_day_df.CaseVar,
                                                        customdata=cust_data,
                                                        name="",
                                                        text=plot_day_df.AreaLabel,
                                                        hovertemplate="<b>%{text}</b><br>" + \
                                                                        "<b>Cases</b>: %{customdata[0]:,}<br>" + \
                                                                        "<b>" + map_log_hvr_txt + "</b>: %{customdata[1]:.2e}")],
                                name=numpy_dt64_to_str(day))
                fig_frames.append(frame)

                # define the slider step
                slider_step = dict(args=[[numpy_dt64_to_str(day)],
                                        dict(mode="immediate",
                                            frame=dict(duration=300,
                                                        redraw=True))],
                                method="animate",
                                label=numpy_dt64_to_str(day))
                sliders_dict["steps"].append(slider_step)

        # Assemble the entire figure based on the components defined above
        fig = subplots.make_subplots(rows=1, cols=2, column_widths=[0.2, 0.8],
                                    subplot_titles=("Top 10 " + title, ""),
                                    horizontal_spacing=0.05,
                                    specs=[[{"type": "bar"},
                                            {"type": "choroplethmapbox"}]])
        fig.add_trace(bar_fig_data, row=1, col=1)
        fig.add_trace(heat_fig_data, row=1, col=2)
        fig.update_layout(mapbox_style="light",
                        mapbox_zoom=init_zoom,
                        mapbox_accesstoken=token,
                        mapbox_center=map_center,
                        margin={"r":10,"t":30,"l":10,"b":10},
                        plot_bgcolor="white",
                        sliders=[sliders_dict],
                        updatemenus=fig_ctrls)
        fig["frames"] = fig_frames
        
        # update the bar plot axes
        if no_data:
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
        else:
            fig.update_xaxes(type=bar_scale_type,
                            ticks="outside",
                            range=bar_range,
                            showgrid=True,
                            gridwidth=0.5,
                            gridcolor="#CCCCCC")

        if no_data:
            # add annotation when theres no data explaining as such
            fig["layout"]["annotations"] = [dict(x=0,
                                                y=0,
                                                xref="x1", 
                                                yref="y1",
                                                text="All<br>" + title + "<br>have reported<br>zero " + \
                                                    map_var + "<br>cases to date",
                                                showarrow=False,
                                                font=dict(size=16))]
        else:
            # modify the bar plot title font properties
            fig["layout"]["annotations"][0]["font"] = dict(size=16)
    
    # return the figure and hide the dbc.Spinner which is shown during initial app loading
    return fig, {"display": "none"}

#################################################################
# 11 Callback for Adding Rows to curve_plot_df (dataframe define curves to plot)
@app.callback(
    [Output("data-table", "data"),
     Output("curve-plot-df", "children"),
     Output("curve-drop", "options"),
     Output("curve-add-click-count", "children"),
     Output("curve-clear-click-count", "children")],
    [Input("curve-add", "n_clicks"),
     Input("curve-clear", "n_clicks"),
     Input("curve-drop", "value")],
    [State("curve-country", "value"),
     State("curve-state", "value"),
     State("curve-county", "value"),
     State("curve-plot-df", "children"),
     State("curve-add-click-count", "children"),
     State("curve-clear-click-count", "children")],
)
def edit_plotted_curves(add_click, clear_click, drop_row_id, country, state, \
                        county, df_as_json, add_click_last, clear_click_last):
    
    # read the df from the hidden div json data
    curve_plot_df = pd.read_json(df_as_json[0], orient='split')

    # determine whether this callback was triggered by the Add button, the Clear All button
    # or Drop Row dropdown
    if add_click > add_click_last:
        if state.startswith("All of "):
            state = "nan"
            county = "nan"
        elif county.startswith("All of "):
            county = "nan"
        nrows = curve_plot_df.shape[0]
        curve_plot_df.loc[nrows] = [nrows, country, state, county]
    
    elif clear_click > clear_click_last:
        curve_plot_df = curve_plot_df.loc[curve_plot_df["Row ID"] == -999]
    
    elif drop_row_id != "":
        curve_plot_df = curve_plot_df.loc[curve_plot_df["Row ID"] != int(drop_row_id)]
        curve_plot_df = curve_plot_df.reset_index(drop=True)
        curve_plot_df["Row ID"] = curve_plot_df.index

    # write the new df to the ui data table and to the hidden div
    return curve_plot_df.replace("nan", "").to_dict("records"), \
           [curve_plot_df.to_json(date_format='iso', orient='split')], \
           [{"label": val, "value": val} for val in [""] + curve_plot_df["Row ID"].tolist()], \
           add_click, clear_click

#################################################################
# 12 Callback for Updating Curves Plot Figure
@app.callback(
    Output("curves", "figure"),
    [Input("curve-plot-df", "children"),
     Input("curve-calc", "value"),
     Input("curve-norm-type", "value"),
     Input("curve-norm-val", "value"),
     Input("curve-zero", "value"),
     Input("curve-type", "value"),
     Input("curve-scale", "value"),
     Input("curve-avg-period", "value")]
)
def update_curves_plot(curve_plot_df_as_json, calc, norm_type, norm_val, zero_opt, \
                       types_ls, y_axis_type, avg_period):
    
    # define function which gives a string label for order of magnitude (log10)
    def logtxt(val):
        log_txt_opts = ["1e-6", "1e-5", "1e-4", ".001", ".01", ".1", \
                        "1", "10", "100", "1K", "10K", "100K", "1M"]
        log_txt = log_txt_opts[int(np.log10(val)) + 6]
        return log_txt

    # define a function which will scatter points and a fit curve line corresponding to
    # place and type of variable
    def add_cust_traces(fig, var, place_name, df, color):

        # determine the basic variable type to be plotted
        if var[:1] == "A":
            var_type = "Active"
        elif var[:1] == "C":
            var_type = "Confirmed"
        elif var[:1] == "R":
            var_type = "Recovered"
        elif var[:1] == "D":
            var_type = "Deaths"

        # assign marker and line styles depending on how many basic types
        # of variables are to plotted
        var_id = np.where(np.array(types_ls) == var_type)[0][0]
        dash_ls = ["solid", "dash", "dot", "longdash"]
        symbol_ls = ["circle", "square", "diamond", "triangle-up"]

        if zero_opt == "None":
            x_axis_var = "Date"
        else:
            x_axis_var = "Zero_Day"

        # define hover text for scatter points
        per_cap = " Cases per " + logtxt(norm_val) + " Capita"
        base_hover_txt = "<b>" + place_name + "</b><br>" + \
                        "<b>Date</b>: %{text}" + \
                        "<br><b>Days Elapsed</b>: %{customdata[0]}"
        if calc == "":
            hover_txt = base_hover_txt + \
                            "<br><b>Total " + var_type + " To Date</b>: %{customdata[1]:,.0f}" + \
                            "<br><b>Total " + var_type + " To Date</b>:<br>" + \
                            "%{customdata[2]:.2e}" + per_cap
        elif calc == "PerDate":
            hover_txt = base_hover_txt + \
                            "<br><b>New " + var_type + " On Date</b>: %{customdata[1]:,.0f}" + \
                            "<br><b>New " + var_type + " On Date</b>:<br>" + \
                            "%{customdata[2]:.2e}" + per_cap

        # plot scatter data points
        fig.add_trace(go.Scatter(x=df[x_axis_var],
                                 y=df[var],
                                 mode='markers',
                                 name="",
                                 marker=dict(symbol=symbol_ls[var_id],
                                             size=8,
                                             color=color,
                                             opacity=0.4),
                                 customdata=np.dstack((df.loc[:, "Zero_Day"].values, \
                                                       df.loc[:, var_type + calc].values, \
                                                       df.loc[:, var_type + calc + "PerCapita"].values))[0],
                                 text=df.Date.dt.strftime('%B %d, %Y'),
                                 hovertemplate=hover_txt,
                                 showlegend=False))

        # define the hover text for the fit curve line
        fit_hover_txt = "<b>Curve Fit for " + place_name + "</b><br>" + \
                        "<b>Date</b>: %{text}" + \
                        "<br><b>Days Elapsed</b>: %{customdata[0]}"
        if calc == "":
            fit_hover_txt = fit_hover_txt + \
                            "<br><b>Fit Total " + var_type + " To Date</b>: %{customdata[1]:,.0f}" + \
                            "<br><b>Fit Total " + var_type + " To Date</b>:<br>" + \
                            "%{customdata[2]:.2e}" + per_cap
        elif calc == "PerDate":
            fit_hover_txt = fit_hover_txt + \
                            "<br><b>Fit New " + var_type + " On Date</b>: %{customdata[1]:,.0f}" + \
                            "<br><b>Fit New " + var_type + " On Date</b>:<br>" + \
                            "%{customdata[2]:.2e}" + per_cap
        
        # plot the fit curve line
        fig.add_trace(go.Scatter(x=df[x_axis_var],
                                 y=df[var + "Avg"],
                                 mode='lines',
                                 name="",
                                 line=dict(width=3, dash=dash_ls[var_id], color=color),
                                 customdata=np.dstack((df.loc[:, "Zero_Day"].values, \
                                                       df.loc[:, var_type + calc + "Avg"].values, \
                                                       df.loc[:, var_type + calc + "PerCapita" + "Avg"].values))[0],
                                 text=df.Date.dt.strftime('%B %d, %Y'),
                                 hovertemplate=fit_hover_txt,
                                 showlegend=False))
        return fig
    
    # set null values of plot parameters
    if calc == "Total":
        calc = ""
    if norm_type == "None":
        norm_type = ""
    
    # make a list of all curves to be plotted
    plot_vars_ls = [plot_type + calc + norm_type for plot_type in types_ls]

    # read the df from the hidden div json data
    # this df defines the country/state/county areas which are to be plotted
    curve_plot_df = pd.read_json(curve_plot_df_as_json[0], orient='split')

    # setup matplotlib colors for distinguishing curves
    nplaces = curve_plot_df.shape[0]
    ncolors = max(nplaces, len(types_ls))
    cmap = cm.get_cmap("tab10", ncolors)    # PiYG
    colors = ["" for i in range(ncolors)]
    for i in range(cmap.N):
        rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
        colors[i] = matplotlib.colors.rgb2hex(rgb)
    
    # set options for deriving the Zero_Day column & X-axis label
    max_zero_day = 0
    y_min, y_max = 0, 1
    if zero_opt == "None":
        zero_thresh = 1
        thresh_var = "Confirmed"
    elif zero_opt == "Total":
        zero_thresh = 1
        thresh_var = "Confirmed"
    elif zero_opt == "PerCapita":
        zero_thresh = 1/10000
        thresh_var = "ConfirmedPerCapita"

    # define a blank figure as the default
    fig = go.Figure()

    # fill the figure with data if places have been identified by the user
    if nplaces > 0:
    
        # pandas doesn't like df == np.nan, so tests are needed to determine proper syntax
        # define a function which generically filters for country, state & county
        def filter_mask_csc(df, country, state, county):
            if isinstance(county, str):
                mask = (df["Country/Region"] == country) & \
                       (df["Province/State"] == state) & \
                       (df["County"] == county)
            elif isinstance(state, str):
                mask = (df["Country/Region"] == country) & \
                       (df["Province/State"] == state) & \
                       (df["County"] == "nan")
            else:
                mask = (df["Country/Region"] == country) & \
                       (df["Province/State"] == "nan") & \
                       (df["County"] == "nan")
            return mask

        # generate a local df containing only the data that will be plotted
        # this will make subsequent df manipulation faster
        mask_bool_ls = [filter_mask_csc(df, curve_plot_df["Country/Region"][i],
                                        curve_plot_df["Province/State"][i],
                                        curve_plot_df.County[i]
                                        )
                        for i in range(nplaces)
                       ]

        # the list of masks needs to consolidated via OR into a single mask
        mask_bool = np.array([False for i in range(df.shape[0])])
        for mask_bool_item in mask_bool_ls:
            mask_bool = mask_bool | mask_bool_item
        plot_df = df[mask_bool]

        # ensure line plots will move left to right
        plot_df = plot_df.sort_values(["Date"]).reset_index()

        # initialize values to be ammended in subsequent for loops
        item_counter = 0
        min_date = plot_df.Date.max()
        max_date = plot_df.Date.min()

        # build the figure piecewise, adding traces within for loops
        for place_i in range(nplaces):
            
            # isolate data for place_i
            curve_row = curve_plot_df.iloc[place_i, :]
            var_mask_bool = filter_mask_csc(plot_df, \
                                            curve_row["Country/Region"], \
                                            curve_row["Province/State"], \
                                            curve_row["County"])
            plot_var_df = plot_df[var_mask_bool]

            # calculate zero day column for place_i
            plot_var_df["Zero_Day"] = 0
            started_df = plot_var_df[plot_var_df[thresh_var] >= zero_thresh]
            start_date_series = started_df.Date[:1] - pd.Timedelta(days=1)
            plot_var_df.Zero_Day = plot_var_df.Date - start_date_series.squeeze()
            plot_var_df.Zero_Day = plot_var_df.Zero_Day.dt.days
            plot_var_df = plot_var_df[plot_var_df.Zero_Day > 0]

            # keep track of x-axis range limits across all plotted places
            max_zero_day = max([max_zero_day, plot_var_df.Zero_Day.max()])
            min_date = min([min_date, plot_var_df.Date.min()])
            max_date = max([max_date, plot_var_df.Date.max()])

            # calculate moving average columns for place_i
            for plot_type in types_ls:

                # calculate moving average for accumulating cases
                var_to_avg = plot_type + calc
                plot_var_df[var_to_avg + "Avg"] = \
                    plot_var_df[var_to_avg].rolling(avg_period, center=True, min_periods=1).mean()

                # calculate moving average for new cases
                var_pc_to_avg = plot_type + calc + "PerCapita"
                plot_var_df[var_pc_to_avg + "Avg"] = \
                    plot_var_df[var_pc_to_avg].rolling(avg_period, center=True, min_periods=1).mean()
            
            # get the name of place_i
            place_elements = [elem for elem in curve_row.replace(np.nan, "").values[1:] if elem != ""]
            place_name = ", ".join(place_elements)

            # add traces for each variable type to be plotted for place_i
            for var in plot_vars_ls:
                fig = add_cust_traces(fig, var, place_name, plot_var_df, colors[item_counter])
                
                # add dummy trace for legend only if a single place is being plotted
                # this will utilize different colors for variable types
                if nplaces == 1:
                    fig.add_trace(go.Scatter(x=[None],
                                             y=[None],
                                             mode='lines+markers',
                                             name=types_ls[item_counter],
                                             line=dict(dash="solid",
                                             color=colors[item_counter]),
                                  showlegend=True))
                    item_counter += 1

            # add a dummy trace for legend only if more than one place is being plotted
            # this will utilize different colors for places
            if nplaces > 1:
                fig.add_trace(go.Scatter(x=[None],
                                         y=[None],
                                         mode='lines',
                                         name=place_name,
                                         line=dict(dash="solid", color=colors[item_counter]),
                              showlegend=True))
                item_counter += 1

    axopts = dict(linecolor = "gray", linewidth = 0.5, showline = True, mirror=True)
    fig.update_layout(
        paper_bgcolor=invis,
        plot_bgcolor=invis,
        margin=go.layout.Margin(l=50, r=20, b=10, t=10),
        xaxis=axopts,
        yaxis=axopts,
        showlegend=True,
        legend=go.layout.Legend(
            x=0,
            y=-0.25,
            traceorder="reversed",
            font=dict(
                      family="sans-serif",
                      size=12,
                      color="black"),
            bgcolor="white",
            bordercolor="gray",
            borderwidth=0.5),
        legend_orientation="h")

    # add dummy trace for basic variable types as parrt of custom legend
    if nplaces > 1:
        if "Confirmed" in types_ls:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines+markers',
                                     marker=dict(size=8, color='black', symbol="circle"),
                                     line=dict(dash="solid"), showlegend=True, name='Confirmed'))

        if "Recovered" in types_ls:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines+markers',
                                     marker=dict(size=8, color='black', symbol="square"),
                                     line=dict(dash="dash"), showlegend=True, name='Recovered'))
        
        if "Deaths" in types_ls:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines+markers',
                                     marker=dict(size=8, color='black', symbol="diamond"),
                                     line=dict(dash="dot"), showlegend=True, name='Deaths'))

    # setup x-axis
    if zero_opt == "None":
        x_margin = 0.1*(max_date - min_date)
        fig.update_xaxes(title_text="Date", showspikes=True, spikesnap="data", spikemode="across", \
                         spikethickness=2)
        fig.layout.xaxis.range = [pd.to_datetime(min_date - x_margin), pd.to_datetime(max_date + x_margin)]
    else:
        fig.update_xaxes(title_text="Days Having " + str(zero_thresh) + " or More Confirmed Cases", \
                         showspikes=True, spikesnap="data", spikemode="across", spikethickness=2)
        fig.layout.xaxis.range = [0, max_zero_day*1.1]
    
    if nplaces > 0:
        # setup y-axis
        if y_axis_type == "linear":
            y_min = 0
            y_max = 1.1*plot_df[plot_vars_ls].max().max()
        elif (y_axis_type == "log") & (norm_type == ""):
            y_vals_arr = plot_df[plot_vars_ls].values.flatten()
            y_nonzero_vals_arr = y_vals_arr[y_vals_arr != 0]
            y_min = 0.2
            y_max = max([0.9*np.log10(max(y_nonzero_vals_arr)), 1.1*np.log10(max(y_nonzero_vals_arr))])
        elif (y_axis_type == "log") & (norm_type == "PerCapita"):
            y_vals_arr = plot_df[plot_vars_ls].values.flatten()
            y_nonzero_vals_arr = y_vals_arr[y_vals_arr != 0]
            y_min = min([0.9*np.log10(min(y_nonzero_vals_arr)), 1.1*np.log10(min(y_nonzero_vals_arr))])
            y_max = max([0.9*np.log10(max(y_nonzero_vals_arr)), 1.1*np.log10(max(y_nonzero_vals_arr))])

    if calc == "":
        if norm_type == "":
            fig.update_yaxes(title_text="Cumulative Cases", type=y_axis_type, showspikes=True,
                             spikesnap="data", spikemode="across", spikethickness=2, \
                             range=[y_min, y_max])
        elif norm_type == "PerCapita":
            fig.update_yaxes(title_text="Cumulative Cases Per " + logtxt(norm_val) + " Capita", \
                             showspikes=True, spikesnap="data", spikemode="across",
                             spikethickness=2, type=y_axis_type, range=[y_min, y_max])
    elif calc == "PerDate":
        if norm_type == "":
            fig.update_yaxes(title_text="Cases Per Date", showspikes=True, type=y_axis_type,
                             spikesnap="data", spikemode="across", spikethickness=2, \
                             range=[y_min, y_max])
        elif norm_type == "PerCapita":
            fig.update_yaxes(title_text="Cases Per Date Per " + logtxt(norm_val) + " Capita", \
                             showspikes=True, spikesnap="data", spikemode="across",
                             spikethickness=2, type=y_axis_type, range=[y_min, y_max])
    
    return fig

#################################################################
# 13 Callback for Updating the first Epidemiology Sandbox Figure
@app.callback(
    [Output("sandbox1-area-fig", "figure"),
     Output("sandbox1-line-fig", "figure"),
     Output("sandbox1-scenario1-d-text", "children"),
     Output("sandbox1-scenario1-r-text", "children"),
     Output("sandbox1-scenario2-d-text", "children"),
     Output("sandbox1-scenario2-r-text", "children")],
    [Input("sandbox1-scenario1-d", "value"),
     Input("sandbox1-scenario1-r", "value"),
     Input("sandbox1-scenario2-d", "value"),
     Input("sandbox1-scenario2-r", "value")]
)
def update_sandbox1(d_scenario1, r_scenario1, d_scenario2, r_scenario2):
    
    # setup matplotlib colors for distinguishing curves
    ncolors = 10
    cmap = cm.get_cmap("tab10", ncolors)    # PiYG
    colors = ["" for i in range(ncolors)]
    for i in range(cmap.N):
        rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
        colors[i] = matplotlib.colors.rgb2hex(rgb)
    
    scenario1 = predict_sir(10000, [[0, d_scenario1, r_scenario1]], np.inf)
    scenario2 = predict_sir(10000, [[0, d_scenario2, r_scenario2]], np.inf)

    # re-run whichever scenario terminated first so that it will have the 
    # same duration as the longer scenario
    scenario1_dur = scenario1.shape[0]
    scenario2_dur = scenario2.shape[0]
    if scenario1_dur > scenario2_dur:
        scenario2 = predict_sir(10000, [[0, d_scenario2, r_scenario2]], scenario1_dur)
    elif scenario1_dur < scenario2_dur:
        scenario1 = predict_sir(10000, [[0, d_scenario1, r_scenario1]], scenario2_dur)
    scenarios_ls = [scenario1, scenario2]
    scenario_names = ["Scenario #1", "Scenario #2"]

    area_fig = subplots.make_subplots(rows=2, cols=1, vertical_spacing=0.03, \
                                      subplot_titles=scenario_names)

    names = ["Susceptible", "Infected", "Removed"]
    for i, scenario in enumerate(scenarios_ls):
        for j in range(3):
            area_fig.add_trace(
                go.Bar(
                    name="",
                    x=scenario[:,0],
                    y=scenario[:, j + 1],
                    marker=dict(line=dict(width=0), color=colors[j]),
                    text=[names[j] for k in range(len(scenario[:,0]))],
                    hovertemplate="Day %{x}<br>%{y:,.0f} %{text}",
                    showlegend=False),
            row=i + 1, col=1)

    area_fig.add_traces([
        go.Bar(name="Removed", x=[None], y=[None], showlegend=True, \
            marker=dict(line=dict(width=0), color=colors[2])),
        go.Bar(name="Infected", x=[None], y=[None], showlegend=True, \
            marker=dict(line=dict(width=0), color=colors[1])),
        go.Bar(name="Susceptible", x=[None], y=[None], showlegend=True, \
            marker=dict(line=dict(width=0), color=colors[0])),
    ])

    axopts = dict(linecolor = "gray", linewidth = 0.5, showline = False, mirror=True)
    area_fig.update_layout(
        barmode="stack",
        bargap=0,
        showlegend=True,
        legend=go.layout.Legend(x=0, y=1.07, traceorder="reversed", orientation="h"),
        paper_bgcolor=invis,
        plot_bgcolor=invis,
        height=600,
        margin=go.layout.Margin(l=20, r=20, b=10, t=10),
        xaxis=axopts,
        yaxis=axopts)
    area_fig.update_xaxes(row=1, col=1, title_text=None, \
                          showspikes=True, spikemode="marker", spikecolor=colors[3])
    area_fig.update_xaxes(row=2, col=1, title_text="Days of Epidemic", \
                          showspikes=True, spikemode="marker", spikecolor=colors[3])
    area_fig.update_yaxes(row=1, col=1, title_text="Population")
    area_fig.update_yaxes(row=2, col=1, title_text="Population")

    # moving plot titles inside plot area, increasing their font size & making them white
    area_fig.layout.annotations[0].update(y=0.88)
    area_fig.layout.annotations[0].font.update(size=18)
    area_fig.layout.annotations[0].font.update(color="#ffffff")
    area_fig.layout.annotations[1].update(y=0.36)
    area_fig.layout.annotations[1].font.update(size=18)
    area_fig.layout.annotations[1].font.update(color="#ffffff")

    lines_fig = go.Figure()
    scenario_colors = [colors[1], colors[9]]
    for i, scenario in enumerate(scenarios_ls):
        lines_fig.add_trace(go.Scatter(
            name="",
            mode="lines",
            x=scenario[:,0],
            y=scenario[:, 2],
            line=dict(color=scenario_colors[i], width=4),
            text=[scenario_names[i] for k in range(len(scenario[:,0]))],
            hovertemplate="<b>%{text}</b><br>Day %{x}<br>%{y:,.0f} Infected",
            showlegend=False,
        ))

        # add dummy traces to fill the legend
        lines_fig.add_trace(go.Scatter(
            name=scenario_names[i],
            mode="lines",
            x=[None],
            y=[None],
            line=dict(color=scenario_colors[i], width=4),
            showlegend=True,
        ))
    
    lines_fig.update_layout(
        title={
            "text": "Comparing Infection Waves",
            "y":0.9,
            "x":0.5,
            "xanchor": "center",
            "yanchor": "top"
        },
        xaxis_title="Days of Epidemic",
        yaxis_title="Infected People",
        legend=go.layout.Legend(x=0, y=-0.35, orientation="h"),
        paper_bgcolor=invis,
        plot_bgcolor=invis,
        height=200,
        margin=go.layout.Margin(l=20, r=20, b=10, t=10),
        xaxis=axopts,
        yaxis=axopts)

    return [area_fig, lines_fig, d_scenario1, f"{r_scenario1:.1f}", \
            d_scenario2, f"{r_scenario2:.1f}"]

#################################################################
# 14 Callbacks to Update UI of the Second Epidemiology Sandbox
# update UI text elements for sandbox2 new slider values
@app.callback(
    [Output("sandbox2-new-d-text", "children"),
     Output("sandbox2-new-r-text", "children")],
    [Input("sandbox2-new-d", "value"),
     Input("sandbox2-new-r", "value")]
)
def update_sandbox2_ui(d_new, r_new):

    return d_new, f"{r_new:.1f}"

# add callback for adding rows to the sandbox2_df
@app.callback(
    [Output("sandbox2-baseline-d-text", "children"),
     Output("sandbox2-baseline-r-text", "children"),
     Output("sandbox2-add-click-count", "children"),
     Output("sandbox2-clear-click-count", "children"),
     Output("sandbox2-data-table", "data"),
     Output("sandbox2-df", "children")],
    [Input("sandbox2-baseline-d", "value"),
     Input("sandbox2-baseline-r", "value"),
     Input("sandbox2-add", "n_clicks"),
     Input("sandbox2-clear", "n_clicks"),
     Input("sandbox2-drop", "value")],
    [State("sandbox2-new-d", "value"),
     State("sandbox2-new-r", "value"),
     State("sandbox2-new-t", "value"),
     State("sandbox2-add-click-count", "children"),
     State("sandbox2-clear-click-count", "children"),
     State("sandbox2-df", "children")]
)
def update_sandbox2_ui(d_baseline, r_baseline, add_click, clear_click, \
                       drop_row_id, d_new, r_new, t_new, add_click_last, \
                       clear_click_last, df_as_json):

    # read the df from the hidden div json data
    sandbox2_df = pd.read_json(df_as_json[0], orient='split')
    sandbox2_df.iloc[0, :] = [0, d_baseline, r_baseline, True, True]

    # format t_new
    if t_new is not None:
        t_new = float(t_new)

    # determine whether this callback was triggered by the Add button, the Clear All button
    # or Drop Row dropdown
    if (add_click > add_click_last) & (t_new is not None):

        if t_new in sandbox2_df.t.values.tolist():
            # update row values where t_new collides
            sandbox2_df.loc[sandbox2_df.t == t_new] = [t_new, d_new, r_new, True, True]
        
        else:

            # add row to bottom of dataframe, then sort by time column 
            nrows = sandbox2_df.shape[0]
            sandbox2_df.loc[nrows] = [t_new, d_new, r_new, True, True]
            sandbox2_df = sandbox2_df.sort_values(["t"]).reset_index(drop=True)
    
    elif clear_click > clear_click_last:
        sandbox2_df = sandbox2_df[sandbox2_df.t == 0]
    
    elif drop_row_id != "":
        sandbox2_df = sandbox2_df.loc[sandbox2_df.t != float(drop_row_id)]
        sandbox2_df = sandbox2_df.reset_index(drop=True)
    
    # correct the In Base & In Alt columns
    sandbox2_df["In Alt"] = True
    sandbox2_df["In Base"] = True
    if sandbox2_df.shape[0] > 1:
        sandbox2_df.iloc[-1,-2] = False

    return d_baseline, f"{r_baseline:.1f}", add_click, clear_click, \
           sandbox2_df.replace(True, "Yes").replace(False, "No").to_dict("records"), \
           [sandbox2_df.to_json(date_format='iso', orient='split')]

#################################################################
# 15 Callback for Updating the Second Epidemiology Sandbox Figure
@app.callback(
    [Output("sandbox2-area-fig", "figure"),
     Output("sandbox2-lines-fig", "figure"),
     Output("sandbox2-drop", "options")],
    [Input("sandbox2-df", "children")]
)
def update_sandbox2_fig(df_as_json):

    # read the df from the hidden div json data
    sandbox2_df = pd.read_json(df_as_json[0], orient='split')

    # set the options for rows to drop from the dataframe
    row_opts_ls = sandbox2_df.loc[sandbox2_df.t != 0, "t"].values.tolist()
    row_opts_format_ls = [{"label": val, "value": val} for  val in row_opts_ls]

    # setup matplotlib colors for distinguishing curves
    ncolors = 10
    cmap = cm.get_cmap("tab10", ncolors)    # PiYG
    colors = ["" for i in range(ncolors)]
    for i in range(cmap.N):
        rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
        colors[i] = matplotlib.colors.rgb2hex(rgb)
    
    n_scenarios = sandbox2_df.shape[0]
    scenario2_params = [sandbox2_df.iloc[i, :4].values for i in range(n_scenarios)]
    scenario2 = predict_sir(10000, scenario2_params, np.inf)
    if n_scenarios > 1:
        scenario1_params = [sandbox2_df.iloc[i, :4].values for i in range(n_scenarios - 1)]
        scenario1 = predict_sir(10000, scenario1_params, np.inf)
    else:
        scenario1 = scenario2

    # re-run whichever scenario terminated first so that it will have the 
    # same duration as the longer scenario
    scenario1_dur = scenario1.shape[0]
    scenario2_dur = scenario2.shape[0]
    if scenario1_dur > scenario2_dur:
        scenario2 = predict_sir(10000, scenario2_params, scenario1_dur)
    elif scenario1_dur < scenario2_dur:
        scenario1 = predict_sir(10000, scenario1_params, scenario2_dur)
    scenarios_ls = [scenario1, scenario2]
    scenario_names = ["Baseline Scenario", "Alternate Scenario"]

    area_fig = subplots.make_subplots(rows=2, cols=1, vertical_spacing=0.03, \
                                      subplot_titles=scenario_names)

    names = ["Susceptible", "Infected", "Removed"]
    for i, scenario in enumerate(scenarios_ls):
        for j in range(3):
            area_fig.add_trace(
                go.Bar(
                    name="",
                    x=scenario[:,0],
                    y=scenario[:, j + 1],
                    marker=dict(line=dict(width=0), color=colors[j]),
                    text=[names[j] for k in range(len(scenario[:,0]))],
                    hovertemplate="Day %{x}<br>%{y:,.0f} %{text}",
                    showlegend=False),
            row=i + 1, col=1)

    area_fig.add_traces([
        go.Bar(name="Removed", x=[None], y=[None], showlegend=True, \
            marker=dict(line=dict(width=0), color=colors[2])),
        go.Bar(name="Infected", x=[None], y=[None], showlegend=True, \
            marker=dict(line=dict(width=0), color=colors[1])),
        go.Bar(name="Susceptible", x=[None], y=[None], showlegend=True, \
            marker=dict(line=dict(width=0), color=colors[0])),
    ])

    axopts = dict(linecolor = "gray", linewidth = 0.5, showline = False, mirror=True)
    area_fig.update_layout(
        barmode="stack",
        bargap=0,
        showlegend=True,
        legend=go.layout.Legend(x=0, y=1.07, traceorder="reversed", orientation="h"),
        paper_bgcolor=invis,
        plot_bgcolor=invis,
        height=600,
        margin=go.layout.Margin(l=20, r=20, b=10, t=10),
        xaxis=axopts,
        yaxis=axopts)
    area_fig.update_xaxes(row=1, col=1, title_text=None, \
                          showspikes=True, spikemode="marker", spikecolor=colors[3])
    area_fig.update_xaxes(row=2, col=1, title_text="Days of Epidemic", \
                          showspikes=True, spikemode="marker", spikecolor=colors[3])
    area_fig.update_yaxes(row=1, col=1, title_text="Population")
    area_fig.update_yaxes(row=2, col=1, title_text="Population")

    # moving plot titles inside plot area, increasing their font size & making them white
    area_fig.layout.annotations[0].update(y=0.88)
    area_fig.layout.annotations[0].font.update(size=18)
    area_fig.layout.annotations[0].font.update(color="#ffffff")
    area_fig.layout.annotations[1].update(y=0.36)
    area_fig.layout.annotations[1].font.update(size=18)
    area_fig.layout.annotations[1].font.update(color="#ffffff")

    lines_fig = go.Figure()
    scenario_colors = [colors[1], colors[9]]
    for i, scenario in enumerate(scenarios_ls):
        lines_fig.add_trace(go.Scatter(
            name="",
            mode="lines",
            x=scenario[:,0],
            y=scenario[:, 2],
            line=dict(color=scenario_colors[i], width=4),
            text=[scenario_names[i] for k in range(len(scenario[:,0]))],
            hovertemplate="<b>%{text}</b><br>Day %{x}<br>%{y:,.0f} Infected",
            showlegend=False,
        ))

        # add dummy traces to fill the legend
        lines_fig.add_trace(go.Scatter(
            name=scenario_names[i],
            mode="lines",
            x=[None],
            y=[None],
            line=dict(color=scenario_colors[i], width=4),
            showlegend=True,
        ))
    
    lines_fig.update_layout(
        title={
            "text": "Comparing Infection Waves",
            "y":0.9,
            "x":0.5,
            "xanchor": "center",
            "yanchor": "top"
        },
        xaxis_title="Days of Epidemic",
        yaxis_title="Infected People",
        legend=go.layout.Legend(x=0, y=-0.35, orientation="h"),
        paper_bgcolor=invis,
        plot_bgcolor=invis,
        height=200,
        margin=go.layout.Margin(l=20, r=20, b=10, t=10),
        xaxis=axopts,
        yaxis=axopts)

    return area_fig, lines_fig, row_opts_format_ls


if __name__ == '__main__':
    app.run_server(debug=False)