# DASH IMPORTS
import dash
import dash_core_components as dcc
import dash_html_components as html
#import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash.exceptions import PreventUpdate
from flask_caching import Cache
# DATA ANALYSIS IMPORTS
import pandas as pd, numpy as np, geopandas as gpd
import os

# GRAPHING IMPORTS
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# import options from options.py
from options import hfs, counties

##TAB STYLES
tabs_styles = {
    'height': '60px',
}
tab_style = {
    'borderBottom': '2px solid #ff0000',
    'borderTop': '2px solid #d6d6d6',
    'justifyContent': 'center',
    'alignItems':'center',
    'fontWeight': 'bold',
    'textAlign': 'center',
    "marginLeft": "auto",
    'fontSize': '17px',
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '5px solid #ff9100',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'fontSize': '20px',
  #  'padding': '6px',
    'textAlign': 'center',
    "marginLeft": "auto"
}


######################################################################################################################
app = dash.Dash(__name__,
                  suppress_callback_exceptions=True,
                  meta_tags=[{"name": "viewport", "content": "width=device-width"}])
server = app.server

os.chdir(os.path.dirname(os.path.abspath(__file__)))

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_THRESHOLD': 200,
})

app.index_string = ''' 
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>SHRP2 Dashboard</title>
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
        <div></div>
    </body>
</html>
'''

# DATA
his = pd.read_csv('data/all_routes.csv')
#si = pd.read_csv('data/intersect.csv')
hr = pd.read_csv('data/speed_file.csv')
#hr15 = pd.read_csv('data/speeds_15_17.csv')
lottr = pd.read_csv('data/lottr.csv', usecols=['LinkDir', 'LOTTR', 'Year'])
streets = gpd.read_file('data/shp/prj.shp')
streets = streets.dropna(subset=['geometry'])
streets.reset_index(drop=True, inplace=True)

# GLOBAL VARIABLES
mapbox_access_token = "pk.eyJ1IjoiZWFuMjQ2IiwiYSI6ImNrbGkwNG9kMTFmdTUybm82ZTBtZXYzbjMifQ.md30QLwkApQpQ_2U3mEO7Q"
embed_link = "https://uky-edu.maps.arcgis.com/apps/webappviewer/index.html?id=9b04f7cf872b492897f77f04a73fbe75"
s = "dark"
rt = list(his.route.unique())
rtpref = list(his.rtprefix.unique())
bar_color = 'rgb(123, 199, 255)'

# APP LAYOUT
app.layout = html.Div(id="mainContainer", style={"display": "flex", "flex-direction": "column", "max-width":"2200px", "margin":"auto"}, children=[
    html.Div(id="output-clientside"),  # empty Div to trigger javascript file for resizing
    html.Div(className="row flex-display", id="header", style={"margin-bottom": "25px"}, children=[
        html.Div(className="one-third column", children=[
            html.Img(
                src=app.get_asset_url("KTC_png.png"),
                id="ktc-image",
            )
        ]),
        html.Div(className="one-half column", id="title", children=[
            html.Div([
                html.H3('SHRP2 DASHBOARD PROTOTYPE', id='toptext', ),
                #html.H5("PROTOTYPE OVERVIEW", id='bottomtext', ),
            ])
        ]),
        html.Div(className="one-third column", id="button", children=[
            html.A(
                html.Button("See Statewide Data", id="learn-more-button"),
                href="https://uky-edu.maps.arcgis.com/apps/opsdashboard/index.html#/762ae87adc5f4d5ca21f02ed5f97c2da",
                target="_blank"
            )
        ]),
    ]),
    html.Div(className="row flex-display", children=[
        html.Div(className="pretty_container four columns", id="cross-filter-options", children=[
            html.P("Select filters below:", className="control_label", style={'font-weight': 'bold'}),
            html.Label("KYTC District:"),
            dcc.Dropdown(id="district",
                         options=[{"label": str(x), "value": x} for x in range(1, 13)],
                         value=None,
                         placeholder="select a district",
                         clearable=True,
                         persistence=False,
                         # persistence_type="memory",
                         className="dcc_control"),

            html.Label("County: "),
            dcc.Dropdown(id="county",
                         options=[{"label": str(x), "value": x} for x in sorted(counties.keys())],
                         value=None,
                         placeholder="select a county",
                         clearable=True,
                         persistence=False,
                         # persistence_type="memory",
                         className="dcc_control"),

            html.Label("Route: "),
            dcc.Dropdown(id="route",
                         options=[{"label": str(x), "value": x} for x in sorted(rt)],
                         value=None,
                         placeholder="select a route",
                         clearable=True,
                         persistence=False,
                         # persistence_type="memory",
                         className="dcc_control"),

            html.P(id="mprange", className="control_label"),

            html.Br(),
            html.Label("MilePoint Range:"),
            dcc.Input(id="bmp",
                      debounce=True,
                      inputMode="numeric",
                      placeholder="begin milepoint",
                      persistence=False,
                      # persistence_type="memory",
                      type="number",
                      className="dcc_control"),

            dcc.Input(id="emp",
                      debounce=True,
                      inputMode="numeric",
                      placeholder="ending milepoint",
                      persistence=False,
                      # persistence_type="memory",
                      type="number",
                      className="dcc_control"),

            html.P(id="selectedRange", className="control_label"),

        ]),

        html.Div(className="eight columns", id="right-column", children=[

            html.Div(className="pretty_container", id="mapboxGraphContainer", children=[
                # dcc.Graph(id="mapbox_plot")
                html.Iframe(id="embed",
                            src=embed_link,
                            width='100%',
                            height='700px')
            ]),
        ]),

    ]),

    #   html.Div(id="hiddenDiv",),

    html.Div(id="graphdiv", children=[
        html.Div(id='tabs_div', children=[dcc.Tabs(id='graph-tabs',
                        value='tab-1',
                        parent_className='custom_tabs',
                        className= 'custom_tabs_container',
                        children=[
            dcc.Tab(label='2018 - 2019 Statistics', value='tab-1', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='2015 - 2019 Statistics', value='tab-2', style=tab_style, selected_style=tab_selected_style),
                ]),
        ]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        dcc.Loading(id='loading',
                            type='circle',
                            fullscreen=True,
                            children=html.Div(id='tabs-content')),
    ]),
])


# HELPER FUNCTIONS
def routes(x):
    f = x.split('-')[1:3]
    return "-".join(f)


def countyfunc(x):
    f = x.split('-')[0]
    return f


def diction(lnk, dx, lg, v):
    dr = dict(zip(list(lnk), list(dx)))
    lt = dict(zip(list(lnk), list(lg)))
    vo = dict(zip(list(lnk), list(v)))
    return dr, lt, vo


def hvol(lnk, v):
    vo = dict(zip(list(lnk), list(v)))
    return vo


streets['route'] = streets['ROUTE_ID'].map(routes)


# define color for mapbox plot speeds
def color(x, df):
    if x < (np.average(df['SPEED_LIMI'], weights=df['SECTION_LE']) / 1.5):
        cl = 'red'
    elif x < (np.average(df['SPEED_LIMI'], weights=df['SECTION_LE']) / 1.2):
        cl = 'gold'
    else:
        cl = 'green'
    return cl


# def zoomlevel()

def generate_street_colormap(streets, style, x, y, z):
    street_colors = ['#006600', '#99ff00', '#ffff33', '#ff9900', '#ff6600']
    layout = go.Layout(
        title_text='Road Heatmap',
        autosize=True,
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(lat=x, lon=y),  # center coordinate dynamic
            pitch=0,
            zoom=z,
            style=style,
        ),
    )

    data = []

    for i in range(len(streets)):
        new_trace = go.Scattermapbox(
            lat=list(streets.loc[i]['geometry'].coords.xy[1]),
            lon=list(streets.loc[i]['geometry'].coords.xy[0]),
            hoverinfo='text',
            mode="markers+lines",
            hovertext='Peak hour speed: {0}   Milepoints: [{1}, {2}]'.format(str(streets.loc[i, 'PeakHrSpee']),
                                                                             str(streets.loc[i, 'BEGIN_POIN']),
                                                                             str(streets.loc[i, 'END_POINT'])),
            marker=go.scattermapbox.Marker(color=color(streets.loc[i, 'PeakHrSpee'], streets),
                                           size=4,
                                           opacity=0.7),
        )
        data.append(new_trace)
    return {"data": data, "layout": layout}


def getkeys(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item in listOfItems:  # get keys based on items
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return listOfKeys


def routequery1819(route, bmp, emp, district=None, county=None):
    if county is not None:
        df = his.loc[(his['ROUTE_ID'].str.contains(route)) &
                     (his['BEGIN_POINT'].astype(float).between(bmp, emp, inclusive=True)) &
                     (his['County'] == county)]
    elif (county is None) and (district is not None):
        df = his.loc[(his['ROUTE_ID'].str.contains(route)) &
                     (his['BEGIN_POINT'].astype(float).between(bmp, emp, inclusive=True)) &
                     (his['District'] == district)]
    else:
        df = his.loc[(his['ROUTE_ID'].str.contains(route)) & (
            his['BEGIN_POINT'].astype(float).between(bmp, emp, inclusive=True))]

    return df


def lendiction(lnk, lg):
    lt = dict(zip(list(lnk), list(lg)))
    return lt


# CALLBACKS

# SCREEN_SIZING CALLBACK
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("mapbox_plot", "figure")],
)


# CALLBACKS FOR USER OPTIONS
@app.callback(
    [Output("county", "options"),
     Output("county", "value"), ],
    Input("district", "value"),
    prevent_initial_call=True
)
def updateCountyOptions(d):
    if d is not None:
        return [{"label": str(x), "value": x} for x in
                sorted(list(his.loc[(his['District'] == d), 'County'].unique()))], None
    else:
        return [{"label": str(x), "value": x} for x in sorted(counties.keys())], None


@app.callback(
    [Output("route", "options"), Output("route", "value"), ],
    [Input("district", "value"), Input("county", "value")],
    prevent_initial_call=True
)
def updateRouteOptions(d, c):
    global r
    if d is None:
        if (c is None):
            r = rt
        elif (c is not None):
            r = list(his.loc[(his['County'] == c), 'route'].unique())

    elif (d is not None):
        if (c is None):
            r = list(his.loc[(his['District'] == d), 'route'].unique())
        elif (c is not None):
            r = list(his.loc[(his['County'] == c), 'route'].unique())

    return [{"label": str(x), "value": x} for x in sorted(r)], None

    # only interstates are continuous over county and district lines. produce an error to show that.


@app.callback(
    Output("mprange", "children"),
    [Input("route", "value"), Input("district", "value"), Input("county", "value")],
    prevent_initial_call=True
)
def showMPRange(r, d, c):  # optimize this later
    if r is None:
        sentence = ''
    elif r is not None:
        if c is not None:
            x = his.loc[(his['County'] == c) & (his['route'] == r), 'BEGIN_POINT'].min()
            y = his.loc[(his['County'] == c) & (his['route'] == r), 'END_POINT'].max()
        elif d is not None:
            x = his.loc[(his['District'] == d) & (his['route'] == r), 'BEGIN_POINT'].min()
            y = his.loc[(his['District'] == d) & (his['route'] == r), 'END_POINT'].max()
        else:
            x = his.loc[(his['route'] == r), 'BEGIN_POINT'].min()
            y = his.loc[(his['route'] == r), 'END_POINT'].max()
        sentence = "Route begins from milepoint {} to {}".format(x, y)

    return sentence


@app.callback(
    [Output("bmp", "min"), Output("bmp", "max"), Output("emp", "min"), Output("emp", "max"), Output("bmp", "value"),
     Output("emp", "value")],
    [Input("route", "value"), Input("district", "value"), Input("county", "value")],
    prevent_initial_call=True
)
def define_mpinput_min_max(r, d, c):  # optimize this later. create a hidden div to share data between callbacks
    if c is not None:
        bmn = his.loc[(his['County'] == c) & (his['route'] == r), 'BEGIN_POINT'].min()
        bmx = his.loc[(his['County'] == c) & (his['route'] == r), 'BEGIN_POINT'].max()
        emn = his.loc[(his['County'] == c) & (his['route'] == r), 'END_POINT'].min()
        emx = his.loc[(his['County'] == c) & (his['route'] == r), 'END_POINT'].max()
    elif d is not None:
        bmn = his.loc[(his['District'] == d) & (his['route'] == r), 'BEGIN_POINT'].min()
        bmx = his.loc[(his['District'] == d) & (his['route'] == r), 'BEGIN_POINT'].max()
        emn = his.loc[(his['District'] == d) & (his['route'] == r), 'END_POINT'].min()
        emx = his.loc[(his['District'] == d) & (his['route'] == r), 'END_POINT'].max()
    else:
        bmn = his.loc[(his['route'] == r), 'BEGIN_POINT'].min()
        bmx = his.loc[(his['route'] == r), 'BEGIN_POINT'].max()
        emn = his.loc[(his['route'] == r), 'END_POINT'].min()
        emx = his.loc[(his['route'] == r), 'END_POINT'].max()
    return bmn, bmx, emn, emx, bmn, emx


# CALLBACKS FOR DATA ANALYSIS AND GRAPH GENERATION.
@app.callback(
    Output("mapboxGraphContainer", "children"),
    [Input("route", "value"),
     Input("bmp", "value"),
     Input("emp", "value"),
     Input("district", "value"),
     Input("county", "value"), ],
    prevent_initial_call=True
)
def showMapboxplot(r, b, e, d, c):
    if r is not None:
        if c is not None:
            dst = streets.loc[(streets['route'].str.contains(r)) &
                              (streets['BEGIN_POIN'].astype(float).between(b, e, inclusive=True)) &
                              (streets['County'] == c)]
        elif (c is None) and (d is not None):
            dst = streets.loc[(streets['route'].str.contains(r)) &
                              (streets['BEGIN_POIN'].astype(float).between(b, e, inclusive=True)) &
                              (streets['District'] == d)]
        else:
            dst = streets.loc[(streets['route'].str.contains(r)) &
                              (streets['BEGIN_POIN'].astype(float).between(b, e, inclusive=True))]

        dst.reset_index(drop=True, inplace=True)
        #length = dst.SECTION_LE.sum()
        length = e - b

        latb = list(dst[dst.routedir == '000'].iloc[0]['geometry'].coords.xy[1])
        lonb = list(dst[dst.routedir == '000'].iloc[0]['geometry'].coords.xy[0])

        late = list(dst[dst.routedir == '000'].iloc[-1]['geometry'].coords.xy[1])
        lone = list(dst[dst.routedir == '000'].iloc[-1]['geometry'].coords.xy[0])

        latb.extend(late)
        lonb.extend(lone)

        x, y = np.mean([max(latb), min(latb)]), np.mean([max(lonb), min(lonb)])  # add zoom here

        z = -1.42 * np.log(length) + 12.987

        d = generate_street_colormap(dst, s, x, y, z)
        fig = go.Figure(
            data=d['data'],
            layout=d['layout'])
        fig.update_layout({'height': 650})
        ret = dcc.Graph(id="mapbox_plot", figure=fig)  # ret=return

    else:
        if c is not None:
            ret = html.Iframe(id="embed", src=f"{embed_link}&find={c}", width='100%', height='700px')
        elif d is not None:
            ret = html.Iframe(id="embed", src=f"{embed_link}&find={d}", width='100%', height='700px')
        else:
            ret = html.Iframe(id="embed", src=embed_link, width='100%', height='700px')
    return ret




##################################################################################
############################         TABS           #######################################
##################################################################################
@app.callback(
    Output("tabs-content", "children"),
    [Input("graph-tabs", 'value'),
     Input("route", "value"),
     Input("bmp", "value"),
     Input("emp", "value"),
     Input("district", "value"),
     Input("county", "value"),
    ],
    prevent_initial_call=True
)
@cache.memoize()
def showHS(tab, r, b, e, d, c):
    if r is not None:
        if tab == 'tab-1':
            df = routequery1819(r, b, e, district=d, county=c)
            c = set(df.loc[df['AllRds_Dir'] == 'Cardinal', 'LinkDir'].unique())
            nc = set(df.loc[df['AllRds_Dir'] != 'Cardinal', 'LinkDir'].unique())
            limit = np.mean(df.loc[df['AllRds_Dir'] == 'Cardinal', 'SPEED_LIMIT_LWA'].unique())
            limitn = np.mean(df.loc[df['AllRds_Dir'] != 'Cardinal', 'SPEED_LIMIT_LWA'].unique())

            # AGGREGATION FUNCS
            wmdc = lambda x: round(np.average(x, weights=dc.loc[x.index, 'Length']), 2)
            lwlt = lambda x: round(np.average(x, weights=lot.loc[x.index, 'Length']), 2)
            nsum = lambda x: round(np.sum(x), 1)

            # CARDINAL
            dc = hr[hr['LinkDir'].isin(c)]  #
            lot = lottr[lottr['LinkDir'].isin(c)]

            drt, leng, adt = diction(df['LinkDir'], df['AllRds_Dir'], df['SECTION_LENGTH'], df['AADT'])
            dc['Length'] = dc['LinkDir'].map(lambda x: leng[x])
            dc['AADT'] = dc['LinkDir'].map(lambda x: adt[x])
            lot['Length'] = lot['LinkDir'].map(lambda x: leng[x])

            dtc = dc.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdc})
            seg_len = np.sum(dc.Length.unique())

            d = dc.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdc, 'AADT': wmdc})
            lotr = lot.groupby('Year', as_index=False).agg({'LOTTR': lwlt})
            lotr = lotr.loc[lotr.Year>2017]

            if 1 in df['F_SYSTEM'].unique():
                try:
                    rf = round(np.average(dc['refspeed'], weights=dc['Length']),
                               2)  # if working check if more than speed limit
                    rfspd = limit if rf > limit else rf
                    d['refspeed'] = rfspd

                except:
                    rfspd = np.mean(df['SPEED_LIMIT_LWA'])
                    d['refspeed'] = np.mean(df['SPEED_LIMIT_LWA'])

            else:
                try:
                    rf = round(np.average(dc.loc[(dc.Hour >= 6) & (dc.Hour < 20), 'refspeed'],
                                          weights=dc.loc[(dc.Hour >= 6) & (dc.Hour < 20), 'Length']),
                               2)
                    rfspd = limit if rf > limit else rf
                    d['refspeed'] = rfspd

                except:
                    rfspd = np.mean(df['SPEED_LIMIT_LWA'])
                    d['refspeed'] = np.mean(df['SPEED_LIMIT_LWA'])

            d['TT'] = seg_len / d['AvgSpeed']
            d['TTref'] = seg_len / d['refspeed']

            d['delay'] = d['TT'] - d['TTref']
            d.loc[(d.delay < 0), 'delay'] = 0

            d['perc'] = d['Hour'].map(lambda x: hfs[x])
            d['Vol'] = d['AADT'] * d['perc']
            d['delT'] = d['delay'] * d['Vol'] * 52 * 5
            dl = d.loc[(d.Hour >= 6) & (d.Hour < 20)].reset_index(drop=True)
            dl = d.groupby(["Hour", "Year"], as_index=False).agg({'delT': nsum})

            dtc['refspeed'] = rfspd
            tdl = dl.groupby('Year', as_index=False).agg({'delT': nsum})

            hrv = pd.DataFrame(data={'Hour': range(0, 24), 'AADT': [np.mean(d.AADT.unique())] * 24, })
            hrv['perc'] = hrv['Hour'].map(lambda x: hfs[x])
            hrv['hvol'] = np.ceil((hrv['AADT'] * hrv['perc']))

            ##############################################################################
            ##############################     HEATMAP     ######################################
            ##############################################################################
            filters = ['BEGIN_POINT', 'END_POINT', 'LinkDir', 'AllRds_Dir']

            ###
            dfc = df.loc[df['AllRds_Dir'] == 'Cardinal', filters].sort_values('BEGIN_POINT')
            dfc['Length'] = dfc['END_POINT'] - dfc['BEGIN_POINT']
            islc = np.sum(dfc['Length'])  # sum of individual segment lengths (isl)
            tslc = max(dfc['END_POINT']) - min(dfc['BEGIN_POINT'])  # total segment length (tdl)

            dfca = dfc.append(dfc.iloc[-1])
            dfca.iloc[-1, 0] = max(dfc['END_POINT'])
            dfca = dfca.append(dfc.iloc[0])
            dfca.iloc[-1, 1] = min(dfc['BEGIN_POINT'])
            dfca['BEGIN_POINT'] = dfca['BEGIN_POINT'].round(3)
            dfca['END_POINT'] = dfca['END_POINT'].round(3)
            dfca.reset_index(drop=True, inplace=True)

            null_mpc = []
            if tslc != islc:
                for ind, row in dfca.iterrows():  # iter over rows
                    if row['END_POINT'] not in np.array(dfca['BEGIN_POINT']):  # if end mp not in bmp
                        null_mpc.append(row['END_POINT'] + 0.001)  # get its value and increment it slightly
                        null_mpc.append(round(dfca.loc[ind + 1, "BEGIN_POINT"] - 0.001, 3))  # get next bmp
                        dfca = dfca.append(dfc.iloc[ind + 1])
                        dfca.iloc[-1, 1] = dfca.iloc[-1, 0]
                        dfca.reset_index(drop=True, inplace=True)

            speedsc = dc[['LinkDir', 'Hour', 'AvgSpeed', 'Year']]
            speedsc = pd.merge(dfca, speedsc, on='LinkDir', how='inner')
            speedsc = speedsc.groupby(['Hour', 'BEGIN_POINT', 'END_POINT'], as_index=False).agg(
                {'AvgSpeed': lambda x: np.mean(x)})

            pvc = pd.pivot_table(speedsc, index=['END_POINT'], columns=[
                'Hour'], values='AvgSpeed')

            for i in null_mpc:
                pvc = pvc.append(pd.Series(name=i))
            pvc = pvc.sort_index()

            tc = []
            for x in range(0, len(pvc.index)):
                tc.append(list(pvc.iloc[x]))

            #####################################################################################################
            # NON-CARDINAL
            # noncardinal hourly speed
            dn = hr[hr['LinkDir'].isin(nc)]  #
            lotn = lottr[lottr['LinkDir'].isin(nc)]
            wmdn = lambda x: round(np.average(x, weights=dn.loc[x.index, 'Length']), 2)
            lwltn = lambda x: round(np.average(x, weights=lotn.loc[x.index, 'Length']), 2)

            dn['Length'] = dn['LinkDir'].map(lambda x: leng[x])
            lotn['Length'] = lotn['LinkDir'].map(lambda x: leng[x])
            dn['AADT'] = dn['LinkDir'].map(lambda x: adt[x])

            dtn = dn.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdn})
            seg_lenn = np.sum(dn.Length.unique())

            dnc = dn.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdn, 'AADT': wmdn})
            lotrn = lotn.groupby('Year', as_index=False).agg({'LOTTR': lwltn})
            lotrn = lotrn.loc[lotrn.Year > 2017]

            if 1 in df['F_SYSTEM'].unique():
                try:
                    rfn = round(np.average(dn['refspeed'], weights=dn['Length']),
                                2)  # if working check if more than speed limit
                    rfspdn = limitn if rfn > limitn else rfn
                    dnc['refspeed'] = rfspdn
                except:
                    rfspdn = np.mean(df['SPEED_LIMIT_LWA'])
                    dnc['refspeed'] = np.mean(df['SPEED_LIMIT_LWA'])
            else:
                try:
                    rfn = round(np.average(dn.loc[(dn.Hour >= 6) & (dn.Hour < 20), 'refspeed'],
                                           weights=dn.loc[(dn.Hour >= 6) & (dn.Hour < 20), 'Length']),
                                2)
                    rfspdn = limitn if rfn > limitn else rfn
                    dnc['refspeed'] = rfspdn
                except:
                    rfspdn = np.mean(df['SPEED_LIMIT_LWA'])
                    dnc['refspeed'] = np.mean(df['SPEED_LIMIT_LWA'])

            dnc['TT'] = seg_lenn / dnc['AvgSpeed']
            dnc['TTref'] = seg_lenn / dnc['refspeed']

            dnc['delay'] = dnc['TT'] - dnc['TTref']
            dnc.loc[(dnc.delay < 0), 'delay'] = 0

            dnc['perc'] = dnc['Hour'].map(lambda x: hfs[x])
            dnc['Vol'] = dnc['AADT'] * dnc['perc']
            dnc['delT'] = dnc['delay'] * dnc['Vol'] * 52 * 5

            dln = dnc.loc[(dnc.Hour >= 6) & (dnc.Hour < 20)].reset_index(drop=True)
            dln = dnc.groupby(["Hour", "Year"], as_index=False).agg({'delT': nsum})

            dtn['refspeed'] = rfspdn

            tdn = dln.groupby('Year', as_index=False).agg({'delT': nsum})

            ############# non-cardinal heatmap ##############################################

            dfnc = df.loc[df['AllRds_Dir'] != 'Cardinal', filters].sort_values('BEGIN_POINT')
            dfnc['Length'] = dfnc['END_POINT'] - dfnc['BEGIN_POINT']
            islnc = np.sum(dfnc['Length'])  # sum of individual segment lengths (isl)
            tslnc = max(dfnc['END_POINT']) - min(dfnc['BEGIN_POINT'])  # total segment length (tdl)

            dfnca = dfnc.append(dfnc.iloc[-1])
            dfnca.iloc[-1, 0] = max(dfnc['END_POINT'])
            dfnca = dfnca.append(dfnc.iloc[0])
            dfnca.iloc[-1, 1] = min(dfnc['BEGIN_POINT'])
            dfnca['BEGIN_POINT'] = dfnca['BEGIN_POINT'].round(3)
            dfnca['END_POINT'] = dfnca['END_POINT'].round(3)
            dfnca.reset_index(drop=True, inplace=True)

            null_mpnc = []
            if tslnc != islnc:
                for ind, row in dfnca.iterrows():  # iter over rows
                    if row['END_POINT'] not in np.array(dfnca['BEGIN_POINT']):  # if end mp not in bmp
                        null_mpnc.append(row['END_POINT'] + 0.001)  # get its value and increment it slightly
                        null_mpnc.append(round(dfnca.loc[ind + 1, "BEGIN_POINT"] - 0.001, 3))  # get next bmp
                        dfnca = dfnca.append(dfnc.iloc[ind + 1])
                        dfnca.iloc[-1, 1] = dfnca.iloc[-1, 0]
                        dfnca.reset_index(drop=True, inplace=True)

            speedsnc = dn[['LinkDir', 'Hour', 'AvgSpeed', 'Year']]
            speedsnc = pd.merge(dfnca, speedsnc, on='LinkDir', how='inner')
            speedsnc = speedsnc.groupby(['Hour', 'BEGIN_POINT', 'END_POINT'], as_index=False).agg(
                {'AvgSpeed': lambda x: np.mean(x)})

            pvnc = pd.pivot_table(speedsnc, index=['END_POINT'], columns=[
                'Hour'], values='AvgSpeed')

            for i in null_mpnc:
                pvnc = pvnc.append(pd.Series(name=i))
            pvnc = pvnc.sort_index()

            tnc = []
            for x in range(0, len(pvnc.index)):
                tnc.append(list(pvnc.iloc[x]))

            # chart y range hr graph
            min_y = dtc['AvgSpeed'].min() - 1 if (dtc['AvgSpeed'].min() < dtn['AvgSpeed'].min()) else dtn[
                                                                                                          'AvgSpeed'].min() - 1
            max_y = dtc['AvgSpeed'].max() + 1 if (dtc['AvgSpeed'].max() > dtn['AvgSpeed'].max()) else dtn[
                                                                                                          'AvgSpeed'].max() + 1

            # chart y range hr delay graph
            min_d = dl['delT'].min() - 1 if (dl['delT'].min() < dln['delT'].min()) else dln['delT'].min() - 1
            max_d = dl['delT'].max() + 1 if (dl['delT'].max() > dln['delT'].max()) else dln['delT'].max() + 1

            # cardinal hourly speed figure
            figc = make_subplots(specs=[[{"secondary_y": True}]])

            figc.add_trace(go.Bar(name='Volume',
                                  x=hrv["Hour"],
                                  y=hrv['hvol'],
                                  marker_color=bar_color,
                                  opacity=0.5, ),
                           secondary_y=False, )

            figc.add_trace(go.Scatter(x=dtc.loc[dtc.Year == 2018, "Hour"],
                                      y=dtc.loc[dtc.Year == 2018, "AvgSpeed"],
                                      mode='lines+markers',
                                      name='2018 Speed',
                                      marker=dict(color='red', ),
                                      line=dict(color='red')),
                           secondary_y=True, )

            figc.add_trace(go.Scatter(x=dtc.loc[dtc.Year == 2019, "Hour"],
                                      y=dtc.loc[dtc.Year == 2019, "AvgSpeed"],
                                      mode='lines+markers',
                                      name='2019 Speed',
                                      marker=dict(color='blue', ),
                                      line=dict(color='blue')),
                           secondary_y=True, )
            figc.add_trace(go.Scatter(x=dtc.loc[dtc.Year == 2019, "Hour"],
                                      y=dtc.loc[dtc.Year == 2019, "refspeed"],
                                      line=dict(color='green', width=2, dash='dash'),
                                      name='Reference Speed'),
                           secondary_y=True, )

            # figc.data = figc.data[::-1]

            figc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Average hourly speeds',
                'hovermode': 'x'
            })

            figc.update_xaxes(title_text="Hour")

            figc.update_yaxes(title_text="Speed(mph)", range=[min_y, max_y], secondary_y=True)
            figc.update_yaxes(title_text="Volume", secondary_y=False)

            # cardinal hourly delay figure
            figdc = go.Figure()
            figdc.add_trace(go.Scatter(x=dl.loc[dl.Year == 2018, "Hour"],
                                       y=dl.loc[dl.Year == 2018, "delT"],
                                       mode='lines+markers',
                                       marker=dict(color='red', ),
                                       line=dict(color='red', ),
                                       name='2018'))
            figdc.add_trace(go.Scatter(x=dl.loc[dl.Year == 2019, "Hour"],
                                       y=dl.loc[dl.Year == 2019, "delT"],
                                       mode='lines+markers',
                                       marker=dict(color='blue', ),
                                       line=dict(color='blue', ),
                                       name='2019'))
            figdc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Average hourly Delay',
                'hovermode': 'x'
            })

            figdc.update_yaxes(title_text="Delay - hr", range=[min_d, max_d])
            figdc.update_xaxes(title_text="Hour")

            # cardinal total year delay figure
            figtdc = go.Figure(data=[
                go.Bar(name='cardinal', x=["2018", "2019"], y=list(tdl['delT']))
            ])

            figtdc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Total Delay by Year',
                'hovermode': 'x'
            })

            figtdc.update_xaxes(title_text="Year")
            figtdc.update_yaxes(title_text="veh-hr")

            # cardinal heatmap figure
            fighmc = go.Figure(data=go.Heatmap(
                x=np.array(pvc.columns),
                y=np.array(pvc.index),
                z=tc,
                zsmooth=False,
                zmin=rfspd // 1.5,
                zmax=rfspd,
                connectgaps=False,
                colorscale='RdYlGn'))

            fighmc.update_layout({'title': "Speed Distribution"})
            fighmc.update_xaxes(title_text = "Hour")
            fighmc.update_yaxes(title_text = "Milepoint")
            fighmc.update_traces(colorbar_title_text="Speed", colorbar_title_side="right")
            #fighmc.update_coloraxes(colorbar_title_text = "Speed", colorbar_title_side = "right")

            # cardinal LOTTR figure
            figltc = go.Figure(data=[
                go.Bar(name='cardinal', x=["2018", "2019"], y=list(lotr['LOTTR']))
            ])

            figltc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'LOTTR by Year',
                'hovermode': 'x'
            })
            figltc.update_xaxes(title_text="Year")
            figltc.update_yaxes(title_text="LOTTR")

            # non-cardinal hourly speed figure
            fignc = make_subplots(specs=[[{"secondary_y": True}]])

            fignc.add_trace(go.Bar(name='Volume',
                                   x=hrv["Hour"],
                                   y=hrv['hvol'],
                                   marker_color=bar_color,
                                   opacity=0.5, ),
                            secondary_y=False, )

            fignc.add_trace(go.Scatter(x=dtn.loc[dtn.Year == 2018, "Hour"],
                                       y=dtn.loc[dtn.Year == 2018, "AvgSpeed"],
                                       mode='lines+markers',
                                       name='2018 Speed',
                                       marker=dict(color='red', ),
                                       line=dict(color='red')),
                            secondary_y=True, ),
            fignc.add_trace(go.Scatter(x=dtn.loc[dtn.Year == 2019, "Hour"],
                                       y=dtn.loc[dtn.Year == 2019, "AvgSpeed"],
                                       mode='lines+markers',
                                       name='2019 Speed',
                                       marker=dict(color='blue'),
                                       line=dict(color='blue')),
                            secondary_y=True, ),
            fignc.add_trace(go.Scatter(x=dtn.loc[dtn.Year == 2019, "Hour"],
                                       y=dtn.loc[dtn.Year == 2019, "refspeed"],
                                       line=dict(color='green', width=2, dash='dash'),
                                       name='Reference Speed'),
                            secondary_y=True, ),
            fignc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Average hourly speeds ',
                'hovermode': 'x'
            })

            fignc.update_xaxes(title_text="Hour of Day")

            fignc.update_yaxes(title_text="Speed (mph)", range=[min_y, max_y], secondary_y=True)
            fignc.update_yaxes(title_text="Volume", secondary_y=False)

            # non-cardinal hourly delay figure
            figdnc = go.Figure()
            figdnc.add_trace(go.Scatter(x=dln.loc[dln.Year == 2018, "Hour"],
                                        y=dln.loc[dln.Year == 2018, "delT"],
                                        mode='lines+markers',
                                        name='2018'))
            figdnc.add_trace(go.Scatter(x=dln.loc[dln.Year == 2019, "Hour"],
                                        y=dln.loc[dln.Year == 2019, "delT"],
                                        mode='lines+markers',
                                        name='2019'))
            figdnc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Average hourly Delay',
                'hovermode': 'x'
            })

            figdnc.update_yaxes(title_text="Delay - hr", range=[min_d, max_d])
            figdnc.update_xaxes(title_text="Hour")

            # non-cardinal total year delay figure
            figtdnc = go.Figure(data=[
                go.Bar(name='noncardinal', x=['2018', '2019'], y=list(tdn['delT']))
            ])
            figtdnc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Total Delay by Year ',
                'hovermode': 'x'
            })
            figtdnc.update_xaxes(title_text="Year")
            figtdnc.update_yaxes(title_text="veh-hr")

            # non-cardinal heatmap figure
            fighmnc = go.Figure(data=go.Heatmap(
                x=np.array(pvnc.columns),
                y=np.array(pvnc.index),
                z=tnc,
                zsmooth=False,
                zmin=rfspdn // 1.5,
                zmax=rfspdn,
                connectgaps=False,
                colorscale='RdYlGn'))

            fighmnc.update_layout({'title': "Speed Distribution "})
            fighmnc.update_xaxes(title_text="Hour")
            fighmnc.update_yaxes(title_text="Milepoint")
            fighmnc.update_traces(colorbar_title_text= "Speed",colorbar_title_side="right")
            #fighmnc.update_coloraxes(colorbar_title_text="Speed")#, colorbar_title_side="right")

            # non-cardinal LOTTR figure
            figltnc = go.Figure(data=[
                go.Bar(name='non-cardinal', x=["2018", "2019"], y=list(lotrn['LOTTR']))
            ])

            figltnc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'LOTTR by Year ',
                'hovermode': 'x'
            })
            figltnc.update_xaxes(title_text="Year")
            figltnc.update_yaxes(title_text="LOTTR")

            graphs_return1819 = html.Div([
                html.Div(className="row flex-display", children=[
                    html.Div([html.H3("Cardinal", id="ctext", )], className="six columns"),
                    html.Div([html.H3("Non-Cardinal", id="nctext", )],
                             className="six columns")
                ]),
                html.Div(className="row flex-display", children=[
                    html.Div([dcc.Graph(id="HrSpdc", figure=figc)],
                             className="pretty_container six columns"),
                    html.Div([dcc.Graph(id="HrSpdnc", figure=fignc)],
                             className="pretty_container six columns")
                ]),
                html.Div(className="row flex-display", children=[
                    html.Div([dcc.Graph(id="DTc", figure=figdc)],
                             className="pretty_container six columns"),
                    html.Div([dcc.Graph(id="DTnc", figure=figdnc)],
                             className="pretty_container six columns")
                ]),
                html.Div(className="row flex-display", children=[
                    html.Div([dcc.Graph(id="Delc", figure=figtdc)],
                             className="pretty_container six columns"),
                    html.Div([dcc.Graph(id="Delnc", figure=figtdnc)],
                             className="pretty_container six columns")
                ]),
                html.Div(className="row flex-display", children=[
                    html.Div([dcc.Graph(id="Hmapc", figure=fighmc)],
                             className="pretty_container six columns"),
                    html.Div([dcc.Graph(id="Hmapnc", figure=fighmnc)],
                             className="pretty_container six columns")
                ]),
                 html.Div(className="row flex-display", children=[
                   html.Div([dcc.Graph(id="Lottrc", figure=figltc)],
                            className="pretty_container six columns"),
                   html.Div([dcc.Graph(id="Lottrnc", figure=figltnc)],
                            className="pretty_container six columns")
                 ])
            ]),
            return graphs_return1819

        elif tab == 'tab-2':

            global si, hr15
            if 'si' not in globals() and 'hr15' not in globals():
                si = pd.read_csv('data/intersect.csv')
                hr15 = pd.read_csv('data/speeds_15_17.csv')

            df = routequery1819(r, b, e, district=d, county=c)

            if c is not None:
                df15 = si.loc[(si['route'].str.contains(r)) &
                              (si['BEGIN_MP_1'].astype(float).between(b, e, inclusive=True)) &
                              (si['County'] == c)]
            elif (c is None) and (d is not None):
                df15 = si.loc[(si['route'].str.contains(r)) &
                              (si['BEGIN_MP_1'].astype(float).between(b, e, inclusive=True)) &
                              (si['District'] == d)]
            else:
                df15 = si.loc[(si['route'].str.contains(r)) &
                              (si['BEGIN_MP_1'].astype(float).between(b, e, inclusive=True))]

            c = set(df.loc[df['AllRds_Dir'] == 'Cardinal', 'LinkDir'].unique())
            c15 = set(df15.loc[df15['AllRds_Dir_1'] == 'Cardinal', 'LIDDir'].unique())
            nc = set(df.loc[df['AllRds_Dir'] != 'Cardinal', 'LinkDir'].unique())
            nc15 = set(df15.loc[df15['AllRds_Dir_1'] != 'Cardinal', 'LIDDir'].unique())
            limit = np.mean(df.loc[df['AllRds_Dir'] == 'Cardinal', 'SPEED_LIMIT_LWA'].unique())
            limitn = np.mean(df.loc[df['AllRds_Dir'] != 'Cardinal', 'SPEED_LIMIT_LWA'].unique())

            # AGGREGATION FUNCS
            wmdc = lambda x: round(np.average(x, weights=dc.loc[x.index, 'Length']), 2)
            wmdc15 = lambda x: round(np.average(x, weights=dc15.loc[x.index, 'Length']), 2)
            lwlt = lambda x: round(np.average(x, weights=lot.loc[x.index, 'Length']), 2)
            nsum = lambda x: round(np.sum(x), 1)

            # CARDINAL
            dc = hr[hr['LinkDir'].isin(c)]  # 18-19
            dc15 = hr15[hr15['LinkDir'].isin(c15)]  # 15-17
            lot = lottr[lottr['LinkDir'].isin(c)]

            drt, leng, adt = diction(df['LinkDir'], df['AllRds_Dir'], df['SECTION_LENGTH'], df['AADT'])
            dc['Length'] = dc['LinkDir'].map(lambda x: leng[x])
            dc['AADT'] = dc['LinkDir'].map(lambda x: adt[x])
            aadt = np.mean(dc['AADT'].unique())  # not being able to get adt for 15 use average for 18/19
            ll = lendiction(df15['LIDDir'], df15['SECTION_LENGTH'])
            dc15['Length'] = dc15['LinkDir'].map(lambda x: ll[x])

            lot['Length'] = lot['LinkDir'].map(lambda x: leng[x])  # check 15-17 too

            dtc = dc.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdc})
            dtc15 = dc15.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdc15})
            seg_len = np.sum(dc.Length.unique())
            dc15['AADT'] = aadt

            d = dc.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdc, 'AADT': wmdc})
            d15 = dc15.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdc15, 'AADT': wmdc15})
            lotr = lot.groupby('Year', as_index=False).agg({'LOTTR': lwlt})

            if 1 in df['F_SYSTEM'].unique():
                try:
                    rf18 = round(np.average(dc['refspeed'], weights=dc['Length']),
                                 2)  # if working check if more than speed limit
                    rf15 = round(np.average(dc15['refspeed'], weights=dc15['Length']),
                                 2)
                    rf = np.average([rf15, rf18])
                    rfspd = limit if rf18 > limit else rf18
                    rfspd_all = limit if rf > limit else rf
                    d['refspeed'] = rfspd
                    d15['refspeed'] = rfspd_all
                except:
                    rfspd, rfspd_all = np.mean(df['SPEED_LIMIT_LWA']), np.mean(df['SPEED_LIMIT_LWA'])
                    d['refspeed'] = np.mean(df['SPEED_LIMIT_LWA'])
                    d15['refspeed'] = np.mean(df['SPEED_LIMIT_LWA'])
            else:
                try:
                    rf18 = round(np.average(dc.loc[(dc.Hour >= 6) & (dc.Hour < 20), 'refspeed'],
                                            weights=dc.loc[(dc.Hour >= 6) & (dc.Hour < 20), 'Length']), 2)
                    rf15 = round(np.average(dc15.loc[(dc15.Hour >= 6) & (dc15.Hour < 20), 'refspeed'],
                                            weights=dc15.loc[(dc15.Hour >= 6) & (dc15.Hour < 20), 'Length']), 2)
                    rf = np.average([rf15, rf18])
                    rfspd = limit if rf18 > limit else rf18
                    rfspd_all = limit if rf > limit else rf
                    d['refspeed'] = rfspd
                    d15['refspeed'] = rfspd_all
                except:
                    rfspd, rfspd_all = np.mean(df['SPEED_LIMIT_LWA']), np.mean(df['SPEED_LIMIT_LWA'])
                    d['refspeed'] = np.mean(df['SPEED_LIMIT_LWA'])
                    d15['refspeed'] = np.mean(df['SPEED_LIMIT_LWA'])

            d['TT'] = seg_len / d['AvgSpeed']
            d15['TT'] = seg_len / d15['AvgSpeed']
            d['TTref'] = seg_len / d['refspeed']
            d15['TTref'] = seg_len / d15['refspeed']
            d['delay'] = d['TT'] - d['TTref']
            d15['delay'] = d15['TT'] - d15['TTref']
            d.loc[(d.delay < 0), 'delay'] = 0
            d15.loc[(d15.delay < 0), 'delay'] = 0

            d['perc'] = d['Hour'].map(lambda x: hfs[x])
            d15['perc'] = d15['Hour'].map(lambda x: hfs[x])
            d['Vol'] = d['AADT'] * d['perc']
            d15['Vol'] = d15['AADT'] * d15['perc']
            d['delT'] = d['delay'] * d['Vol'] * 52 * 5
            d15['delT'] = d15['delay'] * d15['Vol'] * 52 * 5
            dl = d.loc[(d.Hour >= 6) & (d.Hour < 20)].reset_index(drop=True)
            dl15 = d15.loc[(d15.Hour >= 6) & (d15.Hour < 20)].reset_index(drop=True)
            dl = d.groupby(["Hour", "Year"], as_index=False).agg({'delT': nsum})  # 18-19
            dl15 = d15.groupby(["Hour", "Year"], as_index=False).agg({'delT': nsum})  # 15-17

            dtc['refspeed'] = rfspd
            dtc15['refspeed'] = rfspd_all
            tdl = dl.groupby('Year', as_index=False).agg({'delT': nsum})
            tdl15 = dl15.groupby('Year', as_index=False).agg({'delT': nsum})  # total delay
            tdelay = pd.concat([tdl15, tdl], ignore_index=True)  # all
            hrv = pd.DataFrame(data={'Hour': range(0, 24), 'AADT': [np.mean(d.AADT.unique())] * 24, })
            hrv15 = pd.DataFrame(data={'Hour': range(0, 24), 'AADT': [np.mean(d15.AADT.unique())] * 24, })
            hrv['perc'] = hrv['Hour'].map(lambda x: hfs[x])
            hrv15['perc'] = hrv15['Hour'].map(lambda x: hfs[x])
            hrv['hvol'] = np.ceil((hrv['AADT'] * hrv['perc']))  # hourly volume

            #####################################################################
            ############################## HEATMAP ###############################

            filters = ['BEGIN_POINT', 'END_POINT', 'LinkDir', 'AllRds_Dir']
            filters15 = ['BEGIN_MP_1', 'END_MP_1', 'LinkDir', 'AllRds_Dir_1']

            ###
            dfc = df.loc[df['AllRds_Dir'] == 'Cardinal', filters].sort_values('BEGIN_POINT')
            dfc15 = df15.loc[df15['AllRds_Dir_1'] == 'Cardinal', filters15].sort_values('BEGIN_MP_1')
            dfc['Length'] = dfc['END_POINT'] - dfc['BEGIN_POINT']
            dfc15['Length'] = dfc15['END_MP_1'] - dfc15['BEGIN_MP_1']
            islc = np.sum(dfc['Length'])  # sum of individual segment lengths (isl)
            tslc = max(dfc['END_POINT']) - min(dfc['BEGIN_POINT'])  # total segment length (tdl)

            dfca = dfc.append(dfc.iloc[-1])
            dfca.iloc[-1, 0] = max(dfc['END_POINT'])
            dfca = dfca.append(dfc.iloc[0])
            dfca.iloc[-1, 1] = min(dfc['BEGIN_POINT'])
            dfca['BEGIN_POINT'] = dfca['BEGIN_POINT'].round(3)
            dfca['END_POINT'] = dfca['END_POINT'].round(3)
            dfca.reset_index(drop=True, inplace=True)

            null_mpc = []
            if tslc != islc:
                for ind, row in dfca.iterrows():  # iter over row
                    if row['END_POINT'] not in np.array(dfca['BEGIN_POINT']):  # if end mp not in bmp
                        null_mpc.append(row['END_POINT'] + 0.001)  # get its value and increment it slightly
                        null_mpc.append(round(dfca.loc[ind + 1, "BEGIN_POINT"] - 0.001, 3))  # get next bmp
                        dfca = dfca.append(dfc.iloc[ind + 1])
                        dfca.iloc[-1, 1] = dfca.iloc[-1, 0]
                        dfca.reset_index(drop=True, inplace=True)

            speedsc = dc[['LinkDir', 'Hour', 'AvgSpeed', 'Year']]  # heatmap 18/19
            speedsc15 = dc15[['LinkDir', 'Hour', 'AvgSpeed', 'Year']]  # 15/17
            speedsc_t = pd.concat([speedsc15, speedsc], ignore_index=True)  # all
            speedsc = pd.merge(dfca, speedsc, on='LinkDir', how='inner')
            speedsc_t = pd.merge(dfca, speedsc_t, on='LinkDir', how='inner')
            speedsc = speedsc.groupby(['Hour', 'BEGIN_POINT', 'END_POINT'], as_index=False).agg(
                {'AvgSpeed': lambda x: np.mean(x)})
            speedsc_t = speedsc_t.groupby(['Hour', 'BEGIN_POINT', 'END_POINT'], as_index=False).agg(
                {'AvgSpeed': lambda x: np.mean(x)})

            pvc = pd.pivot_table(speedsc, index=['END_POINT'], columns=['Hour'], values='AvgSpeed')  # 18/19
            pvc_t = pd.pivot_table(speedsc_t, index=['END_POINT'], columns=['Hour'], values='AvgSpeed')  # all

            for i in null_mpc:
                pvc = pvc.append(pd.Series(name=i))
            pvc = pvc.sort_index()
            tc = []
            for x in range(0, len(pvc.index)):
                tc.append(list(pvc.iloc[x]))

            for i in null_mpc:
                pvc_t = pvc_t.append(pd.Series(name=i))
            pvc_t = pvc_t.sort_index()
            tc_t = []
            for x in range(0, len(pvc_t.index)):
                tc_t.append(list(pvc_t.iloc[x]))  # tc_t=15/19, tc=18/19

            ##########################################################################################
            # NON-CARDINAL

            # noncardinal hourly speed
            dn = hr[hr['LinkDir'].isin(nc)]  #
            dn15 = hr15[hr15['LinkDir'].isin(nc15)]

            lotn = lottr[lottr['LinkDir'].isin(nc)]

            wmdn = lambda x: round(np.average(x, weights=dn.loc[x.index, 'Length']), 2)
            wmdn15 = lambda x: round(np.average(x, weights=dn15.loc[x.index, 'Length']), 2)

            lwltn = lambda x: round(np.average(x, weights=lotn.loc[x.index, 'Length']), 2)

            dn['Length'] = dn['LinkDir'].map(lambda x: leng[x])
            lotn['Length'] = lotn['LinkDir'].map(lambda x: leng[x])
            dn['AADT'] = dn['LinkDir'].map(lambda x: adt[x])
            aadtn = dn['AADT'].unique().mean()  # not being able to get adt for 15 use average for 18/19

            dn15['Length'] = dn15['LinkDir'].map(lambda x: ll[x])

            dtn = dn.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdn})
            dtn15 = dn15.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdn15})
            seg_lenn = np.sum(dn.Length.unique())
            dn15['AADT'] = aadtn

            dnc = dn.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdn, 'AADT': wmdn})
            dnc15 = dn15.groupby(['Hour', 'Year'], as_index=False).agg({'AvgSpeed': wmdn15, 'AADT': wmdn15})

            lotrn = lotn.groupby('Year', as_index=False).agg({'LOTTR': lwltn})

            if 1 in df['F_SYSTEM'].unique():
                try:
                    rf18n = round(np.average(dn['refspeed'], weights=dn['Length']),
                                  2)  # if working check if more than speed limit
                    rf15n = round(np.average(dn15['refspeed'], weights=dn15['Length']),
                                  2)
                    rfn = np.average([rf15n, rf18n])
                    rfspdn = limitn if rf18n > limitn else rf18n
                    rfspd_alln = limitn if rfn > limitn else rfn
                    dnc['refspeed'] = rfspdn
                    dnc15['refspeed'] = rfspd_alln
                except:
                    dnc['refspeed'] = np.mean(df['SPEED_LIMIT_LWA'])
                    dnc15['refspeed'] = np.mean(df['SPEED_LIMIT_LWA'])
                    rfspdn, rfspd_alln = np.mean(df['SPEED_LIMIT_LWA']), np.mean(df['SPEED_LIMIT_LWA'])
            else:
                try:
                    rf18n = round(np.average(dn.loc[(dn.Hour >= 6) & (dn.Hour < 20), 'refspeed'],
                                             weights=dn.loc[(dn.Hour >= 6) & (dn.Hour < 20), 'Length']), 2)
                    rf15n = round(np.average(dn15.loc[(dn15.Hour >= 6) & (dn15.Hour < 20), 'refspeed'],
                                             weights=dn15.loc[(dn15.Hour >= 6) & (dn15.Hour < 20), 'Length']), 2)
                    rfn = np.average([rf15n, rf18n])
                    rfspdn = limitn if rf18n > limitn else rf18n
                    rfspd_alln = limitn if rfn > limitn else rfn
                    dnc['refspeed'] = rfspdn
                    dnc15['refspeed'] = rfspd_alln
                except:
                    rfspdn, rfspd_alln = np.mean(df['SPEED_LIMIT_LWA']), np.mean(df['SPEED_LIMIT_LWA'])
                    dnc['refspeed'] = np.mean(df['SPEED_LIMIT_LWA'])
                    dnc15['refspeed'] = np.mean(df['SPEED_LIMIT_LWA'])

            dnc['TT'] = seg_lenn / dnc['AvgSpeed']
            dnc15['TT'] = seg_lenn / dnc15['AvgSpeed']
            dnc['TTref'] = seg_lenn / dnc['refspeed']
            dnc15['TTref'] = seg_lenn / dnc15['refspeed']
            dnc['delay'] = dnc['TT'] - dnc['TTref']
            dnc15['delay'] = dnc15['TT'] - dnc15['TTref']
            dnc.loc[(dnc.delay < 0), 'delay'] = 0
            dnc15.loc[(dnc15.delay < 0), 'delay'] = 0

            dnc['perc'] = dnc['Hour'].map(lambda x: hfs[x])
            dnc15['perc'] = dnc15['Hour'].map(lambda x: hfs[x])
            dnc['Vol'] = dnc['AADT'] * dnc['perc']
            dnc15['Vol'] = dnc15['AADT'] * dnc15['perc']
            dnc['delT'] = dnc['delay'] * dnc['Vol'] * 52 * 5
            dnc15['delT'] = dnc15['delay'] * dnc15['Vol'] * 52 * 5

            dln = dnc.loc[(dnc.Hour >= 6) & (dnc.Hour < 20)].reset_index(drop=True)
            dln15 = dnc15.loc[(dnc15.Hour >= 6) & (dnc15.Hour < 20)].reset_index(drop=True)
            dln = dnc.groupby(["Hour", "Year"], as_index=False).agg({'delT': nsum})  # 18-19
            dln15 = dnc15.groupby(["Hour", "Year"], as_index=False).agg({'delT': nsum})  # 15-17

            dtn['refspeed'] = rfspdn
            dtn15['refspeed'] = rfspd_alln

            tdn = dln.groupby('Year', as_index=False).agg({'delT': nsum})
            tdn15 = dln15.groupby('Year', as_index=False).agg({'delT': nsum})  # total delay
            tdelayn = pd.concat([tdn15, tdn], ignore_index=True)  # all

            ############# non-cardinal heatmap ##############################################

            dfnc = df.loc[df['AllRds_Dir'] != 'Cardinal', filters].sort_values('BEGIN_POINT')
            dfnc15 = df15.loc[df15['AllRds_Dir_1'] != 'Cardinal', filters15].sort_values('BEGIN_MP_1')
            dfnc['Length'] = dfnc['END_POINT'] - dfnc['BEGIN_POINT']
            dfnc15['Length'] = dfnc15['END_MP_1'] - dfnc15['BEGIN_MP_1']
            islnc = np.sum(dfnc['Length'])  # sum of individual segment lengths (isl)
            # islnc15 = np.sum(dfnc15['Length'])
            tslnc = max(dfnc['END_POINT']) - min(dfnc['BEGIN_POINT'])  # total segment length (tdl)
            # tslnc15 = max(dfnc15['END_MP_1']) - min(dfnc15['BEGIN_MP_1'])

            dfnca = dfnc.append(dfnc.iloc[-1])
            # dfnca15 = dfnc15.append(dfnc15.iloc[-1])
            dfnca.iloc[-1, 0] = max(dfnc['END_POINT'])
            # dfnca15.iloc[-1, 0] = max(dfnc15['END_MP_1'])
            dfnca = dfnca.append(dfnc.iloc[0])
            # dfnca15 = dfnca15.append(dfnc15.iloc[0])
            dfnca.iloc[-1, 1] = min(dfnc['BEGIN_POINT'])
            # dfnca15.iloc[-1, 1] = min(dfnc15['BEGIN_MP_1'])
            dfnca['BEGIN_POINT'] = dfnca['BEGIN_POINT'].round(3)
            # dfnca15['BEGIN_MP_1'] = dfnca15['BEGIN_MP_1'].round(3)
            dfnca['END_POINT'] = dfnca['END_POINT'].round(3)
            # dfnca15['END_MP_1'] = dfnca15['END_MP_1'].round(3)
            dfnca.reset_index(drop=True, inplace=True)
            # dfnca15.reset_index(drop=True, inplace=True)

            null_mpnc = []
            if tslnc != islnc:
                for ind, row in dfnca.iterrows():  # iter over rows
                    if row['END_POINT'] not in np.array(dfnca['BEGIN_POINT']):  # if end mp not in bmp
                        null_mpnc.append(row['END_POINT'] + 0.001)  # get its value and increment it slightly
                        null_mpnc.append(round(dfnca.loc[ind + 1, "BEGIN_POINT"] - 0.001, 3))  # get next bmp
                        dfnca = dfnca.append(dfnc.iloc[ind + 1])
                        dfnca.iloc[-1, 1] = dfnca.iloc[-1, 0]
                        dfnca.reset_index(drop=True, inplace=True)

            speedsnc = dn[['LinkDir', 'Hour', 'AvgSpeed', 'Year']]  # heatmap 18/19
            speedsnc15 = dn15[['LinkDir', 'Hour', 'AvgSpeed', 'Year']]  # 15/17
            speedsnc_t = pd.concat([speedsnc15, speedsnc], ignore_index=True)  # all
            speedsnc = pd.merge(dfnca, speedsnc, on='LinkDir', how='inner')
            speedsnc_t = pd.merge(dfnca, speedsnc_t, on='LinkDir', how='inner')
            speedsnc = speedsnc.groupby(['Hour', 'BEGIN_POINT', 'END_POINT'], as_index=False).agg(
                {'AvgSpeed': lambda x: np.mean(x)})
            speedsnc_t = speedsnc_t.groupby(['Hour', 'BEGIN_POINT', 'END_POINT'], as_index=False).agg(
                {'AvgSpeed': lambda x: np.mean(x)})

            pvnc = pd.pivot_table(speedsnc, index=['END_POINT'], columns=['Hour'], values='AvgSpeed')  # 18/19
            pvnc_t = pd.pivot_table(speedsnc_t, index=['END_POINT'], columns=['Hour'], values='AvgSpeed')  # all

            for i in null_mpnc:
                pvnc = pvnc.append(pd.Series(name=i))
            pvnc = pvnc.sort_index()

            tnc = []
            for x in range(0, len(pvnc.index)):
                tnc.append(list(pvnc.iloc[x]))

            for i in null_mpnc:
                pvnc_t = pvnc_t.append(pd.Series(name=i))
            pvnc_t = pvnc_t.sort_index()
            tnc_t = []
            for x in range(0, len(pvnc_t.index)):
                tnc_t.append(list(pvnc_t.iloc[x]))  # tc_t=15/19, tc=18/19

            # chart y range hr graph
            min_y = min(dtc['AvgSpeed'].min(), dtn['AvgSpeed'].min(), dtc15['AvgSpeed'].min(),
                        dtn15['AvgSpeed'].min()) - 1
            max_y = max(dtc['AvgSpeed'].max(), dtn['AvgSpeed'].max(), dtc15['AvgSpeed'].max(),
                        dtn15['AvgSpeed'].max()) + 1

            min_d = min(dl['delT'].min(), dln['delT'].min(), dl15['delT'].min(), dln15['delT'].min()) - 1
            max_d = max(dl['delT'].max(), dln['delT'].max(), dl15['delT'].max(), dln15['delT'].max()) + 1

            # cardinal hourly speed figure
            figc = make_subplots(specs=[[{"secondary_y": True}]])

            figc.add_trace(go.Bar(name='Volume',
                                  x=hrv["Hour"],
                                  y=hrv['hvol'],
                                  marker_color=bar_color,
                                  opacity=0.5, ),
                           secondary_y=False, )

            figc.add_trace(go.Scatter(x=dtc.loc[dtc.Year == 2018, "Hour"],
                                      y=dtc.loc[dtc.Year == 2018, "AvgSpeed"],
                                      mode='lines+markers',
                                      name='2018 Speed',
                                      marker=dict(color='red', ),
                                      line=dict(color='red')),
                           secondary_y=True, )

            figc.add_trace(go.Scatter(x=dtc.loc[dtc.Year == 2019, "Hour"],
                                      y=dtc.loc[dtc.Year == 2019, "AvgSpeed"],
                                      mode='lines+markers',
                                      name='2019 Speed',
                                      marker=dict(color='blue', ),
                                      line=dict(color='blue')),
                           secondary_y=True, )
            figc.add_trace(go.Scatter(x=dtc.loc[dtc.Year == 2019, "Hour"],
                                      y=dtc.loc[dtc.Year == 2019, "refspeed"],
                                      line=dict(color='green', width=2, dash='dash'),
                                      name='Reference Speed'),
                           secondary_y=True, )

            # figc.data = figc.data[::-1]

            figc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Average hourly speeds',
                'hovermode': 'x'
            })

            figc.update_xaxes(title_text="Hour")

            figc.update_yaxes(title_text="Speed(mph)", range=[min_y, max_y], secondary_y=True)
            figc.update_yaxes(title_text="Volume", secondary_y=False)

            # all
            figct = make_subplots(specs=[[{"secondary_y": True}]])

            figct.add_trace(go.Bar(name='Volume',
                                   x=hrv["Hour"],
                                   y=hrv['hvol'],
                                   marker_color=bar_color,
                                   opacity=0.5, ),
                            secondary_y=False, )

            figct.add_trace(go.Scatter(x=dtc15.loc[dtc15.Year == 2015, "Hour"],
                                       y=dtc15.loc[dtc15.Year == 2015, "AvgSpeed"],
                                       mode='lines+markers',
                                       name='2015 Speed',
                                       marker=dict(color='#FFA15A', ),
                                       line=dict(color='#FFA15A')),
                            secondary_y=True, )

            figct.add_trace(go.Scatter(x=dtc15.loc[dtc15.Year == 2016, "Hour"],
                                       y=dtc15.loc[dtc15.Year == 2016, "AvgSpeed"],
                                       mode='lines+markers',
                                       name='2016 Speed',
                                       marker=dict(color='#19D3F3', ),
                                       line=dict(color='#19D3F3')),
                            secondary_y=True, )

            figct.add_trace(go.Scatter(x=dtc15.loc[dtc15.Year == 2017, "Hour"],
                                       y=dtc15.loc[dtc15.Year == 2017, "AvgSpeed"],
                                       mode='lines+markers',
                                       name='2017 Speed',
                                       marker=dict(color='#EF553B', ),
                                       line=dict(color='#EF553B')),
                            secondary_y=True, )

            figct.add_trace(go.Scatter(x=dtc.loc[dtc.Year == 2018, "Hour"],
                                       y=dtc.loc[dtc.Year == 2018, "AvgSpeed"],
                                       mode='lines+markers',
                                       name='2018 Speed',
                                       marker=dict(color='red', ),
                                       line=dict(color='red')),
                            secondary_y=True, )

            figct.add_trace(go.Scatter(x=dtc.loc[dtc.Year == 2019, "Hour"],
                                       y=dtc.loc[dtc.Year == 2019, "AvgSpeed"],
                                       mode='lines+markers',
                                       name='2019 Speed',
                                       marker=dict(color='blue', ),
                                       line=dict(color='blue')),
                            secondary_y=True, )
            figct.add_trace(go.Scatter(x=dtc15.loc[dtc15.Year == 2016, "Hour"],
                                       y=dtc15.loc[dtc15.Year == 2016, "refspeed"],
                                       line=dict(color='green', width=2, dash='dash'),
                                       name='Reference Speed'),
                            secondary_y=True, )

            # figc.data = figc.data[::-1]

            figct.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Average hourly speeds',
                'hovermode': 'x'
            })

            figct.update_xaxes(title_text="Hour")

            figct.update_yaxes(title_text="Speed(mph)", range=[min_y, max_y], secondary_y=True)
            figct.update_yaxes(title_text="Volume", secondary_y=False)

            # cardinal hourly delay figure
            figdc = go.Figure()
            figdc.add_trace(go.Scatter(x=dl.loc[dl.Year == 2018, "Hour"],
                                       y=dl.loc[dl.Year == 2018, "delT"],
                                       mode='lines+markers',
                                       marker=dict(color='red', ),
                                       line=dict(color='red', ),
                                       name='2018'))
            figdc.add_trace(go.Scatter(x=dl.loc[dl.Year == 2019, "Hour"],
                                       y=dl.loc[dl.Year == 2019, "delT"],
                                       mode='lines+markers',
                                       marker=dict(color='blue', ),
                                       line=dict(color='blue', ),
                                       name='2019'))
            figdc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Average hourly Delay',
                'hovermode': 'x'
            })

            figdc.update_yaxes(title_text="Delay - hr", range=[min_d, max_d])
            figdc.update_xaxes(title_text="Hour")

            # cardinal hourly delay figure
            figdct = go.Figure()

            figdct.add_trace(go.Scatter(x=dl15.loc[dl15.Year == 2015, "Hour"],
                                        y=dl15.loc[dl15.Year == 2015, "delT"],
                                        mode='lines+markers',
                                        marker=dict(color='#FFA15A', ),
                                        line=dict(color='#FFA15A', ),
                                        name='2015'))

            figdct.add_trace(go.Scatter(x=dl15.loc[dl15.Year == 2016, "Hour"],
                                        y=dl15.loc[dl15.Year == 2016, "delT"],
                                        mode='lines+markers',
                                        marker=dict(color='#19D3F3', ),
                                        line=dict(color='#19D3F3', ),
                                        name='2016'))

            figdct.add_trace(go.Scatter(x=dl15.loc[dl15.Year == 2017, "Hour"],
                                        y=dl15.loc[dl15.Year == 2017, "delT"],
                                        mode='lines+markers',
                                        marker=dict(color='#EF553B', ),
                                        line=dict(color='#EF553B', ),
                                        name='2017'))

            figdct.add_trace(go.Scatter(x=dl.loc[dl.Year == 2018, "Hour"],
                                        y=dl.loc[dl.Year == 2018, "delT"],
                                        mode='lines+markers',
                                        marker=dict(color='red', ),
                                        line=dict(color='red', ),
                                        name='2018'))
            figdct.add_trace(go.Scatter(x=dl.loc[dl.Year == 2019, "Hour"],
                                        y=dl.loc[dl.Year == 2019, "delT"],
                                        mode='lines+markers',
                                        marker=dict(color='blue', ),
                                        line=dict(color='blue', ),
                                        name='2019'))
            figdct.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Average hourly Delay',
                'hovermode': 'x'
            })

            figdct.update_yaxes(title_text="Delay - hr", range=[min_d, max_d])
            figdct.update_xaxes(title_text="Hour")

            # cardinal total year delay figure
            figtdc = go.Figure(data=[
                go.Bar(name='cardinal', x=["2018", "2019"], y=list(tdl['delT']))
            ])

            figtdc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Total Delay by Year ',
                'hovermode': 'x'
            })

            figtdc.update_xaxes(title_text="Year")
            figtdc.update_yaxes(title_text="veh-hr")

            figtdct = go.Figure(data=[
                go.Bar(name='cardinal', x=["2015", "2016", "2017", "2018", "2019"], y=list(tdelay['delT']))
            ])

            figtdct.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Total Delay by Year',
                'hovermode': 'x'
            })

            figtdct.update_xaxes(title_text="Year")
            figtdct.update_yaxes(title_text="veh-hr")

            # cardinal heatmap figure
            fighmc = go.Figure(data=go.Heatmap(
                x=np.array(pvc.columns),
                y=np.array(pvc.index),
                z=tc,
                zsmooth=False,
                zmin=rfspd // 1.5,
                zmax=rfspd,
                connectgaps=False,
                colorscale='RdYlGn'))

            fighmc.update_layout({'title': "Speed Distribution Cardinal"})

            fighmct = go.Figure(data=go.Heatmap(
                x=np.array(pvc_t.columns),
                y=np.array(pvc_t.index),
                z=tc_t,
                zsmooth=False,
                zmin=rfspd_all // 1.5,
                zmax=rfspd_all,
                connectgaps=False,
                colorscale='RdYlGn'))

            fighmct.update_layout({'title': "Speed Distribution"})

            # cardinal LOTTR figure
            figltc = go.Figure(data=[
                go.Bar(name='cardinal', x=["2018", "2019"], y=list(lotr['LOTTR']))
            ])

            figltc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'LOTTR by Year ',
                'hovermode': 'x'
            })
            figltc.update_xaxes(title_text="Year")
            figltc.update_yaxes(title_text="LOTTR")

            figltct = go.Figure(data=[
                go.Bar(name='cardinal', x=["2015", "2016", "2017", "2018", "2019"], y=list(lotr['LOTTR']))
            ])

            figltct.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'LOTTR by Year ',
                'hovermode': 'x'
            })
            figltct.update_xaxes(title_text="Year")
            figltct.update_yaxes(title_text="LOTTR")

            # non-cardinal hourly speed figure
            fignc = make_subplots(specs=[[{"secondary_y": True}]])

            fignc.add_trace(go.Bar(name='Volume',
                                   x=hrv["Hour"],
                                   y=hrv['hvol'],
                                   marker_color=bar_color,
                                   opacity=0.5, ),
                            secondary_y=False, )

            fignc.add_trace(go.Scatter(x=dtn.loc[dtn.Year == 2018, "Hour"],
                                       y=dtn.loc[dtn.Year == 2018, "AvgSpeed"],
                                       mode='lines+markers',
                                       name='2018 Speed',
                                       marker=dict(color='red', ),
                                       line=dict(color='red')),
                            secondary_y=True, ),
            fignc.add_trace(go.Scatter(x=dtn.loc[dtn.Year == 2019, "Hour"],
                                       y=dtn.loc[dtn.Year == 2019, "AvgSpeed"],
                                       mode='lines+markers',
                                       name='2019 Speed',
                                       marker=dict(color='blue'),
                                       line=dict(color='blue')),
                            secondary_y=True, ),
            fignc.add_trace(go.Scatter(x=dtn.loc[dtn.Year == 2019, "Hour"],
                                       y=dtn.loc[dtn.Year == 2019, "refspeed"],
                                       line=dict(color='green', width=2, dash='dash'),
                                       name='Reference Speed'),
                            secondary_y=True, ),
            fignc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Average hourly speeds',
                'hovermode': 'x'
            })

            fignc.update_xaxes(title_text="Hour of Day")

            fignc.update_yaxes(title_text="Speed (mph)", range=[min_y, max_y], secondary_y=True)
            fignc.update_yaxes(title_text="Volume", secondary_y=False)

            fignct = make_subplots(specs=[[{"secondary_y": True}]])

            fignct.add_trace(go.Bar(name='Volume',
                                    x=hrv["Hour"],
                                    y=hrv['hvol'],
                                    marker_color=bar_color,
                                    opacity=0.5, ),
                             secondary_y=False, )

            fignct.add_trace(go.Scatter(x=dtn15.loc[dtn15.Year == 2015, "Hour"],
                                        y=dtn15.loc[dtn15.Year == 2015, "AvgSpeed"],
                                        mode='lines+markers',
                                        name='2015 Speed',
                                        marker=dict(color='#FFA15A', ),
                                        line=dict(color='#FFA15A')),
                             secondary_y=True, )

            fignct.add_trace(go.Scatter(x=dtn15.loc[dtn15.Year == 2016, "Hour"],
                                        y=dtn15.loc[dtn15.Year == 2016, "AvgSpeed"],
                                        mode='lines+markers',
                                        name='2016 Speed',
                                        marker=dict(color='#19D3F3', ),
                                        line=dict(color='#19D3F3')),
                             secondary_y=True, )

            fignct.add_trace(go.Scatter(x=dtn15.loc[dtn15.Year == 2017, "Hour"],
                                        y=dtn15.loc[dtn15.Year == 2017, "AvgSpeed"],
                                        mode='lines+markers',
                                        name='2017 Speed',
                                        marker=dict(color='#EF553B', ),
                                        line=dict(color='#EF553B')),
                             secondary_y=True, )

            fignct.add_trace(go.Scatter(x=dtn.loc[dtn.Year == 2018, "Hour"],
                                        y=dtn.loc[dtn.Year == 2018, "AvgSpeed"],
                                        mode='lines+markers',
                                        name='2018 Speed',
                                        marker=dict(color='red', ),
                                        line=dict(color='red')),
                             secondary_y=True, )

            fignct.add_trace(go.Scatter(x=dtn.loc[dtn.Year == 2019, "Hour"],
                                        y=dtn.loc[dtn.Year == 2019, "AvgSpeed"],
                                        mode='lines+markers',
                                        name='2019 Speed',
                                        marker=dict(color='blue', ),
                                        line=dict(color='blue')),
                             secondary_y=True, )
            fignct.add_trace(go.Scatter(x=dtn15.loc[dtn15.Year == 2016, "Hour"],
                                        y=dtn15.loc[dtn15.Year == 2016, "refspeed"],
                                        line=dict(color='green', width=2, dash='dash'),
                                        name='Reference Speed'),
                             secondary_y=True, )

            fignct.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Average hourly speeds',
                'hovermode': 'x'
            })

            fignct.update_xaxes(title_text="Hour")

            fignct.update_yaxes(title_text="Speed(mph)", range=[min_y, max_y], secondary_y=True)
            fignct.update_yaxes(title_text="Volume", secondary_y=False)

            # non-cardinal hourly delay figure
            figdnc = go.Figure()
            figdnc.add_trace(go.Scatter(x=dln.loc[dln.Year == 2018, "Hour"],
                                        y=dln.loc[dln.Year == 2018, "delT"],
                                        mode='lines+markers',
                                        name='2018'))
            figdnc.add_trace(go.Scatter(x=dln.loc[dln.Year == 2019, "Hour"],
                                        y=dln.loc[dln.Year == 2019, "delT"],
                                        mode='lines+markers',
                                        name='2019'))
            figdnc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Average hourly Delay ',
                'hovermode': 'x'
            })

            figdnc.update_yaxes(title_text="Delay - hr", range=[min_d, max_d])
            figdnc.update_xaxes(title_text="Hour")

            figdnct = go.Figure()

            figdnct.add_trace(go.Scatter(x=dln15.loc[dln15.Year == 2015, "Hour"],
                                         y=dln15.loc[dln15.Year == 2015, "delT"],
                                         mode='lines+markers',
                                         marker=dict(color='#FFA15A', ),
                                         line=dict(color='#FFA15A', ),
                                         name='2015'))

            figdnct.add_trace(go.Scatter(x=dln15.loc[dln15.Year == 2016, "Hour"],
                                         y=dln15.loc[dln15.Year == 2016, "delT"],
                                         mode='lines+markers',
                                         marker=dict(color='#19D3F3', ),
                                         line=dict(color='#19D3F3', ),
                                         name='2016'))

            figdnct.add_trace(go.Scatter(x=dln15.loc[dln15.Year == 2017, "Hour"],
                                         y=dln15.loc[dln15.Year == 2017, "delT"],
                                         mode='lines+markers',
                                         marker=dict(color='#EF553B', ),
                                         line=dict(color='#EF553B', ),
                                         name='2017'))

            figdnct.add_trace(go.Scatter(x=dln.loc[dln.Year == 2018, "Hour"],
                                         y=dln.loc[dln.Year == 2018, "delT"],
                                         mode='lines+markers',
                                         marker=dict(color='red', ),
                                         line=dict(color='red', ),
                                         name='2018'))
            figdnct.add_trace(go.Scatter(x=dln.loc[dln.Year == 2019, "Hour"],
                                         y=dln.loc[dln.Year == 2019, "delT"],
                                         mode='lines+markers',
                                         marker=dict(color='blue', ),
                                         line=dict(color='blue', ),
                                         name='2019'))
            figdnct.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Average hourly Delay',
                'hovermode': 'x'
            })

            figdnct.update_yaxes(title_text="Delay - hr", range=[min_d, max_d])
            figdnct.update_xaxes(title_text="Hour")

            # non-cardinal total year delay figure
            figtdnc = go.Figure(data=[
                go.Bar(name='noncardinal', x=['2018', '2019'], y=list(tdn['delT']))
            ])
            figtdnc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Total Delay by Year ',
                'hovermode': 'x'
            })


            figtdnct = go.Figure(data=[
                go.Bar(name='cardinal', x=["2015", "2016", "2017", "2018", "2019"], y=list(tdelayn['delT']))
            ])

            figtdnct.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'Total Delay by Year',
                'hovermode': 'x'
            })

            figtdnct.update_xaxes(title_text="Year")
            figtdnct.update_yaxes(title_text="veh-hr")

            # non-cardinal heatmap figure
            fighmnc = go.Figure(data=go.Heatmap(
                x=np.array(pvnc.columns),
                y=np.array(pvnc.index),
                z=tnc,
                zsmooth=False,
                zmin=rfspdn // 1.5,
                zmax=rfspdn,
                connectgaps=False,
                colorscale='RdYlGn'))

            fighmnc.update_layout({'title': "Speed Distribution "})

            fighmnct = go.Figure(data=go.Heatmap(
                x=np.array(pvnc_t.columns),
                y=np.array(pvnc_t.index),
                z=tnc_t,
                zsmooth=False,
                zmin=rfspd_alln // 1.5,
                zmax=rfspd_alln,
                connectgaps=False,
                colorscale='RdYlGn'))

            fighmnct.update_layout({'title': "Speed Distribution"})

            # non-cardinal LOTTR figure
            figltnc = go.Figure(data=[
                go.Bar(name='non-cardinal', x=["2018", "2019"], y=list(lotrn['LOTTR']))
            ])

            figltnc.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'LOTTR by Year ',
                'hovermode': 'x'
            })
            figltnc.update_xaxes(title_text="Year")
            figltnc.update_yaxes(title_text="LOTTR")

            figltnct = go.Figure(data=[
                go.Bar(name='non-cardinal', x=["2015", "2016", "2017", "2018", "2019"], y=list(lotrn['LOTTR']))
            ])

            figltnct.update_layout({
                'plot_bgcolor': "#F9F9F9",
                'paper_bgcolor': "#F9F9F9",
                'title': 'LOTTR by Year ',
                'hovermode': 'x'
            })
            figltnct.update_xaxes(title_text="Year")
            figltnct.update_yaxes(title_text="LOTTR")

            graphs_return15= html.Div([
                            html.Div(className="row flex-display", children=[
                                html.Div([html.H3("Cardinal", id="ctext", )],
                                         className="six columns"),
                                html.Div([html.H3("Non-Cardinal", id="nctext", )],
                                         className="six columns")
                            ]),
                            html.Div(className="row flex-display", children=[
                                html.Div([dcc.Graph(id="HrSpdc", figure=figct)],
                                         className="pretty_container six columns"),
                                html.Div([dcc.Graph(id="HrSpdnc", figure=fignct)],
                                         className="pretty_container six columns")
                            ]),
                            html.Div(className="row flex-display", children=[
                                html.Div([dcc.Graph(id="DTc", figure=figdct)],
                                         className="pretty_container six columns"),
                                html.Div([dcc.Graph(id="DTnc", figure=figdnct)],
                                         className="pretty_container six columns")
                            ]),
                            html.Div(className="row flex-display", children=[
                                html.Div([dcc.Graph(id="Delc", figure=figtdct)],
                                         className="pretty_container six columns"),
                                html.Div([dcc.Graph(id="Delnc", figure=figtdnct)],
                                         className="pretty_container six columns")
                            ]),

                            # html.Div(className="row flex-display", children=[
                            #     html.Div([dcc.Graph(id="Hmapc", figure=fighmct)],
                            #              className="pretty_container six columns"),
                            #     html.Div([dcc.Graph(id="Hmapnc", figure=fighmnct)],
                            #              className="pretty_container six columns")
                            # ]),

                             html.Div(className="row flex-display", children=[
                               html.Div([dcc.Graph(id="Lottrc", figure=figltct)],
                                          className="pretty_container six columns"),
                               html.Div([dcc.Graph(id="Lottrnc", figure=figltnct)],
                                          className="pretty_container six columns")
                             ])

                        ]),
            return graphs_return15
    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run_server()
