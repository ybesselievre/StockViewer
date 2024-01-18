import pathlib, dash, os, signal, threading, webbrowser

from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from flask import Flask, redirect
from flask_caching import Cache
from stockviewer import eiten
from stockviewer.eiten import save_obj

server = Flask(__name__)
@server.route('/')
def index_redirect():
    return redirect(f'/stocks')

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    use_pages = True,
    server = server
)
app.title = "Stock Viewer"

CACHE_CONFIG = {'CACHE_TYPE': 'FileSystemCache',
                'CACHE_DIR' : './cache/'}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)

APP_PATH = str(pathlib.Path(__file__).parent.resolve())


#Layout : banner and tabs
banner = html.Div(
            [
                html.Div(
                    id = "imgbox",
                    children = html.Img(id="logo", src=app.get_asset_url("icon_stockv.png"))),
                html.Div(
                    id="banner-text",
                    children=[
                        html.H5("Stock Viewer"),
                        html.H6("Stock and Portfolio Management Strategy Dashboard"),
                    ],
                ),
                html.Div(
                    id="banner-logo",
                    children=[dbc.Button("SHUTDOWN", id="shutdown-button", color="danger", size = "lg")
                    ],
                )
            ],
            id="banner",
            className="banner",
        )


sidebar = html.Div(
        [
            dbc.Nav(
                [
                    dbc.NavLink(
                        [
                            html.Img(src=app.get_asset_url("icon_stock_transparent.gif"), style= {"height" : "8rem"}),
                            html.Span("Stocks"),
                        ],
                        href=dash.page_registry['pages.stocks']['relative_path'],
                        active="exact",
                        className= "nav-link"
                    ),
                    html.Hr(),
                    dbc.NavLink(
                        [
                            html.Img(src=app.get_asset_url("icon_strategy_transparent.gif"), style= {"height" : "8rem"}),
                            html.Span("Strategies"),
                        ],
                        href=dash.page_registry['pages.strategies']['relative_path'],
                        active="exact",
                        className= "nav-link"
                    ),
                ],
                vertical=True,
                pills=True,
                justified=True,
            ),
        ],
        id="sidebar",
        className="nav-sidebar",
    )

shutdown_modal = html.Div(
        id="shutdown-markdown",
        className="modal",
        children=(
            html.Div(
                id="shutdown-markdown-container",
                className="markdown-container",
                children=[
                    html.Div(
                        className="markdown-text",
                        children=dcc.Markdown(
                            children=(
                                """
                        #### Shutdown confirmation

                        Are you sure you want to shut down the app ?
                    """
                            )
                        ),
                    ),
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Confirm Shutdown",
                            id="confirmed-shutdown-button",
                            n_clicks=0,
                            className="shutdownButton",
                        ),
                    ),
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Close",
                            id="shutdown-markdown_close",
                            n_clicks=0,
                            className="closeButton",
                        ),
                    ),
                ],
            )
        ),
    )

shell = html.Div(
        [
            banner,
            sidebar,
            shutdown_modal,
        ],
        id="app-shell",
        className="shell"
        )


#Init Eiten object
pfm = eiten.Eiten(eiten.Args())
pfm.load_data()


app.layout = html.Div(
    id="big-app-container",
    children=[
        dcc.Store("memory", data = False),
        shell,
        dash.page_container,
        html.P(id='shutdown-placeholder')
    ],
)

@app.callback(
    Output("memory", "data"),
    Input("memory", "data")
)
def update_click_output(is_pfm_computed):
    if not is_pfm_computed:
        save_obj(pfm, "./current_pfm")
    return True

# "SHUTDOWN" modal window
@app.callback(
    Output("shutdown-markdown", "style"),
    [Input("shutdown-button", "n_clicks"), Input("shutdown-markdown_close", "n_clicks")],
)
def update_click_output(button_click, close_click):
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "shutdown-button":
            return {"display": "block"}

    return {"display": "none"}

# Definitive SHUTDOWN action
@app.callback(
    Output("shutdown-placeholder", "children"),
    [Input("confirmed-shutdown-button", "n_clicks")],
)
def update_click_output(button_click):
    if button_click:
        os.kill(os.getpid(), signal.SIGTERM)
    return None


# Running the server
if __name__ == "__main__":
    threading.Timer(1.25, lambda: webbrowser.open("http://127.0.0.1:9000/") ).start()
    app.run_server(debug = True, port=9000)