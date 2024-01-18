import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from stockviewer.eiten import load_obj

app = dash.get_app()
dash.register_page(__name__, path = "/strategies", title = "Stocks evolution")

def generate_section_banner(title):
    return html.Div(className="section-banner", children = title)

def build_chart_panel(fig, title = "Test"):

    return html.Div( id= f"{title.replace(' ', '-')}-chart-container",
                    className = "chart-panel",
                    children = [generate_section_banner(title),
                                dcc.Graph(className="chart-panel-graph", figure = fig) ] )

def build_chart_row(row_panels):
    return dbc.Row(
        [
            dbc.Col([
                row_panel
                ]) for row_panel in row_panels
        ]
    )

def build_chart_table(all_panels, nb_charts_per_row = 1):
    panels_table = [all_panels[i:i+nb_charts_per_row] for i in range(0, len(all_panels), nb_charts_per_row)]
    return [ build_chart_row(row_panels) for row_panels in panels_table ]
                                    
@app.callback(
    Output("strategies-dashboard", "children"),
    Input("memory", "data"),
)
def build_dashboard(is_pfm_computed):
    if is_pfm_computed:
        pfm = load_obj("./current_pfm")
        fig_list = pfm.get_strategies_figs()
        all_panels = [build_chart_panel(fig = fig, title = title) for (fig, title) in fig_list]#[build_chart_panel()]*6

        nb_charts_per_row = 2
        return build_chart_table(all_panels, nb_charts_per_row)

layout = html.Div(
    id= "strategies-container",
    className = "app-container",
    children = [ 
        html.Div(id = "strategies-dashboard"),
        dcc.Loading( 
            id = "strategies-loading",
            children = [html.Div(id = "strategies-content", className= "strategies-content")],
            type = "circle", color = "green", className = "loading-spinner"
            ) 
            ]
)