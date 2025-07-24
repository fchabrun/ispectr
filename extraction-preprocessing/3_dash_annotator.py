from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import os
import json
import numpy as np

json_rootdirectory = r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data\2025\lemans\preannotation"

app = Dash()

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("iSPECTR annotation tool", className="display-4"),
        html.Hr(),
        dcc.Dropdown(os.listdir(os.path.join(json_rootdirectory, "input_jsons")), '', id='json-dropdown-selection'),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div([html.H1(children='iSPECTR annotation tool'),
                    dcc.Graph(id='elp-graph-content')
                    ],
                   id="page-content",
                   style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@callback(
    Output('elp-graph-content', 'figure'),
    Input('json-dropdown-selection', 'value')
)
def update_graph(json_filename):
    if json_filename == '':
        return None

    with open(os.path.join(json_rootdirectory, "input_jsons", json_filename), "r") as f:
        sample_data = json.load(f)

    plot_df = pd.DataFrame({"Point": np.arange(304) + 1,
                            "Relative intensity": np.array(sample_data["traces"]["ELP"]["data"])})

    fig = px.line(plot_df, x="Point", y="Relative intensity", labels={"Trace": "ELP"})
    return fig


if __name__ == '__main__':
    app.run()