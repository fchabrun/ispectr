from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os
import json
import numpy as np

json_rootdirectory = r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data\2025\lemans\preannotation"
json_filenames = os.listdir(os.path.join(json_rootdirectory, "input_jsons"))
json_filename = json_filenames[0]

app = Dash()

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
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
        html.H3("iSPECTR annotation tool", className="display-4"),
        html.Hr(),
        dcc.Dropdown(json_filenames, json_filenames[0], id='json-dropdown-selection'),
        html.P(id="old-lemans-text"),
    ],
    style=SIDEBAR_STYLE,
)

graphs_layout = html.Div(
    [
        html.Div(
            [
                dcc.Graph(id='elp-graph-content'),
                dcc.Graph(id='k-graph-content'),
                dcc.Graph(id='l-graph-content'),
            ], style={"display": "inline-block", "width": "40%"}
        ),
        html.Div(
            [
                dcc.Graph(id='g-graph-content'),
                dcc.Graph(id='a-graph-content'),
                dcc.Graph(id='m-graph-content'),
            ], style={"display": "inline-block", "width": "40%"}
        ),
    ]
)

content = html.Div([graphs_layout,
                    ],
                   id="page-content",
                   style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


def get_trace_plot(trace_data, name):
    trace_data = np.array(trace_data)
    plot_df = pd.DataFrame({name: np.arange(len(trace_data)) + 1,
                            "Relative intensity": trace_data})
    fig = px.line(plot_df, x=name, y="Relative intensity")
    fig.update_layout(
        autosize=False,
        width=600,
        height=200,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=4
        ),
    )
    return fig


@callback(
    Output('elp-graph-content', 'figure'),
    Output('g-graph-content', 'figure'),
    Output('a-graph-content', 'figure'),
    Output('m-graph-content', 'figure'),
    Output('k-graph-content', 'figure'),
    Output('l-graph-content', 'figure'),
    # Output('ref-graph-content', 'figure'),
    # Output('spe-graph-content', 'figure'),
    Output('old-lemans-text', 'children'),
    Input('json-dropdown-selection', 'value')
)
def update_graph(json_filename):
    if json_filename == '':
        return None

    with open(os.path.join(json_rootdirectory, "input_jsons", json_filename), "r") as f:
        sample_data = json.load(f)

    # TODO show REF and SPEP

    # TODO dropdown replace by something more convenient

    # TODO show previous data when existing

    # TODO allow to place peaks

    # TODO show SPECTR analysis

    # TODO automatically pre-place peaks from SPECTR output

    # TODO should have TWO MODES:
    # 1) annotate -> do NOT load output files, just input files + pre-annotations (SPECTR) + Le Mans OLD. EVEN IF OUTPUT EXISTS. Enable user to save output annotation. DOES NOT BY DEFAULT DISPLAY SAMPLES FOR WHICH AN ANNOTATION EXISTS.
    # 2) review -> do NOT suggest m-spike location, just look at existing (saved) OUTPUT annotations. Look only at samples for which an OUTPUT file EXISTS. Should enable user to overwrite existing annotation OR remove it (so we can start from scratch)

    traces = []
    # for trace_name in ("ELP", "IgG", "IgA", "IgM", "K", "L", "Ref", "SPE"):
    for trace_name in ("ELP", "IgG", "IgA", "IgM", "K", "L",):
        new_trace = get_trace_plot(sample_data["traces"][trace_name]["data"], trace_name)
        traces.append(new_trace)

    if os.path.exists(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename)):
        old_lemans_text = "Previous (2020) data exists"

        # groundtruth_class
        # groundtruth_maps
    else:
        old_lemans_text = "No previous data exists"

    return *traces, old_lemans_text


if __name__ == '__main__':
    app.run()