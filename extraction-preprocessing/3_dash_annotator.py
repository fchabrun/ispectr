from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os
import json
import numpy as np

json_rootdirectory = r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data\2025\lemans\preannotation"
json_filenames = os.listdir(os.path.join(json_rootdirectory, "input_jsons"))
default_json_filename = json_filenames[0]
json_list_dropdown_data = []
for json_filename in json_filenames:
    previous_data_exists = os.path.exists(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename))
    if previous_data_exists:
        json_color = 'Gold'
    else:
        json_color = 'Black'
    new_json_entry = {'label': html.Span([json_filename], style={'color': json_color}),
                      'value': json_filename,
                      'search': json_filename}
    json_list_dropdown_data.append(new_json_entry)

app = Dash()

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "24rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "26rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H3("iSPECTR annotation tool", className="display-4"),
        html.Hr(),
        dcc.Dropdown(json_list_dropdown_data, default_json_filename, id='json-dropdown-selection'),
        html.P(id="old-lemans-text"),
        html.P(id="old-lemans-class"),
    ],
    style=SIDEBAR_STYLE,
)

graphs_layout = html.Div(
    [
        html.Div(
            [
                html.P("SPEP", style={"display": "block", "text-align": "left", "margin": "1rem", "font-weight": "bold"}),
                dcc.Graph(id='spe-graph-content'),
                html.P("ELP", style={"display": "block", "text-align": "left", "margin": "1rem", "font-weight": "bold"}),
                dcc.Graph(id='elp-graph-content-left'),
                html.P("Kappa", style={"display": "block", "text-align": "left", "margin": "1rem", "font-weight": "bold"}),
                dcc.Graph(id='k-graph-content'),
                html.P("Lambda", style={"display": "block", "text-align": "left", "margin": "1rem", "font-weight": "bold"}),
                dcc.Graph(id='l-graph-content'),
            ], style={"display": "inline-block", "width": "50%"}
        ),
        html.Div(
            [
                html.P("ELP", style={"display": "block", "text-align": "left", "margin": "1rem", "font-weight": "bold"}),
                dcc.Graph(id='elp-graph-content-right'),
                html.P("IgG", style={"display": "block", "text-align": "left", "margin": "1rem", "font-weight": "bold"}),
                dcc.Graph(id='g-graph-content'),
                html.P("IgA", style={"display": "block", "text-align": "left", "margin": "1rem", "font-weight": "bold"}),
                dcc.Graph(id='a-graph-content'),
                html.P("IgM", style={"display": "block", "text-align": "left", "margin": "1rem", "font-weight": "bold"}),
                dcc.Graph(id='m-graph-content'),
            ], style={"display": "inline-block", "width": "50%"}
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
    fig.update_xaxes(title='', visible=True, showticklabels=True,
                     showline=True,
                     linecolor='black',
                     gridcolor='lightgrey')
    fig.update_yaxes(title='', visible=True, showticklabels=True,
                     showline=True,
                     linecolor='black',
                     gridcolor='lightgrey')
    fig.update_layout(
        plot_bgcolor='white',
        autosize=False,
        width=600,
        height=200,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=2
        ),
    )
    return fig


@callback(
    Output('elp-graph-content-left', 'figure'),
    Output('elp-graph-content-right', 'figure'),
    Output('g-graph-content', 'figure'),
    Output('a-graph-content', 'figure'),
    Output('m-graph-content', 'figure'),
    Output('k-graph-content', 'figure'),
    Output('l-graph-content', 'figure'),
    # Output('ref-graph-content', 'figure'),
    Output('spe-graph-content', 'figure'),
    Output('old-lemans-text', 'children'),
    Output('old-lemans-class', 'children'),
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
    for trace_name in ("ELP", "ELP", "IgG", "IgA", "IgM", "K", "L", "SPE"):
        new_trace = get_trace_plot(sample_data["traces"][trace_name]["data"], trace_name)
        traces.append(new_trace)

    if os.path.exists(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename)):
        old_lemans_text = "Previous (2020) data exists"

        with open(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename), "r") as f:
            previous_sample_data = json.load(f)

        old_lemans_class = previous_sample_data["groundtruth_class"]

        # groundtruth_class
        # groundtruth_maps
    else:
        old_lemans_text = "No previous data exists"
        old_lemans_class = ""

    return *traces, old_lemans_text, old_lemans_class


if __name__ == '__main__':
    app.run()