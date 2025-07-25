from dash import Dash, html, dcc, callback, Output, Input, State, dash_table, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os
import json
import numpy as np

json_rootdirectory = r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data\2025\lemans\preannotation"
json_filenames = os.listdir(os.path.join(json_rootdirectory, "input_jsons"))
# default_json_filename = json_filenames[0]
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

app = Dash(prevent_initial_callbacks=True)

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "24rem",
    "padding": "0rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "26rem",
    "margin-right": "2rem",
    "padding": "0rem 1rem",
}

sidebar = html.Div(
    [
        html.H3("iSPECTR annotation tool", className="display-4"),
        html.Hr(),
        dcc.Dropdown(json_list_dropdown_data, "", id='json-dropdown-selection'),
        html.Button('Populate from SPECTR', id='spectr-to-peaks-button', n_clicks=0),
        html.Button('Populate from PREVIOUS (2020)', id='previous-2020-to-peaks-button', n_clicks=0),
        html.Br(),
        html.Br(),
        dash_table.DataTable(
            id='output-peaks-data-table',
            columns=[
                {
                    'name': 'Start',
                    'id': 'start',
                    'deletable': False,
                    'renamable': False,
                },
                {
                    'name': 'End',
                    'id': 'end',
                    'deletable': False,
                    'renamable': False
                },
                {
                    'name': 'HC',
                    'id': 'hc',
                    'presentation': 'dropdown',
                    'deletable': False,
                    'renamable': False
                },
                {
                    'name': 'LC',
                    'id': 'lc',
                    'presentation': 'dropdown',
                    'deletable': False,
                    'renamable': False
                },
            ],
            dropdown={
                'hc': {
                    'options': [
                        {'label': "IgG", 'value': "IgG"},
                        {'label': "IgA", 'value': "IgA"},
                        {'label': "IgM", 'value': "IgM"},
                    ]
                },
                'lc': {
                    'options': [
                        {'label': "K", 'value': "K"},
                        {'label': "L", 'value': "L"},
                    ]
                },
            },
            data=[],
            editable=True,
            row_deletable=True
        ),
        html.Br(),
        html.Button('Add peak', id='add-peak-button', n_clicks=0),
        html.Hr(),
        html.P(id="old-lemans-text"),
        html.P("Local institution comments", style={"font-weight": "bold"}),
        html.P(id="short-comments-text"),
        html.P(id="long-comments-text"),
    ],
    style=SIDEBAR_STYLE,
)

graphs_layout = html.Div(
    [
        html.Div(
            [
                html.P("ELP", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='elp-graph-content-right'),
                html.P("IgG", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='g-graph-content'),
                html.P("IgA", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='a-graph-content'),
                html.P("IgM", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='m-graph-content'),
            ], style={"display": "inline-block", "width": "50%"}
        ),
        html.Div(
            [
                html.P("ELP", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='elp-graph-content-left'),
                html.P("Kappa", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='k-graph-content'),
                html.P("Lambda", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='l-graph-content'),
            ], style={"display": "inline-block", "width": "50%"}
        ),
    ]
)

content = html.Div([graphs_layout,
                    ],
                   id="page-content",
                   style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


def get_trace_plot(trace_data, spectr_preds=None, trace_peak_data=None):
    trace_data = np.array(trace_data)
    color_discrete_map = {"Relative intensity": 'black'}
    if spectr_preds is not None:
        plot_df = pd.DataFrame({"Relative time": np.arange(len(trace_data)) + 1,
                                "Relative intensity": trace_data,
                                "SPECTR": spectr_preds})
        y_cols = ["Relative intensity", "SPECTR"]
        color_discrete_map["SPECTR"] = "blue"
    else:
        plot_df = pd.DataFrame({"Relative time": np.arange(len(trace_data)) + 1,
                                "Relative intensity": trace_data})
        y_cols = ["Relative intensity", ]

    if trace_peak_data is not None:  # add peaks
        # create map from peak
        for i in range(len(trace_peak_data)):
            start = trace_peak_data.start.iloc[i]
            end = trace_peak_data.end.iloc[i]
            if end < start:
                continue
            if end > len(trace_data):
                continue
            if start < 0:
                continue
            ontrace = trace_peak_data.ontrace.iloc[i]
            peak_map = np.concatenate([np.repeat(0, start),
                                       np.repeat(1, end - start),
                                       np.repeat(0, len(trace_data) - end)])
            plot_df[f"Peak {i+1}"] = peak_map
            y_cols.append(f"Peak {i+1}")
            if ontrace:
                color_discrete_map[f"Peak {i+1}"] = "red"
            else:
                color_discrete_map[f"Peak {i+1}"] = "#cccccc"

    fig = px.line(plot_df, x="Relative time", y=y_cols, color_discrete_map=color_discrete_map)
    fig.update_xaxes(title='', visible=True, showticklabels=True,
                     showline=True,
                     linecolor='black',
                     gridcolor='lightgrey')
    fig.update_yaxes(title='', visible=True, showticklabels=False,
                     showline=False,
                     linecolor='white',
                     gridcolor='white')
    fig.update_layout(
        showlegend=False,
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
    Output('output-peaks-data-table', 'data', allow_duplicate=True),
    Output('elp-graph-content-left', 'figure'),
    Output('elp-graph-content-right', 'figure'),
    Output('g-graph-content', 'figure'),
    Output('a-graph-content', 'figure'),
    Output('m-graph-content', 'figure'),
    Output('k-graph-content', 'figure'),
    Output('l-graph-content', 'figure'),
    # Output('ref-graph-content', 'figure'),
    # Output('spe-graph-content', 'figure'),
    Output('short-comments-text', 'children'),
    Output('long-comments-text', 'children'),
    Output('old-lemans-text', 'children'),
    Input('json-dropdown-selection', 'value'),
    Input('output-peaks-data-table', 'data'),
    State('output-peaks-data-table', 'data'),
    prevent_initial_call=True
)
def update_graph(json_filename, peak_data, rows):
    if json_filename == '':
        return None

    if ctx.triggered_id == "json-dropdown-selection":  # we changed the json file => reset peaks
        rows = []

    if len(peak_data) > 0:
        peak_data = pd.DataFrame(peak_data)
        peak_data.start = pd.to_numeric(peak_data.start)
        peak_data.end = pd.to_numeric(peak_data.end)
        peak_data = peak_data[peak_data.lc.isin(["K", "L"]) | peak_data.hc.isin(["IgG", "IgA", "IgM"])]
        peak_data = peak_data[~peak_data.start.isna()]
        peak_data = peak_data[~peak_data.end.isna()]
    else:
        peak_data = None

    # load input data
    with open(os.path.join(json_rootdirectory, "input_jsons", json_filename), "r") as f:
        sample_data = json.load(f)

    short_comments = sample_data["short_comments"]
    long_comments = sample_data["long_comments"]

    short_comments = [html.P(txt, style={"padding": "0", "margin": "1rem 0"}) for txt in short_comments.split("\\n") if len(txt) > 0]
    long_comments = [html.P(txt, style={"padding": "0", "margin": "1rem 0"}) for txt in long_comments.split("\\n") if len(txt) > 0]

    # load spectr data
    with open(os.path.join(json_rootdirectory, "spectr_jsons", json_filename), "r") as f:
        sample_spectr_data = json.load(f)
    spectr_elp_preds = np.array(sample_spectr_data["elp_spep_s_predictions"])

    # TODO should have TWO MODES:
    # 1) annotate -> do NOT load output files, just input files + pre-annotations (SPECTR) + Le Mans OLD. EVEN IF OUTPUT EXISTS. Enable user to save output annotation. DOES NOT BY DEFAULT DISPLAY SAMPLES FOR WHICH AN ANNOTATION EXISTS.
    # 2) review -> do NOT suggest m-spike location, just look at existing (saved) OUTPUT annotations. Look only at samples for which an OUTPUT file EXISTS. Should enable user to overwrite existing annotation OR remove it (so we can start from scratch)

    # TODO save

    # TODO reload existing output

    # TODO in comments, automatically detect words like "IgG", "kappa", etc and put them in BOLD RED

    traces = []
    for trace_name in ("ELP", "ELP", "IgG", "IgA", "IgM", "K", "L"):
        if (trace_name != "ELP") and (peak_data is not None):
            trace_peak_data = peak_data.copy()
            trace_peak_data["ontrace"] = (trace_peak_data.lc == trace_name) | (trace_peak_data.hc == trace_name)
        else:
            trace_peak_data = None
        new_trace = get_trace_plot(sample_data["traces"][trace_name]["data"], spectr_preds=spectr_elp_preds if trace_name == "ELP" else None, trace_peak_data=trace_peak_data)
        traces.append(new_trace)

    if os.path.exists(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename)):
        with open(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename), "r") as f:
            previous_sample_data = json.load(f)

        old_lemans_class_ = previous_sample_data["groundtruth_class"]
        old_lemans_text = f"Previous (2020) data exists: {old_lemans_class_}"
    else:
        old_lemans_text = None

    return rows, *traces, short_comments, long_comments, old_lemans_text


@callback(
    Output('output-peaks-data-table', 'data', allow_duplicate=True),
    Input('add-peak-button', 'n_clicks'),
    Input('spectr-to-peaks-button', 'n_clicks'),
    Input('previous-2020-to-peaks-button', 'n_clicks'),
    State('json-dropdown-selection', 'value'),
    State('output-peaks-data-table', 'data'),
    State('output-peaks-data-table', 'columns'),
    prevent_initial_call=True
)
def add_row(add_n_clicks, spectr_n_clicks, previous_2020_n_clicks, json_filename, rows, columns):

    if add_n_clicks > 0:
        if ctx.triggered_id == "add-peak-button":  # we changed the json file => reset peaks
            rows.append({c['id']: '' for c in columns})

    if spectr_n_clicks > 0:
        if ctx.triggered_id == "spectr-to-peaks-button":  # we changed the json file => reset peaks
            rows = []

            # load spectr data
            with open(os.path.join(json_rootdirectory, "spectr_jsons", json_filename), "r") as f:
                sample_spectr_data = json.load(f)
            # extract prediction map
            spectr_elp_preds = np.array(sample_spectr_data["elp_spep_s_predictions"])
            # clean and get predicted positions
            spectr_elp_preds = (spectr_elp_preds > .1) * 1
            # compute diff (increase/decrease)
            spectr_elp_preds_diff = np.diff(spectr_elp_preds)
            # compute start and end positions
            peak_starts = np.where(spectr_elp_preds_diff == 1)[0]
            peak_ends = np.where(spectr_elp_preds_diff == -1)[0]
            if len(peak_starts) == len(peak_ends):  # only act if n(starts) == n(ends)
                for start, end in zip(peak_starts, peak_ends):
                    if end <= start:  # prevent errors
                        continue
                    if end > 299:  # prevent peaks outside of the spep
                        continue
                    if start < 150:  # prevent peaks too early (e.g. albumin)
                        continue
                    rows.append({'start': start, 'end': end, 'hc': "", 'lc': ""})

    if previous_2020_n_clicks:
        if ctx.triggered_id == "previous-2020-to-peaks-button":  # we changed the json file => reset peaks
            rows = []

            if os.path.exists(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename)):

                with open(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename), "r") as f:
                    previous_sample_data = json.load(f)

                for isotype in ["IgG", "IgA", "IgM", "K", "L"]:
                    iso_trace = np.array(previous_sample_data["groundtruth_maps"][isotype])
                    # compute diff (increase/decrease)
                    iso_trace_diff = np.diff(iso_trace)
                    # compute start and end positions
                    peak_starts = np.where(iso_trace_diff == 1)[0]
                    peak_ends = np.where(iso_trace_diff == -1)[0]
                    if len(peak_starts) == len(peak_ends):  # only act if n(starts) == n(ends)
                        for start, end in zip(peak_starts, peak_ends):
                            if end <= start:  # prevent errors
                                continue
                            if end > 299:  # prevent peaks outside of the spep
                                continue
                            if start < 150:  # prevent peaks too early (e.g. albumin)
                                continue
                            # append or if LC: check if a HC was already placed at this location and precise lc
                            if isotype in ('IgA', 'IgG', 'IgM'):
                                rows.append({'start': start, 'end': end, 'hc': isotype, 'lc': ''})
                            else:
                                added_to_hc = False
                                for row in rows:
                                    if (row['start'] == start) and (row['end'] == end):
                                        row['lc'] = isotype
                                        added_to_hc = True
                                if not added_to_hc:
                                    rows.append({'start': start, 'end': end, 'hc': '', 'lc': isotype})

    return rows


if __name__ == '__main__':
    app.run()