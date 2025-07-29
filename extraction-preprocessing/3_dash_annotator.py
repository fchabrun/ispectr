from dash import Dash, html, dcc, callback, Output, Input, State, dash_table, ctx
# import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os
import json
import numpy as np
from datetime import datetime
# import shutil

# OVERWRITE_OUTPUT_JSON_WITH_NEW_INPUT_DATA = False

json_rootdirectory = r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data\2025\lemans\preannotation"


# if OVERWRITE_OUTPUT_JSON_WITH_NEW_INPUT_DATA:
#     for json_filename in os.listdir(os.path.join(json_rootdirectory, "output_jsons")):
#         # load input json
#         with open(os.path.join(json_rootdirectory, "input_jsons", json_filename), "r") as f:
#             input_json_content = json.load(f)
#         # load output (annotated) json
#         with open(os.path.join(json_rootdirectory, "output_jsons", json_filename), "r") as f:
#             output_json_content = json.load(f)
#         # add annotations to input json
#         input_json_content["peak_data"] = output_json_content["peak_data"]
#         input_json_content["doubtful"] = output_json_content["doubtful"]
#         input_json_content["exclude"] = output_json_content["exclude"]
#         input_json_content["annotated_by"] = output_json_content["annotated_by"]
#         # overwrite previous output (annotated) json
#         with open(os.path.join(json_rootdirectory, "output_jsons", json_filename), 'w') as f:
#             json.dump(input_json_content, f, indent=4)


def json_file_lists_to_dropdown_options(full_json_list, mode):
    json_list_dropdown_data = []
    for json_info in full_json_list:
        if mode == "annotate":
            json_color = 'Gold' if json_info["previous"] else 'Black'
        elif mode == "review":
            json_color = 'Red' if json_info["exclude"] else 'Orange' if json_info["doubtful"] else 'Green'
        elif mode == "confirm":
            json_color = 'Red' if json_info["exclude"] else 'Orange' if json_info["doubtful"] else 'Green'
        else:
            assert False, f"Unknown {mode=}"
        new_json_entry = {'label': html.Span([json_info["json_filename"]], style={'color': json_color}),
                          'value': json_info["json_filename"],
                          'search': json_info["json_filename"]}
        json_list_dropdown_data.append(new_json_entry)
    return json_list_dropdown_data


# pre load the list of json files
# list all json files (annotated and unannotated)
full_json_list = []
existingpreviousdata_json_filenames = os.listdir(os.path.join(json_rootdirectory, "previous_2020_output_jsons"))
for json_filename in os.listdir(os.path.join(json_rootdirectory, "input_jsons")):
    full_json_list.append({"json_filename": json_filename, "previous": (json_filename in existingpreviousdata_json_filenames)})

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
        dcc.Input(
            id='input-reviewer',
            type="text",
            placeholder="Reviewer's name"
        ),
        html.Hr(),
        dcc.RadioItems(
            options={
                'annotate': 'Annotate',
                'confirm': 'Confirm',
                'review': 'Review'
            },
            inline=True,
            value='annotate',
            id="mode-radio"
        ),
        html.Br(),
        html.Div(id='n-json-files-found'),
        dcc.Dropdown([], "", id='json-dropdown-selection'),
        html.Div(id='reviewer-id-text', style={'font-style': 'italic'}),
        html.Hr(),
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
        html.Br(),
        html.Br(),
        dcc.Checklist(['Doubtful', 'Exclude', ], [], id="comments-checkbox"),
        html.Div([
            html.Button('SAVE', id='save-output-button', n_clicks=0, style={"font-weight": "bold"}),
            html.Br(),
            html.Br(),
            html.Button('DISCARD', id='discard-output-button', n_clicks=0, style={}),
        ], style={"text-align": "center"}),
        html.Br(),
        html.Hr(),
        html.P(id="old-lemans-text"),
        html.P("Comments", style={"font-weight": "bold"}),
        html.P(id="short-comments-text"),
        html.P(id="long-comments-text"),
    ],
    style=SIDEBAR_STYLE,
)

graphs_layout = html.Div(
    [
        dcc.ConfirmDialog(
            id='error-dialog',
            message='Unable to save annotations: incomplete or invalid peak data',
        ),
        html.Div(
            [
                html.P("ELP", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='elp-graph-content-right', style={"display": "block", "height": "20vh"}),
                html.P("IgG", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='g-graph-content', style={"display": "block", "height": "20vh"}),
                html.P("IgA", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='a-graph-content', style={"display": "block", "height": "20vh"}),
                html.P("IgM", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='m-graph-content', style={"display": "block", "height": "20vh"}),
            ], style={"display": "inline-block", "width": "50%"}
        ),
        html.Div(
            [
                html.P("SPEP", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='spe-graph-content', style={"display": "block", "height": "20vh"}),
                html.P("ELP", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='elp-graph-content-left', style={"display": "block", "height": "20vh"}),
                html.P("Kappa", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='k-graph-content', style={"display": "block", "height": "20vh"}),
                html.P("Lambda", style={"display": "block", "text-align": "left", "margin": "0.5rem", "font-weight": "bold"}),
                dcc.Graph(id='l-graph-content', style={"display": "block", "height": "20vh"}),
            ], style={"display": "inline-block", "width": "50%"}
        ),
    ]
)

content = html.Div([graphs_layout,
                    ],
                   id="page-content",
                   style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


def get_trace_plot(trace_data, doubtful, exclude, spectr_preds=None, trace_peak_data=None, peak_color="red", peak_alt_color="#cccccc"):
    trace_data = np.array(trace_data)
    color_discrete_map = {"Relative intensity": 'red' if exclude else 'orange' if doubtful else 'black'}
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

    if (trace_peak_data is not None) and (not exclude):  # add peaks
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
                color_discrete_map[f"Peak {i+1}"] = peak_color
            else:
                color_discrete_map[f"Peak {i+1}"] = peak_alt_color

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
        # autosize=True,
        # autosize=False,
        # width=600,
        # height=200,
        # minreducedwidth=600,
        # minreducedheight=200,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        ),
    )
    return fig


def comments_to_spans(comments, keyword="(IgG|IgA|IgM|kappa|Kappa|lambda|Lambda)"):
    import re

    spans = []
    for com_txt in comments:
        new_span = [html.Span(txt, style={"color": ("red" if i % 2 == 1 else "black")}) for i, txt in enumerate(re.split(keyword, com_txt))]
        spans.append(new_span)
    return spans


@callback(
    Output('error-dialog', 'displayed'),
    Output('error-dialog', 'message'),
    Output('n-json-files-found', 'children'),
    Output('json-dropdown-selection', 'options'),
    Output('json-dropdown-selection', 'value'),
    Output('save-output-button', 'children'),
    Output('discard-output-button', 'children'),
    Output('save-output-button', 'style'),
    Output('discard-output-button', 'style'),
    Input('save-output-button', 'n_clicks'),
    Input('discard-output-button', 'n_clicks'),
    Input('mode-radio', 'value'),
    State('output-peaks-data-table', 'data'),
    State('json-dropdown-selection', 'options'),
    State('json-dropdown-selection', 'value'),
    State('comments-checkbox', 'value'),
    State('save-output-button', 'children'),
    State('discard-output-button', 'children'),
    State('save-output-button', 'style'),
    State('discard-output-button', 'style'),
    State('n-json-files-found', 'children'),
    State('input-reviewer', 'value')
)
def save_and_update_json_files_list(n_clicks_save, n_clicks_discard, mode, rows, prev_json_options, json_filename, comments_checkbox, prev_save_button_txt, prev_discard_button_txt, prev_save_button_style, prev_discard_button_style, prev_n_json_txt, reviewer_id):
    # if save => save
    if (n_clicks_save > 0) or (n_clicks_discard > 0):
        if ctx.triggered_id == "save-output-button":
            # reject if annotations are not OK
            if (reviewer_id is None) or (reviewer_id == ""):
                error_dialog_msg = "Please enter a reviewer name"
                return True, error_dialog_msg, prev_n_json_txt, prev_json_options, json_filename, prev_save_button_txt, prev_discard_button_txt, prev_save_button_style, prev_discard_button_style
            for row in rows:
                if (int(row["start"]) <= 150) or (int(row["end"]) <= 150) or (int(row["start"]) >= 300) or (int(row["end"]) >= 300):
                    return True, "Unable to save annotations: invalid peak position", prev_n_json_txt, prev_json_options, json_filename, prev_save_button_txt, prev_discard_button_txt, prev_save_button_style, prev_discard_button_style
                if int(row["start"]) >= int(row["end"]):
                    return True, "Unable to save annotations: invalid peak size", prev_n_json_txt, prev_json_options, json_filename, prev_save_button_txt, prev_discard_button_txt, prev_save_button_style, prev_discard_button_style
                if row["hc"] not in ["IgG", "IgA", "IgM", ""]:
                    return True, "Unable to save annotations: invalid peak HC", prev_n_json_txt, prev_json_options, json_filename, prev_save_button_txt, prev_discard_button_txt, prev_save_button_style, prev_discard_button_style
                if row["lc"] not in ["K", "L", ""]:
                    return True, "Unable to save annotations: invalid peak LC", prev_n_json_txt, prev_json_options, json_filename, prev_save_button_txt, prev_discard_button_txt, prev_save_button_style, prev_discard_button_style
                if (row["hc"] == "") and (row["lc"] == ""):
                    return True, "Unable to save annotations: HC and LC cannot be both missing", prev_n_json_txt, prev_json_options, json_filename, prev_save_button_txt, prev_discard_button_txt, prev_save_button_style, prev_discard_button_style

            if mode == "annotate":
                # reload json input file and add new annotations to it
                with open(os.path.join(json_rootdirectory, "input_jsons", json_filename), "r") as f:
                    json_content = json.load(f)
                # create json content
                json_content["peak_data"] = rows
                json_content["doubtful"] = "Doubtful" in comments_checkbox
                json_content["exclude"] = "Exclude" in comments_checkbox
                json_content["annotated_by"] = reviewer_id
                json_content["annotated_at"] = f"{datetime.now()}"
                # save new json
                with open(os.path.join(json_rootdirectory, "output_jsons", json_filename), 'w') as f:
                    json.dump(json_content, f, indent=4)
            elif mode == "confirm":  # copy in validated folder
                # shutil.copy(os.path.join(json_rootdirectory, "output_jsons", json_filename),
                #             os.path.join(json_rootdirectory, "confirmed_jsons", json_filename))
                # new with reviewer name
                # reload json input file and add new annotations to it
                with open(os.path.join(json_rootdirectory, "output_jsons", json_filename), "r") as f:
                    json_content = json.load(f)
                # add reviewer name
                json_content["confirmed_by"] = reviewer_id
                json_content["confirmed_at"] = f"{datetime.now()}"
                # save new json
                with open(os.path.join(json_rootdirectory, "confirmed_jsons", json_filename), 'w') as f:
                    json.dump(json_content, f, indent=4)

        if ctx.triggered_id == "discard-output-button":
            if mode == "confirm":  # discard
                os.remove(os.path.join(json_rootdirectory, "output_jsons", json_filename))
            elif mode == "review":  # discard
                os.remove(os.path.join(json_rootdirectory, "output_jsons", json_filename))
                os.remove(os.path.join(json_rootdirectory, "confirmed_jsons", json_filename))

    # according to review mode or not, determine the list of json files to show and the default (first) to choose
    if mode == "annotate":
        # load list of json that were already annotated
        annotated_json_filenames = os.listdir(os.path.join(json_rootdirectory, "output_jsons"))
        # filter them out from the full list
        tmp_json_list = [e for e in full_json_list if e["json_filename"] not in annotated_json_filenames]
        # color
        json_options = json_file_lists_to_dropdown_options(tmp_json_list, mode=mode)
        # send back
        if len(json_options) > 10:
            n_json_files_found_txt = f"{len(json_options)} unannotated files found, displaying first 10"
            json_options = json_options[:10]
            return False, "", n_json_files_found_txt, json_options, json_options[0]["value"] if len(json_options) > 0 else "", 'SAVE OUTPUT', "", {"font-weight": "bold"}, {"display": 'none'}
        n_json_files_found_txt = f"{len(json_options)} unannotated files found"
        return False, "", n_json_files_found_txt, json_options, json_options[0]["value"] if len(json_options) > 0 else "", 'SAVE OUTPUT', "", {"font-weight": "bold"}, {"display": 'none'}
    elif mode == "confirm":
        # load list of json that were already annotated
        annotated_json_filenames = os.listdir(os.path.join(json_rootdirectory, "output_jsons"))
        confirmed_json_filenames = os.listdir(os.path.join(json_rootdirectory, "confirmed_jsons"))
        # filter them out from the full list
        tmp_json_list = [e for e in full_json_list if (e["json_filename"] in annotated_json_filenames) and (e["json_filename"] not in confirmed_json_filenames)]
        # annotate exclude/doubtful
        for json_info in tmp_json_list:
            with open(os.path.join(json_rootdirectory, "output_jsons", json_info["json_filename"]), "r") as f:
                saved_output_data = json.load(f)
            json_info["doubtful"] = saved_output_data["doubtful"]
            json_info["exclude"] = saved_output_data["exclude"]
        # color
        json_options = json_file_lists_to_dropdown_options(tmp_json_list, mode=mode)
        # send back
        # if len(json_options) > 100:
        #     n_json_files_found_txt = f"{len(json_options)} annotated files found, displaying first 100"
        #     json_options = json_options[:100]
        #     return False, "", n_json_files_found_txt, json_options, json_options[0]["value"] if len(json_options) > 0 else "", 'CONFIRM SAVED OUTPUT', "DISCARD SAVED OUTPUT", {"font-weight": "bold"}, {}
        n_json_files_found_txt = f"{len(json_options)} annotated files found"
        return False, "", n_json_files_found_txt, json_options, json_options[0]["value"] if len(json_options) > 0 else "", 'CONFIRM SAVED OUTPUT', "DISCARD SAVED OUTPUT", {"font-weight": "bold"}, {}
    elif mode == "review":
        # load list of json that were already annotated
        annotated_json_filenames = os.listdir(os.path.join(json_rootdirectory, "output_jsons"))
        confirmed_json_filenames = os.listdir(os.path.join(json_rootdirectory, "confirmed_jsons"))
        # filter them out from the full list
        tmp_json_list = [e for e in full_json_list if (e["json_filename"] in annotated_json_filenames) and (e["json_filename"] in confirmed_json_filenames)]
        # annotate exclude/doubtful
        for json_info in tmp_json_list:
            with open(os.path.join(json_rootdirectory, "output_jsons", json_info["json_filename"]), "r") as f:
                saved_output_data = json.load(f)
            json_info["doubtful"] = saved_output_data["doubtful"]
            json_info["exclude"] = saved_output_data["exclude"]
        # color
        json_options = json_file_lists_to_dropdown_options(tmp_json_list, mode=mode)
        # send back
        n_json_files_found_txt = f"{len(json_options)} confirmed files found"
        return False, "", n_json_files_found_txt, json_options, json_options[0]["value"] if len(json_options) > 0 else "", '', "DISCARD SAVED OUTPUT", {"display": 'none', "font-weight": "bold"}, {}


@callback(
    Output('output-peaks-data-table', 'data', allow_duplicate=True),
    Output('comments-checkbox', 'value'),
    Output('previous-2020-to-peaks-button', 'style'),
    Output('reviewer-id-text', 'children'),
    Output('elp-graph-content-left', 'figure'),
    Output('elp-graph-content-right', 'figure'),
    Output('g-graph-content', 'figure'),
    Output('a-graph-content', 'figure'),
    Output('m-graph-content', 'figure'),
    Output('k-graph-content', 'figure'),
    Output('l-graph-content', 'figure'),
    # Output('ref-graph-content', 'figure'),
    Output('spe-graph-content', 'figure'),
    Output('short-comments-text', 'children'),
    Output('long-comments-text', 'children'),
    Output('old-lemans-text', 'children'),
    Input('json-dropdown-selection', 'value'),
    Input('output-peaks-data-table', 'data'),
    Input('comments-checkbox', 'value'),
    State('mode-radio', 'value'),
    prevent_initial_call=True
)
def update_graph(json_filename, rows, comments_checkbox, mode):
    reviewer_id = ""

    if json_filename == '':
        return ([], [],
                dict(display='none'),
                reviewer_id,
                {}, {}, {}, {}, {}, {}, {}, {},
                # None, None, None, None, None, None, None, None,
                None, None, None)

    saved_output_data = None
    if ctx.triggered_id == "json-dropdown-selection":  # we changed the json file => reset peaks
        comments_checkbox = []  # reset comments
        if mode in ('confirm', 'review'):
            # load previous output
            load_dir = "output_jsons" if (mode == "confirm") else "confirmed_jsons"
            with open(os.path.join(json_rootdirectory, load_dir, json_filename), "r") as f:
                saved_output_data = json.load(f)
            rows = saved_output_data["peak_data"]
            if saved_output_data["doubtful"]:
                comments_checkbox.append("Doubtful")
            if saved_output_data["exclude"]:
                comments_checkbox.append("Exclude")
            reviewer_id = "annotated by " + saved_output_data["annotated_by"]
            if mode == "review":
                reviewer_id += (", confirmed by " + saved_output_data["confirmed_by"])
        else:
            # not in review mode: just reset current peaks
            rows = []

    if len(rows) > 0:
        peak_data = pd.DataFrame(rows)
        peak_data.start = pd.to_numeric(peak_data.start)
        peak_data.end = pd.to_numeric(peak_data.end)
        peak_data = peak_data[peak_data.lc.isin(["K", "L"]) | peak_data.hc.isin(["IgG", "IgA", "IgM"])]
        peak_data = peak_data[~peak_data.start.isna()]
        peak_data = peak_data[~peak_data.end.isna()]
    else:
        peak_data = None

    # load input data
    if mode == "annotate":
        # load input data without any annotations
        with open(os.path.join(json_rootdirectory, "input_jsons", json_filename), "r") as f:
            sample_data = json.load(f)
    elif mode in ('confirm', 'review'):
        if saved_output_data is not None:
            sample_data = saved_output_data
        else:
            # load annotated output file
            load_dir = "output_jsons" if (mode == "confirm") else "confirmed_jsons"
            with open(os.path.join(json_rootdirectory, load_dir, json_filename), "r") as f:
                sample_data = json.load(f)
    else:
        assert False, f"Unknown {mode=}"

    short_comments = sample_data["short_comments"]
    long_comments = sample_data["long_comments"]

    # separate comments by lines
    short_comments = [txt for txt in short_comments.split("\\n") if len(txt) > 0]
    long_comments = [txt for txt in long_comments.split("\\n") if len(txt) > 0]

    # look for keywords
    short_spans = comments_to_spans(short_comments)
    long_spans = comments_to_spans(long_comments)

    # format as paragraphs
    short_comments = [html.P(span, style={"padding": "0", "margin": "0.5rem 0"}) for span in short_spans]
    long_comments = [html.P(span, style={"padding": "0", "margin": "0.5rem 0"}) for span in long_spans]

    # load spectr data
    with open(os.path.join(json_rootdirectory, "spectr_jsons", json_filename), "r") as f:
        sample_spectr_data = json.load(f)
    spectr_elp_preds = np.array(sample_spectr_data["elp_spep_s_predictions"])

    doubtful = "Doubtful" in comments_checkbox
    exclude = "Exclude" in comments_checkbox

    traces = []
    for trace_name in ("ELP", "ELP", "IgG", "IgA", "IgM", "K", "L", "SPE"):
        if sample_data["traces"][trace_name]["exists"]:
            peak_color = "red"
            peak_alt_color = "#cccccc"
            if trace_name == "SPE":
                trace_peak_data = None
                if (sample_data["traces"][trace_name]["peaks"] is not None) and (len(sample_data["traces"][trace_name]["peaks"]) > 0) and (len(sample_data["traces"][trace_name]["peaks"]) % 2 == 0):
                    p_start = sample_data["traces"][trace_name]["peaks"][::2]
                    p_end = sample_data["traces"][trace_name]["peaks"][1::2]
                    trace_peak_data = pd.DataFrame({"start": p_start,"end": p_end, "ontrace": [True, ] * len(p_start)})
                peak_color = "purple"
                peak_alt_color = "purple"
            elif (trace_name not in ["ELP", "Ref"]) and (peak_data is not None):
                trace_peak_data = peak_data.copy()
                trace_peak_data["ontrace"] = (trace_peak_data.lc == trace_name) | (trace_peak_data.hc == trace_name)
            else:
                trace_peak_data = None
            new_trace = get_trace_plot(trace_data=sample_data["traces"][trace_name]["data"],
                                       doubtful=doubtful,
                                       exclude=exclude,
                                       spectr_preds=spectr_elp_preds if trace_name == "ELP" else None,
                                       trace_peak_data=trace_peak_data,
                                       peak_color=peak_color,
                                       peak_alt_color=peak_alt_color)
        else:
            new_trace = {}
        traces.append(new_trace)

    if os.path.exists(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename)):
        with open(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename), "r") as f:
            previous_sample_data = json.load(f)

        old_lemans_class_ = previous_sample_data["groundtruth_class"]
        old_lemans_text = html.Span(f"Previous (2020) data exists: {old_lemans_class_}", style={"color": "red"})
        lemans_button_style = dict()
    else:
        old_lemans_text = None
        lemans_button_style = dict(display='none')

    return rows, comments_checkbox, lemans_button_style, reviewer_id, *traces, short_comments, long_comments, old_lemans_text


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
