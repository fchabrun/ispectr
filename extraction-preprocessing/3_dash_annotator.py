"""
Code Floris 2025
"""

from dash import Dash, html, dcc, callback, Output, Input, State, dash_table, ctx, no_update
# import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os
try:
    import ujson as json
    print("imported ujson as json")
except:
    print(f"ujson not found, importing json instead")
    import json
import numpy as np
from datetime import datetime, timedelta
import re
import time
from tqdm import tqdm

STORE_H5 = False
if STORE_H5:
    import h5py

# OVERWRITE_OUTPUT_JSON_WITH_NEW_INPUT_DATA = False
root_paths = [r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data",
              r"C:\Users\f.chabrun\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data",
              r"C:\Users\Minion3\Documents\iSPECTR\data", ]
valid_root_path = False
for root_path in root_paths:
    if os.path.exists(root_path):
        valid_root_path = True
        break
assert valid_root_path, "Unable to find data location"

# CAPE TOWN 2025:
json_rootdirectory = os.path.join(root_path, r"2025\preannotation\2025_12_09\capetown\preannotation")

# FIRST REVIEW
# LE MANS 2025:
# json_rootdirectory = os.path.join(root_path, r"2025\lemans\preannotation")
# SECOND REVIEW (inconsistent samples only)
# json_rootdirectory = os.path.join(root_path, r"2025\2025_12_10\de")

# OVERWRITE_OUTPUT_JSON_WITH_NEW_INPUT_DATA = False
# if OVERWRITE_OUTPUT_JSON_WITH_NEW_INPUT_DATA:
#     from tqdm import tqdm
#
#     for json_filename in tqdm(os.listdir(os.path.join(json_rootdirectory, "output_jsons"))):
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
#         input_json_content["annotated_at"] = output_json_content["annotated_at"]
#         # input_json_content["confirmed_by"] = output_json_content["confirmed_by"]  # not in output files (only confirmed files)
#         # input_json_content["confirmed_at"] = output_json_content["confirmed_at"]  # not in output files (only confirmed files)
#         # overwrite previous output (annotated) json
#         with open(os.path.join(json_rootdirectory, "output_jsons", json_filename), 'w') as f:
#             json.dump(input_json_content, f, indent=4)

# ADD_PLUS_TWO_TO_AUTO_PEAKS = False
# if ADD_PLUS_TWO_TO_AUTO_PEAKS:
#     from tqdm import tqdm
#
#     for json_filename in tqdm(os.listdir(os.path.join(json_rootdirectory, "output_jsons"))):
#         # load input json
#         with open(os.path.join(json_rootdirectory, "spectr_jsons", json_filename), "r") as f:
#             spectr_json_content = json.load(f)
#         # load output (annotated) json
#         with open(os.path.join(json_rootdirectory, "output_jsons", json_filename), "r") as f:
#             output_json_content = json.load(f)
#
#         spectr_json_content["elp_spep_s_predictions"]
#         spectr_elp_preds = np.array(spectr_json_content["elp_spep_s_predictions"])
#         peak_starts, peak_ends = gate_peaks_from_spectr_preds(spectr_elp_preds)
#
#         # now compare with peak ends annotations
#         # automatically add +2
#         edits = False
#         for peak_info in output_json_content["peak_data"]:
#             if peak_info["end"] in peak_ends:
#                 peak_info["end"] += 2
#                 edits = True
#
#         if edits:
#             output_json_content["peak_ends_updated_plus2"] = True
#         else:
#             output_json_content["peak_ends_updated_plus2"] = False
#         # overwrite previous output (annotated) json
#         with open(os.path.join(json_rootdirectory, "output_jsons", json_filename), 'w') as f:
#             json.dump(output_json_content, f, indent=4)

# ADD_PLUS_TWO_TO_PREV_PEAKS = False
# if ADD_PLUS_TWO_TO_PREV_PEAKS:
#
#     def gate_peaks_from_spectr_preds(spectr_elp_preds):
#         # clean and get predicted positions
#         spectr_elp_preds = (spectr_elp_preds > .1) * 1
#         # compute diff (increase/decrease)
#         spectr_elp_preds_diff = np.diff(spectr_elp_preds)
#         # compute start and end positions
#         peak_starts = np.where(spectr_elp_preds_diff == 1)[0]
#         peak_ends = np.where(spectr_elp_preds_diff == -1)[0]
#         out_peak_starts, out_peak_ends = [], []
#         if len(peak_starts) == len(peak_ends):  # only act if n(starts) == n(ends)
#             for start, end in zip(peak_starts, peak_ends):
#                 if end <= start:  # prevent errors
#                     continue
#                 if end > 299:  # prevent peaks outside of the spep
#                     continue
#                 if start < 150:  # prevent peaks too early (e.g. albumin)
#                     continue
#                 out_peak_starts.append(int(start))
#                 out_peak_ends.append(int(end))
#         return out_peak_starts, out_peak_ends
#
#     modified = []
#     for json_filename in tqdm(os.listdir(os.path.join(json_rootdirectory, "output_jsons"))):
#         # load input json
#         if not os.path.exists(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename)):
#             continue
#         with open(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename), "r") as f:
#             prev_json_content = json.load(f)
#         # load output (annotated) json
#         with open(os.path.join(json_rootdirectory, "output_jsons", json_filename), "r") as f:
#             output_json_content = json.load(f)
#
#         prev_peak_locs = []
#         for isotype in ["IgG", "IgA", "IgM", "K", "L"]:
#             iso_trace = np.array(prev_json_content["groundtruth_maps"][isotype])
#             peak_starts, peak_ends = gate_peaks_from_spectr_preds(iso_trace)
#             for peak_start, peak_end in zip(peak_starts, peak_ends):
#                 prev_peak_locs.append((peak_start, peak_end))
#
#         # now compare with peak ends annotations
#         # automatically add +2
#         edits = False
#         for peak_info in output_json_content["peak_data"]:
#             for prev_peak_info in prev_peak_locs:
#                 if (peak_info["start"] == prev_peak_info[0]) and (peak_info["end"] == prev_peak_info[1]):
#                     peak_info["end"] += 2
#                     edits = True
#
#         if edits:
#             modified.append(json_filename)
#             output_json_content["prev_peak_ends_updated_plus2"] = True
#         else:
#             output_json_content["prev_peak_ends_updated_plus2"] = False
#         # overwrite previous output (annotated) json
#         with open(os.path.join(json_rootdirectory, "output_jsons", json_filename), 'w') as f:
#             json.dump(output_json_content, f, indent=4)
#
#     len(modified)  # 84

# count doubtful
# annotated_json_filenames = os.listdir(os.path.join(json_rootdirectory, "output_jsons"))
# n_doubtful, n_confident = 0, 0
# for json_fn in tqdm(annotated_json_filenames):
#     with open(os.path.join(json_rootdirectory, "output_jsons", json_fn), "r") as f:
#         buffer_data = json.load(f)
#     if buffer_data["doubtful"]:
#         n_doubtful += 1
#     else:
#         n_confident += 1
# n_doubtful  # 540
# n_doubtful / (n_doubtful + n_confident)  # 18%

SHOW_ELAPSED_TIMES = True


class time_marker():
    def __init__(self, task, auto_start=True):
        if auto_start:
            self.start()
        self.task = task


    def start(self):
        self.t0 = time.time()
        self.t_elapsed = -1


    def end(self):
        self.t_elapsed = time.time() - self.t0


    def print(self, auto_end=True):
        if auto_end:
            self.end()
        if SHOW_ELAPSED_TIMES:
            if self.t_elapsed > 0:
                if self.t_elapsed < 1:
                    print(f"[{self.task}]: completed in {1000*self.t_elapsed:.0f} milliseconds")
                else:
                    print(f"[{self.task}]: completed in {self.t_elapsed:.2f} seconds")


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
if os.path.exists(os.path.join(json_rootdirectory, "previous_2020_output_jsons")):
    existingpreviousdata_json_filenames = os.listdir(os.path.join(json_rootdirectory, "previous_2020_output_jsons"))
else:
    existingpreviousdata_json_filenames = []
# add info of whether the files have prior pre-2020 annotations
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
        dcc.Dropdown([], "", id='reviewer-selection'),
        dcc.Checklist([{'label': 'Only doubtful/exclude', 'value': 'de_only'}],
                       ['de_only'], id="de-only-checkbox"),
        html.Br(),
        html.Div(id='n-json-files-found'),
        dcc.Dropdown([], "", id='json-dropdown-selection'),
        html.Div(id='reviewer-id-text', style={'font-style': 'italic'}),
        html.Hr(),
        html.Button('Populate from PREVIOUS (2020)', id='previous-2020-to-peaks-button', n_clicks=0),
        html.Button('Populate from SPECTR', id='spectr-to-peaks-button', n_clicks=0),
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
        dcc.Checklist([{'label': 'Doubtful', 'value': 'Doubtful', 'disabled': False},
                       {'label': 'Exclude', 'value': 'Exclude', 'disabled': False}], [], id="comments-checkbox"),
        html.Br(),
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
        html.Div([
            html.P(id="short-comments-text"),
            html.P(id="long-comments-text"),
            html.P(id="other-short-comments-text"),
            html.P(id="other-long-comments-text"),
        ], style={"overflow-y": "auto", "height": "20rem"}),
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

content = html.Div([graphs_layout,],
                   id="page-content",
                   style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"),
                       dcc.Loading([sidebar, content],
                                   id="general-loading",
                                   type="default",
                                   overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                                   ),
                       ])


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
    fig.update_traces(line=dict(width=1), selector=dict(name='Relative intensity'))
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


def comments_to_spans(comments, prefix_name, keyword="(IgG|IgA|IgM|kappa|Kappa|lambda|Lambda)"):
    if prefix_name is None:
        spans = []
    else:
        spans = [html.Span(prefix_name, style={"color": "black", "font-style": "italic"}),]
    for com_txt in comments:
        new_span = [html.Span(txt, style={"color": ("red" if i % 2 == 1 else "black")}) for i, txt in enumerate(re.split(keyword, com_txt))]
        spans.append(new_span)
    return spans


def look_for_hc_lc_in_comments(comments, hc_keys="(IgG|IgA|IgM)", lc_keys="([kK][aA][pP][pP][aA]|[lL][aA][mM][bB][dD][aA])"):
    # get list of unique HC
    found_hc = list(set(re.findall(hc_keys, comments)))
    # get list of unique LC
    found_lc = list(set([lc[0].upper() for lc in re.findall(lc_keys, comments)]))
    # return
    return found_hc, found_lc


def qc_peak_info(rows, MIN_PEAK_POS=140, MAX_PEAK_POS=299):
    # make sure types are OK
    for row in rows:
        row["start"] = int(row["start"])  # make sure peak locators are numeric
        row["end"] = int(row["end"])
        if (row["hc"] is None):  # replace None by ""
            row["hc"] = ""
        # do not replace lc: should never be missing (peak can be FLC but not HC only (in this context)
    # qc values
    for row in rows:
        if (row["start"] < MIN_PEAK_POS) or (row["start"] < MIN_PEAK_POS) or (row["end"] > MAX_PEAK_POS) or (row["end"] > MAX_PEAK_POS):
            return None, "Unable to save annotations: invalid peak position"
        if int(row["start"]) >= int(row["end"]):
            return None, "Unable to save annotations: invalid peak size"
        if row["hc"] not in ["IgG", "IgA", "IgM", ""]:
            return None, "Unable to save annotations: invalid peak HC: " + f'"{row["hc"]}"'
        if row["lc"] not in ["K", "L"]:
            return None, "Unable to save annotations: invalid peak LC: " + f'"{row["lc"]}"'
        if (row["hc"] == "") and (row["lc"] == ""):
            return None, "Unable to save annotations: HC and LC cannot be both missing"
    # automatically reorder by start position
    rows = [rows[i] for i in np.argsort([r["start"] for r in rows])]
    # check if overlapping  # now authorized on august 1st due to SA sample SA02611879
    # if not np.all(np.diff([v for row in rows for v in [row['start'], row['end']]]) > 0):
    #     return None, "Unable to save annotations: peak positions overlapping"
    # send back clean data
    return rows, None


def gate_peaks_from_spectr_preds_update_plus2(spectr_elp_preds):
    # clean and get predicted positions
    spectr_elp_preds = (spectr_elp_preds > .1) * 1
    # compute diff (increase/decrease)
    spectr_elp_preds_diff = np.diff(spectr_elp_preds)
    # compute start and end positions
    peak_starts = np.where(spectr_elp_preds_diff == 1)[0]
    peak_ends = np.where(spectr_elp_preds_diff == -1)[0] + 2
    out_peak_starts, out_peak_ends = [], []
    if len(peak_starts) == len(peak_ends):  # only act if n(starts) == n(ends)
        for start, end in zip(peak_starts, peak_ends):
            if end <= start:  # prevent errors
                continue
            if end > 299:  # prevent peaks outside of the spep
                continue
            if start < 150:  # prevent peaks too early (e.g. albumin)
                continue
            out_peak_starts.append(int(start))
            out_peak_ends.append(int(end))
    return out_peak_starts, out_peak_ends


def save_output_data(json_content, mode, reviewer_id):
    print(f"save_output_data called with {mode=}")
    if STORE_H5:
        assert False, f"Unhandled {STORE_H5=} for saving output data"
    else:
        if mode == "annotate":
            json_content["annotated_by"] = reviewer_id
            json_content["annotated_at"] = f"{datetime.now()}"
            # not working anymore
            fn = os.path.join(json_rootdirectory, "output_jsons", json_content['aaid'] + ".json")
            print(f"saving to {fn}")
            with open(fn, 'w') as f:
                json.dump(json_content, f, indent=4)
        elif mode == "confirm":
            json_content["confirmed_by"] = reviewer_id
            json_content["confirmed_at"] = f"{datetime.now()}"
            fn = os.path.join(json_rootdirectory, "confirmed_jsons", json_content['aaid'] + ".json")
            print(f"saving to {fn}")
            with open(fn, 'w') as f:
                json.dump(json_content, f, indent=4)
        else:
            assert False, f"Unhandled {mode=} for saving output data"


def delete_stored_data(json_filename, mode):
    if STORE_H5:
        assert False, f"Unhandled {STORE_H5=} for deleting output data"
    else:
        if mode == "confirm":  # discard
            os.remove(os.path.join(json_rootdirectory, "output_jsons", json_filename))
        elif mode == "review":  # discard
            os.remove(os.path.join(json_rootdirectory, "output_jsons", json_filename))
            os.remove(os.path.join(json_rootdirectory, "confirmed_jsons", json_filename))
        else:
            assert False, f"Unhandled {mode=} for deleting output data"


def reload_output_data(json_filename, source):
    if STORE_H5:
        assert False, f"Unhandled {STORE_H5=} for reloading output data"
    else:
        if source == "unannotated":
            with open(os.path.join(json_rootdirectory, "input_jsons", json_filename), "r") as f:
                json_content = json.load(f)
            return json_content
        elif source == "annotated":
            with open(os.path.join(json_rootdirectory, "output_jsons", json_filename), "r") as f:
                json_content = json.load(f)
            return json_content
        else:
            assert False, f"{source=} is not a valid source"


def parse_doubtul_excluded_files_and_reviewers_list(tmp_json_list, mode):
    if STORE_H5:
        assert False, f"Unhandled {STORE_H5=} for parsing output data"
    else:
        reviewers_list = ["(all)", ]
        if mode == "confirm":
            json_sub_path = "output_jsons"
            reviewer_key = "annotated_by"
        elif mode == "review":
            json_sub_path = "confirmed_jsons"
            reviewer_key = "confirmed_by"
        else:
            assert False, f"{mode=} is not a valid mode"

        for json_info in tmp_json_list:
            # new 27.11: only check if different
            if ("doubtful" not in json_info.keys()) or ("exclude" not in json_info.keys()) or (reviewer_key not in json_info.keys()):
                with open(os.path.join(json_rootdirectory, json_sub_path, json_info["json_filename"]), "r") as f:
                    saved_output_data = json.load(f)
                json_info["doubtful"] = saved_output_data["doubtful"]
                json_info["exclude"] = saved_output_data["exclude"]
                # check annotator and retain list
                json_info[reviewer_key] = saved_output_data[reviewer_key]
            if json_info[reviewer_key] not in reviewers_list:
                reviewers_list.append(json_info[reviewer_key])

        return tmp_json_list, reviewers_list


def get_list_of_annotated_json_files(only_confirmed=False):
    if STORE_H5:
        assert False, f"Unhandled {STORE_H5=} for listing output data"
    else:
        if only_confirmed:
            return os.listdir(os.path.join(json_rootdirectory, "confirmed_jsons"))
        return os.listdir(os.path.join(json_rootdirectory, "output_jsons"))


def load_json_data(json_filename, mode):
    if STORE_H5:
        assert False, f"Unhandled {STORE_H5=} for loading output data"
    else:
        load_dir = "input_jsons" if (mode == "annotate") else "output_jsons" if (mode == "confirm") else "confirmed_jsons"
        with open(os.path.join(json_rootdirectory, load_dir, json_filename), "r") as f:
            saved_data = json.load(f)
        return saved_data


def load_spectr_data(json_filename):
    with open(os.path.join(json_rootdirectory, "spectr_jsons", json_filename), "r") as f:
        saved_data = json.load(f)
    return saved_data


def load_previous_2020_annotations(json_filename):
    if os.path.exists(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename)):
        with open(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename), "r") as f:
            saved_data = json.load(f)
        return saved_data
    return None


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
    Output('reviewer-selection', 'options'),
    Output('reviewer-selection', 'value'),
    Output('reviewer-selection', 'style'),
    Output("output-peaks-data-table", "editable"),
    Output("output-peaks-data-table", "row_deletable"),
    Output('previous-2020-to-peaks-button', 'disabled'),
    Output('spectr-to-peaks-button', 'disabled'),
    Output('add-peak-button', 'disabled'),
    Output('comments-checkbox', 'options'),
    Output("general-loading", "children"),
    Output('de-only-checkbox', 'style'),
    Input("general-loading", "value"),
    Input('save-output-button', 'n_clicks'),
    Input('discard-output-button', 'n_clicks'),
    Input('mode-radio', 'value'),
    Input('reviewer-selection', 'value'),
    Input('de-only-checkbox', 'value'),
    State('output-peaks-data-table', 'data'),
    State('json-dropdown-selection', 'value'),
    State('comments-checkbox', 'value'),
    State('input-reviewer', 'value'),
)
def save_and_update_json_files_list(loading_value, n_clicks_save, n_clicks_discard, mode, reviewer_selected, de_only_checkbox,
                                    rows, json_filename, comments_checkbox, reviewer_id):
    sub0_timelapse = time_marker("UI refresh/global")
    if mode in ("annotate", "confirm", ):
        updates_disable_ret = [True, True, False, False, False,
                               [{'label': 'Doubtful', 'value': 'Doubtful', 'disabled': False},
                                {'label': 'Exclude', 'value': 'Exclude', 'disabled': False}],
                               ]
    else:
        updates_disable_ret = [False, False, True, True, True,
                               [{'label': 'Doubtful', 'value': 'Doubtful', 'disabled': True},
                                {'label': 'Exclude', 'value': 'Exclude', 'disabled': True}],
                               ]

    # if save => save
    if (n_clicks_save > 0) or (n_clicks_discard > 0):
        sub1_timelapse = time_marker("UI refresh/button push")
        if ctx.triggered_id == "save-output-button":
            # reject if annotations are not OK
            if (reviewer_id is None) or (reviewer_id == ""):
                error_dialog_msg = "Please enter a reviewer name"
                sub0_timelapse.print()
                return True, error_dialog_msg, no_update, no_update, json_filename, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

            # qc and clean peak data
            peak_data, peak_data_err = qc_peak_info(rows)
            if peak_data_err is not None:
                sub0_timelapse.print()
                return True, peak_data_err, no_update, no_update, json_filename, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

            if mode == "annotate":
                # reload json input file and add new annotations to it
                json_content = reload_output_data(json_filename=json_filename, source="unannotated")
                # create json content
                json_content["peak_data"] = peak_data
                json_content["doubtful"] = "Doubtful" in comments_checkbox
                json_content["exclude"] = "Exclude" in comments_checkbox
                # save
                save_output_data(json_content=json_content, mode=mode, reviewer_id=reviewer_id)
            elif mode == "confirm":  # copy in validated folder
                # OLD VERSION -- SIMPLY COPY
                # shutil.copy(os.path.join(json_rootdirectory, "output_jsons", json_filename),
                #             os.path.join(json_rootdirectory, "confirmed_jsons", json_filename))
                # LESS OLD VERSION -- REOPEN AND ADD REVIEWER ID AND REVIEWING DATE
                # new with reviewer name
                # reload json input file and add new annotations to it
                json_content = reload_output_data(json_filename=json_filename, source="annotated")
                # EVEN NEWER: JUST RECOMPUTE IN CASE WE CHANGED ANYTHING (AND ALLOW MODIFICATIONS)
                json_content["peak_data"] = peak_data  # overwrite any changes
                json_content["doubtful"] = "Doubtful" in comments_checkbox  # overwrite any changes
                json_content["exclude"] = "Exclude" in comments_checkbox  # overwrite any changes
                # save
                save_output_data(json_content=json_content, mode=mode, reviewer_id=reviewer_id)

        if ctx.triggered_id == "discard-output-button":
            delete_stored_data(json_filename=json_filename, mode=mode)
        sub1_timelapse.print()

    # according to review mode or not, determine the list of json files to show and the default (first) to choose
    if mode == "annotate":
        # load list of json that were already annotated
        sub1_timelapse = time_marker("UI refresh/list files (annotate)")
        annotated_json_filenames = get_list_of_annotated_json_files()
        sub1_timelapse.print()
        # filter them out from the full list
        sub1_timelapse = time_marker("UI refresh/file list -> UI (annotate)")
        tmp_json_list = [e for e in full_json_list if e["json_filename"] not in annotated_json_filenames]
        sub1_timelapse.print()
        # color
        sub1_timelapse = time_marker("UI refresh/file list -> color (annotate)")
        json_options = json_file_lists_to_dropdown_options(tmp_json_list, mode=mode)
        sub1_timelapse.print()
        # send back
        LIMIT_TO_N_FILES = 10  # we can set 10 max displayed
        if len(json_options) > LIMIT_TO_N_FILES:
            n_json_files_found_txt = f"{len(json_options)} unannotated files found, displaying first {LIMIT_TO_N_FILES}"
            json_options = json_options[:LIMIT_TO_N_FILES]
        else:
            n_json_files_found_txt = f"{len(json_options)} unannotated files found"
        sub0_timelapse.print()
        return (False, "", n_json_files_found_txt, json_options,
                json_options[0]["value"] if len(json_options) > 0 else "", 'SAVE ANNOTATIONS', "",
                {"font-weight": "bold"}, {"display": 'none'}, [], "", {"display": 'none'},
                *updates_disable_ret, no_update, {"display": 'none'})
    elif mode == "confirm":
        # load list of json that were already annotated
        sub1_timelapse = time_marker("UI refresh/list output files (confirm)")
        annotated_json_filenames = get_list_of_annotated_json_files(only_confirmed=False)
        sub1_timelapse.print()
        sub1_timelapse = time_marker("UI refresh/list confirmed files (confirm)")
        confirmed_json_filenames = get_list_of_annotated_json_files(only_confirmed=True)
        sub1_timelapse.print()
        # filter them out from the full list
        tmp_json_list = [e for e in full_json_list if (e["json_filename"] in annotated_json_filenames) and (e["json_filename"] not in confirmed_json_filenames)]
        # annotate exclude/doubtful
        sub1_timelapse = time_marker("UI refresh/fetch json data (confirm)")
        tmp_json_list, reviewers_list = parse_doubtul_excluded_files_and_reviewers_list(tmp_json_list=tmp_json_list, mode=mode)
        sub1_timelapse.print()
        if "de_only" in de_only_checkbox:
            tmp_json_list = [e for e in tmp_json_list if (e["doubtful"] or e["exclude"])]
        if (reviewer_selected == "") or reviewer_selected not in reviewers_list:
            if "Master 1" in reviewers_list:
                reviewer_selected = "Master 1"  # default -> Master 1, then (all)
            else:
                reviewer_selected = "(all)"  # default -> (all)
        # keep only selected reviewers
        if reviewer_selected != "(all)":
            tmp_json_list = [e for e in tmp_json_list if e["annotated_by"] == reviewer_selected]
        # color
        sub1_timelapse = time_marker("UI refresh/file list -> color (confirm)")
        json_options = json_file_lists_to_dropdown_options(tmp_json_list, mode=mode)
        sub1_timelapse.print()
        # send back
        LIMIT_TO_N_FILES = 9999  # we can set 10
        if len(json_options) > LIMIT_TO_N_FILES:
            n_json_files_found_txt = f"{len(json_options)} annotated files found, displaying first {LIMIT_TO_N_FILES}"
            json_options = json_options[:LIMIT_TO_N_FILES]
        else:
            n_json_files_found_txt = f"{len(json_options)} annotated files found"
        sub0_timelapse.print()
        # old version -> no changes allowed while reviewing existing annotations
        # return (False, "", n_json_files_found_txt, json_options,
        #         json_options[0]["value"] if len(json_options) > 0 else "", 'CONFIRM SAVED OUTPUT', "DISCARD SAVED OUTPUT",
        #         {"font-weight": "bold"}, {}, reviewers_list, reviewer_selected, {},
        #         *updates_disable_ret, no_update, {})
        return (False, "", n_json_files_found_txt, json_options,
                json_options[0]["value"] if len(json_options) > 0 else "", 'OVERWRITE AND CONFIRM ANNOTATIONS', "DISCARD SAVED ANNOTATIONS",
                {"font-weight": "bold"}, {}, reviewers_list, reviewer_selected, {},
                *updates_disable_ret, no_update, {})
    elif mode == "review":
        # load list of json that were already annotated
        annotated_json_filenames = get_list_of_annotated_json_files(only_confirmed=False)
        confirmed_json_filenames = get_list_of_annotated_json_files(only_confirmed=True)
        # filter them out from the full list
        tmp_json_list = [e for e in full_json_list if (e["json_filename"] in annotated_json_filenames) and (e["json_filename"] in confirmed_json_filenames)]
        # annotate exclude/doubtful
        tmp_json_list, reviewers_list = parse_doubtul_excluded_files_and_reviewers_list(tmp_json_list=tmp_json_list, mode=mode)
        if "de_only" in de_only_checkbox:
            tmp_json_list = [e for e in tmp_json_list if (e["doubtful"] or e["exclude"])]
        if (reviewer_selected == "") or reviewer_selected not in reviewers_list:
            if "Master 1" in reviewers_list:
                reviewer_selected = "Master 1"  # default -> Master 1, then (all)
            else:
                reviewer_selected = "(all)"  # default -> (all)
        # keep only selected reviewers
        if reviewer_selected != "(all)":
            tmp_json_list = [e for e in tmp_json_list if e["confirmed_by"] == reviewer_selected]
        # color
        json_options = json_file_lists_to_dropdown_options(tmp_json_list, mode=mode)
        # send back
        n_json_files_found_txt = f"{len(json_options)} confirmed files found"
        sub0_timelapse.print()
        return (False, "", n_json_files_found_txt, json_options,
                json_options[0]["value"] if len(json_options) > 0 else "", '', "DISCARD ALL ANNOTATIONS",
                {"display": 'none', "font-weight": "bold"}, {}, reviewers_list, reviewer_selected, {},
                *updates_disable_ret, no_update, {})


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
    Output('other-short-comments-text', 'children'),
    Output('other-long-comments-text', 'children'),
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
                None, None, None, None, None)

    saved_output_data = None
    if ctx.triggered_id == "json-dropdown-selection":  # we changed the json file => reset peaks
        comments_checkbox = []  # reset comments
        if mode in ('confirm', 'review'):
            # load previous output
            saved_output_data = load_json_data(json_filename=json_filename, mode=mode)
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
    if saved_output_data is not None:
        # reuse
        sample_data = saved_output_data
    else:
        # load annotated output file
        sample_data = load_json_data(json_filename=json_filename, mode=mode)

    comments_paragraph_list = []
    for com_field, com_prefix in zip(["short_comments", "long_comments", "patient_other_short_comments", "patient_other_long_comments"],
                             ["(SPEP)", "(IT)", "(PAID SPEP)", "(PAID IT)"]):
        comments_text = None
        if com_field in sample_data.keys():
            comments_text = sample_data[com_field]

        if comments_text is None:
            comments_text = "---No data---"

        # separate comments by lines
        comments_lines = [txt for txt in comments_text.split("\\n") if len(txt) > 0]

        # look for keywords
        comments_spans = comments_to_spans(comments_lines, prefix_name=com_prefix)

        # format as paragraphs
        comments_paragraph = [html.P(span, style={"padding": "0", "margin": "0.5rem 0"}) for span in comments_spans]

        # add to list
        comments_paragraph_list.append(comments_paragraph)

    # load spectr data
    sample_spectr_data = load_spectr_data(json_filename=json_filename)
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

    previous_sample_data = load_previous_2020_annotations(json_filename=json_filename)
    if previous_sample_data is not None:
        old_lemans_class_ = previous_sample_data["groundtruth_class"]
        old_lemans_text = html.Span(f"Previous (2020) data exists: {old_lemans_class_}", style={"color": "red"})
        lemans_button_style = dict()
    else:
        old_lemans_text = None
        lemans_button_style = dict(display='none')

    return rows, comments_checkbox, lemans_button_style, reviewer_id, *traces, *comments_paragraph_list, old_lemans_text


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
def fill_peak_rows(add_n_clicks, spectr_n_clicks, previous_2020_n_clicks, json_filename, rows, columns):

    if add_n_clicks > 0:
        if ctx.triggered_id == "add-peak-button":  # we changed the json file => reset peaks
            rows.append({c['id']: '' for c in columns})

    if spectr_n_clicks > 0:
        if ctx.triggered_id == "spectr-to-peaks-button":  # we changed the json file => reset peaks
            rows = []

            # load spectr data
            sample_spectr_data = load_spectr_data(json_filename=json_filename)
            # extract prediction map
            spectr_elp_preds = np.array(sample_spectr_data["elp_spep_s_predictions"])
            # # clean and get predicted positions
            # spectr_elp_preds = (spectr_elp_preds > .1) * 1
            # # compute diff (increase/decrease)
            # spectr_elp_preds_diff = np.diff(spectr_elp_preds)
            # # compute start and end positions
            # peak_starts = np.where(spectr_elp_preds_diff == 1)[0]
            # peak_ends = np.where(spectr_elp_preds_diff == -1)[0]
            # if len(peak_starts) == len(peak_ends):  # only act if n(starts) == n(ends)
            #     for start, end in zip(peak_starts, peak_ends):
            #         if end <= start:  # prevent errors
            #             continue
            #         if end > 299:  # prevent peaks outside of the spep
            #             continue
            #         if start < 150:  # prevent peaks too early (e.g. albumin)
            #             continue
            #         rows.append({'start': start, 'end': end, 'hc': "", 'lc': ""})

            peak_starts, peak_ends = gate_peaks_from_spectr_preds_update_plus2(spectr_elp_preds)  # edit 22 aug 2025
            for start, end in zip(peak_starts, peak_ends):
                rows.append({'start': start, 'end': end, 'hc': "", 'lc': ""})

            # try to automatically populate HC and LC (if only 1 type of HC and LC mentioned in the comments) new 30_07_2025
            sample_data = load_json_data(json_filename, mode="annotate")

            comments = sample_data["short_comments"] + " " + sample_data["long_comments"]
            found_hc, found_lc = look_for_hc_lc_in_comments(comments)

            if (len(found_hc) == 1) and (len(found_lc) == 1):  # exactly 1 HC and 1 LC
                found_hc = found_hc[0]
                found_lc = found_lc[0]
                for row in rows:
                    row["hc"] = found_hc
                    row["lc"] = found_lc

    if previous_2020_n_clicks:
        if ctx.triggered_id == "previous-2020-to-peaks-button":  # we changed the json file => reset peaks
            rows = []

            previous_sample_data = load_previous_2020_annotations(json_filename=json_filename)
            if previous_sample_data is not None:

                for isotype in ["IgG", "IgA", "IgM", "K", "L"]:
                    iso_trace = np.array(previous_sample_data["groundtruth_maps"][isotype])
                    # compute diff (increase/decrease)
                    iso_trace_diff = np.diff(iso_trace)
                    # compute start and end positions
                    peak_starts = np.where(iso_trace_diff == 1)[0]
                    peak_ends = np.where(iso_trace_diff == -1)[0] + 2  # edit 22 aug 2025
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
