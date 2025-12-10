import shutil
import pandas as pd
import os
# import ujson as json
import json
import numpy as np
from datetime import datetime, timedelta
import re
from tqdm import tqdm

root_paths = [r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data",
              r"C:\Users\f.chabrun\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data",
              r"C:\Users\Minion3\Documents\iSPECTR\data", ]
root_path = None
valid_root_path = False
for root_path in root_paths:
    if os.path.exists(root_path):
        valid_root_path = True
        break
assert valid_root_path, "Unable to find data location"

# LE MANS 2025:
# json_rootdirectory = os.path.join(root_path, r"2025\lemans\preannotation")
json_rootdirectory = os.path.join(root_path, r"2025\2025_12_09\lemans\preannotation")


def load_json_data(json_filename, mode):
    load_dir = "input_jsons" if (mode == "annotate") else "output_jsons" if (mode == "confirm") else "confirmed_jsons"
    with open(os.path.join(json_rootdirectory, load_dir, json_filename), "r") as f:
        saved_data = json.load(f)
    return saved_data


def prev_annotations_to_rows(previous_sample_data):
    rows = []
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


def load_previous_2020_annotations(json_filename):
    if os.path.exists(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename)):
        with open(os.path.join(json_rootdirectory, "previous_2020_output_jsons", json_filename), "r") as f:
            saved_data = json.load(f)
        return saved_data
    return None


# %%


def compile_annotations_into_dataframe(period):
    annotations = []
    if period == "new":
        subdir = "input_jsons"
    elif period == "pre2020":
        subdir = "previous_2020_output_jsons"
    else:
        assert False, f"Period {period} not recognized"

    for json_filename in tqdm(os.listdir(os.path.join(json_rootdirectory, subdir))):
        if period == "new":
            try:  # first fetch any confirmed annotations
                json_data = load_json_data(json_filename=json_filename, mode="review")
            except FileNotFoundError:  # if unavailable use unconfirmed annotations
                json_data = load_json_data(json_filename=json_filename, mode="confirm")
            peak_data = json_data['peak_data']
        elif period == "pre2020":
            json_data = load_previous_2020_annotations(json_filename=json_filename)
            peak_data = prev_annotations_to_rows(json_data)

        isotypes = [f"{e['hc']}_{e['lc']}" if (len(e['hc']) > 1) else f"LC_{e['lc']}" for e in peak_data]
        unique_isotypes = list(set(isotypes))
        annotation_summary = "?"
        if len(unique_isotypes) == 0:
            annotation_summary = "Negative"
        elif len(unique_isotypes) == 1:
            annotation_summary = unique_isotypes[0]
        else:
            annotation_summary = []
            unique_isotypes.sort()
            for isotype in unique_isotypes:
                n_isotype = sum([1 for e in isotypes if e == isotype])
                annotation_summary.append(f"{n_isotype}x{isotype}")
            annotation_summary = ",".join(annotation_summary)

        item_annotations = {'Annotation_summary': annotation_summary}
        annotations.append({'aaid': json_filename, **item_annotations})

    annotations = pd.DataFrame(annotations)

    return annotations


old_data = compile_annotations_into_dataframe(period="pre2020")
new_data = compile_annotations_into_dataframe(period="new")

def simplify_annotation_summary(s):
    if s == 'Negative':
        return s
    if s.startswith('LC_'):
        return 'LC'
    if s.startswith('Ig'):
        return s[:3]
    return 'Complex/biclonal'

old_data['Annotation_simple'] = old_data.Annotation_summary.apply(simplify_annotation_summary)
new_data['Annotation_simple'] = new_data.Annotation_summary.apply(simplify_annotation_summary)

old_data.Annotation_simple.value_counts()
# Annotation_simple
# IgG                 1038
# IgM                  405
# IgA                  156
# Complex/biclonal      73
# Negative              52

(100 * old_data.Annotation_simple.value_counts() / len(old_data)).round(1)
# Annotation_simple
# IgG                 60.2
# IgM                 23.5
# IgA                  9.0
# Complex/biclonal     4.2
# Negative             3.0

new_data.Annotation_simple.value_counts()
# Annotation_simple
# IgG                 3295
# IgM                 1092
# IgA                  565
# LC                    45
# Negative             489
# Complex/biclonal     360

(100 * new_data.Annotation_simple.value_counts() / len(new_data)).round(1)
# Annotation_simple
# IgG                 56.4
# IgM                 18.7
# IgA                  9.7
# Negative             8.4
# Complex/biclonal     6.2
# LC                   0.8
