import shutil
import pandas as pd
import os
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
json_rootdirectory = os.path.join(root_path, r"2025\final_json_annotations\2025_12_13\lemans")


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
        # open json metadata
        item_annotations = {'AAID': json_filename, 'Doubtful': '', 'Exclude': ''}
        if period == "new":
            try:  # first fetch any confirmed annotations
                json_data = load_json_data(json_filename=json_filename, mode="review")
            except FileNotFoundError:  # if unavailable use unconfirmed annotations
                json_data = load_json_data(json_filename=json_filename, mode="confirm")
            peak_data = json_data['peak_data']
            item_annotations['Doubtful'] = json_data['doubtful']
            item_annotations['Exclude'] = json_data['exclude']
        elif period == "pre2020":
            json_data = load_previous_2020_annotations(json_filename=json_filename)
            peak_data = prev_annotations_to_rows(json_data)

        # read isotype data
        isotypes = [f"{e['hc']}_{e['lc']}" if (len(e['hc']) > 1) else f"LC_{e['lc']}" for e in peak_data]
        unique_isotypes = list(set(isotypes))
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
        # store
        item_annotations['Annotation_summary'] = annotation_summary

        # save
        annotations.append(item_annotations)

    annotations = pd.DataFrame(annotations)

    return annotations


def simplify_annotation_summary(s):
    if s == 'Negative':
        return s
    if s.startswith('LC_'):
        return 'LC'
    if s.startswith('Ig'):
        return s[:3]
    return 'Complex/biclonal'


old_data = compile_annotations_into_dataframe(period="pre2020")
new_data = compile_annotations_into_dataframe(period="new")

old_data['Annotation_simple'] = old_data.Annotation_summary.apply(simplify_annotation_summary)
new_data['Annotation_simple'] = new_data.Annotation_summary.apply(simplify_annotation_summary)

def print_count_stats(df_to_process, column):
    print(pd.concat([df_to_process[column].value_counts(),
               (100 * df_to_process[column].value_counts(normalize=True)).round(1)], axis=1).apply(lambda row: f"n={row[0]:.0f} ({row[1]}%)", axis=1))


print_count_stats(old_data, "Annotation_simple")
# IgG                 n=1038 (60.2%)
# IgM                  n=405 (23.5%)
# IgA                   n=156 (9.0%)
# Complex/biclonal       n=73 (4.2%)
# Negative               n=52 (3.0%)

print_count_stats(new_data, "Annotation_simple")
# IgG                 n=3296 (56.4%)
# IgM                 n=1094 (18.7%)
# IgA                   n=569 (9.7%)
# Negative              n=476 (8.1%)
# Complex/biclonal      n=364 (6.2%)
# LC                     n=47 (0.8%)

# some stats about doubtful/to exclude
print_count_stats(new_data, "Doubtful")
# Doubtful
# False    n=5007 (85.6%)
# True      n=839 (14.4%)

print_count_stats(new_data, "Exclude")
# Exclude
# False    n=5842 (99.9%)
# True         n=4 (0.1%)

# merge both
final_data = new_data.merge(old_data[['AAID', 'Annotation_summary', 'Annotation_simple']].rename(columns={'Annotation_summary': 'OLD_Annotation_summary', 'Annotation_simple': 'OLD_Annotation_simple'}), how="left", on="AAID")

# export the df for further analysis
final_data.to_excel(r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data\2025\m1\annotations_extraction.xlsx")
