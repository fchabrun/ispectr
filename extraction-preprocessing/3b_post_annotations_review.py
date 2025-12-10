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


annotations = []
for json_filename in tqdm(os.listdir(os.path.join(json_rootdirectory, "input_jsons"))):
    sample_data = None
    # confirmed = False
    try:  # first fetch any confirmed annotations
        sample_data = load_json_data(json_filename=json_filename, mode="review")
        # confirmed = True
    except FileNotFoundError:  # if unavailable use unconfirmed annotations
        sample_data = load_json_data(json_filename=json_filename, mode="confirm")

    # if previous (2020) annotations exist, compare new annotations to previous
    prev_annotations = load_previous_2020_annotations(json_filename=json_filename)
    any_diff = False
    if prev_annotations is not None:
        prev_peak_data = prev_annotations_to_rows(prev_annotations)
        if len(prev_peak_data) != len(sample_data['peak_data']):
            any_diff = True
        else:
            for prev_e, new_e in zip(prev_peak_data, sample_data['peak_data']):
                for k in ['start', 'end', 'hc', 'lc']:
                    if prev_e[k] != new_e[k]:
                        any_diff = True
                        break
                if any_diff:
                    break

    # if comment exists, compare suggested annotations
    antibodies_mentioned = {}
    for lc in ('kappa', 'lambda'):
        found_full_size_antibodies = False
        for hc in ('gam'):
            n = len(re.findall(f"ig{hc} [aà] cha[iî]nes l[eé]g[eè]res {lc}", sample_data['long_comments'].lower()))
            n += len(re.findall(f"ig{hc} {lc}", sample_data['long_comments'].lower()))
            antibodies_mentioned[f"Ig{hc.upper()}_{lc[0].upper()}"] = n > 0
            if n > 0:
                found_full_size_antibodies = True
        if not found_full_size_antibodies:
            n = len(re.findall(f"cha[iî]nes l[eé]g[eè]res {lc}", sample_data['long_comments'].lower()))
            antibodies_mentioned[f"LC_{lc[0].upper()}"] = n > 0
        else:
            antibodies_mentioned[f"LC_{lc[0].upper()}"] = False
    antibodies_mentioned = {f"Routine_{k}": v for k, v in antibodies_mentioned.items()}

    antibodies_found = {}
    for lc in ('kappa', 'lambda'):
        antibodies_found[f"LC_{lc[0].upper()}"] = False
        for hc in ('gam'):
            antibodies_found[f"Ig{hc.upper()}_{lc[0].upper()}"] = False
    for e in sample_data['peak_data']:
        if len(e['hc']) > 1:
            antibodies_found[f"{e['hc']}_{e['lc']}"] = True
        else:
            antibodies_found[f"LC_{e['lc']}"] = True
    antibodies_found = {f"Research_{k}": v for k, v in antibodies_found.items()}

    annotations.append({'aaid': json_filename, **antibodies_mentioned, **antibodies_found, 'Pre2020Difference': any_diff})

annotations = pd.DataFrame(annotations)

# highlight the ones that have differences
for hc in ["IgG", "IgA", "IgM", "LC"]:
    for lc in ["K", "L"]:
        print(pd.crosstab(annotations[f"Routine_{hc}_{lc}"], annotations[f"Research_{hc}_{lc}"]))
        print("")

# NOW MAKE NEW FOLDERS

# 1st make list of files we have to re-review due to pre 2020 different
list_of_different_from_2020 = annotations[annotations.Pre2020Difference].aaid.tolist()
len(list_of_different_from_2020)  # 107

# 2nd list files potentially false negatives
list_of_potential_false_negatives = []
for hc in ["IgG", "IgA", "IgM", "LC"]:
    for lc in ["K", "L"]:
        hclc_false_negatives = annotations[annotations[f"Routine_{hc}_{lc}"] & (~annotations[f"Research_{hc}_{lc}"])].aaid.tolist()
        list_of_potential_false_negatives.extend(hclc_false_negatives)
# make unique
list_of_potential_false_negatives = list(set(list_of_potential_false_negatives))
len(list_of_potential_false_negatives)  # 179

# 3rd list files potentially false positives
list_of_potential_false_positives = []
for hc in ["IgG", "IgA", "IgM", "LC"]:
    for lc in ["K", "L"]:
        hclc_false_positives = annotations[(~annotations[f"Routine_{hc}_{lc}"]) & annotations[f"Research_{hc}_{lc}"]].aaid.tolist()
        list_of_potential_false_positives.extend(hclc_false_positives)
# make unique
list_of_potential_false_positives = list(set(list_of_potential_false_positives))
len(list_of_potential_false_positives)  # 1488

# 4th make unique
list_of_potential_false_positives = [e for e in list_of_potential_false_positives if (e not in list_of_potential_false_negatives) and (e not in list_of_different_from_2020)]
list_of_potential_false_negatives = [e for e in list_of_potential_false_negatives if e not in list_of_different_from_2020]

len(list_of_different_from_2020)  # 107
len(list_of_potential_false_negatives)  # 459
len(list_of_potential_false_positives)  # 810

# COPY TO NEW FOLDER
review_output_path = r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data\2025\2025_12_10"

for json_list_name, json_list in zip(['vs2020', 'false_negatives', 'false_positives'],
                                     [list_of_different_from_2020, list_of_potential_false_negatives, list_of_potential_false_positives]):
    for json_folder in ['input_jsons', 'confirmed_jsons', 'output_jsons', 'previous_2020_output_jsons', 'spectr_jsons']:
        os.makedirs(os.path.join(review_output_path, json_list_name, json_folder), exist_ok=True)
        for json_filename in tqdm(json_list, desc=f"Processing {json_list_name}/{json_folder}"):
            src_file = os.path.join(json_rootdirectory, json_folder, json_filename)
            if os.path.exists(src_file):
                shutil.copy(src=os.path.join(json_rootdirectory, json_folder, json_filename),
                            dst=os.path.join(review_output_path, json_list_name, json_folder, json_filename))
            else:
                if json_folder in ['input_jsons', 'output_jsons', 'spectr_jsons']:
                    assert False, f"unresolved error for {json_filename=}!"
