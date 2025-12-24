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
DSET_NAME = "lemans"
VERSION = "2025_12_24"
json_rootdirectory = os.path.join(root_path, r"2025\final_json_annotations\2025_12_13\lemans")

# CAPE TOWN 2025:
# DSET_NAME = "capetown"
# VERSION = "2025_12_24"
# json_rootdirectory = os.path.join(root_path, r"2025\final_json_annotations\2025_12_24\capetown")


def load_json_data(json_filename, mode):
    load_dir = "input_jsons" if (mode == "annotate") else "output_jsons" if (mode == "confirm") else "confirmed_jsons"
    with open(os.path.join(json_rootdirectory, load_dir, json_filename), "r") as f:
        saved_data = json.load(f)
    return saved_data


def annotation_complete_to_simple(s):
    if s == 'Negative':
        return s
    if s.startswith('LC_'):
        return 'LC'
    if s.startswith('Ig'):
        return s[:3]
    return 'Complex/biclonal'


def annotation_complete_to_classic(s):
    if s == 'Negative':
        return 'Normal'
    if s.startswith('LC_'):
        return 'Complex'
    if s.startswith('Ig'):
        return 'Ig ' + s[2] + s[4].lower()
    return 'Complex'


def hc_str_to_int(s):
    if len(s) == 0:
        return None
    if s[-1] == 'G':
        return 0
    if s[-1] == 'A':
        return 1
    if s[-1] == 'M':
        return 2
    assert False, f"Unknown hc={s}"

def lc_str_to_int(s):
    if s[-1] == 'K':
        return 3
    if s[-1] == 'L':
        return 4
    assert False, f"Unknown lc={s}"


def print_count_stats(df_to_process, column):
    print(pd.concat([df_to_process[column].value_counts(),
                     (100 * df_to_process[column].value_counts(normalize=True)).round(1)], axis=1).apply(lambda row: f"n={row[0]:.0f} ({row[1]}%)", axis=1))


# %%


max_n_peaks = 5
max_n_fractions = 7

annotations = []
X = []
X_spe = []
y = []

for json_filename in tqdm(os.listdir(os.path.join(json_rootdirectory, "input_jsons"))):
    try:  # first fetch any confirmed annotations
        json_data = load_json_data(json_filename=json_filename, mode="review")
    except FileNotFoundError:  # if unavailable use unconfirmed annotations
        json_data = load_json_data(json_filename=json_filename, mode="confirm")
    # start copy of information that does not need processing
    item_annotations = {}
    for key in ['paid',
                'aaid',
                'age',
                'sex',
                'total_protein',
                'short_comments',
                'long_comments',
                'patient_other_short_comments',
                'patient_other_long_comments',
                'doubtful',
                'exclude',
                'annotated_by',
                'annotated_at',
                'confirmed_by',
                'confirmed_at']:
        if key in json_data.keys():
            item_annotations[key] = json_data[key]
        else:
            item_annotations[key] = np.nan

    # process peaks
    peak_data = json_data['peak_data']
    assert len(peak_data) <= max_n_peaks, "Too many peaks!"
    for p in range(max_n_peaks):
        if p < len(peak_data):
            item_annotations[f'p{p+1}_start'] = peak_data[p]['start']
            item_annotations[f'p{p+1}_end'] = peak_data[p]['end']
            item_annotations[f'p{p+1}_hc'] = peak_data[p]['hc']
            item_annotations[f'p{p+1}_lc'] = peak_data[p]['lc']
        else:
            item_annotations[f'p{p+1}_start'] = np.nan
            item_annotations[f'p{p+1}_end'] = np.nan
            item_annotations[f'p{p+1}_hc'] = np.nan
            item_annotations[f'p{p+1}_lc'] = np.nan

    # process peak annotations
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
    annotation_simple = annotation_complete_to_simple(annotation_summary)
    annotation_classic = annotation_complete_to_classic(annotation_summary)
    item_annotations['annot_complete'] = annotation_summary
    item_annotations['annot_simple'] = annotation_simple
    item_annotations['annot_classic'] = annotation_classic

    # process SPEP manual annotations (fractions, fraction names, peaks)
    item_annotations[f'SPE_data'] = json_data['traces']['SPE']['exists']
    if json_data['traces']['SPE']['exists']:
        if ('peaks' not in json_data['traces']['SPE'].keys()) or (json_data['traces']['SPE']['peaks'] is None):
            json_data['traces']['SPE']['peaks'] = []
        item_x_spe = json_data['traces']['SPE']['data']
        assert len(json_data['traces']['SPE']['fractions']['coords']) <= (max_n_fractions + 1), f"Unhandled number of fraction coords for SPEP: {json_data['traces']['SPE']['fractions']['coords']=}"
        assert len(json_data['traces']['SPE']['fractions']['names']) <= max_n_fractions, f"Unhandled number of fraction names for SPEP: {json_data['traces']['SPE']['fractions']['names']=}"
        # do not check (there will be some errors stored in the db!)
        # assert len(json_data['traces']['SPE']['fractions']['names']) == (len(json_data['traces']['SPE']['fractions']['coords']) - 1), f"Inconsistent number of fraction names compared to coords for SPEP"
        assert len(json_data['traces']['SPE']['peaks']) <= (2 * max_n_peaks), f"Unhandled number of peaks for SPEP: {json_data['traces']['SPE']['peaks']=}"
        assert len(json_data['traces']['SPE']['peaks']) % 2 == 0, f"Uneven number of peak coords for SPEP: {json_data['traces']['SPE']['peaks']=}"
        for i in range(max_n_fractions + 1):
            if i < len(json_data['traces']['SPE']['fractions']['coords']):
                item_annotations[f'f{i + 1}'] = json_data['traces']['SPE']['fractions']['coords'][i]
            else:
                item_annotations[f'f{i + 1}'] = np.nan
        for i in range(max_n_fractions):
            if i < len(json_data['traces']['SPE']['fractions']['names']):
                if type(json_data['traces']['SPE']['fractions']['names']) is list:
                    item_annotations[f'fn{i + 1}'] = json_data['traces']['SPE']['fractions']['names'][i]
                elif type(json_data['traces']['SPE']['fractions']['names']) is dict:
                    item_annotations[f'fn{i + 1}'] = json_data['traces']['SPE']['fractions']['names'][f"fraction{i + 1}_name"]
                else:
                    assert False, "Unrecognized type for json_data/traces/SPE/fractions/names"
            else:
                item_annotations[f'fn{i + 1}'] = np.nan
        for i in range(max_n_peaks):
            if i < (len(json_data['traces']['SPE']['peaks']) // 2):
                item_annotations[f'p{i+1}s'] = json_data['traces']['SPE']['peaks'][i * 2]
                item_annotations[f'p{i+1}f'] = json_data['traces']['SPE']['peaks'][i * 2 + 1]
            else:
                item_annotations[f'p{i + 1}s'] = np.nan
                item_annotations[f'p{i + 1}f'] = np.nan
    else:
        item_x_spe = np.zeros((300, )) * np.nan
        for i in range(max_n_fractions + 1):
            item_annotations[f'f{i+1}'] = np.nan
        for i in range(max_n_fractions):
            item_annotations[f'fn{i+1}'] = np.nan
        for i in range(max_n_peaks):
            item_annotations[f'p{i+1}s'] = np.nan
            item_annotations[f'p{i+1}f'] = np.nan
    X_spe.append(item_x_spe)

    # process traces
    item_x = np.zeros((7, 304)) * np.nan
    for t, trace in enumerate(['ELP', 'IgG', 'IgA', 'IgM', 'K', 'L', 'Ref']):
        trace_exists = json_data['traces'][trace]['exists']
        item_annotations[f'{trace}_data'] = trace_exists
        if trace_exists:
            if type(json_data['traces'][trace]['data']) is dict:
                item_x[t, :] = np.array([json_data['traces'][trace]['data'][f"{trace}_x{i + 1}"] for i in range(304)])
            elif type(json_data['traces'][trace]['data']) is list:
                item_x[t, :] = json_data['traces'][trace]['data']
            else:
                assert False, "Trace data is not stored in a dict or list"
    X.append(item_x)

    # process peaks as traces
    if json_data['exclude']:
        item_y = np.zeros((5, 304)) * np.nan
    else:
        item_y = np.zeros((5, 304))
        for p, pdata in enumerate(peak_data):
            hc_i, lc_i = hc_str_to_int(pdata['hc']), lc_str_to_int(pdata['lc'])
            ps, pf = int(pdata['start']), (int(pdata['end']) + 1)
            if hc_i is not None:
                item_y[hc_i, ps:pf] = 1
            item_y[lc_i, ps:pf] = 1
    y.append(item_y)

    # save
    annotations.append(item_annotations)

# concat everything
annotations = pd.DataFrame(annotations)
X = np.stack(X, axis=0)
X_spe = np.stack(X_spe, axis=0)
y = np.stack(y, axis=0)

# add partitioning info
training_samples = np.random.RandomState(seed=0).choice(len(annotations), int(len(annotations) * .8), replace=False)
annotations["partition"] = "validation"
annotations.iloc[training_samples, -1] = "training"

# TODO re-run for Le Mans
# create the full dataset
full_dataset = annotations.copy()
# add traces
for X_i, X_name in enumerate(['ELP', 'IgG', 'IgA', 'IgM', 'K', 'L', 'Ref']):
    tmp_df = pd.DataFrame(X[:, X_i, :], columns=[f"{X_name}_{i + 1}" for i in range(304)])
    full_dataset = pd.concat([full_dataset, tmp_df], axis=1)
# add spep trace
# also normalize SPE between 0 and 1 if not done yet (for now: not supported, will raise an error)
assert np.nanmin(X_spe) == 0, "X_spe is not normalized between 0 and 1! (max != 0)"
assert np.nanmax(X_spe) == 1, "X_spe is not normalized between 0 and 1! (max != 1)"
if X_spe.shape[1] == 300:  # add padding
    X_spe = np.concatenate([np.zeros([len(X_spe), 2]),
                            X_spe,
                            np.zeros([len(X_spe), 2])], axis=1)
elif X_spe.shape[1] != 304:
    assert False, f"Expected X_spe to have a shape of (:,300) or (:,304), got {X_spe.shape=}"
tmp_df = pd.DataFrame(X_spe, columns=[f"SPE_{i + 1}" for i in range(304)])
full_dataset = pd.concat([full_dataset, tmp_df], axis=1)
# add y annotations
for y_i, y_name in enumerate(['IgG', 'IgA', 'IgM', 'K', 'L']):
    tmp_df = pd.DataFrame(y[:, y_i, :], columns=[f"segm_{y_name}_{i + 1}" for i in range(304)])
    full_dataset = pd.concat([full_dataset, tmp_df], axis=1)


# %%

print_count_stats(full_dataset, "partition")
# LEMANS
# training      n=4676 (80.0%)
# validation    n=1170 (20.0%)

# CAPE TOWN
# training      n=575 (80.0%)
# validation    n=144 (20.0%)

pd.crosstab(full_dataset.annot_simple, full_dataset.partition)
# LEMANS
# partition         training  validation
# annot_simple
# Complex/biclonal       296          68
# IgA                    463         106
# IgG                   2630         666
# IgM                    876         218
# LC                      36          11
# Negative               375         101

# CAPE TOWN
# partition         training  validation
# annot_simple
# Complex/biclonal        51          15
# IgA                     61          14
# IgG                    332          79
# IgM                     29           7
# LC                      17           6
# Negative                85          23

print_count_stats(full_dataset, "annot_simple")
# LEMANS
# IgG                 n=3296 (56.4%)
# IgM                 n=1094 (18.7%)
# IgA                   n=569 (9.7%)
# Negative              n=476 (8.1%)
# Complex/biclonal      n=364 (6.2%)
# LC                     n=47 (0.8%)

# CAPE TOWN
# IgG                 n=411 (57.2%)
# Negative            n=108 (15.0%)
# IgA                  n=75 (10.4%)
# Complex/biclonal      n=66 (9.2%)
# IgM                   n=36 (5.0%)
# LC                    n=23 (3.2%)

# some stats about doubtful/to exclude
print_count_stats(full_dataset, "doubtful")
# LEMANS
# False    n=5007 (85.6%)
# True      n=839 (14.4%)

# CAPE TOWN
# False    n=646 (89.8%)
# True      n=73 (10.2%)

print_count_stats(full_dataset, "exclude")
# LEMANS
# False    n=5842 (99.9%)
# True         n=4 (0.1%)

# CAPE TOWN
# False    n=692 (96.2%)
# True       n=27 (3.8%)

print_count_stats(full_dataset, "annot_classic")
# LEMANS
# Ig Gk      n=2204 (37.7%)
# Ig Gl      n=1092 (18.7%)
# Ig Mk       n=847 (14.5%)
# Normal       n=476 (8.1%)
# Complex      n=411 (7.0%)
# Ig Ak        n=308 (5.3%)
# Ig Al        n=261 (4.5%)
# Ig Ml        n=247 (4.2%)

# CAPE TOWN
# Ig Gk      n=271 (37.7%)
# Ig Gl      n=140 (19.5%)
# Normal     n=108 (15.0%)
# Complex     n=89 (12.4%)
# Ig Ak        n=47 (6.5%)
# Ig Mk        n=28 (3.9%)
# Ig Al        n=28 (3.9%)
# Ig Ml         n=8 (1.1%)

# %%

# save
output_path = os.path.join(root_path, r"2025\final_datasets", DSET_NAME, VERSION)
os.makedirs(output_path, exist_ok=True)

# FILTER ONLY NOT EXCLUDE AND OK ITEMS!!!
KEEP_FILTER = (~full_dataset.exclude) & full_dataset.ELP_data & full_dataset.IgG_data & full_dataset.IgA_data & full_dataset.IgM_data & full_dataset.K_data & full_dataset.L_data

print(f"Final dataset size: {full_dataset.shape=}")
print(f"Final number of samples retained for training+validation sets: {np.sum(KEEP_FILTER)}")
# LE MANS
# Final dataset size: full_dataset.shape=(5846, 4024)
# Final number of samples retained for training+validation sets: 5842

print(f"Saving at: {output_path}")

# csv
print(f"Saving to .csv")
full_dataset.to_csv(os.path.join(output_path, f"full_dataset.csv"), index=True)
# excel
# full_dataset.to_excel(os.path.join(output_path, f"full_dataset.xlsx"), index=True)
# h5
print(f"Saving to .h5")
full_dataset.to_hdf(os.path.join(output_path, f"full_dataset.h5"), key="dataset")

# save files to make them compatible with old version of the training/inference scripts
# reload final json files, turn everything into a dataset that we can load in python for training deep learning models
# we want the data to match the previous format used in previous scripts:

# ref_data_path = r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data\proc\lemans_2018"
# ref_if_x = np.load(os.path.join(ref_data_path, "if_v1_x.npy"))
# ref_if_y = np.load(os.path.join(ref_data_path, "if_v1_y.npy"))
# ref_if_simple_y = pd.read_csv(os.path.join(ref_data_path, "if_simple_y.csv"))
# ref_train_samples = np.load(os.path.join(ref_data_path, "train_samples.npy"))
#
# ref_if_x.shape  # a matrix of shape (N, 304, 6) with N=n samples, 304 points, 6 dimensions (ELP, G, A, M, k, l)
# ref_if_x.min()
# ref_if_x.max()  # already normalized between 0-1
# ref_if_y.shape  # a matrix of shape (N, 304, 5)
# ref_if_y.min()
# ref_if_y.max()  # same, normalized between 0 and 1
# np.unique(ref_if_y)  # only 0s and 1s as expected
# ref_if_simple_y.shape  # a csv with a pd dataframe, no index, one column named 'Abnormality', with values:
# ref_if_simple_y.value_counts()
# # Abnormality
# # Ig Gk          701
# # Ig Gl          381
# # Ig Mk          335
# # Ig Ak          102
# # Ig Ml           85
# # Complex         76
# # Ig Al           70
# # Normal          53
# ref_train_samples.shape  # a matrix of shape (N',), with N'= the number of training samples

# extract
if_x = np.stack([full_dataset[KEEP_FILTER][[f"{X_name}_{i + 1}" for i in range(304)]].values for X_name in ['ELP', 'IgG', 'IgA', 'IgM', 'K', 'L']], axis=2)
if_y = np.stack([full_dataset[KEEP_FILTER][[f"segm_{X_name}_{i + 1}" for i in range(304)]].values for X_name in ['IgG', 'IgA', 'IgM', 'K', 'L']], axis=2)
if_simple_y = pd.DataFrame(full_dataset[KEEP_FILTER]['annot_classic'].values, columns=['Abnormality'])
train_samples = np.where(full_dataset[KEEP_FILTER].partition == "training")[0]

# save
print(f"Saving files for training")
print(f"Saving X to .npy")
np.save(os.path.join(output_path, f"if_v1_x.npy"), if_x)
print(f"Saving y to .npy")
np.save(os.path.join(output_path, f"if_v1_y.npy"), if_y)
print(f"Saving y to .csv")
if_simple_y.to_csv(os.path.join(output_path, f"if_v1_simple_y.csv"), index=False)
print(f"Saving training sample list")
np.save(os.path.join(output_path, f"train_samples.npy"), train_samples)
