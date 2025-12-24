# TODO il y a eu un problème avec les fichiers json initiaux de Cape Town
# quand un fichier n'était pas retrouvé dans la liste des annotations, les annotations du fichier précédent étaient automatiquement utilisées à la place
# pour corriger cela, on a mis à jour le script 1_
# le problème est qu'on a déjà commencé à annoter des données de Cape Town
# et on a pas envie de tout refaire de 0
# donc ici on va récupérer les données de Cape Town, isoler les fichiers qui ont changé, et créer deux sous-dossiers afin de plus facilement
# les reprendre

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

json_rootdirectory_WRONGANNOTS = os.path.join(root_path, r"2025\preannotation\2025_12_09\capetown\preannotation")
json_rootdirectory_REPCDANNOTS = os.path.join(root_path, r"2025\preannotation\2025_12_24\capetown")


def load_json_data(json_filename, mode):
    load_dir = json_rootdirectory_WRONGANNOTS if (mode == "WRONG") else json_rootdirectory_REPCDANNOTS if (mode == "REPROCESSED") else "???"
    with open(os.path.join(load_dir, "input_jsons", json_filename), "r") as f:
        saved_data = json.load(f)
    return saved_data


# first parse and see how many differences
changed_count, didnotchange_count = 0, 0
for json_filename in tqdm(os.listdir(os.path.join(json_rootdirectory_REPCDANNOTS, "input_jsons"))):
    WRONG_json_data = load_json_data(json_filename=json_filename, mode="WRONG")
    CORRECTED_json_data = load_json_data(json_filename=json_filename, mode="REPROCESSED")

    nochange_1 = WRONG_json_data['short_comments'] == CORRECTED_json_data['short_comments']
    nochange_2 = WRONG_json_data['long_comments'] == CORRECTED_json_data['long_comments']
    nochange_3 = CORRECTED_json_data['short_comments'] != "<NO CAPETOWN ANNNOTATIONS>"
    assert all([nochange_1, nochange_2, nochange_3]) or (not any([nochange_1, nochange_2, nochange_3])), "Some comments were changed but not all?"

    if nochange_1:
        didnotchange_count += 1
    else:
        changed_count += 1

print(f"{didnotchange_count} files did not change, {changed_count} files were changed")
# only 20 files changed -> let's just copy all output files from the initial directory with existing annotations
# WITHOUT COPY/PASTING the annotations for samples which changed
# it won't take long to re-annotate those 20 files (not all may have been annotated in the first place)

# discard everything in the output, confirmed folders
for folder in ["output_jsons", "confirmed_jsons", "spectr_jsons"]:
    # NOTE: you have to manually remove them beforehand
    os.makedirs(os.path.join(json_rootdirectory_REPCDANNOTS, folder), exist_ok=False)

changed_count, didnotchange_count = 0, 0
for json_filename in tqdm(os.listdir(os.path.join(json_rootdirectory_REPCDANNOTS, "input_jsons"))):
    WRONG_json_data = load_json_data(json_filename=json_filename, mode="WRONG")
    CORRECTED_json_data = load_json_data(json_filename=json_filename, mode="REPROCESSED")

    nochange_1 = WRONG_json_data['short_comments'] == CORRECTED_json_data['short_comments']
    nochange_2 = WRONG_json_data['long_comments'] == CORRECTED_json_data['long_comments']
    nochange_3 = CORRECTED_json_data['short_comments'] != "<NO CAPETOWN ANNNOTATIONS>"
    assert all([nochange_1, nochange_2, nochange_3]) or (not any([nochange_1, nochange_2, nochange_3])), "Some comments were changed but not all?"

    if nochange_1:
        # NO CHANGE -> copy output_jsons and/or confirmed_jsons
        didnotchange_count += 1
        for folder in ["output_jsons", "confirmed_jsons"]:
            input_fp = os.path.join(json_rootdirectory_WRONGANNOTS, folder, json_filename)
            if os.path.exists(input_fp):
                shutil.copy(src=input_fp,
                            dst=os.path.join(json_rootdirectory_REPCDANNOTS, folder, json_filename))
    else:
        # DO NOTHING -> we do not want incorrect annotations to be carried over to this cleaned version of the json file list
        changed_count += 1

    # regardless: copy spectr's analysis output json file
    shutil.copy(src=os.path.join(json_rootdirectory_WRONGANNOTS, "spectr_jsons", json_filename),
                dst=os.path.join(json_rootdirectory_REPCDANNOTS, "spectr_jsons", json_filename))

print(f"{didnotchange_count} unchanged files successfully handled, {changed_count} updated files successfully handled")
