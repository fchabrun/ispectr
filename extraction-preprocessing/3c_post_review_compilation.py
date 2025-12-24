import shutil
import os
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

# input folders
LEMANS_DOUBLEREVIEW_rootfolder = os.path.join(root_path, r"2025\preannotation\2025_12_12")
LEMANS_SINGLEREVIEW_folder = os.path.join(root_path, r"2025\preannotation\2025_12_09\lemans\preannotation")
# CAPE TOWN -> no need, because we did not confirm anything
# so we can just copy paste the content of the 2025/12/24 folder into the final json folder

# output folders
LEMANS_OUTPUT_FOLDER = os.path.join(root_path, r"2025\final_json_annotations\2025_12_13\lemans")
for folder in ["input_jsons", "output_jsons", "confirmed_jsons", "spectr_jsons", "previous_2020_output_jsons"]:
    os.makedirs(os.path.join(LEMANS_OUTPUT_FOLDER, folder), exist_ok=True)

# le mans

n_doublereview = 0
n_singlereview = 0
for json_filename in tqdm(os.listdir(os.path.join(LEMANS_SINGLEREVIEW_folder, "input_jsons"))):
    # which annotations do we take?
    selected_source_folder = None
    # prioritize annotations that were reviewed twice
    for subreview_folder in ['de', 'false_negatives', 'false_positives', 'vs2020']:
        input_fp = os.path.join(LEMANS_DOUBLEREVIEW_rootfolder, subreview_folder, "input_jsons", json_filename)
        if os.path.exists(input_fp):
            # re-reviewed version exists
            selected_source_folder = os.path.join(LEMANS_DOUBLEREVIEW_rootfolder, subreview_folder)
            n_doublereview += 1
            break
    if selected_source_folder is None:
        selected_source_folder = LEMANS_SINGLEREVIEW_folder
        n_singlereview += 1
    # copy/paste everything
    for folder in ["input_jsons", "output_jsons", "confirmed_jsons", "spectr_jsons", "previous_2020_output_jsons"]:
        input_fp = os.path.join(selected_source_folder, folder, json_filename)
        if os.path.exists(input_fp):
            shutil.copy(src=input_fp,
                        dst=os.path.join(LEMANS_OUTPUT_FOLDER, folder, json_filename))
        else:
            # make sure each sample has at least 1 input data and 1 annotation files
            assert folder != "input_jsons", f"Sample {json_filename=} has no input data!"
            assert folder != "output_jsons", f"Sample {json_filename=} has no output data!"

print(f"Successfully copied files for {n_doublereview} LE MANS samples (double reviewed)")
print(f"Successfully copied files for {n_singlereview} LE MANS samples (single review)")
