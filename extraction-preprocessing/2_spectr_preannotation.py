# This script can be used after we extract the Phoresis DB and create a list of JSON files that will be used to manually annotate samples
# Here, we load those JSON files and use SPECTR pretrained model to detect peaks
# So we can automatically pre-annotate samples
# And we'll just have to check if annotations are consistent

import argparse
import os
# import h5py
import tflite_runtime.interpreter as tflite
import numpy as np
import json
from coding_assets.python import config_manager as cfm
from tqdm import tqdm


def LOAD_S_MODEL(model_path):
    loaded_models = []
    print('Loading models ({})'.format(model_path))
    for key in ['s', ]:
        print('Loading model: {}'.format(key))
        # reconstruct then load weights
        keymodel_path = os.path.join(model_path, "{}-model.tflite".format(key))
        tmp_m = tflite.Interpreter(model_path=keymodel_path)
        print('Model successfully loaded')
        loaded_models.append(tmp_m)
    print('All models successfully loaded')
    return loaded_models


def SPECTR_FOR_IT(input_directory, output_directory, model_path, loaded_models=None, spe_width=304):
    # load data
    print(f"Listed {len(os.listdir(input_directory))} files at input directory <{input_directory}>")

    json_metadata = []
    X_data = []
    for json_file in os.listdir(input_directory):
        try:
            with open(os.path.join(input_directory, json_file), 'r') as f:
                json_data = json.load(f)
        except:
            continue
        json_metadata.append({"json_filename": json_file, "paid": json_data["paid"], "aaid": json_data["aaid"]})
        X_data.append(np.array(json_data["traces"]["ELP"]["data"]))
    X_data = np.stack(X_data, axis=0)

    print(f'Successfully loaded X data with shape={X_data.shape}')

    if loaded_models is None:
        loaded_models = LOAD_S_MODEL(model_path)

    print('Converting curves to AI-readable data')
    X_data = X_data[:, :, None, None]
    X_data = X_data.astype(np.float32)

    print('Data successfully converted')

    # launch prediction
    print('Running predictions')

    interpreter = loaded_models[0]  # only 1: the S model
    interpreter.allocate_tensors()
    # get input/output tensors details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # run inference for all samples
    results = []
    for i in range(len(X_data)):
        # set input
        interpreter.set_tensor(input_details[0]['index'], X_data[[i]])
        # run inference
        interpreter.invoke()
        # get output tensor values
        results.append(interpreter.get_tensor(output_details[0]['index']))
    results = np.concatenate(results, axis=0)

    # export
    print("Creating output directory if not already existing")
    os.makedirs(output_directory, exist_ok=True)
    print('Exporting results to files')
    for i, json_info in tqdm(enumerate(json_metadata), total=len(json_metadata)):
        json_content = {"paid": json_info["paid"],
                        "aaid": json_info["aaid"],
                        "elp_spep_s_predictions": results[i, :, 0, 0].tolist(),
                        }
        with open(os.path.join(output_directory, json_info["json_filename"]), 'w') as f:
            json.dump(json_content, f)
    print('Data successfully exported')


if __name__ == "__main__":
    print('Loading arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--mode")
    parser.add_argument("--model_path", type=str, default=r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\code\SPECTR\R\SPECTRWebApp2023\tflite")
    # LE MANS 2025:
    # parser.add_argument("--input_json_dir", type=str, default=r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data\2025\lemans\preannotation\input_jsons")
    # parser.add_argument("--output_json_dir", type=str, default=r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data\2025\lemans\preannotation\spectr_jsons")
    # CAPE TOWN 2025:
    parser.add_argument("--input_json_dir", type=str, default=r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data\2025\capetown\preannotation\input_jsons")
    parser.add_argument("--output_json_dir", type=str, default=r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data\2025\capetown\preannotation\spectr_jsons")

    FLAGS = parser.parse_args()

    cfm.print_config(FLAGS, "FLAGS")

    print('Running AI')
    SPECTR_FOR_IT(input_directory=FLAGS.input_json_dir,
                  output_directory=FLAGS.output_json_dir,
                  loaded_models=None,
                  model_path=FLAGS.model_path)
    print('Done')
