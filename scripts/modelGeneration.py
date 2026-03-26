import os
from Unet import *
import glob
import re


def extract_number(f):
    s = re.findall("\d+", f)
    return (int(s[len(s) - 1]) if s else -1, f)


def loadOrGenerateModel(loaded_model_name, load_models_file_path):
    generator = None
    did_load_model = False

    try:
        saved_models = os.path.join('{}/{}/'.format(load_models_file_path, loaded_model_name), '*.pkl')
        saved_models_files = sorted(glob.glob(saved_models), key=os.path.getmtime)

        most_trained_model = max(saved_models_files, key=extract_number)

        generator = torch.load(most_trained_model)
        # Uncomment below if no GPU
        # generator = torch.load(most_trained_model, map_location=torch.device('cpu'))  # This is for laptop only
        print("Found model {}, loading...".format(most_trained_model))
        did_load_model = True
        try:
            with open('{}/{}/Train Info about Models.txt'.format(load_models_file_path,
                                                                 loaded_model_name), 'a') as f:
                f.write("Found model {}, loading...\n".format(most_trained_model))
        except FileNotFoundError:
            print("Could not find Train Info about Models.txt file for models...")
    except (FileNotFoundError, ValueError):
        print("Did not find model {}, will create model using name specified: {}".format(loaded_model_name,
                                                                                         loaded_model_name))

    if generator is None:
        print("No model specified for loading, a new model will be created.")

        model_directory = "{}/".format(load_models_file_path) + loaded_model_name
        if not os.path.isdir(model_directory):
            print("Directory for model will also be created.")
            os.mkdir(model_directory)
            os.mkdir("{}/{}/result".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/model".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/model/argmax".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/model/gtruth".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/model/original".format(load_models_file_path, loaded_model_name))

            os.mkdir("{}/{}/result/test".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/test/model".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/test/model/argmax".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/test/model/gtruth".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/test/model/original".format(load_models_file_path, loaded_model_name))

            os.mkdir("{}/{}/result/test/threshold".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/test/threshold/argmax".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/test/threshold/gtruth".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/test/threshold/original".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/test/threshold/before".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/test/threshold/after".format(load_models_file_path, loaded_model_name))

            os.mkdir("{}/{}/result/train/threshold".format(load_models_file_path, loaded_model_name))

            os.mkdir("{}/{}/result/train/threshold/cma".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/threshold/cma/before".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/threshold/cma/after".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/threshold/cma/argmax".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/threshold/cma/gtruth".format(load_models_file_path, loaded_model_name))
            os.mkdir(
                "{}/{}/result/train/threshold/cma/predicted_change".format(load_models_file_path, loaded_model_name))

            os.mkdir("{}/{}/result/train/threshold/pso".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/threshold/pso/before".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/threshold/pso/after".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/threshold/pso/argmax".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/threshold/pso/gtruth".format(load_models_file_path, loaded_model_name))
            os.mkdir(
                "{}/{}/result/train/threshold/pso/predicted_change".format(load_models_file_path, loaded_model_name))

            os.mkdir("{}/{}/result/train/threshold/ga".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/threshold/ga/before".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/threshold/ga/after".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/threshold/ga/argmax".format(load_models_file_path, loaded_model_name))
            os.mkdir("{}/{}/result/train/threshold/ga/gtruth".format(load_models_file_path, loaded_model_name))
            os.mkdir(
                "{}/{}/result/train/threshold/ga/predicted_change".format(load_models_file_path, loaded_model_name))

            threshold_values_to_write = "threshold_value_1 = 0.4\nthreshold_value_2 = 0.6\nthreshold_value_3 = 0.8\n" \
                                        "threshold_value_4 = 1.0\nthreshold_value_5 = 1.2"

            print("Threshold values will also be generated for new model.")

            with open('{}/threshold_values.txt'.format(model_directory), 'w') as f:
                f.write(threshold_values_to_write)

    if not os.path.isdir("{}/{}/result/train/threshold/PyHopper".format(load_models_file_path, loaded_model_name)):
        os.mkdir("{}/{}/result/train/threshold/PyHopper".format(load_models_file_path, loaded_model_name))
        os.mkdir("{}/{}/result/train/threshold/PyHopper/before".format(load_models_file_path, loaded_model_name))
        os.mkdir("{}/{}/result/train/threshold/PyHopper/after".format(load_models_file_path, loaded_model_name))
        os.mkdir("{}/{}/result/train/threshold/PyHopper/argmax".format(load_models_file_path, loaded_model_name))
        os.mkdir("{}/{}/result/train/threshold/PyHopper/gtruth".format(load_models_file_path, loaded_model_name))
        os.mkdir(
            "{}/{}/result/train/threshold/PyHopper/predicted_change".format(load_models_file_path,
                                                                                loaded_model_name))

    return generator, did_load_model
