from datetime import datetime
from configparser import ConfigParser


def load_Parameters_Type(type_load):
    if type_load == "train_CNN":
        return load_train_CNN_Parameters()
    elif type_load == "train_threshold":
        return load_train_threshold_Parameters()
    else:
        print("Do not recognise parameter type for execution.")


def load_train_CNN_Parameters():
    print("Attempting to load parameters to train a CNN.")

    config_object = ConfigParser()

    try:
        config_object.read("parameters_train_CNN.config")
    except FileNotFoundError:
        print("Could not find parameters file, aborting...")
        exit(0)

    trainCNNInfo = config_object["TRAINCNN"]

    path_to_models = trainCNNInfo["path_to_models"]

    loaded_model_name = trainCNNInfo["model_to_load"]

    csv_file_to_use = trainCNNInfo["csv_file_to_use"]

    path_to_usable_images = trainCNNInfo["path_to_usable_images"]

    output_images = trainCNNInfo["output_images"]

    output_every_x_images = int(trainCNNInfo["output_every_x_images"])

    image_size = int(trainCNNInfo["image_size"])

    number_of_classes = int(trainCNNInfo["number_of_classes"])

    image_channels = int(trainCNNInfo["image_channels"])

    batch_size = int(trainCNNInfo["batch_size"])

    learning_rate = float(trainCNNInfo["learning_rate"])

    number_of_epochs = int(trainCNNInfo["number_of_epochs"])

    training_ratio = float(trainCNNInfo["training_ratio"])

    if loaded_model_name == "null":
        loaded_model_name = datetime.now().strftime("%H_%M_%S")

    print("Loaded parameters.")
    return {"batch_size": batch_size,
            "image_size": image_size,
            "number_of_classes": number_of_classes,
            "image_channels": image_channels,
            "learning_rate": learning_rate,
            "number_of_epochs": number_of_epochs,
            "path_to_models": path_to_models,
            "loaded_model_name": loaded_model_name,
            "csv_file_to_use": csv_file_to_use,
            "path_to_usable_images": path_to_usable_images,
            "training_ratio": training_ratio,
            "output_images": output_images,
            "output_every_x_images": output_every_x_images
            }


def load_train_threshold_Parameters():
    print("Attempting to load parameters.")

    config_object = ConfigParser()

    try:
        config_object.read("parameters_experiment.config")
    except FileNotFoundError:
        print("Could not find parameters file, aborting...")
        exit(0)

    trainInfo = config_object["EXPERIMENTTHRESHOLD"]

    optimisation_algorithm = trainInfo["optimisation_algorithm"]

    number_of_classes = int(trainInfo["number_of_classes"])

    image_channels = int(trainInfo["image_channels"])

    dataset_used = trainInfo["dataset_used"]

    number_of_change_images = int(trainInfo["number_of_change_images"])

    number_of_epochs = int(trainInfo["number_of_epochs"])

    training_ratio = float(trainInfo["training_ratio"])

    fitness_function = trainInfo["fitness_function"]

    path_to_models = trainInfo["path_to_models"]

    learning_rate = float(trainInfo["learning_rate"])

    loaded_model_name = trainInfo["loaded_model_name"]

    if loaded_model_name == "null":
        loaded_model_name = datetime.now().strftime("%H_%M_%S")

    csv_file_to_use = trainInfo["csv_file_to_use"]

    csv_file_to_use_test = trainInfo["csv_file_to_use_test"]

    path_to_usable_images = trainInfo["path_to_usable_images"]

    output_images = trainInfo["output_images"]

    output_every_x_images = int(trainInfo["output_every_x_images"])

    print("Loaded parameters.")
    return {"optimisation_algorithm": optimisation_algorithm,
            "number_of_classes": number_of_classes,
            "image_channels": image_channels,
            "dataset_used": dataset_used,
            "number_of_change_images": number_of_change_images,
            "number_of_epochs": number_of_epochs,
            "path_to_models": path_to_models,
            "loaded_model_name": loaded_model_name,
            "csv_file_to_use": csv_file_to_use,
            "csv_file_to_use_test": csv_file_to_use_test,
            "path_to_usable_images": path_to_usable_images,
            "learning_rate": learning_rate,
            "training_ratio": training_ratio,
            "fitness_function": fitness_function,
            "output_images": output_images,
            "output_every_x_images": output_every_x_images
            }
