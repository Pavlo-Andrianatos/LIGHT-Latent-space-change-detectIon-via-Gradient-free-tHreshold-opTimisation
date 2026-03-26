import train_CNN
import loadParameters
import modelGeneration
import shutil


def trainCNN_method():
    parametersObject = loadParameters.load_Parameters_Type("train_CNN")
    generator, did_load_model = modelGeneration.loadOrGenerateModel(
        loaded_model_name=parametersObject["loaded_model_name"],
        load_models_file_path=parametersObject["path_to_models"])
    print("Completed model initialization stage")
    print("Will attempt to train model")
    train_CNN.trainModelFunction(generator, did_load_model, parametersObject)

    CNN_directory = "{}/{}".format(parametersObject.path_to_models, (
        parametersObject.model_to_load))

    shutil.copyfile("parameters_train_CNN.config", CNN_directory + "/parameters_train_CNN.config")

    print("CNN training complete.")


def start_training():
    trainCNN_method()

