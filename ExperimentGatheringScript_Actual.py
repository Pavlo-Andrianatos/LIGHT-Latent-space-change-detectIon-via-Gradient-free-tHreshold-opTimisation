import configparser
import main_train_Thresholds
import os
import shutil
import Utils
from pathlib import Path


def run_experiment(paramObject_var, param):
    if paramObject_var["optimisation_algorithm"] == "pyHopper":
        main_train_Thresholds.train_threshold_method(parametersObject=paramObject_var, param=param)
    else:
        main_train_Thresholds.train_threshold_method(parametersObject=paramObject_var, param=param)


def edit_parameters_file(file_path,
                         optimisation_algorithm,
                         number_of_classes,
                         image_channels,
                         dataset_used,
                         number_of_change_images,
                         number_of_epochs,
                         path_to_models,
                         loaded_model_name,
                         csv_file_to_use,
                         csv_file_to_use_test,
                         path_to_usable_images,
                         learning_rate,
                         training_ratio,
                         fitness_function,
                         output_images,
                         output_every_x_images):
    config = configparser.ConfigParser()
    config.read(file_path)

    config.set('EXPERIMENTTHRESHOLD', 'optimisation_algorithm', optimisation_algorithm)
    config.set('EXPERIMENTTHRESHOLD', 'number_of_epochs', str(number_of_epochs))
    config.set('EXPERIMENTTHRESHOLD', 'learning_rate', str(learning_rate))
    config.set('EXPERIMENTTHRESHOLD', 'training_ratio', str(training_ratio))
    config.set('EXPERIMENTTHRESHOLD', 'fitness_function', fitness_function)

    config.set('EXPERIMENTTHRESHOLD', 'path_to_models', path_to_models)
    config.set('EXPERIMENTTHRESHOLD', 'loaded_model_name', loaded_model_name)
    config.set('EXPERIMENTTHRESHOLD', 'csv_file_to_use', csv_file_to_use)
    config.set('EXPERIMENTTHRESHOLD', 'csv_file_to_use_test', csv_file_to_use_test)
    config.set('EXPERIMENTTHRESHOLD', 'path_to_usable_images', path_to_usable_images)

    config.set('EXPERIMENTTHRESHOLD', 'output_images', str(output_images))
    config.set('EXPERIMENTTHRESHOLD', 'output_every_x_images', str(output_every_x_images))
    config.set('EXPERIMENTTHRESHOLD', 'number_of_classes', str(number_of_classes))
    config.set('EXPERIMENTTHRESHOLD', 'image_channels', str(image_channels))
    config.set('EXPERIMENTTHRESHOLD', 'dataset_used', dataset_used)
    config.set('EXPERIMENTTHRESHOLD', 'number_of_change_images', str(number_of_change_images))

    with open(file_path, 'w') as configfile:
        config.write(configfile)


def test_method():
    csv_file_to_use_french_14 = "csvFiles/HRSCD_2025/2006_2012_D14_train_extra_gt_backup.csv"
    csv_file_to_use_vaihingen = "csvFiles/Vaihingen/Vaihingen_change_train.csv"
    csv_file_to_use_syntheworld = "csvFiles/SyntheWorld/SyntheWorld_threshold_train.csv"

    csv_file_to_use_french_14_test = "csvFiles/HRSCD_2025/2006_2012_D14_test_extra_gt_backup.csv"
    csv_file_to_use_vaihingen_test = "csvFiles/Vaihingen/Vaihingen_change_test.csv"
    csv_file_to_use_syntheworld_test = "csvFiles/SyntheWorld/SyntheWorld_threshold_test.csv"

    # Paths to images
    path_to_usable_images_french = ""
    path_to_usable_images_vaihingen = ""
    path_to_usable_images_syntheworld = ""

    optimisation_algorithms = ["cma", "pso", "ga", "pyHopper"]

    number_of_epochs = 20
    learning_rate = 0.0002  # Irrelevant - not used
    training_ratio = 0.8
    fitness_function = "score"

    # Name of directory storing models, used when saving and copying experiment data
    path_to_models = "unet_models"

    # Absolute Path to directory of models
    path = "C:\\Users\\Pavlo\\Documents\\University Stuff\\Masters\\Project\\unet_models"

    directories = [d.name for d in Path(path).iterdir() if d.is_dir()]

    vaihingen_dirs = [d for d in directories if 'vaihingen' in d.lower() and 'twentyfive' in d.lower()]
    french_14_dirs = [d for d in directories if '14' in d.lower() and 'twentyfive' in d.lower()]
    syntheworld_dirs = [d for d in directories if 'syn' in d.lower() and 'fifty' in d.lower()]

    all_models = [vaihingen_dirs, french_14_dirs, syntheworld_dirs]

    # If you want images to be saved
    output_images = "True"
    # How many images are saved. 1 is every image, 2 is every second, 3 is every third, etc.
    output_every_x_images = 1
    # Image channels, RGB is 3. But this is never used. Images are always assumed to be RGB
    image_channels = 3

    # Below variables are overridden
    number_of_classes = 0
    dataset_used = ""
    csv_file_to_use = ""
    csv_file_to_use_test = ""
    path_to_usable_images = ""

    number_of_change_images_set = [5, 10, 15, 20]

    csv_file_path = 'parameters_experiment.config'

    for main_dataset in all_models:
        for unet_model in main_dataset:
            for number_of_change_images in number_of_change_images_set:
                loaded_model_name = unet_model
                if 'vaihingen' in loaded_model_name.lower():
                    number_of_classes = 3
                    csv_file_to_use = csv_file_to_use_vaihingen
                    csv_file_to_use_test = csv_file_to_use_vaihingen_test
                    path_to_usable_images = path_to_usable_images_vaihingen
                    dataset_used = "vaihingen"
                elif '14' in loaded_model_name.lower():
                    number_of_classes = 5
                    csv_file_to_use = csv_file_to_use_french_14
                    csv_file_to_use_test = csv_file_to_use_french_14_test
                    path_to_usable_images = path_to_usable_images_french
                    dataset_used = "french"
                elif 'syn' in loaded_model_name.lower():
                    number_of_classes = 8
                    csv_file_to_use = csv_file_to_use_syntheworld
                    csv_file_to_use_test = csv_file_to_use_syntheworld_test
                    path_to_usable_images = path_to_usable_images_syntheworld
                    dataset_used = "syn"

                for optimisation_algorithm in optimisation_algorithms:
                    edit_parameters_file(csv_file_path,
                                         optimisation_algorithm,
                                         number_of_classes,
                                         image_channels,
                                         dataset_used,
                                         number_of_change_images,
                                         number_of_epochs,
                                         path_to_models,
                                         loaded_model_name,
                                         csv_file_to_use,
                                         csv_file_to_use_test,
                                         path_to_usable_images,
                                         learning_rate,
                                         training_ratio,
                                         fitness_function,
                                         output_images,
                                         output_every_x_images
                                         )
                    for experiment_number in range(1, 2):
                        experiment_type = "_test_{}_{}_".format(experiment_number, number_of_change_images)
                        experiment_directory = "{}/{}".format("Experiments", (
                                loaded_model_name + experiment_type + optimisation_algorithm))
                        if os.path.isdir(experiment_directory):
                            print("Experiment {} already exists. Continuing...".format(experiment_directory))
                            continue
                        tuned_params = None

                        # Set tune_params_operation to True if you want to use PyHopper to tune hyperparameters
                        tune_params_operation = False
                        if tune_params_operation:
                            if not optimisation_algorithm == "pyHopper":
                                print(
                                    "Starting Hyperparameter optimisation for experiment {}, with algorithm {}".format(
                                        experiment_number, optimisation_algorithm))
                                if optimisation_algorithm == "pso":
                                    with open('{}/{}/threshold_values_train.txt'.format(path_to_models,
                                                                                        loaded_model_name),
                                              "w") as threshold_values_file:
                                        threshold_values_file.write("")
                                    with open('{}/{}/PSO Train Info about Models.txt'.format(path_to_models,
                                                                                             loaded_model_name),
                                              'w') as f:
                                        f.write("")
                                    with open('{}/{}/PSO_pyHopper_parameters.txt'.format(path_to_models,
                                                                                         loaded_model_name),
                                              "w") as pyHopper_parameters:
                                        pyHopper_parameters.write("")
                                    tuned_params = main_train_Thresholds.pyHopper_hp_pso(number_of_epochs)
                                elif optimisation_algorithm == "ga":
                                    with open('{}/{}/threshold_values_train.txt'.format(path_to_models,
                                                                                        loaded_model_name),
                                              "w") as threshold_values_file:
                                        threshold_values_file.write("")
                                    with open('{}/{}/GA Train Info about Models.txt'.format(path_to_models,
                                                                                            loaded_model_name),
                                              'w') as f:
                                        f.write("")
                                    with open('{}/{}/GA_pyHopper_parameters.txt'.format(path_to_models,
                                                                                        loaded_model_name),
                                              "w") as pyHopper_parameters:
                                        pyHopper_parameters.write("")
                                    Utils.global_ga_iteration_number = 0
                                    tuned_params = main_train_Thresholds.pyHopper_hp_ga(number_of_epochs)
                                elif optimisation_algorithm == "cma":
                                    with open('{}/{}/threshold_values_train.txt'.format(path_to_models,
                                                                                        loaded_model_name),
                                              "w") as threshold_values_file:
                                        threshold_values_file.write("")
                                    with open('{}/{}/CMA Train Info about Models.txt'.format(path_to_models,
                                                                                             loaded_model_name),
                                              'w') as f:
                                        f.write("")
                                    with open('{}/{}/CMA_pyHopper_parameters.txt'.format(path_to_models,
                                                                                         loaded_model_name),
                                              "w") as pyHopper_parameters:
                                        pyHopper_parameters.write("")
                                    Utils.global_cma_iteration_number = 0
                                    tuned_params = main_train_Thresholds.pyHopper_hp_cma(number_of_epochs)
                                print("Done Hyperparameter optimisation for experiment {}".format(experiment_number))
                                print("Best Tuned parameters for experiment {}: {}".format(experiment_number,
                                                                                           tuned_params))
                            else:
                                print("Optimisation algorithm is PyHopper, so no hyperparameter optimisation required "
                                      "for experiment {}".format(experiment_number))

                        edit_parameters_file(csv_file_path,
                                             optimisation_algorithm,
                                             number_of_classes,
                                             image_channels,
                                             dataset_used,
                                             number_of_change_images,
                                             number_of_epochs,
                                             path_to_models,
                                             loaded_model_name,
                                             csv_file_to_use,
                                             csv_file_to_use_test,
                                             path_to_usable_images,
                                             learning_rate,
                                             training_ratio,
                                             fitness_function,
                                             output_images,
                                             output_every_x_images
                                             )

                        paramObject = {"optimisation_algorithm": optimisation_algorithm,
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

                        Utils.global_pyhopper_iteration_number = 0
                        Utils.global_ga_iteration_number = 0
                        Utils.global_cma_iteration_number = 0

                        run_experiment(paramObject, param=tuned_params)
                        Utils.global_pyhopper_iteration_number = 0

                        print("Now copying files...")

                        experiment_type_name = ""
                        if optimisation_algorithm == "pso":
                            experiment_type_name = "PSO"
                        elif optimisation_algorithm == "cma":
                            experiment_type_name = "CMA"
                        elif optimisation_algorithm == "ga":
                            experiment_type_name = "GA"
                        elif optimisation_algorithm == "pyHopper":
                            experiment_type_name = "PyHopper"

                        if not os.path.isdir(experiment_directory):
                            os.mkdir("{}/".format(experiment_directory))

                        shutil.copyfile(path_to_models + "/" + loaded_model_name + "/threshold_values_train.txt",
                                        experiment_directory + "/threshold_values_train.txt")
                        shutil.copyfile(
                            path_to_models + "/" + loaded_model_name + "/{} Train Info about Models.txt".format(
                                experiment_type_name),
                            experiment_directory + "/{} Train Info about Models.txt".format(
                                experiment_type_name))
                        if optimisation_algorithm != "pyHopper":
                            shutil.copyfile(
                                path_to_models + "/" + loaded_model_name + "/{}_pyHopper_parameters.txt".format(
                                    experiment_type_name),
                                experiment_directory + "/{}_pyHopper_parameters.txt".format(
                                    experiment_type_name))

                        if optimisation_algorithm != "pyHopper":
                            shutil.copytree(
                                path_to_models + "/" + loaded_model_name + "/result/train/threshold/" + optimisation_algorithm,
                                experiment_directory + "/images/train/", dirs_exist_ok=True)
                            shutil.copytree(
                                path_to_models + "/" + loaded_model_name + "/result/test/threshold/" + optimisation_algorithm,
                                experiment_directory + "/images/test/", dirs_exist_ok=True)
                        else:
                            shutil.copytree(
                                path_to_models + "/" + loaded_model_name + "/result/train/threshold/" + experiment_type_name,
                                experiment_directory + "/images/train/", dirs_exist_ok=True)
                            shutil.copytree(
                                path_to_models + "/" + loaded_model_name + "/result/test/threshold/" + experiment_type_name,
                                experiment_directory + "/images/test/", dirs_exist_ok=True)

                        shutil.copyfile("parameters_experiment.config",
                                        experiment_directory + "/parameters_experiment.config")

                        print("Experiment {} Complete.".format(experiment_number))

                        Utils.global_pyhopper_iteration_number = 0
                        Utils.global_ga_iteration_number = 0


def start_testing():
    test_method()
