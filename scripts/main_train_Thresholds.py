import modelGeneration
import train_CMA
import train_PSO
from train_PSO import testPyHopper
import train_GA
from train_PyHopper import train_method_pyhopper
import numpy as np
import pyhopper
import os


def train_threshold_method(parametersObject=None, param=None):
    path_to_models_temp_var = parametersObject["path_to_models"]
    loaded_model_name_temp_var = parametersObject["loaded_model_name"]
    with open('{}/{}/threshold_values_train.txt'.format(path_to_models_temp_var,
                                                        loaded_model_name_temp_var),
              "w") as threshold_values_file:
        threshold_values_file.write("")
    if parametersObject["optimisation_algorithm"] == "pyHopper":
        with open('{}/{}/PyHopper Train Info about Models.txt'.format(path_to_models_temp_var,
                                                                      loaded_model_name_temp_var), 'w') as f:
            f.write("")

        pyHopper_dir_after = '{}/{}/result/train/threshold/PyHopper/after/'.format(path_to_models_temp_var,
                                                                                   loaded_model_name_temp_var)
        pyHopper_dir_argmax = '{}/{}/result/train/threshold/PyHopper/argmax/'.format(path_to_models_temp_var,
                                                                                     loaded_model_name_temp_var)
        pyHopper_dir_before = '{}/{}/result/train/threshold/PyHopper/before/'.format(path_to_models_temp_var,
                                                                                     loaded_model_name_temp_var)
        pyHopper_dir_gtruth = '{}/{}/result/train/threshold/PyHopper/gtruth/'.format(path_to_models_temp_var,
                                                                                     loaded_model_name_temp_var)
        pyHopper_dir_change = '{}/{}/result/train/threshold/PyHopper/predicted_change/'.format(path_to_models_temp_var,
                                                                                               loaded_model_name_temp_var)

        pyHopper_dir_after_test = '{}/{}/result/test/threshold/PyHopper/after/'.format(path_to_models_temp_var,
                                                                                       loaded_model_name_temp_var)
        pyHopper_dir_argmax_test = '{}/{}/result/test/threshold/PyHopper/argmax/'.format(path_to_models_temp_var,
                                                                                         loaded_model_name_temp_var)
        pyHopper_dir_before_test = '{}/{}/result/test/threshold/PyHopper/before/'.format(path_to_models_temp_var,
                                                                                         loaded_model_name_temp_var)
        pyHopper_dir_gtruth_test = '{}/{}/result/test/threshold/PyHopper/gtruth/'.format(path_to_models_temp_var,
                                                                                         loaded_model_name_temp_var)
        pyHopper_dir_change_test = '{}/{}/result/test/threshold/PyHopper/predicted_change/'.format(
            path_to_models_temp_var,
            loaded_model_name_temp_var)

        for filename in os.listdir(pyHopper_dir_after):
            if os.path.isfile(os.path.join(pyHopper_dir_after, filename)):
                os.remove(os.path.join(pyHopper_dir_after, filename))

        for filename in os.listdir(pyHopper_dir_argmax):
            if os.path.isfile(os.path.join(pyHopper_dir_argmax, filename)):
                os.remove(os.path.join(pyHopper_dir_argmax, filename))

        for filename in os.listdir(pyHopper_dir_before):
            if os.path.isfile(os.path.join(pyHopper_dir_before, filename)):
                os.remove(os.path.join(pyHopper_dir_before, filename))

        for filename in os.listdir(pyHopper_dir_gtruth):
            if os.path.isfile(os.path.join(pyHopper_dir_gtruth, filename)):
                os.remove(os.path.join(pyHopper_dir_gtruth, filename))

        for filename in os.listdir(pyHopper_dir_change):
            if os.path.isfile(os.path.join(pyHopper_dir_change, filename)):
                os.remove(os.path.join(pyHopper_dir_change, filename))

        for filename in os.listdir(pyHopper_dir_after_test):
            if os.path.isfile(os.path.join(pyHopper_dir_after_test, filename)):
                os.remove(os.path.join(pyHopper_dir_after_test, filename))

        for filename in os.listdir(pyHopper_dir_argmax_test):
            if os.path.isfile(os.path.join(pyHopper_dir_argmax_test, filename)):
                os.remove(os.path.join(pyHopper_dir_argmax_test, filename))

        for filename in os.listdir(pyHopper_dir_before_test):
            if os.path.isfile(os.path.join(pyHopper_dir_before_test, filename)):
                os.remove(os.path.join(pyHopper_dir_before_test, filename))

        for filename in os.listdir(pyHopper_dir_gtruth_test):
            if os.path.isfile(os.path.join(pyHopper_dir_gtruth_test, filename)):
                os.remove(os.path.join(pyHopper_dir_gtruth_test, filename))

        for filename in os.listdir(pyHopper_dir_change_test):
            if os.path.isfile(os.path.join(pyHopper_dir_change_test, filename)):
                os.remove(os.path.join(pyHopper_dir_change_test, filename))

        train_pyhopper(parametersObject["number_of_epochs"])
    else:
        print("Got params: {}".format(param))
        generator, did_load_model = modelGeneration.loadOrGenerateModel(
            loaded_model_name=parametersObject["loaded_model_name"],
            load_models_file_path=parametersObject["path_to_models"])
        print("Completed model initialization stage")
        print("Will attempt to optimise threshold values")

        optimisation_algorithm = parametersObject["optimisation_algorithm"]

        dir_after = '{}/{}/result/train/threshold/{}/after/'.format(path_to_models_temp_var,
                                                                    loaded_model_name_temp_var,
                                                                    optimisation_algorithm)
        dir_argmax = '{}/{}/result/train/threshold/{}/argmax/'.format(path_to_models_temp_var,
                                                                      loaded_model_name_temp_var,
                                                                      optimisation_algorithm)
        dir_before = '{}/{}/result/train/threshold/{}/before/'.format(path_to_models_temp_var,
                                                                      loaded_model_name_temp_var,
                                                                      optimisation_algorithm)
        dir_gtruth = '{}/{}/result/train/threshold/{}/gtruth/'.format(path_to_models_temp_var,
                                                                      loaded_model_name_temp_var,
                                                                      optimisation_algorithm)
        dir_change = '{}/{}/result/train/threshold/{}/predicted_change/'.format(path_to_models_temp_var,
                                                                                loaded_model_name_temp_var,
                                                                                optimisation_algorithm)

        dir_after_test = '{}/{}/result/test/threshold/{}/after/'.format(path_to_models_temp_var,
                                                                        loaded_model_name_temp_var,
                                                                        optimisation_algorithm)
        dir_argmax_test = '{}/{}/result/test/threshold/{}/argmax/'.format(path_to_models_temp_var,
                                                                          loaded_model_name_temp_var,
                                                                          optimisation_algorithm)
        dir_before_test = '{}/{}/result/test/threshold/{}/before/'.format(path_to_models_temp_var,
                                                                          loaded_model_name_temp_var,
                                                                          optimisation_algorithm)
        dir_gtruth_test = '{}/{}/result/test/threshold/{}/gtruth/'.format(path_to_models_temp_var,
                                                                          loaded_model_name_temp_var,
                                                                          optimisation_algorithm)
        dir_change_test = '{}/{}/result/test/threshold/{}/predicted_change/'.format(path_to_models_temp_var,
                                                                                    loaded_model_name_temp_var,
                                                                                    optimisation_algorithm)

        for filename in os.listdir(dir_after):
            if os.path.isfile(os.path.join(dir_after, filename)):
                os.remove(os.path.join(dir_after, filename))

        for filename in os.listdir(dir_argmax):
            if os.path.isfile(os.path.join(dir_argmax, filename)):
                os.remove(os.path.join(dir_argmax, filename))

        for filename in os.listdir(dir_before):
            if os.path.isfile(os.path.join(dir_before, filename)):
                os.remove(os.path.join(dir_before, filename))

        for filename in os.listdir(dir_gtruth):
            if os.path.isfile(os.path.join(dir_gtruth, filename)):
                os.remove(os.path.join(dir_gtruth, filename))

        for filename in os.listdir(dir_change):
            if os.path.isfile(os.path.join(dir_change, filename)):
                os.remove(os.path.join(dir_change, filename))

        for filename in os.listdir(dir_after_test):
            if os.path.isfile(os.path.join(dir_after_test, filename)):
                os.remove(os.path.join(dir_after_test, filename))

        for filename in os.listdir(dir_argmax_test):
            if os.path.isfile(os.path.join(dir_argmax_test, filename)):
                os.remove(os.path.join(dir_argmax_test, filename))

        for filename in os.listdir(dir_before_test):
            if os.path.isfile(os.path.join(dir_before_test, filename)):
                os.remove(os.path.join(dir_before_test, filename))

        for filename in os.listdir(dir_gtruth_test):
            if os.path.isfile(os.path.join(dir_gtruth_test, filename)):
                os.remove(os.path.join(dir_gtruth_test, filename))

        for filename in os.listdir(dir_change_test):
            if os.path.isfile(os.path.join(dir_change_test, filename)):
                os.remove(os.path.join(dir_change_test, filename))

        if parametersObject["optimisation_algorithm"] == "pso":
            with open('{}/{}/PSO Train Info about Models.txt'.format(path_to_models_temp_var,
                                                                     loaded_model_name_temp_var), 'w') as f:
                f.write("")
            with open('{}/{}/PSO_pyHopper_parameters.txt'.format(path_to_models_temp_var, loaded_model_name_temp_var),
                      "w") as pyHopper_parameters:
                pyHopper_parameters.write("")
            train_PSO.train_Thresholds_Function(generator, did_load_model, parametersObject, param=param)
        elif parametersObject["optimisation_algorithm"] == "cma":
            with open('{}/{}/CMA Train Info about Models.txt'.format(path_to_models_temp_var,
                                                                     loaded_model_name_temp_var), 'w') as f:
                f.write("")
            with open('{}/{}/CMA_pyHopper_parameters.txt'.format(path_to_models_temp_var, loaded_model_name_temp_var),
                      "w") as pyHopper_parameters:
                pyHopper_parameters.write("")
            train_CMA.train_Thresholds_Function(generator, did_load_model, parametersObject, param=param)
        elif parametersObject["optimisation_algorithm"] == "ga":
            with open('{}/{}/GA Train Info about Models.txt'.format(path_to_models_temp_var,
                                                                    loaded_model_name_temp_var), 'w') as f:
                f.write("")
            with open('{}/{}/GA_pyHopper_parameters.txt'.format(path_to_models_temp_var, loaded_model_name_temp_var),
                      "w") as pyHopper_parameters:
                pyHopper_parameters.write("")
            train_GA.train_Thresholds_Function(generator, did_load_model, parametersObject, param=param)
    print("Threshold training complete.")


def train_pyhopper(number_of_epochs):
    seed = np.random.randint(0, 2 ** 12 - 1)

    search = pyhopper.Search(
        {
            "threshold_one_pyhopper": pyhopper.float(1),
            "threshold_two_pyhopper": pyhopper.float(1),
            "threshold_three_pyhopper": pyhopper.float(1),
            "threshold_four_pyhopper": pyhopper.float(1),
            "threshold_five_pyhopper": pyhopper.float(1),
            "seed": seed
        }
    )

    best_params = search.run(
        train_method_pyhopper,
        "maximize",
        steps=number_of_epochs,
        quiet=True,
    )
    return best_params


def pyHopper_hp_pso(number_of_epochs):
    seed = np.random.randint(0, 2 ** 12 - 1)

    search = pyhopper.Search(
        {
            "c1": pyhopper.float(1),
            "c2": pyhopper.float(1),
            "w": pyhopper.float(1),
            "seed": seed
        }
    )

    best_params = search.run(
        testPyHopper,
        "maximize",
        steps=number_of_epochs,
        quiet=True,
    )
    return best_params


def pyHopper_hp_ga(number_of_epochs):
    seed = np.random.randint(0, 2 ** 12 - 1)

    search = pyhopper.Search(
        {
            "parent_selection_type": pyhopper.choice("sss", "rws", "sus", "random", "tournament", "rank"),
            "crossover_type": pyhopper.choice("single_point", "two_points", "uniform", "scattered"),
            "mutation_type": pyhopper.choice("random", "swap", "scramble", "inversion"),  # ,
            "seed": seed
        }
    )

    best_params = search.run(
        train_GA.testPyHopper,
        "maximize",
        steps=number_of_epochs,
        quiet=True,
    )
    return best_params


def pyHopper_hp_cma(number_of_epochs):
    seed = np.random.randint(0, 2 ** 12 - 1)

    search = pyhopper.Search(
        {
            "sigma0": pyhopper.float(1),
            "seed": seed
        }
    )

    best_params = search.run(
        train_CMA.testPyHopper,
        "maximize",
        steps=number_of_epochs,
        quiet=True,
    )
    return best_params
