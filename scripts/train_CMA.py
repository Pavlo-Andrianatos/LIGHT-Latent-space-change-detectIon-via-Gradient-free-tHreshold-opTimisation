import copy
import math
import os
import cma
import imageio
import pandas
import torchvision.transforms as transforms
import torchvision.utils as v_utils
from PIL import Image
from cma.fitness_transformations import Function
from torch.utils.data import DataLoader, Dataset
import Utils
import loadParameters
import modelGeneration
from Unet import *

# Change to True if you want to use static baseline thresholds [0.4, 0.6, 0.8, 1.0, 1.2]
is_running_baseline = False


class CustomDataset(Dataset):
    def __init__(self, data_frame, root_dir, image_channels):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.image_channels = image_channels

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        transform = transforms.Compose([transforms.ToTensor(), ])

        image_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        if self.image_channels == 1:
            image = Image.open(image_name).convert('L')
        else:
            image = Image.open(image_name).convert('RGB')
        image = transform(image)

        second_image_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])
        if self.image_channels == 1:
            second_image = Image.open(second_image_name).convert('L')
        else:
            second_image = Image.open(second_image_name).convert('RGB')
        second_image = transform(second_image)

        change_image_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 2])
        change_image = Image.open(change_image_name).convert('L')
        change_image = transform(change_image)

        return image, second_image, change_image


def testPyHopper(param):
    parametersObject = loadParameters.load_Parameters_Type("train_threshold")
    generator, did_load_model = modelGeneration.loadOrGenerateModel(
        loaded_model_name=parametersObject["loaded_model_name"],
        load_models_file_path=parametersObject["path_to_models"])

    return train_Thresholds_Function(generator, did_load_model, parametersObject, param)


def displayGenerated(model_output, i, k, loaded_model_name, load_models_file_path, dataset_used, semantic=False,
                     semantic_number=0, change_image_pso=None, test_change=False):
    length = model_output.shape[2]
    width = model_output.shape[3]

    y_argmax_val = torch.argmax(model_output.cpu().detach().data, dim=1)

    y_argmax_val = y_argmax_val.numpy()

    y_argmax_val = y_argmax_val.astype(np.uint8)

    image_to_save_classify = np.zeros((length, width, 3), dtype=np.uint8)

    if dataset_used == "vaihingen":
        y_argmax_val = y_argmax_val.squeeze(0)

        image_to_save_classify[y_argmax_val == 0] = [255, 0, 0]
        image_to_save_classify[y_argmax_val == 1] = [0, 255, 0]
        image_to_save_classify[y_argmax_val == 2] = [100, 100, 100]

    if dataset_used == "syn":
        y_argmax_val = y_argmax_val.squeeze(0)

        image_to_save_classify[y_argmax_val == 0] = [225, 225, 85]
        image_to_save_classify[y_argmax_val == 1] = [65, 200, 75]
        image_to_save_classify[y_argmax_val == 2] = [50, 50, 50]
        image_to_save_classify[y_argmax_val == 3] = [150, 150, 150]
        image_to_save_classify[y_argmax_val == 4] = [25, 100, 0]
        image_to_save_classify[y_argmax_val == 5] = [0, 0, 255]
        image_to_save_classify[y_argmax_val == 6] = [0, 255, 0]
        image_to_save_classify[y_argmax_val == 7] = [255, 0, 0]

    if dataset_used == "french":
        y_argmax_val = y_argmax_val.squeeze(0)

        image_to_save_classify[y_argmax_val == 0] = [255, 0, 0]
        image_to_save_classify[y_argmax_val == 1] = [0, 255, 0]
        image_to_save_classify[y_argmax_val == 2] = [34, 139, 34]
        image_to_save_classify[y_argmax_val == 3] = [55, 36, 24]
        image_to_save_classify[y_argmax_val == 4] = [0, 0, 255]

    if not semantic:
        if not test_change:
            path = "{}/{}/result/train/threshold/cma/argmax/argmax_{}_{}.png".format(
                load_models_file_path, loaded_model_name, i, k)
        else:
            path = "{}/{}/result/test/threshold/cma/argmax/argmax_{}_{}.png".format(
                load_models_file_path, loaded_model_name, i, k)

        image = Image.fromarray(image_to_save_classify)

        image.save(path)

        if not test_change:
            path = "{}/{}/result/train/threshold/cma/predicted_change/predicted_change_{}_{}.png".format(
                load_models_file_path, loaded_model_name, i, k)
        else:
            path = "{}/{}/result/test/threshold/cma/predicted_change/predicted_change_{}_{}.png".format(
                load_models_file_path, loaded_model_name, i, k)

        image = Image.fromarray(image_to_save_classify)

        image = image.convert('L')

        temp_array = np.array(image)

        if dataset_used == "vaihingen":

            most_frequent = np.bincount(y_argmax_val.ravel()).argmax()

            temp_array = ((y_argmax_val != most_frequent) * 255).astype(np.uint8)

            temp_array = np.squeeze(temp_array)
        elif dataset_used == "french":
            temp_array = (temp_array <= 128).astype(np.uint8) * 255
        elif dataset_used == "syn":
            temp_array[:] = 0
            temp_array[y_argmax_val == 7] = 255
        else:
            change_image_pso = change_image_pso.cpu().detach().numpy()

            change_image_pso = change_image_pso.squeeze(0)

            change_image_pso[change_image_pso == 1] = 0
            change_image_pso[change_image_pso > 0] = 1

            change_image_pso = change_image_pso.astype(np.uint8)

            change_image_pso = change_image_pso.squeeze(0)

            temp_array[change_image_pso == 0] = 0
            temp_array[temp_array == 2] = 1

        image = Image.fromarray(temp_array)

        image.save(path)
    else:
        if not test_change:
            path = "{}/{}/result/train/threshold/cma/argmax/argmax_semantic_{}_{}_{}.png".format(
                load_models_file_path, loaded_model_name, semantic_number, i, k)
        else:
            path = "{}/{}/result/test/threshold/cma/argmax/argmax_semantic_{}_{}_{}.png".format(
                load_models_file_path, loaded_model_name, semantic_number, i, k)

        image = Image.fromarray(image_to_save_classify)

        image.save(path)


def displayTruth(y_truth, i, k, loaded_model_name, load_models_file_path, dataset_used, test_change=False):
    length = y_truth.shape[2]
    width = y_truth.shape[3]

    y_truth = np.reshape(y_truth, (length, width, 1))

    dis = np.zeros((length, width, 1))

    if dataset_used == "vaihingen" or dataset_used == "french" or dataset_used == "syn":
        dis[y_truth == 0] = 0
        dis[y_truth != 0] = 255

    dis = dis.astype(np.uint8)

    if not test_change:
        imageio.imwrite("{}/{}/result/train/threshold/cma/gtruth/gtruth_change_{}_{}.png".format(
            load_models_file_path, loaded_model_name, i, k), dis)
    else:
        imageio.imwrite("{}/{}/result/test/threshold/cma/gtruth/gtruth_change_{}_{}.png".format(
            load_models_file_path, loaded_model_name, i, k), dis)


def train_Thresholds_Function(generator, did_load_model, parametersObject, param=None):
    path_to_models = parametersObject["path_to_models"]  # Path to CNN
    loaded_model_name = parametersObject["loaded_model_name"]  # Name of CNN
    number_of_classes = parametersObject["number_of_classes"]
    learning_rate = parametersObject["learning_rate"]  # Not used
    number_of_epochs = parametersObject["number_of_epochs"]
    csv_file_to_use = parametersObject["csv_file_to_use"]  # Path to train images
    csv_file_to_use_test = parametersObject["csv_file_to_use_test"]  # Path to test iamges
    training_ratio = parametersObject["training_ratio"]
    path_to_usable_images = parametersObject["path_to_usable_images"]  # Path to images
    output_every_x_images = parametersObject["output_every_x_images"]  # Number of images to output, not used
    output_images = parametersObject["output_images"]  # True or False
    image_channels = parametersObject["image_channels"]  # 1 or 3 (black/white and RGB)

    Utils.global_cma_iteration_number = 0

    dataset_used = parametersObject["dataset_used"]
    number_of_change_images = parametersObject["number_of_change_images"]
    fitness_function = parametersObject["fitness_function"]

    print("CMA Training is commencing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(UnetGenerator(image_channels, number_of_classes), device_ids=[i for i in range(1)]).to(
        device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if did_load_model:
        model.load_state_dict(generator['model_state_dict'])
        optimizer.load_state_dict(generator['optimizer_state_dict'])
        model.train(False)
        model.to(device)
    else:
        print("Need to load a model to perform this task")
        exit(0)

    train_dataframe = pandas.read_csv(csv_file_to_use)

    test_dataframe = pandas.read_csv(csv_file_to_use_test)

    if param is not None:
        seed = param["seed"]
    else:
        seed = np.random.randint(0, 2 ** 12 - 1)

    np.random.seed(seed)

    image_ids_for_dataframe = train_dataframe['image']

    image_ids_for_dataframe = np.random.choice(
        image_ids_for_dataframe,
        size=number_of_change_images,
        replace=False
    )

    image_ids_for_dataframe_test = test_dataframe['image']

    # Uncomment below for lower test image amount

    # image_ids_for_dataframe_test = np.random.choice(
    #     image_ids_for_dataframe_test,
    #     size=number_of_change_images,
    #     replace=False
    # )

    trainingRatio = int(training_ratio * len(image_ids_for_dataframe))

    validationRatio = int(float(training_ratio) * len(image_ids_for_dataframe))

    train_ids = image_ids_for_dataframe[:trainingRatio]

    valid_ids = image_ids_for_dataframe[trainingRatio:trainingRatio + validationRatio]

    test_ids = image_ids_for_dataframe_test

    train_dataframeSplit = train_dataframe[train_dataframe['image'].isin(train_ids)]

    valid_dataframeSplit = train_dataframe[train_dataframe['image'].isin(valid_ids)]

    test_dataframe = test_dataframe[test_dataframe['image'].isin(test_ids)]

    train_dataset = CustomDataset(train_dataframeSplit, path_to_usable_images, image_channels)

    validation_dataset = CustomDataset(valid_dataframeSplit, path_to_usable_images, image_channels)

    test_dataset = CustomDataset(test_dataframe, path_to_usable_images, image_channels)

    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=0 * torch.cuda.device_count(),
                                   pin_memory=True,
                                   )

    validation_data_loader = DataLoader(dataset=validation_dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=0 * torch.cuda.device_count(),
                                        pin_memory=True
                                        )

    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=0 * torch.cuda.device_count(),
                                  pin_memory=True
                                  )

    torch.autograd.set_detect_anomaly(mode=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)
    torch.autograd.profiler.profile(enabled=False)

    fitness_function_index = 9

    if fitness_function == "precision":
        fitness_function_index = 0
    elif fitness_function == "recall":
        fitness_function_index = 1
    elif fitness_function == "non_changed":
        fitness_function_index = 2
    elif fitness_function == "changed":
        fitness_function_index = 3
    elif fitness_function == "mIoU":
        fitness_function_index = 4
    elif fitness_function == "f1":
        fitness_function_index = 5
    elif fitness_function == "OA":
        fitness_function_index = 6
    elif fitness_function == "Kappa":
        fitness_function_index = 7
    elif fitness_function == "SeK":
        fitness_function_index = 8
    elif fitness_function == "score":
        fitness_function_index = 9

    def calculateScore(model_output, change_image_cma, dataset_used_score, all_metrics=False):
        change_image_cma = change_image_cma.cpu().detach().numpy()

        change_image_cma = change_image_cma.squeeze(0)

        change_image_cma = change_image_cma.astype(np.uint8)

        y_argmax_val = torch.argmax(model_output.cpu().detach().data, dim=1)

        y_argmax_val = y_argmax_val.numpy()

        y_argmax_val = y_argmax_val.astype(np.uint8)

        image_to_save_classify = np.zeros((model_output.shape[2], model_output.shape[3], 3), dtype=np.uint8)

        if dataset_used_score == "vaihingen":
            y_argmax_val = y_argmax_val.squeeze(0)

            image_to_save_classify[y_argmax_val == 0] = [255, 0, 0]
            image_to_save_classify[y_argmax_val == 1] = [0, 255, 0]
            image_to_save_classify[y_argmax_val == 2] = [100, 100, 100]

        if dataset_used_score == "syn":
            y_argmax_val = y_argmax_val.squeeze(0)

            image_to_save_classify[y_argmax_val == 0] = [225, 225, 85]
            image_to_save_classify[y_argmax_val == 1] = [65, 200, 75]
            image_to_save_classify[y_argmax_val == 2] = [50, 50, 50]
            image_to_save_classify[y_argmax_val == 3] = [150, 150, 150]
            image_to_save_classify[y_argmax_val == 4] = [25, 100, 0]
            image_to_save_classify[y_argmax_val == 5] = [0, 0, 255]
            image_to_save_classify[y_argmax_val == 6] = [0, 255, 0]
            image_to_save_classify[y_argmax_val == 7] = [255, 0, 0]

        if dataset_used_score == "french":
            y_argmax_val = y_argmax_val.squeeze(0)

            image_to_save_classify[y_argmax_val == 0] = [255, 0, 0]
            image_to_save_classify[y_argmax_val == 1] = [0, 255, 0]
            image_to_save_classify[y_argmax_val == 2] = [34, 139, 34]
            image_to_save_classify[y_argmax_val == 3] = [55, 36, 24]
            image_to_save_classify[y_argmax_val == 4] = [0, 0, 255]

        image_temp = Image.fromarray(image_to_save_classify)

        image_temp = image_temp.convert('L')

        temp_array = np.array(image_temp)

        if dataset_used_score == "vaihingen":
            most_frequent = np.bincount(temp_array.ravel()).argmax()

            y_argmax_val = (temp_array != most_frequent).astype(np.uint8)
        elif dataset_used_score == "french":
            y_argmax_val = (temp_array <= 128).astype(np.uint8)
        elif dataset_used_score == "syn":
            y_argmax_val[:] = (y_argmax_val == 7).astype(np.uint8)
        else:
            y_argmax_val[change_image_cma == 0] = 0
            y_argmax_val[y_argmax_val == 2] = 1

        change_image_cma = change_image_cma.transpose(1, 2, 0)

        change_image_cma = change_image_cma.squeeze(2)

        true_positives = np.sum(np.logical_and(y_argmax_val == 1, change_image_cma == 1), dtype='int64')
        true_negatives = np.sum(np.logical_and(y_argmax_val == 0, change_image_cma == 0), dtype='int64')
        false_positives = np.sum(np.logical_and(y_argmax_val == 1, change_image_cma == 0), dtype='int64')
        false_negatives = np.sum(np.logical_and(y_argmax_val == 0, change_image_cma == 1), dtype='int64')

        epsilon = 1e-7

        # Precision
        precision_pso = true_positives / (true_positives + false_positives + epsilon)

        # Recall
        recall_pso = true_positives / (true_positives + false_negatives + epsilon)

        # Intersection over Union (IoU)
        non_changed_type = true_negatives / (true_negatives + false_positives + false_negatives + epsilon)
        changed_type = true_positives / (true_positives + false_negatives + false_positives + epsilon)
        mean_intersection_over_union = 0.5 * (non_changed_type + changed_type)

        # F1 Score
        f1_score_pso = 2 * (precision_pso * recall_pso) / (precision_pso + recall_pso + epsilon)

        # Overall Accuracy
        overall_accuracy_pso = (true_positives + true_negatives) / (
                true_positives + false_positives + true_negatives + false_negatives + epsilon
        )

        # Kappa Statistic
        p_zero = (true_positives + true_negatives) / (
                true_positives + false_positives + true_negatives + false_negatives + epsilon
        )

        total = true_positives + true_negatives + false_positives + false_negatives

        p_e = ((true_positives + false_positives) * (true_positives + false_negatives) + (
                false_negatives + true_negatives) * (false_positives + true_negatives)) / (total * total)

        Kappa = (p_zero - p_e) / (1 - p_e + epsilon)

        # SeK
        SeK = Kappa * math.exp(changed_type - 1)

        # Final Score
        score = (0.3 * mean_intersection_over_union) + (0.7 * SeK)

        return_metrics = [precision_pso, recall_pso, non_changed_type, changed_type,
                          mean_intersection_over_union, f1_score_pso, overall_accuracy_pso,
                          Kappa, SeK, score]

        if all_metrics:
            return return_metrics
        else:
            return return_metrics[fitness_function_index]

    class cmaLossFunc(Function):
        def _eval(self, x):
            model.train(False)
            total_difference_score_cma = 0
            total_number_tested = 0
            for l_index_1_cma, (image_cma, second_image_cma, change_image_cma) in enumerate(train_data_loader):
                image_cma = image_cma.to(device, non_blocking=True)
                second_image_cma = second_image_cma.to(device, non_blocking=True)

                with torch.no_grad():
                    if not is_running_baseline:
                        y_cma = model.forward(inputImage=image_cma, secondImage=second_image_cma,
                                              threshold_down1=float(x[0]),
                                              threshold_down2=float(x[1]),
                                              threshold_down3=float(x[2]),
                                              threshold_down4=float(x[3]),
                                              threshold_bridge=float(x[4]))
                    if is_running_baseline:
                        y_cma = model.forward(inputImage=image_cma, secondImage=second_image_cma,
                                              threshold_down1=float(0.4),
                                              threshold_down2=float(0.6),
                                              threshold_down3=float(0.8),
                                              threshold_down4=float(1.0),
                                              threshold_bridge=float(1.2))

                score_cma = calculateScore(copy.deepcopy(y_cma), copy.deepcopy(change_image_cma),
                                           dataset_used)

                total_number_tested += 1

                total_difference_score_cma = total_difference_score_cma + score_cma

            difference_value_train = (total_difference_score_cma / total_number_tested)
            difference_value_train = (1 - difference_value_train)
            return difference_value_train

    if param is not None:
        sigma0 = param["sigma0"]
    else:
        sigma0 = 0.5

    cmaLoss = cmaLossFunc()

    threshold_values = [np.random.random_sample() for _ in range(5)]

    es = cma.CMAEvolutionStrategy(threshold_values, sigma0, {'maxiter': number_of_epochs, 'seed': seed, 'popsize': 3})

    while Utils.global_cma_iteration_number != number_of_epochs:
        X = es.ask()

        es.tell(X, [cmaLoss(x) for x in X])
        es.disp()

        model.train(False)
        total_difference_score_val = 0
        total_difference_metrics = []
        number_validated = 0
        for l_index_1_val, (image_val, second_image_val, change_image_val) in enumerate(validation_data_loader):
            image_val = image_val.to(device, non_blocking=True)
            second_image_val = second_image_val.to(device, non_blocking=True)

            with torch.no_grad():
                if not is_running_baseline:
                    y_cma_val = model.forward(inputImage=image_val, secondImage=second_image_val,
                                              threshold_down1=float(es.best.x[0]),
                                              threshold_down2=float(es.best.x[1]),
                                              threshold_down3=float(es.best.x[2]),
                                              threshold_down4=float(es.best.x[3]),
                                              threshold_bridge=float(es.best.x[4]))
                if is_running_baseline:
                    y_cma_val = model.forward(inputImage=image_val, secondImage=second_image_val,
                                              threshold_down1=float(0.4),
                                              threshold_down2=float(0.6),
                                              threshold_down3=float(0.8),
                                              threshold_down4=float(1.0),
                                              threshold_bridge=float(1.2))

                y_before = model.forward(inputImage=image_val)
                y_after = model.forward(inputImage=second_image_val)
                score_ga_val = calculateScore(copy.deepcopy(y_cma_val), copy.deepcopy(change_image_val),
                                              dataset_used, True)

            if output_images == "True":
                if number_of_classes == 3:
                    v_utils.save_image(copy.deepcopy(y_cma_val),
                                       "{}/{}/result/train/threshold/cma/argmax/gen_image_{}_{}.png".format(
                                           path_to_models,
                                           loaded_model_name,
                                           Utils.global_cma_iteration_number, l_index_1_val))

                displayGenerated(copy.deepcopy(y_cma_val), Utils.global_cma_iteration_number, l_index_1_val,
                                 loaded_model_name,
                                 path_to_models, dataset_used, change_image_pso=copy.deepcopy(change_image_val))

                displayGenerated(y_before, Utils.global_cma_iteration_number, l_index_1_val, loaded_model_name,
                                 path_to_models, dataset_used, semantic=True, semantic_number=0)
                displayGenerated(y_after, Utils.global_cma_iteration_number, l_index_1_val, loaded_model_name,
                                 path_to_models, dataset_used, semantic=True, semantic_number=1)

                displayTruth(copy.deepcopy(change_image_val), Utils.global_cma_iteration_number, l_index_1_val,
                             loaded_model_name,
                             path_to_models, dataset_used)
                v_utils.save_image(image_val.cpu().data,
                                   "{}/{}/result/train/threshold/cma/before/before_image_{}_{}.png".format(
                                       path_to_models,
                                       loaded_model_name,
                                       Utils.global_cma_iteration_number, l_index_1_val))
                v_utils.save_image(second_image_val.cpu().data,
                                   "{}/{}/result/train/threshold/cma/after/after_image_{}_{}.png".format(
                                       path_to_models,
                                       loaded_model_name,
                                       Utils.global_cma_iteration_number, l_index_1_val))

            total_difference_score_val = total_difference_score_val + score_ga_val[fitness_function_index]

            if not total_difference_metrics:
                total_difference_metrics = score_ga_val
            else:
                total_difference_metrics = [sum(x) for x in zip(total_difference_metrics, score_ga_val)]

            number_validated += 1

        difference_value_val = (total_difference_score_val / number_validated)
        total_difference_metrics = [element / float(number_validated) for element in
                                    total_difference_metrics]
        print("Validation Score: " + str(difference_value_val) + ", for Iteration: " +
              str(Utils.global_cma_iteration_number) + ", higher is better.")

        model.train(False)
        total_difference_score_test = 0
        total_difference_metrics_test = []
        number_validated_test = 0
        for l_index_1, (image_test, second_image_test, change_image_test) in enumerate(test_data_loader):
            image_test = image_test.to(device, non_blocking=True)
            second_image_test = second_image_test.to(device, non_blocking=True)

            with torch.no_grad():
                if not is_running_baseline:
                    y_test = model.forward(inputImage=image_test, secondImage=second_image_test,
                                           threshold_down1=float(es.best.x[0]),
                                           threshold_down2=float(es.best.x[1]),
                                           threshold_down3=float(es.best.x[2]),
                                           threshold_down4=float(es.best.x[3]),
                                           threshold_bridge=float(es.best.x[4]))
                if is_running_baseline:
                    y_test = model.forward(inputImage=image_test, secondImage=second_image_test,
                                           threshold_down1=float(0.4),
                                           threshold_down2=float(0.6),
                                           threshold_down3=float(0.8),
                                           threshold_down4=float(1.0),
                                           threshold_bridge=float(1.2))

                y_before_test = model.forward(inputImage=image_test)
                y_after_test = model.forward(inputImage=second_image_test)
                score_test = calculateScore(copy.deepcopy(y_test), copy.deepcopy(change_image_test),
                                            dataset_used, True)

            if output_images == "True":
                if number_of_classes == 3:
                    v_utils.save_image(copy.deepcopy(y_test),
                                       "{}/{}/result/test/threshold/cma/argmax/gen_image_{}_{}.png".format(
                                           path_to_models,
                                           loaded_model_name,
                                           Utils.global_cma_iteration_number, l_index_1))

                displayGenerated(copy.deepcopy(y_test), Utils.global_cma_iteration_number, l_index_1,
                                 loaded_model_name,
                                 path_to_models, dataset_used, change_image_pso=copy.deepcopy(change_image_test),
                                 test_change=True)

                displayGenerated(y_before_test, Utils.global_cma_iteration_number, l_index_1, loaded_model_name,
                                 path_to_models, dataset_used, semantic=True, semantic_number=0, test_change=True)
                displayGenerated(y_after_test, Utils.global_cma_iteration_number, l_index_1, loaded_model_name,
                                 path_to_models, dataset_used, semantic=True, semantic_number=1, test_change=True)

                displayTruth(copy.deepcopy(change_image_test), Utils.global_cma_iteration_number, l_index_1,
                             loaded_model_name,
                             path_to_models, dataset_used, test_change=True)
                v_utils.save_image(image_test.cpu().data,
                                   "{}/{}/result/test/threshold/cma/before/before_image_{}_{}.png".format(
                                       path_to_models,
                                       loaded_model_name,
                                       Utils.global_cma_iteration_number, l_index_1))
                v_utils.save_image(second_image_test.cpu().data,
                                   "{}/{}/result/test/threshold/cma/after/after_image_{}_{}.png".format(
                                       path_to_models,
                                       loaded_model_name,
                                       Utils.global_cma_iteration_number, l_index_1))

            total_difference_score_test = total_difference_score_test + score_test[fitness_function_index]

            if not total_difference_metrics_test:
                total_difference_metrics_test = score_test
            else:
                total_difference_metrics_test = [sum(x) for x in zip(total_difference_metrics_test, score_test)]

            number_validated_test += 1

        difference_value_test = (total_difference_score_test / number_validated_test)
        total_difference_metrics_test = [element / float(number_validated_test) for element in
                                         total_difference_metrics_test]
        print("Test Score: " + str(difference_value_test) + ", for Iteration: " +
              str(Utils.global_cma_iteration_number) + ", higher is better.\n")

        threshold_values_to_write = "Iteration {}\nthreshold_value_1 = {}\nthreshold_value_2 = {}\nthreshold_value_3 = {}\nthreshold_value_4 = {}\nthreshold_value_5 = {}\n\n".format(
            Utils.global_cma_iteration_number, es.best.x[0], es.best.x[1], es.best.x[2],
            es.best.x[3],
            es.best.x[4])
        with open('{}/{}/threshold_values_train.txt'.format(path_to_models, loaded_model_name),
                  "a") as threshold_values_file_val:
            threshold_values_file_val.write(threshold_values_to_write)

        if param is not None:
            with open('{}/{}/CMA_pyHopper_parameters.txt'.format(path_to_models, loaded_model_name),
                      "a") as pyHopper_parameters_val:
                pyHopper_parameters_val.write(str(param))
                pyHopper_parameters_val.write("\n")

        with open('{}/{}/CMA Train Info about Models.txt'.format(path_to_models, loaded_model_name),
                  'a') as f_cma_info:
            f_cma_info.write("Iteration {} Training Metrics: "
                             "\n\tSeed: {}"
                             "\n\tValidation Score: {}"
                             "\n\tTest Score: {}"
                             "\n\tPrecision: {}"
                             "\n\tRecall: {}"
                             "\n\tNon-Changed: {}"
                             "\n\tChanged: {}"
                             "\n\tmIoU: {}"
                             "\n\tF1-Score: {}"
                             "\n\tOverall Accuracy: {}"
                             "\n\tKappa: {}"
                             "\n\tSeK: {}"
                             "\n\tscore: {}"
                             "\n\n".format(Utils.global_cma_iteration_number, seed,
                                           difference_value_val, difference_value_test,
                                           total_difference_metrics[0], total_difference_metrics[1],
                                           total_difference_metrics[2], total_difference_metrics[3],
                                           total_difference_metrics[4], total_difference_metrics[5],
                                           total_difference_metrics[6], total_difference_metrics[7],
                                           total_difference_metrics[8], total_difference_metrics[9]))

            f_cma_info.write("Iteration {} Test Metrics: "
                             "\n\tSeed: {}"
                             "\n\tValidation Score: {}"
                             "\n\tTest Score: {}"
                             "\n\tPrecision: {}"
                             "\n\tRecall: {}"
                             "\n\tNon-Changed: {}"
                             "\n\tChanged: {}"
                             "\n\tmIoU: {}"
                             "\n\tF1-Score: {}"
                             "\n\tOverall Accuracy: {}"
                             "\n\tKappa: {}"
                             "\n\tSeK: {}"
                             "\n\tscore: {}"
                             "\n\n".format(Utils.global_cma_iteration_number, seed,
                                           difference_value_val, difference_value_test,
                                           total_difference_metrics_test[0], total_difference_metrics_test[1],
                                           total_difference_metrics_test[2], total_difference_metrics_test[3],
                                           total_difference_metrics_test[4], total_difference_metrics_test[5],
                                           total_difference_metrics_test[6], total_difference_metrics_test[7],
                                           total_difference_metrics_test[8], total_difference_metrics_test[9]))

        Utils.global_cma_iteration_number = Utils.global_cma_iteration_number + 1

    return total_difference_metrics_test[9]
