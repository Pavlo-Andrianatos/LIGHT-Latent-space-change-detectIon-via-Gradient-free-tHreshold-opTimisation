import copy
import torchvision.utils as v_utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.backends import cudnn
from Unet import *
import os
import numpy as np
import imageio
import pandas
import datetime
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, data_frame, root_dir):
        self.data_frame = data_frame
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        transform = transforms.Compose([transforms.ToTensor(), ])

        image_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(image_name)
        image = transform(image)

        ground_truth_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])
        ground_truth = Image.open(ground_truth_name)

        ground_truth = torch.from_numpy(np.array(ground_truth))
        ground_truth = ground_truth.long()

        new_target = torch.empty_like(ground_truth)

        # Uncomment below when training CNN for desired dataset
        # Below for Vaihingen

        # new_target[ground_truth == 29] = 0
        # new_target[ground_truth == 150] = 1
        # new_target[ground_truth == 255] = 2

        # Below for HRSCD

        new_target[ground_truth == 50] = 0
        new_target[ground_truth == 100] = 1
        new_target[ground_truth == 150] = 2
        new_target[ground_truth == 200] = 3
        new_target[ground_truth == 250] = 4

        # Below for SyntheWorld

        # new_target[ground_truth == 1] = 0
        # new_target[ground_truth == 2] = 1
        # new_target[ground_truth == 3] = 2
        # new_target[ground_truth == 4] = 3
        # new_target[ground_truth == 5] = 4
        # new_target[ground_truth == 6] = 5
        # new_target[ground_truth == 7] = 6
        # new_target[ground_truth == 8] = 7

        return image, new_target


def displayGenerated(y_argmax, i, k, loaded_model_name, load_models_file_path, image_size):
    # y_argmax = y_argmax.squeeze(0)
    length = y_argmax.shape[1]
    width = y_argmax.shape[2]
    gen = y_argmax.cpu().numpy()
    gen = np.reshape(gen, (image_size, image_size, 1))

    dis = np.zeros((length, width, 3), dtype=np.uint8)

    # Uncomment below when training CNN for desired dataset

    # Below for Vaihingen

    # dis[np.where((gen == 0).all(axis=2))] = [255, 0, 0]
    # dis[np.where((gen == 1).all(axis=2))] = [0, 255, 0]
    # dis[np.where((gen == 2).all(axis=2))] = [100, 100, 100]

    # Below for HRSCD

    dis[np.where((gen == 0).all(axis=2))] = [255, 0, 0]
    dis[np.where((gen == 1).all(axis=2))] = [0, 255, 0]
    dis[np.where((gen == 2).all(axis=2))] = [34, 139, 34]
    dis[np.where((gen == 3).all(axis=2))] = [55, 36, 24]
    dis[np.where((gen == 4).all(axis=2))] = [0, 0, 255]

    # Below for SyntheWorld

    # dis[np.where((gen == 0).all(axis=2))] = [225, 225, 85]
    # dis[np.where((gen == 1).all(axis=2))] = [65, 200, 75]
    # dis[np.where((gen == 2).all(axis=2))] = [50, 50, 50]
    # dis[np.where((gen == 3).all(axis=2))] = [150, 150, 150]
    # dis[np.where((gen == 4).all(axis=2))] = [25, 100, 0]
    # dis[np.where((gen == 5).all(axis=2))] = [0, 0, 255]
    # dis[np.where((gen == 6).all(axis=2))] = [0, 255, 0]
    # dis[np.where((gen == 7).all(axis=2))] = [255, 0, 0]

    imageio.imwrite("{}/{}/result/train/model/argmax/argmax_{}_{}.png".format(load_models_file_path,
                                                                              loaded_model_name,
                                                                              i, k), dis)


def displayTruth(y_, i, k, loaded_model_name, load_models_file_path, image_size):
    truth = y_.cpu().numpy()
    truth = np.reshape(truth, (image_size, image_size, 1))
    length = truth.shape[0]
    width = truth.shape[1]

    dis = np.zeros((length, width, 1), dtype=np.uint8)

    # Uncomment below when training CNN for desired dataset
    # Below for Vaihingen

    # dis[truth == 0] = 29
    # dis[truth == 1] = 150
    # dis[truth == 2] = 255

    # Below for HRSCD

    dis[truth == 0] = 50
    dis[truth == 1] = 100
    dis[truth == 2] = 150
    dis[truth == 3] = 200
    dis[truth == 4] = 250

    # Below for SyntheWorld

    # dis[truth == 0] = 240
    # dis[truth == 1] = 210
    # dis[truth == 2] = 60
    # dis[truth == 3] = 90
    # dis[truth == 4] = 180
    # dis[truth == 5] = 150
    # dis[truth == 6] = 180
    # dis[truth == 7] = 30

    imageio.imwrite("{}/{}/result/train/model/gtruth/gtruth_{}_{}.png".format(load_models_file_path,
                                                                              loaded_model_name,
                                                                              i, k), dis)


def trainModelFunction(generator, did_load_model, parametersObject):
    load_models_file_path = parametersObject["path_to_models"]  # Path to CNN
    loaded_model_name = parametersObject["loaded_model_name"]  # Name of CNN
    number_of_classes = parametersObject["number_of_classes"]
    learning_rate = parametersObject["learning_rate"]
    epochs = parametersObject["number_of_epochs"]
    csv_file_to_use = parametersObject["csv_file_to_use"]
    training_ratio = parametersObject["training_ratio"]
    path_to_usable_images = parametersObject["path_to_usable_images"]  # Path
    batch_size = parametersObject["batch_size"]  # Always 1
    output_every_x_images = parametersObject["output_every_x_images"]  # number of images
    output_images = parametersObject["output_images"]  # True or False
    image_size = parametersObject["image_size"]  # 320 or 512
    image_channels = parametersObject["image_channels"]

    torch.backends.cudnn.benchmark = True
    print("Training is commencing...")
    try:
        with open('{}/{}/Train Info about Models.txt'.format(load_models_file_path,
                                                             loaded_model_name), 'a') as f:
            f.write("Training started, Datetime: " + datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + "\n")
    except FileNotFoundError:
        print("Could not find Train Info about Models.txt file for models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(UnetGenerator(image_channels, number_of_classes), device_ids=[i for i in range(1)]).to(
        device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if did_load_model:
        model.load_state_dict(generator['model_state_dict'])
        optimizer.load_state_dict(generator['optimizer_state_dict'])
        start_epoch = generator['epoch'] + 1
    else:
        start_epoch = 0

    model.train(True)
    model.to(device)
    end_epoch = start_epoch + epochs

    train_dataframe = pandas.read_csv(csv_file_to_use)

    image_ids_for_dataframe = train_dataframe['image'].unique()

    # Split data set into training and generalization sets
    trainingRatio = int(training_ratio * len(image_ids_for_dataframe))

    train_ids = image_ids_for_dataframe[:trainingRatio]
    valid_ids = image_ids_for_dataframe[trainingRatio:]

    train_dataframeSplit = train_dataframe[train_dataframe['image'].isin(train_ids)]

    valid_dataframeSplit = train_dataframe[train_dataframe['image'].isin(valid_ids)]

    # Creates two data sets that will look in the directories passed to it for images
    train_dataset = CustomDataset(train_dataframeSplit, path_to_usable_images)

    validation_dataset = CustomDataset(valid_dataframeSplit, path_to_usable_images)

    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4 * torch.cuda.device_count(),
                                   pin_memory=True,
                                   )

    validation_data_loader = DataLoader(dataset=validation_dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=4 * torch.cuda.device_count(),
                                        pin_memory=True
                                        )

    torch.autograd.set_detect_anomaly(mode=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)
    torch.autograd.profiler.profile(enabled=False)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer.zero_grad(set_to_none=True)

    for i in range(start_epoch, end_epoch):
        print("Training, epoch: {}".format(i))
        train_loss = 0.0

        confTrain = ConfusionMatrix(number_of_classes)

        for l_index, (image, ground_truth) in enumerate(train_data_loader):
            x = image.to(device, non_blocking=True)
            y_ = ground_truth.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                y = model.forward(x)
                loss = loss_function(y, y_)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            y_argmax = torch.argmax(y, dim=1)

            confTrain += torch.flatten(y_argmax), torch.flatten(ground_truth.to(device))

        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, '{}/{}/{}_{}.pkl'.format(load_models_file_path, loaded_model_name, loaded_model_name,
                                    str(i)))

        with torch.no_grad():
            print("Completed Training, epoch: {}".format(i))
            print("Validating, epoch: {}".format(i))
            model.train(False)
            k = 0
            valid_loss = 0.0
            confVal = ConfusionMatrix(number_of_classes)

            for l_index, (image, ground_truth) in enumerate(validation_data_loader):
                x = image.to(device, non_blocking=True)
                y_ = ground_truth.to(device, non_blocking=True)
                y = model.forward(x)

                valid_loss_temp = loss_function(y, y_)

                valid_loss += valid_loss_temp.item()

                y_argmax = torch.argmax(copy.deepcopy(y), dim=1)

                confVal += torch.flatten(y_argmax), torch.flatten(ground_truth.to(device))

                if k % int(output_every_x_images) == 0 and output_images == "True":
                    y_argmax = torch.argmax(y, dim=1)
                    displayGenerated(y_argmax, i, k, loaded_model_name, load_models_file_path, image_size)
                    v_utils.save_image(x.cpu().data,
                                       "{}/{}/result/train/model/original/original_image_{}_{}.png".format(
                                           load_models_file_path,
                                           loaded_model_name,
                                           i, k))
                    displayTruth(y_, i, k, loaded_model_name, load_models_file_path, image_size)
                k = k + 1

        model.train(True)

        train_loss = train_loss / len(train_data_loader.sampler)
        valid_loss = valid_loss / len(validation_data_loader.sampler)

        # Convert the confusion matrix tensors to numpy arrays
        conf_train_np = confTrain.value.cpu().numpy()
        conf_val_np = confVal.value.cpu().numpy()

        # Initialize confusion matrix metrics
        train_tp = train_fn = train_fp = train_tn = 0
        valid_tp = valid_fn = valid_fp = valid_tn = 0

        # Loop over all classes to compute metrics
        for d in range(number_of_classes):
            # Training confusion matrix components
            train_tp += conf_train_np[d, d]
            train_fp += np.sum(conf_train_np[:, d]) - conf_train_np[d, d]
            train_fn += np.sum(conf_train_np[d, :]) - conf_train_np[d, d]
            train_tn += np.sum(conf_train_np) - (train_tp + train_fp + train_fn)

            # Validation confusion matrix components
            valid_tp += conf_val_np[d, d]
            valid_fp += np.sum(conf_val_np[:, d]) - conf_val_np[d, d]
            valid_fn += np.sum(conf_val_np[d, :]) - conf_val_np[d, d]
            valid_tn += np.sum(conf_val_np) - (valid_tp + valid_fp + valid_fn)

        train_accuracy = (train_tp + train_tn) / (train_tp + train_fp + train_fn + train_tn)
        train_sensitivity = train_tp / (train_tp + train_fn)
        train_specificity = train_tn / (train_tn + train_fp)
        train_precision = train_tp / (train_tp + train_fp)
        train_f1score = 2 * ((train_precision * train_sensitivity) / (train_precision + train_sensitivity))

        valid_accuracy = (valid_tp + valid_tn) / (valid_tp + valid_fp + valid_fn + valid_tn)
        valid_sensitivity = valid_tp / (valid_tp + valid_fn)
        valid_specificity = valid_tn / (valid_tn + valid_fp)
        valid_precision = valid_tp / (valid_tp + valid_fp)
        valid_f1score = 2 * ((valid_precision * valid_sensitivity) / (valid_precision + valid_sensitivity))

        print("Training loss: ", train_loss)
        print("Training accuracy: ", train_accuracy)
        print("Training sensitivity: ", train_sensitivity)
        print("Training specificity: ", train_specificity)
        print("Training precision: ", train_precision)
        print("Training f1score: ", train_f1score)
        print()

        print("Validation loss: ", valid_loss)
        print("Validation accuracy: ", valid_accuracy)
        print("Validation sensitivity: ", valid_sensitivity)
        print("Validation specificity: ", valid_specificity)
        print("Validation precision: ", valid_precision)
        print("Validation f1score: ", valid_f1score)
        try:
            with open('{}/{}/Train Info about Models.txt'.format(load_models_file_path,
                                                                 loaded_model_name), 'a') as f:
                if i == start_epoch:
                    f.write("Epoch {}: Training Loss: {}\n".format(i, train_loss))
                else:
                    f.write("\n")
                    f.write("Epoch {}: Training Loss: {}\n".format(i, train_loss))
                f.write("Epoch {}: Training Accuracy: {}\n".format(i, train_accuracy))
                f.write("Epoch {}: Training Sensitivity: {}\n".format(i, train_sensitivity))
                f.write("Epoch {}: Training Specificity: {}\n".format(i, train_specificity))
                f.write("Epoch {}: Training Precision: {}\n".format(i, train_precision))
                f.write("Epoch {}: Training F1-Score: {}\n\n".format(i, train_f1score))
                f.write("Epoch {}: Validation Loss: {}\n".format(i, valid_loss))
                f.write("Epoch {}: Validation Accuracy: {}\n".format(i, valid_accuracy))
                f.write("Epoch {}: Validation Sensitivity: {}\n".format(i, valid_sensitivity))
                f.write("Epoch {}: Validation Specificity: {}\n".format(i, valid_specificity))
                f.write("Epoch {}: Validation Precision: {}\n".format(i, valid_precision))
                f.write("Epoch {}: Validation F1-Score: {}\n".format(i, valid_f1score))
        except FileNotFoundError:
            print("Could not find Train Info about Models.txt file for models, aborting...")
    try:
        with open('{}/{}/Train Info about Models.txt'.format(load_models_file_path, loaded_model_name), 'a') as f:
            f.write("Training ended, Datetime: " + datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + "\n\n")
    except FileNotFoundError:
        print("Could not find Train Info about Models.txt file for models, aborting...")


class ConfusionMatrix:
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, n_classes: int = 10):
        self._matrix = torch.zeros(n_classes * n_classes).to(self._device)
        self._n = n_classes

    def cpu(self):
        self._matrix.cpu()

    def cuda(self):
        self._matrix.cuda()

    def to(self, device: str):
        self._matrix.to(device)

    def __add__(self, other):
        if isinstance(other, ConfusionMatrix):
            self._matrix.add_(other._matrix)
        elif isinstance(other, tuple):
            self.update(*other)
        else:
            raise NotImplemented
        return self

    def update(self, prediction: torch.tensor, label: torch.tensor):
        conf_data = prediction * self._n + label
        conf = conf_data.bincount(minlength=self._n * self._n)
        self._matrix.add_(conf)

    @property
    def value(self):
        return self._matrix.view(self._n, self._n).T
