# from main_train_CNN import start_training
from ExperimentGatheringScript_Actual import start_testing
from main_train_CNN import start_training

trainCNNModel = False
trainThresholds = False

if __name__ == '__main__':
    if trainCNNModel and trainThresholds:
        print("Cannot train model and optimise thresholds at the same time")
        exit(0)

    if trainCNNModel:
        start_training()
    elif trainThresholds:
        start_testing()
