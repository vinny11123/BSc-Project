from preprocess import *
from ml_models import *
from dataloading import ChessDataSet

import os
import torch
import time
from datetime import datetime
from torch.utils.data import DataLoader
from torch import nn

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report





if __name__ == "__main__":

        # Path to location of the chess feature sets.
    path = "D:/Programming work/University/University Year 3 work/Project/Collected_Chess_Data/new_data"
    
    # Get all the file names in the the given path
    file_names = list_files_in_directory(path)

    # Create the file dictionary.
    dataset_dict = create_dict_pkl_files(path, file_names)

    # Display all the file names we have found
    for i in file_names:
        print(file_names)

        # Display the dictionary keys and the len of the dataset they are storing in the value.
    for i,v in dataset_dict.items():
        print(i, len(v))


    # First Models Data
    # chess_data_set = dataset_dict[file_names[3]] + dataset_dict[file_names[4]] + dataset_dict[file_names[5]] + dataset_dict[file_names[6]] + dataset_dict[file_names[7]] + dataset_dict[file_names[8]] + dataset_dict[file_names[9]] + dataset_dict[file_names[10]] + dataset_dict[file_names[11]]

    # New Models Data
    chess_data_set = dataset_dict[file_names[0]]
    
    # Display the size of the current overall dataset and the amount of each class label.
    display_number_of_each_class(chess_data_set, 3)


        # Balance the dataset among all 3 classes
    chess_data_set = balance_datasets(chess_data_set, 3)

    
    features, labels = preprocess_data(chess_data_set)
    

    train, validation, test = train_validation_test_data_sets(features, labels, True)


    #check if cuda device is availabe, and print the device.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # We only need a test set as we are just testing the model
    test_set = ChessDataSet(test)
    print('test set has {} instances'.format(len(test_set)))


    # Create the dataloader for testing.
    test_loader = DataLoader(test_set, batch_size=20, shuffle=False)


    # Create a version of the model that is actually being tested.
    # Old Model
    # model = EvenWiderSixLayerFCNetwork().to(device)

    # New Model
    model = EvenWiderSixLayerFCNetworkHighDropout().to(device)

    # Load the state of the saved model, weights etc.
    model.load_state_dict(torch.load("model5_best_epoch5259"))

    # set loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Set the model to evaluation.
    model.eval()


    # create variables to store test accuracy, test loss.
    test_accuracy = 0
    test_loss = 0

    # temporary variables used to get the test_accuracy and test_loss.
    t_running_loss = 0
    test_acc_counter = 0

    # accumulate predicted and true labels so we can calculate precision, recall and f1-score at the end of the testing loop.
    actual_labels = []
    predicted_labels = []


    with torch.no_grad():
        for i, tdata in enumerate(test_loader): # iterate over every batch within test_loader.
            
            # get the test features and test labels and pass the test features to the model
            t_featires = tdata["features"].to(device)
            t_labels = tdata["label"].to(device)
            t_outputs = model(t_featires)

            # calculate the current loss within this batch and add it to the t_running_vloss
            t_loss = loss_fn(t_outputs, t_labels)
            t_running_loss += t_loss

            # get the test predictions for this batch and compare the the test labels for this batch
            # adding the amount of correct predictions to test_acc_counter.
            t_pred_probab = nn.Softmax(dim=1)(t_outputs)
            t_predicted_values = t_pred_probab.argmax(1)
            test_acc_counter += torch.sum(t_predicted_values == t_labels)
            
            # Append the predicted labels and true labels to the accumulation lists
            actual_labels.extend(t_labels.tolist())
            predicted_labels.extend(t_predicted_values.tolist())


    # calculate the test_loss and the test_accuracy and print the values
    test_loss = t_running_loss / len(test_loader)
    test_accuracy = test_acc_counter / len(test_set)
    
    print(f"test loss = {test_loss}")
    print(f"test accuracy = {test_accuracy}")

    # calculate precision, recall, and f1-score per class.
    precision_per_class = precision_score(actual_labels, predicted_labels, average=None)
    recall_per_class = recall_score(actual_labels, predicted_labels, average=None)
    f1_score_per_class = f1_score(actual_labels, predicted_labels, average=None)

    # Optionally, calculate and print macro-average or weighted-average metrics. Using macro here as we will be changing
    # the dataset to balanced and imbalanced, and weighted-average accounts for dataset imbalance, but macro doesnt.
    # therefore we use macro so that we can see the difference when we actually use a imbalanced dataset vs a balanced dataset
    # of class labels.
    macro_avg_precision = precision_score(actual_labels, predicted_labels, average='macro')
    macro_avg_recall = recall_score(actual_labels, predicted_labels, average='macro')
    macro_avg_f1_score = f1_score(actual_labels, predicted_labels, average='macro')

    # Print precision, recall, and F1-score per class
    for class_idx in range(len(precision_per_class)):
        print(f"Class {class_idx}:")
        print(f"  Precision: {precision_per_class[class_idx]:.4f}")
        print(f"  Recall: {recall_per_class[class_idx]:.4f}")
        print(f"  F1-score: {f1_score_per_class[class_idx]:.4f}")

    print("\nMacro-average metrics:")
    print(f"  Precision: {macro_avg_precision:.4f}")
    print(f"  Recall: {macro_avg_recall:.4f}")
    print(f"  F1-score: {macro_avg_f1_score:.4f}")


    


    












    
        