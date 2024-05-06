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

def train_one_epoch(epoch_index):
    running_loss = 0.

    correct = 0
    
    # Enumerate is used so that the batch index can be tracked for intra-epoch reporting
    for i, data in enumerate(training_loader):
        
        # Gets batch of training data from the DataLoader.
        features = data["features"].to(device)
        labels = data["label"].to(device)

        # labels = labels.view(1, 4)[0]
        labels = labels.long()

        # Zero gradients for every batch
        optimizer.zero_grad()

        # Make prediction
        outputs = model(features)
        pred_probab = nn.Softmax(dim=1)(outputs)
        y_pred = pred_probab.argmax(1)

        # print(y_pred)

        correct += torch.sum(y_pred == labels)
        # Computer the loss and its gradients

        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()

        #Gather data and report
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(training_loader)
    print(f"Training loss = {epoch_loss}")
    print(f"Accuracy = {correct / len(training_set)}")
    accuracy = correct / len(training_set)
    
    return epoch_loss, accuracy


if __name__ == "__main__":

    # Path to location of the chess feature sets.
    path = "D:/Programming work/University/University Year 3 work/Project/Collected_Chess_Data/chess_feature_sets"

    file_names = list_files_in_directory(path)

    dataset_dict = create_dict_pkl_files(path, file_names)

    # Display all the file names we have found
    for i in file_names:
        print(file_names)

    # Display the dictionary keys and the len of the dataset they are storing in the value.
    for i,v in dataset_dict.items():
        print(i, len(v))

    # Pick the datasets we want, here its currently gm_wgm

    # ALL AVAILABLE DATA
    # chess_data_set = dataset_dict[file_names[0]] + dataset_dict[file_names[1]] + dataset_dict[file_names[2]] + dataset_dict[file_names[3]] + dataset_dict[file_names[4]] + dataset_dict[file_names[5]] + dataset_dict[file_names[6]] + dataset_dict[file_names[7]] + dataset_dict[file_names[8]] + dataset_dict[file_names[9]] + dataset_dict[file_names[10]] + dataset_dict[file_names[11]]

    # TEST DATA
    chess_data_set = dataset_dict[file_names[3]] + dataset_dict[file_names[4]] + dataset_dict[file_names[5]] + dataset_dict[file_names[6]] + dataset_dict[file_names[7]] + dataset_dict[file_names[8]] + dataset_dict[file_names[9]] + dataset_dict[file_names[10]] + dataset_dict[file_names[11]]

    # Display the size of the current overall dataset and the amount of each class label.
    display_number_of_each_class(chess_data_set, 3)
    
    # Balance the dataset among all 3 classes
    chess_data_set = balance_datasets(chess_data_set, 3)
    
    features, labels = preprocess_data(chess_data_set)

    train, validation, test = train_validation_test_data_sets(features, labels, True)

    # check if cuda device is availabe, and print the device.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Load in the required data using dataloading
    training_set = ChessDataSet(train)
    validation_set = ChessDataSet(validation)
    test_set = ChessDataSet(test)

    print('\n')
    print('Training set has {} instances'.format(len(training_set)))
    print('validation set has {} instances'.format(len(validation_set)))
    print('test set has {} instances'.format(len(test_set)))
    
    # dataloader = DataLoader(chess_dataset, batch_size=4, shuffle=True, num_workers=0)
    training_loader = DataLoader(training_set, batch_size=100, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=20, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=20, shuffle=False)

    # create an instance of the simple fully connected neural network and send to the device.
    model = EvenWiderSixLayerFCNetwork().to(device)

    # Create the loss function.
    loss_fn = torch.nn.CrossEntropyLoss()
     
    # Optimizers specified in the torch.optim package

    #learning_rate = 0.00001
    learning_rate = 0.001
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"LEARNING RATE = {learning_rate}")

    epoch_number = 0

    # Set the number of epochs
    EPOCHS = 600

    # use for early stopping.
    best_vloss = 1_000_000.

    # store train accuracies and train losses for our model.
    train_accuracies = []
    train_losses = []

    # store validation accuracies and losses for our model.
    validation_accuracies = []
    validation_losses = []

    # iterate for the length of set epochs.
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
    
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        # training -> returns epoch loss and training accuracy.
        epoch_loss, accuracy = train_one_epoch(epoch_number)

        # append the train losses and accuracies to the created lists for graphing purposes.
        train_losses.append(epoch_loss)
        train_accuracies.append(accuracy.cpu())

        # set running validation loss to 0
        running_vloss = 0.0
        
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
    
        # Disable gradient computation and reduce memory consumption.
        validation_acc_counter = 0
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader): # for every batch within validation loader.

                # Get the features and labels, and train the model using the features.
                v_features = vdata["features"].to(device)
                v_labels = vdata["label"].to(device)
                v_outputs = model(v_features)

                # get the current loss for this batch using the models outputs and and validation labels.
                # add the vloss to the running vloss
                vloss = loss_fn(v_outputs, v_labels)
                running_vloss += vloss

                # get the predicted values from this batch and count how many the model got correct compared to the labels.
                pred_probab = nn.Softmax(dim=1)(v_outputs)
                predicted_values = pred_probab.argmax(1)
                validation_acc_counter += torch.sum(predicted_values == v_labels)


        # append the validation loss and accuracy into the validation loss and accuracy lists.
        validation_losses.append((running_vloss / len(validation_loader)).cpu())
        validation_accuracies.append((validation_acc_counter / len(validation_set)).cpu())

        print(f"Validation loss = {running_vloss / len(validation_loader)}")
        print(f"Accuracy = {validation_acc_counter / len(validation_set)}")

        if (running_vloss / len(validation_loader)).cpu() < best_vloss:
            best_vloss = (running_vloss / len(validation_loader)).cpu()
            torch.save(model.state_dict(), f"model_epoch_{epoch_number}")

        epoch_number += 1

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


    # Save the current model in its current form after X number of epochs.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = 'model_{}_{}'.format(timestamp, epoch_number)
    torch.save(model.state_dict(), model_path)

    # create a list that is [0, 1, 2, ..., number of epochs]
    epoch_list = list(range(1, EPOCHS+1))
    
    # Create separate plots for accuracy and loss
    plt.figure(figsize=(12, 6))

    # Plotting Accuracy
    plt.subplot(1, 2, 1)  # Subplot 1 for Accuracy
    plt.plot(epoch_list, [i * 100 for i in train_accuracies], label='Training Accuracy', color='blue', marker='o')
    plt.plot(epoch_list, [i * 100 for i in validation_accuracies], label='Validation Accuracy', color='orange', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracies over Epochs')
    plt.grid(True)
    plt.legend()
    
    # Plotting Loss
    plt.subplot(1, 2, 2)  # Subplot 2 for Loss
    plt.plot(epoch_list, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epoch_list, validation_losses, label='Validation Loss', color='orange', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses over Epochs')
    plt.grid(True)
    plt.legend()
    
    # Show the combined subplots
    plt.tight_layout()
    plt.show()
    






    






