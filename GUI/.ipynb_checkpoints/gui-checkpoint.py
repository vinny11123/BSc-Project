
from PyQt5 import QtCore, QtGui, QtWidgets
import requests
import torch
import os
import numpy as np
from torch import nn

class EvenWiderSixLayerFCNetworkHighDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(21, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Professional Chess Match Outcome Predictor")
        MainWindow.resize(1750, 1000)
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(825, 420, 101, 41))
        self.pushButton.setObjectName("pushButton")
        
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(605, 10, 541, 281))
        font = QtGui.QFont()
        font.setPointSize(20)
        
        self.title.setFont(font)
        self.title.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.title.setObjectName("title")
        
        self.white_username = QtWidgets.QLineEdit(self.centralwidget)
        self.white_username.setGeometry(QtCore.QRect(729, 300, 291, 20))
        self.white_username.setObjectName("white_username")
        self.white_username.setPlaceholderText("Enter White Player Username....")
        
        self.black_username = QtWidgets.QLineEdit(self.centralwidget)
        self.black_username.setGeometry(QtCore.QRect(729, 330, 291, 20))
        self.black_username.setObjectName("black_username")
        self.black_username.setPlaceholderText("Enter Black Player Username....")
        
        self.blitz_viewer = QtWidgets.QListView(self.centralwidget)
        self.blitz_viewer.setGeometry(QtCore.QRect(610, 500, 265, 281))
        self.blitz_viewer.setObjectName("blitz_viewer")
        self.blitz_viewer.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        
        self.bullet_viewer = QtWidgets.QListView(self.centralwidget)
        self.bullet_viewer.setGeometry(QtCore.QRect(880, 500, 265, 281))
        self.bullet_viewer.setObjectName("bullet_viewer")
        self.bullet_viewer.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1750, 21))
        self.menubar.setObjectName("menubar")
        
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.on_button_press)

        self.model = EvenWiderSixLayerFCNetworkHighDropout()
        self.model.load_state_dict(torch.load("model5_best_epoch5259"))
        self.model.eval()


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Professional Chess Match Outcome Prediction"))
        self.pushButton.setText(_translate("MainWindow", "Go!"))
        self.title.setText(_translate("MainWindow", "Professional Chess Match Outcome Prediction"))

    def on_button_press(self):
        # check the names entered into the username boxes
        white_username_text = self.white_username.text()
        black_username_text = self.black_username.text()
        
        # Clear the entered text after button is pressed
        self.clear_line_edits()

        # Obtain whites data.
        white_data = self.get_player_data(white_username_text)

        # Obtain blacks data.
        black_data = self.get_player_data(black_username_text)

        
        # No data for white player
        if len(white_data) == 1:
            QtWidgets.QMessageBox.critical(None, "Error", f"No data found for white username {white_username_text}")

        # No data for black player
        elif len(black_data) == 1:
            QtWidgets.QMessageBox.critical(None, "Error", f"No data found for black username {black_username_text}")
            

        else:# Run as normal and fill the viewers with the information
            
            # Create the sample, excluding the game type, 0 or 2, blitz or bullet
            sample = white_data + black_data

            # Returns the blitz and bullet predictions.
            blitz_results, bullet_results = self.get_match_results(sample)

            class_dict = {0: "Draw", 1: "White Win", 2: "Black Win"}

            # Display the retrieved text in the QListView widgets
            blitz_model = QtGui.QStandardItemModel()
            blitz_model.appendRow(QtGui.QStandardItem(f"Game Type: Blitz"))
            blitz_model.appendRow(QtGui.QStandardItem(f"White Username: {white_username_text}"))
            blitz_model.appendRow(QtGui.QStandardItem(f"Black Username: {black_username_text}"))
            blitz_model.appendRow(QtGui.QStandardItem(f"Predicted: {class_dict[blitz_results[3]]}"))
            blitz_model.appendRow(QtGui.QStandardItem(f"Draw prob: {blitz_results[0]:.2f}"))
            blitz_model.appendRow(QtGui.QStandardItem(f"White win prob: {blitz_results[1]:.2f}"))
            blitz_model.appendRow(QtGui.QStandardItem(f"Black win prob: {blitz_results[2]:.2f}"))
            self.blitz_viewer.setModel(blitz_model)
            
            bullet_model = QtGui.QStandardItemModel()
            bullet_model.appendRow(QtGui.QStandardItem(f"Game Type: Bullet"))
            bullet_model.appendRow(QtGui.QStandardItem(f"White Username: {white_username_text}"))
            bullet_model.appendRow(QtGui.QStandardItem(f"Black Username: {black_username_text}"))
            bullet_model.appendRow(QtGui.QStandardItem(f"Predicted: {class_dict[bullet_results[3]]}"))
            bullet_model.appendRow(QtGui.QStandardItem(f"Draw prob: {bullet_results[0]:.2f}"))
            bullet_model.appendRow(QtGui.QStandardItem(f"White win prob: {bullet_results[1]:.2f}"))
            bullet_model.appendRow(QtGui.QStandardItem(f"Black win prob: {bullet_results[2]:.2f}"))
            self.bullet_viewer.setModel(bullet_model)
  
    
    # Takes in a sample, modifies the sample, returns the predicted results for blitz and bullet games
    def get_match_results(self, sample):
        
        # Append the correct label, 0 for blitz, 2 for bullet.
        blitz_sample = [i for i in sample]
        bullet_sample = [i for i in sample]

        # Append correct labels to each of the lists.
        blitz_sample.append(0)
        bullet_sample.append(2)

        # turn both lists in tensors
        blitz_sample = torch.Tensor([blitz_sample])
        bullet_sample = torch.Tensor([bullet_sample])

        # feed tensors into the model and get the ANNs return.
        blitz_output = self.model(blitz_sample)
        bullet_output = self.model(bullet_sample)

        # Apply the softmax function to normalise betwenn 0 - 1
        blitz_prob = nn.Softmax(dim=1)(blitz_output)
        bullet_prob = nn.Softmax(dim=1)(bullet_output)

        blitz_final_list = blitz_prob.tolist()[0]
        bullet_final_list = bullet_prob.tolist()[0]

        blitz_final_list.append(blitz_prob.argmax(1).tolist()[0])
        bullet_final_list.append(bullet_prob.argmax(1).tolist()[0])

        return blitz_final_list, bullet_final_list


    # Clear the line edit boxes
    def clear_line_edits(self):
        white_username_text = self.white_username.clear()
        black_username_text = self.black_username.clear()


    # Method to return data for two users, white username and black username.
    # If user cannot be found, then returns empty list.
    def get_player_data(self, username):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)  Chrome/124.0.0.0 Safari/537.36'}

        data_list = []
        try:
            # Get white players statistics
            res_player_stats = requests.get(f'https://api.chess.com/pub/player/{username}/stats', headers = headers).json()

            data_list.append([
                res_player_stats['chess_blitz']['last']['rating'],
                res_player_stats['chess_blitz']['best']['rating'],
                res_player_stats['chess_blitz']['record']['draw'],
                res_player_stats['chess_blitz']['record']['win'],
                res_player_stats['chess_blitz']['record']['loss'],
                res_player_stats['chess_bullet']['last']['rating'],
                res_player_stats['chess_bullet']['best']['rating'],
                res_player_stats['chess_bullet']['record']['draw'],
                res_player_stats['chess_bullet']['record']['win'],
                res_player_stats['chess_bullet']['record']['loss'],
            ])
            return data_list[0]
            
        # Something went wrong, return an empty list.
        except Exception as e:
            data_list = [e]
            return data_list


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
