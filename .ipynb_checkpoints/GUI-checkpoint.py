import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg


# Class to create a mainwindow
class MainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()

        # Add a title to MainWindow
        self.setWindowTitle("Professional Chess Match Outcome Prediction")

        # set layout to vertical
        self.setLayout(qtw.QVBoxLayout())

        # Create Label for title, set font and add it to the MainWIndow
        title = qtw.QLabel("Chess Match Outcome Prediction")
        title.setFont(qtg.QFont('Consolas', 20))
        self.layout().addWidget(title)

        # Create White Username Entry Boxes
        white_username_entry = qtw.QLineEdit()
        white_username_entry.setObjectName("white_username")
        white_username_entry.setText("Enter White Username")
        self.layout().addWidget(white_username_entry)

        # Create Black Username Entry Boxes
        black_username_entry = qtw.QLineEdit()
        black_username_entry.setObjectName("black_username")
        black_username_entry.setText("Enter Black Username")
        self.layout().addWidget(black_username_entry)

        # Create Game Type Entry Boxes
        game_type_entry = qtw.QLineEdit()
        game_type_entry.setObjectName("game_type")
        game_type_entry.setText("Enter Game Type, Blitz or Bullet")
        self.layout().addWidget(game_type_entry)
        
        # Create a button
        go_button = qtw.QPushButton("Go!")
        self.layout().addWidget(go_button)
        



        

        # Show the Main Window
        self.show()


            
    

app = qtw.QApplication([])
mw = MainWindow()

# Run the App
app.exec_()