# pyinstaller ui.py --onefile --noconsole

import fcntl

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from diffusers import StableDiffusionPipeline

import sys
import signal

from os.path import expanduser
 


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
 
        self.setWindowTitle("Python ")
        self.setGeometry(100, 100, 600, 400)
        self.UiComponents()


        ## image
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

        self.resize(600, 800)

        self.show()

        

        
    def UiComponents(self):
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(280,40)


        button = QPushButton("CLICK", self)
        button.setGeometry(20, 70, 100, 40)
        button.clicked.connect(self.clickme)
        text = button.text()
 
    def clickme(self):
        textboxValue = self.textbox.text()
        print(textboxValue)
        print("pressed")
        
        self.createImage(textboxValue)
        
    def createImage(self, prompt):
        self.load()
        n_prompt = "bad fingers"
        self.pipe(prompt, negative_prompt=n_prompt,  num_inference_steps=1)

        image = self.pipe(prompt, negative_prompt=n_prompt).images[0]
        
        # self.canvas.setImage(image)
        
        home = expanduser("~")
        fileName = home + "/episode1.jpg"
        image.save(fileName)

        image = QImage(fileName)
        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.imageLabel.move(20, 150)

        self.imageLabel.adjustSize()


        
    def load(self):
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.pipe = self.pipe.to("mps")
        self.pipe.enable_attention_slicing()


def main():
    App = QApplication(sys.argv)
    window = Window()
    sys.exit(App.exec())


if __name__ == '__main__':
    with open('/tmp/ui_lockfile', 'w') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            print('Another instance is running')
            exit(0)
        try:
            main()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
