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
        self.show()

        
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.pipe = self.pipe.to("mps")
        self.pipe.enable_attention_slicing()

        
    def UiComponents(self):
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(280,40)


        button = QPushButton("CLICK", self)
        button.setGeometry(200, 150, 100, 40)
        button.clicked.connect(self.clickme)
        text = button.text()
 
    def clickme(self):
        textboxValue = self.textbox.text()
        print(textboxValue)
        print("pressed")
        
        self.createImage(textboxValue)
        
    def createImage(self, prompt):
        # prompt = "airplane and clouds"
        n_prompt = "bad fingers"

        # warmup for mac
        self.pipe(prompt, negative_prompt=n_prompt,  num_inference_steps=1)

        image = self.pipe(prompt, negative_prompt=n_prompt).images[0]

        home = expanduser("~")
        image.save(home + "/episode1.jpg")

        
App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
