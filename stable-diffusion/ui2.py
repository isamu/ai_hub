# pyinstaller ui.py --onefile --noconsole

import fcntl

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import sys
import signal

# from os.path import expanduser

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
 
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
 
        self.setWindowTitle("Python ")
        self.setGeometry(100, 100, 300, 200)
        self.UiComponents()

        ## image
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

        # w/h
        self.resize(900, 1200)

        
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

        # home = expanduser("~")
        fileName = "/tmp/ui-episode2.jpg"

        self.load()
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
        self.pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0].save(fileName)

        
        image = QImage(fileName)
        pixmap = QPixmap.fromImage(image)
        pixmap2 = pixmap.scaledToWidth(800)
        self.imageLabel.setPixmap(pixmap2)
        self.imageLabel.move(20, 150)

        self.imageLabel.adjustSize()


    def load(self):
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("mps", torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="mps"))
        self.pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("mps")
    

        
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
