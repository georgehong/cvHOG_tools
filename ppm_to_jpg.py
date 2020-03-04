from PIL import Image
import os


for root, dirs, files in os.walk("./hog_pos", topdown=False):
    for name in files:
        file_path = os.path.join(root, name)
        im = Image.open(file_path)
        im.save(file_path[:-4] + str(".jpg"))