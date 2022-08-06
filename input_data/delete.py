import os
import shutil

fileTest = "C:/Users/alalala/Downloads/dataset"

try:
    shutil.rmtree(fileTest)
except OSError as e:
    print(e)
else:
    print("File is deleted successfully")