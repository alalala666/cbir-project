from PIL import Image
import os

path=("C:/Users/alalala/Downloads/dataset/139,35/") 
entries = os.listdir(path)
print("path = " + path)
pic_name = path[35::]
pic_name = pic_name[:-1]
print(pic_name)

for i in entries:
    if(i[-3:] != 'png'):
        continue
    old_name = path + i
    new_name = (old_name[-12:])[:4] + ".jpg"
    print(new_name)
    os.rename(old_name, new_name)
