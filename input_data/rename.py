import os

path=("C:/Users/alalala/Downloads/dataset/") 
entries = os.listdir(path)
print("path = " + path)

def rename(pic_path):
    entries = os.listdir(pic_path)
    for i in entries:
        if(i[-3:] != 'png'):
            continue
        old_name = pic_path + i
        new_name = pic_path + (old_name[-12:])[:4] + ".jpg"
        print(new_name)
        os.rename(old_name, new_name)
    print("----------rename pic_path = " + pic_path +" ----------") 

for filename in os.listdir(path):
    rename(path + filename +"/")

print("------all finish-------")      