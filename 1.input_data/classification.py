import os

path=("C:/git/cbir-project/2.cbir/dataset/")  

for filename in os.listdir(path):

    entries = os.listdir(path + filename)
    for i in entries:
        #print(i)
        if(i[-3:] != 'jpg'):
            continue
        old_name = path + filename+"/" + i
        new_name = path + i
        #print(old_name)
        os.rename(old_name, new_name)
        #print(old_name)
    #print("----------rename pic_path = " + path +" ----------")

#print("----------rename pic_path = " + pic_path +" ----------")