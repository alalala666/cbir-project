import os

# path=("要重新命名的資料夾") 
# entries = os.listdir(path)
# print("path = " + path)

#單個資料夾
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

#全部的資料夾
def rename_all(path):
    for filename in os.listdir(path):
        rename(path + filename +"/")

#rename_all("C:/git/cbir-project/2.cbir/dataset/")
#rename("C:\git\cbir-project/2.cbir\dataset\-62,-17\-62,-17/")
print("------all finish-------")      