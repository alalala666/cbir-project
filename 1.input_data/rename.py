import os

# path=("要重新命名的資料夾") 
# entries = os.listdir(path)
# print("path = " + path)

#單個資料夾 
#年分.jpg
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

#把經緯度加到圖片名稱 
#經緯度_年分.jpg
def rename_location(pic_path,dname):
    print(pic_path+"\n"+dname)
    entries = os.listdir(pic_path)
    for i in entries:
        if(i[-3:] != 'jpg'):
              continue
        old_name = pic_path + i
        new_name = path +dname +"/" +dname+ "_" +(old_name[-8:])[:4] + ".jpg"
        print(new_name)
        os.rename(old_name, new_name)
    print("----------rename pic_path = " + pic_path +" ----------") 


#全部的資料夾
def rename_all(path):
    for filename in os.listdir(path):
        rename(path + filename +"/")
        rename_location(path + filename +"/",filename)

# path = "C:/Users/alalala/Downloads/dataset/"
# rename_all(path)

print("------all finish-------")      