import os 
import csv
import shutil
from get_images import auto_get_images_1984to2012,auto_get_images_2012to2022
#from delete import del_emp_dir
from modify_image import modify_all
 
#輸入圖片路徑
download_images_path = "C:/git/cbir-project/1.input_data/dataset/"

#先把舊檔案刪除 不用就註掉
# try:
#     shutil.rmtree(download_images_path)
# except OSError as e:
#     print(e)  
# else:
#     print("File is deleted successfully")


#讀取經緯度的csv檔案
loaction_path = '1.input_data/lonlat.csv'
with open(loaction_path) as location:
    print("reading " + loaction_path + " : ")
    rows = csv.reader(location)
    #跳過第一列
    next(rows)
    next(rows)
    for i in rows:
        print ("-----lon : " + i[0] + "-----lat : "+ i[1] +"-----")
        lon = float(i[0])
        lat = float(i[1])
        
        #開始抓圖
        try: 
            auto_get_images_1984to2012(lon,lat,download_images_path)
            auto_get_images_2012to2022(lon,lat,download_images_path)  
        except:
            del_emp_dir(download_images_path)
            continue       


#複製一份 原圖留著
shutil.copytree("C:/git/cbir-project/1.input_data/dataset", "C:/git/cbir-project/1.input_data/original_dataset")

# #重新命名
# from rename import rename_all
# #rename_all(download_images_path)

# #影像處理 調整對比度 亮度
# modify_all(download_images_path)

# #複製到2
# shutil.copytree("C:/git/cbir-project/1.input_data/dataset", "C:/git/cbir-project/2.cbir/dataset")

print("-------------finish-------------")