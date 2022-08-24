import cv2
import numpy as np
import os

def modify_image(image_path):
    if(image_path[-3:] == 'gif'):
        return 0
    img = cv2.imread(image_path)
    contrast = 35
    brightness = -5
    output = img * (contrast/127 + 1) - contrast + brightness # 轉換公式
    # 轉換公式參考 https://stackoverflow.com/questions/50474302/how-do-i-adjust-brightness-contrast-and-vibrance-with-opencv-python

    # 調整後的數值大多為浮點數，且可能會小於 0 或大於 255
    # 為了保持像素色彩區間為 0～255 的整數，所以再使用 np.clip() 和 np.uint8() 進行轉換
    output = np.clip(output, 0, 255)
    output = np.uint8(output)

    # 顯示照片
    # cv2.imshow('C:/git/cbir-project/1.input_data/original_dataset/-115,36/LC08_039035_20130531.png', img)    # 原始圖片
    # cv2.imshow('C:/git/cbir-project/1.input_data/original_dataset/LC08_039035_20130531.png', output) # 調整亮度對比的圖片
    # cv2.waitKey(0)                    # 按下任意鍵停止
    # cv2.destroyAllWindows()
    cv2.imwrite(image_path, output)
    print(image_path + "  ok")
    #modify_image(original_path)

def modify_images(path,filename):
    entries = os.listdir(path)
    for i in entries:
        modify_image(path + i)
        #print(path + i)

def modify_all(path):
    for filename in os.listdir(path):
        modify_images(path + filename +"/",filename)
        #print(filename)

# path = "C:/git/cbir-project/2.cbir/mAP_database/  4_-62,-17_1997.jpg"
# modify_image(path)
