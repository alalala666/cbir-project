import os 
import geemap 
import ee 
import matplotlib.pyplot as plt 
from geemap import cartoee
import cartopy.crs as ccrs
from sqlalchemy import false, true
import pandas as pd
import csv
import shutil
geemap.ee_initialize()

#抓圖函數
def auto_get_image(lat,lon):
    #經緯度要到小數點後面一點 
    lat = lat
    lon = lon
  
    #設定範圍
    point = ee.Geometry.Point(lon,lat)

    #設定時間
    start_year = 1984
    end_year = 1986 #應該是2012 改1986測比較快
    years = ee.List.sequence(start_year, end_year)

    #選圖片 每月選一張做為代表
    def get_best_image(year):
        start_date = ee.Date.fromYMD(year, 1, 1)
        end_date = ee.Date.fromYMD(year, 12, 31)
        image = (
            ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA")
            .filterBounds(point)#設定範圍
            .filter(ee.Filter.calendarRange(1, 12, 'month'))#指定月份 1-12月
            .filterDate(start_date, end_date)
            .sort('CLOUD_COVER').first()#去雲
            )
        return ee.Image(image)

    #圖像可視化
    #https://developers.google.com/earth-engine/guides/image_visualization
    vis_params = {"bands": ['B3', 'B2', 'B1'],#rbg
                "min": 0.0,
                "max": 0.4,
                "gamma" : 1.2, #亮度             
                }

    collection = ee.ImageCollection(years.map(get_best_image))
    image = ee.Image(collection.first())

    #設定範圍
    w = 0.3
    h = 0.3
    region = [lon - w, lat - h, lon + w, lat + h]

    #下載圖片
    #設定路徑及資料夾名稱
    downloads_name = str(int(lon)) +","+ str(int(lat))
    downloads_path = "~/Downloads/dataset/" + downloads_name

    cartoee.get_image_collection_gif(
        ee_ic=collection,
        out_dir=os.path.expanduser(downloads_path),
        out_gif="1984-2012_timelapse.gif",
        vis_params=vis_params,
        region=region,#選取範圍
        fps=3,
        plot_title=" ",
        date_format='YYYY-MM-dd',
        fig_size=(10, 8),
        dpi_plot=100,
        file_format="png",
        verbose=True,
        )

#先把舊檔案刪除 不用就註掉
fileTest = "C:/Users/alalala/Downloads/dataset"
try:
    shutil.rmtree(fileTest)
except OSError as e:
    print(e)
else:
    print("File is deleted successfully")

#讀取經緯度的csv檔案
loaction_path = 'C:\git\中技社\input_data\lonlat.csv'
with open(loaction_path) as location:
    print("reading " + loaction_path + " : ")
    rows = csv.reader(location)
    #跳過第一列
    next(rows)
    for i in rows:
        print ("-----lon : " + i[0] + "-----lat : "+ i[1] +"-----")
        lon = float(i[0])
        lat = float(i[1])
        #開始抓圖
        auto_get_image(lon,lat)         

print("-------------finish-------------")