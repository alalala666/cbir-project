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

def auto_get_images_1984to2012(lon,lat,path):

    
    #經緯度要到小數點後面一點 
    lat = lat
    lon = lon
  
    #設定範圍
    point = ee.Geometry.Point(lon,lat)

    #設定時間
    start_year = 1984
    end_year = 2012 #應該是2012 改1986測比較快****************************************************2012
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
    downloads_path = path + downloads_name

    cartoee.get_image_collection_gif(
        ee_ic=collection,
        out_dir=os.path.expanduser(downloads_path),
        out_gif="1984-2012_timelapse.gif",
        vis_params=vis_params,
        region=region,#選取範圍
        fps=3,
        #plot_title=" ",
        #date_format='YYYY-MM-dd',
        fig_size=(10, 8),
        dpi_plot=100,
        file_format="png",
        verbose=True,
        )

def auto_get_images_2012to2022(lon,lat,path):

    point = ee.Geometry.Point(lon,lat)
    #設定時間
    start_year = 2013
    end_year = 2022#*********************************************************************************2022
    years = ee.List.sequence(start_year, end_year)

    def get_best_image(year):
        start_date = ee.Date.fromYMD(year, 1, 1)
        end_date = ee.Date.fromYMD(year, 12, 31)
        image = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
            .filterBounds(point)#設定範圍
            .filter(ee.Filter.calendarRange(1, 12, 'month'))#指定月份 1-12月
            .filterDate(start_date, end_date)
            .sort('CLOUD_COVER').first()#去雲
            )
        return ee.Image(image)

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

    #設定路徑及資料夾名稱
    downloads_name = str(int(lon)) +","+ str(int(lat))
    downloads_path = path + downloads_name

    #下載圖片 要用再解註
    cartoee.get_image_collection_gif(
        ee_ic=collection,
        out_dir=os.path.expanduser(downloads_path),
        out_gif="2012-now_timelapse.gif",
        vis_params=vis_params,
        region=region,#選取範圍
        fps=3,
        #plot_title=" ",
        #date_format='YYYY-MM-dd',
        fig_size=(10, 8),
        dpi_plot=100,
        file_format="png",
        verbose=True,
        )

# path = "C:/git/cbir-project/2.cbir/dataset/-62,-17/"
# lon = -62.4953
# lat = -17.2853

# auto_get_images_1984to2012(lon, lat,path)
# auto_get_images_2012to2022(lon, lat, path)
print("finish")