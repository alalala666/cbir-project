import os 
import csv

with open('C:\git\cbir-project/1.input_data\lonlat.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(['Longitude', 'latitude'])
    for i in range(-180,180):
        for j in range(-90,90):
             writer.writerow([i, j])

#coding=utf-8
import os
def del_emp_dir(path):
  for (root,dirs,files) in os.walk(path):
    for item in dirs:
      dir = os.path.join(root,item)
      try:
        os.rmdir(dir)
        print(dir)
      except Exception as e:
        print('Exception',e)

dir = r'C:/git/cbir-project/2.cbir/dataset/'
del_emp_dir(dir)
               