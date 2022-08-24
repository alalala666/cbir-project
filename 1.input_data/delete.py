#coding=utf-8
#刪除空的資料夾
import os
def del_emp_dir(path):
  for (root,dirs,files) in os.walk(path):
    for item in dirs:
      dir = os.path.join(root,item)
      try:
        os.rmdir(dir)
        #print(dir)
      except Exception as e:
        print('Exception',e)

# dir = r'C:/git/cbir-project/2.cbir/dataset/'
# del_emp_dir(dir)