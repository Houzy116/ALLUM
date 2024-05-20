# import geopandas as gpd
# from glob import glob
# import numpy as np
# import os
# import os, sys, gdal
# from gdalconst import *
# import numpy as np
# import torch
# import rasterio
# import gc
# from matplotlib import pyplot as plt
# import math
# from gdalconst import *
# import shutil
# from tqdm import tqdm

import time
from tool import *
import warnings
warnings.filterwarnings("ignore")
import torch

albedo_data1=torch.load(f'/mnt/nvme1n1/hk/albedo/result_2001_1_albedo.pth')
albedo_data2=torch.load(f'/mnt/nvme1n1/hk/albedo/result_2020_1_albedo.pth')
#提取网格信息

checkpoint_lat=list(range(80,-90,-10))+[-89]
result={}
for lat in range(90,-90,-1):
    n=str(abs(lat))+'   '
    print(n,end="",flush = True)
    time1=time.time()
    for lon in range(-180,180):
        try:
            result[str(lat)+','+str(lon)]=get_landchange_matrix([lon,lat],albedo_data1,albedo_data2)
        except:
            print([lon,lat])
        if lon%3==0:
            if np.max(np.array(result[str(lat)+','+str(lon)][0]))==1:
                print('*',end="",flush = True)
            else:
                print('-',end="",flush = True)
    time2=time.time()
    print('    ',end="",flush = True)
    print(int(time2-time1),end="",flush = True)
    print('\n',end="",flush = True)
    if lat in checkpoint_lat:
        torch.save(result,f'/mnt/nvme1n1/hk/albedo/landchange_matrix_200101-202001_{lat}.pth')
        result={}
