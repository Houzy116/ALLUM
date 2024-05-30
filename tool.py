import geopandas as gpd
import datetime
from glob import glob
import numpy as np
import os, sys, gdal
from gdalconst import *
import rasterio
import gc
from matplotlib import pyplot as plt
import math
from gdalconst import *
import shutil
import time
import osr
import warnings
from rasterio import fill
import netCDF4 as nc
warnings.filterwarnings("ignore")
from tqdm import tqdm
import torch
from dateutil.relativedelta import relativedelta
import pandas as pd
import requests
import json
# from generate_sankey import *
from matplotlib import cm
import random
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from math import sin, asin, cos, acos, tan, radians, pi, degrees
from osgeo import gdal, gdalconst
names=['Evergreen Needleleaf Forests',
 'Evergreen Broadleaf Forests',
 'Deciduous Needleleaf Forests',
 'Deciduous Broadleaf Forests',
 'Mixed Forests',
 'Closed Shrublands',
 'Open Shrublands',
 'Woody Savannas',
 'Savannas',
 'Grasslands',
 'Permanent Wetlands',
 'Croplands',
 'Urban',
 'Cropland/Natural Vegetation Mosaics',
 'Barren',
 'Water Bodies',
 'Snow']
global_trf=(-180.00441663186973, 0.004491576420597608, 0.0, 90.00220831593487, 0.0, -0.004491576420597608)
pi = 3.1415926;
R = 6371007.181			
pixel_with=0.004491576420597608
root_path='/data/hk/albedo/'
type_code={0:'EN_Forests              ',#Evergreen_Needleleaf_Forests
           1:'EB_Forests              ',#Evergreen_Broadleaf_Forests
           2:'DN_Forests              ',#Deciduous_Needleleaf_Forests
           3:'DB_Forests              ',#Deciduous_Broadleaf_Forests
           4:'M_Forests               ',#Mixed_Forests
           5:'C_Shrublands            ',#CLosed_Shrublands
           6:'O_Shrublands            ',#Open_Shrublands
           7:'W_Savannas              ',#Woody_Savannas
           8:'Savannas                ',
           9:'Grasslands              ',
           10:'Permanent_Wetlands      ',
           11:'Croplands               ',
           12:'Urban                   ',
           13:'Cropland Natural_Mosaics',#Cropland/Natural_Vegetation_Mosaics
           14:'Permanent_Snow          ',
           15:'Barren                  ',
           16:'Water_Bodies            ',
           17:'Snow                    '
           }
def get_extent(fn):
    ds = gdal.Open(fn)
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    # 获取图像角点坐标
    transform = ds.GetGeoTransform()
#     print(transform)
#     print(ds.GetProjection())
#     print(transform)
    minX = transform[0]
    maxY = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]#是负值（important）
    maxX = minX + (cols * pixelWidth)
    minY = maxY + (rows * pixelHeight)
    return minX, maxY, maxX, minY

def splice(files,output_file,pixel_size=0.0003,band_num=1,type='int32',masked=None):
    # print(files[0])
    MinX, MaxY, MaxX, MinY = get_extent(files[0])

    for fn in files[1:]:
        minX, maxY, maxX, minY = get_extent(fn)
        MinX = min(MinX, minX)
        MaxY = max(MaxY, maxY)
        MaxX = max(MaxX, maxX)
        MinY = min(MinY, minY)


    ds = gdal.Open(files[0])
    transform = ds.GetGeoTransform()
    pixelWidth = transform[1]
    pixelHeight = transform[5]
    #获取输出图像的行与列
    cols = int((MaxX - MinX) / pixel_size)
    rows = int((MaxY - MinY) / pixel_size)
    # print(cols)
    NP2GDAL_CONVERSION = {
      "uint8": 1,
      "int8": 1,
      "uint16": 2,
      "int16": 3,
      "uint32": 4,
      "int32": 5,
      "float32": 6,
      "float64": 7,
      "complex64": 10,
      "complex128": 11,
    }
    driver = gdal.GetDriverByName('GTiff')
    dsOut = driver.Create(output_file,
                            cols,rows,1,NP2GDAL_CONVERSION[type],
                            ['COMPRESS=LZW','BIGTIFF=YES'])
    bandOut = dsOut.GetRasterBand(1)

    # 设置输出图像的几何信息和投影信息
    geotransform = [MinX, pixel_size, 0, MaxY, 0, pixel_size*(-1)]
    dsOut.SetGeoTransform(geotransform)
    dsOut.SetProjection(ds.GetProjection())
    label=np.zeros([rows,cols],dtype=type)
    n=0
    for fn in tqdm(files, desc=f'Band {band_num} '):
        ds = gdal.Open(fn)
        transform = ds.GetGeoTransform()
        pixelWidth = transform[1]
        pixelHeight = transform[5]

        rows = ds.RasterYSize
        cols = ds.RasterXSize
        transform = ds.GetGeoTransform()
        minX = transform[0]
        maxY = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]#是负值（important）

        maxX = minX + (cols * pixelWidth)
        minY = maxY + (rows * pixelHeight)

        xOffset = round((minX - MinX) / pixel_size)
        yOffset = round((maxY - MaxY) / (pixel_size*(-1)))
        cols=int(cols*pixelWidth/pixel_size)
        rows=int(rows*pixelHeight/(pixel_size*(-1)))
        band = ds.GetRasterBand(band_num)
        # data = band.ReadAsArray(buf_xsize=cols,buf_ysize = rows)
        data = band.ReadAsArray().astype(type)
        if masked is not None:
            data[data>=masked]=np.nan
        # print(data.max())
        label_part=label[yOffset:yOffset+rows,xOffset:xOffset+cols]
        # label_part2=label_part+data
        # label_part2[(label_part>0)&(data>0)]=np.maximum(data[(label_part>0)&(data>0)],label_part[(label_part>0)&(data2>0)])
        # label[yOffset:yOffset+rows,xOffset:xOffset+cols]=np.maximum(data,label_part)
        label[yOffset:yOffset+rows,xOffset:xOffset+cols]=data
        n+=1
        # print('\n拼接图像'+str(n))
    # print(label.max())
    dsOut.GetRasterBand(1).WriteArray(label)
    dsOut.FlushCache()
    del dsOut
def get_img(y,m,bandname,coord=None,path=None):
    # time1=time.time()
    bandmap={'BSA_vis_snow':1,
         'WSA_vis_snow':2,
         'BSA_nir_snow':3,
         'WSA_nir_snow':4,
         'BSA_shortwave_snow':5,
         'WSA_shortwave_snow':6,
         'BSA_vis_snowfree':7,
         'WSA_vis_snowfree':8,
         'BSA_nir_snowfree':9,
         'WSA_nir_snowfree':10,
         'BSA_shortwave_snowfree':11,
         'WSA_shortwave_snowfree':12,
         'count':13,
         'snow_count':14
         }
    global_trf=(-180.00441663186973, 0.004491576420597608, 0.0, 90.00220831593487, 0.0, -0.004491576420597608)
    if path is None:
        if bandname=='landcover':
            path=root_path+f"landcover/{y}_landcover/{y}_landcover.tif"
        elif bandname=='snow_fre':
            c=str(y)+str(m).zfill(2)
            path=root_path+f'snow/snow_monthly/{c}.tif'
        else:
            path=root_path+f'albedo/{y}_{m}_albedo/{y}_{m}_band{bandmap[bandname]}.tif'
    # time2=time.time()
    ds=gdal.Open(path)
    # if ds.GetGeoTransform()!=global_trf:
    #     raise('TRF ERROR')
    # time3=time.time()
    if coord==None:
        return ds.ReadAsArray()
    else:
        x=round((coord[0]-global_trf[0])/global_trf[1])
        x_s=round((coord[0]+1-global_trf[0])/global_trf[1])-x
        y=round((coord[1]-global_trf[3])/global_trf[5])
        y_s=round((coord[1]-1-global_trf[3])/global_trf[5])-y
        img=ds.ReadAsArray(xoff=x,xsize=x_s,yoff=y,ysize=y_s)
        # time4=time.time()
        # print(time2-time1)
        # print(time3-time2)
        # print(time4-time3)
        return img
    
def get_pixelarea(coord,x_s,y_s):
    pi = 3.1415926;
    R = 6371007.181			
    pixel_with=0.004491576420597608
    k=[]
    for i in range(y_s):
        pixel_area = (pi/180.0)*R*R*abs(math.sin((coord[1]-pixel_with*i)/180.0*pi) - math.sin((coord[1]-pixel_with*(i+1))/180.0*pi)) * pixel_with
        k.append(pixel_area)
    area=(np.array([[1]*x_s])*(np.array([k]).T))/1000000
    return area

def tif_save(img,save_name,trf,p='4326',novalue=None,valuetype=6):
    driver=gdal.GetDriverByName('GTiff')
    new_img=driver.Create(save_name,img.shape[1],img.shape[0],1,valuetype,['COMPRESS=LZW','BIGTIFF=YES'])
    new_img.SetGeoTransform(trf)
    
    if p=='4326':
        sref = osr.SpatialReference()
        sref.ImportFromEPSG(4326)
        proj = sref.ExportToWkt()
    else:
        proj=p
    new_img.SetProjection(proj)
    if novalue is not None:
        new_img.GetRasterBand(1).SetNoDataValue(novalue)
    new_img.GetRasterBand(1).WriteArray(img)
    new_img.FlushCache()
    del new_img

def resample(img,output_path):
    driver = gdal.GetDriverByName('MEM')
    src_ds = driver.Create('', img.shape[1],img.shape[0], 1, 5)
    src_ds.GetRasterBand(1).WriteArray(img)
    src_ds.GetRasterBand(1).SetNoDataValue(255)
    src_ds.FlushCache()
    width=80152
    height=40076
    data = src_ds.GetRasterBand(1).ReadAsArray(buf_xsize=width,buf_ysize=height,resample_alg = gdalconst.GRIORA_Bilinear)
    driver=gdal.GetDriverByName('GTiff')
    new_img=driver.Create(output_path,
                                   width,height,1,1,['COMPRESS=LZW','BIGTIFF=YES'])
    transform_new=(-180.00441663186973, 0.004491576420597608, 0.0, 90.00220831593487, 0.0, -0.004491576420597608)
    new_img.SetGeoTransform(transform_new)
    new_img.SetProjection(gdal.Open(root_path+f'albedo/2001_1_albedo/2001_1_band1.tif').GetProjection())
    new_img.GetRasterBand(1).SetNoDataValue(255)
    new_img.GetRasterBand(1).WriteArray(data)
    new_img.FlushCache()
    print('重采样完成：',output_path)
    del src_ds
    del new_img
    del data

def TS_IDM_4d(v,interp_indice,max_distance,distence_w_dim):
        value_bands=v[:,zero(interp_indice[0]-int(max_distance/distence_w_dim[0])):interp_indice[0]+int(max_distance/distence_w_dim[0])+1,
                zero(interp_indice[1]-int(max_distance/distence_w_dim[1])):interp_indice[1]+int(max_distance/distence_w_dim[1])+1,
                zero(interp_indice[2]-int(max_distance/distence_w_dim[2])):interp_indice[2]+int(max_distance/distence_w_dim[2])+1,
                zero(interp_indice[3]-int(max_distance/distence_w_dim[3])):interp_indice[3]+int(max_distance/distence_w_dim[3])+1]
        value=value_bands[0]
        non_nan_indices = np.where(~np.isnan(value))
        non_nan_points = np.array(list(zip(non_nan_indices[0], non_nan_indices[1], non_nan_indices[2],non_nan_indices[3])))
         
        if len(non_nan_points)==0:
                interpolated_values=[np.nan]
                dws=np.nan
        else:
                k=np.array([zero(-interp_indice[0]+int(max_distance/distence_w_dim[0]))-int(max_distance/distence_w_dim[0]),
                        zero(-interp_indice[1]+int(max_distance/distence_w_dim[1]))-int(max_distance/distence_w_dim[1]),
                        zero(-interp_indice[2]+int(max_distance/distence_w_dim[2]))-int(max_distance/distence_w_dim[2]),
                        zero(-interp_indice[3]+int(max_distance/distence_w_dim[3]))-int(max_distance/distence_w_dim[3])])
                # print(k)
                # print(non_nan_points)
                non_nan_points+=k
                distance_s = np.array([np.linalg.norm(np.array(non_nan_points[i])*np.array(distence_w_dim)) for i in range(len(non_nan_points))])
                if len(distance_s)<10:
                        interpolated_values=[np.nan]
                        dws=np.nan
                else:
                        
                        kth = np.partition(distance_s, 9)[9]
                        distance1=distance_s.copy()
                        distance1=distance1[distance1<=kth]
                        weights = 1.0 / distance1**2
                        dws=np.sum(weights)
                        weights /= dws
                        interpolated_values=[]
                        for i in range(value_bands.shape[0]):
                                distance=distance_s.copy()
                                value=value_bands[i]
                                non_nan_values = value[non_nan_indices]
                                non_nan_values=non_nan_values[distance<=kth]
                                interpolated_value = np.sum(weights * non_nan_values)
                                interpolated_values.append(interpolated_value)
                        
        return interpolated_values,dws
def search_nonan(v,interp_indice,max_distance,distence_w_dim):
    search_n=1
    k,dws=TS_IDM_4d(v,interp_indice,max_distance=max_distance*search_n,distence_w_dim=distence_w_dim)
    while np.isnan(k[0]):
        search_n*=2
        k,dws=TS_IDM_4d(v,interp_indice,max_distance=max_distance*search_n,distence_w_dim=distence_w_dim)
    return k,dws
def zero(k):
    if k<0:
        return 0
    else:
        return k


def fill_albedo(month,type):
    max_distance=3
    distence_w_dim=[10,2,1,1]
    bands=['albedo_BSA_vis', 
        'albedo_WSA_vis', 
        'albedo_BSA_nir', 
        'albedo_WSA_nir', 
        'albedo_BSA_shortwave', 
        'albedo_WSA_shortwave']
    # band='albedo_BSA_vis'

    # m_index=[(month-3)%12+1,(month-2)%12+1,(month-1)%12+1,(month)%12+1,(month+1)%12+1]
    m_index=[(month-2)%12+1,(month-1)%12+1,(month)%12+1]
    offset=[1 if m_index[j]-m_index[0]>=0 else -1 for j in range(len(m_index))]
    if np.array(offset).sum()==len(offset):
        offset=[0]*len(m_index)
    # print(month,m_index,offset)

    vs=[]
    ms=[]
    land_proportions=[]
    for i in range(len(offset)):    
        NC=nc.Dataset(root_path+f'information_val/month_{m_index[i]}.nc','r')
        for band in bands:
            vs.append(NC.variables[f'{band}-{type}'][:])
        if type!='ocean':
            ms.append(NC.variables[f'area-{type}'][:])
        # qa=NC.variables['qa'][:]
        area_grid=NC.variables['area-grid'][:]
        area_land=NC.variables['area-land'][:]
        land_proportions.append(area_land/area_grid)
        NC.close()
    offset_bands=[]
    for i in offset:
        offset_bands+=[i]*len(bands)
    if offset[0]!=0:
        nan_array=np.zeros((1,180,360))*np.nan
        vs=[np.concatenate((nan_array,vs[i]),axis=0) if offset_bands[i]==1 else np.concatenate((vs[i],nan_array),axis=0) for i in range(len(offset_bands))]
        land_proportions=[np.concatenate((nan_array,land_proportions[i]),axis=0) if offset[i]==1 else np.concatenate((land_proportions[i],nan_array),axis=0) for i in range(len(offset))]
        if type!='ocean':
            ms=[np.concatenate((nan_array,ms[i]),axis=0) if offset[i]==1 else np.concatenate((ms[i],nan_array),axis=0) for i in range(len(offset))]
        else:
            ms=land_proportions  
    else:
        if type=='ocean':
            ms=land_proportions  
    M,H,W=vs[0].shape
    v=np.stack(vs).reshape((len(offset),len(bands),M,H,W)).transpose((1,0,2,3,4))
    # vv=[v[:,i] for i in range(len(bands))]
    m=np.stack(ms)
    # plt.imshow(v[0,1,1])
    # plt.show()
    # land_proportion=np.stack(land_proportions)
    # print(month,m_index,v.shape)
    all_indices = np.indices(v[0].shape).reshape(4, -1).T
    if type!='ocean':
        interp_indices = all_indices[((np.isnan(v[0])) & (m>0)).reshape(-1)]
    else:
        interp_indices = all_indices[((np.isnan(v[0])) & (m<1) & (m>0)).reshape(-1)]
    nonan_indices=all_indices[(~np.isnan(v[0])).reshape(-1)]
    # print(len(nonan_indices))
    interp_indices=[j for j in interp_indices if j[0]==int(len(offset)/2)]   
    # interp_values=[]
    # search_n=1
    print(month,type,len(interp_indices),round(len(interp_indices)/(len(nonan_indices)+0.001)*100,2),'%')
    # for interp_indice in tqdm(interp_indices):
    #     k=TS_IDM_4d(interp_indice,max_distance=max_distance*search_n,distence_w_dim=distence_w_dim)
    #     while np.isnan(k):
    #         search_n*=2
    #         k=TS_IDM_4d(interp_indice,max_distance=max_distance*search_n,distence_w_dim=distence_w_dim)

    #     interp_values.append(k)
    kk=[search_nonan(v,interp_indice,max_distance=max_distance,distence_w_dim=distence_w_dim) for interp_indice in interp_indices]
    interp_values=np.array([a[0] for a in kk])
    dws=np.array([a[1] for a in kk])
    print('fill value '+str((1-np.isnan(interp_values).sum()/interp_values.shape[0])*100)+'%')
    dws_v=np.zeros(v.shape[1:])
    for z in range(len(interp_indices)):
        v[:,interp_indices[z][0],interp_indices[z][1],interp_indices[z][2],interp_indices[z][3]]=np.array(interp_values[z])
        dws_v[interp_indices[z][0],interp_indices[z][1],interp_indices[z][2],interp_indices[z][3]]=dws[z]
    # print(offset)
    # return v[:,int(len(offset)/2)],dws_v[int(len(offset)/2)]
    if offset[int(len(offset)/2)]>=0:
        return v[:,int(len(offset)/2),offset[int(len(offset)/2)]:],dws_v[int(len(offset)/2),offset[int(len(offset)/2)]:]
    else:
        return v[:,int(len(offset)/2),:-1],dws_v[int(len(offset)/2),:-1]    






 
 
def mk(x, alpha=0.1):
    n = len(x)

    # 计算趋势slope
    model = LinearRegression()
    model.fit(np.arange(1,n+1).reshape(-1,1),x)
    slope = model.coef_[0]
 
    # 计算S的值
    s = 0
    for j in range(n - 1):
        for i in range(j + 1, n):
            s += np.sign(x[i] - x[j])
 
    # 判断x里面是否存在重复的数，输出唯一数队列unique_x,重复数数量队列tp
    unique_x, tp = np.unique(x, return_counts=True)
    g = len(unique_x)
 
    # 计算方差VAR(S)
    if n == g:  # 如果不存在重复点
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
    else:
        var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18
 
    # 计算z_value
    if n <= 10:  # n<=10属于特例
        z = s / (n * (n - 1) / 2)
    else:
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
 
    # 计算p_value，可以选择性先对p_value进行验证
    p = 2 * (1 - norm.cdf(abs(z)))
 
    # 计算Z(1-alpha/2)
    h = abs(z) > norm.ppf(1 - alpha / 2)
 
    # 趋势判断
    if (z < 0) and h:
        trend = -1#'decreasing'
    elif (z > 0) and h:
        trend = 1#'increasing'
    else:
        trend = 0#'no trend'
 
    return trend,p,slope,z,[model.coef_,model.intercept_]


    
    
def get_sza(y=2001):
    def judge_a(Hour, snoon, a):
        
        if(a >= 0 and a <= pi and Hour < snoon):
            a = pi - a
        elif(a >= 0 and a <= pi and Hour == snoon):
            pass
        elif(a >= 0 and a <= pi and Hour > snoon):
            a = pi + a

        return a

    def get_h0_a(lon, lat, date, time, timezone):
       
        # 第零步：将日期转化为日期序数，1,2,3......，365；将时间转化为24小时制浮点数，如18:30转为18.5
        date = datetime.datetime.strptime(date, '%Y/%m/%d')
        Dn = int(date.strftime('%j'))
        time = datetime.datetime.strptime(time, '%H:%M')
        Hour = time.hour + time.minute/60.0

        # 第一步：计算太阳倾角(太阳直射点纬度)decl和equation of time
        gamma = 2*pi*(Dn - 1 + (Hour - 12)/24)/365

        # 计算eqtime
        eqtime = 229.18*(0.000075 + 0.001868*cos(gamma) - 0.032077*sin(gamma) - 0.014615*cos(2*gamma) - 0.040849*sin(2*gamma))

        # 计算decl
        f1 = 0.006918
        f2 = 0.399912*cos(gamma)
        f3 = 0.070257*sin(gamma)
        f4 = 0.006758*cos(gamma*2)
        f5 = 0.000907*sin(gamma*2)
        f6 = 0.002697*cos(gamma*3)
        f7 = 0.001480*sin(gamma*3)
        decl = f1 - f2 + f3 - f4 + f5 - f6 + f7

        # 第二步：计算太阳时角和方位角180度时的时间
        time_offset = eqtime + 4*lon - 60*timezone
        tst = Hour*60 + time_offset
        ha = (tst/4 - 180)
        snoon = (720 - 4*lon - eqtime)/60 + timezone

        # 第三步：计算太阳高度角
        lat = radians(lat)
        ha = radians(ha)  # 注意转为弧度
        h0 = asin(sin(lat)*sin(decl) + cos(lat)*cos(decl)*cos(ha))
        sza=90-degrees(h0)
        
        if sza>85:
            sza_L=0
        elif sza>70:
            sza_L=1
        else:
            sza_L=2
        return sza,sza_L
    def get_month_daynum(y=2001):
        md_num=[]
        n=0
        m=1
        for d in range(365):
            now=datetime.datetime.strptime(f'{y}-1-1', '%Y-%m-%d')+relativedelta(days=d)

            if m==now.month:
                n+=1
            else:
                md_num.append(n)
                n=1
                m=now.month
        md_num.append(n)
        return md_num
    def get_h0_month(lon, lat, y, m, time, timezone):
        md_num=get_month_daynum(y=y)
        m_szas=[]
        for md in [i+1 for i in range(md_num[m-1])]:
            m_szas.append(get_h0_a(lon, lat, f'{y}/{m}/{md}', time, timezone)[0])
        m_sza=np.array(m_szas).mean()
        if m_sza>85:
                sza_L=0
        elif m_sza>70:
            sza_L=1
        else:
            sza_L=2
        return m_sza,sza_L
    
    
    sza_ar=np.zeros((20,12,180,360))
    for l in [i-0.5 for i in range(90,-90,-1)]:
        st=str(abs(int(l+0.5))).zfill(2)+'     '
        for m in range(1,13):
            sza_info=get_h0_month(0, l, y, m, '12:00', 0.0)
            # sza_info=get_h0_a(0, l, f'{y}/{m}/15', '12:00', 0.0)
            sza_ar[:,m-1,int(-l-0.5+90),:]=sza_info[0]

    sza_L=np.zeros((20,12,180,360))
    sza_L[sza_ar>70]=1
    sza_L[sza_ar>85]=2 
    return sza_ar,sza_L


def to_str(a,suffix=None,k=2,l=8):
    
    if suffix is None:
        a=str(round(a,k))
    else:
        a=str(round(a,k))+suffix
    a+=' '*(l-len(a))
    return a
def month_fill(data,mask):
    data_mean=np.nanmean(data,axis=0)
    for i in range(12):
        data[i][mask[i]]=data_mean[mask[i]]
    return data

def mann_kendall_test(x):
    """
    Mann-Kendall trend test for a given data sequence x.
    Args:
        x: A list or numpy array of data sequence.
    Returns:
        trend: The calculated trend (positive, negative or no trend).
        p_value: The p-value of the test.
    """
    alpha=0.05
    n = len(x)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(x[j] - x[i])
    
    # Calculate the variance of the test statistic.
    var_s = (n * (n - 1) * (2 * n + 5)) / 18
    
    # Calculate the standardized test statistic.
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    # Calculate the p-value of the test.
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    # Determine the trend based on the sign of the test statistic.
    if p_value < alpha:
        if z > 0:
            trend = 'increasing'
        elif z < 0:
            trend = 'decreasing'
        else:
            trend = 'no trend'
    else:
        trend = 'no trend'
    return trend, p_value