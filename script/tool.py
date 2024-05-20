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
from generate_sankey import *
from matplotlib import cm
import random
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from math import sin, asin, cos, acos, tan, radians, pi, degrees
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

    

def tif_save_snowfre(img,save_name,trf,p='4326',novalue=None,valuetype=6):
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
from osgeo import gdal, gdalconst
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
def resample_wb_fraction(img,width=360,height=180,resample_alg = gdalconst.GRIORA_Bilinear):
    driver = gdal.GetDriverByName('MEM')
    src_ds = driver.Create('', img.shape[1],img.shape[0], 1, 6)
    src_ds.GetRasterBand(1).WriteArray(img)
    src_ds.GetRasterBand(1).SetNoDataValue(2)
    src_ds.FlushCache()

    data = src_ds.GetRasterBand(1).ReadAsArray(buf_xsize=width,buf_ysize=height,resample_alg = resample_alg)

    return data
def tif_save(img,save_name,trf,p='4326'):
    driver=gdal.GetDriverByName('GTiff')
    new_img=driver.Create(save_name,img.shape[1],img.shape[0],1,6,['COMPRESS=LZW','BIGTIFF=YES'])
    new_img.SetGeoTransform(trf)
    
    if p=='4326':
        sref = osr.SpatialReference()
        sref.ImportFromEPSG(4326)
        proj = sref.ExportToWkt()
    else:
        proj=p
    new_img.SetProjection(proj)
    new_img.GetRasterBand(1).SetNoDataValue(0)
    new_img.GetRasterBand(1).WriteArray(img)
    new_img.FlushCache()
    del new_img
def get_land_area(y,coord):
    landcover=get_img(y,0,'landcover',coord)
    area=((pi/180.0)*R*R*abs(math.sin(coord[1]/180.0*pi) - math.sin((coord[1]-pixel_with*landcover.shape[0])/180.0*pi)) * pixel_with*landcover.shape[1])/1000000
 
    #check grid availability
    condition=1#can be used gril
    
    if (landcover!=20).sum()==0:#all is ocean
        condition=2#too mach ocean
        areas=[area,0]
    else:
        pixel_area=get_pixelarea(coord,landcover.shape[1],landcover.shape[0])
        land_area=np.sum((landcover!=20)*pixel_area)#land area
        if (land_area/area)<0.05:#too mach ocean (>95%)
            condition=2
            areas=[area,land_area]
        else:
            condition=13
            areas=[area,land_area]
    return condition,areas

def get_snow_cover(f):
    nf=nc.Dataset(f)
    k=np.flip(nf.variables['scfv'][:][0],axis=0)
    k2=np.flip(nf.variables['scfv_unc'][:][0],axis=0)
    k[k2==210]=0#water
    k[k2==215]=100#Permanent_Snow_and_Ice
    return k

def monthly_snow_cover(y,m,save=False):
    fs=glob(root_path+f'snow/Snow/{str(y).zfill(4)+str(m).zfill(2)}*.nc')
    sum=np.zeros((18000, 36000)).astype(np.float32)
    count=np.zeros((18000, 36000)).astype(np.float32)
    for i in tqdm(range(len(fs)),desc=f' {y}-{m} '):
        k=get_snow_cover(fs[i])
        k2=k.copy()
        k2[k2>100]=0
        sum+=k2
        count+=(k<=100).astype(np.int8)
    sum=sum.astype(np.float32)
    sum[count==0]=np.nan
    kk=sum/count
    if save:
        np.save(root_path+f'snow/snow_monthly/{str(y).zfill(4)+str(m).zfill(2)}.npy',kk)
    return kk
def fill_snow_cover(y,m,save=False):
    adds=[]
    for i in range(1,21):
        add=[y-i,y+i]
        for j in range(2):
            if add[j]>2020 or add[j]<2001:
                add[j]=None
        if not np.array([add[0] is None,add[1] is None]).sum()==2:
            adds.append(add)
    img=np.load(root_path+f'snow/snow_monthly/{y}{str(m).zfill(2)}.npy')
    for add in tqdm(adds,desc=f' {y}-{m} '):
        add_pathes=[root_path+f'snow/snow_monthly/{i}{str(m).zfill(2)}.npy' for i in add if not i is None]
        add_imgs=[np.load(i) for i in add_pathes]
        if len(add_imgs)==1:
            add_img=add_imgs[0]
        else:
            add_img=np.nanmean(np.stack(add_imgs),axis=0)
        img[np.isnan(img)]=add_img[np.isnan(img)]
    if save:
        np.save(root_path+f'snow/snow_monthly/{y}{str(m).zfill(2)}_fill.npy',img)
    return img
def fill_snow_cover2(y,m):
    k=[[y,m-1],[y,m+1]]
    for j in range(2):
        if k[j][1]==0:
            k[j][1]=12
            k[j][0]=k[j][0]-1
        if k[j][1]==13:
            k[j][1]=1
            k[j][0]=k[j][0]+1
        if k[j][0] in [2000,2021]:
            k[j]=None
    code=[str(j[0])+str(j[1]).zfill(2) for j in k if j is not None ]
    k_pathes=[root_path+f'snow/snow_monthly/{c}_fill.npy' for c in code]
    print(k_pathes)
    z=str(y)+str(m).zfill(2)
    img=np.load(root_path+f'snow/snow_monthly/{z}_fill.npy')
    print((np.isnan(img))[24*100:-24*100].sum())
    img_interp=np.nanmean(np.stack([np.load(j) for j in k_pathes]),axis=0)
    
    img[np.isnan(img)]=img_interp[np.isnan(img)]
    print((np.isnan(img))[24*100:-24*100].sum())
    np.save(root_path+f'snow/snow_monthly/{z}_fill2.npy',img)
class get_10lat_information():
    '''
    return condition, areas and albedomean
    condition:
              1--grid have enough data to Statistics
              2--all ocean
              4--snowcover data have nan value
    if condition=2 return areas is [area of the grid, 0]
                   return albedomean is []
    elif condition=3 return areas is [area of 17-tpyes landcover and snow,area of the grid, land area, no nan land area]
                   return albedomean is []
    elif condition=4 return areas is [area of the grid, land area, no nan land area]
                    return albedomean is []
    else return a areas list with 21 length
    
    1-17 is 17 area of 17 land types
    18 is snowcover area
    19-21 are area of the grid, land area, no nan land area
    
    return a albedomean array for 20*6 size:
    20: 17 types land, snow, total and ocean
    6: 6 albedo bands
    
    there total is the albedo by average albedo of all nonan pixels in grid  (weighting snow and snowfree by snow frequence) directly, 
    instand of the albedo by weighting the area of different types of albedo
    '''
    def __init__(self,y,m,lat):
        self.y=y
        self.m=m
        self.lat=lat
        self.bandmap={'BSA_vis_snow':1,
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
        self.global_trf=(-180.00441663186973, 0.004491576420597608, 0.0, 90.00220831593487, 0.0, -0.004491576420597608)
        # print(f'Loading {self.y}/{self.m}/{self.lat} Image...')
        
        self.result={}
    def get_10lat_information(self):
        # print('Computing Information...')
        # time.sleep(1)
        self.get_all_band()
        for lat in range(self.lat,self.lat-10,-1):
            for lon in range(-180,180):
                self.result[str(lat)+','+str(lon)]=self.get_1grid_information([lon,lat])
        torch.save(self.result,root_path+f'information/result3_{self.y}_{self.m}_{self.lat}.pth')
    
    def get_all_band(self):
        self.landcover=self.get_img_10lat('landcover')
        self.count=self.get_img_10lat('count')
        self.snow_count=self.get_img_10lat('snow_count')
        self.snow_fre=self.get_img_10lat('snow_fre')
        bands_snow=['BSA_vis_snow', 
            'WSA_vis_snow', 
            'BSA_nir_snow', 
            'WSA_nir_snow', 
            'BSA_shortwave_snow', 
            'WSA_shortwave_snow']
        bands_snowfree=['BSA_vis_snowfree', 
            'WSA_vis_snowfree', 
            'BSA_nir_snowfree', 
            'WSA_nir_snowfree', 
            'BSA_shortwave_snowfree', 
            'WSA_shortwave_snowfree']
        albedo_snow=[]
        albedo_snowfree=[]
        for b in bands_snow:
            albedo_snow.append(self.get_img_10lat(b))
        for b in bands_snowfree:
            albedo_snowfree.append(self.get_img_10lat(b))
        self.albedo_snow=np.stack(albedo_snow)
        self.albedo_snowfree=np.stack(albedo_snowfree)
    def get_img_10lat(self,bandname):
        if bandname=='landcover':
            path=root_path+f"landcover/{self.y}_landcover/{self.y}_landcover.tif"
        elif bandname=='snow_fre':
            c=str(self.y)+str(self.m).zfill(2)
            path=root_path+f'snow/snow_monthly/{c}.tif'
        else:
            path=root_path+f'albedo/{self.y}_{self.m}_albedo/{self.y}_{self.m}_band{self.bandmap[bandname]}.tif'
        ds=gdal.Open(path)
        self.coord_y=round((self.lat-self.global_trf[3])/self.global_trf[5])
        coord_y_s=round((self.lat-10-self.global_trf[3])/self.global_trf[5])-self.coord_y
        img=ds.ReadAsArray(xoff=0,xsize=80152,yoff=self.coord_y,ysize=coord_y_s)
        return img

    def get_img_fromallband(self,coord,band):
        x=round((coord[0]-global_trf[0])/global_trf[1])
        x_s=round((coord[0]+1-global_trf[0])/global_trf[1])-x
        y=round((coord[1]-global_trf[3])/global_trf[5])
        y_s=round((coord[1]-1-global_trf[3])/global_trf[5])-y
        if band=='landcover':
            return self.landcover[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
        if band=='count':
            return self.count[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
        if band=='snow_count':
            return self.snow_count[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
        if band=='snow_fre':
            return self.snow_fre[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
        if band=='albedo_snow':
            return self.albedo_snow[:,y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
        if band=='albedo_snowfree':
            return self.albedo_snowfree[:,y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
    # def get_1grid_information(self,coord):
    #     condition=1
    #     snow_fre=self.get_img_fromallband(coord,'snow_fre').astype(np.float32)/100
    #     if (snow_fre==2.55).sum()>0:#snow fre have nan value
    #         condition=4
    #     return condition
    def get_1grid_information(self,coord):
        landcover=self.get_img_fromallband(coord,'landcover')
        area=((pi/180.0)*R*R*abs(math.sin(coord[1]/180.0*pi) - math.sin((coord[1]-pixel_with*landcover.shape[0])/180.0*pi)) * pixel_with*landcover.shape[1])/1000000
    
        #check grid availability
        condition=1#can be used gril
        
        if (landcover!=20).sum()==0:#all is ocean
            condition=2#too mach ocean
            areas=[area,0]
            albedomean=[]
        else:
            pixel_area=get_pixelarea(coord,landcover.shape[1],landcover.shape[0])
            land_area=np.sum((landcover!=20)*pixel_area)#land area
            # if (land_area/area)<0.05:#too mach ocean (>95%)
            #     condition=2
            #     areas=[area,land_area]
            #     albedomean=[] 
            # else:    
            # snowcount=get_img(y,m,'snow_count',coord)
            count=self.get_img_fromallband(coord,'count').astype(np.float32)
            count_snow=self.get_img_fromallband(coord,'snow_count').astype(np.float32)
            count_snowfree=count-count_snow 
            count_withocean=count.copy()      
            count[landcover==20]=0
            land_data_area=np.sum(((count>0) & (landcover!=20))*pixel_area)#have data land area
            

            snow_fre=self.get_img_fromallband(coord,'snow_fre').astype(np.float32)/100
            snow_fre[landcover==15]=1
            if (snow_fre==2.55).sum()>0:#snow fre have nan value
                condition=4
                areas=[area,land_area,land_data_area]
                albedomean=[]
            else:
                #ocean data mask ，1 if landcover == ocean and exist albedo data else nan
                ocean_data_mask=(landcover==20).astype(np.float32)
                ocean_data_mask[ocean_data_mask==0]=np.nan
                #no nan mask ，1 if have data else nan
                nonan_mask_snow=(count_snow>0).astype(np.float32)
                nonan_mask_snow[landcover==20]=0
                nonan_mask_snow[nonan_mask_snow==0]=np.nan
                
                nonan_mask_snowfree=(count_snowfree>0).astype(np.float32)
                nonan_mask_snowfree[landcover==20]=0
                nonan_mask_snowfree[nonan_mask_snowfree==0]=np.nan
                # plt.imshow(snow_fre)
                # plt.colorbar()
                # plt.show()  
                snow_fre[landcover==20]=np.nan#piexl percent for snow (have interploted)
                    
                #landcover area
                landcover_classes_areaperc=[]
                for i in range(1,18):
                    landcover_class_areaperc=(landcover==i)*(1-snow_fre)
                    landcover_classes_areaperc.append(landcover_class_areaperc)
                landcover_classes_areaperc=np.stack(landcover_classes_areaperc)#piexl percent for every land types
                landcover_classes_area=list(np.nansum((landcover_classes_areaperc*pixel_area).reshape((17,-1)),axis=1))# 17 land types area km2
                
                #snow area
                snow_dataarea=np.nansum(snow_fre*pixel_area)# snowcover area km2
                
                areas=landcover_classes_area+[snow_dataarea,area,land_area,land_data_area]
                    
                # if (land_data_area/land_area)<0.25:#too mach nan value (nan>75%)
                #     condition=3
                #     albedomean=[]
                # else:#available grid    
                albedo_snow=self.get_img_fromallband(coord,'albedo_snow')
                albedo_snowfree=self.get_img_fromallband(coord,'albedo_snowfree')
                #landcover albedo
                landcover_classes_mask=np.expand_dims(nonan_mask_snowfree,axis=0)*landcover_classes_areaperc
                landcover_classes_mask[landcover_classes_mask>0]=1
                landcover_classes_mask[landcover_classes_mask==0]=np.nan
                k_snowfree=(landcover_classes_mask.reshape((17,1,-1)))*(albedo_snowfree.reshape((1,6,-1)))#(17,1,n*n)*(1,6,n*n)
                landcover_classes_albedomean=np.nanmean(k_snowfree.reshape(17*6,-1),axis=1).reshape(17,6)#(17,6) average albedo for different bands and differen land types

                    
                #snow albedo
                snow_fre_mask=nonan_mask_snow*snow_fre
                snow_fre_mask[snow_fre_mask>0]=1
                snow_fre_mask[snow_fre_mask<=0]=np.nan
                
                k_snow=(snow_fre_mask.reshape((1,-1)))*(albedo_snow.reshape((6,-1)))#(1,n*n)*(6,n*n)
                snow_albedomean=np.nanmean(k_snow.reshape(6,-1),axis=1).reshape(1,6)#(1,6) average albedo for snow
                
                #total albedo
                count[count==0]=np.nan
                albedo=(albedo_snow*count_snow.reshape((1,landcover.shape[0],landcover.shape[1]))+albedo_snowfree*count_snowfree.reshape((1,landcover.shape[0],landcover.shape[1])))/(count.reshape(1,landcover.shape[0],landcover.shape[1]))
                albedo_totalmean=np.nanmean(albedo.reshape(6,-1),axis=1).reshape(1,6)
                
                #inshore ocean albedo
                count_withocean[count_withocean==0]=np.nan
                ocean_data_mask[np.isnan(count_withocean)]=0
                albedo_withocean=(albedo_snow*count_snow.reshape((1,landcover.shape[0],landcover.shape[1]))+albedo_snowfree*count_snowfree.reshape((1,landcover.shape[0],landcover.shape[1])))/(count_withocean.reshape(1,landcover.shape[0],landcover.shape[1]))
                k_ocean=(ocean_data_mask.reshape((1,-1)))*(albedo_withocean.reshape((6,-1)))#(1,n*n)*(6,n*n)
                ocean_albedomean=np.nanmean(k_ocean.reshape(6,-1),axis=1).reshape(1,6)#(1,6) average albedo for snow


                albedomean=np.concatenate([landcover_classes_albedomean,snow_albedomean,albedo_totalmean,ocean_albedomean],axis=0)


        return condition,areas,albedomean
def get_wb_fraction_data(type):
    NC=nc.Dataset(root_path+f'white_sky_fraction/{type}sf.sfc.mon.mean.nc')
    t_str = '1800-01-01 00:00:00'
    d = datetime.datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')
    t=[d+datetime.timedelta(hours=i) for i in NC.variables['time'][:]][636:876]
    lat=NC.variables['lat'][:]
    lon=NC.variables['lon'][:]
    data=NC.variables[f'{type}sf'][636:876]
    return [data,t,lat,lon]

def convert_to_nc(month):
    data,area,qa=get_month_raster_from_information(month)
    qa[:,139:151,310:]=2
    area[:18,:,139:151,310:]=0
    area[-1,:,139:151,310:]=np.nan
    area[-2,:,139:151,310:]=0
    data[:,:,139:151,310:]=np.nan
    new_NC = nc.Dataset(root_path+f'information_fill2/month_{month}.nc', 'w', format='NETCDF4')
    new_NC.createDimension('time', 20)
    new_NC.createDimension('latitude', 180)
    new_NC.createDimension('longitude', 360)

    new_NC.createVariable("time", 'i2', ("time"))
    new_NC.createVariable("latitude", 'f', ("latitude"))
    new_NC.createVariable("longitude", 'f', ("longitude"))

    new_NC.variables['time'][:] = np.array(range(2001,2021))
    new_NC.variables['latitude'][:] = np.array(range(90,-90,-1))-0.5
    new_NC.variables['longitude'][:] = np.array(range(-180,180))+0.5

    bands=['BSA_vis', 
        'WSA_vis', 
        'BSA_nir', 
        'WSA_nir', 
        'BSA_shortwave', 
        'WSA_shortwave']
    landtypes=['landtype'+str(i) for i in range(1,18)]+['snow','total','ocean']
    area_types=['landtype'+str(i) for i in range(1,18)]+['snow','grid','land','no_nan_of_land']

    for i in range(6):
        for j in range(20):
            var_name=f"albedo_{bands[i]}-{landtypes[j]}"
            new_NC.createVariable(var_name, 'f4', ("time", "latitude", "longitude"))
            new_NC.variables[var_name][:]=data[j,i]
            
    for j in range(21):
        var_name=f"area-{area_types[j]}"
        new_NC.createVariable(var_name, 'f4', ("time", "latitude", "longitude"))
        new_NC.variables[var_name][:]=area[j]
    new_NC.createVariable('qa','i2',("time", "latitude", "longitude"))
    new_NC.variables['qa'][:]=qa
    new_NC.close()
    
def get_month_raster_from_information(month):
    years=list(range(2001,2021))
    print(f'convert to cube for {month} 2001-2020')
    # print( 'data-----(landtype,band,year,lat,lon)')
    # print( 'area-----(landtype,year,lat,lon)')
    # print( 'qa-----(year,lat,lon)')
    data=np.zeros([20,6,20,180,360],dtype=np.float32)
    area=np.zeros([21,20,180,360],dtype=np.float32)
    qa=np.zeros([20,180,360],dtype=np.int8)
    time.sleep(1)
    for i in tqdm(range(20)):
        # print(f'convert to cube for {years[i]}')
        data_month_y=torch.load(root_path+f'information2/result3_{years[i]}_{month}_all.pth')
        for lat in range(90,-90,-1):
            for lon in range(-180,180):
                d=data_month_y[str(lat)+','+str(lon)]
                if d[0]==1:#perfact grid
                    qa[i,90-lat,lon+180]=1
                    area[:,i,90-lat,lon+180]=d[1]
                    data[:,:,i,90-lat,lon+180]=d[2]
                elif d[0]==2:#all ocean
                    qa[i,90-lat,lon+180]=2
                    area[:,i,90-lat,lon+180]=np.zeros((21))
                    area[-3,i,90-lat,lon+180]=d[1][0]
                    area[-2,i,90-lat,lon+180]=0
                    area[-1,i,90-lat,lon+180]=np.nan
                    data[:,:,i,90-lat,lon+180]=np.nan
                # elif d[0]==3:#too much nan value in land(>75%)
                #     qa[i,90-lat,lon+180]=3
                #     area[:,i,90-lat,lon+180]=d[1]
                #     data[:,:,i,90-lat,lon+180]=np.nan
                elif d[0]==4:#have snow nan
                    qa[i,90-lat,lon+180]=4
                    area[:,i,90-lat,lon+180]=np.nan
                    
                    area[-3:,i,90-lat,lon+180]=d[1]
                    data[:,:,i,90-lat,lon+180]=np.nan
    return data,area,qa

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



def simplify_landcover(coord,albedo_data,imgs):
    coord_str=str(coord[1])+','+str(coord[0])
    condition,area,_=albedo_data[coord_str]
    if condition!=4:#1 or 2
        landcover=imgs.get_img_fromallband(coord,'landcover').astype(np.int16)
        snow_fre=imgs.get_img_fromallband(coord,'snow_fre').astype(np.float32)/100
        snow_fre[landcover==20]=0
        snow_fre[landcover==15]=1
        if condition==1:
            area=area[:17]
            use_landtype=[i for i in range(1,18) if area[i-1]>0]
        else:
            use_landtype=[]
        return landcover,snow_fre,use_landtype,condition
    else:
        return [condition]
def get_landchange_matrix(y,m,coord,albedo_data1,albedo_data2,imgs1,imgs2):
    data1=list(simplify_landcover(coord,albedo_data1,imgs1))
    data2=list(simplify_landcover(coord,albedo_data2,imgs2))
    
    if (len(data1)+len(data2))<6:
        return [[data1[-1],data2[-1]],None,None]
    else:
        data1[2]=[0]+data1[2]
        data2[2]=[0]+data2[2]
        data1[2]+=[20]
        data2[2]+=[20]
        # if ((data1[0]==20) & (data2[0]!=20)).sum()>0:
        #     data1[2]+=[20]
        # if ((data1[0]!=20) & (data2[0]==20)).sum()>0:
        #     data2[2]+=[20]
        
        transfer_id=((np.array(data1[2]).reshape((1,-1))*100)+(np.array(data2[2]).reshape((-1,1)))).reshape(-1,1,1)

        # print(transfer_id)
        transfer_id=transfer_id.repeat(data1[1].shape[0],axis=1).repeat(data1[1].shape[1],axis=2)
        transfer=data1[0]*100+data2[0]
        transfer=np.expand_dims(transfer,0).repeat(transfer_id.shape[0],axis=0)
        # data1[1][transfer[0]==2020]=np.nan
        # data2[1][transfer[0]==2020]=np.nan
        max_snowfre=np.max(np.stack([data1[1],data2[1]]),axis=0)
        min_snowfre=np.min(np.stack([data1[1],data2[1]]),axis=0)
        dif_snowfre=data2[1]-data1[1]
        pixel_area=get_pixelarea(coord,data1[1].shape[1],data1[1].shape[0])
        transfer_masks=(transfer==transfer_id).astype(np.float32)
        transfer_masks[transfer_masks==0]=np.nan
        transfer_masks=transfer_masks*np.expand_dims(pixel_area,0)
        transfer_masks=transfer_masks*np.expand_dims((1-max_snowfre),0)
        transfer_area_matrix=np.nansum(transfer_masks.reshape((-1,pixel_area.shape[0]*pixel_area.shape[1])),axis=1).reshape(-1,(len(data1[2])))
        transfer_id_matrix=np.array(list(transfer_id[:,0,0])).reshape(-1,(len(data1[2])))

        snow_to_snow=np.nansum(min_snowfre*pixel_area)

        dif_snowfre_P=dif_snowfre.copy()
        dif_snowfre_N=dif_snowfre.copy()
        dif_snowfre_P[dif_snowfre_P<=0]=np.nan
        dif_snowfre_N[dif_snowfre_N>0]=np.nan

        #snowfre2>snowfre1
        transfer_id2=np.array(data2[2]).reshape(-1,1,1).repeat(data1[1].shape[0],axis=1).repeat(data1[1].shape[1],axis=2)
        transfer2=np.expand_dims(data2[0],axis=0).repeat(transfer_id2.shape[0],axis=0)
        transfer_masks2=(transfer2==transfer_id2).astype(np.float32)
        transfer_masks2[transfer_masks2==0]=np.nan
        transfer_masks2=transfer_masks2*np.expand_dims(pixel_area,0)
        transfer_masks2=transfer_masks2*np.expand_dims(-dif_snowfre_N,0)
        transfer_area_snowto=np.nansum(transfer_masks2.reshape((-1,pixel_area.shape[0]*pixel_area.shape[1])),axis=1).reshape((len(data2[2]),-1))

        #snowfre2<snowfre1
        transfer_id1=np.array(data1[2]).reshape(-1,1,1).repeat(data1[1].shape[0],axis=1).repeat(data1[1].shape[1],axis=2)
        transfer1=np.expand_dims(data1[0],axis=0).repeat(transfer_id1.shape[0],axis=0)
        transfer_masks1=(transfer1==transfer_id1).astype(np.float32)
        transfer_masks1[transfer_masks1==0]=np.nan
        transfer_masks1=transfer_masks1*np.expand_dims(pixel_area,0)
        transfer_masks1=transfer_masks1*np.expand_dims(dif_snowfre_P,0)
        transfer_area_tosnow=np.nansum(transfer_masks1.reshape((-1,pixel_area.shape[0]*pixel_area.shape[1])),axis=1).reshape((-1,len(data1[2])))

        transfer_area_matrix=np.concatenate([transfer_area_matrix,transfer_area_tosnow],axis=0)
        transfer_area_snowto=np.array(list(transfer_area_snowto.flatten())+[snow_to_snow]).reshape(-1,1)
        transfer_area_matrix=np.concatenate([transfer_area_matrix,transfer_area_snowto],axis=1)
        transfer_area_matrix=transfer_area_matrix[1:,1:]
    return [[data1[-1],data2[-1]],transfer_area_matrix,transfer_id_matrix]
def get_10lat_confusionmatrix(y,m,lat_10):
    imgs1=get_10lat_information(2001,m,lat_10)
    imgs2=get_10lat_information(y,m,lat_10)
    imgs1.landcover=imgs1.get_img_10lat('landcover')
    imgs1.snow_fre=imgs1.get_img_10lat('snow_fre')
    imgs2.landcover=imgs2.get_img_10lat('landcover')
    imgs2.snow_fre=imgs2.get_img_10lat('snow_fre')
    albedo_data1=torch.load(root_path+f'information/result3_2001_{m}_{lat_10}.pth')
    albedo_data2=torch.load(root_path+f'information/result3_{y}_{m}_{lat_10}.pth')
    result={}
    for lat in range(lat_10,lat_10-10,-1):
        for lon in range(-180,180):
            result[str(lat)+','+str(lon)]=get_landchange_matrix(y,m,[lon,lat],albedo_data1,albedo_data2,imgs1,imgs2)
    torch.save(result,root_path+f'confuse_matrix/result2_{y}_{m}_{lat_10}.pth')
def convert_to_matrix(year,month):
    f=torch.load(f'/data/hk/albedo/confuse_matrix/result2_{year}_{month}_all.pth')
    data=np.zeros((180,360,18,18)).astype(np.float32)
    for lat in tqdm(range(90,-90,-1)):
        for lon in range(-180,180):
            f_sub=f[str(lat)+','+str(lon)]
            if f_sub[0][0]!=1 or f_sub[0][1]!=1:
                output=np.full((18,18),np.nan)
            else:
                output=np.zeros((18,18)).astype(np.float32)
                from_index=list((f_sub[2][0]/100).astype(np.int16)-1)[1:-1]
                to_index=list(f_sub[2][:,0]-1)[1:-1]
                for y in range(len(from_index)):
                    for x in range(len(to_index)):
                        output[from_index[y],to_index[x]]=f_sub[1][x,y]
                for y in range(len(from_index)):
                    output[from_index[y],-1]=f_sub[1][-1,y]
                for x in range(len(to_index)):
                    output[-1,to_index[x]]=f_sub[1][x,-1]
                output[-1,-1]=f_sub[1][-1,-1]   
            data[90-lat,lon+180]=output
    return data

 
 
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
from tool import *
class get_10lat_information_LandS():
    '''
    return condition, areas and albedomean
    condition:
              1--grid have enough data to Statistics
              2--all ocean
              4--snowcover data have nan value
    if condition=2 return areas is [area of the grid, 0]
                   return albedomean is []
    elif condition=4 return areas is [area of the grid, land area]
                    return albedomean is []
                    
    else return a deep list
    
    [
    
    condition,
    
    [[area of the grid, land area],
    17 land types mean snowcover,
    17 land types mean snowcover in areas with LAI data,
    17 land types mean snowcover in areas with SM data]],
    
    [17 land types mean LAI,
    17 land types mean SM,]
    
    ]
    '''
    def __init__(self,y,m,lat):
        self.y=y
        self.m=m
        self.lat=lat
        self.global_trf=(-180.00441663186973, 0.004491576420597608, 0.0, 90.00220831593487, 0.0, -0.004491576420597608)
        # print(f'Loading {self.y}/{self.m}/{self.lat} Image...')
        
        self.result={}
    def get_10lat_information(self):
        # print('Computing Information...')
        # time.sleep(1)
        self.get_all_band()
        for lat in range(self.lat,self.lat-10,-1):
            for lon in range(-180,180):
                self.result[str(lat)+','+str(lon)]=self.get_1grid_information([lon,lat])
        torch.save(self.result,root_path+f'information_LandS/result-LAIandSM_{self.y}_{self.m}_{self.lat}.pth')
    
    def get_all_band(self):
        self.landcover=self.get_img_10lat('landcover')
        self.snow_fre=self.get_img_10lat('snow_fre')

        self.LAI=self.get_img_10lat('LAI')
        self.SM=self.get_img_10lat('SM')
    def get_img_10lat(self,bandname):
        if bandname=='landcover':
            path=root_path+f"landcover/{self.y}_landcover/{self.y}_landcover.tif"
        elif bandname=='snow_fre':
            c=str(self.y)+str(self.m).zfill(2)
            path=root_path+f'snow/snow_monthly/{c}.tif'
        elif bandname=='SM':
            path=root_path+f'SM/{self.y}_{self.m}_500m.tif'
        elif bandname=='LAI':
            path=root_path+f'LAI/{self.y}_{self.m}.tif'
        else:
            raise('ERROR bandname')
        ds=gdal.Open(path)
        self.coord_y=round((self.lat-self.global_trf[3])/self.global_trf[5])
        coord_y_s=round((self.lat-10-self.global_trf[3])/self.global_trf[5])-self.coord_y
        img=ds.ReadAsArray(xoff=0,xsize=80152,yoff=self.coord_y,ysize=coord_y_s)
        return img

    def get_img_fromallband(self,coord,band):
        x=round((coord[0]-global_trf[0])/global_trf[1])
        x_s=round((coord[0]+1-global_trf[0])/global_trf[1])-x
        y=round((coord[1]-global_trf[3])/global_trf[5])
        y_s=round((coord[1]-1-global_trf[3])/global_trf[5])-y
        if band=='landcover':
            return self.landcover[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
        if band=='snow_fre':
            return self.snow_fre[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
        if band=='LAI':
            return self.LAI[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
        if band=='SM':
            return self.SM[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
    # def get_1grid_information(self,coord):
    #     condition=1
    #     snow_fre=self.get_img_fromallband(coord,'snow_fre').astype(np.float32)/100
    #     if (snow_fre==2.55).sum()>0:#snow fre have nan value
    #         condition=4
    #     return condition
    def get_1grid_information(self,coord):
        landcover=self.get_img_fromallband(coord,'landcover')

        area=((pi/180.0)*R*R*abs(math.sin(coord[1]/180.0*pi) - math.sin((coord[1]-pixel_with*landcover.shape[0])/180.0*pi)) * pixel_with*landcover.shape[1])/1000000
    
        #check grid availability
        condition=1#can be used gril
        
        if (landcover!=20).sum()==0:#all is ocean
            condition=2#too mach ocean
            areas=[area,0]
            landcover_classes_LandS=[]
        else:
            pixel_area=get_pixelarea(coord,landcover.shape[1],landcover.shape[0])
            land_area=np.sum((landcover!=20)*pixel_area)#land area
            

            snow_fre=self.get_img_fromallband(coord,'snow_fre').astype(np.float32)/100
            snow_fre[landcover==15]=1
            if (snow_fre==2.55).sum()>0:#snow fre have nan value
                condition=4
                areas=[area,land_area]
                landcover_classes_LandS=[]
            else:
                LAI=self.get_img_fromallband(coord,'LAI')
                SM=self.get_img_fromallband(coord,'SM')
                LAI[landcover==20]=np.nan
                SM[landcover==20]=np.nan
                snow_fre[landcover==20]=np.nan#piexl percent for snow (have interploted)
             
                landcover_classes_areaperc=[]
                for i in range(1,18):
                    landcover_class_areaperc=(landcover==i)#
                    landcover_classes_areaperc.append(landcover_class_areaperc)
                landcover_classes_areaperc=np.stack(landcover_classes_areaperc)
                
                
                landcover_classes_sc=np.ones([17,snow_fre.shape[0],snow_fre.shape[1]])
                landcover_classes_sc[landcover_classes_areaperc==0]=np.nan
                landcover_classes_meansc=list(np.nanmean((landcover_classes_sc*snow_fre).reshape((17,-1)),axis=1))
                
                landcover_classes_sc_LAI=landcover_classes_sc.copy()
                LAI_17=np.expand_dims(LAI,axis=0).repeat(17,axis=0)
                landcover_classes_sc_LAI[np.isnan(LAI_17)]=np.nan
                landcover_classes_meansc_LAI=list(np.nanmean((landcover_classes_sc_LAI*snow_fre).reshape((17,-1)),axis=1))
                
                landcover_classes_sc_SM=landcover_classes_sc.copy()
                SM_17=np.expand_dims(SM,axis=0).repeat(17,axis=0)
                landcover_classes_sc_SM[np.isnan(SM_17)]=np.nan
                landcover_classes_meansc_SM=list(np.nanmean((landcover_classes_sc_SM*snow_fre).reshape((17,-1)),axis=1))

                
                areas=[[area,land_area],
                       landcover_classes_meansc,# 17 land types snow fre
                       landcover_classes_meansc_LAI,# 17 land types snow fre in areas with LAI data
                       landcover_classes_meansc_SM]# 17 land types snow fre in areas with SM data
                    
                #landcover albedo

                
                
                landcover_classes_LandS=[]

                for i in range(2):
                    if i==0:
                        var=LAI
                        landcover_classes_mask=landcover_classes_sc_LAI
                    else:
                        var=SM
                        landcover_classes_mask=landcover_classes_sc_SM

                    k_var=(landcover_classes_mask.reshape((17,1,-1)))*(var.reshape((1,1,-1)))#(17,1,n*n)*(1,6,n*n)
                    landcover_classes_var=np.nanmean(k_var.reshape(17*1,-1),axis=1).reshape(17,1)#(17,6) average albedo for different bands and differen land types
                    landcover_classes_LandS.append(list(landcover_classes_var.flatten()))
        return condition,areas,landcover_classes_LandS
    
    
    
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