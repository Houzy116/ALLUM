import warnings,datetime,time,rasterio,random,torch,math,gc,squarify,ternary,os, sys, osr,pymannkendall
warnings.filterwarnings("ignore")
from dateutil.relativedelta import relativedelta
from gdalconst import *
import geopandas as gpd
from glob import glob
from math import sin, asin, cos, acos, tan, radians, pi, degrees
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle,Rectangle
from mpl_chord_diagram import chord_diagram
import netCDF4 as nc
import numpy as np
from osgeo import gdal, gdalconst
import pandas as pd
from rasterio import fill
from scipy.stats import norm
from scipy import stats,signal
import seaborn as sns
from shapely import geometry
from sklearn.linear_model import LinearRegression,TheilSenRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


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
R = 6371007.181			
pixel_with=0.004491576420597608
root_path='/data/hk/albedo/'
type_code={'IGBP':{0:'Evergreen Needleleaf Forests',#Evergreen_Needleleaf_Forests
           1:'Permanent Snow',
           2:'Evergreen Broadleaf Forests',#Evergreen_Broadleaf_Forests
           3:'Deciduous Needleleaf Forests',#Deciduous_Needleleaf_Forests
           4:'Deciduous Broadleaf Forests',#Deciduous_Broadleaf_Forests
           5:'Mixed Forests',#Mixed_Forests
           6:'CLosed Shrublands',#CLosed_Shrublands
           7:'Open Shrublands',#Open_Shrublands
           8:'Woody Savannas',#Woody_Savannas
           9:'Savannas',
           10:'Grasslands',
           11:'Permanent Wetlands',
           12:'Croplands',
           13:'Urban',
           14:'Cropland/Natural Vegetation Mosaics',#Cropland/Natural_Vegetation_Mosaics
           15:'Barren',
           16:'Snow'
           },
            'LCCS1':{0:'Barren',#Evergreen_Needleleaf_Forests
           1:'Permanent Snow',
           2:'Evergreen Needleleaf Forests',
           3:'Evergreen Broadleaf Forests',#Evergreen_Broadleaf_Forests
           4:'Deciduous Needleleaf Forests',#Deciduous_Needleleaf_Forests
           5:'Deciduous Broadleaf Forests',#Deciduous_Broadleaf_Forests
           6:'Mixed Broadleaf/Needleleaf Forests',#Mixed_Forests
           7:'Mixed Broadleaf Evergreen/Deciduous Forests',
           8:'Open Forests',#CLosed_Shrublands
           9:'Sparse Forests',#Open_Shrublands
           10:'Dense Herbaceous',#Woody_Savannas
           11:'Sparse Herbaceous',
           12:'Dense Shrublands',
           13:'Shrubland/Grassland Mosaics',
           14:'Sparse Shrublands',
           15:'Snow'
           },
           'LCCS2':
            {0:'Barren',
            1:'Permanent Snow',
            2:'Urban',
            3:'Dense Forests',
            4:'Open Forests',
            5:'Forest/Cropland Mosaics',
            6:'Natural Herbaceous',
            7:'Natural Herbaceous/Cropland Mosaics',
            8:'Herbaceous Croplands',
            9:'Shrublands',
            10:'Snow' 
           },
            'LCCS3':
            {0:'Barren',
            1:'Permanent Snow',
            2:'Dense Forests',
            3:'Open Forests',
            4:'Woody Wetlands',
            5:'Grasslands',
            6:'Shrublands',
            7:'Herbaceous Wetlands',
            8:'Tundra',
            9:'Snow'
           }
}


def splice(files,output_file,pixel_size=0.0003,band_num=1,type='int32',masked=None):
    def get_extent(fn):
        ds = gdal.Open(fn)
        rows = ds.RasterYSize
        cols = ds.RasterXSize
        # 获取图像角点坐标
        transform = ds.GetGeoTransform()
        minX = transform[0]
        maxY = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]#是负值（important）
        maxX = minX + (cols * pixelWidth)
        minY = maxY + (rows * pixelHeight)
        return minX, maxY, maxX, minY
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
    cols = int((MaxX - MinX) / pixel_size)
    rows = int((MaxY - MinY) / pixel_size)
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
        data = band.ReadAsArray().astype(type)
        if masked is not None:
            data[data>=masked]=np.nan
        label[yOffset:yOffset+rows,xOffset:xOffset+cols]=data
        n+=1
    dsOut.GetRasterBand(1).WriteArray(label)
    dsOut.FlushCache()
    del dsOut
    
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
    new_img=driver.Create(output_path,width,height,1,1,['COMPRESS=LZW','BIGTIFF=YES'])
    transform_new=(-180.00441663186973, 0.004491576420597608, 0.0, 90.00220831593487, 0.0, -0.004491576420597608)
    new_img.SetGeoTransform(transform_new)
    new_img.SetProjection(gdal.Open(root_path+f'albedo/2001_1_albedo/2001_1_band1.tif').GetProjection())
    new_img.GetRasterBand(1).SetNoDataValue(255)
    new_img.GetRasterBand(1).WriteArray(data)
    new_img.FlushCache()
    print('complete：',output_path)
    del src_ds
    del new_img
    del data

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
        date = datetime.datetime.strptime(date, '%Y/%m/%d')
        Dn = int(date.strftime('%j'))
        time = datetime.datetime.strptime(time, '%H:%M')
        Hour = time.hour + time.minute/60.0
        gamma = 2*pi*(Dn - 1 + (Hour - 12)/24)/365
        eqtime = 229.18*(0.000075 + 0.001868*cos(gamma) - 0.032077*sin(gamma) - 0.014615*cos(2*gamma) - 0.040849*sin(2*gamma))
        f1 = 0.006918
        f2 = 0.399912*cos(gamma)
        f3 = 0.070257*sin(gamma)
        f4 = 0.006758*cos(gamma*2)
        f5 = 0.000907*sin(gamma*2)
        f6 = 0.002697*cos(gamma*3)
        f7 = 0.001480*sin(gamma*3)
        decl = f1 - f2 + f3 - f4 + f5 - f6 + f7
        time_offset = eqtime + 4*lon - 60*timezone
        tst = Hour*60 + time_offset
        ha = (tst/4 - 180)
        snoon = (720 - 4*lon - eqtime)/60 + timezone
        lat = radians(lat)
        ha = radians(ha) 
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

def t_test(a,b):
    if len(a.shape)==1:
        a=a.reshape((1,-1))
    if len(b.shape)==1:
        b=b.reshape((1,-1))
    def t_test_single(a_single,b_single):
        r=stats.ttest_ind(a_single, b_single)
        # return [r.__getattribute__("statistic"),r.__getattribute__("pvalue")]
        return [(r.__getattribute__("statistic")>0).astype(np.int8),(r.__getattribute__("pvalue")<0.05).astype(np.int8)]
    k=[t_test_single(a[i],b[i]) for i in range(a.shape[0])]
    k2=[k[i][0] for i in range(len(k))]
    k3=[k[i][1] for i in range(len(k))]
    return [k2,k3]

def trend_map(a):
    def LinearRegression_trend(kk):
        if np.isnan(kk).any():
            mask=0
            return [-1,0,mask]
        else:
            mask=1
            # p,slope=mk(kk)[1:3]
            result = mk(kk)
            return [result[-1],result[0],mask]
    k=[LinearRegression_trend(a[i]) for i in range(a.shape[0])]
    ps=[k[i][0] for i in range(len(k))]
    slopes=[k[i][1] for i in range(len(k))]
    masks=[k[i][2] for i in range(len(k))]
    return [ps,slopes,masks]

def mk(x):
    n = len(x)
    result = pymannkendall.original_test(x,alpha=0.05)
    model = TheilSenRegressor()
    model.fit(np.arange(1,n+1).reshape(-1,1),x)
    return model.coef_[0],model.intercept_,result.trend,result.p

def linear_outlier_detection(data, threshold=1.5):
    X = np.arange(len(data)).reshape(-1, 1)
    y = np.array(data).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    residuals = np.abs(y - model.predict(X))
    outliers = np.where(residuals > threshold * np.std(residuals))[0]
    return outliers