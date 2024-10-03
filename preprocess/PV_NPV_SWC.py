import sys
sys.path.append('..')
from tool import *
from preprocess.DDRF import resample_wb_fraction 

#Splice
def splice(files,output_file,pixel_size=0.0003,band_num=1,type='int32',masked=None):
    def get_extent(fn):
        ds = gdal.Open(fn)
        rows = ds.RasterYSize
        cols = ds.RasterXSize
        transform = ds.GetGeoTransform()
        minX = transform[0]
        maxY = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]
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
                            cols,rows,1,5,
                            ['COMPRESS=LZW','BIGTIFF=YES'])
    bandOut = dsOut.GetRasterBand(1)

    # 设置输出图像的几何信息和投影信息
    geotransform = [MinX, pixel_size, 0, MaxY, 0, pixel_size*(-1)]
    dsOut.SetGeoTransform(geotransform)
    dsOut.SetProjection(ds.GetProjection())
    # label=np.ones([rows,cols],dtype=type)*(-1001)
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
        data = band.ReadAsArray()
        if masked is not None:
            data[data>=masked]=np.nan
        # label_part=label[yOffset:yOffset+rows,xOffset:xOffset+cols]
        label[yOffset:yOffset+rows,xOffset:xOffset+cols]=data
        n+=1
    label[np.isnan(label)]=0
    # Limit outliers
    label[label>=1000]=1000
    label[label<=-1000]=-1000
    label*=100
    label=label.astype('int32')
    dsOut.GetRasterBand(1).WriteArray(label)
    dsOut.FlushCache()
    del dsOut

#Synthesize blue-sky
def bm(b,m):
    print(b,m)
    W=rasterio.open(f'/data2/hzy/ssd_hzy/G3/W_{b}{m}.tif').read(1)
    B=rasterio.open(f'/data2/hzy/ssd_hzy/G3/B_{b}{m}.tif').read(1)
    WB_sky_fraction=torch.load("/data/hk/albedo/white_sky_fraction/white_sky_fraction2.pth")
    wb=(WB_sky_fraction[(2020-2001)*12+m-1]+WB_sky_fraction[(2001-2001)*12+m-1])/2
    wb=(WB_sky_fraction[(2020-2001)*12+m-1]+WB_sky_fraction[(2001-2001)*12+m-1])/2
    wb_resample=resample_wb_fraction(wb,width=80152,height=40076,resample_alg = gdalconst.GRIORA_NearestNeighbour)
    wb_resample[wb_resample==2]=np.nan


    W[np.isnan(W)]=0
    B[np.isnan(B)]=0
    a=(W*wb_resample+B*(1-wb_resample))/100000

    tif_save(a,f'/data2/hzy/ssd_hzy/G3/{b}{m}_2.tif',global_trf,p='4326')
    del a,B,W,wb_resample
    gc.collect()
    img=rasterio.open(f'/data2/hzy/ssd_hzy/G3/{b}{m}_2.tif').read(1)
    plt.imshow(img,vmin=-0.05,vmax=0.05,cmap='seismic')
    plt.colorbar()
    plt.show()
        
if __name__=='__main__':
    
    
    #Splice
    for t in ['W','B']:
        for b in ['NDVI','SSI','NMDI']:
            for m in range(1,13):
                try:
                    print(t,b,m)
                    fs=glob(f'/ssd_hzy/G2/{t}_{b}{m}-*')

                    splice(fs,f'/ssd_hzy/G3/{t}_{b}{m}.tif',pixel_size=0.004491576420597608,band_num=1,type='float32')
                except:
                    print(t,b,m)
                    fs=glob(f'/ssd_hzy/G2/{t}_{b}{m}-*')

                    splice(fs,f'/ssd_hzy/G3/{t}_{b}{m}.tif',pixel_size=0.004491576420597608,band_num=1,type='float32')
    
    
    #Synthesize blue-sky
    for b in ['NDVI','SSI','NMDI']:
        for m in range(1,13):
            bm(b,m)


    #Calculate contribution of albedo (SZA<70)
    sza,sza_l=torch.load('/data/hk/albedo/SZA.pth')
    for b in ['NMDI']:
        count=np.zeros((40076, 80152))
        all=np.zeros((40076, 80152))
        for m in tqdm(range(1,13)):
            now=datetime.datetime.strptime(f'2001-{m}-1', '%Y-%m-%d')
            month_time=(now+relativedelta(months=1)-now).total_seconds()
            img=rasterio.open(f'/data2/hzy/ssd_hzy/G3/{b}{m}_2.tif').read(1)
            SZA_L=np.ones((40076, 80152))
            SZA_L=SZA_L*(sza_l[m-1].reshape(-1,1))
            img[SZA_L>0]=np.nan
            sc_01=rasterio.open(f"/data2/hzy/albedo2/snowcover/snowcover/2001_{m}_snowcover_v3_LCCS2.tif").read(1)
            sc_20=rasterio.open(f"/data2/hzy/albedo2/snowcover/snowcover/2020_{m}_snowcover_v3_LCCS2.tif").read(1)
            sc=(1000-np.maximum(sc_01,sc_20))/1000
            img*=sc
            img[img==0]=np.nan
            tif_save(img,f'/data2/hzy/ssd_hzy/G3/{b}{m}_albedo_final.tif',global_trf,p='4326')
            count[~np.isnan(img)]+=month_time
            img[np.isnan(img)]=0
            img=img*month_time
            all+=img
            del img 
            gc.collect()
        all/=count
        tif_save(all,f'/data2/hzy/ssd_hzy/G3/{b}_final.tif',global_trf,p='4326')


    #Calculate contribution of albedo (SZA<85)
    for b in ['NDVI','SSI','NMDI']:
        count=np.zeros((40076, 80152))
        all=np.zeros((40076, 80152))
        for m in tqdm(range(1,13)):
            img=rasterio.open(f'/data2/hzy/ssd_hzy/G3/{b}{m}_2.tif').read(1)
            SZA_L=np.ones((40076, 80152))
            SZA_L=SZA_L*(sza_l[m-1].reshape(-1,1))
            img[SZA_L>1]=np.nan
            sc_01=rasterio.open(f"/data2/hzy/albedo2/snowcover/snowcover/2001_{m}_snowcover_v3_LCCS2.tif").read(1)
            sc_20=rasterio.open(f"/data2/hzy/albedo2/snowcover/snowcover/2020_{m}_snowcover_v3_LCCS2.tif").read(1)
            sc=(1000-np.maximum(sc_01,sc_20))/1000
            img*=sc
            img[img==0]=np.nan
            tif_save(img,f'/data2/hzy/ssd_hzy/G3/{b}{m}_85_final.tif',global_trf,p='4326')
            count[~np.isnan(img)]+=1
            img[np.isnan(img)]=0
            all+=img
        all/=count
        tif_save(all,f'/data2/hzy/ssd_hzy/G3/{b}_85_final.tif',global_trf,p='4326')


    #Calculate albedo-induced RF using seven kernels
    kernels=torch.load(f"/data2/hzy/albedo2/kernel/kernels.pth")
    sza,sza_l=torch.load('/data/hk/albedo/SZA.pth')
    for key in ['HadGEM2', 'HadGEM3', 'CAM3', 'CAM5', 'ECHAM6','ERAI','ERA5']:
        sw_y=kernels[key]
        print(key)
        for b in ['SSI','NMDI','NDVI']:
            img_y=np.zeros((40076, 80152)).astype(np.float32)
            for m in tqdm(range(1,13)):
                SZA_L=np.ones((40076, 80152))
                SZA_L=SZA_L*(sza_l[m-1].reshape(-1,1))
                now=datetime.datetime.strptime(f'2001-{m}-1', '%Y-%m-%d')
                month_time=(now+relativedelta(months=1)-now).total_seconds()
                img=rasterio.open(f'/data2/hzy/ssd_hzy/G3/{b}{m}_85_final.tif').read(1)
                sw=sw_y[m-1]
                sw=resample_wb_fraction(sw,width=80152,height=40076,resample_alg = gdalconst.GRIORA_NearestNeighbour)
                img[img==0]=np.nan
                img[SZA_L>1]=np.nan
                eg=(img)*sw*month_time
                eg[np.isnan(eg)]=0
                img_y+=eg
            y_len=365*24*3600
            img_y/=y_len
            tif_save(img_y,f'/data2/hzy/ssd_hzy/G3/{b}_{key}_eg_final.tif',global_trf,p='4326')
