#拼接
#run splice.py
from tool import *
if __name__=='__main__':
    # for y in range(2001,2021):
    #     for m in range(1,13):
    #         print('LAI   :',y,m)
    #         fs=glob(f'/data2/LAI/{y}_{m}*.tif')
    #         splice(fs,f'/data/hk/albedo/LAI/{y}_{m}.tif',pixel_size=0.004491576420597608,band_num=1,type='float32',masked=200)

    # for year in range(2002,2011):
    #     for month in range(1,13):
    #         print('SM:   ',year,month)
    #         ds=gdal.Open(f'/data/hk/albedo/SM/{year}_{month}.tif')
    #         # print(ds.GetGeoTransform()) 
    #         img=ds.ReadAsArray()
    #         new_img=resample_wb_fraction(img,width=80152,height=40076,resample_alg = gdalconst.GRIORA_NearestNeighbour)
    #         tif_save_snowfre(new_img,f'/data/hk/albedo/SM/{year}_{month}_500m.tif',global_trf,p='4326',novalue=None,valuetype=6)        
    # for y in [2001,2003,2005,2006]:
    for y in [2011]: 
        if y in [2001,2009]:
            for m in range(9,13):
                print(f'------------------ {y}/{m} ------------------')
                for lat in tqdm(range(90,-90,-10)):
                    cc=get_10lat_information_LandS(y,m,lat)
                    cc.get_10lat_information()
        else:
            for m in range(1,13):
                print(f'------------------ {y}/{m} ------------------')
                for lat in tqdm(range(90,-90,-10)):
                    cc=get_10lat_information_LandS(y,m,lat)
                    cc.get_10lat_information()            