import sys
sys.path.append('..')
from tool import *
warnings.filterwarnings("ignore")

    
def resample_wb_fraction(img,width=360,height=180,resample_alg = gdalconst.GRIORA_Bilinear):
    driver = gdal.GetDriverByName('MEM')
    src_ds = driver.Create('', img.shape[1],img.shape[0], 1, 6)
    src_ds.GetRasterBand(1).WriteArray(img)
    src_ds.GetRasterBand(1).SetNoDataValue(2)
    src_ds.FlushCache()

    data = src_ds.GetRasterBand(1).ReadAsArray(buf_xsize=width,buf_ysize=height,resample_alg = resample_alg)

    return data
def trf_lon(img):
    img2=np.zeros(img.shape)
    img2[:,:180]=img[:,180:]
    img2[:,180:]=img[:,:180]
    return img2

if __name__=='__main__':

    nf=nc.Dataset("/data2/hzy/albedo/alb.kernel.nc")
    for m in range(12):
        img=nf.variables['FSNT'][m]
        img=-np.flip(img,axis=0)*100
        
        img2=resample_wb_fraction(img)
        img3=trf_lon(img2)
        torch.save(img3,f"/data/hk/albedo/{m}_SW_1degree_kernel.pth")