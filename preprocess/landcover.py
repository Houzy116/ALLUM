import sys
sys.path.append('..')
from tool import *
warnings.filterwarnings("ignore")

if __name__=='__main__':
    for y in range(2001,2021):
        print('*'*20,y,'*'*20)
        fs=glob(f"/data/hk/albedo/landcover/{y}_landcover/{y}_landcover-*.tif")
        splice(fs,f"/data/hk/albedo/landcover/{y}_landcover/{y}_landcover.tif",pixel_size=0.004491576420597608,band_num=1,type='int32')
        
    ys=[os.path.basename(i).split('_')[0] for i in glob(root_path+f"landcover/*_landcover/*_landcover.tif")]
    ys.sort()
    #修正
    for y in ys:
        print(y)
        img=get_img(y,1,'landcover')
        img[31000:34000,70000:]=20
        path=root_path+f"landcover/{y}_landcover/{y}_landcover.tif"
        tif_save(img,path,global_trf)

    for y in range(2001,2021):
        path=f"/data/hk/albedo/landcover/{y}_landcover/{y}_landcover.tif"
        ds=gdal.Open(path)
        print(ds.GetGeoTransform())
        img=ds.ReadAsArray()
        print(img.shape)
        img[img==0]=20
        plt.imshow(img)
        plt.colorbar()
        plt.show()
        tif_save(img,path,global_trf,p='4326',novalue=0,valuetype=1)
        
