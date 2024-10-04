import sys
sys.path.append('..')
from tool import *
warnings.filterwarnings("ignore")
if __name__=='__main__':    
    
    #Check the pixel size of the file.
    for y in range(2001,2021):
        for m in range(1,13):
            print('*'*20,y,m,'*'*20)
            # /data/hk/albedo/albedo/2020_1_albedo/
            fs=glob(f'/data/hk/albedo/albedo/{y}_{m}_albedo/{y}_{m}-*.tif')
            print(len(fs))
            for f in fs:
                ds=gdal.Open(f)
                if ds.GetGeoTransform()[1]!=0.004491576420597608:
                    print(f)
                    
    #Splice
    print(y,m)
    fs=glob(f'/data/hk/albedo/{y}_{m}_albedo2/{y}_{m}*.tif')
    for i in range(1,15):
        print('-'*20,i,'-'*20)
        time.sleep(1)
        splice(fs,f'/data/hk/albedo/{y}_{m}_albedo2/{y}_{m}_band{i}.tif',pixel_size=0.004491576420597608,band_num=i,type='int32')
    
    #Delete source files
    print('*'*20,y,m,'*'*20)
    fs=glob(f'/data/hk/albedo/{y}_{m}_albedo2/{y}_{m}-*.tif')
    for f in fs:
        os.remove(f)
        
    #Visual inspection
    print('*'*20,y,m,'*'*20)
    img=get_img(y,m,'BSA_vis_snow',coord=None,path=None)
    plt.imshow(img)
    plt.colorbar()
    plt.show()
        
        
