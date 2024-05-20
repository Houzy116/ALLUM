from tool import *
warnings.filterwarnings("ignore")
if __name__=='__main__':
    # for y in [2006]:

    #     for m in range(2,13):
    #         print('*'*20,y,m,'*'*20)
    #         fs=glob(f'/data/hk/albedo/albedo/{y}_{m}_albedo/{y}_{m}-*.tif')
    #         for i in range(1,15):
    #             # print('-'*20,i,'-'*20)
    #             # time.sleep(1)
    #             splice(fs,f'/data/hk/albedo/albedo/{y}_{m}_albedo/{y}_{m}_band{i}.tif',pixel_size=0.004491576420597608,band_num=i,type='int32')
    # for y in [2009]:
    
    #     for m in range(1,13):
    #         print('*'*20,y,m,'*'*20)
    #         fs=glob(f'/data/hk/albedo/albedo/{y}_{m}_albedo/{y}_{m}-*.tif')
    #         for i in range(1,15):
    #             # print('-'*20,i,'-'*20)
    #             # time.sleep(1)
    #             splice(fs,f'/data/hk/albedo/albedo/{y}_{m}_albedo/{y}_{m}_band{i}.tif',pixel_size=0.004491576420597608,band_num=i,type='int32')
    
    # for m in range(8,13):
    #     print('*'*20,y,m,'*'*20)
    #     fs=glob(f'/data/hk/albedo/albedo/{y}_{m}_albedo/{y}_{m}-*.tif')
    #     for i in range(1,15):
    #         # print('-'*20,i,'-'*20)
    #         # time.sleep(1)
    #         splice(fs,f'/data/hk/albedo/albedo/{y}_{m}_albedo/{y}_{m}_band{i}.tif',pixel_size=0.004491576420597608,band_num=i,type='int32')

    for y in [2003,2004,2005,2006,2007,2008,2009]:
        print('*'*20,y,'*'*20)
        fs=glob(f"/data/hk/albedo/landcover/{y}_landcover/{y}_landcover-*.tif")
        splice(fs,f"/data/hk/albedo/landcover/{y}_landcover/{y}_landcover.tif",pixel_size=0.004491576420597608,band_num=1,type='int32')