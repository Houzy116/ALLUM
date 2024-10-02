import sys
sys.path.append('..')
from tool import *
warnings.filterwarnings("ignore")

if __name__=='__main__':
    
    #Split
    for y in range(2001,2021):
        fs=glob(f"/data2/hzy/albedo2/LCCS/landcover_LCCS_source/{y}*.tif")
        names=['LCCS1-cf','LCCS1-cf','LCCS2','LCCS3','LCCS2-cf','LCCS3-cf','IGBP']
        for i in range(4):
            output_file=f'/data2/hzy/albedo2/LCCS/LCCS/{y}_{names[i]}.tif'

            splice(fs,output_file,pixel_size=0.004491576420597608,band_num=i+1,type='int8',masked=None)

    #Assign new numbers to the LULC categories.
    for name in ['LCCS1','LCCS2','LCCS3','IGBP']:
        for y in range(2001,2021):
            print(y,name)
            img=rasterio.open(f'/data2/hzy/albedo2/LCCS/LCCS/{y}_{name}.tif').read(1)       
            
            #water bodies:100
            if name=='LCCS3':
                fm_LCCS=[[3,100],[10,3],[20,4],[27,5],[30,6],[40,7],[50,8],[51,9]]
            elif name=='LCCS2':
                fm_LCCS=[[3,100],[9,3],[10,4],[20,5],[25,6],[30,7],[35,8],[36,9],[40,10]]
            elif name=='IGBP':
                fm_LCCS=[[18,100],[2,1],[16,2],[17,16]]
            elif name=='LCCS1':
                fm_LCCS=[[3,100],[11,3],[12,4],[13,5],[14,6],[15,7],[16,8],[21,9],[22,10],[31,11],[32,12],[41,13],[42,14],[43,15]]
            
            if name=='IGBP':
                img[31000:34000,70000:]=17
                img=img+1
            for i in range(len(fm_LCCS)):
                img[img==fm_LCCS[i][0]]=fm_LCCS[i][1]
            tif_save(img,f'/data2/hzy/albedo2/LCCS/LCCS/{y}_{name}_v2.tif',global_trf,p='4326',novalue=None,valuetype=1)
        
