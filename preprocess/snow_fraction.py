import sys
sys.path.append('..')
from tool import *
warnings.filterwarnings("ignore")

#IDW fill missing value
def fill_snow_cover(y,m,landcover_type):
    print(f'{y}-{m}  ')
    img=rasterio.open(f"/data2/hzy/albedo2/snowcover/snowcover/{y}_{m}_snowcover_v2_{landcover_type}.tif").read(1)
    se=[]
    
    #Extract regions where the solar zenith angle (SZA) is less than 85 degrees
    if img[0,0]!=3000:
        se.append(0)
        for i in range(40076):
            if img[i,0]==3000:
                se.append(i)
                break
    else:
        for i in range(40076):
            if img[i,0]!=3000:
                se.append(i)
                break
        for i in range(se[0],40076):
            if img[i,0]==3000:
                se.append(i)
                break
    if len(se)==1:
        se.append(40076)         
    k=(img==2000).sum()
    img2=img[se[0]:se[1]].copy()
    
    # IDW fill missing value  
    msd=25#IDW max search distance
    for i in range(10):
        out = fill.fillnodata(img2, (img2!=2000),max_search_distance=msd)
        nan_count=(out==2000).sum()
        if nan_count==0:
            img[se[0]:se[1]]=out
            output_file=f"/data2/hzy/albedo2/snowcover/snowcover/{y}_{m}_snowcover_v3_{landcover_type}.tif"
            tif_save(img,output_file,global_trf,p='4326',novalue=None,valuetype=3)
            break
        else:
            msd+=5
    return k
    
    
if __name__=='__main__':
    
    #Get SZA
    _,sza_L=torch.load(root_path+'sza_and_szaL.pth')
    year,month=2001,1
    kk=np.zeros((20,12,40076))
    for year in range(2001,2021):
        for month in range(1,13):
            
            for lat in range(90,-90,-1):
                    y=round((lat-global_trf[3])/global_trf[5])
                    y_s=round((lat-1-global_trf[3])/global_trf[5])-y
                    kk[year-2001,month-1,y:y+y_s]=sza_L[year-2001,month-1,90-lat,0]
                    
    #missing value->2000
    #ice->1000
    #water->0
    #SZA>85->3000                
    for landcover_type in ['LCCS1','LCCS2','LCCS3','IGBP']:
        for y in range(2001,2021):
            landcover=rasterio.open(f'/data2/hzy/albedo2/LCCS/LCCS/{y}_{landcover_type}_v2.tif').read(1)
            for m in range(1,13):
                print(y,m)
                #Monthly snow cover fraction
                img=rasterio.open(f"/data2/hzy/albedo2/snowcover/snowcover/{y}_{m}_snowcover.tif").read(1)
                mask=rasterio.open(f"/data2/hzy/albedo2/snowcover/snowcover_mask/{y}_{m}_snowcover_msk.tif").read(1)
                img[mask==0]=2000#missing value
                img[landcover==2]=1000#ice
                img[landcover==100]=0#water
                img[landcover==0]=0#water
                SZA_L=np.ones((40076, 80152))
                SZA_L=SZA_L*(kk[y-2001,m-1].reshape(-1,1))
                img[SZA_L==2]=3000#sza>85
                output_file=f"/data2/hzy/albedo2/snowcover/snowcover/{y}_{m}_snowcover_v2_{landcover_type}.tif"
                tif_save(img,output_file,global_trf,p='4326',novalue=None,valuetype=3)
    
    #IDW fill missing value (Less than 0.5%)
    for landcover_type in ['LCCS1','LCCS2','LCCS3','IGBP']:
        kk=[]
        for y in range(2001,2021):
            for m in range(1,13):
                k=fill_snow_cover(y,m,landcover_type)
                kk.append(k)
        torch.save(kk,f'fill_count_{landcover_type}.pth')