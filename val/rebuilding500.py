import sys
sys.path.append('..')
from tool import *

def rebuilding_all(year,month,landcover,stat,landcover_type):
    if landcover_type=='LCCS2':
        lc_len=10
    elif landcover_type=='LCCS3':
        lc_len=9
    elif landcover_type=='IGBP':
        lc_len=16
    elif landcover_type=='LCCS1':
        lc_len=15 
    albedo=stat[f'{year}-{month}-albedo']
    c=str(year)+str(month).zfill(2)
    snow_cover=rasterio.open(f"/data2/hzy/albedo2/snowcover/snowcover/{year}_{month}_snowcover_v3_{landcover_type}.tif").read(1).astype(np.float32)/10
    # snow_cover/=10landcover[landcover==100]=20
    # snow_cover[snow_cover==300]=np.nan
    landcover[landcover==100]=20
    landcover[landcover==0]=20
    snow_cover[landcover==2]=100
    snow_cover[landcover==20]=0
    albedo_500=np.zeros((40076, 80152)).astype(np.float32)
    print(year,month)
    time.sleep(1)
    for lat in tqdm(range(90,-90,-1), desc=f'{year}-{month}'):
        for lon in range(-180,180):
            coord=[lon,lat]
            x=round((coord[0]-global_trf[0])/global_trf[1])
            x_s=round((coord[0]+1-global_trf[0])/global_trf[1])-x
            y=round((coord[1]-global_trf[3])/global_trf[5])
            y_s=round((coord[1]-1-global_trf[3])/global_trf[5])-y
            k=np.zeros((y_s,x_s))
            albedo[np.isnan(albedo)]=0
            albedo_snow=albedo[-1,90-coord[1],coord[0]+180]
            
            if sza_L[year-2001,month-1,90-coord[1],coord[0]+180]==2:
                kk=np.ones((y_s,x_s))*np.nan
            else:
                for t in range(lc_len):
                    if t==1:
                        continue
                    else:
                        k[landcover[y:y+y_s,x:x+x_s]==t+1]=albedo[t,90-coord[1],coord[0]+180]
                kk=k*(1-snow_cover[y:y+y_s,x:x+x_s]/100)+albedo_snow*snow_cover[y:y+y_s,x:x+x_s]/100
                if sza_L[year-2001,month-1,90-coord[1],coord[0]+180]==1:
                    kk+=1
            albedo_500[y:y+y_s,x:x+x_s]=kk
    tif_save(albedo_500,f"/data2/hzy/ssd_hzy/albedo_rebuilding/rebuilding2_albedo_{year}_{month}_{landcover_type}.tif",global_trf,p='4326')


if __name__=='__main__':

    for landcover_type in ['LCCS3','IGBP']: 
        print(landcover_type)
        if landcover_type=='LCCS2':
            lc_len=10
        elif landcover_type=='LCCS3':
            lc_len=9
        elif landcover_type=='IGBP':
            lc_len=16
        elif landcover_type=='LCCS1':
            lc_len=15 

        _,sza_L=torch.load(root_path+'sza_and_szaL.pth')
        WB_sky_fraction=torch.load("/data/hk/albedo/white_sky_fraction/white_sky_fraction2.pth")
        landtypes=['landtype'+str(i) for i in range(1,lc_len+1)]+['snow']
        stat={}
        for month in range(1,13):
            NC=nc.Dataset(f"/data2/hzy/albedo2/albedo_information_nc2/fill/month_{month}_{landcover_type}.nc",'r')
            for year in range(2001,2021):
                types_stat_albedo=[]
                for type in landtypes:
                    now=datetime.datetime.strptime(f'2001-{month}-1', '%Y-%m-%d')
                    month_time=(now+relativedelta(months=1)-now).total_seconds()
                    bs=NC.variables[f'albedo_BSA_shortwave-{type}'][year-2001]
                    ws=NC.variables[f'albedo_WSA_shortwave-{type}'][year-2001]
                    wf=WB_sky_fraction[(year-2001)*12+month-1]
                    types_stat_albedo.append(((ws*wf+bs*(1-wf))/1000))
                types_stat_albedo=np.stack(types_stat_albedo)
                types_stat_albedo[types_stat_albedo==0]=np.nan
                stat[f'{year}-{month}-albedo']=types_stat_albedo 
            NC.close() 
            
        for y in [2001,2020]:
            landcover=rasterio.open(f'/data2/hzy/albedo2/LCCS/LCCS/{year}_{landcover_type}_v2.tif').read(1)
            for m in range(1,13):
                albedo_500=rebuilding_all(y,m,landcover,stat,landcover_type)