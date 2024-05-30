import sys
sys.path.append('..')
from tool import *
from preprocess.DDRF import resample_wb_fraction
def rebuilding_all(year,month,landcover,stat):
    albedo=stat[f'{year}-{month}-albedo']
    c=str(year)+str(month).zfill(2)
    snow_cover=rasterio.open(root_path+f'snow/snow_monthly/{c}.tif').read(1)
    snow_cover[landcover==15]=100
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
                for t in range(17):
                    if t==14:
                        continue
                    else:
                        k[landcover[y:y+y_s,x:x+x_s]==t+1]=albedo[t,90-coord[1],coord[0]+180]
                kk=k*(1-snow_cover[y:y+y_s,x:x+x_s]/100)+albedo_snow*snow_cover[y:y+y_s,x:x+x_s]/100
                if sza_L[year-2001,month-1,90-coord[1],coord[0]+180]==1:
                    kk+=1
            albedo_500[y:y+y_s,x:x+x_s]=kk
    tif_save(albedo_500,f"/data2/hzy/ssd_hzy/albedo_rebuilding/rebuilding2_albedo_{year}_{month}_2.tif",global_trf,p='4326')
    

def get_ma(y,m,):
    f_white_data=torch.load("/data/hk/albedo/white_sky_fraction/white_sky_fraction2.pth")
    wb=f_white_data[(y-2001)*12+m-1]
    wb_resample=resample_wb_fraction(wb,width=80152,height=40076,resample_alg = gdalconst.GRIORA_NearestNeighbour)
    wb_resample[wb_resample==2]=np.nan
    WS=rasterio.open(root_path+f'albedo/{y}_{m}_albedo/{y}_{m}_WS2.tif').read(1)
    BS=rasterio.open(root_path+f'albedo_rebuilding/{y}_{m}_BS.tif').read(1)
    a=((WS*wb_resample+BS*(1-wb_resample))/1000)
    return a

if __name__=='__main__':
    #MCD43A3均值合成
    for y in range(2001,2021):
        landcover=rasterio.open(root_path+f"landcover/{y}_landcover/{y}_landcover.tif").read(1)

        for m in range(1,13):
            a=get_ma(y,m)
            tif_save(a,f'/data2/hzy/ssd_hzy/albedo_rebuilding/source2_{y}_{m}_sw_albedo2_2.tif',global_trf,p='4326')
            del a
            gc.collect()
            time.sleep(0.5)

    _,sza_L=torch.load(root_path+'sza_and_szaL.pth')
    WB_sky_fraction=torch.load("/data/hk/albedo/white_sky_fraction/white_sky_fraction2.pth")
    landtypes=['landtype'+str(i) for i in range(1,18)]+['snow']
    stat={}
    for month in range(1,13):
        NC=nc.Dataset(root_path+f'information_fill2/month_{month}.nc','r')
        for year in range(2001,2021):
            types_stat=[]
            types_stat_albedo=[]
            types_stat_area=[]
            sw=torch.load(f"/data/hk/albedo/kernel/{month}_SW_1degree2.tif")
            for type in landtypes:
                now=datetime.datetime.strptime(f'2001-{month}-1', '%Y-%m-%d')
                month_time=(now+relativedelta(months=1)-now).total_seconds()
                bs=NC.variables[f'albedo_BSA_shortwave-{type}'][year-2001]
                ws=NC.variables[f'albedo_WSA_shortwave-{type}'][year-2001]
                area_type=NC.variables[f'area-{type}'][year-2001] 
                area_land=NC.variables[f'area-land'][year-2001] 
                wf=WB_sky_fraction[(year-2001)*12+month-1]
                # area=NC.variables['area-grid'][year-2001]
                # types_stat.append((1-(ws*wf+bs*(1-wf))/1000)*area_type*1e6*sw*month_time)
                types_stat_albedo.append(((ws*wf+bs*(1-wf))/1000))
                area_type[sza_L[year-2001,month-1]>=2]=np.nan
                types_stat_area.append(area_type)
                                #   (1-(ws*wf+bs*(1-wf))/1000)*area_type*1e6*month_time
                # types_stat.append((ws*wf+bs*(1-wf))*area_type)
            # types_stat=np.stack(types_stat)
            types_stat_albedo=np.stack(types_stat_albedo)
            types_stat_area=np.stack(types_stat_area)
            # stat[f'{year}-{month}']=types_stat  
            
            types_stat_albedo[types_stat_albedo==0]=np.nan
            stat[f'{year}-{month}-albedo']=types_stat_albedo
            stat[f'{year}-{month}-typearea']=types_stat_area
            stat[f'{year}-landarea']=area_land   
        NC.close() 
        
    for year in range(2001,2021):
        year_stat=[]
        year_stat_albedo_weightarea=[]
        for month in range(1,13):
            year_stat_albedo_weightarea.append(np.nansum((stat[f'{year}-{month}-albedo']*stat[f'{year}-{month}-typearea']/stat[f'{year}-landarea']),axis=0))
        year_stat_albedo=np.nanmean(year_stat_albedo_weightarea,axis=0)
        stat[f'{year}-albedo']=year_stat_albedo
        
    for y in range(2001,2021):
        landcover=rasterio.open(root_path+f"landcover/{y}_landcover/{y}_landcover.tif").read(1)
        for m in range(1,13):
            rebuilding_all(y,m,landcover,stat)
