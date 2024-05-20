from tool import *
if __name__=='__main__':

        
        
    
    def rebuilding_all(year,month,landcover):
        albedo=stat[f'{year}-{month}-albedo']
        c=str(year)+str(month).zfill(2)
        snow_cover=rasterio.open(root_path+f'snow/snow_monthly/{c}.tif').read(1)
        snow_cover[landcover==15]=100
        albedo_500=np.zeros((40076, 80152)).astype(np.float32)
        print(year,month)
        time.sleep(1)
    #     break
    # break
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
        tif_save(albedo_500,root_path+f"albedo_rebuilding/rebuilding2_albedo_{year}_{month}.tif",global_trf,p='4326')
        
    # for y in range(2001,2021):
    #     landcover=rasterio.open(root_path+f"landcover/{y}_landcover/{y}_landcover.tif").read(1)
    #     for m in range(1,13):
    #         rebuilding_all(y,m,landcover)
            
            
            
            
            
            
            
            
            
            
            
            
    # from tool import *
    # def get_ma(y,m):
    #     black_nir=get_wb_fraction_data('nbd')
    #     white_nir=get_wb_fraction_data('ndd')
    #     black_vis=get_wb_fraction_data('vbd')
    #     white_vis=get_wb_fraction_data('vdd')
    #     f_white=(white_nir[0]+white_vis[0])/(white_nir[0]+black_nir[0]+white_vis[0]+black_vis[0])
    #     f_white_data=f_white.data
    #     f_white_mask=f_white.mask
    #     f_white_data[f_white_mask==1]=2
    #     f_white_data[f_white_mask==1]=2
    #     # ds=[resample_wb_fraction(f_white_data[i]) for i in range(240)]
    #     # f_white_resample=np.stack(ds)
    #     # f_white_resample[f_white_resample==2]=np.nan
    #     wb=f_white_data[(y-2001)*12+m-1]
    #     wb_resample=resample_wb_fraction(wb,width=80152,height=40076,resample_alg = gdalconst.GRIORA_NearestNeighbour)
    #     wb_resample[wb_resample==2]=np.nan
    #     WS=rasterio.open(root_path+f'albedo/{y}_{m}_albedo/{y}_{m}_WS2.tif').read(1)
    #     BS=rasterio.open(root_path+f'albedo/{y}_{m}_albedo/{y}_{m}_BS2.tif').read(1)
    #     a=((WS*wb_resample+BS*(1-wb_resample))/1000)
    #     return a
    
    
    def get_ma2(y,m,WS):

        black_nir=get_wb_fraction_data('nbd')
        white_nir=get_wb_fraction_data('ndd')
        black_vis=get_wb_fraction_data('vbd')
        white_vis=get_wb_fraction_data('vdd')
        f_white=(white_nir[0]+white_vis[0])/(white_nir[0]+black_nir[0]+white_vis[0]+black_vis[0])
        f_white_data=f_white.data
        f_white_mask=f_white.mask
        f_white_data[f_white_mask==1]=2
        f_white_data[f_white_mask==1]=2
        # ds=[resample_wb_fraction(f_white_data[i]) for i in range(240)]
        # f_white_resample=np.stack(ds)
        # f_white_resample[f_white_resample==2]=np.nan
        wb=f_white_data[(y-2001)*12+m-1]
        wb_resample=resample_wb_fraction(wb,width=80152,height=40076,resample_alg = gdalconst.GRIORA_NearestNeighbour)
        wb_resample[wb_resample==2]=np.nan
        # WS=rasterio.open(root_path+f'albedo/{y}_{m}_albedo/{y}_{m}_WS2.tif').read(1)
        BS=rasterio.open(root_path+f'albedo_rebuilding/{y}_{m}_BS.tif').read(1)
        a=((WS*wb_resample+BS*(1-wb_resample))/1000)
        return a
    #直接均值合成
    for y in range(2001,2021):
        landcover=rasterio.open(root_path+f"landcover/{y}_landcover/{y}_landcover.tif").read(1)

        for m in range(1,13):
            # rebuilding_all(y,m,landcover)
            print(y,m)
            BS_img=rasterio.open(f'/ssd_hzy/albedo/{y}_{m}_albedo/{y}_{m}_band5.tif').read(1)
            # print(1)
            BS_img_snowfree=rasterio.open(f'/ssd_hzy/albedo/{y}_{m}_albedo/{y}_{m}_band11.tif').read(1)
            # print(1)
            count=rasterio.open(f'/ssd_hzy/albedo/{y}_{m}_albedo/{y}_{m}_band13.tif').read(1)
            # print(1)
            snow_count=rasterio.open(f'/ssd_hzy/albedo/{y}_{m}_albedo/{y}_{m}_band14.tif').read(1)
            # print(1)
            
            count=count.astype(np.float32)
            count[count==0]=np.nan

            BS=(BS_img*snow_count+BS_img_snowfree*(count-snow_count))/count
            del BS_img,BS_img_snowfree,count,snow_count
            

            tif_save_snowfre(BS,root_path+f'albedo_rebuilding/{y}_{m}_BS.tif',global_trf,p='4326')
            
            del BS
            gc.collect()
            
        # for m in range(1,13):
            print(y,m)
            BS_img=rasterio.open(f'/ssd_hzy/albedo/{y}_{m}_albedo/{y}_{m}_band6.tif').read(1)
            BS_img_snowfree=rasterio.open(f'/ssd_hzy/albedo/{y}_{m}_albedo/{y}_{m}_band12.tif').read(1)
            count=rasterio.open(f'/ssd_hzy/albedo/{y}_{m}_albedo/{y}_{m}_band13.tif').read(1)
            snow_count=rasterio.open(f'/ssd_hzy/albedo/{y}_{m}_albedo/{y}_{m}_band14.tif').read(1)
            
            count=count.astype(np.float32)
            count[count==0]=np.nan
            
            WS=(BS_img*snow_count+BS_img_snowfree*(count-snow_count))/count
            del BS_img,BS_img_snowfree,count,snow_count
            gc.collect()

            # tif_save_snowfre(BS,root_path+f'albedo_rebuilding/{y}_{m}_WS.tif',global_trf,p='4326')


        # b=np.zeros((40076,80152)).astype(np.float32)
        # for m in range(1,13):
            print(y,m)
            a=get_ma2(y,m,WS)
            tif_save_snowfre(a,root_path+f'albedo_rebuilding/source2_{y}_{m}_sw_albedo2.tif',global_trf,p='4326')
            del a,WS
            gc.collect()
            # shutil.remove(root_path+f'albedo/{y}_{m}_albedo/{y}_{m}_WS2.tif')
            os.remove(root_path+f'albedo_rebuilding/{y}_{m}_BS.tif')
            time.sleep(0.5)
        #     b+=a
        # b=b/12
        # tif_save_snowfre(b,root_path+f'albedo/{y}_sw_albedo2.tif',global_trf,p='4326')
        
        
        
        
        
        
        
    _,sza_L=torch.load(root_path+'sza_and_szaL.pth')
    WB_sky_fraction=torch.load("/data/hk/albedo/white_sky_fraction/white_sky_fraction.pth")
    landtypes=['landtype'+str(i) for i in range(1,18)]+['snow']
    stat={}
    for month in range(1,13):
        NC=nc.Dataset(root_path+f'information_fill2/month_{month}.nc','r')
        for year in range(2001,2021):
            types_stat=[]
            types_stat_albedo=[]
            types_stat_area=[]
            sw=torch.load(f"/data/hk/albedo/SW_MERRA2/{month}_SW_1degree.tif")
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
            rebuilding_all(y,m,landcover)
