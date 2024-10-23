
import sys
sys.path.append('..')
from tool import *
warnings.filterwarnings("ignore")

class get_10lat_information():
    '''
    return condition, areas and albedomean
    condition:
              1--grid have enough data to Statistics
              2--all ocean
              4--SZA>85
    if condition=2 return areas is [area of the grid, 0]
                   return albedomean is []
    elif condition=1 return areas is [area of 17-tpyes landcover and snow,area of the grid, land area, no nan land area]
                   return albedomean is []
    elif condition=4 return areas is [area of the grid, land area, no nan land area]
                    return albedomean is []

    '''
    def __init__(self,y,m,lat):
        self.y=y
        self.m=m
        self.lat=lat
        self.bandmap={'BSA_vis_snow':1,
            'WSA_vis_snow':2,
            'BSA_nir_snow':3,
            'WSA_nir_snow':4,
            'BSA_shortwave_snow':5,
            'WSA_shortwave_snow':6,
            'BSA_vis_snowfree':7,
            'WSA_vis_snowfree':8,
            'BSA_nir_snowfree':9,
            'WSA_nir_snowfree':10,
            'BSA_shortwave_snowfree':11,
            'WSA_shortwave_snowfree':12,
            'count':13,
            'snow_count':14
            }
        self.global_trf=(-180.00441663186973, 0.004491576420597608, 0.0, 90.00220831593487, 0.0, -0.004491576420597608)
        # print(f'Loading {self.y}/{self.m}/{self.lat} Image...')
        
        
    def get_10lat_information(self):
        # print('Computing Information...')
        # time.sleep(1)
        self.get_all_band()
        for landcover_type in ['LCCS1','IGBP']:
            result={}
            for lat in range(self.lat,self.lat-10,-1):
                for lon in range(-180,180):
                    result[str(lat)+','+str(lon)]=self.get_1grid_information([lon,lat],landcover_type)
            torch.save(result,f'/data2/hzy/albedo2/albedo_information/result_{self.y}_{self.m}_{self.lat}_{landcover_type}.pth')
    
    def get_all_band(self):
        self.landcover2=self.get_img_10lat('LCCS1')
        self.landcover3=self.get_img_10lat('IGBP')
        self.count=self.get_img_10lat('count')
        self.snow_count=self.get_img_10lat('snow_count')
        self.snow_fre2=self.get_img_10lat('snow_fre_LCCS1')
        self.snow_fre3=self.get_img_10lat('snow_fre_IGBP')
        bands_snow=['BSA_vis_snow', 
            'WSA_vis_snow', 
            'BSA_nir_snow', 
            'WSA_nir_snow', 
            'BSA_shortwave_snow', 
            'WSA_shortwave_snow']
        bands_snowfree=['BSA_vis_snowfree', 
            'WSA_vis_snowfree', 
            'BSA_nir_snowfree', 
            'WSA_nir_snowfree', 
            'BSA_shortwave_snowfree', 
            'WSA_shortwave_snowfree']
        albedo_snow=[]
        albedo_snowfree=[]
        for b in bands_snow:
            albedo_snow.append(self.get_img_10lat(b))
        for b in bands_snowfree:
            albedo_snowfree.append(self.get_img_10lat(b))
        self.albedo_snow=np.stack(albedo_snow)
        self.albedo_snowfree=np.stack(albedo_snowfree)
    def get_img_10lat(self,bandname):
        if bandname=='landcover_d':
            path=root_path+f"landcover/{self.y}_landcover/{self.y}_landcover.tif"
        elif bandname=='snow_fre_LCCS1':
            # c=str(self.y)+str(self.m).zfill(2)
            # path=root_path+f'snow/snow_monthly/{c}.tif'
            path=f"/data2/hzy/albedo2/snowcover/snowcover/{self.y}_{self.m}_snowcover_v3_LCCS1.tif"
        elif bandname=='snow_fre_IGBP':
            # c=str(self.y)+str(self.m).zfill(2)
            # path=root_path+f'snow/snow_monthly/{c}.tif'
            path=f"/data2/hzy/albedo2/snowcover/snowcover/{self.y}_{self.m}_snowcover_v3_IGBP.tif"
        elif bandname in ['LCCS2','LCCS3','LCCS1','IGBP']:
            path=f'/data2/hzy/albedo2/LCCS/LCCS/{self.y}_{bandname}_v2.tif'
        
        else:
            if self.y not in [2016,2017,2018]:
                path=f'/data/hk/albedo/albedo/{self.y}_{self.m}_albedo/{self.y}_{self.m}_band{self.bandmap[bandname]}.tif'
            else:
            # raise()
            # path=root_path+f'albedo/{self.y}_{self.m}_albedo/{self.y}_{self.m}_band{self.bandmap[bandname]}.tif'
                path=f'/data2/hzy/albedo2/albedo/{self.y}_{self.m}_band{self.bandmap[bandname]}.tif'
        ds=gdal.Open(path)
        self.coord_y=round((self.lat-self.global_trf[3])/self.global_trf[5])
        coord_y_s=round((self.lat-10-self.global_trf[3])/self.global_trf[5])-self.coord_y
        img=ds.ReadAsArray(xoff=0,xsize=80152,yoff=self.coord_y,ysize=coord_y_s)
        return img

    def get_img_fromallband(self,coord,band,landcover_type='IGBP'):
        x=round((coord[0]-global_trf[0])/global_trf[1])
        x_s=round((coord[0]+1-global_trf[0])/global_trf[1])-x
        y=round((coord[1]-global_trf[3])/global_trf[5])
        y_s=round((coord[1]-1-global_trf[3])/global_trf[5])-y
        if band=='landcover':
            if landcover_type=='aaa':
                return self.landcover[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
            elif landcover_type=='LCCS1':
                return self.landcover2[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
            elif landcover_type=='IGBP':
                return self.landcover3[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
        if band=='count':
            return self.count[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
        if band=='snow_count':
            return self.snow_count[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
        if band=='snow_fre':
            if landcover_type=='aaa':
                return self.snow_fre[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
            elif landcover_type=='LCCS1':
                return self.snow_fre2[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
            elif landcover_type=='IGBP':
                return self.snow_fre3[y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
        if band=='albedo_snow':
            return self.albedo_snow[:,y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
        if band=='albedo_snowfree':
            return self.albedo_snowfree[:,y-self.coord_y:y+y_s-self.coord_y,x:x+x_s]
    # def get_1grid_information(self,coord):
    #     condition=1
    #     snow_fre=self.get_img_fromallband(coord,'snow_fre').astype(np.float32)/100
    #     if (snow_fre==2.55).sum()>0:#snow fre have nan value
    #         condition=4
    #     return condition
    def get_1grid_information(self,coord,landcover_type):
        if landcover_type=='IGBP':
            typen=16
        elif landcover_type=='LCCS1':
            typen=15
            # raise() 
        landcover=self.get_img_fromallband(coord,'landcover',landcover_type)
        landcover[landcover==100]=20
        landcover[landcover==0]=20
        area=((pi/180.0)*R*R*abs(math.sin(coord[1]/180.0*pi) - math.sin((coord[1]-pixel_with*landcover.shape[0])/180.0*pi)) * pixel_with*landcover.shape[1])/1000000
    
        #check grid availability
        condition=1#can be used gril
        
        if (landcover!=20).sum()==0:#all is ocean
            condition=2#too mach ocean
            areas=[area,0]
            albedomean=[]
        else:
            pixel_area=get_pixelarea(coord,landcover.shape[1],landcover.shape[0])
            land_area=np.sum((landcover!=20)*pixel_area)#land area
            # if (land_area/area)<0.05:#too mach ocean (>95%)
            #     condition=2
            #     areas=[area,land_area]
            #     albedomean=[] 
            # else:    
            # snowcount=get_img(y,m,'snow_count',coord)
            count=self.get_img_fromallband(coord,'count').copy().astype(np.float32)
            count_snow=self.get_img_fromallband(coord,'snow_count').copy().astype(np.float32)
            count_snowfree=count-count_snow 
            count_withocean=count.copy()      
            count[landcover==20]=0
            land_data_area=np.sum(((count>0) & (landcover!=20))*pixel_area)#have data land area
            

            snow_fre=self.get_img_fromallband(coord,'snow_fre',landcover_type).copy().astype(np.float32)/1000
            # snow_fre[landcover==15]=1
            if (snow_fre==3).sum()>0:#snow fre have nan value
                condition=4
                areas=[area,land_area,land_data_area]
                albedomean=[]
            else:
                #ocean data mask ，1 if landcover == ocean and exist albedo data else nan
                ocean_data_mask=(landcover==20).astype(np.float32)
                ocean_data_mask[ocean_data_mask==0]=np.nan
                #no nan mask ，1 if have data else nan
                nonan_mask_snow=(count_snow>0).astype(np.float32)
                nonan_mask_snow[landcover==20]=0
                nonan_mask_snow[nonan_mask_snow==0]=np.nan
                
                nonan_mask_snowfree=(count_snowfree>0).astype(np.float32)
                nonan_mask_snowfree[landcover==20]=0
                nonan_mask_snowfree[nonan_mask_snowfree==0]=np.nan
                # plt.imshow(snow_fre)
                # plt.colorbar()
                # plt.show()  
                snow_fre[landcover==20]=np.nan#piexl percent for snow (have interploted)
                    
                #landcover area
                landcover_classes_areaperc=[]
                for i in range(1,typen+1):
                    landcover_class_areaperc=(landcover==i)*(1-snow_fre)
                    landcover_classes_areaperc.append(landcover_class_areaperc)
                landcover_classes_areaperc=np.stack(landcover_classes_areaperc)#piexl percent for every land types
                landcover_classes_area=list(np.nansum((landcover_classes_areaperc*pixel_area).reshape((typen,-1)),axis=1))# 17 land types area km2
                
                #snow area
                snow_dataarea=np.nansum(snow_fre*pixel_area)# snowcover area km2
                
                areas=landcover_classes_area+[snow_dataarea,area,land_area,land_data_area]
                    
                # if (land_data_area/land_area)<0.25:#too mach nan value (nan>75%)
                #     condition=3
                #     albedomean=[]
                # else:#available grid    
                albedo_snow=self.get_img_fromallband(coord,'albedo_snow').copy()
                albedo_snowfree=self.get_img_fromallband(coord,'albedo_snowfree').copy()
                #landcover albedo
                landcover_classes_mask=np.expand_dims(nonan_mask_snowfree,axis=0)*landcover_classes_areaperc
                landcover_classes_mask[landcover_classes_mask>0]=1
                landcover_classes_mask[landcover_classes_mask==0]=np.nan
                k_snowfree=(landcover_classes_mask.reshape((typen,1,-1)))*(albedo_snowfree.reshape((1,6,-1)))#(17,1,n*n)*(1,6,n*n)
                landcover_classes_albedomean=np.nanmean(k_snowfree.reshape(typen*6,-1),axis=1).reshape(typen,6)#(17,6) average albedo for different bands and differen land types

                    
                #snow albedo
                snow_fre_mask=nonan_mask_snow*snow_fre
                snow_fre_mask[snow_fre_mask>0]=1
                snow_fre_mask[snow_fre_mask<=0]=np.nan
                
                k_snow=(snow_fre_mask.reshape((1,-1)))*(albedo_snow.reshape((6,-1)))#(1,n*n)*(6,n*n)
                snow_albedomean=np.nanmean(k_snow.reshape(6,-1),axis=1).reshape(1,6)#(1,6) average albedo for snow
                
                #total albedo
                count[count==0]=np.nan
                albedo=(albedo_snow*count_snow.reshape((1,landcover.shape[0],landcover.shape[1]))+albedo_snowfree*count_snowfree.reshape((1,landcover.shape[0],landcover.shape[1])))/(count.reshape(1,landcover.shape[0],landcover.shape[1]))
                albedo_totalmean=np.nanmean(albedo.reshape(6,-1),axis=1).reshape(1,6)
                
                #inshore ocean albedo
                count_withocean[count_withocean==0]=np.nan
                ocean_data_mask[np.isnan(count_withocean)]=0
                albedo_withocean=(albedo_snow*count_snow.reshape((1,landcover.shape[0],landcover.shape[1]))+albedo_snowfree*count_snowfree.reshape((1,landcover.shape[0],landcover.shape[1])))/(count_withocean.reshape(1,landcover.shape[0],landcover.shape[1]))
                k_ocean=(ocean_data_mask.reshape((1,-1)))*(albedo_withocean.reshape((6,-1)))#(1,n*n)*(6,n*n)
                ocean_albedomean=np.nanmean(k_ocean.reshape(6,-1),axis=1).reshape(1,6)#(1,6) average albedo for snow


                albedomean=np.concatenate([landcover_classes_albedomean,snow_albedomean,albedo_totalmean,ocean_albedomean],axis=0)


        return condition,areas,albedomean
    
def convert_to_nc(month,landcover_type):
    data,area,qa=get_month_raster_from_information(month,landcover_type)
    new_NC = nc.Dataset(f'/data2/hzy/albedo2/albedo_information_nc/month_{month}_{landcover_type}.nc', 'w', format='NETCDF4')
    new_NC.createDimension('time', 20)
    new_NC.createDimension('latitude', 180)
    new_NC.createDimension('longitude', 360)

    new_NC.createVariable("time", 'i2', ("time"))
    new_NC.createVariable("latitude", 'f', ("latitude"))
    new_NC.createVariable("longitude", 'f', ("longitude"))

    new_NC.variables['time'][:] = np.array(range(2001,2021))
    new_NC.variables['latitude'][:] = np.array(range(90,-90,-1))-0.5
    new_NC.variables['longitude'][:] = np.array(range(-180,180))+0.5

    bands=['BSA_vis', 
        'WSA_vis', 
        'BSA_nir', 
        'WSA_nir', 
        'BSA_shortwave', 
        'WSA_shortwave']
    if landcover_type=='IGBP':
        lc_len=16
    if landcover_type=='LCCS1':
        lc_len=15
        # raise()
    landtypes=['landtype'+str(i) for i in range(1,lc_len+1)]+['snow','total','water']
    area_types=['landtype'+str(i) for i in range(1,lc_len+1)]+['snow','grid','land','no_nan_of_land']

    for i in range(6):
        for j in range(lc_len+3):
            var_name=f"albedo_{bands[i]}-{landtypes[j]}"
            new_NC.createVariable(var_name, 'f4', ("time", "latitude", "longitude"))
            new_NC.variables[var_name][:]=data[j,i]
            
    for j in range(lc_len+4):
        var_name=f"area-{area_types[j]}"
        new_NC.createVariable(var_name, 'f4', ("time", "latitude", "longitude"))
        new_NC.variables[var_name][:]=area[j]
    new_NC.createVariable('qa','i2',("time", "latitude", "longitude"))
    new_NC.variables['qa'][:]=qa
    new_NC.close()

def get_month_raster_from_information(month,landcover_type):
    years=list(range(2001,2021))
    print(f'convert to cube for {month} 2001-2020')
    # print( 'data-----(landtype,band,year,lat,lon)')
    # print( 'area-----(landtype,year,lat,lon)')
    # print( 'qa-----(year,lat,lon)')
    if landcover_type=='LCCS2':
        data=np.zeros([13,6,20,180,360],dtype=np.float32)
        area=np.zeros([14,20,180,360],dtype=np.float32)
        qa=np.zeros([20,180,360],dtype=np.int8)
    if landcover_type=='LCCS3':
        data=np.zeros([12,6,20,180,360],dtype=np.float32)
        area=np.zeros([13,20,180,360],dtype=np.float32)
        qa=np.zeros([20,180,360],dtype=np.int8)
    if landcover_type=='IGBP':
        data=np.zeros([19,6,20,180,360],dtype=np.float32)
        area=np.zeros([20,20,180,360],dtype=np.float32)
        qa=np.zeros([20,180,360],dtype=np.int8)
    if landcover_type=='LCCS1':
        data=np.zeros([18,6,20,180,360],dtype=np.float32)
        area=np.zeros([19,20,180,360],dtype=np.float32)
        qa=np.zeros([20,180,360],dtype=np.int8)
        # raise()
    time.sleep(1)
    for i in tqdm(range(20)):
        # print(f'convert to cube for {years[i]}')
        
        data_month_y=torch.load(f"/data2/hzy/albedo2/albedo_information/result_{years[i]}_{month}_all_{landcover_type}.pth")
        for lat in range(90,-90,-1):
            for lon in range(-180,180):
                d=data_month_y[str(lat)+','+str(lon)]
                if d[0]==1:#perfact grid
                    qa[i,90-lat,lon+180]=1
                    area[:,i,90-lat,lon+180]=d[1]
                    data[:,:,i,90-lat,lon+180]=d[2]
                elif d[0]==2:#all ocean
                    qa[i,90-lat,lon+180]=2
                    area[:,i,90-lat,lon+180]=np.zeros((area.shape[0]))
                    area[-3,i,90-lat,lon+180]=d[1][0]
                    area[-2,i,90-lat,lon+180]=0
                    area[-1,i,90-lat,lon+180]=np.nan
                    data[:,:,i,90-lat,lon+180]=np.nan
                # elif d[0]==3:#too much nan value in land(>75%)
                #     qa[i,90-lat,lon+180]=3
                #     area[:,i,90-lat,lon+180]=d[1]
                #     data[:,:,i,90-lat,lon+180]=np.nan
                elif d[0]==4:#have snow nan
                    qa[i,90-lat,lon+180]=4
                    area[:,i,90-lat,lon+180]=np.nan
                    
                    area[-3:,i,90-lat,lon+180]=d[1]
                    data[:,:,i,90-lat,lon+180]=np.nan
    return data,area,qa

if __name__=='__main__':
    
    #Generate grid data of ALLUMs (albedo for different LULC types)
    for y in range(2001,2021):
        for m in [11]:
            print(f'------------------ {y}/{m} ------------------')
            for lat in tqdm(range(90,-90,-10)):
                cc=get_10lat_information(y,m,lat)
                cc.get_10lat_information()
                
                
    #Concate files of ALLUMs
    for landcover_type in ['LCCS1','LCCS2','LCCS3','IGBP']:
        for y in range(2001,2021):
            for m in tqdm(range(1,13),desc=f' {y} '):
                fs_y=glob(f'/data2/hzy/albedo2/albedo_information/result_{y}_{m}_*0_{landcover_type}.pth')
                if len(fs_y)!=18:
                    raise(f'ERROR {len(fs_y)}')
                k={}
                for i in range(len(fs_y)):
                    k.update(torch.load(fs_y[i]))
                torch.save(k,f'/data2/hzy/albedo2/albedo_information/result_{y}_{m}_all_{landcover_type}.pth')

    #Convert to nc files      
    for landcover_type in ['LCCS1','LCCS2','LCCS3','IGBP']:
        for month in range(1,13):
            convert_to_nc(month,landcover_type)
