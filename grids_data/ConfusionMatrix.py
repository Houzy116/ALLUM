from albedo import *
import sys
sys.path.append('..')
from tool import *
warnings.filterwarnings("ignore")

def get_10lat_confusionmatrix(y,m,lat_10,landcover_type):
    imgs1=get_10lat_information(2001,m,lat_10)
    imgs2=get_10lat_information(y,m,lat_10)
    imgs1.landcover=imgs1.get_img_10lat(landcover_type)
    imgs1.snow_fre=imgs1.get_img_10lat('snow_fre_'+landcover_type)
    imgs2.landcover=imgs2.get_img_10lat(landcover_type)
    imgs2.snow_fre=imgs2.get_img_10lat('snow_fre_'+landcover_type)
    result={}
    for lat in range(lat_10,lat_10-10,-1):
        for lon in range(-180,180):
            result[str(lat)+','+str(lon)]=get_landchange_matrix([lon,lat],imgs1,imgs2,landcover_type)
    torch.save(result,f'/data2/hzy/albedo2/confuse_matrix/result_{y}_{m}_{lat_10}_{landcover_type}.pth')
      
def get_landchange_matrix(coord,imgs1,imgs2,landcover_type):
    data1=list(simplify_ocean(coord,imgs1,landcover_type))
    data2=list(simplify_ocean(coord,imgs2,landcover_type))
    if (len(data1)+len(data2))<6:
        return [[data1[-1],data2[-1]],None,None]
    elif data1[-1]==2 & data2[-1]==2:
        all_area=get_pixelarea(coord,data1[1].shape[1],data1[1].shape[0]).sum()
        return [[2, 2],
        np.array([[all_area,     0.        ],
                [    0.        ,     0.        ]]),
        np.array([[   0, 2000],
                [  20, 2020]])]
    else:
        data1[2]=[0]+data1[2]
        data2[2]=[0]+data2[2]
        data1[2]+=[20]
        data2[2]+=[20]      
        transfer_id=((np.array(data1[2]).reshape((1,-1))*100)+(np.array(data2[2]).reshape((-1,1)))).reshape(-1,1,1)
        transfer_id=transfer_id.repeat(data1[1].shape[0],axis=1).repeat(data1[1].shape[1],axis=2)
        transfer=data1[0]*100+data2[0]
        transfer=np.expand_dims(transfer,0).repeat(transfer_id.shape[0],axis=0)
        max_snowfre=np.max(np.stack([data1[1],data2[1]]),axis=0)
        min_snowfre=np.min(np.stack([data1[1],data2[1]]),axis=0)
        dif_snowfre=data2[1]-data1[1]
        pixel_area=get_pixelarea(coord,data1[1].shape[1],data1[1].shape[0])
        transfer_masks=(transfer==transfer_id).astype(np.float32)
        transfer_masks[transfer_masks==0]=np.nan
        transfer_masks=transfer_masks*np.expand_dims(pixel_area,0)
        transfer_masks=transfer_masks*np.expand_dims((1-max_snowfre),0)
        transfer_area_matrix=np.nansum(transfer_masks.reshape((-1,pixel_area.shape[0]*pixel_area.shape[1])),axis=1).reshape(-1,(len(data1[2])))
        transfer_id_matrix=np.array(list(transfer_id[:,0,0])).reshape(-1,(len(data1[2])))
        snow_to_snow=np.nansum(min_snowfre*pixel_area)
        dif_snowfre_P=dif_snowfre.copy()
        dif_snowfre_N=dif_snowfre.copy()
        dif_snowfre_P[dif_snowfre_P<=0]=np.nan
        dif_snowfre_N[dif_snowfre_N>0]=np.nan

        #snowfre2>snowfre1
        transfer_id2=np.array(data2[2]).reshape(-1,1,1).repeat(data1[1].shape[0],axis=1).repeat(data1[1].shape[1],axis=2)
        transfer2=np.expand_dims(data2[0],axis=0).repeat(transfer_id2.shape[0],axis=0)
        transfer_masks2=(transfer2==transfer_id2).astype(np.float32)
        transfer_masks2[transfer_masks2==0]=np.nan
        transfer_masks2=transfer_masks2*np.expand_dims(pixel_area,0)
        transfer_masks2=transfer_masks2*np.expand_dims(-dif_snowfre_N,0)
        transfer_area_snowto=np.nansum(transfer_masks2.reshape((-1,pixel_area.shape[0]*pixel_area.shape[1])),axis=1).reshape((len(data2[2]),-1))

        #snowfre2<snowfre1
        transfer_id1=np.array(data1[2]).reshape(-1,1,1).repeat(data1[1].shape[0],axis=1).repeat(data1[1].shape[1],axis=2)
        transfer1=np.expand_dims(data1[0],axis=0).repeat(transfer_id1.shape[0],axis=0)
        transfer_masks1=(transfer1==transfer_id1).astype(np.float32)
        transfer_masks1[transfer_masks1==0]=np.nan
        transfer_masks1=transfer_masks1*np.expand_dims(pixel_area,0)
        transfer_masks1=transfer_masks1*np.expand_dims(dif_snowfre_P,0)
        transfer_area_tosnow=np.nansum(transfer_masks1.reshape((-1,pixel_area.shape[0]*pixel_area.shape[1])),axis=1).reshape((-1,len(data1[2])))

        transfer_area_matrix=np.concatenate([transfer_area_matrix,transfer_area_tosnow],axis=0)
        transfer_area_snowto=np.array(list(transfer_area_snowto.flatten())+[snow_to_snow]).reshape(-1,1)
        transfer_area_matrix=np.concatenate([transfer_area_matrix,transfer_area_snowto],axis=1)
        transfer_area_matrix=transfer_area_matrix[1:,1:]
    return [[data1[-1],data2[-1]],transfer_area_matrix,transfer_id_matrix]

def simplify_ocean(coord,imgs,landcover_type):
    snow_fre=imgs.get_img_fromallband(coord,'snow_fre','aaa').astype(np.float32)/1000
    if (snow_fre==3).sum()!=0:#>85
        condition=4
        return [condition]
    else:
        landcover=imgs.get_img_fromallband(coord,'landcover','aaa')
        landcover[landcover==100]=20
        landcover[landcover==0]=20
        landcover=landcover.astype(np.int16)
        if (landcover!=20).sum()==0:#all water
            condition=2#too much ocean
            use_landtype=[]
            return landcover,snow_fre,use_landtype,condition
        else:
            condition=1
            if landcover_type=='LCCS2':
                use_landtype=[i for i in range(1,11) if (landcover==i).sum()!=0]#LCCS2
            if landcover_type=='LCCS3':
                use_landtype=[i for i in range(1,10) if (landcover==i).sum()!=0]#LCCS3
            if landcover_type=='IGBP':
                use_landtype=[i for i in range(1,17) if (landcover==i).sum()!=0]#LCCS3
            if landcover_type=='LCCS1':
                use_landtype=[i for i in range(1,16) if (landcover==i).sum()!=0]#LCCS3
            return landcover,snow_fre,use_landtype,condition
        
def convert_to_matrix(year,month,landcover_type):
    if landcover_type=='LCCS2':
        lc_len=10
    if landcover_type=='LCCS3':
        lc_len=9
    if landcover_type=='IGBP':
        lc_len=16
    if landcover_type=='LCCS1':
        lc_len=15
    f=torch.load(f'/data2/hzy/albedo2/confuse_matrix/result_{year}_{month}_all_{landcover_type}.pth')
    data=np.zeros((180,360,lc_len+1,lc_len+1)).astype(np.float32)
    for lat in tqdm(range(90,-90,-1)):
        for lon in range(-180,180):
            f_sub=f[str(lat)+','+str(lon)]
            if f_sub[0][0]!=1 or f_sub[0][1]!=1:
                output=np.full((lc_len+1,lc_len+1),np.nan)
            else:
                output=np.zeros((lc_len+1,lc_len+1)).astype(np.float32)
                from_index=list((f_sub[2][0]/100).astype(np.int16)-1)[1:-1]
                to_index=list(f_sub[2][:,0]-1)[1:-1]
                for y in range(len(from_index)):
                    for x in range(len(to_index)):
                        output[from_index[y],to_index[x]]=f_sub[1][x,y]
                for y in range(len(from_index)):
                    output[from_index[y],-1]=f_sub[1][-1,y]
                for x in range(len(to_index)):
                    output[-1,to_index[x]]=f_sub[1][x,-1]
                output[-1,-1]=f_sub[1][-1,-1]   
            data[90-lat,lon+180]=output
    return data   
 
if __name__=='__main__':
    
    #Generate grid data of ALLUMs (areas for different LULC conversion and non-converion regions)
    for y in range(2002,2021):  
        for m in range(1,13):
            print(f'------------------ {y}/{m} ------------------')
            for lat_10 in tqdm(range(90,-90,-10)):
                for landcover_type in ['LCCS1','LCCS2','LCCS3','IGBP']:
                    get_10lat_confusionmatrix(y,m,lat_10,landcover_type)

    #Concate files of ALLUMs
    for landcover_type in ['LCCS1','LCCS2','LCCS3','IGBP']:
        for y in range(2002,2021):
            for m in range(1,13):
                all={}
                for p in glob(f'/data2/hzy/albedo2/confuse_matrix/result_{y}_{m}_*_{landcover_type}.pth'):
                    all.update(torch.load(p))
                print(y,m,len(all.keys()))
                torch.save(all,f'/data2/hzy/albedo2/confuse_matrix/result_{y}_{m}_{landcover_type}_all.pth')
            
    #Convert to LULC confuse matrix
    for landcover_type in ['LCCS1','LCCS2','LCCS3','IGBP']:
        for year in range(2002,2021):
            print(year)
            data_m=[]
            for month in range(1,13):
                # print(month)
                data_m.append(convert_to_matrix(year,month,'landcover_type'))
            data_m=np.stack(data_m)
            torch.save(data_m,f'/data2/hzy/albedo2/confuse_matrix/confuse_matrix_{year}_{landcover_type}.pth')
