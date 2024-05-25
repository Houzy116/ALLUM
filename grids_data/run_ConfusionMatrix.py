from run_albedo import *
import sys
sys.path.append('..')
from tool import *
warnings.filterwarnings("ignore")

def get_10lat_confusionmatrix(y,m,lat_10):
    imgs1=get_10lat_information(2001,m,lat_10)
    imgs2=get_10lat_information(y,m,lat_10)
    imgs1.landcover=imgs1.get_img_10lat('landcover')
    imgs1.snow_fre=imgs1.get_img_10lat('snow_fre')
    imgs2.landcover=imgs2.get_img_10lat('landcover')
    imgs2.snow_fre=imgs2.get_img_10lat('snow_fre')
    albedo_data1=torch.load(root_path+f'information/result3_2001_{m}_{lat_10}.pth')
    albedo_data2=torch.load(root_path+f'information/result3_{y}_{m}_{lat_10}.pth')
    result={}
    for lat in range(lat_10,lat_10-10,-1):
        for lon in range(-180,180):
            result[str(lat)+','+str(lon)]=get_landchange_matrix(y,m,[lon,lat],albedo_data1,albedo_data2,imgs1,imgs2)
    torch.save(result,root_path+f'confuse_matrix/result2_{y}_{m}_{lat_10}.pth')
      
def get_landchange_matrix(y,m,coord,albedo_data1,albedo_data2,imgs1,imgs2):
    data1=list(simplify_ocean(coord,albedo_data1,imgs1))
    data2=list(simplify_ocean(coord,albedo_data2,imgs2))
    
    if (len(data1)+len(data2))<6:
        return [[data1[-1],data2[-1]],None,None]
    else:
        data1[2]=[0]+data1[2]
        data2[2]=[0]+data2[2]
        data1[2]+=[20]
        data2[2]+=[20]
        # if ((data1[0]==20) & (data2[0]!=20)).sum()>0:
        #     data1[2]+=[20]
        # if ((data1[0]!=20) & (data2[0]==20)).sum()>0:
        #     data2[2]+=[20]
        
        transfer_id=((np.array(data1[2]).reshape((1,-1))*100)+(np.array(data2[2]).reshape((-1,1)))).reshape(-1,1,1)

        # print(transfer_id)
        transfer_id=transfer_id.repeat(data1[1].shape[0],axis=1).repeat(data1[1].shape[1],axis=2)
        transfer=data1[0]*100+data2[0]
        transfer=np.expand_dims(transfer,0).repeat(transfer_id.shape[0],axis=0)
        # data1[1][transfer[0]==2020]=np.nan
        # data2[1][transfer[0]==2020]=np.nan
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

def simplify_ocean(coord,albedo_data,imgs):
    coord_str=str(coord[1])+','+str(coord[0])
    condition,area,_=albedo_data[coord_str]
    if condition!=4:#1 or 2
        landcover=imgs.get_img_fromallband(coord,'landcover').astype(np.int16)
        snow_fre=imgs.get_img_fromallband(coord,'snow_fre').astype(np.float32)/100
        snow_fre[landcover==20]=0
        snow_fre[landcover==15]=1
        if condition==1:#数据充足
            area=area[:17]
            use_landtype=[i for i in range(1,18) if area[i-1]>0]
        else:#全是海洋
            use_landtype=[]
        return landcover,snow_fre,use_landtype,condition
    else:
        return [condition]
    
def convert_to_matrix(year,month):
    f=torch.load(f'/data/hk/albedo/confuse_matrix/result2_{year}_{month}_all.pth')
    data=np.zeros((180,360,18,18)).astype(np.float32)
    for lat in tqdm(range(90,-90,-1)):
        for lon in range(-180,180):
            f_sub=f[str(lat)+','+str(lon)]
            if f_sub[0][0]!=1 or f_sub[0][1]!=1:
                output=np.full((18,18),np.nan)
            else:
                output=np.zeros((18,18)).astype(np.float32)
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
    for y in range(2001,2021):    
        for m in range(1,13):
            print(f'------------------ {y}/{m} ------------------')
            for lat_10 in tqdm(range(90,-90,-10)):
                get_10lat_confusionmatrix(y,m,lat_10)
    #合并
    for y in range(2002,2021):
        for m in range(1,13):
            all={}
            for p in glob(f'/data/hk/albedo/confuse_matrix/result2_{y}_{m}_*.pth'):
                all.update(torch.load(p))
            print(y,m,len(all.keys()))
            torch.save(all,f'/data/hk/albedo/confuse_matrix/result2_{y}_{m}_all.pth')
            
    #转换成矩阵
    for year in range(2002,2021):
        print(year)
        data_m=[]
        for month in range(1,13):
            # print(month)
            data_m.append(convert_to_matrix(year,month))
        data_m=np.stack(data_m)
        torch.save(data_m,f'/data/hk/albedo/confuse_matrix/confuse_matrix_{year}.pth')