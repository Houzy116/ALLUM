import sys
sys.path.append('..')
from tool import *
def fill_albedo2(month,type,sza_L,landcover_type,interp_2000=False):
    max_distance=3
    if type=='snow':
        distence_w_dim=torch.load(f'/data2/hzy/albedo2/data/weight_LCCS2.pth')[type]
    else:
        distence_w_dim=torch.load(f'/data2/hzy/albedo2/data/weight_{landcover_type}.pth')[type]
    sza_L_month=sza_L[0,month-1]
    bands=['albedo_BSA_vis', 
        'albedo_WSA_vis', 
        'albedo_BSA_nir', 
        'albedo_WSA_nir', 
        'albedo_BSA_shortwave', 
        'albedo_WSA_shortwave']
    m_index=[(month-2)%12+1,(month-1)%12+1,(month)%12+1]
    offset=[1 if m_index[j]-m_index[0]>=0 else -1 for j in range(len(m_index))]
    if np.array(offset).sum()==len(offset):
        offset=[0]*len(m_index)
    vs=[]
    ms=[]
    land_proportions=[]
    for i in range(len(offset)):    
        NC=nc.Dataset(f'/data2/hzy/albedo2/albedo_information_nc2/LCCS2_snow/month_{m_index[i]}_{landcover_type}.nc','r')
        for band in bands:
            vs.append(NC.variables[f'{band}-{type}'][:])
        if type!='ocean':
            ms.append(NC.variables[f'area-{type}'][:])
        # qa=NC.variables['qa'][:]
        area_grid=NC.variables['area-grid'][:]
        area_land=NC.variables['area-land'][:]
        land_proportions.append(area_land/area_grid)
        NC.close()
    offset_bands=[]
    for i in offset:
        offset_bands+=[i]*len(bands)
    if offset[0]!=0:
        nan_array=np.zeros((1,180,360))*np.nan
        vs=[np.concatenate((nan_array,vs[i]),axis=0) if offset_bands[i]==1 else np.concatenate((vs[i],nan_array),axis=0) for i in range(len(offset_bands))]
        land_proportions=[np.concatenate((nan_array,land_proportions[i]),axis=0) if offset[i]==1 else np.concatenate((land_proportions[i],nan_array),axis=0) for i in range(len(offset))]
        if type!='ocean':
            ms=[np.concatenate((nan_array,ms[i]),axis=0) if offset[i]==1 else np.concatenate((ms[i],nan_array),axis=0) for i in range(len(offset))]
        else:
            ms=land_proportions  
    else:
        if type=='ocean':
            ms=land_proportions  
    M,H,W=vs[0].shape
    v=np.stack(vs).reshape((len(offset),len(bands),M,H,W)).transpose((1,0,2,3,4))#(6, 3, 21, 180, 360)
    m=np.stack(ms)
    all_indices = np.indices(v[0].shape).reshape(4, -1).T
    if interp_2000:
        if type!='ocean':
                interp_indices = all_indices[((v[0]==2000) & (m>0)).reshape(-1)]
        else:
            interp_indices = all_indices[((v[0]==2000) & (m<1) & (m>0)).reshape(-1)]
        nonan_indices=all_indices[((~np.isnan(v[0])).reshape(-1)) & ((v[0]!=2000).reshape(-1))]
    else:
    
        if type!='ocean':
            interp_indices = all_indices[(np.isnan(v[0]) & (m>0)).reshape(-1)]
        else:
            interp_indices = all_indices[(np.isnan(v[0]) & (m<1) & (m>0)).reshape(-1)]
        nonan_indices=all_indices[(~np.isnan(v[0])).reshape(-1)]
    interp_indices=[j for j in interp_indices if j[0]==int(len(offset)/2)]  
    interp_indices=[j for j in interp_indices if sza_L_month[j[2],j[3]]<2] 
    nonan_indices2=[j for j in nonan_indices if j[0]==int(len(offset)/2)]  
    nonan_indices2=[j for j in nonan_indices2 if sza_L_month[j[2],j[3]]<2]
    str_output=f'{month}-{type}  '+str(len(interp_indices))+'  '+str(round(len(interp_indices)/(len(nonan_indices2)+len(interp_indices)+0.01)*100,2))+'%'
    time.sleep(1)
    kk=[search_nonan2(v,interp_indice,max_distance=max_distance,distence_w_dim=distence_w_dim,interp_2000=interp_2000) for interp_indice in tqdm(interp_indices,desc=str_output)]
    interp_values=np.array([a[0] for a in kk])
    dws=np.array([a[1] for a in kk])
    dws_v=np.zeros(v.shape)
    for z in range(len(interp_indices)):
        v[:,interp_indices[z][0],interp_indices[z][1],interp_indices[z][2],interp_indices[z][3]]=np.array(interp_values[z])
        dws_v[:,interp_indices[z][0],interp_indices[z][1],interp_indices[z][2],interp_indices[z][3]]=np.array(dws[z])
    if offset[int(len(offset)/2)]>=0:
        return v[:,int(len(offset)/2),offset[int(len(offset)/2)]:],dws_v[:,int(len(offset)/2),offset[int(len(offset)/2)]:]
    else:
        return v[:,int(len(offset)/2),:-1],dws_v[:,int(len(offset)/2),:-1]
def TS_IDW_4d_2(v,interp_indice,max_distance,distence_w_dim,interp_2000):
        value=v[zero(interp_indice[0]-int(max_distance/distence_w_dim[0])):interp_indice[0]+int(max_distance/distence_w_dim[0])+1,
                zero(interp_indice[1]-int(max_distance/distence_w_dim[1])):interp_indice[1]+int(max_distance/distence_w_dim[1])+1,
                zero(interp_indice[2]-int(max_distance/distence_w_dim[2])):interp_indice[2]+int(max_distance/distence_w_dim[2])+1,
                zero(interp_indice[3]-int(max_distance/distence_w_dim[3])):interp_indice[3]+int(max_distance/distence_w_dim[3])+1]
        non_nan_indices = np.where(~np.isnan(value))
        if interp_2000:
            v[v==2000]=np.nan
        non_nan_points = np.array(list(zip(non_nan_indices[0], non_nan_indices[1], non_nan_indices[2],non_nan_indices[3])))
        if len(non_nan_points)==0:
                interpolated_value=np.nan
                dws=np.nan
        else:
                k=np.array([zero(-interp_indice[0]+int(max_distance/distence_w_dim[0]))-int(max_distance/distence_w_dim[0]),
                        zero(-interp_indice[1]+int(max_distance/distence_w_dim[1]))-int(max_distance/distence_w_dim[1]),
                        zero(-interp_indice[2]+int(max_distance/distence_w_dim[2]))-int(max_distance/distence_w_dim[2]),
                        zero(-interp_indice[3]+int(max_distance/distence_w_dim[3]))-int(max_distance/distence_w_dim[3])])
                non_nan_points+=k
                distance_s = np.array([np.linalg.norm(np.array(non_nan_points[i])*np.array(distence_w_dim)) for i in range(len(non_nan_points))])
                if len(distance_s)<10:
                        interpolated_value=np.nan
                        dws=np.nan
                else:
                        kth = np.partition(distance_s, 9)[9]
                        distance=distance_s.copy()
                        distance1=distance_s.copy()
                        distance1=distance1[distance_s<=kth]
                        weights = 1.0 / distance1**2
                        dws=np.sum(weights)
                        weights /= dws
                        non_nan_values = value[non_nan_indices]
                        non_nan_values=non_nan_values[distance<=kth]
                        interpolated_value = np.sum(weights * non_nan_values)
        return interpolated_value,dws
    
def search_nonan2(v,interp_indice,max_distance,distence_w_dim,interp_2000):
    k_bands=[]
    dws_bands=[]
    search_n=1
    for i in range(6):
        k,dws=TS_IDW_4d_2(v[i],interp_indice,max_distance=max_distance*search_n,distence_w_dim=distence_w_dim[i],interp_2000=interp_2000)
        while np.isnan(k):
            search_n*=2
            k,dws=TS_IDW_4d_2(v[i],interp_indice,max_distance=max_distance*search_n,distence_w_dim=distence_w_dim[i],interp_2000=interp_2000)
        k_bands.append(k)
        dws_bands.append(dws)
    return k_bands,dws_bands

def zero(k):
    if k<0:
        return 0
    else:
        return k
    
if __name__ == '__main__':
    #fill albedo
    for landcover_type in ['LCCS1','LCCS2','LCCS3','IGBP']:
        if landcover_type=='LCCS2':
            lc_len=10
        elif landcover_type=='LCCS3':
            lc_len=9
        elif landcover_type=='IGBP':
            lc_len=16
        elif landcover_type=='LCCS1':
            lc_len=15 
        bands=['albedo_BSA_vis', 
        'albedo_WSA_vis', 
        'albedo_BSA_nir', 
        'albedo_WSA_nir', 
        'albedo_BSA_shortwave',
        'albedo_WSA_shortwave']
        _,sza_L=torch.load(root_path+'sza_and_szaL.pth')
        landtypes=['landtype'+str(i) for i in range(1,lc_len+1)]+['snow']
        for month in range(1,13):
            print(landcover_type)
            NC=nc.Dataset(f'/data2/hzy/albedo2/albedo_information_nc2/fill/month_{month}_{landcover_type}.nc','r+')
            for type in landtypes:
                if type=='landtype2':
                    continue
                v_fill,dws=fill_albedo2(month,type,sza_L,landcover_type)
                for band_i in range(len(bands)):
                    NC.variables[f'{bands[band_i]}-{type}'][:]=v_fill[band_i]
                    try:
                        NC.createVariable(f'DWS-{type}-{bands[band_i]}', 'f4', ("time", "latitude", "longitude"))
                    except:
                        pass
                    NC.variables[f'DWS-{type}-{bands[band_i]}'][:]=dws[band_i]
            NC.close()