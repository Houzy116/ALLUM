import sys
sys.path.append('..')
from tool import *
warnings.filterwarnings("ignore")

def get_snow_cover(f):
    nf=nc.Dataset(f)
    k=np.flip(nf.variables['scfv'][:][0],axis=0)
    k2=np.flip(nf.variables['scfv_unc'][:][0],axis=0)
    k[k2==210]=0#water
    k[k2==215]=100#Permanent_Snow_and_Ice
    return k

def monthly_snow_cover(y,m,save=False):
    fs=glob(root_path+f'snow/Snow/{str(y).zfill(4)+str(m).zfill(2)}*.nc')
    sum=np.zeros((18000, 36000)).astype(np.float32)
    count=np.zeros((18000, 36000)).astype(np.float32)
    for i in tqdm(range(len(fs)),desc=f' {y}-{m} '):
        k=get_snow_cover(fs[i])
        k2=k.copy()
        k2[k2>100]=0
        sum+=k2
        count+=(k<=100).astype(np.int8)
    sum=sum.astype(np.float32)
    sum[count==0]=np.nan
    kk=sum/count
    if save:
        np.save(root_path+f'snow/snow_monthly/{str(y).zfill(4)+str(m).zfill(2)}.npy',kk)
    return kk


def fill_snow_cover(y,m,save=False):
    adds=[]
    for i in range(1,21):
        add=[y-i,y+i]
        for j in range(2):
            if add[j]>2020 or add[j]<2001:
                add[j]=None
        if not np.array([add[0] is None,add[1] is None]).sum()==2:
            adds.append(add)
    img=np.load(root_path+f'snow/snow_monthly/{y}{str(m).zfill(2)}.npy')
    for add in tqdm(adds,desc=f' {y}-{m} '):
        add_pathes=[root_path+f'snow/snow_monthly/{i}{str(m).zfill(2)}.npy' for i in add if not i is None]
        add_imgs=[np.load(i) for i in add_pathes]
        if len(add_imgs)==1:
            add_img=add_imgs[0]
        else:
            add_img=np.nanmean(np.stack(add_imgs),axis=0)
        img[np.isnan(img)]=add_img[np.isnan(img)]
    if save:
        np.save(root_path+f'snow/snow_monthly/{y}{str(m).zfill(2)}_fill.npy',img)
    return img
def fill_snow_cover2(y,m):
    k=[[y,m-1],[y,m+1]]
    for j in range(2):
        if k[j][1]==0:
            k[j][1]=12
            k[j][0]=k[j][0]-1
        if k[j][1]==13:
            k[j][1]=1
            k[j][0]=k[j][0]+1
        if k[j][0] in [2000,2021]:
            k[j]=None
    code=[str(j[0])+str(j[1]).zfill(2) for j in k if j is not None ]
    k_pathes=[root_path+f'snow/snow_monthly/{c}_fill.npy' for c in code]
    print(k_pathes)
    z=str(y)+str(m).zfill(2)
    img=np.load(root_path+f'snow/snow_monthly/{z}_fill.npy')
    print((np.isnan(img))[24*100:-24*100].sum())
    img_interp=np.nanmean(np.stack([np.load(j) for j in k_pathes]),axis=0)
    
    img[np.isnan(img)]=img_interp[np.isnan(img)]
    print((np.isnan(img))[24*100:-24*100].sum())
    np.save(root_path+f'snow/snow_monthly/{z}_fill2.npy',img)
    
    
    
if __name__=='__main__':
    #月合成
    for y in range(2011,2021):
        for m in range(1,13):
            _=monthly_snow_cover(y,m,True)
    #临近年依次fill nodata
    for y in range(2001,2021):
        for m in range(1,13):
            fill_snow_cover(y,m,True)
    #临近月fill nodata
    for y in range(2001,2021):
        for m in range(1,13):
            fill_snow_cover2(y,m)
    #IDW填充+平滑+重采样
    from tool import *
    for y in range(2011,2021):
        for m in range(1,13):
            c=str(y)+str(m).zfill(2)
            pp=f'/data/hk/albedo/snow/snow_monthly/{c}_fill2.npy'
            # print(pp)
            img=np.load(pp)

            for j in range(1):
                img[np.isnan(img)]=255
                img=img.astype(np.uint8)
                need_interp=(img==255)[24*100:-24*100].sum()
                print(need_interp)
                if need_interp!=0:
                    # print(need_interp)
                    out = fill.fillnodata(img, (img!=255),max_search_distance=20,smoothing_iterations=1)
                    need_interp=(out==255)[24*100:-24*100].sum()
                    if need_interp!=0:
                        
                        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                        print(need_interp)
                else:
                    break
            resample(out,pp.replace('_fill2.npy','.tif'))