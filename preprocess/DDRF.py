import sys
sys.path.append('..')
from tool import *
warnings.filterwarnings("ignore")

#Read files
def get_wb_fraction_data(type):
    NC=nc.Dataset(root_path+f'white_sky_fraction/{type}sf.sfc.mon.mean.nc')
    t_str = '1800-01-01 00:00:00'
    d = datetime.datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')
    t=[d+datetime.timedelta(hours=i) for i in NC.variables['time'][:]][636:876]
    lat=NC.variables['lat'][:]
    lon=NC.variables['lon'][:]
    data=NC.variables[f'{type}sf'][636:876]
    return [data,t,lat,lon]

#Longitude coordinate system transformation.
def trf_lon(img):
    img2=np.zeros(img.shape)
    img2[:,:180]=img[:,180:]
    img2[:,180:]=img[:,:180]
    return img2

if __name__=='__main__':
    
    #Shortwave white_sky_fraction
    black_nir=get_wb_fraction_data('nbd')
    white_nir=get_wb_fraction_data('ndd')
    black_vis=get_wb_fraction_data('vbd')
    white_vis=get_wb_fraction_data('vdd')
    f_white=(white_nir[0]+white_vis[0])/(white_nir[0]+black_nir[0]+white_vis[0]+black_vis[0])
    f_white_data=f_white.data
    f_white_mask=f_white.mask
    f_white_data[f_white_mask==1]=2
    f_white_data[f_white_mask==1]=2
    ds=[trf_lon(resample_output(f_white_data[i])) for i in range(240)]
    f_white_resample=np.stack(ds)
    f_white_resample[f_white_resample==2]=np.nan
    torch.save(f_white_resample,root_path+'white_sky_fraction/white_sky_fraction.pth')

    #Nir white_sky_fraction
    fn=white_nir[0]/(white_nir[0]+black_nir[0])
    fn_white_data=fn.data
    fn_white_mask=fn.mask
    fn_white_data[fn_white_mask==1]=2
    fn_white_data[fn_white_mask==1]=2
    ds=[trf_lon(resample_output(fn_white_data[i])) for i in range(240)]
    fn_white_resample=np.stack(ds)
    fn_white_resample[fn_white_resample==2]=np.nan
    torch.save(fn_white_resample,root_path+'white_sky_fraction/white_sky_fraction_nir.pth')

    #Vis white_sky_fraction
    fv=white_vis[0]/(white_vis[0]+black_vis[0])
    fv_white_data=fv.data
    fv_white_mask=fv.mask
    fv_white_data[fv_white_mask==1]=2
    fv_white_data[fv_white_mask==1]=2
    ds=[trf_lon(resample_output(fv_white_data[i])) for i in range(240)]
    fv_white_resample=np.stack(ds)
    fv_white_resample[fv_white_resample==2]=np.nan
    torch.save(fv_white_resample,root_path+'white_sky_fraction/white_sky_fraction_vis.pth')