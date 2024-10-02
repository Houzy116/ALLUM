import sys
sys.path.append('..')
from tool import *
warnings.filterwarnings("ignore")

#Resample using Bilinear
def resample(img,width=360,height=180,resample_alg = gdalconst.GRIORA_Bilinear):
    driver = gdal.GetDriverByName('MEM')
    src_ds = driver.Create('', img.shape[1],img.shape[0], 1, 6)
    src_ds.GetRasterBand(1).WriteArray(img)
    src_ds.GetRasterBand(1).SetNoDataValue(1000000)
    src_ds.FlushCache()
    data = src_ds.GetRasterBand(1).ReadAsArray(buf_xsize=width,buf_ysize=height,resample_alg = resample_alg)
    return data

#Longitude coordinate system transformation.
def trf_lon(img):
    img2=np.zeros(img.shape)
    img2[:,:180]=img[:,180:]
    img2[:,180:]=img[:,:180]
    return img2

if __name__=='__main__':
    
    #Standardize the data format and resample.

    # ERAI: Huang, Y., Xia, Y., and Tan, X.: On the pattern of CO2 radiative forcing and poleward energy transport, J. Geophys. Res.-Atmos., 122, 10578–10593, 2017. 
    # ERA5: Huang, H. and Huang, Y.: Radiative sensitivity quantified by a new set of radiation flux kernels based on the ECMWF Reanalysis v5 (ERA5), Earth Syst. Sci. Data, 15, 3001–3021, https://doi.org/10.5194/essd-15-3001-2023, 2023.
    # CAM3: Shell, K. M., Kiehl, J. T. & Shields, C. A. Using the radiative kernel technique to calculate climate feedbacks in NCAR’s community atmospheric model. J. Climate 21, 2269–2282 (2008).
    # CAM5: Pendergrass, A. G., Conley, A. & Vitt, F. M. Surface and top-of-atmosphere radiative feedback kernels for CESM-CAM5. Earth Syst. Sci. Data 10, 317–324 (2018).
    # HadGEM2: Smith, C. J. et al. Understanding rapid adjustments to diverse forcing agents. Geophys. Res. Lett. 45, 12,023–12,031 (2018).
    # HadGEM3: Smith, C. J., Kramer, R. J. & Sima, A. The HadGEM3-GA7.1 radiative kernel: the importance of a well-resolved stratosphere. Earth Syst. Sci. Data 12, 2157–2168 (2020).
    # ECHAM6: Block, K. & Mauritsen, T. Forcing and feedback in the MPI‐ESM‐LR coupled model under abruptly quadrupled CO2. J. Adv. Model Earth Sy. 5, 676–691 (2013).
    
    names=['HadGEM2', 'HadGEM3', 'CAM3', 'CAM5', 'ECHAM6','ERAI','ERA5']
    v_names=['albedo_sw','albedo_sw','alb_all','FSNT','A_srad0','alb_toa_all','TOA_all']
    for i in range(7):
        name=names[i]
        NC=nc.Dataset(f"/data2/hzy/albedo2/kernel/{name}.nc",'r')
        print(NC.variables[v_names[i]])
        data=[]
        for m in range(1,13):
            
            img=NC.variables[v_names[i]][m-1]*(100)
            if i==2:
                img[0]=img[1]
                img[-1]=img[-2]
            if i in [0,1,3]:
                img=np.flip(img,axis=0)
            img=resample(img)
            img=trf_lon(img)
            data.append(img)
        data=np.stack(data)
        torch.save(data,f"/data2/hzy/albedo2/kernel/{name}.pth")
    
    #Calculate the maximum, minimum, and mean values, and store all the data in a single file.
    datas=[]
    output={}
    for i in range(7):
        datas.append(torch.load(f"/data2/hzy/albedo2/kernel/{names[i]}.pth"))
        output[names[i]]=torch.load(f"/data2/hzy/albedo2/kernel/{names[i]}.pth")
    datas=np.stack(datas)
    max=np.max(datas,axis=0)
    min=np.min(datas,axis=0)
    mean=np.mean(datas,axis=0)
    median=np.median(datas,axis=0)
    output['min']=min
    output['max']=max
    output['mean']=mean
    output['median']=median
    torch.save(output,f"/data2/hzy/albedo2/kernel/kernels.pth")