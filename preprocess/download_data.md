## Data download

For **MCD43A3 C6.1 albedo data**, **MCD12Q1 C6.1 LULC data**, and **MOD10A1 C6.1 snow cover fraction data**, some preprocessing and downloads were performed via Google Earth Engine. Below are the code script links required for downloading and preprocessing these datasets:

- **MCD43A3 C6.1 albedo data**: [https://code.earthengine.google.com/93f02b37610af47e8d340c1ae7060735](https://code.earthengine.google.com/93f02b37610af47e8d340c1ae7060735)  
- **MCD12Q1 C6.1 LULC data**: [https://code.earthengine.google.com/6969e1242cf3a008dec62de623de9928](https://code.earthengine.google.com/6969e1242cf3a008dec62de623de9928)  
- **MOD10A1 C6.1 snow cover fraction data**: [https://code.earthengine.google.com/261319c795656b2a014962c520948da4](https://code.earthengine.google.com/261319c795656b2a014962c520948da4)  


The contributions of **photosynthetic vegetation (PV)**, **non-photosynthetic vegetation (NPV)**, and **surface water content (SWC)** changes to albedo change were calculated using Google Earth Engine. Below is the link to the code script for calculating and downloading intermediate results:  [https://code.earthengine.google.com/40f2ec8176a6ee2ee09ecbb3e118f230](https://code.earthengine.google.com/40f2ec8176a6ee2ee09ecbb3e118f230)  

Additional datasets can be directly downloaded using the following links:  
- **The diffuse and direct solar surface radiation fluxes**: [https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html)  
- **The in-situ albedo observations from AWS sites**: [https://doi.org/10.22008/FK2/IW73UU](https://doi.org/10.22008/FK2/IW73UU)  
- **GLASS02B2 albedo product**: [https://www.glass.hku.hk/download.html](https://www.glass.hku.hk/download.html)  

The methods for downloading radiative kernel data can be found in the following references:  

- **ERAI**:  
  Huang, Y., Xia, Y., and Tan, X.: On the pattern of CO2 radiative forcing and poleward energy transport, *J. Geophys. Res.-Atmos.*, **122**, 10578–10593, 2017.  

- **ERA5**:  
  Huang, H. and Huang, Y.: Radiative sensitivity quantified by a new set of radiation flux kernels based on the ECMWF Reanalysis v5 (ERA5), *Earth Syst. Sci. Data*, **15**, 3001–3021, 2023.  

- **CAM5**:  
  Pendergrass, A. G., Conley, A., and Vitt, F. M.: Surface and top-of-atmosphere radiative feedback kernels for CESM-CAM5, *Earth Syst. Sci. Data*, **10**, 317–324, 2018.  

- **HadGEM2**:  
  Smith, C. J. et al.: Understanding rapid adjustments to diverse forcing agents, *Geophys. Res. Lett.*, **45**, 12,023–12,031, 2018.  

- **HadGEM3**:  
  Smith, C. J., Kramer, R. J., and Sima, A.: The HadGEM3-GA7.1 radiative kernel: The importance of a well-resolved stratosphere, *Earth Syst. Sci. Data*, **12**, 2157–2168, 2020.  

- **ECHAM6**:  
  Block, K. and Mauritsen, T.: Forcing and feedback in the MPI‐ESM‐LR coupled model under abruptly quadrupled CO2, *J. Adv. Model Earth Sy.*, **5**, 676–691, 2013.  
```

