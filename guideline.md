
## Guideline  
If you are attempting to reproduce this study, this document provides a step-by-step guideline for data processing and result analysis.

1. **Download Required Data**  
   Follow the instructions in `./preprocess/download_data.md` to download all the data required for this study. The downloads and preprocessing of **MCD43A3 C6.1 albedo data**, **MCD12Q1 C6.1 LULC data**, and **MOD10A1 C6.1 snow cover fraction data** are performed via Google Earth Engine. The document provides executable code script links for these tasks. Additionally, the code script link for calculating the contributions of **photosynthetic vegetation (PV)**, **non-photosynthetic vegetation (NPV)**, and **surface water content (SWC)** changes to albedo change is also included in `download_data.md`.  

2. **Preprocess the Data**  
   The downloaded data needs further preprocessing, including format conversion, missing value imputation, and resampling. The `./preprocess` folder provides scripts for preprocessing different datasets. Use these scripts sequentially to preprocess the data.  

3. **Generate the ALLUMs Dataset**  
   The `./grdis_data` folder contains all the scripts required to construct the ALLUMs dataset.  
   - First, run `./grdis_data/albedo.py` to calculate the albedo data for the ALLUMs dataset.  
   - Next, execute `./grdis_data/ConfusionMatrix.py` to calculate the area of different LULC conversion pathways within the ALLUMs dataset.  
   - After these steps, use `./grdis_data/weights.ipynb` to compute the anisotropy weights for filling in missing albedo data in the ALLUMs dataset.  
   - Finally, run `./grdis_data/TS_IDW_4d.ipynb` to fill in the missing albedo data.  

   Alternatively, you can directly download the precomputed ALLUMs dataset from Zenodo: [https://zenodo.org/records/10.5281/zenodo.13981586](https://zenodo.org/records/10.5281/zenodo.13981586).  
4. **Data Analysis**  
   Before exporting the analysis results, you need to perform some statistical calculations, such as:  
   - Computing global mean land surface albedo changes  
   - Analyzing the contributions of different factors  
   - Calculating albedo change trends and their significance  

   These calculations can be executed using `./anasys/stat.ipynb`. Other scripts in the `./anasys/` folder are used for exporting analysis results and generating relevant figures:  
   - **`./anasys/albedo.ipynb`**: Outputs results related to albedo.  
   - **`./anasys/RF.ipynb`**: Outputs results related to radiative forcing.  
   - **`./anasys/area.ipynb`**: Outputs results on LULC (land use/land cover) changes.  
   - **`./anasys/PV_NPV_SWC.ipynb`**: Outputs results on the contributions of **PV**, **NPV**, and **SWC** changes to albedo change. 
5. **Result Validation**  
   The result validation is divided into five parts, and the required scripts are included in the `./val` folder:

   - **`./val/DIW.py`**: Validates the accuracy of the four-dimensional spatio-temporal Inverse Distance Weighted (IDW) interpolation model used to fill in missing data for ALLUMs.  
   - **`./val/rebuilding500.py`**: Reconstructs the global albedo map at 500m resolution.  
   - **`./val/500rebuilding_val.ipynb`**: Outputs the validation results by comparing the reconstructed 500m albedo map with MCD43A3 data.  
   - **`./val/regression_val.ipynb`**: Outputs the accuracy of the PV-NPV-SWC albedo model.  
   - **`./val/calibration.ipynb`**: Outputs the validation of MODIS calibration errors based on desert calibration sites and provides robustness test results related to calibration error limits.  
   - **`./val/GLASS.ipynb`**: Includes the processing of GLASS albedo production data and provides all code for comparing and validating it against ALLUMs.  

