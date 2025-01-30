
## Guideline  
If you attempt to reproduce this study, this document provides a step-by-step guideline for data processing and result analysis.

1. **Download the Data**  
   Follow the instructions in **`./preprocess/download_data.md`** to download all the required data. 
   The acquisition and preprocessing of **MCD43A3 C6.1 albedo data**, **MCD12Q1 C6.1 LULC data**, and **MOD10A1 C6.1 snow cover fraction data** are performed via Google Earth Engine. The document provides executable code script links for these tasks. Additionally, the code script link for calculating the contributions of **photosynthetic vegetation (PV)**, **non-photosynthetic vegetation (NPV)**, and **surface water content (SWC)** changes to surface albedo change is included in **`download_data.md`**.  

2. **Preprocess the Data**  
   The obtained data needs further preprocessing, including format conversion, missing value interpolation, and resampling. The **`./preprocess`** folder provides scripts for preprocessing different datasets. Use these scripts in sequence to preprocess the data.  

3. **Generate the ALLUMs Dataset**  
   The **`./grdis_data`** folder contains all the scripts required to construct the ALLUMs dataset.  
   - First, run **`./grids_data/albedo.py`** to calculate the albedo data for the ALLUMs dataset.  
   - Next, execute **`./grids_data/ConfusionMatrix.py`** to calculate the area of different LULC conversion pathways within the ALLUMs dataset.  
   - After the above steps, use **`./grids_data/weights.ipynb`** to compute the anisotropy weights to interpolate the missing albedo data in the ALLUMs dataset.  
   - Finally, run **`./grids_data/TS_IDW_4d.ipynb`** to fill in the missing albedo data.  

   Alternatively, you can directly download the precomputed ALLUMs dataset from Zenodo: [https://zenodo.org/records/10.5281/zenodo.13981586](https://zenodo.org/records/10.5281/zenodo.13981586).  
4. **Data Analysis**  
   The following statistical calculations are needed before the analysis results are obtained.  
   - Computing global mean land surface albedo changes.  
   - Analyzing the contributions of different factors.  
   - Calculating albedo change trends and their significance.  

   These calculations can be executed using **`./anasys/stat.ipynb`**. Other scripts in the **`./anasys/`** folder are used to export analysis results and generate relevant figures:  
   - **`./anasys/albedo.ipynb`**: Output the albedo changed.  
   - **`./anasys/RF.ipynb`**: Output the albedo-induced radiative forcing.  
   - **`./anasys/area.ipynb`**: Output the results of LULC (land use/land cover) changes.  
   - **`./anasys/PV_NPV_SWC.ipynb`**: Output the contributions of **PV**, **NPV**, and **SWC** changes to albedo changes. 
5. **Result Validation**  
   The result validation is divided into five parts, and the required scripts are included in the **`./val`** folder:

   - **`./val/DIW.py`**: Validate the accuracy of the four-dimensional spatio-temporal Inverse Distance Weighted (IDW) interpolation model used to interpolated the missing data for ALLUMs.  
   - **`./val/rebuilding500.py`**: Reconstruct the global albedo map at 500m resolution.  
   - **`./val/500rebuilding_val.ipynb`**: Output the validation results by comparing the reconstructed 500m albedo map with MCD43A3 data.  
   - **`./val/regression_val.ipynb`**: Output the accuracy of the PV-NPV-SWC albedo model.  
   - **`./val/calibration.ipynb`**: Output the validation of MODIS calibration errors based on desert calibration sites and provides robustness tests related to calibration error limits.  
   - **`./val/GLASS.ipynb`**: Include the processing of GLASS albedo production data, and provide the codes for comparing GLASS albedo results with ALLUMs.  

