#!/usr/bin/env python
# coding: utf-8
# Written by Melis Celik based on what we learned from Rafael's imaging analysis.

# In[ ]:


#get_ipython().system('pip show nd2')


# In[ ]:


#pip install nd2reader


# In[56]:


import nd2reader #to read nd2 files which is a file format from Nikon microscopes.
import nd2
import os #to interact with the operating systems
import numpy as np # a liibrary for numerical computing in Python
import pandas as pd #  pandas is a data manipulation and analysis library for providing data like frames
import matplotlib.pyplot as plt # plotting library
from matplotlib.table import Table # creating tables
import seaborn as sns # a statistical data visualization library based on matplotlib
from skimage import io, measure, morphology, filters # a library used for image processing
from skimage.color import rgb2gray
from skimage.measure import label, regionprops_table
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import clear_border
from skimage.morphology import area_opening
from scipy.stats import ttest_ind # to perform statistics 
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as nd_label
from nd2reader import ND2Reader
import plotly.graph_objects as go
from tkinter import Tk, filedialog #Pythons standard library for creating graphical user interfaces, I think it is very convenient.


# In[57]:


# Here we are asking user for folder path by using tkinter package
def get_folder_path():
    Tk().withdraw()  
    folder_path = filedialog.askdirectory(title="Select Folder with ND2 Files")
    return folder_path


# In[58]:


# Since we are working with  ND2 files, we need to load and extract channels. There are 4 channels in these images: RED, GREEN, BLUE, FAR-RED
def load_nd2_file(file_path):
    with nd2.ND2File(file_path) as figure:
        image = figure.asarray()  
        print(f"File path: {figure.path}")
        print(f"File shape: {figure.shape}")
        return image


# In[59]:


# Normalize channel images
def normalize_channel(image):
    normalized_image = (image * (1000 / image.max())).astype(np.uint16)
    return normalized_image

# To keep track segmentation, thresholding processes we need to plot and visualize image at different stages, in that way we can be sure that the code and set parameters are working for all images. This is a kind of internal control and critical since sometimes set threshold and segmentation processes can not be fit for all images.
def visualize_image(image, title, cmap='gray'):
    plt.figure(figsize=(4, 4), dpi=150)
    plt.imshow(image, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()


# In[60]:


# Here we process each channel for segmentation 
def process_channel(channel, channel_name):
    # 1) Visualize Original Channel
    visualize_image(channel, f"Original Image - {channel_name}")

    # If the channel is MAP2, apply Otsu thresholding and calculate the area since when I remove borders, I have lost majority of MAP2 signal. MAP2 is located at dentrites in neurons. Other markers (KI67, MAP2 and DAPI) are located in nuclei. That is why for KI67, MAP2 and DAPI, the code will designed to count segmented object while for MAP2 which is located cell body is measured based on area.
    if channel_name == 'MAP2':
        # 2) Background subtraction for MAP2
        img_bg = gaussian(channel, sigma=50, preserve_range=True)
        img_no_bg = channel - img_bg
        visualize_image(img_no_bg, f"Background Corrected - {channel_name}")

        # 3) Apply Otsu thresholding to segment the MAP2 area
        th_val = threshold_otsu(img_no_bg)
        bw_otsu = img_no_bg > th_val
        visualize_image(bw_otsu, f"Thresholding (Otsu) - {channel_name}")

        # 4) Directly calculate the total area from the thresholded regions (without border removal) since I do not want to lose the majority of signal.
        labeled_map, num_features = label(bw_otsu, return_num=True)
        visualize_image(labeled_map, f"Connected Components (MAP2) - {channel_name}", cmap='nipy_spectral')

        # 5) Calculate region properties which is important for comparison and statistics
        properties = ['label', 'area', 'intensity_mean']
        table = regionprops_table(label_image=labeled_map, intensity_image=channel, properties=properties)
        df = pd.DataFrame(table)

        return df, labeled_map  

    # For other channels which are SOX2, KI67, DAPI, here we follow the original steps (including border removal and area opening) based on Rafa's lecture
    else:
        # 1) Background subtraction
        img_bg = gaussian(channel, sigma=50, preserve_range=True)
        img_no_bg = channel - img_bg
        visualize_image(img_bg, f"Background - {channel_name}")
        visualize_image(img_no_bg, f"Background Corrected - {channel_name}")

        # 2) Thresholding (Otsu)
        th_val = threshold_otsu(img_no_bg)
        bw_otsu = img_no_bg > th_val
        visualize_image(bw_otsu, f"Thresholding (Otsu) - {channel_name}")

        # 3) Remove border touching segments and small objects (for channels other than MAP2)
        mask = clear_border(bw_otsu)
        mask = area_opening(mask, area_threshold=200)
        visualize_image(mask, f"After Border Removal & Area Opening - {channel_name}")

        # 4) Label connected components
        lbl = label(mask)
        visualize_image(lbl, f"Connected Component Analysis - {channel_name}", cmap='nipy_spectral')

        # 5) Region properties
        properties = ['label', 'area', 'eccentricity', 'intensity_mean']
        table = regionprops_table(label_image=lbl, intensity_image=channel, properties=properties)
        df = pd.DataFrame(table)
        
        return df, lbl  

## So far we set required parameters for image prosessing like background substraction, thresholding, border removal etc.

# In[61]:


# Main function to execute the comparison between SNP and WT images. In folder I provided 4 SNP and 4 WT image and the code was designed to group SNP and WT files seperately based on file name.
def main():
    # For the analysis, folder path should be given here, so that batch analysis can be performed.
    folder_path = get_folder_path()
    
    # Initialize summary list for all images
    summary_list = []

    # Counter for SNP and WT files in that way giving name will be easier. If file name contains SNP or WT, it will be recognized and to be enumerated.
    snp_count = 1
    wt_count = 1

    # Here the loop starts over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.nd2'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Analyzing {file_name}...") #This is for tracking whether the code is running for all images.

            # Load and process image
            image = load_nd2_file(file_path)

            # Each channels will be assigned based on their markers rather than wavelength of channel
            channels = [image[i].squeeze() for i in range(image.shape[0])]
            channel_names = ['DAPI', 'KI67', 'SOX2', 'MAP2']

            # Store results for each channel
            results = {}
            area_map2 = 0  # Variable to store MAP2 area since we are calculating area instead of object count

            for channel, name in zip(channels, channel_names):
                normalized_channel = normalize_channel(channel)
                df, lbl = process_channel(normalized_channel, name)

                if name == 'MAP2':
                    area_map2 = df['area'].sum()  # Sum area for MAP2
                    results[name] = df[['label', 'area']]  # Store only label and area for MAP2
                else:
                    results[name] = df[['label', 'area']]  # Store label and area for other channels

            # Here I tried to determine if the file belongs to SNP or WT and assign identifier to simplify and categorize different files
            if 'snp' in file_name.lower():
                label_name = f'SNP_{snp_count}'
                snp_count += 1
            else:
                label_name = f'WT_{wt_count}'
                wt_count += 1

            # Here I counted each object as cell and normalized values
            dapi_count = len(results['DAPI'])  # Get DAPI count for normalization
            summary_rows = []  # List to hold each summary row

            for channel in channel_names:
                if channel == 'MAP2': #however for MAP2, we are using area, not cell count since it is not possible to count MAP2 signal like nuclei 
                    cell_count = len(results[channel])
                    normalized_value = area_map2 / dapi_count if dapi_count > 0 else 0  # Normalize MAP2 area by DAPI count
                    summary_rows.append({'Image': label_name, 'Channel': channel, 'Cell Count': cell_count, 
                                         'Area': area_map2, 'Normalized (by DAPI)': normalized_value})
                else:
                    cell_count = len(results[channel])
                    normalized_value = cell_count / dapi_count if dapi_count > 0 else 0
                    summary_rows.append({'Image': label_name, 'Channel': channel, 'Cell Count': cell_count, 
                                         'Area': None, 'Normalized (by DAPI)': normalized_value})

            # Append the results for the current image to the summary list
            summary_list.extend(summary_rows)

    # Convert summary to DataFrame
    summary_df = pd.DataFrame(summary_list)

    # For statistical comparison of SNP and WT, p-values were calculated using t-test for each channel (excluding DAPI) since we are only comparing two groups.
    p_values = {}
    for channel in channel_names[1:]:  # Exclude DAPI since there is no variation in normalised DAPI signal, it is always 1.
        snp_group = summary_df[summary_df['Image'].str.startswith('SNP') & (summary_df['Channel'] == channel)]['Normalized (by DAPI)']
        wt_group = summary_df[summary_df['Image'].str.startswith('WT') & (summary_df['Channel'] == channel)]['Normalized (by DAPI)']

        # To be able to perform t-test, we should have minimum 2 files per condition here it will give error if there are not enough replicates
        if len(snp_group) > 1 and len(wt_group) > 1:
            t_stat, p_val = ttest_ind(snp_group, wt_group)
            p_values[channel] = p_val
        else:
            p_values[channel] = "Please provide more replicates for proper statistical analysis."

    # Since I wanted to see output, I wanted to print the summary table
    print(summary_df)

    # P-values for each channel (except DAPI) will be printed to see whether there is a significant difference between SNP and WT for each channel.
    print("\nT-test p-values (SNP vs WT):")
    for channel, p_val in p_values.items():
        print(f"{channel}: {p_val}")

    # A summary of means for each channel is created here.
    mean_summary = summary_df.groupby(['Channel', 'Image']).agg(
        Mean_Cell_Count=('Cell Count', 'mean'),
        Mean_Area=('Area', 'mean'),
        Mean_Normalized=('Normalized (by DAPI)', 'mean')
    ).reset_index()

    # Create a summary for SNP and WT to see the result of analysis
    final_summary = mean_summary.groupby('Channel').agg(
        Mean_Cell_Count_SNP=('Mean_Cell_Count', lambda x: x[mean_summary['Image'].str.startswith('SNP')].mean()),
        Mean_Cell_Count_WT=('Mean_Cell_Count', lambda x: x[mean_summary['Image'].str.startswith('WT')].mean()),
        Mean_Area_SNP=('Mean_Area', lambda x: x[mean_summary['Image'].str.startswith('SNP')].mean()),
        Mean_Area_WT=('Mean_Area', lambda x: x[mean_summary['Image'].str.startswith('WT')].mean()),
        Mean_Normalized_SNP=('Mean_Normalized', lambda x: x[mean_summary['Image'].str.startswith('SNP')].mean()),
        Mean_Normalized_WT=('Mean_Normalized', lambda x: x[mean_summary['Image'].str.startswith('WT')].mean()),
    ).reset_index()

    # Print mean summary
    print("\nMean Summary of Cell Count, Area, and Normalized Values:")
    print(final_summary)

    # Create separate bar plots for mean normalized values for each channel to be able to use as a visual data and make clear the difference beetween two groups.
    channels = ['KI67', 'SOX2', 'MAP2']
    for channel in channels:
        plt.figure(figsize=(8, 5))
        
        # Filter the mean normalized data for the current channel
        mean_normalized_channel = final_summary[final_summary['Channel'] == channel].melt(
            id_vars='Channel', value_vars=['Mean_Normalized_SNP', 'Mean_Normalized_WT'],
            var_name='Condition', value_name='Mean Normalized Value'
        )
        
        # Here I have tried to use seaborn library to plot bar plot for the current channel
        sns.barplot(data=mean_normalized_channel, x='Channel', y='Mean Normalized Value', hue='Condition', palette='muted')
        plt.title(f'Mean Normalized Values for {channel} (SNP vs WT)')
        plt.ylabel('Mean Normalized Value (by DAPI)')
        plt.xlabel('Channel')
        plt.legend(title='Condition')
        plt.tight_layout()
        plt.show()

# This is crucial for executing main function.
if __name__ == "__main__":
    main()



# In[ ]:




