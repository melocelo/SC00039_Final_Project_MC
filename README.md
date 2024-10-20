# SC00039_Final_Project_MC
## Purpose
This code were designed to analyze ND2 image files containing different channels specifically aimed to evaluate proliferation/differentiation characteristics of different lines (e.g., DAPI, KI67, SOX2, MAP2). It performs image processing tasks such as background subtraction, Otsu thresholding, and segmentation. The processed data is normalized using the DAPI channel, and statistical analysis (t-tests) is applied to determine differences between "SNP" and "WT" samples.
## Prerequisites
Before running the code, be sure that you have the following installed:

1)Conda or use an existing installation of Anaconda/Miniconda.
2)Python 3.9

## Test-data
Since ND2 image files are big, I could not manage to upload them here. Please download Image_Analysis folder contains 4 SNP and 4 WT images into your computer before running the code via Google drive link below:
>> https://drive.google.com/drive/folders/1e7LlDjyH3JKyG7Flhx9VhJ8vnMDfLowr?usp=share_link
## Installation
### Step 1: Clone the Repository
Download the project files to your local machine. You can do this by cloning the repository or downloading it as a zip.
```bash
git clone https://github.com/melocelo/SC00039_Final_Project_MC.git
```
### Step 2: Install Conda
If you don't already have Conda, install it by following the instructions at Miniconda.
## Environment Setup
In this project, we need a Conda environment so you can create the environment from the provided environment.yml file in this repository.
### Step 1: To create the environment that will be used to install Jupyter Lab.
```bash
conda create -y -n trial-env -c conda-forge python=3.9
```
### Step 2: To activate environment
When the virtual environment with the name trial-env is created, you can activate it to install the desired packages.
```bash
conda activate trial-env
```
### Step 3: To install JupyterLab and make trial-env available to JupyterLab
```bash
conda install -c conda-forge jupyterlab
ipython kernel install --user --name=trial-env
```

When the environment is activated, you can verify that the correct packages are installed via:
```bash
conda list
```
If everything is correct, you are supposed to see the required packages (like scikit-image, pandas, seaborn, nd2reader, etc.) in the list.
## Running the code 
You can run the code in terminal or in Jupyter Notebook by following these steps.
### Running in terminal
```bash
conda activate trial-env
python Final_Assignment.py
```
Before runnning the code, be sure that you are in the directory of the code.
### Running in Jupyter Notebook
```bash
conda activate trial-env
jupyter lab
```
You can open the Python script or notebook in Jupyter and run the cells to execute the analysis.

