# SC00039_Final_Project_MC
## Purpose
This code were designed to analyze ND2 image files containing different channels specifically aimed to evaluate proliferation/differentiation characteristics of different lines (e.g., DAPI, KI67, SOX2, MAP2). It performs image processing tasks such as background subtraction, Otsu thresholding, and segmentation. The processed data is normalized using the DAPI channel, and statistical analysis (t-tests) is applied to determine differences between "SNP" and "WT" samples.
## Prerequisites
Before running the code, be sure that you have the following installed:
1)Conda or use an existing installation of Anaconda/Miniconda.
2)Python 3.9
## Installation
# Step 1: Clone the Repository
Download the project files to your local machine. You can do this by cloning the repository or downloading it as a zip.
git clone <repository-url>
cd <project-folder>
# Step 2: Install Conda
If you don't already have Conda, install it by following the instructions at Miniconda.
## Environment Setup
