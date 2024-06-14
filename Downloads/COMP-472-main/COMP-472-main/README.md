# COMP 472 Artificial Intelligence (Summer 2024)

## Group Number: AK_9 

Christian Dingayan 		40176620

Krupesh Patel  		 		40175794

Abdul-Hakim Skaik			40198779



## Libraries: 

### For Training, Processing & Data Visualization

numpy, pillow, os, csv, matplotlib, scikit-learn


## Command used to install:

pip install pillow numpy matplotlib scikit-learn


## Description of each file 

### process.py

The purpose of this file is to process the dataset and images within the directory. Therefore, we are cleaning the data to specific format which is "48x48" size and grayscaled. The processed images are then placed back in to folder in which the dataset came from which is "Comp472OriginalDataSet". 

### labelling.py 

The purpose of this file to label the processed dataset and images. Specifically, we are generating a CSV file with three parameters (Set, Category, and Image Path). This file is newly re-generated each time that this file is ran. 

### data_viz.py 

The purpose of this file is to visualize our dataset according to given specifications. Firstly, we are generating the plot for Class Distribution which is showing the number of images in each class. Secondly, the second plot generates the pixel intensity distribution for each of our respective classes. Lastly, the plotting of the sample images of their histograms with the pixel intensity distribution is separated onto 4 different plots. 


## How to execute code

1. Execute the code by simply calling "python main.py" into the terminal. An alternative is to simply run the main.py file.

2. The data cleaning is done through the process.py file. This file is executed first as per the sequential structure of the main.py file.

3. The data visualization is done with the corresponding plots following the process.py and labelling.py (ran right after process.py) files are completed. 
