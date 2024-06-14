import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.utils import shuffle

# Define the paths to your directories, categories and sets
base_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'Comp472OriginalDataSet')
categories = ["Angry", "Happy", "Engaged", "Neutral"]
sets = ["Train", "Test"]

# Utility function to load images
def load_images(base_dir, sets, categories):
    """
    Loads images from specified directories and categorizes them by sets and categories.

    Parameters:
    - base_dir (str): Base directory path where images are stored.
    - sets (list): List of dataset sets (e.g., ['Train', 'Test']).
    - categories (list): List of image categories (e.g., ['Angry', 'Happy', 'Engaged', 'Neutral']).

    Returns:
    - images (dict): Dictionary containing images categorized by sets and categories.
    """
    images = {set_name: {category: [] for category in categories} for set_name in sets}
    
    for set_name in sets:
        for category in categories:
            category_path = os.path.join(base_dir, set_name, category)
            # Use glob to find all image files in the category path
            for file_path in glob.glob(os.path.join(category_path, '*')):
                # Check if the file is a valid image file
                if any(file_path.lower().endswith(extension) for extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
                    with Image.open(file_path) as img:
                        images[set_name][category].append(np.array(img))
    
    return images

# Class distribution visualization
def plot_class_distribution(images, sets, categories):
    """
    Plots the distribution of classes across different dataset sets.

    Parameters:
    - images (dict): Dictionary containing images categorized by sets and categories.
    - sets (list): List of dataset sets (e.g., ['Train', 'Test']).
    - categories (list): List of image categories (e.g., ['Angry', 'Happy', 'Engaged', 'Neutral']).
    """
    # Calculate the number of images per class for each dataset set
    class_counts = {set_name: {category: len(images[set_name][category]) for category in categories} for set_name in sets}
    
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for i, set_name in enumerate(sets):
        axs[i].bar(class_counts[set_name].keys(), class_counts[set_name].values())
        axs[i].set_title(f'{set_name} Set Class Distribution')
        axs[i].set_xlabel('Class')
        axs[i].set_ylabel('Number of Images')
    plt.tight_layout()
    plt.show()

# Pixel intensity distribution visualization
def plot_pixel_intensity_distribution(images, categories):
    """
    Plots the pixel intensity distribution for each image category.

    Parameters:
    - images (dict): Dictionary containing images categorized by sets and categories.
    - categories (list): List of image categories (e.g., ['Angry', 'Happy', 'Engaged', 'Neutral']).
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    for i, category in enumerate(categories):
        all_pixels = []
        for set_name in images:
            for img in images[set_name][category]:
                all_pixels.extend(img.flatten())
        
        axs[i].hist(all_pixels, bins=50, color='gray', alpha=0.5)
        axs[i].set_title(f'{category} Pixel Intensity Distribution')
        axs[i].set_xlabel('Pixel Intensity')
        axs[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Sample images visualization with pixel intensity histograms
def plot_sample_images_with_histograms(images, categories, num_samples=15):
    """
    Plots sample images from each category with their corresponding pixel intensity histograms.

    Parameters:
    - images (dict): Dictionary containing images categorized by sets and categories.
    - categories (list): List of image categories (e.g., ['Angry', 'Happy', 'Engaged', 'Neutral']).
    - num_samples (int): Number of sample images per category to display (default is 15).
    """
    num_rows = 5  # 5 rows for images per category
    num_columns = 6  # 3 images and 3 histograms per row
    
    for i, category in enumerate(categories):
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(20, num_rows * 2))
        
        sample_images = []
        for set_name in images:
            sample_images.extend(images[set_name][category])
        sample_images = shuffle(sample_images, random_state=42)[:num_samples]
        
        for j, img in enumerate(sample_images):
            row = j // 3
            col_image = (j % 3) * 2
            col_hist = col_image + 1
            
            axs[row, col_image].imshow(img, cmap='gray')
            axs[row, col_image].axis('off')
            
            pixels = img.flatten()
            axs[row, col_hist].hist(pixels, bins=50, color='gray', alpha=0.5)
            
            axs[row, col_hist].set_title(f'Histogram for {category} Image', fontsize=10)
        
        fig.suptitle(f'Sample Images and Histograms for {category}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()



# If you'd like to run this Python file solely 
# And, get the data visualization then uncomment the following function calls
   
# images = load_images(base_dir, sets, categories)
# plot_class_distribution(images, sets, categories)
# plot_pixel_intensity_distribution(images, categories)
# plot_sample_images_with_histograms(images, categories,num_samples=15)