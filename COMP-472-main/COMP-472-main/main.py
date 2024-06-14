import os
from process import process_images
from labelling import generate_csv
from data_viz import load_images, plot_class_distribution, plot_pixel_intensity_distribution,plot_sample_images_with_histograms

def main():
    # Define the paths to your directories for the datasets 
    # Which are "Comp472OriginalDataSet"
    # Also, redefines the categories and sets as arguments to pass for the processing and labelling functionalities.
    base_dir_original = os.path.join(os.path.dirname(__file__), 'dataset', 'Comp472OriginalDataSet')
    categories = ["Angry", "Happy", "Engaged", "Neutral"]
    sets = ["Train", "Test"]

    # Call the function to process the images
    print("Starting image processing...")
    process_images(base_dir_original, sets, categories)
    print("Image processing completed.")

    # Call the function to generate the CSV file
    # Creates a new CSV file 
    csv_file_path = os.path.join(os.path.dirname(__file__), 'dataset_labels.csv')

    print("Generating CSV file...")
    generate_csv(base_dir_original, sets, categories, csv_file_path)
    print(f"CSV file created at {csv_file_path}")

    # Call the function to load the processed images and to visualize the data
    # Load processed images
    images = load_images(base_dir_original, sets, categories)

    # Visualize data functionalities such as 
    # Class distribution, Pixel intensity distribution and sample images
    print("Visualizing class distribution...")
    plot_class_distribution(images, sets, categories)

    print("Visualizing pixel intensity distribution...")
    plot_pixel_intensity_distribution(images, categories)

    print("Visualizing sample images...")
    plot_sample_images_with_histograms(images, categories, num_samples=15)


if __name__ == "__main__":
    main()