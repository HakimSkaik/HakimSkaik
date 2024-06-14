import os
from PIL import Image

# Define the paths to your directories
# This should work on your local computer, just make sure to change the last part to your dataset name if you change it.
base_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'Comp472OriginalDataSet')
categories = ["Angry", "Happy", "Engaged", "Neutral"]
sets = ["Train", "Test"]

# Define the desired format
# We keep it to 48x48 and grayscale since 1500 of the dataset images are originally in that format.
desired_size = (48, 48)
desired_mode = "L"  # 'L' mode is for grayscale

def check_and_reformat_image(image_path):
    with Image.open(image_path) as img:
        if img.size != desired_size or img.mode != desired_mode:
            print(f"Reformatting: {image_path}")
            # Resize the image
            img = img.resize(desired_size)
            # Convert to grayscale
            img = img.convert(desired_mode)
            # Save the image
            img.save(image_path)

# Iterate through the directories 
# This part essentially goes through each set (training and testing) and then through each category making sure every image is properly formatted.
def process_images(base_dir, sets, categories):
    for set_name in sets:
        for category in categories:
            category_path = os.path.join(base_dir, set_name, category)
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                if os.path.isfile(file_path):
                    check_and_reformat_image(file_path)
