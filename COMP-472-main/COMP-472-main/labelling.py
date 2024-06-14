import os
import csv

# Define the relative path to your directories
# Same thing again we just define the paths, just make sure that the last part matches the name in your file directory. 
# Also if you can run the script again and it updates if there were any changes in the original dataset i.e: when we add our pictures.
base_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'Comp472OriginalDataSet')
categories = ["Angry", "Happy", "Engaged", "Neutral"]
sets = ["Train", "Test"]

# CSV file path
csv_file_path = os.path.join(os.path.dirname(__file__), 'dataset_labels.csv')

# Create a CSV file and write the header
def generate_csv(base_dir, sets, categories, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Set', 'Category', 'ImagePath'])

    # Iterate through the directories
        for set_name in sets:
            for category in categories:
                category_path = os.path.join(base_dir, set_name, category)
                for filename in os.listdir(category_path):
                    file_path = os.path.join(category_path, filename)
                    if os.path.isfile(file_path):
                        relative_path = os.path.relpath(file_path, os.path.dirname(__file__))
                        csv_writer.writerow([set_name, category, relative_path])

print(f"CSV file created at {csv_file_path}")