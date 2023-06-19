import os

# Define data directory and target file paths
data_root = r'./data/PlantVillage'
train_file = 'train.txt'
val_file = 'val.txt'
test_file = 'test.txt'
label_file = 'labels.txt'

# Get subdirectories in the data root directory
subdirectories = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

# Create training, validation, and test files
with open(train_file, 'w') as train_f, open(val_file, 'w') as val_f, open(test_file, 'w') as test_f:
    for class_index, subdirectory in enumerate(subdirectories):
        subdirectory_path = os.path.join(data_root, subdirectory)
        image_names = os.listdir(subdirectory_path)
        num_images = len(image_names)

        # Determine the number of images for training, validation, and test
        num_train = int(0.8 * num_images)  # 80% for training
        num_val = int(0.1 * num_images)  # 10% for validation
        num_test = num_images - num_train - num_val  # Remaining 10% for test

        # Write image names and labels to the corresponding files
        for i, image_name in enumerate(image_names):
            image_path = os.path.join(subdirectory_path, image_name)
            if i < num_train:
                train_f.write(f"{image_path},{class_index}\n")
            elif i < num_train + num_val:
                val_f.write(f"{image_path},{class_index}\n")
            else:
                test_f.write(f"{image_path},{class_index}\n")

# Save labels to a file
with open(label_file, 'w') as label_f:
    label_f.write('\n'.join(subdirectories))
