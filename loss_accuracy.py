import argparse
import matplotlib.pyplot as plt

# Define the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--log_file', type=str, default='Resnettraining1e-6.log', help='Path to the training log file')
args = parser.parse_args()

# Read the training log file
with open(args.log_file, 'r') as file:
    lines = file.readlines()

# Convert predicted class to class name
class_names = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
               "Potato___healthy", "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
               "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
               "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy"]

# Extract loss and accuracy values from the log file
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
train_times = []
val_times = []
epochs = []
recalls = [[] for _ in range(15)]
print(f"Start reading data {args.log_file}...")
for line in lines:
    try:
        if 'Average Training Loss' in line:
            train_loss = float(line.split('Average Training Loss: ')[1].split(',')[0])
            train_losses.append(train_loss)
            train_accuracy = float(line.split('Average Training Accuracy: ')[1].split('%')[0])
            train_accuracies.append(train_accuracy)
            val_loss = float(line.split('Average Validation Loss: ')[1].split(',')[0])
            val_losses.append(val_loss)
            val_accuracy = float(line.split('Average Validation Accuracy: ')[1].split('%')[0])
            val_accuracies.append(val_accuracy)
            if train_loss is None or train_accuracy is None or val_loss is None or val_accuracy is None:
                raise ValueError("Invalid data found in log file.")
        elif ', Training Loss:' in line:
            train_time = float(line.split('Time: ')[1].split('s')[0])
            train_times.append(train_time)
        elif 'Validation Loss' in line:
            val_time = float(line.split('%, Time: ')[1].split('s')[0])
            val_times.append(val_time)
        elif 'Recall for class' in line:
            epoch_recall = float(line.split(': ')[1].split('%')[0])
            class_index = int(line.split('class ')[1].split(':')[0])
            recalls[class_index].append(epoch_recall)

            if len(epochs) < len(recalls[class_index]):
                epochs.append(len(epochs) + 1)
    except Exception as e:
        print(f"Error occurred while parsing line: {line}")
        print(f"Error message: {str(e)}")
print(f"Successful data reading {args.log_file}...")

# Plot the loss
print(f"Start printing loss {args.log_file}...")
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, linestyle='--', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.ylim([0, 10])  # Set the y-axis limits
plt.xlim([0, 100])  # Set the x-axis limits
plt.savefig(f'{args.log_file}_loss_plot.png')
print(f"Successful save loss {args.log_file}...")

# Plot the accuracy
print(f"Start printing accuracy {args.log_file}...")
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, linestyle='--', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.ylim([0, 100])  # Set the y-axis limits
plt.xlim([0, 100])  # Set the x-axis limits
plt.savefig(f'{args.log_file}_accuracy_plot.png')
print(f"Successful save accuracy {args.log_file}...\n")

# Calculate total time
total_train_time = sum(train_times)
total_val_time = sum(val_times)

# Plot the accuracy
print(f"Start printing Time {args.log_file}...")
plt.figure(figsize=(10, 5))
plt.plot(train_times, color='black', linestyle='-', label='Training Time')  # 设置训练时间折线为黑色的实线
plt.plot(val_times, color='black', linestyle='--', label='Validation Time')  # 设置验证时间折线为黑色的虚线
plt.xlabel('Epoch')
plt.ylabel('Time')
plt.title('Training and Validation Time')
plt.legend()

# 在合适的位置添加总时间文本
#plt.text(len(train_times) - 1, train_times[-1], f'Total Train Time: {total_train_time:.2f} s', color='black',
#         fontsize=10, horizontalalignment='right', verticalalignment='bottom')
#plt.text(len(val_times) - 1, val_times[-1], f'Total Validation Time: {total_val_time:.2f} s', color='black',
#         fontsize=10, horizontalalignment='right', verticalalignment='bottom')

plt.savefig(f'{args.log_file}_Time.png')
print(f"Successful save Time {args.log_file}...\n")

# Calculate total time
print(f"{args.log_file} Total training time: {total_train_time} seconds")
print(f"{args.log_file} Total validation time: {total_val_time} seconds\n")



# Plot the recall for each class
plt.figure(figsize=(20, 10))
for class_index, recall_values in enumerate(recalls):
    plt.scatter(epochs, recall_values, label=f'Class {class_names[class_index]}')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall for Each Class')
plt.ylim([0, 100])  # Set the y-axis limits
plt.legend()
plt.savefig(f'{args.log_file}_recall_plot.png')