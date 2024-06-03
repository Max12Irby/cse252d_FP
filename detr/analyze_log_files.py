import os
import json
import matplotlib.pyplot as plt


def read_log_files(directory):
    data_structure = []
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    for subdir in subdirectories:
        subdir_path = os.path.join(directory, subdir)
        log_file_path = os.path.join(subdir_path, 'log.txt')
        with open(log_file_path, 'r') as file:
            log_data = json.load(file)                
            data_structure.append(log_data)
    
    data_structure.sort(key=lambda x: x.get('epoch', 0))
    return data_structure


def plot_losses(data_structure):
    epochs = [epoch['epoch'] for epoch in data_structure]
    train_losses = [epoch['train_loss'] for epoch in data_structure]
    test_losses = [epoch['test_loss'] for epoch in data_structure]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='s')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title('Train and Test Losses vs. Epoch Number')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_ap_values(data_structure):
    epochs = [epoch['epoch'] for epoch in data_structure]
    train_ap_50 = [epoch['train_loss_bbox_0'] for epoch in data_structure]
    train_ap_small = [epoch['train_loss_bbox_2'] for epoch in data_structure]
    train_ap_medium = [epoch['train_loss_bbox_3'] for epoch in data_structure]
    train_ap_large = [epoch['train_loss_bbox_4'] for epoch in data_structure]
    
    test_ap_50 = [epoch['test_loss_bbox_0'] for epoch in data_structure]
    test_ap_small = [epoch['test_loss_bbox_2'] for epoch in data_structure]
    test_ap_medium = [epoch['test_loss_bbox_3'] for epoch in data_structure]
    test_ap_large = [epoch['test_loss_bbox_4'] for epoch in data_structure]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_ap_50, label='Train AP loss (50% threshold)', marker='o', color='blue')
    plt.plot(epochs, train_ap_small, label='Train AP loss for Small Objects', marker='s', color='blue')
    plt.plot(epochs, train_ap_medium, label='Train AP loss for Medium Objects', marker='^', color='blue')
    plt.plot(epochs, train_ap_large, label='Train AP loss for Large Objects', marker='x', color='blue')
    
    plt.plot(epochs, test_ap_50, label='Test AP loss (50% threshold)', marker='o', linestyle='--', color='orange')
    plt.plot(epochs, test_ap_small, label='Test AP loss for Small Objects', marker='s', linestyle='--', color='orange')
    plt.plot(epochs, test_ap_medium, label='Test AP loss for Medium Objects', marker='^', linestyle='--', color='orange')
    plt.plot(epochs, test_ap_large, label='Test AP loss for Large Objects', marker='x', linestyle='--', color='orange')
    
    plt.xlabel('Epoch Number')
    plt.ylabel('AP Value')
    plt.title('Train and Test AP Loss Values for Small, Medium, and Large Objects vs. Epoch Number')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    directory_name = "detr/output_backbone"
    log_data_structure = read_log_files(directory_name)
    #print(log_data_structure)
    print([log_data['epoch'] for log_data in log_data_structure])
    plot_losses(log_data_structure)
    plot_ap_values(log_data_structure)