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



if __name__ == '__main__':
    directory_name = "detr/output_backbone"

    log_data_structure = read_log_files(directory_name)
    #print(log_data_structure)
    print([log_data['epoch'] for log_data in log_data_structure])
    plot_losses(log_data_structure)
    #plot_ap_values(log_data_structure)