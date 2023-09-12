import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import psutil


def report_to_df(report):
    report = [x.split(" ") for x in report.split("\n")]
    header = ["Class Name"] + [x for x in report[0] if x != ""]
    values = []
    for row in report[1:-1]:
        row = [value for value in row if value != ""]
        if row != []:
            while row.__len__() > header.__len__():
                tmp = list([row[0] + ' ' + row[1]])
                new_row = tmp + row[2:]
                row = new_row
            values.append(row)
    df = pd.DataFrame(data=values, columns=header)
    return df


def plot_loss(loss_values, config):
    # Create a plot of the loss values
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, config.training.num_epoch + 1), loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    # Specify the path where you want to save the plot
    plot_path = config.training_path / 'loss_plot.png'

    # Save the plot to the specified path
    plt.savefig(plot_path)

    # display the plot using plt.show()
    # plt.show()

    # Close the plot to release resources
    plt.close()


def memory_usage(config, model, device):

    # Create a dummy input tensor
    input_tensor = torch.randn((1, 1, config.sampling.target_size[0], config.sampling.target_size[1]))
    input_tensor = input_tensor.to(device)

    # Measure memory usage
    if device.type == 'cuda':
        # Measure GPU memory usage
        gpu_memory_before = torch.cuda.memory_allocated(device=device)
        _ = model(input_tensor)
        gpu_memory_after = torch.cuda.memory_allocated(device=device)
        gpu_memory_used = gpu_memory_after - gpu_memory_before

        print(f"GPU Memory Used (MB): {gpu_memory_used / (1024 * 1024)}")
    else:
        # Measure CPU memory usage
        cpu_memory_before = psutil.virtual_memory().used
        _ = model(input_tensor)
        cpu_memory_after = psutil.virtual_memory().used
        cpu_memory_used = cpu_memory_after - cpu_memory_before

        print(f"CPU Memory Used (MB): {cpu_memory_used / (1024 * 1024)}")