import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import psutil
import seaborn as sns
import os


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


def plot_loss(loss_values, num_epoch, training_path, name='loss_plot.png'):
    # Create a plot of the loss values
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, num_epoch + 1), loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    # Specify the path where you want to save the plot
    plot_path = os.path.join(training_path, name)

    # Save the plot to the specified path
    plt.savefig(plot_path)

    # display the plot using plt.show()
    # plt.show()

    # Close the plot to release resources
    plt.close()


def plot_results(report_df, cm, out_path, target_names):

    # Normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(16, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=target_names, yticklabels=target_names)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    # plt.show()

    cm_path = os.path.join(out_path, 'confusion_matrix.eps')
    plt.savefig(cm_path, dpi=600)

    # Extract class names and F1-scores
    class_names = report_df['Class Name'][:report_df.shape[0]-3]
    f1_scores = report_df['f1-score'][:report_df.shape[0]-3].astype(float)

    # Extract macro and weighted averages
    macro_avg = report_df.iloc[-2][2:].astype(float)
    weighted_avg = report_df.iloc[-1][2:].astype(float)  #

    # Create a bar chart
    plt.figure(figsize=(16, 10))

    # Plot class F1-scores
    plt.bar(class_names, f1_scores, color='skyblue', label='F1-Score')

    # Add a bar for Macro Average and Weighted Average
    plt.bar('Macro Avg', macro_avg['f1-score'], color='orange', label='Macro Avg')
    plt.bar('Weighted Avg', weighted_avg['f1-score'], color='green', label='Weighted Avg')

    plt.xlabel('Class Name')
    plt.ylabel('Scores')
    plt.title('Scores by Class')
    plt.xticks(rotation=45)
    plt.legend()
    # plt.show()

    cr_path = os.path.join(out_path, 'classification_report.eps')
    plt.savefig(cr_path, dpi=600)


def plot_f1_scores(out_path_name, model_names, *dataframes):
    # Combine the reports into a single DataFrame
    combined_report = pd.concat(dataframes, keys=model_names)

    # Filter out the row with the class name "accuracy" from each DataFrame
    combined_report = combined_report[combined_report["Class Name"] != "accuracy"]

    # Set the style of seaborn
    sns.set(style="whitegrid")

    # Set up the bar plot using seaborn
    plt.figure(figsize=(16, 10))
    ax = sns.barplot(x="Class Name", y="f1-score", hue="level_0", data=combined_report.reset_index(), width=0.35)

    # Rotate x-axis labels by 90 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Customize the plot
    plt.xlabel('')
    plt.ylabel('f1-score')
    # plt.title('f1-scores for Each Class by Model')
    legend = plt.legend(title="Model", loc="upper left", bbox_to_anchor=(1, 1))

    # Show the plot
    plt.tight_layout()
    # plt.show()

    plt.savefig(out_path_name, dpi=600)


def plot_f1_scores_diff(out_path_name, model_names, *dataframes):

    combined_report = pd.concat(dataframes, keys=model_names)
    combined_report = combined_report[combined_report["Class Name"] != "accuracy"]

    # Calculate improvement in F1-score
    base_model_scores = combined_report.loc["Base Model", "f1-score"]
    model1_scores = combined_report.loc["Model1", "f1-score"]
    model2_scores = combined_report.loc["Model2", "f1-score"]
    model3_scores = combined_report.loc["Model3", "f1-score"]

    diff_model1 = model1_scores - base_model_scores
    diff_model2 = model2_scores - base_model_scores
    diff_model3 = model3_scores - base_model_scores

    diff_df = pd.DataFrame({
        "Class Name": combined_report.reset_index()["Class Name"].unique(),
        "diff_Model1": diff_model1.values,
        "diff_Model2": diff_model2.values,
        "diff_Model3": diff_model3.values
    })

    diff_df = pd.melt(diff_df, id_vars=["Class Name"], var_name="Model", value_name="diff")

    # sns.set(style="whitegrid")
    # plt.figure(figsize=(12, 6))

    # ax = sns.barplot(x="Class Name", y="diff", hue="Model", data=diff_df, width=0.35)

    # Pivot the DataFrame
    df_pivot = diff_df.pivot(index='Class Name', columns='Model', values='diff').reset_index()
    df_pivot = df_pivot[0:-1]
    # Number of subplots required
    num_plots = len(df_pivot)

    # Create subplots
    fig, axes = plt.subplots(int(num_plots / 2), 2, figsize=(10, 6), sharex=True)

    # Plot each class in a separate subplot
    for idx, (row, ax) in enumerate(zip(df_pivot.iterrows(), axes.flatten())):
        sns.lineplot(ax=ax, x=['Base', 'Model1', 'Model2', 'Model3'], y=[0, row[1]['diff_Model1'],
                                                                         row[1]['diff_Model2'],
                                                                         row[1]['diff_Model3']])
        ax.scatter(['Base', 'Model1', 'Model2', 'Model3'], [0, row[1]['diff_Model1'],
                                                            row[1]['diff_Model2'],
                                                            row[1]['diff_Model3']])
        ax.set_title(row[1]['Class Name'])
        ax.set_ylabel('Score')

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # plt.xlabel('Models')

    plt.ylim(-max(abs(diff_df["diff"])), max(abs(diff_df["diff"])))

    # Show the plot
    plt.tight_layout()
    plt.savefig(out_path_name, dpi=600)


def memory_usage(config, model, device):

    # Create a dummy input tensor
    x = torch.randn((2, 1, config.sampling.target_size[0], config.sampling.target_size[1]))
    x = x.to(device)
    normalized_x = (x - x.min()) / (x.max() - x.min())

    # Measure memory usage
    if device.type == 'cuda':
        # Measure GPU memory usage
        gpu_memory_before = torch.cuda.memory_allocated(device=device)
        _ = model(normalized_x)
        gpu_memory_after = torch.cuda.memory_allocated(device=device)
        gpu_memory_used = gpu_memory_after - gpu_memory_before

        return f"GPU Memory Used (MB): {gpu_memory_used / (1024 * 1024)}"
    else:
        # Measure CPU memory usage
        cpu_memory_before = psutil.virtual_memory().used
        _ = model(normalized_x)
        cpu_memory_after = psutil.virtual_memory().used
        cpu_memory_used = cpu_memory_after - cpu_memory_before

        return f"CPU Memory Used (MB): {cpu_memory_used / (1024 * 1024)}"


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    if type(train_data.dataset).__name__ == 'UvpDataset':
        training_labels = np.array(train_data.dataset.data_frame['label'].map(train_data.dataset.label_to_int))
    else:
        if isinstance(train_data, np.ndarray):
            training_labels = np.array(train_data).astype(int)
        else:
            training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)