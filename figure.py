import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
matplotlib.use('PDF')

def show_hist(class_list,total):
    data = class_list
    plt.figure(dpi=128,figsize=(8,8))
    plt.hist(data, bins=total)
    my_x_ticks = np.arange(0, total, int(total/10))
    plt.xticks(my_x_ticks)
    plt.title("Train dataset")
    plt.xlabel("class_id")
    plt.ylabel("class_number")
    plt.savefig('picture/Histogram_{}_{}.pdf'.format(str(total-10),str(10)))

def show_confusion(y_pred, y_true, total):
    y_pred = y_pred.T[0]
    plt.figure(dpi=128, figsize=(total,total))
    C = confusion_matrix(y_true, y_pred)
    plt.matshow(C, cmap=plt.cm.Blues)# 更改颜色
    # plt.colorbar() # 添加图例
    for i in range(len(C)):
        for j in range(len(C)):
            if total <= 40:
                plt.annotate(C[j, i], xy=(i, j),horizontalalignment='center', fontsize=5, verticalalignment='center')
            elif total > 40 and total <= 70:
                plt.annotate(C[j, i], xy=(i, j),horizontalalignment='center', fontsize=3, verticalalignment='center')
            elif total > 70 and total <= 100:
                plt.annotate(C[j, i], xy=(i, j),horizontalalignment='center', fontsize=1, verticalalignment='center')
    if total <= 30:
        plt.tick_params(labelsize=8) # 字体大小
    elif total > 30 and total <= 60:
        plt.tick_params(labelsize=5) # 字体大小
    elif total > 60 and total <= 100:
        plt.tick_params(labelsize=2) # 字体大小
    labels = [str(i) for i in range(total)]
    plt.ylabel('True label', fontsize = 8)
    plt.xlabel('Predicted label', fontsize = 8)
    plt.yticks(range(total), labels)
    plt.xticks(range(total), labels,rotation=45)#X轴字体倾斜45°
    plt.savefig('picture/lt/0.01/der_2stage/ConfusionMetrix_{}.png'.format(str(total)), dpi=128)

def _get_fontsize_and_ticksize(total):
    """
    Helper function to determine the font size and tick size based on the total number.
    """
    if total <= 40:
        return 5, 8
    elif total > 40 and total <= 70:
        return 3, 5
    elif total > 70 and total <= 100:
        return 1, 2


def plot_avg_confidences_per_class(class_confidences_dict, cur_task):
    """
    Plot the average confidences of samples from each class across all classes.
    
    :param class_confidences_dict: Dictionary where keys are class IDs and values are lists of confidences (in all classes) for samples of that class.
    """
    total_classes = len(class_confidences_dict.keys())
    
    fontsize, ticksize = _get_fontsize_and_ticksize(total_classes)

    for class_id, confidences in class_confidences_dict.items():
        # Compute average confidences for samples of this class across all classes
        mean_confidences = np.mean(confidences, axis=0)
        plt.figure(dpi=128, figsize=(total_classes, total_classes))
        plt.bar(np.arange(total_classes), mean_confidences, color='skyblue')
        plt.xlabel('Class ID', fontsize=fontsize)
        plt.ylabel('Average Confidence for True Label_{}'.format(str(class_id)), fontsize=fontsize)
        plt.title('Average Confidence of Samples with True Label {} Across All Classes'.format(str(class_id)), fontsize=fontsize)
        plt.tick_params(labelsize=ticksize)
        plt.xticks(range(total_classes), [str(i) for i in range(total_classes)], rotation=45)
        plt.savefig('picture/lt/0.01/der_2stage/{}_AverageConfidenceForTrueLabel_{}.png'.format(str(cur_task),str(class_id)))
        plt.close()

def visualize_tsne_features(class_features_dict, cur_task, sample_size_per_class=100):
    """
    Use t-SNE to visualize the features of samples from each class in 2D space.

    :param class_features_dict: Dictionary where keys are class IDs and values are lists of feature vectors for samples of that class.
    :param cur_task: current task ID or identifier.
    :param sample_size_per_class: Number of samples to be randomly selected per class for visualization.
    """
    total_classes = len(class_features_dict.keys())
    fontsize, ticksize = _get_fontsize_and_ticksize(total_classes)
    
    # Prepare data for t-SNE
    all_features = []
    all_labels = []

    for class_id, features_list in class_features_dict.items():
        if len(features_list) > sample_size_per_class:
            selected_indices = np.random.choice(len(features_list), sample_size_per_class, replace=False)
            sampled_features = [features_list[i] for i in selected_indices]
        else:
            sampled_features = features_list
        all_features.extend(sampled_features)
        all_labels.extend([class_id] * len(sampled_features))

    all_features = [feat.cpu().numpy() for feat in all_features]
    tsne = TSNE(n_components=2, random_state=0, init='pca')
    reduced_features = tsne.fit_transform(all_features)

    plt.figure(dpi=128, figsize=(total_classes, total_classes))
    for class_id in class_features_dict.keys():
        class_data = reduced_features[np.array(all_labels) == class_id]
        plt.scatter(class_data[:, 0], class_data[:, 1], label=str(class_id), alpha=0.6)

    plt.legend()
    plt.xlabel('t-SNE Dimension 1', fontsize=fontsize)
    plt.ylabel('t-SNE Dimension 2', fontsize=fontsize)
    plt.title('t-SNE Visualization of Features for Task {}'.format(cur_task), fontsize=fontsize)
    plt.tick_params(labelsize=ticksize)
    plt.savefig('picture/lt/0.01/der_2stage/{}_tSNE_Visualization.png'.format(str(cur_task)))
    plt.close()


