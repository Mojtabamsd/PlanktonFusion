from torchvision.utils import save_image
from torch import zeros
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap.umap_ as umap


def visualization_output(img_org, outputs, visualisation_path, epoch, batch_size=32, gray=True):
    if gray:
        dimensionality = 1
    else:
        dimensionality = 3

    img_input = img_org.cpu().data
    # img_input = img_input.reshape(round(img_input.shape[0] / dimensionality), dimensionality,
    #                               image_size, image_size)
    img_output = outputs.cpu().data
    # img_output = img_output.reshape(round(img_output.shape[0] / dimensionality), dimensionality,
    #                                 image_size, image_size)
    # num_patch_show = round(inputs.shape[0] / dimensionality)
    num_patch_show = round(img_org.shape[0])
    if num_patch_show > batch_size:
        num_patch_show = batch_size
    concat = zeros(
        [
            num_patch_show * 2,
            dimensionality,
            img_input.shape[2],
            img_input.shape[3]
        ]
    )

    concat[0::2, ...] = img_input[0:num_patch_show, ...]
    concat[1::2, ...] = img_output[0:num_patch_show, ...]

    concat_img_path = visualisation_path / (
            "reconstruction_" + str(epoch+1).zfill(5) + ".png"
    )
    save_image(concat, concat_img_path, nrow=4)

    # # save first layer edge
    # weight_img_path = visualisation_path / (
    #     "weight_" + str(epoch).zfill(5) + ".png"
    # )
    # model.save_cnn_weight_image(str(weight_img_path))


def tsne_plot(latent_vectors, all_labels, int_to_label, out_path):

    tsne = TSNE(n_components=2, random_state=42, learning_rate=500, n_iter=5000)
    latent_tsne = tsne.fit_transform(np.vstack(latent_vectors))

    # pca = PCA(n_components=2)
    # latent_tsne = pca.fit_transform(np.vstack(latent_vectors))

    # umap_model = umap.UMAP(n_components=2)
    # latent_tsne = umap_model.fit_transform(np.vstack(latent_vectors))

    plt.figure(figsize=(10, 8))

    for label in np.unique(all_labels):
        indices = all_labels == label
        plt.scatter(latent_tsne[indices, 0], latent_tsne[indices, 1],
                    marker=".",
                    # s=1.4,
                    label=int_to_label[label])

    plt.title('t-SNE Plot of Latent Vectors')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    # plt.show()

    out_path_name = out_path / "tsne_plot.png"
    plt.savefig(out_path_name, dpi=600)
