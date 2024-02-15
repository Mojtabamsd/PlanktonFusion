from torchvision.utils import save_image
from torch import zeros


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