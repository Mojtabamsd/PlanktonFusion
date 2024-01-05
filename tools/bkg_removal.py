from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure


def normalize(mask, range=255):
    normalized = (range * (mask - np.min(mask)) / np.ptp(mask)).astype(int).astype('uint8')
    return normalized


def find_min_point_inside(image, mask):
    min_point = np.unravel_index(np.argmin(image[mask]), image.shape)
    return min_point


def region_growing(image, seed, max_iterations=20000):
    mask = np.zeros_like(image, dtype=np.bool)
    region_mean = image[seed]
    tolerance = 25  # Adjust this threshold based on your image characteristics
    iterations = 0

    stack = [seed]

    while stack and iterations < max_iterations:
        current = stack.pop()

        if not mask[current]:
            mask[current] = True

            # Get neighboring pixels
            neighbors = [
                (current[0] + 1, current[1]),
                (current[0] - 1, current[1]),
                (current[0], current[1] + 1),
                (current[0], current[1] - 1),
            ]

            for neighbor in neighbors:
                # Check if the neighbor is within the image boundaries
                if (
                        0 <= neighbor[0] < image.shape[0]
                        and 0 <= neighbor[1] < image.shape[1]
                        and not mask[neighbor]
                        and np.abs(image[neighbor] - region_mean) < tolerance
                ):
                    stack.append(neighbor)

        iterations += 1

    return mask


root = r'D:\mojmas\files\Projects\Lisstholo\test'

for file in Path(root).glob('*.png'):
    input_path = str(file)
    output_path = str(file.parent / 'seg' / (file.stem + ".out.png"))
    img = cv2.imread(input_path, 0)

    gray_image = cv2.medianBlur(img, 3)
    # gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # # Get the initial seed point from the user
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.imshow(gray_image, cmap='gray')
    # ax.set_title('Click to provide a seed point for region growing')

    # seed_point = plt.ginput(1, show_clicks=True)[0]
    # seed_point = (int(seed_point[1]), int(seed_point[0]))

    seed_point = find_min_point_inside(gray_image, gray_image > 0)
    seed_point = (seed_point[0], seed_point[1])

    segmentation_mask = region_growing(gray_image, seed_point)

    contours = measure.find_contours(segmentation_mask, 0.5)

    # Filter out small contours based on area
    min_contour_area = 30  # Adjust this threshold based on your requirements
    filtered_contours = [contour for contour in contours if contour.shape[0] > min_contour_area]

    # # Display the original image and the segmented image with the filtered contour
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.imshow(gray_image, cmap='gray')
    #
    # for contour in filtered_contours:
    #     ax.plot(contour[:, 1], contour[:, 0], '-r', linewidth=2)
    #
    # ax.set_title('Region Growing Contour Segmentation (Filtered)')
    # plt.show()

    threshold_value_point = 45

    while True:
        outside_points = np.where((gray_image < threshold_value_point) & ~segmentation_mask)

        if len(outside_points[0]) == 0:
            break  # No more points below the threshold outside detected contours

        seed_point = (outside_points[0][0], outside_points[1][0])

        new_segmentation_mask = region_growing(gray_image, seed_point)

        segmentation_mask |= new_segmentation_mask

        new_contours = measure.find_contours(new_segmentation_mask, 0.5)

        new_filtered_contours = [contour for contour in new_contours if contour.shape[0] > min_contour_area]

        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.imshow(gray_image, cmap='gray')
        #
        # for contour in filtered_contours + new_filtered_contours:
        #     ax.plot(contour[:, 1], contour[:, 0], '-r', linewidth=2)
        #
        # ax.set_title('Region Growing Contour Segmentation (Filtered + Iterative)')
        # plt.show()
        filtered_contours += new_filtered_contours

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='gray')

    # Remove contours based on average mean intensity threshold
    avg_intensity_threshold = 60  # Adjust this threshold based on your requirements
    filtered_merged_contours = []

    for filtered_contour in filtered_contours:
        contour_coords = np.array(list(zip(filtered_contour[:, 0].astype(int), filtered_contour[:, 1].astype(int))))
        region_mean = np.mean(gray_image[contour_coords[:, 0], contour_coords[:, 1]])

        if region_mean <= avg_intensity_threshold:
            filtered_merged_contours.append(filtered_contour)

    union_area = np.zeros_like(segmentation_mask, dtype=np.bool)
    for contour in filtered_merged_contours:
        union_area |= measure.grid_points_in_poly(segmentation_mask.shape, contour)

    merged_contours = measure.find_contours(union_area, 0.5)
    merged_contours = [contour for contour in merged_contours if contour.shape[0] > min_contour_area]

    for merged_contour in merged_contours:
        ax.plot(merged_contour[:, 1], merged_contour[:, 0], '-r', linewidth=2)

    ax.set_title('Final Segmentation Result')
    plt.savefig(output_path)
    plt.show()
