import cv2
import numpy as np
import os

from scipy.ndimage import morphology, label, measurements
from segment_anything import SamAutomaticMaskGenerator


def save_img(image, fname, points=None):
    if points:
        for x, y in list(points):
            cv2.drawMarker(image, (x, y), (0, 0, 255), cv2.MARKER_STAR, 10)

    cv2.imwrite(fname, image)


def generate_points(width, height):
    # rows = np.array([height // 3, height // 2, height // 3 + height // 2])
    rows = np.arange(start=0, stop=height + 1, step=height // 10)
    x_coords = np.array([width // 10, 9 * width // 10])

    X, Y = np.meshgrid(x_coords, rows)
    back_coords = np.vstack((X.flatten(), Y.flatten())).T

    # column = np.array([width // 3, width // 2, width // 3 + width // 2])
    column = np.arange(start=0, stop=width + 1, step=width // 10)
    y_coords = np.array([5, height - 5])

    X, Y = np.meshgrid(column, y_coords)
    back_coords_2 = np.vstack((X.flatten(), Y.flatten())).T

    coords = np.concatenate((back_coords, back_coords_2))

    types = np.ones(coords.shape[0])

    return coords, types


def predict_2(generator, img_path):
    fname = img_path.split("/")[-1]

    if os.path.exists(f"output_data/{fname}"):
        print(f"Skipping {fname}, already exists in output dir...")
        return

    original_image = cv2.imread(img_path)[:, :, [2, 0, 1]]
    image = original_image

    # generator.set_image(
    #     image,
    #     image_format="BGR",
    # )

    input_point, input_label = generate_points(image.shape[1], image.shape[0])

    masks = generator.generate(image)

    fmask = np.zeros((image.shape[0], image.shape[1]))

    for ann in masks:
        m = ann["segmentation"]
        (bx, by, w, h) = ann["bbox"]

        matched = False
        for x, y in list(input_point):
            if bx <= x and x <= (bx + w) and by <= y and y <= (by + h):
                matched = True
                break
        if not matched:
            fmask[m] = 1
    mask = fmask[:, :, None]
    image = original_image[:, :, [1, 2, 0]] * mask
    save_img(image, f"./output_data/{fname}", input_point)


def predict(predictor, img_path, draw_points=False):
    fname = img_path.split("/")[-1]

    if os.path.exists(f"output_data/{fname}"):
        print(f"Skipping {fname}, already exists in output dir...")
        return

    original_image = cv2.imread(img_path)[:, :, :]
    image = original_image

    predictor.set_image(
        image,
        image_format="BGR",
    )

    import numpy as np

    input_point = np.array([[100, 100]])
    input_label = np.array([0])  # 1 = foreground, 0 = background

    input_point, input_label = generate_points(image.shape[1], image.shape[0])

    masks, scores, logits = predictor.predict(
        point_coords=input_point, point_labels=input_label
    )

    mask = ~masks[-1]

    mask = morphology.binary_opening(mask, iterations=2)
    labeled, num_components = label(mask)
    centroids = measurements.center_of_mass(
        mask, labeled, index=np.arange(num_components) + 1
    )
    component_sizes = measurements.sum(
        mask, labeled, index=np.arange(num_components) + 1
    )
    image_center = np.array(mask.shape) / 2
    distances = np.linalg.norm(centroids - image_center, axis=1)
    distances[component_sizes < 100] = float("inf")
    closest_component_index = np.argmin(distances) + 1

    # Create a new binary mask that contains only the closest component
    closest_component_mask = np.zeros_like(mask)
    closest_component_mask[labeled == closest_component_index] = 1

    mask = closest_component_mask.astype(np.uint8)

    # Define the size of the structuring element for morphological operations
    ksize = 75

    # Perform morphological opening to remove small islands in the visible part of the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = mask[:, :, None]
    rgba = np.ones((image.shape[0], image.shape[1], 4), dtype=np.uint8) * 255
    rgba[:, :, :3] = image
    rgba = rgba * mask

    # Save as png
    fname = "".join(fname.split(".")[:-1]) + ".png"

    #image = cv2.bitwise_and(original_image, mask)
    if not draw_points:
        save_img(rgba, f"./output_data/{fname}")
        return
    
    save_img(rgba, f"./output_data/{fname}", input_point)
