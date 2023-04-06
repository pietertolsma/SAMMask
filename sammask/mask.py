import cv2
import numpy as np
import os

from scipy.ndimage import morphology, label, measurements


def generate_points(width, height):
    column = np.array([height // 3, height // 2, height // 3 + height // 2])
    x_coords = np.array([width // 10, 9 * width // 10])

    X, Y = np.meshgrid(x_coords, column)
    back_coords = np.vstack((X.flatten(), Y.flatten())).T

    types = np.ones(back_coords.shape[0])

    return back_coords, types


def predict(predictor, img_path, scale=10):
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
    distances[component_sizes < 50] = float("inf")
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

    # mask = mask[:, :, None]
    mask = mask[:, :, None]
    image = original_image * mask
    cv2.imwrite(f"./output_data/{fname}", image)
