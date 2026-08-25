import os
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import pandas as pd

LOGGER = logging.getLogger("pefim_utils")
MIN_NUM_FRAMES = 3
DISTANCE_THRESHOLD = 17
POS_THRESH = 0.99
NEG_THRESH = 0.5


def get_stats(data, cluster_results_directory, all_data=True, save_stats=False):
    """
    Computes statistics on the confidence data, generates histograms, and logs information.

    Args:
        data (np.ndarray): Array of confidence values.
        cluster_results_directory (str): Directory to save results.
        all_data (bool, optional): Flag indicating whether to consider all data. Defaults to True.
        save_stats (bool, optional): Flag indicating whether to save histograms and logs. Defaults to False.

    Returns:
        int: The number of data points above the upper quartile.
    """
    name_flag = "TopK" if not all_data else "all"
    if save_stats:
        os.makedirs(f"{cluster_results_directory}", exist_ok=True)
        plt.hist(data, bins=10, alpha=0.65, color="blue", edgecolor="black")
        plt.savefig(f"{cluster_results_directory}/hist_{name_flag}.jpg")
        # plt.savefig(f'testhist{name_flag}.jpg')
        plt.close()
    pos_class_data = data[data > POS_THRESH]
    neg_class_data = data[data < NEG_THRESH]
    upper_quartile = np.percentile(data, 75)
    up_quat_data = data[data > upper_quartile]

    LOGGER.warning(
        f"{name_flag} number of positive samples above threshold {POS_THRESH} is {len(pos_class_data)} saved in {cluster_results_directory}"
    )
    LOGGER.warning(
        f"{name_flag} number of negative samples below threshold {NEG_THRESH} is {len(neg_class_data)} saved in {cluster_results_directory}"
    )
    if all_data:
        LOGGER.warning(
            f"{name_flag} number of positive samples above upper quartile {upper_quartile} is {len(up_quat_data)} saved in {cluster_results_directory}"
        )
        LOGGER.warning(
            f"{name_flag} number of 1/3rd topk frames is {int(len(data)*(1/3))}"
        )
    if save_stats:
        log_path = f"{cluster_results_directory}/log_{name_flag}.txt"
        with open(log_path, "w") as log_file:
            log_file.write(
                f"{name_flag} number of positive samples above threshold {POS_THRESH} is {len(pos_class_data)} saved in {cluster_results_directory}\n"
            )
            log_file.write(
                f"{name_flag} number of negative samples below threshold {NEG_THRESH} is {len(neg_class_data)} saved in {cluster_results_directory}\n"
            )
            if all_data:
                log_file.write(
                    f"{name_flag} number of positive samples above upper quartile {upper_quartile} is {len(up_quat_data)} saved in {cluster_results_directory}\n"
                )
                log_file.write(
                    f"{name_flag} number of 1/3rd topk frames is {int(len(data)*(1/3))}\n"
                )

        LOGGER.debug(f"Log written to {log_path}")
    return len(up_quat_data)  # int(len(data) * (1/3))


def get_topk_frames(
    data, cluster_results_directory, min_num_frames=500, save_stats=True
):
    """
    Retrieves the top-k frames based on confidence scores.

    Args:
        data (list): List of dictionaries containing frame data.
        cluster_results_directory (str): Directory to save results.
        min_num_frames (int, optional): Minimum number of frames required. Defaults to 500.
        save_stats (bool, optional): Flag indicating whether to save statistics. Defaults to True.

    Returns:
        list: List of top-k frame dictionaries.
    """
    topk_frame_info = data
    topk_frame_info.sort(key=lambda x: x["confidences"], reverse=True)
    confidence_data = np.stack(
        [topk_frame_info[i]["confidences"] for i in range(len(topk_frame_info))]
    )
    upper_quartile_len = get_stats(
        confidence_data, cluster_results_directory, all_data=True, save_stats=save_stats
    )
    # num_frames = int(len(data) * (1/3))
    num_frames = upper_quartile_len

    if num_frames < min_num_frames:
        # TODO - Remove videos with low frames
        # raise ValueError(f"The number of frames picked by the upper quartile is lesser than the min frames required {len(topk_frame_info)}")
        LOGGER.debug(
            f"The number of frames picked by the upper quartile is lesser than the min frames required {len(topk_frame_info)}"
        )

    # num_frame = min(len(topk_frame_info), NUM_FRAMES)

    LOGGER.debug(f"Total number of top frames in upper quartiles is {num_frames}")
    topk_frames = topk_frame_info[:num_frames]
    # write the code to plot a histogram of confidences
    # if save_stats:
    #    confidence_data = np.stack([topk_frames[i]['confidences'] for i in range(len(topk_frames))])
    #    get_stats(confidence_data, cluster_results_directory, all_data=False)
    return topk_frames


def create_image_grid(image_paths, grid_size=None, texts=None):
    """
    Create a grid of images given a list of image paths.

    :param image_paths: List of paths to images.
    :param grid_size: Tuple of ints (rows, cols). If None, a square grid will be used.
    :return: Concatenated image object.
    """
    # Load images
    images = [Image.open(path).resize((250, 250)) for path in image_paths]

    # If grid_size is not specified, create a square grid
    if grid_size is None:
        grid_size = (math.ceil(math.sqrt(len(images))),) * 2  # (rows, cols)

    if texts and len(texts) != len(images):
        raise ValueError("The number of texts must match the number of images.")

    # Calculate the size of the grid
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)
    total_width = max_width * grid_size[1]
    total_height = max_height * grid_size[0]

    # Create a new blank image with a white background
    grid_image = Image.new("RGB", (total_width, total_height), color="white")

    # Paste images onto the grid_image
    for index, image in enumerate(images):
        # Calculate image position in grid
        row = index // grid_size[1]
        col = index % grid_size[1]
        box = (col * max_width, row * max_height)

        if texts:
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default(size=30)
            text = texts[index]
            draw.text(
                (image.width // 2, image.height // 2), str(text), font=font, fill="cyan"
            )
        grid_image.paste(image, box)

    return grid_image


def numerical_sort(filename: str) -> int:
    """Extract and return numerical digits from a string.

    This function helps sort the file names numerically as they are returned unordered by the multi-processing step.

    Args:
        filename (str): Input string containing numerical digits.

    Returns:
        int: Extracted numerical value.
    """
    return int("".join(filter(str.isdigit, str(filename))))


def extract_framenumber(filename: str) -> int:
    """
    extracts the framenumber from the file path
    Args:
        filename (str): path to frame number.

    Returns:
        int: Extracted franme number.
    """
    return int(Path(filename).stem)


def get_median(cluster_features):
    """
    Returns the index of the feature closest to the median of the cluster features.

    Args:
        cluster_features (array-like): A collection of feature vectors within a cluster.

    Returns:
        int: The index of the feature vector closest to the median.
    """
    idx = np.argsort(
        np.linalg.norm(cluster_features - np.median(cluster_features, axis=0), axis=1)
    )[0]
    return idx


def get_unique_features_set(topk_frames):
    """
    Extracts unique feature clusters and their indices from the top K frames.

    Args:
        topk_frames (list): A list of dictionaries, each containing:
            - 'image_paths' (str): The image path.
            - 'feature_data' (array-like): The feature data associated with the image.

    Returns:
        tuple:
            indices (list): Indices of the unique features.
            clusters (list): Cluster labels for each feature.

    """
    # image_paths_for_clustering = [frame["image_paths"] for frame in topk_frames]
    features_for_clustering = [frame["feature_data"] for frame in topk_frames]
    features_np = np.array(features_for_clustering)
    clusters, indices = find_unique_features(features_np, DISTANCE_THRESHOLD)
    return indices, clusters


def find_unique_features(features, distance_threshold):
    """
    Clusters feature vectors based on a distance threshold and returns the clusters and their indices.

    Args:
        features (array-like): A collection of feature vectors.
        distance_threshold (float): The maximum distance between features to be considered in the same cluster.

    Returns:
        tuple: A tuple containing:
            - clusters (list): A list of clusters, each containing feature vectors.
            - indexes (list): A list of indices indicating the cluster assignment for each feature.
    """
    clusters = []
    indexes = []
    cluster_centroid = []
    cluster_centroid_indices = []
    # N = features.shape[0]

    for i, f in enumerate(features):
        D = [np.linalg.norm(c - f) for c in cluster_centroid]
        if len(D) and min(D) < distance_threshold:
            min_d = np.argmin(D)
            clusters[min_d].append(f)
            median_feature_idx = get_median(clusters[min_d])
            cluster_centroid[min_d] = clusters[min_d][median_feature_idx]
            cluster_centroid_indices[min_d] = median_feature_idx
            indexes.append(min_d)
        else:
            clusters.append([f])
            cluster_centroid.append(f)
            cluster_centroid_indices.append(i)
            indexes.append(len(clusters) - 1)

    return clusters, indexes


def computer_clusters_from_unique_set(
    cluster_results_directory, indices, clusters, topk_frames, save_data
):
    """
    Processes clusters from a set of unique features, generates cluster images,
    and optionally saves the cluster data.

    Args:
        cluster_results_directory (str): Directory to save cluster results.
        indices (np.ndarray): Array of cluster indices corresponding to each feature.
        clusters (list): List of clusters containing feature vectors.
        topk_frames (list): List of dictionaries containing frame data.
        save_data (bool): Flag indicating whether to save cluster images and data.

    Returns:
        Optional[dict]: A dictionary containing cluster data if clusters are found; otherwise, None.
    """
    # num_clusters = len(set(indices)) - (1 if -1 in indices else 0)
    indices = np.array(indices)
    cluster_data = {}
    dropped_cluster = 0
    for cluster_id in set(indices):
        if cluster_id == -1:
            continue

        cluster_indices = np.where(indices == cluster_id)[0]
        if len(cluster_indices) < MIN_NUM_FRAMES:
            dropped_cluster += 1
            continue
        cluster_image_paths = [topk_frames[i]["image_paths"] for i in cluster_indices]
        cluster_labels = [topk_frames[i]["labels_multicls"] for i in cluster_indices]
        cluster_features = [topk_frames[i]["feature_data"] for i in cluster_indices]
        cluster_confidences = [topk_frames[i]["confidences"] for i in cluster_indices]

        median_feature_idx = np.argsort(
            np.linalg.norm(
                cluster_features - np.median(cluster_features, axis=0), axis=1
            )
        )[0]
        median_feature = cluster_features[median_feature_idx]
        median_image_path = cluster_image_paths[median_feature_idx]

        if save_data:
            image_grid = create_image_grid(
                cluster_image_paths, texts=cluster_confidences
            )
            os.makedirs(f"{cluster_results_directory}/clusters/", exist_ok=True)
            image_grid.save(
                f"{cluster_results_directory}/clusters/cluster_{cluster_id}.jpg", "JPEG"
            )

        cluster_data[cluster_id] = {
            "image_paths": cluster_image_paths,
            "labels": cluster_labels,
            "features": cluster_features,
            "confidences": cluster_confidences,
            "median_feature": median_feature,
            "median_image_path": median_image_path,
        }

    if dropped_cluster == len(set(indices)):
        LOGGER.warning("nothing in this cluster")
        return None
    LOGGER.warning(
        f"Dropped clusters are {dropped_cluster} and the orginal cluster no is {len(set(indices))} and remaining cluster no is {len(set(indices))-dropped_cluster}"
    )
    cluster_centroid_image_list = [
        cluster_data[i]["median_image_path"] for i in cluster_data.keys()
    ]
    cluster_ids_list = [idx for idx in cluster_data.keys()]
    if save_data:
        centroid_image_grid = create_image_grid(
            cluster_centroid_image_list, texts=cluster_ids_list
        )
        centroid_image_grid.save(
            f"{cluster_results_directory}/cluster_{MIN_NUM_FRAMES}_{DISTANCE_THRESHOLD}.jpg",
            "JPEG",
        )
        cluster_df = pd.DataFrame(cluster_data)
        cluster_df.to_pickle(f"{cluster_results_directory}/cluster_info.pkl")

    return cluster_data
