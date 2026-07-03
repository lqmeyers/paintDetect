"""Pseudo pose-estimation: derive a bee's body orientation from segmentation.

Ported from the repo-root ``bee_angle.py``. The original ran a hardcoded batch
job over one user's directory *at import time*; that has been moved into
:func:`batch_process`, so importing this module is side-effect free.

Requires OpenCV (``pip install -e .[analysis]``).
"""

import math
import os

import numpy as np
from PIL import Image, ImageDraw
import cv2

from .segmentation import Segmentation, trained_models


def overlay_masks(mask1, mask2):
    """Overlay two grayscale binary masks (255=fg) into one 'L' image."""
    if mask1.size != mask2.size:
        raise ValueError("Masks must have the same size")

    width, height = mask1.size
    result = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(result)

    for x in range(width):
        for y in range(height):
            if mask1.getpixel((x, y)) == 255:
                draw.point((x, y), 255)
            if mask2.getpixel((x, y)) == 255:
                draw.point((x, y), 200)

    return result


def get_rot_rect(input):
    overlay_result = np.array(input)
    _, thresh = cv2.threshold(overlay_result, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 1:
        combined_contour = np.concatenate(contours)
    else:
        combined_contour = contours[0]

    return cv2.minAreaRect(combined_contour)


def get_centerline_endpoints(center1, center2, rotated_rect):
    center1 = np.array(center1)
    center2 = np.array(center2)

    centerline_direction = center2 - center1
    centerline_direction = centerline_direction / np.linalg.norm(centerline_direction)

    box = cv2.boxPoints(rotated_rect)
    rect_side_vectors = [box[i] - box[(i + 1) % 4] for i in range(4)]
    rect_side_lengths = [np.linalg.norm(side_vector) for side_vector in rect_side_vectors]
    longest_side_idx = np.argmax(rect_side_lengths)

    side_vector1 = box[longest_side_idx] - box[(longest_side_idx + 1) % 4]
    side_vector2 = box[(longest_side_idx + 2) % 4] - box[(longest_side_idx + 3) % 4]

    if np.dot(centerline_direction, side_vector1) > np.dot(centerline_direction, side_vector2):
        centerline_angle = np.arctan2(side_vector1[1], side_vector1[0])
    else:
        centerline_angle = np.arctan2(side_vector2[1], side_vector2[0])

    half_length = 0.5 * rect_side_lengths[longest_side_idx]
    endpoint1 = rotated_rect[0] + half_length * np.array([np.cos(centerline_angle), np.sin(centerline_angle)])
    endpoint2 = rotated_rect[0] - half_length * np.array([np.cos(centerline_angle), np.sin(centerline_angle)])

    return endpoint1, endpoint2


def pythagMe(coord1, coord2):
    """Euclidean distance between two [x, y] points."""
    x1, y1 = coord1[0], coord1[1]
    x2, y2 = coord2[0], coord2[1]
    return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def order_endpoints(h_center, endpoints):
    """Return endpoints ordered so the first is closest to the head center."""
    dist1 = pythagMe(h_center, endpoints[0])
    dist2 = pythagMe(h_center, endpoints[1])
    if dist1 >= dist2:
        return (endpoints[1], endpoints[0])
    return endpoints


def get_angle(image_path, save_vis=False, outpath='./', model_lib=None):
    """Pseudo pose-estimation from head+thorax U-Net segmentations.

    Predicts head/thorax masks, fits a rotated rectangle around them, and returns
    the centerline endpoints with the head-side endpoint first.

    Args:
        image_path: path to the input image.
        save_vis: if True, write a ``*.line_vis.png`` visualization to ``outpath``.
        outpath: directory for the visualization.
        model_lib: dict of part-model paths (defaults to the module registry,
            which must be filled in for your machine).

    Returns:
        (endpoint1, endpoint2) with the head-side endpoint first.
    """
    if model_lib is None:
        model_lib = trained_models

    img = Image.open(image_path)

    seg = Segmentation(model_lib, img)
    head = seg.head
    thorax = seg.thorax

    overlay_result = overlay_masks(head, thorax)

    rotated_rect = get_rot_rect(overlay_result)
    h_rect = get_rot_rect(head)
    t_rect = get_rot_rect(thorax)

    h_cent = h_rect[0]
    t_cent = t_rect[0]

    pose_line = get_centerline_endpoints(h_cent, t_cent, rotated_rect)
    pose_line = order_endpoints(h_cent, pose_line)

    if save_vis:
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.line(img_np, np.int0(pose_line[0]), np.int0(pose_line[1]), (0, 0, 255), 2)
        kp = [cv2.KeyPoint(x=int(pose_line[0][0]), y=int(pose_line[0][1]), size=4)]
        cv2.drawKeypoints(img_np, kp, img_np, (0, 255, 0))
        cv2.imwrite(outpath + os.path.basename(image_path)[:-4] + '.line_vis.png', img_np)

    return pose_line


def batch_process(image_dir, out_dir, every=100, model_lib=None):
    """Run :func:`get_angle` over every ``every``-th image in ``image_dir``.

    This was the import-time loop in the original ``bee_angle.py``; it now only
    runs when explicitly called.
    """
    for root, dirs, files in os.walk(image_dir):
        files.sort()
        for i, f in enumerate(files):
            if i % every == 0 and f[-4:] in ['.jpg', '.png']:
                get_angle(os.path.join(root, f), True, out_dir, model_lib=model_lib)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Bee pose-estimation line from segmentation')
    parser.add_argument('image', help='input image path')
    parser.add_argument('-o', '--outpath', default='./', help='dir for line visualization')
    parser.add_argument('--no-vis', action='store_true', help='do not save a visualization')
    args = parser.parse_args()
    print(get_angle(args.image, save_vis=not args.no_vis, outpath=args.outpath))
