import concurrent.futures
from collections import Counter
from pathlib import Path

import cv2 as cv
import dlib
import numpy as np

dataset_base = Path('dataset/celeba/')
dataset_path = dataset_base / 'img_align_celeba/'
identity_info_path = dataset_base / 'identity_CelebA.txt'
original_base = dataset_base / 'original/'
landmark_base = dataset_base / 'landmark/'
mask_base = dataset_base / 'mask/'

predictor_path = 'lib/shape_predictor_68_face_landmarks.dat'

FACIAL_OUTLINE_INDICES = list(range(17)) + list(range(26, 16, -1))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def convert_and_save(filename, identity):
    original_path = (original_base / identity / filename).with_suffix('.png')
    landmark_path = (landmark_base / identity / filename).with_suffix('.png')
    mask_path = (mask_base / identity / filename).with_suffix('.png')

    # Read an image
    img_path = dataset_path / filename
    img = cv.imread(str(img_path))
    h, w = img.shape[:2]
    dets = detector(img[..., ::-1], 1)

    if len(dets) <= 0:
        print(f'{filename}: Face not detected!')
        return
    elif len(dets) != 1:
        print(f'{filename}: More than one face detected!')
        return

    face = dets[0]
    landmarks = predictor(img[..., ::-1], face)

    # Draw facial landmarks
    landmark_img = np.zeros((h, w, 1), np.uint8)

    jaw = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)], np.int32)
    nose_bridge = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 31)], np.int32)
    cv.polylines(landmark_img, [jaw, nose_bridge], False, (255,), 1, cv.LINE_AA)

    mouth_outline = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)], np.int32)
    cv.polylines(landmark_img, [mouth_outline], True, (255,), 1, cv.LINE_AA)

    # Draw facial mask
    mask_img = np.full((h, w, 1), 255, np.uint8)
    facial_outline = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in FACIAL_OUTLINE_INDICES], np.int32)
    cv.fillPoly(mask_img, [facial_outline], (0,), cv.LINE_AA)

    if not original_path.parent.is_dir():
        original_path.parent.mkdir(parents=True)
    if not landmark_path.parent.is_dir():
        landmark_path.parent.mkdir(parents=True)
    if not mask_path.parent.is_dir():
        mask_path.parent.mkdir(parents=True)

    # Save the original image
    cv.imwrite(str(original_path), img)
    # Save the landmark image
    cv.imwrite(str(landmark_path), landmark_img)
    # Save the mask
    cv.imwrite(str(mask_path), mask_img)


def count_identities():
    identity_counter = {}

    for identity in original_base.iterdir():
        if not identity.is_dir():
            continue
        identity_counter[identity.name] = len(list(identity.glob('*.png')))

    print(f'total identities = {len(identity_counter)}')
    print(f'total images = {sum(identity_counter.values())}')

    c = Counter(identity_counter.values())
    for k in sorted(c.keys()):
        print(f'{k} -> {c[k]}')

    return identity_counter


def filter_identities(thresh: int):
    """Filter out identities that the total number of whose faces are scarce."""
    identity_counter = count_identities()
    filtered_identities = [int(identity) for identity, count in identity_counter.items() if count >= thresh]
    filtered_identities.sort()
    return filtered_identities


if __name__ == '__main__':
    identity_info = identity_info_path.read_text()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(convert_and_save, filename, identity): filename
                   for filename, identity in map(lambda l: l.split(), identity_info.splitlines())}
        for future in concurrent.futures.as_completed(futures):
            pass

    # Collect identities containing at least over 30 images
    filtered_identities = filter_identities(30)
    filtered_identities = np.array(filtered_identities, np.int32)
    np.save('identities.npy', filtered_identities)
