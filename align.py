import os
import argparse
from pathlib import Path

import numpy as np
import cv2
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as Rot

class Transform:
    def __init__(self, translation: np.ndarray, rotation: Rot):
        if translation.ndim > 1:
            self._translation = translation.flatten()
        else:
            self._translation = translation
        assert self._translation.size == 3
        self._rotation = rotation

    @staticmethod
    def from_transform_matrix(transform_matrix: np.ndarray):
        translation = transform_matrix[:3, 3]
        rotation = Rot.from_matrix(transform_matrix[:3, :3])
        return Transform(translation, rotation)

    @staticmethod
    def from_rotation(rotation: Rot):
        return Transform(np.zeros(3), rotation)

    def R_matrix(self):
        return self._rotation.as_matrix()

    def R(self):
        return self._rotation

    def t(self):
        return self._translation

    def T_matrix(self) -> np.ndarray:
        return self._T_matrix_from_tR(self._translation, self._rotation.as_matrix())

    def q(self):
        # returns (x, y, z, w)
        return self._rotation.as_quat()

    def euler(self):
        return self._rotation.as_euler('xyz', degrees=True)

    def __matmul__(self, other):
        # a (self), b (other)
        # returns a @ b
        #
        # R_A | t_A   R_B | t_B   R_A @ R_B | R_A @ t_B + t_A
        # --------- @ --------- = ---------------------------
        # 0   | 1     0   | 1     0         | 1
        #
        rotation = self._rotation * other._rotation
        translation = self._rotation.apply(other._translation) + self._translation
        return Transform(translation, rotation)

    def inverse(self):
        #           R_AB  | A_t_AB
        # T_AB =    ------|-------
        #           0     | 1
        #
        # to be converted to
        #
        #           R_BA  | B_t_BA    R_AB.T | -R_AB.T @ A_t_AB
        # T_BA =    ------|------- =  -------|-----------------
        #           0     | 1         0      | 1
        #
        # This is numerically more stable than matrix inversion of T_AB
        rotation = self._rotation.inv()
        translation = - rotation.apply(self._translation)
        return Transform(translation, rotation)

def load_rectified_events(file_path):
    with np.load(file_path) as data:
        events = [data[key] for key in data]
    return events

def load_rgb_images(file_path):
    with np.load(file_path) as data:
        images = [data[key] for key in data]
    return images

#blue and red event visualization
def main():
    seq_path = Path("../dsec_dataset_zurich_city_09_b")
    assert seq_path.is_dir()
    print(f'start processing: {seq_path}')

    confpath = seq_path / 'calibration' / 'cam_to_cam.yaml'
    assert confpath.exists()
    conf = OmegaConf.load(confpath)

    images_left_dir = seq_path / 'images' / 'left'
    outdir = images_left_dir / 'ev_inf'
    os.makedirs(outdir, exist_ok=True)

    image_in_dir = images_left_dir

    # Get mapping for this sequence:
    K_r0 = np.eye(3)
    K_r0[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect0']['camera_matrix']
    K_r1 = np.eye(3)
    K_r1[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect1']['camera_matrix']

    R_r0_0 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect0']))
    R_r1_1 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect1']))

    T_r0_0 = Transform.from_rotation(R_r0_0)
    T_r1_1 = Transform.from_rotation(R_r1_1)
    T_1_0 = Transform.from_transform_matrix(np.array(conf['extrinsics']['T_10']))

    T_r1_r0 = T_r1_1 @ T_1_0 @ T_r0_0.inverse()
    R_r1_r0_matrix = T_r1_r0.R().as_matrix()
    P_r1_r0 = K_r1 @ R_r1_r0_matrix @ np.linalg.inv(K_r0)

    ht = 480
    wd = 640
    # coords: ht, wd, 2
    coords = np.stack(np.meshgrid(np.arange(wd), np.arange(ht)), axis=-1)
    # coords_hom: ht, wd, 3
    coords_hom = np.concatenate((coords, np.ones((ht, wd, 1))), axis=-1)
    # mapping: ht, wd, 3
    mapping = (P_r1_r0 @ coords_hom[..., None]).squeeze()
    # mapping: ht, wd, 2
    mapping = (mapping/mapping[..., -1][..., None])[..., :2]
    mapping = mapping.astype('float32')

    event_index = 0

    # Ensure we only process as many images as there are events
    for i, entry in enumerate(image_in_dir.iterdir()):
        # if event_index >= len(rectified_events_left):
        #     print(f"Processed all available events at index {event_index}.")
        #     break
        
        if entry.suffix == '.png':
            if i % 2 != 0:  # skip every other image 
                continue

            image_out_file = outdir / entry.name
            if image_out_file.exists():
                event_index += 1
                continue

            image_in = cv2.imread(str(entry))

            # Warp the image using the mapping
            image_out = cv2.remap(image_in, mapping, None, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(image_out_file), image_out)

            # call events from rectified events
            events_path = f"{seq_path}/txt/rectified_events_left_{int(i/2)}.txt"
            events = np.loadtxt(events_path, delimiter=' ', dtype=np.float32, skiprows=1).T
            print(f'Loaded {events.shape[1]} events from {events_path}')
            
            # Visualize events on the warped RGB image
            vis_image = image_out.copy()
            x, y, t, p = events
            for j in range(x.size):
                color = (255, 0, 0) if p[j] == 1 else (0, 0, 255)  # Blue for polarity 1, Red for polarity -1
                cv2.circle(vis_image, (int(x[j]), int(y[j])), 1, color, -1)

            vis_out_file = outdir / f'events_on_image_{event_index}.png'
            cv2.imwrite(str(vis_out_file), vis_image)
            print(f'Saved visualized events to {vis_out_file}')

            event_index += 1

    print(f'done processing: {seq_path}')

if __name__ == "__main__":
    main()




