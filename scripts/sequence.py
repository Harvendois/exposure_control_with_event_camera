import os
import cv2
import h5py
import numpy as np
import torch
import weakref
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from dataset.representations import VoxelGrid  # Ensure this is available
from utils.eventslicer import EventSlicer  # Ensure this is available

def render_rect(x: np.ndarray, y: np.ndarray, pol: np.ndarray, t: np.ndarray, H: int, W: int) -> np.ndarray:
    x = x.squeeze()
    y = y.squeeze()
    pol = pol.squeeze()
    t = t.squeeze()
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img_acc = np.zeros((H, W), dtype='float32').ravel()

    pol = pol.astype('int')
    x0 = x.astype('int')
    y0 = y.astype('int')
    t = t.astype('float64')
    value = 2 * pol - 1

    t_norm = (t - t.min()) / (t.max() - t.min())
    t_norm = t_norm ** 2
    t_norm = t_norm.astype('float32')
    assert t_norm.min() >= 0
    assert t_norm.max() <= 1

    for xlim in [x0, x0 + 1]:
        for ylim in [y0, y0 + 1]:
            mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0)
            interp_weights = value * (1 - np.abs(xlim - x)) * (1 - np.abs(ylim - y)) * t_norm

            index = W * ylim.astype('int') + xlim.astype('int')

            np.add.at(img_acc, index[mask], interp_weights[mask])

    img_acc = np.reshape(img_acc, (H, W))

    img_out = np.full((H, W, 3), fill_value=255, dtype='uint8')

    clip_percentile = 80
    min_percentile = -np.percentile(np.abs(img_acc[img_acc < 0]), clip_percentile)
    max_percentile = np.percentile(np.abs(img_acc[img_acc > 0]), clip_percentile)
    img_acc = np.clip(img_acc, min_percentile, max_percentile)

    img_acc_max = img_acc.max()
    idx_pos = img_acc > 0
    img_acc[idx_pos] = img_acc[idx_pos] / img_acc_max
    val_pos = img_acc[idx_pos]
    img_out[idx_pos] = np.stack((255 - val_pos * 255, 255 - val_pos * 255, np.ones_like(val_pos) * 255), axis=1)

    img_acc_min = img_acc.min()
    idx_neg = img_acc < 0
    img_acc[idx_neg] = img_acc[idx_neg] / img_acc_min
    val_neg = img_acc[idx_neg]
    img_out[idx_neg] = np.stack((np.ones_like(val_neg) * 255, 255 - val_neg * 255, 255 - val_neg * 255), axis=1)
    return img_out

class Sequence(Dataset):
    def __init__(self, seq_path: Path, mode: str='train', delta_t_ms: int=50, num_bins: int=15):
        assert num_bins >= 1
        assert delta_t_ms <= 100, 'Adapt this code if duration is higher than 100 ms'
        assert seq_path.is_dir()

        self.mode = mode
        self.height = 480
        self.width = 640
        self.num_bins = num_bins
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)
        self.locations = ['left', 'right']
        self.delta_t_us = delta_t_ms * 1000
        disp_dir = seq_path / 'disparity'
        assert disp_dir.is_dir()
        self.timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')
        ev_disp_dir = disp_dir / 'event'
        assert ev_disp_dir.is_dir()
        disp_gt_pathstrings = [str(entry) for entry in sorted(ev_disp_dir.iterdir()) if entry.name.endswith('.png')]
        self.disp_gt_pathstrings = disp_gt_pathstrings
        assert len(self.disp_gt_pathstrings) == self.timestamps.size
        assert int(Path(self.disp_gt_pathstrings[0]).stem) == 0
        self.disp_gt_pathstrings.pop(0)
        self.timestamps = self.timestamps[1:]
        self.h5f = {}
        self.rectify_ev_maps = {}
        self.event_slicers = {}
        ev_dir = seq_path / 'events'
        for location in self.locations:
            ev_dir_location = ev_dir / location
            ev_data_file = ev_dir_location / 'events.h5'

            if ev_data_file.is_file():
                print(f"File {ev_data_file} exists.")
            else:
                print(f"File {ev_data_file} does not exist.")

            ev_rect_file = ev_dir_location / 'rectify_map.h5'
            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]

        self.rgb_dirs = {loc: seq_path / 'images' / loc for loc in self.locations}
        for loc, path in self.rgb_dirs.items():
            assert path.is_dir(), f"RGB directory {path} for {loc} does not exist."
        self.rgb_images = {loc: [str(entry) for entry in sorted(path.iterdir()) if entry.name.endswith('.png')] for loc, path in self.rgb_dirs.items()}

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

    def events_to_voxel_grid(self, x, y, p, t, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = t / t[-1]
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32') / 256

    @staticmethod
    def get_rgb_image(filepath: Path):
        assert filepath.is_file()
        img = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    def __len__(self):
        return len(self.disp_gt_pathstrings)

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def __getitem__(self, index):
        ts_end = self.timestamps[index]
        ts_start = ts_end - self.delta_t_us
        print(f"Index: {index}, ts_start: {ts_start}, ts_end: {ts_end}")

        disp_gt_path = Path(self.disp_gt_pathstrings[index])
        file_index = int(disp_gt_path.stem)
        output = {
            'disparity_gt': self.get_disparity_map(disp_gt_path),
            'file_index': file_index,
        }

        rect_events_dict = {}
        for location in self.locations:

            event_data = self.event_slicers[location].get_events(ts_start, ts_end)
            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            xy_rect = self.rectify_events(x,y,location)
            x_rect = xy_rect[:,0]
            y_rect = xy_rect[:,1]

            event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
            if 'representation' not in output:
                output['representation'] = {}
            output['representation'][location] = event_representation

            rect_events_dict[location] = np.array([x_rect, y_rect, t, p])

        return output, rect_events_dict

def save_events_to_txt(filepath: str, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray):
    """
    Saves the event data to a .txt file.

    Args:
        filepath (str): Path to the output text file.
        x (np.ndarray): X-coordinates of events.
        y (np.ndarray): Y-coordinates of events.
        t (np.ndarray): Timestamps of events.
        p (np.ndarray): Polarities of events.
    """
    data = np.vstack((x, y, t, p)).T
    np.savetxt(filepath, data, fmt='%.6f %.6f %.6f %d', header='x y t p', comments='')

def main():
    seq_path = Path("../dsec_dataset_zurich_city_09_b") # edit your sequence path here 

    if not seq_path.is_dir():
        print(f"Error: The directory {seq_path} does not exist.")
        return 0

    dataset = Sequence(seq_path=seq_path, mode='train', delta_t_ms=15, num_bins=15) 
    # this line automatically utilizes the __getitem__ method to extract the data from the dataset, which requires eventslicer.py and representations.py to be available

    for i in range(len(dataset)):
        output, rect_events_dict = dataset[i] # event rectification and voxel grid extraction is done here, provided by DSEC dataset codewriters

        for location in dataset.locations:
            # Extract rectified events
            rectified_events = rect_events_dict[location]
            x_rect, y_rect, t, p = rectified_events

            # Save rectified events to .txt file in the seq_path/txt directory
            if os.path.exists(f'{seq_path}/txt') == False:
                os.mkdir(f'{seq_path}/txt')

            #
            if location == 'left':
                txt_filepath = f'{seq_path}/txt/rectified_events_left_{i}.txt'
                save_events_to_txt(txt_filepath, x_rect, y_rect, t, p)
                print(f'Saved rectified events to {txt_filepath}')

            # Extract voxel grid representation
            voxel_grid = output['representation'][location]

            if os.path.exists(f'{seq_path}/voxel_grid_representations') == False:
                os.mkdir(f'{seq_path}/voxel_grid_representations')

            # Save voxel grid as image
            voxel_grid_image = voxel_grid.sum(dim=0).cpu().numpy()
            voxel_grid_image = (voxel_grid_image - voxel_grid_image.min()) / (voxel_grid_image.max() - voxel_grid_image.min()) * 255
            voxel_grid_image = voxel_grid_image.astype(np.uint8)
            plt.imsave(f'{seq_path}/voxel_grid_representations/voxel_grid_{location}_{i}.png', voxel_grid_image, cmap='gray')

            # Render rectified image using render_rect function
            rectified_image = render_rect(x_rect, y_rect, p, t, dataset.height, dataset.width)

            if os.path.exists(f'{seq_path}/rectified_image') == False:
                os.mkdir(f'{seq_path}/rectified_image')

            # Save rectified image to .png file for visualization 
            rectified_image_path = f'{seq_path}/rectified_image/rectified_image_{location}_{i}.png'
            plt.imsave(rectified_image_path, rectified_image)
            print(f'Saved rectified image to {rectified_image_path}')
            
if __name__ == "__main__":
    main()
