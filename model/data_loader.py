import numpy as np
from collections import defaultdict
import os
import glob
import cv2
import torch.utils.data as data


rng = np.random.RandomState(2020)


def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class DataSet(data.Dataset):
    def __init__(
        self,
        video_folder,
        transform,
        resize_height,
        resize_width,
        time_step=4,
        num_pred=1,
    ):
        self.dir = video_folder
        self.transform = transform
        self.videos = defaultdict(dict)
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, "*"))
        for video in sorted(videos):
            video_name = video.split("/")[-1]
            self.videos[video_name]["path"] = video
            self.videos[video_name]["frames"] = glob.glob(os.path.join(video, "*.jpg"))
            self.videos[video_name]["frames"].sort()
            self.videos[video_name]["length"] = len(self.videos[video_name]["frames"])

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, "*"))
        for video in sorted(videos):
            video_name = video.split("/")[-1]
            frames.extend(self.videos[video_name]["frames"])
        return frames

    def __getitem__(self, index):
        video_name = self.samples[index].split("/")[-2]
        frame_idx = int(self.samples[index].split("/")[-1].split(".")[-2])

        batch = []
        for i in range(self._time_step + self._num_pred):
            try:
                image = np_load_frame(
                    # TODO: this assumes that the number of frames is divisible by time step
                    self.videos[video_name]["frames"][frame_idx + i],
                    self._resize_height,
                    self._resize_width,
                )
            except Exception as e:
                print(frame_idx, i, len(self.videos[video_name]["frames"]))
                raise e
            if self.transform is not None:
                batch.append(self.transform(image))

        return np.concatenate(batch, axis=0)

    def __len__(self):
        return len(self.samples) - self._time_step - self._num_pred
