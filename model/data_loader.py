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
        self.transform = transform
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred

        self.videos = self.load_videos(video_folder)
        self.seqs = self.get_seqs()

    @staticmethod
    def load_videos(directory):
        videos_root_folder = glob.glob(os.path.join(directory, "*"))
        videos = {}
        for video in sorted(videos_root_folder):
            video_name = video.split("/")[-1]
            frames = sorted(glob.glob(os.path.join(video, "*.jpg")))
            videos[video_name] = {
                "path": video,
                "frames": frames,
                "length": len(frames)
            }
        return videos

    def get_seqs(self):
        seqs = []
        seq_len = self._time_step + self._num_pred
        for video in self.videos.values():
            seqs.extend(
                video["frames"][i:i+seq_len] for i in range(video["length"] - seq_len + 1)
            )
        return seqs

    def __getitem__(self, index):
        seq = self.seqs[index]
        frames = [np_load_frame(frame, self._resize_height, self._resize_width) for frame in seq]
        if self.transform is not None:
            frames = [self.transform(frame) for frame in frames]
        return np.concatenate(frames, axis=0)

    def __len__(self):
        return len(self.seqs)
