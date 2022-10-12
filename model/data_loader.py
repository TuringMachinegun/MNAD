from pathlib import Path

import cv2
import numpy as np
import torch.utils.data as data

LABELS_FILENAME = "labels.npy"


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

        self.videos = self.load_videos(Path(video_folder))
        self.seq_idx_originating_video, self.seqs = self.get_seqs()

    @staticmethod
    def load_videos(root_dir: Path):
        videos = {}
        for video_dir in sorted(root_dir.glob("*")):
            video_name = video_dir.stem
            frames = [str(x) for x in sorted(video_dir.glob("*.jpg"))]
            labels = np.load(str(video_dir / LABELS_FILENAME)) if (video_dir / LABELS_FILENAME).exists() else None
            if labels is not None:
                assert labels.size == len(frames)
            videos[video_name] = {
                "path": video_dir,
                "frames": frames,
                "length": len(frames),
                "labels": labels,
            }
        return videos

    def get_seqs(self):
        seqs = []
        seq_idx_originating_video = []
        seq_len = self._time_step + self._num_pred
        for video_name, video in self.videos.items():
            video_seq = [video["frames"][i:i + seq_len] for i in range(video["length"] - seq_len + 1)]
            seqs.extend(video_seq)
            seq_idx_originating_video.extend([video_name] * len(video_seq))
        return seq_idx_originating_video, seqs

    def get_labels(self):
        labels = []
        for video in self.videos.values():
            labels.append(video["labels"][self._time_step:])
        return np.concatenate(labels)

    def __getitem__(self, index):
        seq = self.seqs[index]
        frames = [np_load_frame(frame, self._resize_height, self._resize_width) for frame in seq]
        if self.transform is not None:
            frames = [self.transform(frame) for frame in frames]
        return np.concatenate(frames, axis=0)

    def __len__(self):
        return len(self.seqs)
