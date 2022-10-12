# MNAD
This repo is merely a cleaned-up version of [https://github.com/cvlab-yonsei/MNAD](https://github.com/cvlab-yonsei/MNAD),
please refer to it and cite it directly.

## Train/test

Train with e.g.
```bash
train.py --task prediction --train_path=<path to train videos>
```

Test with 
```bash
evaluate.py --task prediction --test_path=<path to test videos> --model_dir=<model directory> --m_items_dir=<memory items directory> 
```

## Data preparation
Data folder is expected to be of this form
```
<data root folder>
|- <video a>
|  |- labels.npy
|  |- 0000.jpg
|  |- 0001.jpg
|  |- ...
|- <video b>
|  |- labels.npy
|  |- 0000.jpg
|  |- 0001.jpg
|  |- ...  
| ...
```
`labels.npy` is a flat numpy int8 array of 0's and 1's, of the same length as the video, with 1's corresponding to anomalous frames.


