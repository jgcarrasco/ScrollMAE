
# Towards Unsupervised Ink Detection

**NOTE:** This repo heavily borrows from the [SparK repo](https://github.com/keyu-tian/SparK) so make sure to check it out
```
@Article{tian2023designing,
  author  = {Keyu Tian and Yi Jiang and Qishuai Diao and Chen Lin and Liwei Wang and Zehuan Yuan},
  title   = {Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling},
  journal = {arXiv:2301.03580},
  year    = {2023},
}
```

## Introduction



## TO DO
- [ ] Write report
- [ ] Evaluate the encoder vs. randomly initialized
  - [ ] Just the encoder, or take encoder/decoder?
  - [ ] Freeze the encoder or fine-tune?
- [ ] Train on multiple segments of the Grand Prize
- [ ] Put the number of depth layers as an argument (i.e. `in_channels`)

## How to use

### Download segments using rclone

The first that you have to do is to download segment data from the vesuvius server. You can download any segment that you want. It is recommended to use `rclone` for faster download speeds.

- Install `rclone`
- `rclone config`, then create a new remote:
    - Type n for a new remote.
    - Choose a name, e.g., `scrolls_remote`.
    - Select `13` for http connection.
    - When asked for URL of http host to connect to, provide the URL prefix https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/.
    - You can leave other options at their default values.
    - Exit the configuration menu once you've created the remote.
- Now we can copy segments as follows: `rclone copy scrolls_remote:20230519195952.zarr/ ./20230519195952.zarr/ --progress --multi-thread-streams=32 --transfers=32 --size-only` where `20230519195952` can be replaced by any segment ID that you want to download.
- After the setup, you can run `.download_segment.sh SEGMENT_ID` to easily download any segment.

### Pretrain

The pretraining arguments can easily be specified in `pretrain/utils/arg_util.py`. The most important ones are:

- `exp_name`: We just set it to `debug` to indicate that we are just using 1 GPU. The original SparK repo distributes the training across multiple GPUs.
- `exp_dir`: The directory where the logs/checkpoints/tensorboard info will be saved.
- `data_path`: The path to the `.zarr` of the segment. TODO: Implement multisegment
- `dataset_type`: We will set it to `segment` to perform pretraining on segment data.

Then, you can just do:
```
python pretrain/main.py
```

## Installation & Running

- **Loading pretrained model weights in 3 lines**
```python3
# download our weights `resnet50_1kpretrained_timm_style.pth` first
import torch, timm
res50, state = timm.create_model('resnet50'), torch.load('resnet50_1kpretrained_timm_style.pth', 'cpu')
res50.load_state_dict(state.get('module', state), strict=False)     # just in case the model weights are actually saved in state['module']
```





