
# Towards Unsupervised Ink Prediction

This repository contains the first efforts to implement the approach described in [September's report](https://drive.google.com/drive/folders/1S_3BDYpcudAJ2rthNzVqFqGo5GEAgEcr?usp=drive_link). Briefly, the idea here is to use Self-Supervised Learning techniques to train a model on large amounts of unlabeled segment data. If the model learns sufficiently good features, it could be used to detect traces of ink (as shown in a [previous proof-of-concept](https://github.com/jgcarrasco/dino-ink-detection))in another scrolls and serve as a starting point of the iterative labeling approach used in the Grand Prize. The code contained in this repo enables you to:
- Pretrain a ResNet-50 encoder on any segment. It automatically downloads segments and temporarily stores it locally so that it is not required to re-download until the machine shuts downs or reboots.
- Train a UNet with a ResNet backbone to perform ink prediction on a single segment.
- Load the weights of the pretrained encoder to see if the performance improves.

Together with the repo, a report is also included, showing the results of different experiments performed to assess the quality of the learned representations. The idea of this early work is to provide us with a way of evaluating the pretrained encoder so that we are able to improve it until it is able to detect traces of ink in an unsupervised manner.

> **NOTE:** It is advised to [read the report](./report/october-report.pdf) before executing the code below. 

> **NOTE 2:** The pretraining part uses a different version of `timm` than the fine-tuning part. Therefore, to fine tune it is required to upgrade the package to one of the latest version, whereas to pretrain we require `timm==0.6.13`. This will be (hopefully) fixed in the future.

## Pretraining Instructions

**NOTE:** The pretraining part has been heavily borrowed from the [SparK repo](https://github.com/keyu-tian/SparK) and adapted to work with segment data, so make sure to check it out:
```
@Article{tian2023designing,
  author  = {Keyu Tian and Yi Jiang and Qishuai Diao and Chen Lin and Liwei Wang and Zehuan Yuan},
  title   = {Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling},
  journal = {arXiv:2301.03580},
  year    = {2023},
}
```

To perform pretraining, we just have to input the desired arguments in the file `pretrain/utils/arg_util.py`. Most of the arguments should be left, as-is, the ones that have to be changed are:

- `exp_dir`: Directory where the training logs and pretrained encoder will be save.
- `data_path`: ID of the segment to perform pretraining on. It automatically downloads the segment data and stores it temporarily on disk so that re-downloading is not necessary unless we reboot the machine. Right now it justs support training on a single segment at each run, but multi-segment pretraining will be implemented in the future. 
- `resume_from`: Previous runs will generate a `exp_dir/resnet50_withdecoder_1kpretrained_spark_style.pth` file containing all the information until the training is finished or cancelled. If we want to resume a cancelled run, or train it for longer, we can input the path in this argument. If we don't want to resume any previous run, set it to `None`
- `init_weight`: Same as above, but the weights are loaded and the training is initialized from scratch.
- `ep`: Number of epochs. 

Once that above arguments are set up, we can start training by simply running `python pretrain/main.py`.

> **NOTE:** Training can be visualized via Tensorboard, just open the folder `exp_dir/tensorboard_logs/`.

## Fine-tuning instructions

The training is performed by executing `python finetune/train.py`. The following variables specify the training scheme:

- `exp_name`: Name of the experiment. This is just used to differentiate between training runs.
- `segment_id`: ID of the segment to train on.
- `BATCH_SIZE`: Modify according to your GPU power
- `NUM_EPOCHS`: Number of epochs
- `data_augmentation`: Whether to use data augmentation or not. Using augmentation is recommended.
- `freeze_encoder`: Whether to freeze the decoder or nor.
- `pretrained_path`: Path of the pretrained encoder (stored in `exp_log/resnet50_1kpretrained_timm_style.pth`). Set to `None` if we want the encoder to be randomly initialized.

Once that the training run has finished, we will find in `exp_name` the weights of the best model according to the validation loss, the weights of the last model as well as an image of the ink predicted by the best model.

### Download segments using rclone

**NOTE: The steps below are not required anymore, now the segments are automatically downloaded by using the [`vesuvius`](https://github.com/ScrollPrize/vesuvius) library.**

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
- After the setup, you can run `./download_segment.sh SEGMENT_ID` to easily download any segment.





