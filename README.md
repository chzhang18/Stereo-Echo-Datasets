Stereo-Echo Dataset for Stereo Depth Estimation with Echoes (ECCV 2022)

## Overview
We introduce two Stereo-Echo datasets named **Stereo-Replica** and **Stereo-Matterport3D** from [Replica](https://github.com/facebookresearch/Replica-Dataset) and [Matterport3D](https://niessner.github.io/Matterport/) respectively for multimodal stereo depth estimation benchmarks with echoes. The corresponding echoes are simulated using 3D simulators [Habitat](https://github.com/facebookresearch/habitat-sim) and audio simulator [SoundSpaces](https://github.com/facebookresearch/sound-spaces). The details of our datasets are described in our paper.

![](/images/dataset_overview.PNG)

## Download
**Stereo-Replica** dataset can be obatined from Baidu Cloud links ([Download Extraction Code:jqzm](https://pan.baidu.com/s/100Fx_5CLe1FQLWyx9hPlSA)), which includes stereo images ('Replica_dataset' folder) and echoes ('echoes_navigable' folder). We have used the 128x128 image resolution for our experiment.

**Stereo-Matterport3D** is an extension of existing matterport3D dataset. In order to obtain the raw frames please forward the access request acceptance from the authors of matterport3D dataset. We only release the obtained echoes from Baidu Cloud ([Download Extraction Code: gdf3](https://pan.baidu.com/s/1-LKnKVOCWM5Xw8axnA3jGQ)).


## Stereo Images Synthesis
Please use *replica_parsing.py* to synthesize stereo images of Replica and use *mp3d_parsing.py* to synthesize stereo images of Matterport3D.

## Citation
If you use our Stereo-Echo datasets in your research, please cite this publication:
```
@inproceedings{zhang_2022_ECCV,
title={Stereo Depth Estimation with Echoes},
author={Zhang, Chenghao and Tian, Kun and Ni, Bolin and Meng, Gaofeng and Fan, Bin and Zhang, Zhaoxiang and Pan, Chunhong},
booktitle={European Conference on Computer Vision (ECCV)},
year={2022}}
```

## Acknowledgements
This repository makes liberal use of code from [stereo-from-mono](https://github.com/nianticlabs/stereo-from-mono/) and [beyond-image-to-depth](https://github.com/krantiparida/beyond-image-to-depth).
