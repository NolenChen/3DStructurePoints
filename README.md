# Unsupervised Learning of Intrinsic Structural Representation Points


## Description
This repository contains the code for our cvpr 2020 paper: Unsupervised Learning of Intrinsic Structural Representation Points

<div align="center">
<img src="https://github.com/NolenChen/3DStructurePoints/blob/master/figs/teaser.png" width="70%" height="70%"><br><br>
</div>


## Environment setup

Current Code is tested on ubuntu16.04, python3.6, torch 1.1.1 and torchvision 0.3.0. 
We use a [pytorch version of pointnet++](https://github.com/erikwijmans/Pointnet2_PyTorch) in our pipeline.
```
pip install -r requirements.txt
cd pointnet2
python setup.py build_ext --inplace
```


## Dataset

The training and testing data for 3D semantic correspondence is provided by [LMVCNN](https://people.cs.umass.edu/~hbhuang/local_mvcnn/) and [bhcp](http://www.vovakim.com/projects/CorrsTmplt/doc_data.html) respectively, and you can download the preprocessed training data [here](https://drive.google.com/file/d/1MkUcFF4gbfhQLssPNd_MV9afOOar-Wxn/view?usp=sharing).
The script for preprocessing the testing data will come soon.


## Train

```
python train/train_structure_points.py -data_dir PATH_TO_TRAINING_DATA -num_structure_points 16 -category plane -log_dir PATH_TO_LOG
```
* -data_dir: path to the preprocessed training data.
* -num_structure_points: number of structure points
* -category: category to train on
* -log_dir: path to log dir

The trained model will be saved in PATH_TO_LOG/checkpoints/model

## Test

```
python test/test_structure_points.py -data_dir ./demo_data/plane -model_dir PATH_TO_TRAINED_MODEL -num_structure_points 16 -output_dir OUTPUT_PATH
```
* -model_dir: path to trained model
* -data_dir: path to the testing data.
* -output_dir: output path.
* -num_structure_points: number of structure points, should be the same with training stage.


## Citation
Please cite our paper if you find it useful in your research:

```

```



