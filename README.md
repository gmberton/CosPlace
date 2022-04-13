
# Rethinking Visual Geo-localization for Large-Scale Applications

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-pittsburgh-250k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-250k?p=rethinking-visual-geo-localization-for-large)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-pittsburgh-30k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-30k?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-tokyo247)](https://paperswithcode.com/sota/visual-place-recognition-on-tokyo247?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-mapillary-val)](https://paperswithcode.com/sota/visual-place-recognition-on-mapillary-val?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-st-lucia)](https://paperswithcode.com/sota/visual-place-recognition-on-st-lucia?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-sf-xl-test-v1)](https://paperswithcode.com/sota/visual-place-recognition-on-sf-xl-test-v1?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-sf-xl-test-v2)](https://paperswithcode.com/sota/visual-place-recognition-on-sf-xl-test-v2?p=rethinking-visual-geo-localization-for-large)

This is the official repository for the CVPR 2022 paper [Rethinking Visual Geo-localization for Large-Scale Applications](https://arxiv.org/abs/2204.02287).
The paper presents a new dataset called San Francisco eXtra Large (SF-XL, go [_here_](https://forms.gle/wpyDzhDyoWLQygAT9) to download it), and a highly scalable training method (called CosPlace), which allows to reach SOTA results with compact descriptors.

The images below represent respectively:
1) the map of San Francisco eXtra Large
2) a visualization of how CosPlace Groups (read datasets) are formed
3) results with CosPlace vs other methods on Pitts250k (CosPlace trained on SF-XL, others on Pitts30k)
<p float="left">
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/images/SF-XL%20map.jpg" height="200" />
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/images/map_groups.png" height="200" /> 
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/images/backbones_pitts250k_main.png" height="200" />
</p>



## Train
After downloading the SF-XL dataset, simply run 

`$ python3 train.py --dataset_folder path/to/sf-xl/processed`

the script automatically splits SF-XL in CosPlace Groups, and saves the resulting object in the folder `cache`.
By default training is performed with a ResNet-18 with descriptors dimensionality 512 is used, which fits in less than 4GB of VRAM.

To change the backbone or the output descriptors dimensionality simply run 

`$ python3 train.py --dataset_folder path/to/sf-xl/processed --backbone resnet50 --fc_output_dim 128`

You can also speed up your training with Automatic Mixed Precision (note that all results/statistics from the paper did not use AMP)

`$ python3 train.py --dataset_folder path/to/sf-xl/processed --use_amp16`

Run `$ python3 train.py -h` to have a look at all the hyperparameters that you can change. You will find all hyperparameters mentioned in the paper.

#### Reproducibility
Results from the paper are fully reproducible, and we followed deep learning's best practices (average over multiple runs for the main results, validation and hyperparameter search on the val set).

## Test
You can test a trained model as such

`$ python3 eval.py --dataset_folder path/to/sf-xl/processed --backbone resnet50 --fc_output_dim 128 --resume_model path/to/best_model.pth`

You can download plenty of trained models below.

## Model Zoo

<details>
     <summary><b>Models with different backbones and dimensionality of descriptors, trained on SF-XL</b></summary></br>
    Pretained networks employing different backbones.</br></br>
	<table>
		<tr>
			<th rowspan=2>Model</th>
			<th colspan=7>Dimension of Descriptors</th>
	 	</tr>
	 	<tr>
	  		<td>32</td>
	   		<td>64</td>
	   		<td>128</td>
	   		<td>256</td>
	   		<td>512</td>
	   		<td>1024</td>
	   		<td>2048</td>
	 	</tr>
		<tr>
			<td>ResNet-18</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>-</td>
			<td>-</td>
	 	</tr>
		<tr>
			<td>ResNet-50</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
	 	</tr>
		<tr>
			<td>ResNet-101</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
	 	</tr>
		<tr>
			<td>ResNet-152</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
	 	</tr>
		<tr>
			<td>VGG-16</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>-</td>
			<td>-</td>
	 	</tr>
	</table>
</details>

## Cite
Here is the bibtex to cite our paper
```
@inProceedings{Berton_CVPR_2022_cosPlace,
  author = {Berton, Gabriele and Masone, Carlo and Caputo, Barbara},
  title = {Rethinking Visual Geo-localization for Large-Scale Applications}, 
  booktitle = {CVPR},
  month = {June}, 
  year = {2022}, }
```

## Issues
If you find some problems in our code, or have any advice or questions, feel free to open an issue or send an email to berton.gabri@gmail.com

