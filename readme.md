# Identification of thin-section in metamorphic rocks based on one-step diffusion model
This is the repository of "Identification of thin-section in metamorphic rocks based on one-step diffusion model"

## Run Experiments
You can do a quick test using run.ipynb in a jupyter notebook.
### Setup & Preparation

* Create the environment
```bash
conda create env -n env_name python=3.10 -y
conda activate env_name
```

* Install necessary python libraries:
```bash
pip install pyyaml safetensors einops transformers scipy torchsde aiohttp spandrel kornia requests numpy
pip install open_clip_torch
```

Install diffuser:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```
Install CutMix
```bash
pip install git+https://github.com/ildoonet/cutmix
```

### Datasets
All the datasets should be prepared in `./data/`. 

The first dataset: [Photomicrograph dataset of rocks for petrology teaching in Nanjing University](https://www.scidb.cn/en/detail?dataSetId=732953783604084736&version=V1)

The second dataset: [Micro image data set of some rock forming minerals, typical metamorphic minerals and oolitic thin sections](https://www.scidb.cn/en/detail?dataSetId=684362351280914432)

I will upload the processed dataset compressed package to the network disk, if necessary.

### Run
Use "scripts/exps/expand_diff.sh" to generate more metamorphic rock's thin-section images.

"classification/classificationAtM_Rock.py" is used to cross-validate the recognition accuracy on the original dataset, and "classification/classificationAtExpanedM_Rock.py" is used to cross-validate the recognition accuracy on the expanded dataset.

"fid/fid.py" is used to test the fid value of the generated image.
