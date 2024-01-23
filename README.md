<h1 align = "center">
  Pytorch-Transformer <br>
  <a href="https://github.com/m-np/pytorch-transformer/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/m-np/pytorch-transformer?logo=git&style=plastic"></a>
  <a href="https://github.com/m-np/pytorch-transformer/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/m-np/pytorch-transformer?style=plastic&logo=github"></a>
  <a href="https://github.com/m-np/pytorch-transformer/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/m-np/pytorch-transformer?style=plastic&logo=github"></a>
  <a href="https://github.com/m-np/pytorch-transformer/pulls"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=plastic&logo=open-source-initiative"></a>
</h1>

<div align = "justify">

**Objective:** This project contains my work on building a transformer from scratch for an German-to-English translation. <br>
This project uses <a href = "https://github.com/gordicaleksa/pytorch-original-transformer/tree/main">pytorch-original-transformer</a> work to understand the inner workings of the transformer and how to build it from scratch. Along with the implementation, we are referring to the <a href = "https://arxiv.org/abs/1706.03762">original paper</a> to study transformers.<br>
To understand the repo, check the [**HOWTO.md**](./HOWTO.md) file.

---

</div>

## Setup

Please follow the following steps to run the project locally <br/>

1. `git clone https://github.com/m-np/ai-ml-project-template.git`
2. Open Anaconda console/Terminal and navigate into project directory `cd path_to_repo`
3. Run `conda create --name <env_name> python==3.9`.
4. Run `conda activate <env_name>` (for running scripts from your console or set the interpreter in your IDE)

For adding the new conda environment to the jupyter notebook follow this additional instruction
1. Run `conda install -c anaconda ipykernel`
2. Run `python -m ipykernel install --user --name=<env_name>`

-----

For pytorch installation:

PyTorch pip package will come bundled with some version of CUDA/cuDNN with it,
but it is highly recommended that you install a system-wide CUDA beforehand, mostly because of the GPU drivers. 
I also recommend using Miniconda installer to get conda on your system.
Follow through points 1 and 2 of [this setup](https://github.com/Petlja/PSIML/blob/master/docs/MachineSetup.md)
and use the most up-to-date versions of Miniconda and CUDA/cuDNN for your system.

-----

For other module installation, please follow the following steps:
1. Open Anaconda console/Terminal and navigate into project directory `cd path_to_repo`
2. Run `conda activate <env_name>`
3. Run `pip install -r requirements.txt` found 👉 [`requirements.txt`](./requirements.txt)

-----

## Description 

The model is trained on the Kaggle [Multi30K dataset](https://www.kaggle.com/datasets/devanshusingh/machine-translation-dataset-de-en) and the notebook used for training the data is found [here](./notebooks/training-nw-nb.ipynb)

This model takes the following arguments as represented in the paper.

```
'dk': key dimensions -> 32,
'dv': value dimensions -> 32,
'h': Number of parallel attention heads -> 8,
'src_vocab_size': source vocabulary size (German) -> 8500,
'target_vocab_size': target vocabulary size (English) -> 6500,
'src_pad_idx': Source pad index -> 2,
'target_pad_idx': Target pad index -> 2,
'num_encoders': Number of encoder modules -> 3,
'num_decoders': Number of decoder modules -> 3,
'dim_multiplier': Dimension multiplier for inner dimensions in pointwise FFN (dff = dk*h*dim_multiplier) -> 4,
'pdropout': Dropout probability in the network -> 0.1,
'lr': learning rate used to train the model -> 0.0003,
'N_EPOCHS': Number of Epochs -> 50,
'CLIP': 1,
'patience': 5
```
We use Adam Optimizer along with CrossEntropyLoss to train the model.

We tested the performance of the model on 1000 held-out test data and observed a Bleu score of 30.8

### Hugging Face
The trained model can also be found in the [huggingface repo](https://huggingface.co/Rzoro/Transformer_de_en_multi30K)


## LICENSE 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)


## Resources

The following code is derived from the pytorch-original-transformer 
```
@misc{Gordić2020PyTorchOriginalTransformer,
  author = {Gordić, Aleksa},
  title = {pytorch-original-transformer},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gordicaleksa/pytorch-original-transformer}},
}
```

and using the following [blog](https://medium.com/@hunter-j-phillips/putting-it-all-together-the-implemented-transformer-bfb11ac1ddfe)
