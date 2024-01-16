<h1 align = "center">
  Pytorch-Transformer <br>
  <a href="https://github.com/m-np/pytorch-transformer/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/m-np/pytorch-transformer?logo=git&style=plastic"></a>
  <a href="https://github.com/m-np/pytorch-transformer/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/m-np/pytorch-transformer?style=plastic&logo=github"></a>
  <a href="https://github.com/m-np/pytorch-transformer/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/m-np/pytorch-transformer?style=plastic&logo=github"></a>
  <a href="https://makeapullrequest.com/"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=plastic&logo=open-source-initiative"></a>
</h1>

<div align = "justify">

**Objective:** This project contains my work on building a transformer from scratch for an English-to-German translation. <br>
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
