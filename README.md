# LaViSE
This is the official repository for paper "Explaining Deep Convolutional Neural Networks via Unsupervised 
Visual-Semantic Filter Attention" to appear in CVPR 2022. 

Authors: Yu Yang, Seungbae Kim, Jungseock Joo

![alt text](./img/intro.png)

[//]: # (## Requirements)

## Datasets
### [Common Objects in Context (COCO)](https://cocodataset.org/#home) 

- Please follow the instructions in the 
[COCO API README](https://github.com/cocodataset/cocoapi) and 
[here](data/README.md) to download and setup the COCO data.

### [Visual Genome (VG)](https://visualgenome.org/)

- Please follow the instructions in the 
[README of the python wrapper for the Visual Genome API](https://github.com/ranjaykrishna/visual_genome_python_driver) 
and [here](data/README.md).

### [GloVe](https://nlp.stanford.edu/projects/glove/)

- We load the pretrained GloVe word embeddings directly from the 
[torchtext](https://torchtext.readthedocs.io/en/latest/vocab.html#glove) library.

### Social Media <u>P</u>hotographs <u>o</u>f US <u>P</u>oliticians (PoP)

- The list of entities used to discover new concepts is provided in `data/entities.txt`.

## Getting started 

### Requirements

Required packages can be found in `requirements.txt`.

### Usage

Train an explainer with

```commandline
python train_explainer.py
```

Explain a target filter of any model with

```commandline
python infer_filter.py
```

More features will be added soon! 🍻

## Citation
```
@inproceedings{yang2022explaining,
    author    = {Yang, Yu and Kim, Seungbae and Joo, Jungseock},
    title     = {Explaining Deep Convolutional Neural Networks via Unsupervised Visual-Semantic Filter Attention},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022},
}
```
