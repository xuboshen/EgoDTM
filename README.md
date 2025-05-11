⚠️ I will reorganize this repository and release all code, data, and downstream benchmark evaluations between late May and June. Please stay tuned!

<h1 align="center">
  EgoDTM: Towards 3D-Aware Egocentric Video-Language Pretraining
</h1>


<p align="center" width="100%">
<a target="_blank"><img src="assets/paper/method.png" alt="method" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

We introduce EgoDTM, an **Ego**centric **D**epth- and **T**ext-aware video-language **M**odel. 

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Installation](#installation)
- [Data Generation](#data-generation)
- [Training](#training)
- [Visualization](#visualization)


## Overview
- We introduce EgoDTM, a 3D-aware egocentric video-language model learned from 3D-aware video-language pretraining 
- We develop a lightweight 3D-aware decoder for depth estimation and a ata construction pipeline to enrich captions with spatial information. As a byprouct, we generate millions of egocentric ata, including captions, HOI boxes, HOI masks, and depth maps.
- Extensive experimental results show that EgoDTM significantly enhances performance on video understanding tasks and 3D understanding tasks.

## Installation

The environment depends on the pretrained EgoVLM, see corresponding installation docs in [INSTALL.md](doc/INSTALL.md). 



## Data Generation
<p align="center" width="100%">
<a target="_blank"><img src="assets/figs/DataGeneration.png" alt="Data generation pipeline" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## Training

We provide our training log of EgoVLMs under [EgoDTM_train_log](assets/train_logs/egodtm_log.txt)

## Visualization

data generated from our pipeline
<p align="center" width="100%">
<a target="_blank"><img src="assets/figs/vis_data.png" alt="vis_data" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>


predicted depths by EgoDTM
<p align="center" width="100%">
<a target="_blank"><img src="assets/figs/vis_input_depth.png" alt="vis_input_depth" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

LLM prompt
<p align="center" width="100%">
<a target="_blank"><img src="assets/figs/LLM_prompt.png" alt="LLM_prompt" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>
