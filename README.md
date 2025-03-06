# SGDIFF: Dual-Granularity Semantic Guided Sparse Routing Diffusion Model for General Pansharpening

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

  <h3 align="center">  </h3>
  <p align="center">
    处理遥感图像融合的场景依赖问题——基于多模态MOE架构
    <br />
    <a href="https://github.com/shaojintian/Best_README_template"><strong>探索本项目的文档 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/shaojintian/Best_README_template/issues">报告Bug</a>
    ·
    <a href="https://github.com/shaojintian/Best_README_template/issues">提出新特性</a>
  </p>


# News

 - [ ] Inference Code Release


### 数据集
DownLoad datasets from [PanCollection](https://liangjiandeng.github.io/PanCollection.html)

### 多模态语义先验获取

分别基于PAN和MS图像提取语义先验信息，包括粗粒度场景分类信息和细粒度地物描述信息
建议使用的遥感图文多模态大模型：

[GeoChat](https://github.com/mbzuai-oryx/GeoChat)

[LRHS-BOT](https://github.com/NJU-LHRS/LHRS-Bot)

注：SGDiff不限制MLLM的选型，即使论文基于Geochat实现，但更好的MLLM对模型存在潜力

[Example File Format](tuneavideo/data/example)
### CheckPoint
[Pretrained Checkpoint]()

### QuickStart
###### **requirement**
```
conda create -n SGDiff python==3.10
conda activate SGDiff
pip install -r requirements.txt 
```
###### **inference**

 1. update configs/general_finetune.yaml
```
train_data:
  train_qb:
    dataroot:  the path for h5 data
    grounding_file: the path for grounding description file
    scene_file:  the path for grounding description file
    ...
 validation_data:
   val_QB:
     dataroot:
     grounding_file:
     scene_file:
 resume_from_checkpoint: the path for pretrained model
```
2.  
```
python inference.py
```

 3. look up into your **output_dir** for  .mat output, then compute metrics or visulization pansharpened results

###### **training from scratch**

```
python train_pansharpening.py
```




### 作者

Contact:  321227312@qq.com

<!-- links -->
[your-project-path]:shaojintian/Best_README_template
[contributors-shield]: https://img.shields.io/github/contributors/shaojintian/Best_README_template.svg?style=flat-square
[contributors-url]: https://github.com/shaojintian/Best_README_template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shaojintian/Best_README_template.svg?style=flat-square
[forks-url]: https://github.com/shaojintian/Best_README_template/network/members
[stars-shield]: https://img.shields.io/github/stars/shaojintian/Best_README_template.svg?style=flat-square
[stars-url]: https://github.com/shaojintian/Best_README_template/stargazers
[issues-shield]: https://img.shields.io/github/issues/shaojintian/Best_README_template.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/shaojintian/Best_README_template.svg
[license-shield]: https://img.shields.io/github/license/shaojintian/Best_README_template.svg?style=flat-square
[license-url]: https://github.com/shaojintian/Best_README_template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian

   

### Citication


### Thanks
codebase:        https://github.com/showlab/Tune-A-Video




