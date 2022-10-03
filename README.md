# EmbryosFormer: Deformable Transformer and Collaborative Encoding-Decoding for Embryos Stage Development Classification
The timing of cell divisions in early embryos during In-Vitro Fertilization (IVF) process is a key predictor of embryo viability. However, observing cell divisions in Time-Lapse Monitoring (TLM) is a time-consuming process and highly depends on experts. In this paper, we propose EmbryosFormer, a computational model to automatically detect and classify cell divisions from original time-lapse images. Our proposed network is designed as an encoder-decoder deformable transformer with collaborative heads. The transformer contracting path predicts per-image label and is optimized by a classification head. The transformer expanding path models the temporal coherency between embryos images to ensure monotonic non-decreasing constraint and is optimized by a segmentation head. Both contracting and expanding paths are synergetically learned by a collaboration head. We have benchmarked our proposed EmbryosFormer on two datasets: a public dataset with mouse embryos with 8-cell stage and an in-house dataset with human embryos with 4-cell stage. 

## 1. Installation
- we use PyTorch 1.12.1 and cuda 11.3 (higher versions may be available)

## 2. Dataset preparation
- Extract video features and use [create_annot.sh](data/embryo/create_annot.sh) to create input annotations for the training step (json format)

## 3. Training and Validation
- Use scripts: `scripts/train.sh` and `scripts/test.sh`. Config file is in `cfgs` folder

## Citation
Please consider citing this project in your publications if it helps your research
```
```
The code is used for academic purpose only.

## Acknowledgement
- The Transformer-based network is mainly based on [PDVC](https://github.com/ttengwang/PDVC) and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). We thank the authors for their great works