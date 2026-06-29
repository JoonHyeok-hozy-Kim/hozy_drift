# Drift Model Reconstruction
- Original Paper : Deng et al. 2026, *Generative Modeling via Drifting* [arxiv](https://arxiv.org/abs/2602.04770)
- Authors' Code : [github](https://github.com/lambertae/drifting)
- hozy summary : [hozy blog](https://joonhyeok-hozy-kim.github.io/blog/2026/drift_model/)

<br>

## Implementation Notice
- [x] 2-Dimensional Example
  |Training Data|Inference Result|
  |:-:|:-:|
  |![](./assets/training_data-epoch_1.png)|![](./assets/denoised_sample.png)|
- [ ] Imagenet Example : TBD

<br>

## How to run
```
bash scripts/inference/denoised_sampling/two_dimensional.sh
```

<br>

## How to train
```
bash scripts/train/two_dimensional.sh
```


<br>

## Citation
- Original Paper
  ```
  @misc{deng2026generativemodelingdrifting,
      title={Generative Modeling via Drifting}, 
      author={Mingyang Deng and He Li and Tianhong Li and Yilun Du and Kaiming He},
      year={2026},
      eprint={2602.04770},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.04770}, 
  }
  ```
- Authors' Implementation
  ```
  @article{deng2026generative,
  title={Generative Modeling via Drifting},
  author={Deng, Mingyang and Li, He and Li, Tianhong and Du, Yilun and He, Kaiming},
  journal={arXiv preprint arXiv:2602.04770},
  year={2026}
  }
  ```
- This implementation
  ```
  @misc{kim2026drift_model_reconstruction,
  author = {Joon Hyeok Kim},
  title = {A Reconstruction of Drift Model},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/JoonHyeok-hozy-Kim/hozy_drift}},
  }
  ```