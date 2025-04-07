# InstaMap 

![Alt text](https://journals.iucr.org/d/issues/2025/04/00/sor5003/sor5003fig14.jpg?rand=1744031136769)

InstaMap models heterogeneity by bending space. Cryo-EM images yi (left) with annotated pose and imaging parameters $(R_i, T_i, \text{PSF}_i)$ are used for gradient-based learning. A vector field in a fixed frame is queried at the rotated, shifted and jittered grid to provide a per-image SE(3) equivariant output $F_i$. Space is bent via an additive perturbation on the corresponding rotated, shifted and jittered grid.

## Reference
```
@article{InstaMap2025,
abstract = {Despite the parallels between problems in computer vision and cryo-electron microscopy (cryo-EM), many state-of-the-art approaches from computer vision have yet to be adapted for cryo-EM. Within the computer-vision research community, implicits such as neural radiance fields (NeRFs) have enabled the detailed reconstruction of 3D objects from few images at different camera-viewing angles. While other neural implicits, specifically density fields, have been used to map conformational heterogeneity from noisy cryo-EM projection images, most approaches represent volume with an implicit function in Fourier space, which has disadvantages compared with solving the problem in real space, complicating, for instance, masking, constraining physics or geometry, and assessing local resolution. In this work, we build on a recent development in neural implicits, a multi-resolution hash-encoding framework called instant-NGP, that we use to represent the scalar volume directly in real space and apply it to the cryo-EM density-map reconstruction problem ( InstaMap ). We demonstrate that for both synthetic and real data, InstaMap for homogeneous reconstruction achieves higher resolution at shorter training stages than five other real-spaced representations. We propose a solution to noise overfitting, demonstrate that InstaMap is both lightweight and fast to train, implement masking from a user-provided input mask and extend it to molecular-shape heterogeneity via bending space using a per-image vector field.},
author = {Woollard, Geoffrey and Zhou, Wenda and Thiede, Erik H and Lin, Chen and Grigorieff, Nikolaus and Cossio, Pilar and {Dao Duc}, Khanh and Hanson, Sonya M},
doi = {10.1107/S2059798325002025},
issn = {2059-7983},
journal = {Acta Crystallographica Section D Structural Biology},
keywords = {cryo-em,density maps,end-to-end gradient-based,heterogeneity},
month = {apr},
number = {4},
pages = {147--169},
publisher = {International Union of Crystallography},
title = {{InstaMap : instant-NGP for cryo-EM density maps}},
url = {https://journals.iucr.org/paper?S2059798325002025},
volume = {81},
year = {2025}
}
```