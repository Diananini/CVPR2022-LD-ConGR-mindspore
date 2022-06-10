# LD-ConGR: A Large RGB-D Video Dataset for Long-Distance Continuous Gesture Recognition

## Introduction
This project is the mindspore implementation of [our paper](paper.pdf) (accepted by CVPR2022).

Gesture recognition plays an important role in natural human-computer interaction and sign language recognition. Existing research on gesture recognition is limited to close-range interaction such as vehicle gesture control and face-to-face communication. To apply gesture recognition to long-distance interactive scenes such as meetings and smart homes, we establish a large RGB-D video dataset LD-ConGR. LD-ConGR is distinguished from existing gesture datasets by its **long-distance gesture collection**, **fine-grained annotations**, and **high video quality**. Specifically, 1) the farthest gesture provided by the LD-ConGR is captured 4m away from the camera while existing gesture datasets collect gestures within 1m from the camera; 2) besides the gesture category, the temporal segmentation of gestures and hand location are also annotated in LD-ConGR; 3) videos are captured at high resolution (1280 x 720 for color streams and 640 x 576 for depth streams) and high frame rate (30 fps).

### Citation
If you find useful the LD-Con dataset for your research, please cite the paper:
```
@inproceedings{ld-congr-cvpr2022,
    title={LD-ConGR: A Large RGB-D Video Dataset for Long-Distance Continuous Gesture Recognition},
    author={Dan Liu and Libo Zhang and Yanjun Wu},
    booktitle={CVPR},
    year={2022}
}
```
