# Video acceleration magnification

Python implementation of this paper:

`
   Author    = {Yichao Zhang, Silvia L. Pintea, Jan C. van Gemert},
   Title     = {Video Acceleration Magnification},
   Paper = {CVPR 2017},
   Subject = {Computer Vision and Pattern Recognition}
   Year      = {2017}
`

## Installing
```
git clone https://github.com/nejcgalof/video-acceleration-magnification.git
```

## Usage
```
video_accel_mag.py <input_video> [--py_level] [--alpha]
```

Arguments `--py_level` and `--alpha` defaults to 4, while `input_video` argument is mandatory.

Example of run:
```
 video_accel_mag.py syn_ball.avi --py_level 4 --alpha 5
```

## Results

We show results with alpha 5 and pylevel 4. More results in `./gifs`.

![syn_ball_alpha_5_pylevel_4](./gifs/syn_ball_alpha_5_pylevel_4.gif)
![gun_shot_alpha_5_pylevel_4](./gifs/gun_shot_alpha_5_pylevel_4.gif)
![kitara_alpha_4_pylevel_4](./gifs/kitara_alpha_4_pylevel_4.gif)
![roka_alpha_5_pylevel_4](./gifs/roka_alpha_5_pylevel_4.gif)

## Authors

[Blaž Sitar](https://github.com/BSitar) and Nejc Galof
