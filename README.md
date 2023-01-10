# BIDNeRF

**Bundle adjusted Inverse Neural Radiance Fields** is an updated version of [INeRF](https://yenchenlin.me/inerf/).

It is based on the [JAXNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf) implementation, which is not so user friendly, so I included a wrapper.

This project was created for internal research purposes and is not intended to be used directly as a general tool.


## Installation
Clone the repository.
```bash
git clone --recurse-submodules git@github.com:Egoago/bidnerf.git
cd bidnerf
```
Create the environment. For details see the JAXNeRF repository.
```bash
conda create --name bidnerf python=3.8
conda activate bidnerf
conda install jax cuda-nvcc -c conda-forge -c nvidia

pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tqdm pyyaml Pillow opencv-python flax==0.5.3 tensorflow==2.7.3 jaxlie
```
Download the test [datasets](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) (containing images, camera poses and camera intrinsics) and preptrained JAXNeRF [models](http://storage.googleapis.com/gresearch/jaxnerf/jaxnerf_pretrained_models.zip).
The downloaded files should be extracted and rearranged into the following hierarchy:
```
project
├───datasets
│   ├───blender
│   │   ├───chair
│   │   │   ...
│   │
│   └───llff
│       │   ...
│   
└───models
    ├───blender
    │   ├───chair
    │   │   └───chekpoint_100000
    │   │   ...
    │
    └───llff
        │   ...
```

Render new images from the pretrained models.
```bash
python render_scene.py --config=../config
```

## Usage
The *config.yaml* file contains all the hyperparameters with which the rendering and registering processes can be modified.
An example usage is included in the *pose_estimation.py* file.
```python
import jax
import bidnerf

# initialization
bidnerf.load_flags()
rng = jax.r.PRNGKey(20221012)
rng_keys = jax.random.split(rng, 4)

# loading
data = bidnerf.load_data(rng_keys[0])
render_fn = bidnerf.load_model(rng_keys[1])

# optimalization
bidnerf.fit(data, render_fn, rng_keys[2])

bidnerf.save(data, render_fn, rng_keys[3], "gdr")
```
