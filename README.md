# Face verification Python\*

To install all the required Python modules you can use:

``` sh
conda create -n faceverify python=3.8
conda activate faceverify
pip install -r requirements.txt

```

Download the checkpoint folder from [here](https://drive.google.com/drive/folders/1Q4h-80rfyCoffCC0w0ESjLN_iWN60pEZ?usp=sharing) and save the files to `./checkpoints/buffalo_sc/`

For Face verification, run the below command:

``` sh
python face_verify.py -i1 <path_image1> -i2 <path_image2>

```


