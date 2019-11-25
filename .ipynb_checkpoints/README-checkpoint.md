# Data Analytics at Scale 2019

This code and data is intended for the OII Data Analytics at Scale course only. The code and data should not be used for any other purposes.

## Data

We will use the "meme generator" dataset from the Library of Congress to evaluate our image hashing algorithms. The dataset may be downloaded [from the OII file server as das_images.zip](http://gofile.me/25ezO/eoiSsixZQ). It consists of 55,972 JPEG images belonging to 1,034 groups (or "meme families").

Filenames consist of two parts. The first four numbers followed by an underscore indicate the meme family of the image. For example, all images that begin "0086_" are part of meme family 86 and should be more similar to one another than to images in other groups. The data are not perfect, and thus even a perfect algorithm will not achieve a perfect score.

**Please note that some images may be offensive.**

The md5sum of the das_images.zip file is given below to allow you to check that it has downloaded correctly.

```
$ md5sum das_images.zip 
fcb1ff3c62f3ca39c39985a4bba0d873  das_images.zip
```

You should not need any further information about the dataset to complete the assignment, but the curious can refer to the Library of Congress [webpage](https://www.loc.gov/item/lcwaN0010226/) and [dataset page](https://labs.loc.gov/experiments/webarchive-datasets/)

## Image hashing

Per the assignment, will need to make use of the [imagehash library](https://github.com/JohannesBuchner/imagehash/). You will also need to use the FINd algorithm in this repository.

To run the FINd algorithm, you will need to place `FINd.py` and `matrix.py` in the same directory. To test that the code runs, you can execute it from the terminal with a filename as an argument. For example:

```
$ python FINd.py das_images/0297_21195384.jpg
2e340b5ea54927c35a87f4b7f8518858353cc4bc03c333eed0e7fd01fd8112be,das_images/0297_21195384.jpg
[[0 0 1 0 1 1 1 0 0 0 1 1 0 1 0 0]
 [0 0 0 0 1 0 1 1 0 1 0 1 1 1 1 0]
 [1 0 1 0 0 1 0 1 0 1 0 0 1 0 0 1]
 [0 0 1 0 0 1 1 1 1 1 0 0 0 0 1 1]
 [0 1 0 1 1 0 1 0 1 0 0 0 0 1 1 1]
 [1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1]
 [1 1 1 1 1 0 0 0 0 1 0 1 0 0 0 1]
 [1 0 0 0 1 0 0 0 0 1 0 1 1 0 0 0]
 [0 0 1 1 0 1 0 1 0 0 1 1 1 1 0 0]
 [1 1 0 0 0 1 0 0 1 0 1 1 1 1 0 0]
 [0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 1]
 [0 0 1 1 0 0 1 1 1 1 1 0 1 1 1 0]
 [1 1 0 1 0 0 0 0 1 1 1 0 0 1 1 1]
 [1 1 1 1 1 1 0 1 0 0 0 0 0 0 0 1]
 [1 1 1 1 1 1 0 1 1 0 0 0 0 0 0 1]
 [0 0 0 1 0 0 1 0 1 0 1 1 1 1 1 0]]
```

FINd generates a 256-bit hash of an image. The first line contains this hash in hexadecimal encoding. The matrix that follows shows the hash as a 16x16 grid.

Please see `findExample.ipynb` to see how FINd can be loaded from Python.

