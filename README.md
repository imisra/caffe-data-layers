# caffe-data-layers
Data Layers for Caffe

## Python Data Layer
A major issue when using Python Data Layers with `caffe` is using data prefetching easily. Since Python isn't truly multi-threaded, I use multiple processes for prefetching.
To make the inter-process communication efficient, I bypass using the shared data implementations (`Queue`, etc.) from the `multiprocessing` library (which serialize data and are slow), and instead use `sharedctypes`.

The reference layer here (`multiple_image_multiple_label_data_layer_dist.py`), shows an example implementation. It can trivially be extended to use multiple prefetching processes.
In order to make the most of this, please set the environment variable `TMPDIR = '/dev/shm'` so that `sharedctypes` uses shared memory (Thanks to Carl Doersch for pointing this out).
