"""
Author: Ishan Misra
Email: ishan@cmu.edu

Summary:
reads in a key file and creates tops (images or labels) according to the keys. The keys are essentially image_names without the file extension.
I use these keys as keys inside a hdf5 file. Depending on what you specify for the datatype of the "top",
the key can serve as -- 1)[im] path to an image 2)[lab] key of hdf5 file 3)[txt] ignored

Params:
keys_file: a tab separated list of keys; number of keys per line == number of tops
data_shapes : shape of the tops in caffe style; batch_size must be the same across all tops
label_files: path to hdf5 label files (these are lazy loaded, so no need to create separate chunks)
txt_files: txt files containing numeric data
image_dir, image_ext: image dir and extension
bgr_mean: mean to be used with image
top_dtypes: a list indicating what the dtypes for the top are. Possible values --
    - 'im': image; the key specifies the image filename without extension
    - 'lab': hdf5; the key specifies the key in the hdf file
    - 'txt': text; the key is ignored
prefetch_size: number of blobs to prefetch; setting this to "0" turns off prefetcher. typical value is between 3-5.
A major pain is to figure out when the caffe network process has ended, so that we can kill the prefetcher. I have a slightly hacky way to do this right now.
Also, note that this layer can trivially be extended to use multiple prefetch processes. Check for the "TODO" tag to see how to modify this.

Example usage:
first line of keys_file: "IMG_001\tsegmask-IMG_001\tdummy"
first line of txt_file: "1 0 0 1 0"
param top_dtypes: "[im, lab, txt]"
what the layer does:
uses IMG_001 to read the image
uses segmask-IMG_001 to read a label from the hdf5 file
reads "1 0 0 1 0" -- (requires keys_file and txt_file to be in correspondence)

Example param_str:
param_str: "{ 'keys_file': '/IUS/vmr104/imisra/research/genData/imagenetVideo/rnd5/image_keys.txt', 'data_shapes' : [[3,3,227,227],[3,3,227,227],[3,3,227,227],[3,1,1,1]], 'label_files' : [], 'txt_files': ['/IUS/vmr104/imisra/research/genData/imagenetVideo/rnd5/image_labs.txt'], 'image_dir' : '/scratch/imisra/imnetvideo', 'image_ext' : '.JPEG', bgr_mean: [103.939, 116.779, 123.68], 'top_dtypes': ['im','im','im','txt'], 'prefetch_size': 5 }"

Using shared memory to speed things up (Thanks to Carl Doersch for this pointer)
The sharedctypes library does not use shared memory by default, and creates files in /tmp. This can be slow.
To speed things up, just set the environment variable "TMPDIR" to "/dev/shm" so that it uses shared memory BEFORE you use caffe.
e.g. export TMPDIR="/dev/shm" && <run my network>
"""
__author__ = "Ishan Misra <ishan@cmu.edu>"
__date__ = "2015.11.17"
import caffe
import numpy as np
import yaml
from multiprocessing import Process, Queue
import multiprocessing
import h5py
import math
import code
import traceback as tb
import os
from PIL import Image
import cv2
import scipy.misc
from multiprocessing.sharedctypes import Array as sharedArray
import ctypes
import atexit
import time
import sys
import operator

def prod(ll):
    return float(reduce(operator.mul, ll, 1));

class MultipleImageMultipleLabelDataLayer(caffe.Layer):
    def setup_prefetch(self):
        self._slots_used = Queue(self._max_queue_size);
        self._slots_filled = Queue(self._max_queue_size);
        global shared_mem_list
        shared_mem_list = [[] for t in range(self._num_tops)]
        for t in range(self._num_tops):
            for c in range(self._max_queue_size):
                shared_mem = sharedArray(ctypes.c_float, self._blob_counts[t]);
                with shared_mem.get_lock():
                    s = np.frombuffer(shared_mem.get_obj(), dtype=np.float32);
                    assert(s.size == self._blob_counts[t]), '{} {}'.format(s.size, self._blob_counts)
                shared_mem_list[t].append(shared_mem);
        self._shared_mem_shape = self._data_shapes;
        #TODO:: extend to multiple prefetch processes by running a for loop here.
        #take care to set the random seed, randomization etc. before setting up each process so that the data is randomized for each processes.
        #we want to minimize the IPC here.
        #start the process now
        self._prefetch_process_name = "data prefetcher"
        self._prefetch_process_id_q = Queue(1);
        self._prefetch_process = BlobFetcher(self._prefetch_process_name, self._prefetch_process_id_q,\
                                    self._slots_used, self._slots_filled,\
                                    self._shared_mem_shape, self._num_tops, self.get_next_minibatch_helper)
        for c in range(self._max_queue_size):
            self._slots_used.put(c);
        self._prefetch_process.start();
        self._prefetch_process_id = self._prefetch_process_id_q.get();
        print 'prefetching enabled: %d'%(self._prefetch_process_id);
        print 'setting up prefetcher with queue size: %d'%(self._max_queue_size);
        def cleanup():
            print 'terminate BlobFetcher'
            self._prefetch_process.terminate()
            self._prefetch_process.join();
        atexit.register(cleanup)

    def check_prefetch_alive(self):
        try:
            os.kill(self._prefetch_process_id, 0) #not killing just poking to see if alive
        except err:
            #will raise exception if process is dead
            #can do something more intelligent here rather than raise the same error ...
            raise err

    def caffe_read_image(self, imname, shape, immean):
        #im = cv2.imread(imname,cv2.IMREAD_COLOR);
        #My PIL Image benchmarks show that it is 1.3x faster than opencv
        im = Image.open(imname); im = im[:,:,::-1];
        #resize the image
        h = im.shape[0]
        w = im.shape[1]
        if shape[2] != h or shape[3] != w or len(im.shape)!=3 or im.shape[2]!=3:
            im = scipy.misc.imresize(im, (int(shape[2]), int(shape[3]), shape[1]), interp='bilinear');
        im = im.astype(np.float32)
        im -= immean;
        im = np.transpose(im, axes = (2, 0, 1))
        return im;

    def setup(self, bottom, top):
        """Setup the KeyHDF5Layer."""
        layer_params = yaml.load(self.param_str); #new version of caffe
        self._keys_file = layer_params['keys_file'];
        self._hdf5_files = layer_params['label_files'];
        self._txt_files = layer_params['txt_files'];
        self._image_dir = layer_params['image_dir'];
        self._image_mean = np.array(layer_params['bgr_mean']);
        self._image_ext = layer_params['image_ext'];
        self._top_dtypes = layer_params['top_dtypes'];
        self._data_shapes = layer_params['data_shapes'];
        self._max_queue_size = layer_params['prefetch_size'];
        self._key_sep = '\t';
        if self._max_queue_size > 0:
            self._use_prefetch = True
        else:
            self._use_prefetch = False;
        self._max_queue_size = min(self._max_queue_size, 10);
        #do the tops make sense
        assert(len(top) == len(self._data_shapes));
        assert(len(top) == len(self._top_dtypes));
        for x in self._top_dtypes:
            assert(x in ['im', 'lab', 'txt']);
        self._num_im = self._top_dtypes.count('im');
        self._num_lab = self._top_dtypes.count('lab');
        self._num_txt = self._top_dtypes.count('txt');
        self._num_tops = len(self._top_dtypes)

        self._batch_size = self._data_shapes[0][0];
        t0shape = tuple(self._data_shapes[0]);
        self._blob_counts = [];
        self._single_batch_shapes = [];
        for d in range(len(top)):
            assert self._data_shapes[d][0] == self._batch_size;
            self._blob_counts.append(int(prod(self._data_shapes[d])))
            top[d].reshape(*(self._data_shapes[d]));
            self._single_batch_shapes.append([ 1,self._data_shapes[d][1], self._data_shapes[d][2],self._data_shapes[d][3]]);
        # tb.print_stack();namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)

        #verify keys and labels go together
        with open(self._keys_file,'r') as fh:
            self._data_keys = [line.strip().split(self._key_sep) for line in fh if len(line.strip()) >=1];
        for k in self._data_keys:
            assert(len(k) == self._num_tops);
        #now setup labels
        assert(self._num_lab == len(self._hdf5_files))
        self._hdf5_fhs = [];
        self._txts = [];
        hctr = 0;
        tctr = 0;
        for t in range(self._num_tops):
            fh = None;
            txt = None;
            if self._top_dtypes[t] == 'lab':
                fh = h5py.File(self._hdf5_files[hctr],'r')
                for x in self._data_keys:
                    assert(x[t] in fh)
                hctr+=1;
            elif self._top_dtypes[t] == 'txt':
                tfh = open(self._txt_files[tctr],'r')
                txt = [np.array(line.strip()) for line in tfh if len(line.strip()) >=1]
                assert(len(txt) == len(self._data_keys));
                tctr+=1
            self._hdf5_fhs.append(fh)
            self._txts.append(txt);
        assert(hctr == len(self._hdf5_files)), '%d %d'%(hctr, len(self._hdf5_files));
        assert(tctr == len(self._txt_files)), '%d %d'%(tctr, len(self._txt_files));

        #fetching and prefetching
        self._read_key_index = 0;
        self._disk_key_index = 0;
        if self._use_prefetch:
            self.setup_prefetch()

    def get_next_minibatch(self):
        if not self._use_prefetch:
            return self.get_next_minibatch_helper();
        #is child still alive?
        while self._slots_filled.empty():
            self.check_prefetch_alive();
        deq_slot = self._slots_filled.get();
        im_datas = [];
        for t in range(self._num_tops):
            shared_mem = shared_mem_list[t][deq_slot];
            with shared_mem.get_lock():
                    shared_mem_arr = np.reshape(np.frombuffer(shared_mem.get_obj(),dtype=np.float32), self._data_shapes[t]);
                    im_data = shared_mem_arr[...].astype(np.float32, copy=True); #copy since we will mark this slot as used
                    im_datas.append(im_data);
            # print 'fwd:: ', slot, im_datas[t].min(), im_datas[t].max(), im_datas[t].mean();
        self._read_key_index +=1;
        if self._read_key_index >= len(self._data_keys):
                print 'One epoch finished'
                self._read_key_index = 0;
        self._slots_used.put(deq_slot);
        return im_datas;

    def get_next_minibatch_helper(self):
        im_datas = [];
        for c in range(self._num_tops):
            im_datas.append(np.zeros(tuple(self._data_shapes[c]),dtype=np.float32));
        for b in range(self._batch_size):
            if self._disk_key_index >= len(self._data_keys):
                self._disk_key_index = 0;
            imBatchKeys = self._data_keys[self._disk_key_index];
            for t in range(self._num_tops):
                imName = imBatchKeys[t];
                if self._top_dtypes[t] == 'im':
                    imPath = os.path.join(self._image_dir, imName+self._image_ext);
                    im_datas[t][b,...]= self.caffe_read_image(imPath, shape=self._single_batch_shapes[t], immean=self._image_mean);
                elif self._top_dtypes[t] == 'lab':
                    im_datas[t][b,...]= self._hdf5_fhs[t][imName].value[:].reshape(self._single_batch_shapes[t]);
                elif self._top_dtypes[t] == 'txt':
                    im_datas[t][b,...]= self._txts[t][self._disk_key_index].reshape(self._single_batch_shapes[t]);
                else:
                    raise ValueError('uknown dtype: {}'.format(self._top_dtypes[t]));
            self._disk_key_index +=1;
        # tb.print_stack();namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)
        # print 'min_max:: ', im_datas.min(), im_datas.max(), im_datas.mean();
        return im_datas;

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        im_datas = self.get_next_minibatch();
        for c in range(self._num_tops):
            top[c].data[...] = im_datas[c].astype(np.float32, copy=False)
        # tb.print_stack();namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)
        # time.sleep(2);


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

#modification of Ross Girshick's ROI Data Layer from FRCNN
class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, process_name, process_pid_q, slots_used, slots_filled, shared_shapes, num_tops, func_to_call):
        super(BlobFetcher, self).__init__(name=process_name)
        self._shared_shapes = shared_shapes;
        self._num_tops = num_tops;
        self._slots_used = slots_used
        self._slots_filled = slots_filled;
        self._process_name = process_name
        self._funct_to_call = func_to_call;
        self._prefetch_process_id_q = process_pid_q;
        #self._parent_pid = os.getppid();
        #self._self_pid = os.getpid();
        self.exit = multiprocessing.Event()

    def self_cleanup(self):
        try:
            os.kill(self._parent_pid, 0) #not killing just poking to see if parent is alive
        except:
            #parent is dead and we are not, stop prefetching
            print 'prefetch %s (%d) : shutdown'%(self._process_name, self._self_pid)
            self.exit.set();
            sys.exit()

    def run(self):
        print 'BlobFetcher started: pid %d; ppid %d'%(os.getpid(), os.getppid())
        self._parent_pid = os.getppid()
        self._self_pid = os.getpid();
        self._prefetch_process_id_q.put(self._self_pid);
        global shared_mem_list
        while True:
            #blobs = get_minibatch(minibatch_db, self._num_classes)
            self.self_cleanup();
            if self._slots_used.empty():
                continue;
            slot = self._slots_used.get();
            im_datas = self._funct_to_call();
            for t in range(self._num_tops):
                shared_mem = shared_mem_list[t][slot];
                with shared_mem.get_lock():
                    s = np.frombuffer(shared_mem.get_obj(), dtype=np.float32);
                    # print s.size, self._shared_shapes[t];
                    shared_mem_arr = np.reshape(s, self._shared_shapes[t]);
                    shared_mem_arr[...] = im_datas[t].astype(np.float32, copy=True);
                    # print 'helper:: ',im_datas[t].min(), im_datas[t].max(), im_datas[t].mean()
            self._slots_filled.put(slot);
