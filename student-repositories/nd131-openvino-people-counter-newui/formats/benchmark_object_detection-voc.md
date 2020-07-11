Running with parameters:
    USE_CASE: object_detection
    FRAMEWORK: caffe
    WORKSPACE: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/common/caffe
    DATASET_LOCATION: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/pascal_voc_tfrecord/tfrecord-voc.record
    CHECKPOINT_DIRECTORY: 
    IN_GRAPH: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.prototxt
    SOCKET_ID: 0
    MODEL_NAME: detection_softmax
    MODE: inference
    PRECISION: fp32
    BATCH_SIZE: 1
    NUM_CORES: 2
    BENCHMARK_ONLY: True
    ACCURACY_ONLY: False
    OUTPUT_RESULTS: False
    DISABLE_TCMALLOC: True
    TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD: 2147483648
    NOINSTALL: False
    OUTPUT_DIR: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord
    MPI_NUM_PROCESSES: None
    MPI_NUM_PEOCESSES_PER_SOCKET: 1

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

Reading package lists...
E: Could not open lock file /var/lib/apt/lists/lock - open (13: Permission denied)
E: Unable to lock directory /var/lib/apt/lists/
W: Problem unlinking the file /var/cache/apt/pkgcache.bin - RemoveCaches (13: Permission denied)
W: Problem unlinking the file /var/cache/apt/srcpkgcache.bin - RemoveCaches (13: Permission denied)

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?
E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?
update-alternatives: error: unable to create file '/var/lib/dpkg/alternatives/gcc.dpkg-tmp': Permission denied
update-alternatives: error: unable to create file '/var/lib/dpkg/alternatives/gcc.dpkg-tmp': Permission denied

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?
Requirement already up-to-date: pip in /home/aswin/anaconda3/envs/new_env/lib/python3.6/site-packages (20.1)
Requirement already satisfied: requests in /home/aswin/anaconda3/envs/new_env/lib/python3.6/site-packages (2.23.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/aswin/anaconda3/envs/new_env/lib/python3.6/site-packages (from requests) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /home/aswin/anaconda3/envs/new_env/lib/python3.6/site-packages (from requests) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /home/aswin/anaconda3/envs/new_env/lib/python3.6/site-packages (from requests) (2019.11.28)
Requirement already satisfied: idna<3,>=2.5 in /home/aswin/anaconda3/envs/new_env/lib/python3.6/site-packages (from requests) (2.9)
Installing caffe requirements..\n
Check whether the caffe model is present..\n
/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/common/caffe/start.sh: line 80: [: abcd: integer expression expected
Caffe Python Module is not available..\n
Log output location: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord/benchmark_detection_softmax_inference_fp32_20200518_222334.log
benchmarking
/usr/bin/python3.6 common/caffe/run_tf_benchmark.py --framework=caffe --use-case=object_detection --model-name=detection_softmax --precision=fp32 --mode=inference --benchmark-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks --intelai-models=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/detection_softmax --num-cores=2 --batch-size=1 --socket-id=0 --output-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord --annotations_dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/caffe2-pose-estimation/annotations/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages  --benchmark-only  --verbose --model-source-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet --in-graph=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.prototxt --in-weights=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.caffemodel --data-location=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/pascal_voc_tfrecord/tfrecord-voc.record --num-inter-threads=1 --num-intra-threads=1 --disable-tcmalloc=True                  
2020-05-18 22:23:36.847882: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2993060000 Hz
2020-05-18 22:23:36.848256: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x255a210 executing computations on platform Host. Devices:
2020-05-18 22:23:36.848286: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
WARNING: Logging before InitGoogleLogging() is written to STDERR
W0518 22:23:36.849476 26607 _caffe.cpp:139] DEPRECATION WARNING - deprecated use of Python interface
W0518 22:23:36.849504 26607 _caffe.cpp:140] Use this instead (with the named "weights" parameter):
W0518 22:23:36.849510 26607 _caffe.cpp:142] Net('/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.prototxt', 1, weights='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.caffemodel')
I0518 22:23:36.852499 26607 net.cpp:51] Initializing net from parameters: 
state {
  phase: TEST
  level: 0
}
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 224
      dim: 224
    }
  }
}
layer {
  name: "conv1_1"
  type: "Convolution"
  bottom: "data"
  top: "conv1_1"
  convolution_param {
    num_output: 64
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 3
    kernel_w: 3
    stride_h: 2
    stride_w: 2
    dilation: 1
  }
}
layer {
  name: "conv1_2"
  type: "ReLU"
  bottom: "conv1_1"
  top: "conv1_2"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1_2"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_h: 3
    kernel_w: 3
    stride_h: 2
    stride_w: 2
    pad_h: 0
    pad_w: 0
  }
}
layer {
  name: "fire2/squeeze1x1_1"
  type: "Convolution"
  bottom: "pool1"
  top: "fire2/squeeze1x1_1"
  convolution_param {
    num_output: 16
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire2/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire2/squeeze1x1_1"
  top: "fire2/squeeze1x1_2"
}
layer {
  name: "fire2/expand1x1_1"
  type: "Convolution"
  bottom: "fire2/squeeze1x1_2"
  top: "fire2/expand1x1_1"
  convolution_param {
    num_output: 64
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire2/expand1x1_2"
  type: "ReLU"
  bottom: "fire2/expand1x1_1"
  top: "fire2/expand1x1_2"
}
layer {
  name: "fire2/expand3x3_1"
  type: "Convolution"
  bottom: "fire2/squeeze1x1_2"
  top: "fire2/expand3x3_1"
  convolution_param {
    num_output: 64
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire2/expand3x3_2"
  type: "ReLU"
  bottom: "fire2/expand3x3_1"
  top: "fire2/expand3x3_2"
}
layer {
  name: "fire2/concat"
  type: "Concat"
  bottom: "fire2/expand1x1_2"
  bottom: "fire2/expand3x3_2"
  top: "fire2/concat"
  concat_param {
    axis: 1
  }
}
layer {
  name: "fire3/squeeze1x1_1"
  type: "Convolution"
  bottom: "fire2/concat"
  top: "fire3/squeeze1x1_1"
  convolution_param {
    num_output: 16
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire3/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire3/squeeze1x1_1"
  top: "fire3/squeeze1x1_2"
}
layer {
  name: "fire3/expand1x1_1"
  type: "Convolution"
  bottom: "fire3/squeeze1x1_2"
  top: "fire3/expand1x1_1"
  convolution_param {
    num_output: 64
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire3/expand1x1_2"
  type: "ReLU"
  bottom: "fire3/expand1x1_1"
  top: "fire3/expand1x1_2"
}
layer {
  name: "fire3/expand3x3_1"
  type: "Convolution"
  bottom: "fire3/squeeze1x1_2"
  top: "fire3/expand3x3_1"
  convolution_param {
    num_output: 64
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire3/expand3x3_2"
  type: "ReLU"
  bottom: "fire3/expand3x3_1"
  top: "fire3/expand3x3_2"
}
layer {
  name: "fire3/concat"
  type: "Concat"
  bottom: "fire3/expand1x1_2"
  bottom: "fire3/expand3x3_2"
  top: "fire3/concat"
  concat_param {
    axis: 1
  }
}
layer {
  name: "pool3"
  type: "Pooling"
  bottom: "fire3/concat"
  top: "pool3"
  pooling_param {
    pool: MAX
    kernel_h: 3
    kernel_w: 3
    stride_h: 2
    stride_w: 2
    pad_h: 0
    pad_w: 0
  }
}
layer {
  name: "fire4/squeeze1x1_1"
  type: "Convolution"
  bottom: "pool3"
  top: "fire4/squeeze1x1_1"
  convolution_param {
    num_output: 32
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire4/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire4/squeeze1x1_1"
  top: "fire4/squeeze1x1_2"
}
layer {
  name: "fire4/expand1x1_1"
  type: "Convolution"
  bottom: "fire4/squeeze1x1_2"
  top: "fire4/expand1x1_1"
  convolution_param {
    num_output: 128
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire4/expand1x1_2"
  type: "ReLU"
  bottom: "fire4/expand1x1_1"
  top: "fire4/expand1x1_2"
}
layer {
  name: "fire4/expand3x3_1"
  type: "Convolution"
  bottom: "fire4/squeeze1x1_2"
  top: "fire4/expand3x3_1"
  convolution_param {
    num_output: 128
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire4/expand3x3_2"
  type: "ReLU"
  bottom: "fire4/expand3x3_1"
  top: "fire4/expand3x3_2"
}
layer {
  name: "fire4/concat"
  type: "Concat"
  bottom: "fire4/expand1x1_2"
  bottom: "fire4/expand3x3_2"
  top: "fire4/concat"
  concat_param {
    axis: 1
  }
}
layer {
  name: "fire5/squeeze1x1_1"
  type: "Convolution"
  bottom: "fire4/concat"
  top: "fire5/squeeze1x1_1"
  convolution_param {
    num_output: 32
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire5/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire5/squeeze1x1_1"
  top: "fire5/squeeze1x1_2"
}
layer {
  name: "fire5/expand1x1_1"
  type: "Convolution"
  bottom: "fire5/squeeze1x1_2"
  top: "fire5/expand1x1_1"
  convolution_param {
    num_output: 128
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire5/expand1x1_2"
  type: "ReLU"
  bottom: "fire5/expand1x1_1"
  top: "fire5/expand1x1_2"
}
layer {
  name: "fire5/expand3x3_1"
  type: "Convolution"
  bottom: "fire5/squeeze1x1_2"
  top: "fire5/expand3x3_1"
  convolution_param {
    num_output: 128
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire5/expand3x3_2"
  type: "ReLU"
  bottom: "fire5/expand3x3_1"
  top: "fire5/expand3x3_2"
}
layer {
  name: "fire5/concat"
  type: "Concat"
  bottom: "fire5/expand1x1_2"
  bottom: "fire5/expand3x3_2"
  top: "fire5/concat"
  concat_param {
    axis: 1
  }
}
layer {
  name: "pool5"
  type: "Pooling"
  bottom: "fire5/concat"
  top: "pool5"
  pooling_param {
    pool: MAX
    kernel_h: 3
    kernel_w: 3
    stride_h: 2
    stride_w: 2
    pad_h: 0
    pad_w: 0
  }
}
layer {
  name: "fire6/squeeze1x1_1"
  type: "Convolution"
  bottom: "pool5"
  top: "fire6/squeeze1x1_1"
  convolution_param {
    num_output: 48
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire6/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire6/squeeze1x1_1"
  top: "fire6/squeeze1x1_2"
}
layer {
  name: "fire6/expand1x1_1"
  type: "Convolution"
  bottom: "fire6/squeeze1x1_2"
  top: "fire6/expand1x1_1"
  convolution_param {
    num_output: 192
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire6/expand1x1_2"
  type: "ReLU"
  bottom: "fire6/expand1x1_1"
  top: "fire6/expand1x1_2"
}
layer {
  name: "fire6/expand3x3_1"
  type: "Convolution"
  bottom: "fire6/squeeze1x1_2"
  top: "fire6/expand3x3_1"
  convolution_param {
    num_output: 192
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire6/expand3x3_2"
  type: "ReLU"
  bottom: "fire6/expand3x3_1"
  top: "fire6/expand3x3_2"
}
layer {
  name: "fire6/concat"
  type: "Concat"
  bottom: "fire6/expand1x1_2"
  bottom: "fire6/expand3x3_2"
  top: "fire6/concat"
  concat_param {
    axis: 1
  }
}
layer {
  name: "fire7/squeeze1x1_1"
  type: "Convolution"
  bottom: "fire6/concat"
  top: "fire7/squeeze1x1_1"
  convolution_param {
    num_output: 48
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire7/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire7/squeeze1x1_1"
  top: "fire7/squeeze1x1_2"
}
layer {
  name: "fire7/expand1x1_1"
  type: "Convolution"
  bottom: "fire7/squeeze1x1_2"
  top: "fire7/expand1x1_1"
  convolution_param {
    num_output: 192
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire7/expand1x1_2"
  type: "ReLU"
  bottom: "fire7/expand1x1_1"
  top: "fire7/expand1x1_2"
}
layer {
  name: "fire7/expand3x3_1"
  type: "Convolution"
  bottom: "fire7/squeeze1x1_2"
  top: "fire7/expand3x3_1"
  convolution_param {
    num_output: 192
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire7/expand3x3_2"
  type: "ReLU"
  bottom: "fire7/expand3x3_1"
  top: "fire7/expand3x3_2"
}
layer {
  name: "fire7/concat"
  type: "Concat"
  bottom: "fire7/expand1x1_2"
  bottom: "fire7/expand3x3_2"
  top: "fire7/concat"
  concat_param {
    axis: 1
  }
}
layer {
  name: "fire8/squeeze1x1_1"
  type: "Convolution"
  bottom: "fire7/concat"
  top: "fire8/squeeze1x1_1"
  convolution_param {
    num_output: 64
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire8/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire8/squeeze1x1_1"
  top: "fire8/squeeze1x1_2"
}
layer {
  name: "fire8/expand1x1_1"
  type: "Convolution"
  bottom: "fire8/squeeze1x1_2"
  top: "fire8/expand1x1_1"
  convolution_param {
    num_output: 256
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire8/expand1x1_2"
  type: "ReLU"
  bottom: "fire8/expand1x1_1"
  top: "fire8/expand1x1_2"
}
layer {
  name: "fire8/expand3x3_1"
  type: "Convolution"
  bottom: "fire8/squeeze1x1_2"
  top: "fire8/expand3x3_1"
  convolution_param {
    num_output: 256
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire8/expand3x3_2"
  type: "ReLU"
  bottom: "fire8/expand3x3_1"
  top: "fire8/expand3x3_2"
}
layer {
  name: "fire8/concat"
  type: "Concat"
  bottom: "fire8/expand1x1_2"
  bottom: "fire8/expand3x3_2"
  top: "fire8/concat"
  concat_param {
    axis: 1
  }
}
layer {
  name: "fire9/squeeze1x1_1"
  type: "Convolution"
  bottom: "fire8/concat"
  top: "fire9/squeeze1x1_1"
  convolution_param {
    num_output: 64
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire9/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire9/squeeze1x1_1"
  top: "fire9/squeeze1x1_2"
}
layer {
  name: "fire9/expand1x1_1"
  type: "Convolution"
  bottom: "fire9/squeeze1x1_2"
  top: "fire9/expand1x1_1"
  convolution_param {
    num_output: 256
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire9/expand1x1_2"
  type: "ReLU"
  bottom: "fire9/expand1x1_1"
  top: "fire9/expand1x1_2"
}
layer {
  name: "fire9/expand3x3_1"
  type: "Convolution"
  bottom: "fire9/squeeze1x1_2"
  top: "fire9/expand3x3_1"
  convolution_param {
    num_output: 256
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire9/expand3x3_2"
  type: "ReLU"
  bottom: "fire9/expand3x3_1"
  top: "fire9/expand3x3_2"
}
layer {
  name: "fire9/concat_1"
  type: "Concat"
  bottom: "fire9/expand1x1_2"
  bottom: "fire9/expand3x3_2"
  top: "fire9/concat_1"
  concat_param {
    axis: 1
  }
}
layer {
  name: "fire9/concat_2__fire9/concat_mask"
  type: "Dropout"
  bottom: "fire9/concat_1"
  top: "fire9/concat_2"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "conv10_1"
  type: "Convolution"
  bottom: "fire9/concat_2"
  top: "conv10_1"
  convolution_param {
    num_output: 1000
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "conv10_2"
  type: "ReLU"
  bottom: "conv10_1"
  top: "conv10_2"
}
layer {
  name: "pool10"
  type: "Pooling"
  bottom: "conv10_2"
  top: "pool10"
  pooling_param {
    pool: AVE
    global_pooling: true
  }
}
layer {
  name: "softmaxout"
  type: "Softmax"
  bottom: "pool10"
  top: "softmaxout"
}
I0518 22:23:36.853324 26607 layer_factory.hpp:77] Creating layer data
I0518 22:23:36.853391 26607 net.cpp:84] Creating Layer data
I0518 22:23:36.853425 26607 net.cpp:380] data -> data
I0518 22:23:36.853471 26607 net.cpp:122] Setting up data
I0518 22:23:36.853482 26607 net.cpp:129] Top shape: 1 3 224 224 (150528)
I0518 22:23:36.853492 26607 net.cpp:137] Memory required for data: 602112
I0518 22:23:36.853499 26607 layer_factory.hpp:77] Creating layer conv1_1
I0518 22:23:36.853511 26607 net.cpp:84] Creating Layer conv1_1
I0518 22:23:36.853539 26607 net.cpp:406] conv1_1 <- data
I0518 22:23:36.853554 26607 net.cpp:380] conv1_1 -> conv1_1
I0518 22:23:36.853641 26607 net.cpp:122] Setting up conv1_1
I0518 22:23:36.853674 26607 net.cpp:129] Top shape: 1 64 111 111 (788544)
I0518 22:23:36.853693 26607 net.cpp:137] Memory required for data: 3756288
I0518 22:23:36.853718 26607 layer_factory.hpp:77] Creating layer conv1_2
I0518 22:23:36.853739 26607 net.cpp:84] Creating Layer conv1_2
I0518 22:23:36.853756 26607 net.cpp:406] conv1_2 <- conv1_1
I0518 22:23:36.853775 26607 net.cpp:380] conv1_2 -> conv1_2
I0518 22:23:36.853801 26607 net.cpp:122] Setting up conv1_2
I0518 22:23:36.853821 26607 net.cpp:129] Top shape: 1 64 111 111 (788544)
I0518 22:23:36.853838 26607 net.cpp:137] Memory required for data: 6910464
I0518 22:23:36.853855 26607 layer_factory.hpp:77] Creating layer pool1
I0518 22:23:36.853874 26607 net.cpp:84] Creating Layer pool1
I0518 22:23:36.853893 26607 net.cpp:406] pool1 <- conv1_2
I0518 22:23:36.853910 26607 net.cpp:380] pool1 -> pool1
I0518 22:23:36.853924 26607 net.cpp:122] Setting up pool1
I0518 22:23:36.853943 26607 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0518 22:23:36.853953 26607 net.cpp:137] Memory required for data: 7684864
I0518 22:23:36.853958 26607 layer_factory.hpp:77] Creating layer fire2/squeeze1x1_1
I0518 22:23:36.853971 26607 net.cpp:84] Creating Layer fire2/squeeze1x1_1
I0518 22:23:36.853991 26607 net.cpp:406] fire2/squeeze1x1_1 <- pool1
I0518 22:23:36.854010 26607 net.cpp:380] fire2/squeeze1x1_1 -> fire2/squeeze1x1_1
I0518 22:23:36.854058 26607 net.cpp:122] Setting up fire2/squeeze1x1_1
I0518 22:23:36.854077 26607 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0518 22:23:36.854096 26607 net.cpp:137] Memory required for data: 7878464
I0518 22:23:36.854126 26607 layer_factory.hpp:77] Creating layer fire2/squeeze1x1_2
I0518 22:23:36.854146 26607 net.cpp:84] Creating Layer fire2/squeeze1x1_2
I0518 22:23:36.854158 26607 net.cpp:406] fire2/squeeze1x1_2 <- fire2/squeeze1x1_1
I0518 22:23:36.854166 26607 net.cpp:380] fire2/squeeze1x1_2 -> fire2/squeeze1x1_2
I0518 22:23:36.854174 26607 net.cpp:122] Setting up fire2/squeeze1x1_2
I0518 22:23:36.854192 26607 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0518 22:23:36.854212 26607 net.cpp:137] Memory required for data: 8072064
I0518 22:23:36.854228 26607 layer_factory.hpp:77] Creating layer fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split
I0518 22:23:36.854252 26607 net.cpp:84] Creating Layer fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split
I0518 22:23:36.854270 26607 net.cpp:406] fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split <- fire2/squeeze1x1_2
I0518 22:23:36.854280 26607 net.cpp:380] fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split -> fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split_0
I0518 22:23:36.854290 26607 net.cpp:380] fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split -> fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split_1
I0518 22:23:36.854316 26607 net.cpp:122] Setting up fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split
I0518 22:23:36.854334 26607 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0518 22:23:36.854353 26607 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0518 22:23:36.854372 26607 net.cpp:137] Memory required for data: 8459264
I0518 22:23:36.854388 26607 layer_factory.hpp:77] Creating layer fire2/expand1x1_1
I0518 22:23:36.854413 26607 net.cpp:84] Creating Layer fire2/expand1x1_1
I0518 22:23:36.854431 26607 net.cpp:406] fire2/expand1x1_1 <- fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split_0
I0518 22:23:36.854444 26607 net.cpp:380] fire2/expand1x1_1 -> fire2/expand1x1_1
I0518 22:23:36.854473 26607 net.cpp:122] Setting up fire2/expand1x1_1
I0518 22:23:36.854494 26607 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0518 22:23:36.854513 26607 net.cpp:137] Memory required for data: 9233664
I0518 22:23:36.854533 26607 layer_factory.hpp:77] Creating layer fire2/expand1x1_2
I0518 22:23:36.854553 26607 net.cpp:84] Creating Layer fire2/expand1x1_2
I0518 22:23:36.854570 26607 net.cpp:406] fire2/expand1x1_2 <- fire2/expand1x1_1
I0518 22:23:36.854589 26607 net.cpp:380] fire2/expand1x1_2 -> fire2/expand1x1_2
I0518 22:23:36.854610 26607 net.cpp:122] Setting up fire2/expand1x1_2
I0518 22:23:36.854627 26607 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0518 22:23:36.854646 26607 net.cpp:137] Memory required for data: 10008064
I0518 22:23:36.854662 26607 layer_factory.hpp:77] Creating layer fire2/expand3x3_1
I0518 22:23:36.854681 26607 net.cpp:84] Creating Layer fire2/expand3x3_1
I0518 22:23:36.854691 26607 net.cpp:406] fire2/expand3x3_1 <- fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split_1
I0518 22:23:36.854698 26607 net.cpp:380] fire2/expand3x3_1 -> fire2/expand3x3_1
I0518 22:23:36.854749 26607 net.cpp:122] Setting up fire2/expand3x3_1
I0518 22:23:36.854771 26607 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0518 22:23:36.854791 26607 net.cpp:137] Memory required for data: 10782464
I0518 22:23:36.854810 26607 layer_factory.hpp:77] Creating layer fire2/expand3x3_2
I0518 22:23:36.854821 26607 net.cpp:84] Creating Layer fire2/expand3x3_2
I0518 22:23:36.854827 26607 net.cpp:406] fire2/expand3x3_2 <- fire2/expand3x3_1
I0518 22:23:36.854835 26607 net.cpp:380] fire2/expand3x3_2 -> fire2/expand3x3_2
I0518 22:23:36.854846 26607 net.cpp:122] Setting up fire2/expand3x3_2
I0518 22:23:36.854851 26607 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0518 22:23:36.854858 26607 net.cpp:137] Memory required for data: 11556864
I0518 22:23:36.854862 26607 layer_factory.hpp:77] Creating layer fire2/concat
I0518 22:23:36.854874 26607 net.cpp:84] Creating Layer fire2/concat
I0518 22:23:36.854894 26607 net.cpp:406] fire2/concat <- fire2/expand1x1_2
I0518 22:23:36.854912 26607 net.cpp:406] fire2/concat <- fire2/expand3x3_2
I0518 22:23:36.854925 26607 net.cpp:380] fire2/concat -> fire2/concat
I0518 22:23:36.854938 26607 net.cpp:122] Setting up fire2/concat
I0518 22:23:36.854944 26607 net.cpp:129] Top shape: 1 128 55 55 (387200)
I0518 22:23:36.854950 26607 net.cpp:137] Memory required for data: 13105664
I0518 22:23:36.854955 26607 layer_factory.hpp:77] Creating layer fire3/squeeze1x1_1
I0518 22:23:36.854965 26607 net.cpp:84] Creating Layer fire3/squeeze1x1_1
I0518 22:23:36.854974 26607 net.cpp:406] fire3/squeeze1x1_1 <- fire2/concat
I0518 22:23:36.854981 26607 net.cpp:380] fire3/squeeze1x1_1 -> fire3/squeeze1x1_1
I0518 22:23:36.855015 26607 net.cpp:122] Setting up fire3/squeeze1x1_1
I0518 22:23:36.855036 26607 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0518 22:23:36.855046 26607 net.cpp:137] Memory required for data: 13299264
I0518 22:23:36.855053 26607 layer_factory.hpp:77] Creating layer fire3/squeeze1x1_2
I0518 22:23:36.855060 26607 net.cpp:84] Creating Layer fire3/squeeze1x1_2
I0518 22:23:36.855069 26607 net.cpp:406] fire3/squeeze1x1_2 <- fire3/squeeze1x1_1
I0518 22:23:36.855077 26607 net.cpp:380] fire3/squeeze1x1_2 -> fire3/squeeze1x1_2
I0518 22:23:36.855087 26607 net.cpp:122] Setting up fire3/squeeze1x1_2
I0518 22:23:36.855095 26607 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0518 22:23:36.855103 26607 net.cpp:137] Memory required for data: 13492864
I0518 22:23:36.855108 26607 layer_factory.hpp:77] Creating layer fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split
I0518 22:23:36.855113 26607 net.cpp:84] Creating Layer fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split
I0518 22:23:36.855118 26607 net.cpp:406] fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split <- fire3/squeeze1x1_2
I0518 22:23:36.855125 26607 net.cpp:380] fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split -> fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split_0
I0518 22:23:36.855134 26607 net.cpp:380] fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split -> fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split_1
I0518 22:23:36.855144 26607 net.cpp:122] Setting up fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split
I0518 22:23:36.855149 26607 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0518 22:23:36.855154 26607 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0518 22:23:36.855160 26607 net.cpp:137] Memory required for data: 13880064
I0518 22:23:36.855165 26607 layer_factory.hpp:77] Creating layer fire3/expand1x1_1
I0518 22:23:36.855176 26607 net.cpp:84] Creating Layer fire3/expand1x1_1
I0518 22:23:36.855181 26607 net.cpp:406] fire3/expand1x1_1 <- fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split_0
I0518 22:23:36.855191 26607 net.cpp:380] fire3/expand1x1_1 -> fire3/expand1x1_1
I0518 22:23:36.855221 26607 net.cpp:122] Setting up fire3/expand1x1_1
I0518 22:23:36.855226 26607 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0518 22:23:36.855232 26607 net.cpp:137] Memory required for data: 14654464
I0518 22:23:36.855239 26607 layer_factory.hpp:77] Creating layer fire3/expand1x1_2
I0518 22:23:36.855247 26607 net.cpp:84] Creating Layer fire3/expand1x1_2
I0518 22:23:36.855252 26607 net.cpp:406] fire3/expand1x1_2 <- fire3/expand1x1_1
I0518 22:23:36.855259 26607 net.cpp:380] fire3/expand1x1_2 -> fire3/expand1x1_2
I0518 22:23:36.855268 26607 net.cpp:122] Setting up fire3/expand1x1_2
I0518 22:23:36.855273 26607 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0518 22:23:36.855278 26607 net.cpp:137] Memory required for data: 15428864
I0518 22:23:36.855283 26607 layer_factory.hpp:77] Creating layer fire3/expand3x3_1
I0518 22:23:36.855290 26607 net.cpp:84] Creating Layer fire3/expand3x3_1
I0518 22:23:36.855295 26607 net.cpp:406] fire3/expand3x3_1 <- fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split_1
I0518 22:23:36.855304 26607 net.cpp:380] fire3/expand3x3_1 -> fire3/expand3x3_1
I0518 22:23:36.855370 26607 net.cpp:122] Setting up fire3/expand3x3_1
I0518 22:23:36.855391 26607 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0518 22:23:36.855410 26607 net.cpp:137] Memory required for data: 16203264
I0518 22:23:36.855429 26607 layer_factory.hpp:77] Creating layer fire3/expand3x3_2
I0518 22:23:36.855451 26607 net.cpp:84] Creating Layer fire3/expand3x3_2
I0518 22:23:36.855469 26607 net.cpp:406] fire3/expand3x3_2 <- fire3/expand3x3_1
I0518 22:23:36.855489 26607 net.cpp:380] fire3/expand3x3_2 -> fire3/expand3x3_2
I0518 22:23:36.855500 26607 net.cpp:122] Setting up fire3/expand3x3_2
I0518 22:23:36.855505 26607 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0518 22:23:36.855512 26607 net.cpp:137] Memory required for data: 16977664
I0518 22:23:36.855516 26607 layer_factory.hpp:77] Creating layer fire3/concat
I0518 22:23:36.855526 26607 net.cpp:84] Creating Layer fire3/concat
I0518 22:23:36.855535 26607 net.cpp:406] fire3/concat <- fire3/expand1x1_2
I0518 22:23:36.855540 26607 net.cpp:406] fire3/concat <- fire3/expand3x3_2
I0518 22:23:36.855547 26607 net.cpp:380] fire3/concat -> fire3/concat
I0518 22:23:36.855556 26607 net.cpp:122] Setting up fire3/concat
I0518 22:23:36.855562 26607 net.cpp:129] Top shape: 1 128 55 55 (387200)
I0518 22:23:36.855568 26607 net.cpp:137] Memory required for data: 18526464
I0518 22:23:36.855573 26607 layer_factory.hpp:77] Creating layer pool3
I0518 22:23:36.855588 26607 net.cpp:84] Creating Layer pool3
I0518 22:23:36.855594 26607 net.cpp:406] pool3 <- fire3/concat
I0518 22:23:36.855602 26607 net.cpp:380] pool3 -> pool3
I0518 22:23:36.855612 26607 net.cpp:122] Setting up pool3
I0518 22:23:36.855618 26607 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0518 22:23:36.855624 26607 net.cpp:137] Memory required for data: 18899712
I0518 22:23:36.855628 26607 layer_factory.hpp:77] Creating layer fire4/squeeze1x1_1
I0518 22:23:36.855639 26607 net.cpp:84] Creating Layer fire4/squeeze1x1_1
I0518 22:23:36.855646 26607 net.cpp:406] fire4/squeeze1x1_1 <- pool3
I0518 22:23:36.855655 26607 net.cpp:380] fire4/squeeze1x1_1 -> fire4/squeeze1x1_1
I0518 22:23:36.855710 26607 net.cpp:122] Setting up fire4/squeeze1x1_1
I0518 22:23:36.855720 26607 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0518 22:23:36.855727 26607 net.cpp:137] Memory required for data: 18993024
I0518 22:23:36.855734 26607 layer_factory.hpp:77] Creating layer fire4/squeeze1x1_2
I0518 22:23:36.855741 26607 net.cpp:84] Creating Layer fire4/squeeze1x1_2
I0518 22:23:36.855749 26607 net.cpp:406] fire4/squeeze1x1_2 <- fire4/squeeze1x1_1
I0518 22:23:36.855756 26607 net.cpp:380] fire4/squeeze1x1_2 -> fire4/squeeze1x1_2
I0518 22:23:36.855767 26607 net.cpp:122] Setting up fire4/squeeze1x1_2
I0518 22:23:36.855773 26607 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0518 22:23:36.855780 26607 net.cpp:137] Memory required for data: 19086336
I0518 22:23:36.855784 26607 layer_factory.hpp:77] Creating layer fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split
I0518 22:23:36.855792 26607 net.cpp:84] Creating Layer fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split
I0518 22:23:36.855796 26607 net.cpp:406] fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split <- fire4/squeeze1x1_2
I0518 22:23:36.855806 26607 net.cpp:380] fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split -> fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split_0
I0518 22:23:36.855831 26607 net.cpp:380] fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split -> fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split_1
I0518 22:23:36.855854 26607 net.cpp:122] Setting up fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split
I0518 22:23:36.855870 26607 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0518 22:23:36.855880 26607 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0518 22:23:36.855886 26607 net.cpp:137] Memory required for data: 19272960
I0518 22:23:36.855890 26607 layer_factory.hpp:77] Creating layer fire4/expand1x1_1
I0518 22:23:36.855901 26607 net.cpp:84] Creating Layer fire4/expand1x1_1
I0518 22:23:36.855908 26607 net.cpp:406] fire4/expand1x1_1 <- fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split_0
I0518 22:23:36.855916 26607 net.cpp:380] fire4/expand1x1_1 -> fire4/expand1x1_1
I0518 22:23:36.855948 26607 net.cpp:122] Setting up fire4/expand1x1_1
I0518 22:23:36.855969 26607 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0518 22:23:36.855988 26607 net.cpp:137] Memory required for data: 19646208
I0518 22:23:36.855999 26607 layer_factory.hpp:77] Creating layer fire4/expand1x1_2
I0518 22:23:36.856009 26607 net.cpp:84] Creating Layer fire4/expand1x1_2
I0518 22:23:36.856014 26607 net.cpp:406] fire4/expand1x1_2 <- fire4/expand1x1_1
I0518 22:23:36.856021 26607 net.cpp:380] fire4/expand1x1_2 -> fire4/expand1x1_2
I0518 22:23:36.856029 26607 net.cpp:122] Setting up fire4/expand1x1_2
I0518 22:23:36.856034 26607 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0518 22:23:36.856042 26607 net.cpp:137] Memory required for data: 20019456
I0518 22:23:36.856047 26607 layer_factory.hpp:77] Creating layer fire4/expand3x3_1
I0518 22:23:36.856060 26607 net.cpp:84] Creating Layer fire4/expand3x3_1
I0518 22:23:36.856066 26607 net.cpp:406] fire4/expand3x3_1 <- fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split_1
I0518 22:23:36.856074 26607 net.cpp:380] fire4/expand3x3_1 -> fire4/expand3x3_1
I0518 22:23:36.856182 26607 net.cpp:122] Setting up fire4/expand3x3_1
I0518 22:23:36.856191 26607 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0518 22:23:36.856199 26607 net.cpp:137] Memory required for data: 20392704
I0518 22:23:36.856205 26607 layer_factory.hpp:77] Creating layer fire4/expand3x3_2
I0518 22:23:36.856218 26607 net.cpp:84] Creating Layer fire4/expand3x3_2
I0518 22:23:36.856225 26607 net.cpp:406] fire4/expand3x3_2 <- fire4/expand3x3_1
I0518 22:23:36.856232 26607 net.cpp:380] fire4/expand3x3_2 -> fire4/expand3x3_2
I0518 22:23:36.856242 26607 net.cpp:122] Setting up fire4/expand3x3_2
I0518 22:23:36.856249 26607 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0518 22:23:36.856256 26607 net.cpp:137] Memory required for data: 20765952
I0518 22:23:36.856261 26607 layer_factory.hpp:77] Creating layer fire4/concat
I0518 22:23:36.856271 26607 net.cpp:84] Creating Layer fire4/concat
I0518 22:23:36.856277 26607 net.cpp:406] fire4/concat <- fire4/expand1x1_2
I0518 22:23:36.856283 26607 net.cpp:406] fire4/concat <- fire4/expand3x3_2
I0518 22:23:36.856289 26607 net.cpp:380] fire4/concat -> fire4/concat
I0518 22:23:36.856297 26607 net.cpp:122] Setting up fire4/concat
I0518 22:23:36.856304 26607 net.cpp:129] Top shape: 1 256 27 27 (186624)
I0518 22:23:36.856310 26607 net.cpp:137] Memory required for data: 21512448
I0518 22:23:36.856314 26607 layer_factory.hpp:77] Creating layer fire5/squeeze1x1_1
I0518 22:23:36.856324 26607 net.cpp:84] Creating Layer fire5/squeeze1x1_1
I0518 22:23:36.856331 26607 net.cpp:406] fire5/squeeze1x1_1 <- fire4/concat
I0518 22:23:36.856339 26607 net.cpp:380] fire5/squeeze1x1_1 -> fire5/squeeze1x1_1
I0518 22:23:36.856384 26607 net.cpp:122] Setting up fire5/squeeze1x1_1
I0518 22:23:36.856391 26607 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0518 22:23:36.856398 26607 net.cpp:137] Memory required for data: 21605760
I0518 22:23:36.856405 26607 layer_factory.hpp:77] Creating layer fire5/squeeze1x1_2
I0518 22:23:36.856412 26607 net.cpp:84] Creating Layer fire5/squeeze1x1_2
I0518 22:23:36.856436 26607 net.cpp:406] fire5/squeeze1x1_2 <- fire5/squeeze1x1_1
I0518 22:23:36.856458 26607 net.cpp:380] fire5/squeeze1x1_2 -> fire5/squeeze1x1_2
I0518 22:23:36.856479 26607 net.cpp:122] Setting up fire5/squeeze1x1_2
I0518 22:23:36.856496 26607 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0518 22:23:36.856515 26607 net.cpp:137] Memory required for data: 21699072
I0518 22:23:36.856531 26607 layer_factory.hpp:77] Creating layer fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split
I0518 22:23:36.856550 26607 net.cpp:84] Creating Layer fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split
I0518 22:23:36.856567 26607 net.cpp:406] fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split <- fire5/squeeze1x1_2
I0518 22:23:36.856588 26607 net.cpp:380] fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split -> fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split_0
I0518 22:23:36.856600 26607 net.cpp:380] fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split -> fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split_1
I0518 22:23:36.856609 26607 net.cpp:122] Setting up fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split
I0518 22:23:36.856627 26607 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0518 22:23:36.856647 26607 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0518 22:23:36.856665 26607 net.cpp:137] Memory required for data: 21885696
I0518 22:23:36.856681 26607 layer_factory.hpp:77] Creating layer fire5/expand1x1_1
I0518 22:23:36.856707 26607 net.cpp:84] Creating Layer fire5/expand1x1_1
I0518 22:23:36.856715 26607 net.cpp:406] fire5/expand1x1_1 <- fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split_0
I0518 22:23:36.856724 26607 net.cpp:380] fire5/expand1x1_1 -> fire5/expand1x1_1
I0518 22:23:36.856751 26607 net.cpp:122] Setting up fire5/expand1x1_1
I0518 22:23:36.856760 26607 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0518 22:23:36.856766 26607 net.cpp:137] Memory required for data: 22258944
I0518 22:23:36.856773 26607 layer_factory.hpp:77] Creating layer fire5/expand1x1_2
I0518 22:23:36.856796 26607 net.cpp:84] Creating Layer fire5/expand1x1_2
I0518 22:23:36.856813 26607 net.cpp:406] fire5/expand1x1_2 <- fire5/expand1x1_1
I0518 22:23:36.856833 26607 net.cpp:380] fire5/expand1x1_2 -> fire5/expand1x1_2
I0518 22:23:36.856853 26607 net.cpp:122] Setting up fire5/expand1x1_2
I0518 22:23:36.856870 26607 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0518 22:23:36.856889 26607 net.cpp:137] Memory required for data: 22632192
I0518 22:23:36.856905 26607 layer_factory.hpp:77] Creating layer fire5/expand3x3_1
I0518 22:23:36.856930 26607 net.cpp:84] Creating Layer fire5/expand3x3_1
I0518 22:23:36.856947 26607 net.cpp:406] fire5/expand3x3_1 <- fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split_1
I0518 22:23:36.856971 26607 net.cpp:380] fire5/expand3x3_1 -> fire5/expand3x3_1
I0518 22:23:36.857086 26607 net.cpp:122] Setting up fire5/expand3x3_1
I0518 22:23:36.857096 26607 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0518 22:23:36.857103 26607 net.cpp:137] Memory required for data: 23005440
I0518 22:23:36.857110 26607 layer_factory.hpp:77] Creating layer fire5/expand3x3_2
I0518 22:23:36.857118 26607 net.cpp:84] Creating Layer fire5/expand3x3_2
I0518 22:23:36.857126 26607 net.cpp:406] fire5/expand3x3_2 <- fire5/expand3x3_1
I0518 22:23:36.857134 26607 net.cpp:380] fire5/expand3x3_2 -> fire5/expand3x3_2
I0518 22:23:36.857144 26607 net.cpp:122] Setting up fire5/expand3x3_2
I0518 22:23:36.857151 26607 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0518 22:23:36.857158 26607 net.cpp:137] Memory required for data: 23378688
I0518 22:23:36.857163 26607 layer_factory.hpp:77] Creating layer fire5/concat
I0518 22:23:36.857169 26607 net.cpp:84] Creating Layer fire5/concat
I0518 22:23:36.857177 26607 net.cpp:406] fire5/concat <- fire5/expand1x1_2
I0518 22:23:36.857182 26607 net.cpp:406] fire5/concat <- fire5/expand3x3_2
I0518 22:23:36.857192 26607 net.cpp:380] fire5/concat -> fire5/concat
I0518 22:23:36.857216 26607 net.cpp:122] Setting up fire5/concat
I0518 22:23:36.857224 26607 net.cpp:129] Top shape: 1 256 27 27 (186624)
I0518 22:23:36.857231 26607 net.cpp:137] Memory required for data: 24125184
I0518 22:23:36.857235 26607 layer_factory.hpp:77] Creating layer pool5
I0518 22:23:36.857244 26607 net.cpp:84] Creating Layer pool5
I0518 22:23:36.857249 26607 net.cpp:406] pool5 <- fire5/concat
I0518 22:23:36.857254 26607 net.cpp:380] pool5 -> pool5
I0518 22:23:36.857264 26607 net.cpp:122] Setting up pool5
I0518 22:23:36.857271 26607 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0518 22:23:36.857277 26607 net.cpp:137] Memory required for data: 24298240
I0518 22:23:36.857282 26607 layer_factory.hpp:77] Creating layer fire6/squeeze1x1_1
I0518 22:23:36.857292 26607 net.cpp:84] Creating Layer fire6/squeeze1x1_1
I0518 22:23:36.857313 26607 net.cpp:406] fire6/squeeze1x1_1 <- pool5
I0518 22:23:36.857333 26607 net.cpp:380] fire6/squeeze1x1_1 -> fire6/squeeze1x1_1
I0518 22:23:36.857393 26607 net.cpp:122] Setting up fire6/squeeze1x1_1
I0518 22:23:36.857401 26607 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0518 22:23:36.857409 26607 net.cpp:137] Memory required for data: 24330688
I0518 22:23:36.857416 26607 layer_factory.hpp:77] Creating layer fire6/squeeze1x1_2
I0518 22:23:36.857422 26607 net.cpp:84] Creating Layer fire6/squeeze1x1_2
I0518 22:23:36.857430 26607 net.cpp:406] fire6/squeeze1x1_2 <- fire6/squeeze1x1_1
I0518 22:23:36.857437 26607 net.cpp:380] fire6/squeeze1x1_2 -> fire6/squeeze1x1_2
I0518 22:23:36.857447 26607 net.cpp:122] Setting up fire6/squeeze1x1_2
I0518 22:23:36.857468 26607 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0518 22:23:36.857488 26607 net.cpp:137] Memory required for data: 24363136
I0518 22:23:36.857496 26607 layer_factory.hpp:77] Creating layer fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split
I0518 22:23:36.857502 26607 net.cpp:84] Creating Layer fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split
I0518 22:23:36.857508 26607 net.cpp:406] fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split <- fire6/squeeze1x1_2
I0518 22:23:36.857515 26607 net.cpp:380] fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split -> fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split_0
I0518 22:23:36.857529 26607 net.cpp:380] fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split -> fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split_1
I0518 22:23:36.857538 26607 net.cpp:122] Setting up fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split
I0518 22:23:36.857556 26607 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0518 22:23:36.857576 26607 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0518 22:23:36.857594 26607 net.cpp:137] Memory required for data: 24428032
I0518 22:23:36.857611 26607 layer_factory.hpp:77] Creating layer fire6/expand1x1_1
I0518 22:23:36.857638 26607 net.cpp:84] Creating Layer fire6/expand1x1_1
I0518 22:23:36.857647 26607 net.cpp:406] fire6/expand1x1_1 <- fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split_0
I0518 22:23:36.857656 26607 net.cpp:380] fire6/expand1x1_1 -> fire6/expand1x1_1
I0518 22:23:36.857698 26607 net.cpp:122] Setting up fire6/expand1x1_1
I0518 22:23:36.857714 26607 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0518 22:23:36.857719 26607 net.cpp:137] Memory required for data: 24557824
I0518 22:23:36.857725 26607 layer_factory.hpp:77] Creating layer fire6/expand1x1_2
I0518 22:23:36.857731 26607 net.cpp:84] Creating Layer fire6/expand1x1_2
I0518 22:23:36.857738 26607 net.cpp:406] fire6/expand1x1_2 <- fire6/expand1x1_1
I0518 22:23:36.857743 26607 net.cpp:380] fire6/expand1x1_2 -> fire6/expand1x1_2
I0518 22:23:36.857749 26607 net.cpp:122] Setting up fire6/expand1x1_2
I0518 22:23:36.857767 26607 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0518 22:23:36.857784 26607 net.cpp:137] Memory required for data: 24687616
I0518 22:23:36.857796 26607 layer_factory.hpp:77] Creating layer fire6/expand3x3_1
I0518 22:23:36.857816 26607 net.cpp:84] Creating Layer fire6/expand3x3_1
I0518 22:23:36.857822 26607 net.cpp:406] fire6/expand3x3_1 <- fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split_1
I0518 22:23:36.857829 26607 net.cpp:380] fire6/expand3x3_1 -> fire6/expand3x3_1
I0518 22:23:36.857990 26607 net.cpp:122] Setting up fire6/expand3x3_1
I0518 22:23:36.857997 26607 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0518 22:23:36.858003 26607 net.cpp:137] Memory required for data: 24817408
I0518 22:23:36.858009 26607 layer_factory.hpp:77] Creating layer fire6/expand3x3_2
I0518 22:23:36.858017 26607 net.cpp:84] Creating Layer fire6/expand3x3_2
I0518 22:23:36.858023 26607 net.cpp:406] fire6/expand3x3_2 <- fire6/expand3x3_1
I0518 22:23:36.858029 26607 net.cpp:380] fire6/expand3x3_2 -> fire6/expand3x3_2
I0518 22:23:36.858038 26607 net.cpp:122] Setting up fire6/expand3x3_2
I0518 22:23:36.858042 26607 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0518 22:23:36.858049 26607 net.cpp:137] Memory required for data: 24947200
I0518 22:23:36.858053 26607 layer_factory.hpp:77] Creating layer fire6/concat
I0518 22:23:36.858062 26607 net.cpp:84] Creating Layer fire6/concat
I0518 22:23:36.858067 26607 net.cpp:406] fire6/concat <- fire6/expand1x1_2
I0518 22:23:36.858072 26607 net.cpp:406] fire6/concat <- fire6/expand3x3_2
I0518 22:23:36.858078 26607 net.cpp:380] fire6/concat -> fire6/concat
I0518 22:23:36.858085 26607 net.cpp:122] Setting up fire6/concat
I0518 22:23:36.858091 26607 net.cpp:129] Top shape: 1 384 13 13 (64896)
I0518 22:23:36.858096 26607 net.cpp:137] Memory required for data: 25206784
I0518 22:23:36.858099 26607 layer_factory.hpp:77] Creating layer fire7/squeeze1x1_1
I0518 22:23:36.858106 26607 net.cpp:84] Creating Layer fire7/squeeze1x1_1
I0518 22:23:36.858124 26607 net.cpp:406] fire7/squeeze1x1_1 <- fire6/concat
I0518 22:23:36.858132 26607 net.cpp:380] fire7/squeeze1x1_1 -> fire7/squeeze1x1_1
I0518 22:23:36.858183 26607 net.cpp:122] Setting up fire7/squeeze1x1_1
I0518 22:23:36.858189 26607 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0518 22:23:36.858196 26607 net.cpp:137] Memory required for data: 25239232
I0518 22:23:36.858203 26607 layer_factory.hpp:77] Creating layer fire7/squeeze1x1_2
I0518 22:23:36.858222 26607 net.cpp:84] Creating Layer fire7/squeeze1x1_2
I0518 22:23:36.858237 26607 net.cpp:406] fire7/squeeze1x1_2 <- fire7/squeeze1x1_1
I0518 22:23:36.858254 26607 net.cpp:380] fire7/squeeze1x1_2 -> fire7/squeeze1x1_2
I0518 22:23:36.858273 26607 net.cpp:122] Setting up fire7/squeeze1x1_2
I0518 22:23:36.858289 26607 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0518 22:23:36.858304 26607 net.cpp:137] Memory required for data: 25271680
I0518 22:23:36.858319 26607 layer_factory.hpp:77] Creating layer fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split
I0518 22:23:36.858333 26607 net.cpp:84] Creating Layer fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split
I0518 22:23:36.858348 26607 net.cpp:406] fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split <- fire7/squeeze1x1_2
I0518 22:23:36.858364 26607 net.cpp:380] fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split -> fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split_0
I0518 22:23:36.858384 26607 net.cpp:380] fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split -> fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split_1
I0518 22:23:36.858403 26607 net.cpp:122] Setting up fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split
I0518 22:23:36.858418 26607 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0518 22:23:36.858434 26607 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0518 22:23:36.858450 26607 net.cpp:137] Memory required for data: 25336576
I0518 22:23:36.858464 26607 layer_factory.hpp:77] Creating layer fire7/expand1x1_1
I0518 22:23:36.858486 26607 net.cpp:84] Creating Layer fire7/expand1x1_1
I0518 22:23:36.858494 26607 net.cpp:406] fire7/expand1x1_1 <- fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split_0
I0518 22:23:36.858500 26607 net.cpp:380] fire7/expand1x1_1 -> fire7/expand1x1_1
I0518 22:23:36.858533 26607 net.cpp:122] Setting up fire7/expand1x1_1
I0518 22:23:36.858541 26607 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0518 22:23:36.858546 26607 net.cpp:137] Memory required for data: 25466368
I0518 22:23:36.858551 26607 layer_factory.hpp:77] Creating layer fire7/expand1x1_2
I0518 22:23:36.858557 26607 net.cpp:84] Creating Layer fire7/expand1x1_2
I0518 22:23:36.858575 26607 net.cpp:406] fire7/expand1x1_2 <- fire7/expand1x1_1
I0518 22:23:36.858594 26607 net.cpp:380] fire7/expand1x1_2 -> fire7/expand1x1_2
I0518 22:23:36.858618 26607 net.cpp:122] Setting up fire7/expand1x1_2
I0518 22:23:36.858629 26607 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0518 22:23:36.858642 26607 net.cpp:137] Memory required for data: 25596160
I0518 22:23:36.858654 26607 layer_factory.hpp:77] Creating layer fire7/expand3x3_1
I0518 22:23:36.858667 26607 net.cpp:84] Creating Layer fire7/expand3x3_1
I0518 22:23:36.858680 26607 net.cpp:406] fire7/expand3x3_1 <- fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split_1
I0518 22:23:36.858693 26607 net.cpp:380] fire7/expand3x3_1 -> fire7/expand3x3_1
I0518 22:23:36.858841 26607 net.cpp:122] Setting up fire7/expand3x3_1
I0518 22:23:36.858848 26607 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0518 22:23:36.858853 26607 net.cpp:137] Memory required for data: 25725952
I0518 22:23:36.858858 26607 layer_factory.hpp:77] Creating layer fire7/expand3x3_2
I0518 22:23:36.858863 26607 net.cpp:84] Creating Layer fire7/expand3x3_2
I0518 22:23:36.858877 26607 net.cpp:406] fire7/expand3x3_2 <- fire7/expand3x3_1
I0518 22:23:36.858891 26607 net.cpp:380] fire7/expand3x3_2 -> fire7/expand3x3_2
I0518 22:23:36.858904 26607 net.cpp:122] Setting up fire7/expand3x3_2
I0518 22:23:36.858916 26607 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0518 22:23:36.858929 26607 net.cpp:137] Memory required for data: 25855744
I0518 22:23:36.858940 26607 layer_factory.hpp:77] Creating layer fire7/concat
I0518 22:23:36.858953 26607 net.cpp:84] Creating Layer fire7/concat
I0518 22:23:36.858966 26607 net.cpp:406] fire7/concat <- fire7/expand1x1_2
I0518 22:23:36.858980 26607 net.cpp:406] fire7/concat <- fire7/expand3x3_2
I0518 22:23:36.859000 26607 net.cpp:380] fire7/concat -> fire7/concat
I0518 22:23:36.859009 26607 net.cpp:122] Setting up fire7/concat
I0518 22:23:36.859011 26607 net.cpp:129] Top shape: 1 384 13 13 (64896)
I0518 22:23:36.859025 26607 net.cpp:137] Memory required for data: 26115328
I0518 22:23:36.859037 26607 layer_factory.hpp:77] Creating layer fire8/squeeze1x1_1
I0518 22:23:36.859053 26607 net.cpp:84] Creating Layer fire8/squeeze1x1_1
I0518 22:23:36.859067 26607 net.cpp:406] fire8/squeeze1x1_1 <- fire7/concat
I0518 22:23:36.859081 26607 net.cpp:380] fire8/squeeze1x1_1 -> fire8/squeeze1x1_1
I0518 22:23:36.859141 26607 net.cpp:122] Setting up fire8/squeeze1x1_1
I0518 22:23:36.859148 26607 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0518 22:23:36.859153 26607 net.cpp:137] Memory required for data: 26158592
I0518 22:23:36.859158 26607 layer_factory.hpp:77] Creating layer fire8/squeeze1x1_2
I0518 22:23:36.859163 26607 net.cpp:84] Creating Layer fire8/squeeze1x1_2
I0518 22:23:36.859177 26607 net.cpp:406] fire8/squeeze1x1_2 <- fire8/squeeze1x1_1
I0518 22:23:36.859191 26607 net.cpp:380] fire8/squeeze1x1_2 -> fire8/squeeze1x1_2
I0518 22:23:36.859205 26607 net.cpp:122] Setting up fire8/squeeze1x1_2
I0518 22:23:36.859216 26607 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0518 22:23:36.859230 26607 net.cpp:137] Memory required for data: 26201856
I0518 22:23:36.859242 26607 layer_factory.hpp:77] Creating layer fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split
I0518 22:23:36.859256 26607 net.cpp:84] Creating Layer fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split
I0518 22:23:36.859267 26607 net.cpp:406] fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split <- fire8/squeeze1x1_2
I0518 22:23:36.859284 26607 net.cpp:380] fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split -> fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split_0
I0518 22:23:36.859292 26607 net.cpp:380] fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split -> fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split_1
I0518 22:23:36.859298 26607 net.cpp:122] Setting up fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split
I0518 22:23:36.859310 26607 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0518 22:23:36.859323 26607 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0518 22:23:36.859336 26607 net.cpp:137] Memory required for data: 26288384
I0518 22:23:36.859347 26607 layer_factory.hpp:77] Creating layer fire8/expand1x1_1
I0518 22:23:36.859365 26607 net.cpp:84] Creating Layer fire8/expand1x1_1
I0518 22:23:36.859371 26607 net.cpp:406] fire8/expand1x1_1 <- fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split_0
I0518 22:23:36.859377 26607 net.cpp:380] fire8/expand1x1_1 -> fire8/expand1x1_1
I0518 22:23:36.859421 26607 net.cpp:122] Setting up fire8/expand1x1_1
I0518 22:23:36.859426 26607 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0518 22:23:36.859431 26607 net.cpp:137] Memory required for data: 26461440
I0518 22:23:36.859436 26607 layer_factory.hpp:77] Creating layer fire8/expand1x1_2
I0518 22:23:36.859454 26607 net.cpp:84] Creating Layer fire8/expand1x1_2
I0518 22:23:36.859467 26607 net.cpp:406] fire8/expand1x1_2 <- fire8/expand1x1_1
I0518 22:23:36.859479 26607 net.cpp:380] fire8/expand1x1_2 -> fire8/expand1x1_2
I0518 22:23:36.859493 26607 net.cpp:122] Setting up fire8/expand1x1_2
I0518 22:23:36.859506 26607 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0518 22:23:36.859519 26607 net.cpp:137] Memory required for data: 26634496
I0518 22:23:36.859531 26607 layer_factory.hpp:77] Creating layer fire8/expand3x3_1
I0518 22:23:36.859545 26607 net.cpp:84] Creating Layer fire8/expand3x3_1
I0518 22:23:36.859558 26607 net.cpp:406] fire8/expand3x3_1 <- fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split_1
I0518 22:23:36.859573 26607 net.cpp:380] fire8/expand3x3_1 -> fire8/expand3x3_1
I0518 22:23:36.859817 26607 net.cpp:122] Setting up fire8/expand3x3_1
I0518 22:23:36.859824 26607 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0518 22:23:36.859829 26607 net.cpp:137] Memory required for data: 26807552
I0518 22:23:36.859834 26607 layer_factory.hpp:77] Creating layer fire8/expand3x3_2
I0518 22:23:36.859840 26607 net.cpp:84] Creating Layer fire8/expand3x3_2
I0518 22:23:36.859861 26607 net.cpp:406] fire8/expand3x3_2 <- fire8/expand3x3_1
I0518 22:23:36.859875 26607 net.cpp:380] fire8/expand3x3_2 -> fire8/expand3x3_2
I0518 22:23:36.859889 26607 net.cpp:122] Setting up fire8/expand3x3_2
I0518 22:23:36.859901 26607 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0518 22:23:36.859915 26607 net.cpp:137] Memory required for data: 26980608
I0518 22:23:36.859926 26607 layer_factory.hpp:77] Creating layer fire8/concat
I0518 22:23:36.859941 26607 net.cpp:84] Creating Layer fire8/concat
I0518 22:23:36.859952 26607 net.cpp:406] fire8/concat <- fire8/expand1x1_2
I0518 22:23:36.859966 26607 net.cpp:406] fire8/concat <- fire8/expand3x3_2
I0518 22:23:36.859980 26607 net.cpp:380] fire8/concat -> fire8/concat
I0518 22:23:36.859995 26607 net.cpp:122] Setting up fire8/concat
I0518 22:23:36.860008 26607 net.cpp:129] Top shape: 1 512 13 13 (86528)
I0518 22:23:36.860020 26607 net.cpp:137] Memory required for data: 27326720
I0518 22:23:36.860033 26607 layer_factory.hpp:77] Creating layer fire9/squeeze1x1_1
I0518 22:23:36.860046 26607 net.cpp:84] Creating Layer fire9/squeeze1x1_1
I0518 22:23:36.860059 26607 net.cpp:406] fire9/squeeze1x1_1 <- fire8/concat
I0518 22:23:36.860074 26607 net.cpp:380] fire9/squeeze1x1_1 -> fire9/squeeze1x1_1
I0518 22:23:36.860152 26607 net.cpp:122] Setting up fire9/squeeze1x1_1
I0518 22:23:36.860169 26607 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0518 22:23:36.860185 26607 net.cpp:137] Memory required for data: 27369984
I0518 22:23:36.860198 26607 layer_factory.hpp:77] Creating layer fire9/squeeze1x1_2
I0518 22:23:36.860213 26607 net.cpp:84] Creating Layer fire9/squeeze1x1_2
I0518 22:23:36.860225 26607 net.cpp:406] fire9/squeeze1x1_2 <- fire9/squeeze1x1_1
I0518 22:23:36.860239 26607 net.cpp:380] fire9/squeeze1x1_2 -> fire9/squeeze1x1_2
I0518 22:23:36.860252 26607 net.cpp:122] Setting up fire9/squeeze1x1_2
I0518 22:23:36.860263 26607 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0518 22:23:36.860276 26607 net.cpp:137] Memory required for data: 27413248
I0518 22:23:36.860287 26607 layer_factory.hpp:77] Creating layer fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split
I0518 22:23:36.860306 26607 net.cpp:84] Creating Layer fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split
I0518 22:23:36.860311 26607 net.cpp:406] fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split <- fire9/squeeze1x1_2
I0518 22:23:36.860317 26607 net.cpp:380] fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split -> fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split_0
I0518 22:23:36.860322 26607 net.cpp:380] fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split -> fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split_1
I0518 22:23:36.860339 26607 net.cpp:122] Setting up fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split
I0518 22:23:36.860352 26607 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0518 22:23:36.860364 26607 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0518 22:23:36.860376 26607 net.cpp:137] Memory required for data: 27499776
I0518 22:23:36.860389 26607 layer_factory.hpp:77] Creating layer fire9/expand1x1_1
I0518 22:23:36.860402 26607 net.cpp:84] Creating Layer fire9/expand1x1_1
I0518 22:23:36.860414 26607 net.cpp:406] fire9/expand1x1_1 <- fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split_0
I0518 22:23:36.860431 26607 net.cpp:380] fire9/expand1x1_1 -> fire9/expand1x1_1
I0518 22:23:36.860471 26607 net.cpp:122] Setting up fire9/expand1x1_1
I0518 22:23:36.860476 26607 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0518 22:23:36.860481 26607 net.cpp:137] Memory required for data: 27672832
I0518 22:23:36.860486 26607 layer_factory.hpp:77] Creating layer fire9/expand1x1_2
I0518 22:23:36.860503 26607 net.cpp:84] Creating Layer fire9/expand1x1_2
I0518 22:23:36.860515 26607 net.cpp:406] fire9/expand1x1_2 <- fire9/expand1x1_1
I0518 22:23:36.860528 26607 net.cpp:380] fire9/expand1x1_2 -> fire9/expand1x1_2
I0518 22:23:36.860543 26607 net.cpp:122] Setting up fire9/expand1x1_2
I0518 22:23:36.860554 26607 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0518 22:23:36.860568 26607 net.cpp:137] Memory required for data: 27845888
I0518 22:23:36.860579 26607 layer_factory.hpp:77] Creating layer fire9/expand3x3_1
I0518 22:23:36.860599 26607 net.cpp:84] Creating Layer fire9/expand3x3_1
I0518 22:23:36.860605 26607 net.cpp:406] fire9/expand3x3_1 <- fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split_1
I0518 22:23:36.860613 26607 net.cpp:380] fire9/expand3x3_1 -> fire9/expand3x3_1
I0518 22:23:36.860847 26607 net.cpp:122] Setting up fire9/expand3x3_1
I0518 22:23:36.860854 26607 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0518 22:23:36.860859 26607 net.cpp:137] Memory required for data: 28018944
I0518 22:23:36.860864 26607 layer_factory.hpp:77] Creating layer fire9/expand3x3_2
I0518 22:23:36.860872 26607 net.cpp:84] Creating Layer fire9/expand3x3_2
I0518 22:23:36.860886 26607 net.cpp:406] fire9/expand3x3_2 <- fire9/expand3x3_1
I0518 22:23:36.860899 26607 net.cpp:380] fire9/expand3x3_2 -> fire9/expand3x3_2
I0518 22:23:36.860913 26607 net.cpp:122] Setting up fire9/expand3x3_2
I0518 22:23:36.860925 26607 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0518 22:23:36.860939 26607 net.cpp:137] Memory required for data: 28192000
I0518 22:23:36.860949 26607 layer_factory.hpp:77] Creating layer fire9/concat_1
I0518 22:23:36.860965 26607 net.cpp:84] Creating Layer fire9/concat_1
I0518 22:23:36.860980 26607 net.cpp:406] fire9/concat_1 <- fire9/expand1x1_2
I0518 22:23:36.860992 26607 net.cpp:406] fire9/concat_1 <- fire9/expand3x3_2
I0518 22:23:36.861006 26607 net.cpp:380] fire9/concat_1 -> fire9/concat_1
I0518 22:23:36.861021 26607 net.cpp:122] Setting up fire9/concat_1
I0518 22:23:36.861033 26607 net.cpp:129] Top shape: 1 512 13 13 (86528)
I0518 22:23:36.861047 26607 net.cpp:137] Memory required for data: 28538112
I0518 22:23:36.861058 26607 layer_factory.hpp:77] Creating layer fire9/concat_2__fire9/concat_mask
I0518 22:23:36.861078 26607 net.cpp:84] Creating Layer fire9/concat_2__fire9/concat_mask
I0518 22:23:36.861083 26607 net.cpp:406] fire9/concat_2__fire9/concat_mask <- fire9/concat_1
I0518 22:23:36.861088 26607 net.cpp:380] fire9/concat_2__fire9/concat_mask -> fire9/concat_2
I0518 22:23:36.861099 26607 net.cpp:122] Setting up fire9/concat_2__fire9/concat_mask
I0518 22:23:36.861111 26607 net.cpp:129] Top shape: 1 512 13 13 (86528)
I0518 22:23:36.861124 26607 net.cpp:137] Memory required for data: 28884224
I0518 22:23:36.861135 26607 layer_factory.hpp:77] Creating layer conv10_1
I0518 22:23:36.861150 26607 net.cpp:84] Creating Layer conv10_1
I0518 22:23:36.861162 26607 net.cpp:406] conv10_1 <- fire9/concat_2
I0518 22:23:36.861177 26607 net.cpp:380] conv10_1 -> conv10_1
I0518 22:23:36.861917 26607 net.cpp:122] Setting up conv10_1
I0518 22:23:36.861928 26607 net.cpp:129] Top shape: 1 1000 13 13 (169000)
I0518 22:23:36.861934 26607 net.cpp:137] Memory required for data: 29560224
I0518 22:23:36.861939 26607 layer_factory.hpp:77] Creating layer conv10_2
I0518 22:23:36.861958 26607 net.cpp:84] Creating Layer conv10_2
I0518 22:23:36.861970 26607 net.cpp:406] conv10_2 <- conv10_1
I0518 22:23:36.861985 26607 net.cpp:380] conv10_2 -> conv10_2
I0518 22:23:36.861999 26607 net.cpp:122] Setting up conv10_2
I0518 22:23:36.862011 26607 net.cpp:129] Top shape: 1 1000 13 13 (169000)
I0518 22:23:36.862025 26607 net.cpp:137] Memory required for data: 30236224
I0518 22:23:36.862035 26607 layer_factory.hpp:77] Creating layer pool10
I0518 22:23:36.862053 26607 net.cpp:84] Creating Layer pool10
I0518 22:23:36.862058 26607 net.cpp:406] pool10 <- conv10_2
I0518 22:23:36.862064 26607 net.cpp:380] pool10 -> pool10
I0518 22:23:36.862071 26607 net.cpp:122] Setting up pool10
I0518 22:23:36.862084 26607 net.cpp:129] Top shape: 1 1000 1 1 (1000)
I0518 22:23:36.862097 26607 net.cpp:137] Memory required for data: 30240224
I0518 22:23:36.862108 26607 layer_factory.hpp:77] Creating layer softmaxout
I0518 22:23:36.862123 26607 net.cpp:84] Creating Layer softmaxout
I0518 22:23:36.862134 26607 net.cpp:406] softmaxout <- pool10
I0518 22:23:36.862151 26607 net.cpp:380] softmaxout -> softmaxout
I0518 22:23:36.862166 26607 net.cpp:122] Setting up softmaxout
I0518 22:23:36.862180 26607 net.cpp:129] Top shape: 1 1000 1 1 (1000)
I0518 22:23:36.862195 26607 net.cpp:137] Memory required for data: 30244224
I0518 22:23:36.862210 26607 net.cpp:200] softmaxout does not need backward computation.
I0518 22:23:36.862216 26607 net.cpp:200] pool10 does not need backward computation.
I0518 22:23:36.862221 26607 net.cpp:200] conv10_2 does not need backward computation.
I0518 22:23:36.862224 26607 net.cpp:200] conv10_1 does not need backward computation.
I0518 22:23:36.862228 26607 net.cpp:200] fire9/concat_2__fire9/concat_mask does not need backward computation.
I0518 22:23:36.862232 26607 net.cpp:200] fire9/concat_1 does not need backward computation.
I0518 22:23:36.862236 26607 net.cpp:200] fire9/expand3x3_2 does not need backward computation.
I0518 22:23:36.862241 26607 net.cpp:200] fire9/expand3x3_1 does not need backward computation.
I0518 22:23:36.862253 26607 net.cpp:200] fire9/expand1x1_2 does not need backward computation.
I0518 22:23:36.862267 26607 net.cpp:200] fire9/expand1x1_1 does not need backward computation.
I0518 22:23:36.862278 26607 net.cpp:200] fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split does not need backward computation.
I0518 22:23:36.862290 26607 net.cpp:200] fire9/squeeze1x1_2 does not need backward computation.
I0518 22:23:36.862303 26607 net.cpp:200] fire9/squeeze1x1_1 does not need backward computation.
I0518 22:23:36.862313 26607 net.cpp:200] fire8/concat does not need backward computation.
I0518 22:23:36.862325 26607 net.cpp:200] fire8/expand3x3_2 does not need backward computation.
I0518 22:23:36.862337 26607 net.cpp:200] fire8/expand3x3_1 does not need backward computation.
I0518 22:23:36.862349 26607 net.cpp:200] fire8/expand1x1_2 does not need backward computation.
I0518 22:23:36.862362 26607 net.cpp:200] fire8/expand1x1_1 does not need backward computation.
I0518 22:23:36.862368 26607 net.cpp:200] fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split does not need backward computation.
I0518 22:23:36.862372 26607 net.cpp:200] fire8/squeeze1x1_2 does not need backward computation.
I0518 22:23:36.862377 26607 net.cpp:200] fire8/squeeze1x1_1 does not need backward computation.
I0518 22:23:36.862380 26607 net.cpp:200] fire7/concat does not need backward computation.
I0518 22:23:36.862386 26607 net.cpp:200] fire7/expand3x3_2 does not need backward computation.
I0518 22:23:36.862390 26607 net.cpp:200] fire7/expand3x3_1 does not need backward computation.
I0518 22:23:36.862396 26607 net.cpp:200] fire7/expand1x1_2 does not need backward computation.
I0518 22:23:36.862401 26607 net.cpp:200] fire7/expand1x1_1 does not need backward computation.
I0518 22:23:36.862406 26607 net.cpp:200] fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split does not need backward computation.
I0518 22:23:36.862411 26607 net.cpp:200] fire7/squeeze1x1_2 does not need backward computation.
I0518 22:23:36.862416 26607 net.cpp:200] fire7/squeeze1x1_1 does not need backward computation.
I0518 22:23:36.862421 26607 net.cpp:200] fire6/concat does not need backward computation.
I0518 22:23:36.862426 26607 net.cpp:200] fire6/expand3x3_2 does not need backward computation.
I0518 22:23:36.862432 26607 net.cpp:200] fire6/expand3x3_1 does not need backward computation.
I0518 22:23:36.862435 26607 net.cpp:200] fire6/expand1x1_2 does not need backward computation.
I0518 22:23:36.862439 26607 net.cpp:200] fire6/expand1x1_1 does not need backward computation.
I0518 22:23:36.862444 26607 net.cpp:200] fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split does not need backward computation.
I0518 22:23:36.862449 26607 net.cpp:200] fire6/squeeze1x1_2 does not need backward computation.
I0518 22:23:36.862453 26607 net.cpp:200] fire6/squeeze1x1_1 does not need backward computation.
I0518 22:23:36.862458 26607 net.cpp:200] pool5 does not need backward computation.
I0518 22:23:36.862462 26607 net.cpp:200] fire5/concat does not need backward computation.
I0518 22:23:36.862466 26607 net.cpp:200] fire5/expand3x3_2 does not need backward computation.
I0518 22:23:36.862471 26607 net.cpp:200] fire5/expand3x3_1 does not need backward computation.
I0518 22:23:36.862475 26607 net.cpp:200] fire5/expand1x1_2 does not need backward computation.
I0518 22:23:36.862479 26607 net.cpp:200] fire5/expand1x1_1 does not need backward computation.
I0518 22:23:36.862483 26607 net.cpp:200] fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split does not need backward computation.
I0518 22:23:36.862488 26607 net.cpp:200] fire5/squeeze1x1_2 does not need backward computation.
I0518 22:23:36.862491 26607 net.cpp:200] fire5/squeeze1x1_1 does not need backward computation.
I0518 22:23:36.862495 26607 net.cpp:200] fire4/concat does not need backward computation.
I0518 22:23:36.862499 26607 net.cpp:200] fire4/expand3x3_2 does not need backward computation.
I0518 22:23:36.862504 26607 net.cpp:200] fire4/expand3x3_1 does not need backward computation.
I0518 22:23:36.862509 26607 net.cpp:200] fire4/expand1x1_2 does not need backward computation.
I0518 22:23:36.862512 26607 net.cpp:200] fire4/expand1x1_1 does not need backward computation.
I0518 22:23:36.862519 26607 net.cpp:200] fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split does not need backward computation.
I0518 22:23:36.862522 26607 net.cpp:200] fire4/squeeze1x1_2 does not need backward computation.
I0518 22:23:36.862526 26607 net.cpp:200] fire4/squeeze1x1_1 does not need backward computation.
I0518 22:23:36.862531 26607 net.cpp:200] pool3 does not need backward computation.
I0518 22:23:36.862535 26607 net.cpp:200] fire3/concat does not need backward computation.
I0518 22:23:36.862540 26607 net.cpp:200] fire3/expand3x3_2 does not need backward computation.
I0518 22:23:36.862545 26607 net.cpp:200] fire3/expand3x3_1 does not need backward computation.
I0518 22:23:36.862548 26607 net.cpp:200] fire3/expand1x1_2 does not need backward computation.
I0518 22:23:36.862552 26607 net.cpp:200] fire3/expand1x1_1 does not need backward computation.
I0518 22:23:36.862560 26607 net.cpp:200] fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split does not need backward computation.
I0518 22:23:36.862566 26607 net.cpp:200] fire3/squeeze1x1_2 does not need backward computation.
I0518 22:23:36.862571 26607 net.cpp:200] fire3/squeeze1x1_1 does not need backward computation.
I0518 22:23:36.862576 26607 net.cpp:200] fire2/concat does not need backward computation.
I0518 22:23:36.862581 26607 net.cpp:200] fire2/expand3x3_2 does not need backward computation.
I0518 22:23:36.862586 26607 net.cpp:200] fire2/expand3x3_1 does not need backward computation.
I0518 22:23:36.862591 26607 net.cpp:200] fire2/expand1x1_2 does not need backward computation.
I0518 22:23:36.862596 26607 net.cpp:200] fire2/expand1x1_1 does not need backward computation.
I0518 22:23:36.862601 26607 net.cpp:200] fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split does not need backward computation.
I0518 22:23:36.862605 26607 net.cpp:200] fire2/squeeze1x1_2 does not need backward computation.
I0518 22:23:36.862610 26607 net.cpp:200] fire2/squeeze1x1_1 does not need backward computation.
I0518 22:23:36.862615 26607 net.cpp:200] pool1 does not need backward computation.
I0518 22:23:36.862619 26607 net.cpp:200] conv1_2 does not need backward computation.
I0518 22:23:36.862623 26607 net.cpp:200] conv1_1 does not need backward computation.
I0518 22:23:36.862628 26607 net.cpp:200] data does not need backward computation.
I0518 22:23:36.862632 26607 net.cpp:242] This network produces output softmaxout
I0518 22:23:36.862679 26607 net.cpp:255] Network initialization done.
/home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
WARNING:tensorflow:From /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/detection_softmax/inference/fp32/infer_detections.py:219: string_input_producer (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensor_slices(string_tensor).shuffle(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)`. If `shuffle=False`, omit the `.shuffle(...)`.
WARNING:tensorflow:From /home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/training/input.py:278: input_producer (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensor_slices(input_tensor).shuffle(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)`. If `shuffle=False`, omit the `.shuffle(...)`.
WARNING:tensorflow:From /home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/training/input.py:190: limit_epochs (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensors(tensor).repeat(num_epochs)`.
WARNING:tensorflow:From /home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/training/input.py:199: QueueRunner.__init__ (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
WARNING:tensorflow:From /home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/training/input.py:199: add_queue_runner (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
WARNING:tensorflow:From /home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/training/input.py:202: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
WARNING:tensorflow:From /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/detection_softmax/inference/fp32/infer_detections.py:224: TFRecordReader.__init__ (from tensorflow.python.ops.io_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.TFRecordDataset`.
WARNING:tensorflow:From /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/detection_softmax/inference/fp32/infer_detections.py:228: start_queue_runners (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
graph has been loaded using caffe..
Inference with real data.
/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.prototxt
/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.caffemodel
total iteration is 1000
warm up iteration is 0
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   156.3485 ms
Total samples/sec:     6.3960 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   148.2803 ms
Total samples/sec:     6.7440 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   146.0097 ms
Total samples/sec:     6.8489 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   144.8637 ms
Total samples/sec:     6.9030 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   144.9373 ms
Total samples/sec:     6.8995 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   144.0155 ms
Total samples/sec:     6.9437 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   143.6814 ms
Total samples/sec:     6.9598 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   143.4264 ms
Total samples/sec:     6.9722 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   143.0234 ms
Total samples/sec:     6.9919 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 9, 0.1394791603088379 sec
Batchsize: 1
Time spent per BATCH:   142.6690 ms
Total samples/sec:     7.0092 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.7386 ms
Total samples/sec:     7.0058 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.3977 ms
Total samples/sec:     7.0226 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.1468 ms
Total samples/sec:     7.0350 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.9460 ms
Total samples/sec:     7.0449 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.8630 ms
Total samples/sec:     7.0491 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.8914 ms
Total samples/sec:     7.0476 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.7058 ms
Total samples/sec:     7.0569 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5887 ms
Total samples/sec:     7.0627 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.4543 ms
Total samples/sec:     7.0694 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 19, 0.13912367820739746 sec
Batchsize: 1
Time spent per BATCH:   141.3377 ms
Total samples/sec:     7.0753 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2184 ms
Total samples/sec:     7.0812 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.4268 ms
Total samples/sec:     7.0708 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6877 ms
Total samples/sec:     7.0578 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.1357 ms
Total samples/sec:     7.0355 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.4765 ms
Total samples/sec:     7.0187 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.6568 ms
Total samples/sec:     7.0098 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.5372 ms
Total samples/sec:     7.0157 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.4234 ms
Total samples/sec:     7.0213 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.3287 ms
Total samples/sec:     7.0260 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 29, 0.1421511173248291 sec
Batchsize: 1
Time spent per BATCH:   142.3228 ms
Total samples/sec:     7.0263 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.3680 ms
Total samples/sec:     7.0240 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.6680 ms
Total samples/sec:     7.0093 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.5573 ms
Total samples/sec:     7.0147 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.4546 ms
Total samples/sec:     7.0198 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.3432 ms
Total samples/sec:     7.0253 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.2443 ms
Total samples/sec:     7.0302 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.1579 ms
Total samples/sec:     7.0344 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.0645 ms
Total samples/sec:     7.0391 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.0206 ms
Total samples/sec:     7.0412 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 39, 0.14304375648498535 sec
Batchsize: 1
Time spent per BATCH:   142.0462 ms
Total samples/sec:     7.0400 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.0334 ms
Total samples/sec:     7.0406 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.0226 ms
Total samples/sec:     7.0411 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.2146 ms
Total samples/sec:     7.0316 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.3231 ms
Total samples/sec:     7.0263 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.4315 ms
Total samples/sec:     7.0209 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.5387 ms
Total samples/sec:     7.0156 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.4561 ms
Total samples/sec:     7.0197 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.3791 ms
Total samples/sec:     7.0235 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.3017 ms
Total samples/sec:     7.0273 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 49, 0.13927745819091797 sec
Batchsize: 1
Time spent per BATCH:   142.2412 ms
Total samples/sec:     7.0303 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.1722 ms
Total samples/sec:     7.0337 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.1177 ms
Total samples/sec:     7.0364 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.0904 ms
Total samples/sec:     7.0378 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   142.0390 ms
Total samples/sec:     7.0403 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.9779 ms
Total samples/sec:     7.0433 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.9351 ms
Total samples/sec:     7.0455 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.8764 ms
Total samples/sec:     7.0484 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.8229 ms
Total samples/sec:     7.0510 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.7763 ms
Total samples/sec:     7.0534 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 59, 0.13861393928527832 sec
Batchsize: 1
Time spent per BATCH:   141.7236 ms
Total samples/sec:     7.0560 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.9377 ms
Total samples/sec:     7.0453 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.8872 ms
Total samples/sec:     7.0479 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.8365 ms
Total samples/sec:     7.0504 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.7890 ms
Total samples/sec:     7.0527 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.7886 ms
Total samples/sec:     7.0528 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.7564 ms
Total samples/sec:     7.0544 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.7263 ms
Total samples/sec:     7.0559 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6991 ms
Total samples/sec:     7.0572 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6503 ms
Total samples/sec:     7.0596 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 69, 0.13899660110473633 sec
Batchsize: 1
Time spent per BATCH:   141.6124 ms
Total samples/sec:     7.0615 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6742 ms
Total samples/sec:     7.0584 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6776 ms
Total samples/sec:     7.0583 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6418 ms
Total samples/sec:     7.0601 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6098 ms
Total samples/sec:     7.0617 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5766 ms
Total samples/sec:     7.0633 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5382 ms
Total samples/sec:     7.0652 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5042 ms
Total samples/sec:     7.0669 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.4801 ms
Total samples/sec:     7.0681 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.4493 ms
Total samples/sec:     7.0697 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 79, 0.1387331485748291 sec
Batchsize: 1
Time spent per BATCH:   141.4154 ms
Total samples/sec:     7.0714 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.3839 ms
Total samples/sec:     7.0729 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.3509 ms
Total samples/sec:     7.0746 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.3265 ms
Total samples/sec:     7.0758 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.3722 ms
Total samples/sec:     7.0735 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.3410 ms
Total samples/sec:     7.0751 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.3445 ms
Total samples/sec:     7.0749 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.3797 ms
Total samples/sec:     7.0732 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.4092 ms
Total samples/sec:     7.0717 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.4291 ms
Total samples/sec:     7.0707 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 89, 0.14601850509643555 sec
Batchsize: 1
Time spent per BATCH:   141.4801 ms
Total samples/sec:     7.0681 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5181 ms
Total samples/sec:     7.0662 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5477 ms
Total samples/sec:     7.0648 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5721 ms
Total samples/sec:     7.0635 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5918 ms
Total samples/sec:     7.0626 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6050 ms
Total samples/sec:     7.0619 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6270 ms
Total samples/sec:     7.0608 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6555 ms
Total samples/sec:     7.0594 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.7506 ms
Total samples/sec:     7.0546 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.8600 ms
Total samples/sec:     7.0492 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 99, 0.14235830307006836 sec
Batchsize: 1
Time spent per BATCH:   141.8650 ms
Total samples/sec:     7.0490 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.8691 ms
Total samples/sec:     7.0488 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.8807 ms
Total samples/sec:     7.0482 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.8507 ms
Total samples/sec:     7.0497 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.8187 ms
Total samples/sec:     7.0513 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.8099 ms
Total samples/sec:     7.0517 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.7794 ms
Total samples/sec:     7.0532 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.7498 ms
Total samples/sec:     7.0547 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.7313 ms
Total samples/sec:     7.0556 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.7013 ms
Total samples/sec:     7.0571 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 109, 0.13868951797485352 sec
Batchsize: 1
Time spent per BATCH:   141.6739 ms
Total samples/sec:     7.0585 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6515 ms
Total samples/sec:     7.0596 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6417 ms
Total samples/sec:     7.0601 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6147 ms
Total samples/sec:     7.0614 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5987 ms
Total samples/sec:     7.0622 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5752 ms
Total samples/sec:     7.0634 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5543 ms
Total samples/sec:     7.0644 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5460 ms
Total samples/sec:     7.0648 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5314 ms
Total samples/sec:     7.0656 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5465 ms
Total samples/sec:     7.0648 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 119, 0.14151620864868164 sec
Batchsize: 1
Time spent per BATCH:   141.5463 ms
Total samples/sec:     7.0648 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5306 ms
Total samples/sec:     7.0656 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5077 ms
Total samples/sec:     7.0668 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.4879 ms
Total samples/sec:     7.0677 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.4670 ms
Total samples/sec:     7.0688 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6225 ms
Total samples/sec:     7.0610 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.6026 ms
Total samples/sec:     7.0620 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5786 ms
Total samples/sec:     7.0632 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5607 ms
Total samples/sec:     7.0641 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5399 ms
Total samples/sec:     7.0651 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 129, 0.14672613143920898 sec
Batchsize: 1
Time spent per BATCH:   141.5798 ms
Total samples/sec:     7.0632 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5610 ms
Total samples/sec:     7.0641 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5394 ms
Total samples/sec:     7.0652 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.5170 ms
Total samples/sec:     7.0663 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.4991 ms
Total samples/sec:     7.0672 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.4783 ms
Total samples/sec:     7.0682 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.4632 ms
Total samples/sec:     7.0690 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.4450 ms
Total samples/sec:     7.0699 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.4255 ms
Total samples/sec:     7.0709 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.4134 ms
Total samples/sec:     7.0715 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 139, 0.1389462947845459 sec
Batchsize: 1
Time spent per BATCH:   141.3958 ms
Total samples/sec:     7.0723 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.3824 ms
Total samples/sec:     7.0730 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.3664 ms
Total samples/sec:     7.0738 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.3470 ms
Total samples/sec:     7.0748 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.3278 ms
Total samples/sec:     7.0757 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.3096 ms
Total samples/sec:     7.0767 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2908 ms
Total samples/sec:     7.0776 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2737 ms
Total samples/sec:     7.0785 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2564 ms
Total samples/sec:     7.0793 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2386 ms
Total samples/sec:     7.0802 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 149, 0.13898611068725586 sec
Batchsize: 1
Time spent per BATCH:   141.2235 ms
Total samples/sec:     7.0810 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2047 ms
Total samples/sec:     7.0819 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1879 ms
Total samples/sec:     7.0828 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1712 ms
Total samples/sec:     7.0836 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1691 ms
Total samples/sec:     7.0837 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1519 ms
Total samples/sec:     7.0846 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1513 ms
Total samples/sec:     7.0846 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1354 ms
Total samples/sec:     7.0854 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1503 ms
Total samples/sec:     7.0846 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1332 ms
Total samples/sec:     7.0855 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 159, 0.13846135139465332 sec
Batchsize: 1
Time spent per BATCH:   141.1165 ms
Total samples/sec:     7.0863 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1045 ms
Total samples/sec:     7.0869 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0898 ms
Total samples/sec:     7.0877 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0872 ms
Total samples/sec:     7.0878 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0736 ms
Total samples/sec:     7.0885 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0612 ms
Total samples/sec:     7.0891 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0464 ms
Total samples/sec:     7.0899 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0328 ms
Total samples/sec:     7.0905 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0443 ms
Total samples/sec:     7.0900 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0293 ms
Total samples/sec:     7.0907 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 169, 0.13857603073120117 sec
Batchsize: 1
Time spent per BATCH:   141.0149 ms
Total samples/sec:     7.0915 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0008 ms
Total samples/sec:     7.0922 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9996 ms
Total samples/sec:     7.0922 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9894 ms
Total samples/sec:     7.0927 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9764 ms
Total samples/sec:     7.0934 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9639 ms
Total samples/sec:     7.0940 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9530 ms
Total samples/sec:     7.0946 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9395 ms
Total samples/sec:     7.0952 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9258 ms
Total samples/sec:     7.0959 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9157 ms
Total samples/sec:     7.0964 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 179, 0.1391589641571045 sec
Batchsize: 1
Time spent per BATCH:   140.9060 ms
Total samples/sec:     7.0969 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9026 ms
Total samples/sec:     7.0971 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8907 ms
Total samples/sec:     7.0977 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8795 ms
Total samples/sec:     7.0983 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8679 ms
Total samples/sec:     7.0988 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8690 ms
Total samples/sec:     7.0988 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8576 ms
Total samples/sec:     7.0994 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8558 ms
Total samples/sec:     7.0995 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8449 ms
Total samples/sec:     7.1000 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8407 ms
Total samples/sec:     7.1002 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 189, 0.13945770263671875 sec
Batchsize: 1
Time spent per BATCH:   140.8334 ms
Total samples/sec:     7.1006 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8325 ms
Total samples/sec:     7.1006 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8300 ms
Total samples/sec:     7.1008 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8316 ms
Total samples/sec:     7.1007 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8203 ms
Total samples/sec:     7.1012 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8208 ms
Total samples/sec:     7.1012 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8108 ms
Total samples/sec:     7.1017 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8102 ms
Total samples/sec:     7.1018 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8002 ms
Total samples/sec:     7.1023 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8105 ms
Total samples/sec:     7.1017 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 199, 0.1386432647705078 sec
Batchsize: 1
Time spent per BATCH:   140.7997 ms
Total samples/sec:     7.1023 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7960 ms
Total samples/sec:     7.1025 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7867 ms
Total samples/sec:     7.1029 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7766 ms
Total samples/sec:     7.1035 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7666 ms
Total samples/sec:     7.1040 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7610 ms
Total samples/sec:     7.1042 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7505 ms
Total samples/sec:     7.1048 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7422 ms
Total samples/sec:     7.1052 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7324 ms
Total samples/sec:     7.1057 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7216 ms
Total samples/sec:     7.1062 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 209, 0.14659476280212402 sec
Batchsize: 1
Time spent per BATCH:   140.7496 ms
Total samples/sec:     7.1048 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7395 ms
Total samples/sec:     7.1053 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7294 ms
Total samples/sec:     7.1058 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7206 ms
Total samples/sec:     7.1063 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7102 ms
Total samples/sec:     7.1068 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7019 ms
Total samples/sec:     7.1072 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6968 ms
Total samples/sec:     7.1075 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6877 ms
Total samples/sec:     7.1079 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6830 ms
Total samples/sec:     7.1082 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6738 ms
Total samples/sec:     7.1086 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 219, 0.13875579833984375 sec
Batchsize: 1
Time spent per BATCH:   140.6651 ms
Total samples/sec:     7.1091 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6566 ms
Total samples/sec:     7.1095 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6488 ms
Total samples/sec:     7.1099 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6828 ms
Total samples/sec:     7.1082 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6744 ms
Total samples/sec:     7.1086 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6732 ms
Total samples/sec:     7.1087 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6649 ms
Total samples/sec:     7.1091 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6666 ms
Total samples/sec:     7.1090 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6582 ms
Total samples/sec:     7.1094 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6499 ms
Total samples/sec:     7.1099 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 229, 0.13861584663391113 sec
Batchsize: 1
Time spent per BATCH:   140.6411 ms
Total samples/sec:     7.1103 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6331 ms
Total samples/sec:     7.1107 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6360 ms
Total samples/sec:     7.1106 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6278 ms
Total samples/sec:     7.1110 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6211 ms
Total samples/sec:     7.1113 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6127 ms
Total samples/sec:     7.1117 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6047 ms
Total samples/sec:     7.1121 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6037 ms
Total samples/sec:     7.1122 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5954 ms
Total samples/sec:     7.1126 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5934 ms
Total samples/sec:     7.1127 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 239, 0.13866710662841797 sec
Batchsize: 1
Time spent per BATCH:   140.5854 ms
Total samples/sec:     7.1131 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5773 ms
Total samples/sec:     7.1135 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5702 ms
Total samples/sec:     7.1139 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5861 ms
Total samples/sec:     7.1131 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5782 ms
Total samples/sec:     7.1135 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6038 ms
Total samples/sec:     7.1122 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5969 ms
Total samples/sec:     7.1125 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5955 ms
Total samples/sec:     7.1126 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5871 ms
Total samples/sec:     7.1130 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5814 ms
Total samples/sec:     7.1133 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 249, 0.13855934143066406 sec
Batchsize: 1
Time spent per BATCH:   140.5733 ms
Total samples/sec:     7.1137 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5667 ms
Total samples/sec:     7.1141 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5601 ms
Total samples/sec:     7.1144 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6251 ms
Total samples/sec:     7.1111 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6171 ms
Total samples/sec:     7.1115 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6113 ms
Total samples/sec:     7.1118 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6046 ms
Total samples/sec:     7.1121 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6080 ms
Total samples/sec:     7.1120 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6005 ms
Total samples/sec:     7.1123 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5945 ms
Total samples/sec:     7.1127 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 259, 0.13847756385803223 sec
Batchsize: 1
Time spent per BATCH:   140.5864 ms
Total samples/sec:     7.1131 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5780 ms
Total samples/sec:     7.1135 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5709 ms
Total samples/sec:     7.1138 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5682 ms
Total samples/sec:     7.1140 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5605 ms
Total samples/sec:     7.1144 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5643 ms
Total samples/sec:     7.1142 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5619 ms
Total samples/sec:     7.1143 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5543 ms
Total samples/sec:     7.1147 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5502 ms
Total samples/sec:     7.1149 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5430 ms
Total samples/sec:     7.1153 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 269, 0.13862943649291992 sec
Batchsize: 1
Time spent per BATCH:   140.5359 ms
Total samples/sec:     7.1156 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5303 ms
Total samples/sec:     7.1159 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5229 ms
Total samples/sec:     7.1163 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5165 ms
Total samples/sec:     7.1166 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5106 ms
Total samples/sec:     7.1169 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5035 ms
Total samples/sec:     7.1173 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4965 ms
Total samples/sec:     7.1176 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4937 ms
Total samples/sec:     7.1178 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4884 ms
Total samples/sec:     7.1180 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4817 ms
Total samples/sec:     7.1184 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 279, 0.13909626007080078 sec
Batchsize: 1
Time spent per BATCH:   140.4767 ms
Total samples/sec:     7.1186 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4735 ms
Total samples/sec:     7.1188 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4665 ms
Total samples/sec:     7.1191 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4848 ms
Total samples/sec:     7.1182 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4874 ms
Total samples/sec:     7.1181 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4811 ms
Total samples/sec:     7.1184 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4765 ms
Total samples/sec:     7.1186 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4697 ms
Total samples/sec:     7.1190 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4630 ms
Total samples/sec:     7.1193 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4583 ms
Total samples/sec:     7.1196 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 289, 0.14132094383239746 sec
Batchsize: 1
Time spent per BATCH:   140.4613 ms
Total samples/sec:     7.1194 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4558 ms
Total samples/sec:     7.1197 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4673 ms
Total samples/sec:     7.1191 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4620 ms
Total samples/sec:     7.1194 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4644 ms
Total samples/sec:     7.1192 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4591 ms
Total samples/sec:     7.1195 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4538 ms
Total samples/sec:     7.1198 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4532 ms
Total samples/sec:     7.1198 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4480 ms
Total samples/sec:     7.1201 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4437 ms
Total samples/sec:     7.1203 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 299, 0.13919663429260254 sec
Batchsize: 1
Time spent per BATCH:   140.4396 ms
Total samples/sec:     7.1205 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4403 ms
Total samples/sec:     7.1205 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4343 ms
Total samples/sec:     7.1208 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4296 ms
Total samples/sec:     7.1210 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4230 ms
Total samples/sec:     7.1213 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4170 ms
Total samples/sec:     7.1216 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4165 ms
Total samples/sec:     7.1217 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4107 ms
Total samples/sec:     7.1220 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4210 ms
Total samples/sec:     7.1214 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4145 ms
Total samples/sec:     7.1218 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 309, 0.13878107070922852 sec
Batchsize: 1
Time spent per BATCH:   140.4093 ms
Total samples/sec:     7.1220 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4160 ms
Total samples/sec:     7.1217 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4203 ms
Total samples/sec:     7.1215 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4141 ms
Total samples/sec:     7.1218 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4116 ms
Total samples/sec:     7.1219 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4147 ms
Total samples/sec:     7.1218 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4149 ms
Total samples/sec:     7.1218 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4087 ms
Total samples/sec:     7.1221 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4028 ms
Total samples/sec:     7.1224 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3974 ms
Total samples/sec:     7.1226 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 319, 0.13911843299865723 sec
Batchsize: 1
Time spent per BATCH:   140.3934 ms
Total samples/sec:     7.1228 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3874 ms
Total samples/sec:     7.1231 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3819 ms
Total samples/sec:     7.1234 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3776 ms
Total samples/sec:     7.1236 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3715 ms
Total samples/sec:     7.1240 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3657 ms
Total samples/sec:     7.1242 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3599 ms
Total samples/sec:     7.1245 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3554 ms
Total samples/sec:     7.1248 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3500 ms
Total samples/sec:     7.1250 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3501 ms
Total samples/sec:     7.1250 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 329, 0.14357805252075195 sec
Batchsize: 1
Time spent per BATCH:   140.3599 ms
Total samples/sec:     7.1245 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3543 ms
Total samples/sec:     7.1248 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3509 ms
Total samples/sec:     7.1250 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3457 ms
Total samples/sec:     7.1253 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3467 ms
Total samples/sec:     7.1252 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3411 ms
Total samples/sec:     7.1255 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3355 ms
Total samples/sec:     7.1258 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3306 ms
Total samples/sec:     7.1260 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3250 ms
Total samples/sec:     7.1263 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3240 ms
Total samples/sec:     7.1264 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 339, 0.13847565650939941 sec
Batchsize: 1
Time spent per BATCH:   140.3185 ms
Total samples/sec:     7.1266 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3130 ms
Total samples/sec:     7.1269 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3095 ms
Total samples/sec:     7.1271 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3242 ms
Total samples/sec:     7.1264 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3437 ms
Total samples/sec:     7.1254 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3385 ms
Total samples/sec:     7.1256 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3334 ms
Total samples/sec:     7.1259 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3293 ms
Total samples/sec:     7.1261 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3244 ms
Total samples/sec:     7.1263 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3209 ms
Total samples/sec:     7.1265 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 349, 0.14301133155822754 sec
Batchsize: 1
Time spent per BATCH:   140.3286 ms
Total samples/sec:     7.1261 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3237 ms
Total samples/sec:     7.1264 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3211 ms
Total samples/sec:     7.1265 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3165 ms
Total samples/sec:     7.1267 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3124 ms
Total samples/sec:     7.1270 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3112 ms
Total samples/sec:     7.1270 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3066 ms
Total samples/sec:     7.1273 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3022 ms
Total samples/sec:     7.1275 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2972 ms
Total samples/sec:     7.1277 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2927 ms
Total samples/sec:     7.1280 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 359, 0.13886284828186035 sec
Batchsize: 1
Time spent per BATCH:   140.2887 ms
Total samples/sec:     7.1282 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2833 ms
Total samples/sec:     7.1284 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3049 ms
Total samples/sec:     7.1273 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3202 ms
Total samples/sec:     7.1266 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3165 ms
Total samples/sec:     7.1267 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3112 ms
Total samples/sec:     7.1270 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3065 ms
Total samples/sec:     7.1273 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3019 ms
Total samples/sec:     7.1275 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2979 ms
Total samples/sec:     7.1277 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3069 ms
Total samples/sec:     7.1272 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 369, 0.13843655586242676 sec
Batchsize: 1
Time spent per BATCH:   140.3018 ms
Total samples/sec:     7.1275 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2974 ms
Total samples/sec:     7.1277 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2926 ms
Total samples/sec:     7.1280 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2880 ms
Total samples/sec:     7.1282 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2917 ms
Total samples/sec:     7.1280 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2879 ms
Total samples/sec:     7.1282 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2921 ms
Total samples/sec:     7.1280 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2945 ms
Total samples/sec:     7.1279 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3007 ms
Total samples/sec:     7.1275 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2956 ms
Total samples/sec:     7.1278 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 379, 0.13839173316955566 sec
Batchsize: 1
Time spent per BATCH:   140.2906 ms
Total samples/sec:     7.1281 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2860 ms
Total samples/sec:     7.1283 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2817 ms
Total samples/sec:     7.1285 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2782 ms
Total samples/sec:     7.1287 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2738 ms
Total samples/sec:     7.1289 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2686 ms
Total samples/sec:     7.1292 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2738 ms
Total samples/sec:     7.1289 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2735 ms
Total samples/sec:     7.1289 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2690 ms
Total samples/sec:     7.1292 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2642 ms
Total samples/sec:     7.1294 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 389, 0.13850688934326172 sec
Batchsize: 1
Time spent per BATCH:   140.2597 ms
Total samples/sec:     7.1296 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2556 ms
Total samples/sec:     7.1298 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2512 ms
Total samples/sec:     7.1301 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2539 ms
Total samples/sec:     7.1299 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2506 ms
Total samples/sec:     7.1301 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2464 ms
Total samples/sec:     7.1303 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2424 ms
Total samples/sec:     7.1305 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2380 ms
Total samples/sec:     7.1307 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2350 ms
Total samples/sec:     7.1309 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2378 ms
Total samples/sec:     7.1307 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 399, 0.14478087425231934 sec
Batchsize: 1
Time spent per BATCH:   140.2492 ms
Total samples/sec:     7.1302 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2445 ms
Total samples/sec:     7.1304 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2401 ms
Total samples/sec:     7.1306 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2504 ms
Total samples/sec:     7.1301 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2458 ms
Total samples/sec:     7.1303 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2414 ms
Total samples/sec:     7.1306 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2385 ms
Total samples/sec:     7.1307 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2349 ms
Total samples/sec:     7.1309 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2311 ms
Total samples/sec:     7.1311 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2265 ms
Total samples/sec:     7.1313 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 409, 0.1384751796722412 sec
Batchsize: 1
Time spent per BATCH:   140.2222 ms
Total samples/sec:     7.1315 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2193 ms
Total samples/sec:     7.1317 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2286 ms
Total samples/sec:     7.1312 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2250 ms
Total samples/sec:     7.1314 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2209 ms
Total samples/sec:     7.1316 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2174 ms
Total samples/sec:     7.1318 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2261 ms
Total samples/sec:     7.1313 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2224 ms
Total samples/sec:     7.1315 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2188 ms
Total samples/sec:     7.1317 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2227 ms
Total samples/sec:     7.1315 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 419, 0.1385955810546875 sec
Batchsize: 1
Time spent per BATCH:   140.2188 ms
Total samples/sec:     7.1317 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2151 ms
Total samples/sec:     7.1319 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2109 ms
Total samples/sec:     7.1321 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2082 ms
Total samples/sec:     7.1323 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2195 ms
Total samples/sec:     7.1317 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2162 ms
Total samples/sec:     7.1318 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2125 ms
Total samples/sec:     7.1320 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2108 ms
Total samples/sec:     7.1321 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2081 ms
Total samples/sec:     7.1323 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2056 ms
Total samples/sec:     7.1324 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 429, 0.13833260536193848 sec
Batchsize: 1
Time spent per BATCH:   140.2013 ms
Total samples/sec:     7.1326 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1986 ms
Total samples/sec:     7.1327 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1956 ms
Total samples/sec:     7.1329 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1937 ms
Total samples/sec:     7.1330 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1897 ms
Total samples/sec:     7.1332 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1944 ms
Total samples/sec:     7.1330 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1906 ms
Total samples/sec:     7.1331 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1876 ms
Total samples/sec:     7.1333 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1847 ms
Total samples/sec:     7.1334 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1819 ms
Total samples/sec:     7.1336 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 439, 0.1393740177154541 sec
Batchsize: 1
Time spent per BATCH:   140.1801 ms
Total samples/sec:     7.1337 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1773 ms
Total samples/sec:     7.1338 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1841 ms
Total samples/sec:     7.1335 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1810 ms
Total samples/sec:     7.1336 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1779 ms
Total samples/sec:     7.1338 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1738 ms
Total samples/sec:     7.1340 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1700 ms
Total samples/sec:     7.1342 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1801 ms
Total samples/sec:     7.1337 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1777 ms
Total samples/sec:     7.1338 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1742 ms
Total samples/sec:     7.1340 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 449, 0.13843536376953125 sec
Batchsize: 1
Time spent per BATCH:   140.1703 ms
Total samples/sec:     7.1342 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1665 ms
Total samples/sec:     7.1344 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1628 ms
Total samples/sec:     7.1346 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1597 ms
Total samples/sec:     7.1347 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1556 ms
Total samples/sec:     7.1349 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1523 ms
Total samples/sec:     7.1351 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1551 ms
Total samples/sec:     7.1350 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1522 ms
Total samples/sec:     7.1351 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1594 ms
Total samples/sec:     7.1347 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1600 ms
Total samples/sec:     7.1347 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 459, 0.13859009742736816 sec
Batchsize: 1
Time spent per BATCH:   140.1566 ms
Total samples/sec:     7.1349 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1533 ms
Total samples/sec:     7.1350 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1500 ms
Total samples/sec:     7.1352 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1463 ms
Total samples/sec:     7.1354 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1432 ms
Total samples/sec:     7.1356 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1395 ms
Total samples/sec:     7.1357 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1372 ms
Total samples/sec:     7.1359 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1348 ms
Total samples/sec:     7.1360 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1331 ms
Total samples/sec:     7.1361 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1334 ms
Total samples/sec:     7.1361 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 469, 0.138380765914917 sec
Batchsize: 1
Time spent per BATCH:   140.1296 ms
Total samples/sec:     7.1362 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1268 ms
Total samples/sec:     7.1364 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1234 ms
Total samples/sec:     7.1366 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1329 ms
Total samples/sec:     7.1361 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1292 ms
Total samples/sec:     7.1363 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1256 ms
Total samples/sec:     7.1365 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1220 ms
Total samples/sec:     7.1366 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1222 ms
Total samples/sec:     7.1366 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1264 ms
Total samples/sec:     7.1364 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1244 ms
Total samples/sec:     7.1365 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 479, 0.13854742050170898 sec
Batchsize: 1
Time spent per BATCH:   140.1211 ms
Total samples/sec:     7.1367 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1177 ms
Total samples/sec:     7.1369 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1146 ms
Total samples/sec:     7.1370 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1110 ms
Total samples/sec:     7.1372 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1079 ms
Total samples/sec:     7.1374 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1077 ms
Total samples/sec:     7.1374 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1055 ms
Total samples/sec:     7.1375 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1024 ms
Total samples/sec:     7.1376 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0998 ms
Total samples/sec:     7.1378 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1082 ms
Total samples/sec:     7.1373 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 489, 0.13864469528198242 sec
Batchsize: 1
Time spent per BATCH:   140.1052 ms
Total samples/sec:     7.1375 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1018 ms
Total samples/sec:     7.1377 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1017 ms
Total samples/sec:     7.1377 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0990 ms
Total samples/sec:     7.1378 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0957 ms
Total samples/sec:     7.1380 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0924 ms
Total samples/sec:     7.1381 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0896 ms
Total samples/sec:     7.1383 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0863 ms
Total samples/sec:     7.1385 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0832 ms
Total samples/sec:     7.1386 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0805 ms
Total samples/sec:     7.1388 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 499, 0.14154863357543945 sec
Batchsize: 1
Time spent per BATCH:   140.0835 ms
Total samples/sec:     7.1386 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0804 ms
Total samples/sec:     7.1388 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0893 ms
Total samples/sec:     7.1383 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0866 ms
Total samples/sec:     7.1384 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0870 ms
Total samples/sec:     7.1384 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0843 ms
Total samples/sec:     7.1386 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0827 ms
Total samples/sec:     7.1386 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0795 ms
Total samples/sec:     7.1388 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.0771 ms
Total samples/sec:     7.1389 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1292 ms
Total samples/sec:     7.1363 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 509, 0.1435408592224121 sec
Batchsize: 1
Time spent per BATCH:   140.1359 ms
Total samples/sec:     7.1359 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1428 ms
Total samples/sec:     7.1356 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1467 ms
Total samples/sec:     7.1354 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1654 ms
Total samples/sec:     7.1344 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1730 ms
Total samples/sec:     7.1340 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1703 ms
Total samples/sec:     7.1342 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1670 ms
Total samples/sec:     7.1343 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1654 ms
Total samples/sec:     7.1344 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1635 ms
Total samples/sec:     7.1345 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1614 ms
Total samples/sec:     7.1346 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 519, 0.13901209831237793 sec
Batchsize: 1
Time spent per BATCH:   140.1592 ms
Total samples/sec:     7.1347 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1706 ms
Total samples/sec:     7.1342 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1682 ms
Total samples/sec:     7.1343 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.1818 ms
Total samples/sec:     7.1336 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2040 ms
Total samples/sec:     7.1325 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2138 ms
Total samples/sec:     7.1320 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2175 ms
Total samples/sec:     7.1318 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2212 ms
Total samples/sec:     7.1316 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2274 ms
Total samples/sec:     7.1313 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2245 ms
Total samples/sec:     7.1314 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 529, 0.13841533660888672 sec
Batchsize: 1
Time spent per BATCH:   140.2211 ms
Total samples/sec:     7.1316 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2180 ms
Total samples/sec:     7.1318 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2148 ms
Total samples/sec:     7.1319 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2115 ms
Total samples/sec:     7.1321 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2086 ms
Total samples/sec:     7.1322 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2056 ms
Total samples/sec:     7.1324 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2119 ms
Total samples/sec:     7.1321 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2104 ms
Total samples/sec:     7.1321 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2149 ms
Total samples/sec:     7.1319 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2247 ms
Total samples/sec:     7.1314 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 539, 0.1384439468383789 sec
Batchsize: 1
Time spent per BATCH:   140.2214 ms
Total samples/sec:     7.1316 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2179 ms
Total samples/sec:     7.1318 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2172 ms
Total samples/sec:     7.1318 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2140 ms
Total samples/sec:     7.1320 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2108 ms
Total samples/sec:     7.1321 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2115 ms
Total samples/sec:     7.1321 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2086 ms
Total samples/sec:     7.1322 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2171 ms
Total samples/sec:     7.1318 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2288 ms
Total samples/sec:     7.1312 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2315 ms
Total samples/sec:     7.1311 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 549, 0.14520955085754395 sec
Batchsize: 1
Time spent per BATCH:   140.2405 ms
Total samples/sec:     7.1306 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2445 ms
Total samples/sec:     7.1304 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2587 ms
Total samples/sec:     7.1297 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2609 ms
Total samples/sec:     7.1296 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2616 ms
Total samples/sec:     7.1295 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2623 ms
Total samples/sec:     7.1295 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2687 ms
Total samples/sec:     7.1292 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2657 ms
Total samples/sec:     7.1293 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2676 ms
Total samples/sec:     7.1292 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2643 ms
Total samples/sec:     7.1294 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 559, 0.13852572441101074 sec
Batchsize: 1
Time spent per BATCH:   140.2612 ms
Total samples/sec:     7.1296 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2588 ms
Total samples/sec:     7.1297 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2592 ms
Total samples/sec:     7.1297 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2584 ms
Total samples/sec:     7.1297 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2576 ms
Total samples/sec:     7.1297 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2543 ms
Total samples/sec:     7.1299 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2513 ms
Total samples/sec:     7.1301 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2531 ms
Total samples/sec:     7.1300 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2498 ms
Total samples/sec:     7.1301 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2471 ms
Total samples/sec:     7.1303 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 569, 0.13991355895996094 sec
Batchsize: 1
Time spent per BATCH:   140.2465 ms
Total samples/sec:     7.1303 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2448 ms
Total samples/sec:     7.1304 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2462 ms
Total samples/sec:     7.1303 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2451 ms
Total samples/sec:     7.1304 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2507 ms
Total samples/sec:     7.1301 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2534 ms
Total samples/sec:     7.1299 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2503 ms
Total samples/sec:     7.1301 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2580 ms
Total samples/sec:     7.1297 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2615 ms
Total samples/sec:     7.1295 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2586 ms
Total samples/sec:     7.1297 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 579, 0.1392650604248047 sec
Batchsize: 1
Time spent per BATCH:   140.2569 ms
Total samples/sec:     7.1298 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2616 ms
Total samples/sec:     7.1295 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2661 ms
Total samples/sec:     7.1293 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2676 ms
Total samples/sec:     7.1292 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2646 ms
Total samples/sec:     7.1294 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2691 ms
Total samples/sec:     7.1292 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2704 ms
Total samples/sec:     7.1291 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2787 ms
Total samples/sec:     7.1287 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.2883 ms
Total samples/sec:     7.1282 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3191 ms
Total samples/sec:     7.1266 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 589, 0.14367246627807617 sec
Batchsize: 1
Time spent per BATCH:   140.3247 ms
Total samples/sec:     7.1263 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3217 ms
Total samples/sec:     7.1265 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3253 ms
Total samples/sec:     7.1263 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3231 ms
Total samples/sec:     7.1264 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3205 ms
Total samples/sec:     7.1265 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3216 ms
Total samples/sec:     7.1265 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3206 ms
Total samples/sec:     7.1265 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3180 ms
Total samples/sec:     7.1267 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3174 ms
Total samples/sec:     7.1267 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3176 ms
Total samples/sec:     7.1267 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 599, 0.14467239379882812 sec
Batchsize: 1
Time spent per BATCH:   140.3248 ms
Total samples/sec:     7.1263 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3326 ms
Total samples/sec:     7.1259 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3367 ms
Total samples/sec:     7.1257 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3464 ms
Total samples/sec:     7.1252 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3486 ms
Total samples/sec:     7.1251 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3618 ms
Total samples/sec:     7.1244 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3752 ms
Total samples/sec:     7.1238 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3729 ms
Total samples/sec:     7.1239 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3699 ms
Total samples/sec:     7.1240 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3767 ms
Total samples/sec:     7.1237 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 609, 0.1406395435333252 sec
Batchsize: 1
Time spent per BATCH:   140.3771 ms
Total samples/sec:     7.1237 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3838 ms
Total samples/sec:     7.1233 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3952 ms
Total samples/sec:     7.1227 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3951 ms
Total samples/sec:     7.1228 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.3968 ms
Total samples/sec:     7.1227 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4017 ms
Total samples/sec:     7.1224 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4105 ms
Total samples/sec:     7.1220 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4194 ms
Total samples/sec:     7.1215 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4238 ms
Total samples/sec:     7.1213 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4311 ms
Total samples/sec:     7.1209 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 619, 0.1475667953491211 sec
Batchsize: 1
Time spent per BATCH:   140.4426 ms
Total samples/sec:     7.1203 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4395 ms
Total samples/sec:     7.1205 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4452 ms
Total samples/sec:     7.1202 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4570 ms
Total samples/sec:     7.1196 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4591 ms
Total samples/sec:     7.1195 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4561 ms
Total samples/sec:     7.1197 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4531 ms
Total samples/sec:     7.1198 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4501 ms
Total samples/sec:     7.1200 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4478 ms
Total samples/sec:     7.1201 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4455 ms
Total samples/sec:     7.1202 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 629, 0.13857555389404297 sec
Batchsize: 1
Time spent per BATCH:   140.4426 ms
Total samples/sec:     7.1203 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4409 ms
Total samples/sec:     7.1204 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4377 ms
Total samples/sec:     7.1206 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4378 ms
Total samples/sec:     7.1206 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4349 ms
Total samples/sec:     7.1207 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4349 ms
Total samples/sec:     7.1207 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4319 ms
Total samples/sec:     7.1209 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4306 ms
Total samples/sec:     7.1210 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4305 ms
Total samples/sec:     7.1210 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4386 ms
Total samples/sec:     7.1205 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 639, 0.1385505199432373 sec
Batchsize: 1
Time spent per BATCH:   140.4357 ms
Total samples/sec:     7.1207 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4444 ms
Total samples/sec:     7.1203 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4414 ms
Total samples/sec:     7.1204 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4470 ms
Total samples/sec:     7.1201 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4534 ms
Total samples/sec:     7.1198 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4504 ms
Total samples/sec:     7.1200 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4523 ms
Total samples/sec:     7.1199 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4495 ms
Total samples/sec:     7.1200 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4490 ms
Total samples/sec:     7.1200 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4459 ms
Total samples/sec:     7.1202 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 649, 0.13983416557312012 sec
Batchsize: 1
Time spent per BATCH:   140.4450 ms
Total samples/sec:     7.1202 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4419 ms
Total samples/sec:     7.1204 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4404 ms
Total samples/sec:     7.1205 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4401 ms
Total samples/sec:     7.1205 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4430 ms
Total samples/sec:     7.1203 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4425 ms
Total samples/sec:     7.1204 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4449 ms
Total samples/sec:     7.1202 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4419 ms
Total samples/sec:     7.1204 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4397 ms
Total samples/sec:     7.1205 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4369 ms
Total samples/sec:     7.1206 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 659, 0.13843154907226562 sec
Batchsize: 1
Time spent per BATCH:   140.4339 ms
Total samples/sec:     7.1208 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4331 ms
Total samples/sec:     7.1208 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4305 ms
Total samples/sec:     7.1210 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4298 ms
Total samples/sec:     7.1210 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4271 ms
Total samples/sec:     7.1211 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4343 ms
Total samples/sec:     7.1208 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4316 ms
Total samples/sec:     7.1209 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4333 ms
Total samples/sec:     7.1208 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4467 ms
Total samples/sec:     7.1201 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4690 ms
Total samples/sec:     7.1190 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 669, 0.1504535675048828 sec
Batchsize: 1
Time spent per BATCH:   140.4839 ms
Total samples/sec:     7.1183 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4814 ms
Total samples/sec:     7.1184 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4819 ms
Total samples/sec:     7.1184 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4825 ms
Total samples/sec:     7.1183 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4822 ms
Total samples/sec:     7.1183 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4793 ms
Total samples/sec:     7.1185 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4785 ms
Total samples/sec:     7.1185 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4906 ms
Total samples/sec:     7.1179 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4982 ms
Total samples/sec:     7.1175 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4972 ms
Total samples/sec:     7.1176 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 679, 0.1442277431488037 sec
Batchsize: 1
Time spent per BATCH:   140.5026 ms
Total samples/sec:     7.1173 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5020 ms
Total samples/sec:     7.1173 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.4998 ms
Total samples/sec:     7.1174 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5036 ms
Total samples/sec:     7.1173 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5009 ms
Total samples/sec:     7.1174 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5131 ms
Total samples/sec:     7.1168 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5246 ms
Total samples/sec:     7.1162 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5253 ms
Total samples/sec:     7.1162 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5298 ms
Total samples/sec:     7.1159 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5446 ms
Total samples/sec:     7.1152 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 689, 0.1412656307220459 sec
Batchsize: 1
Time spent per BATCH:   140.5457 ms
Total samples/sec:     7.1151 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5515 ms
Total samples/sec:     7.1148 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5537 ms
Total samples/sec:     7.1147 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5533 ms
Total samples/sec:     7.1147 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5623 ms
Total samples/sec:     7.1143 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5598 ms
Total samples/sec:     7.1144 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5747 ms
Total samples/sec:     7.1137 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5778 ms
Total samples/sec:     7.1135 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5771 ms
Total samples/sec:     7.1135 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5818 ms
Total samples/sec:     7.1133 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 699, 0.14870882034301758 sec
Batchsize: 1
Time spent per BATCH:   140.5934 ms
Total samples/sec:     7.1127 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5942 ms
Total samples/sec:     7.1127 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5911 ms
Total samples/sec:     7.1128 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.5890 ms
Total samples/sec:     7.1129 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6176 ms
Total samples/sec:     7.1115 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6165 ms
Total samples/sec:     7.1115 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6351 ms
Total samples/sec:     7.1106 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6395 ms
Total samples/sec:     7.1104 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6597 ms
Total samples/sec:     7.1094 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6659 ms
Total samples/sec:     7.1090 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 709, 0.1485881805419922 sec
Batchsize: 1
Time spent per BATCH:   140.6770 ms
Total samples/sec:     7.1085 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6783 ms
Total samples/sec:     7.1084 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6863 ms
Total samples/sec:     7.1080 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6917 ms
Total samples/sec:     7.1077 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6900 ms
Total samples/sec:     7.1078 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6870 ms
Total samples/sec:     7.1080 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6855 ms
Total samples/sec:     7.1081 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6855 ms
Total samples/sec:     7.1081 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6832 ms
Total samples/sec:     7.1082 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6801 ms
Total samples/sec:     7.1083 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 719, 0.14417362213134766 sec
Batchsize: 1
Time spent per BATCH:   140.6850 ms
Total samples/sec:     7.1081 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6852 ms
Total samples/sec:     7.1081 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6843 ms
Total samples/sec:     7.1081 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6831 ms
Total samples/sec:     7.1082 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6863 ms
Total samples/sec:     7.1080 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6903 ms
Total samples/sec:     7.1078 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6889 ms
Total samples/sec:     7.1079 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6862 ms
Total samples/sec:     7.1080 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.6936 ms
Total samples/sec:     7.1076 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7006 ms
Total samples/sec:     7.1073 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 729, 0.1435396671295166 sec
Batchsize: 1
Time spent per BATCH:   140.7045 ms
Total samples/sec:     7.1071 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7115 ms
Total samples/sec:     7.1067 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7289 ms
Total samples/sec:     7.1059 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7290 ms
Total samples/sec:     7.1059 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7296 ms
Total samples/sec:     7.1058 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7286 ms
Total samples/sec:     7.1059 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7318 ms
Total samples/sec:     7.1057 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7359 ms
Total samples/sec:     7.1055 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7424 ms
Total samples/sec:     7.1052 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7469 ms
Total samples/sec:     7.1050 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 739, 0.14096975326538086 sec
Batchsize: 1
Time spent per BATCH:   140.7472 ms
Total samples/sec:     7.1049 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7530 ms
Total samples/sec:     7.1046 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7537 ms
Total samples/sec:     7.1046 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7600 ms
Total samples/sec:     7.1043 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7642 ms
Total samples/sec:     7.1041 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7695 ms
Total samples/sec:     7.1038 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7731 ms
Total samples/sec:     7.1036 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7715 ms
Total samples/sec:     7.1037 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7733 ms
Total samples/sec:     7.1036 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7772 ms
Total samples/sec:     7.1034 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 749, 0.13924479484558105 sec
Batchsize: 1
Time spent per BATCH:   140.7752 ms
Total samples/sec:     7.1035 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7733 ms
Total samples/sec:     7.1036 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7724 ms
Total samples/sec:     7.1037 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7697 ms
Total samples/sec:     7.1038 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7692 ms
Total samples/sec:     7.1038 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7666 ms
Total samples/sec:     7.1040 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7642 ms
Total samples/sec:     7.1041 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7682 ms
Total samples/sec:     7.1039 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7800 ms
Total samples/sec:     7.1033 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7830 ms
Total samples/sec:     7.1031 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 759, 0.1412949562072754 sec
Batchsize: 1
Time spent per BATCH:   140.7837 ms
Total samples/sec:     7.1031 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.7999 ms
Total samples/sec:     7.1023 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8016 ms
Total samples/sec:     7.1022 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8011 ms
Total samples/sec:     7.1022 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8009 ms
Total samples/sec:     7.1022 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8002 ms
Total samples/sec:     7.1023 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8066 ms
Total samples/sec:     7.1019 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8159 ms
Total samples/sec:     7.1015 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8133 ms
Total samples/sec:     7.1016 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8171 ms
Total samples/sec:     7.1014 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 769, 0.1473088264465332 sec
Batchsize: 1
Time spent per BATCH:   140.8255 ms
Total samples/sec:     7.1010 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8313 ms
Total samples/sec:     7.1007 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8304 ms
Total samples/sec:     7.1007 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8285 ms
Total samples/sec:     7.1008 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8338 ms
Total samples/sec:     7.1006 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8314 ms
Total samples/sec:     7.1007 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8292 ms
Total samples/sec:     7.1008 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8322 ms
Total samples/sec:     7.1007 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8307 ms
Total samples/sec:     7.1007 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8462 ms
Total samples/sec:     7.0999 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 779, 0.14352989196777344 sec
Batchsize: 1
Time spent per BATCH:   140.8496 ms
Total samples/sec:     7.0998 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8526 ms
Total samples/sec:     7.0996 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8709 ms
Total samples/sec:     7.0987 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8793 ms
Total samples/sec:     7.0983 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8817 ms
Total samples/sec:     7.0982 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8789 ms
Total samples/sec:     7.0983 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8761 ms
Total samples/sec:     7.0984 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8787 ms
Total samples/sec:     7.0983 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8936 ms
Total samples/sec:     7.0976 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.8963 ms
Total samples/sec:     7.0974 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 789, 0.1470797061920166 sec
Batchsize: 1
Time spent per BATCH:   140.9041 ms
Total samples/sec:     7.0970 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9114 ms
Total samples/sec:     7.0967 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9189 ms
Total samples/sec:     7.0963 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9185 ms
Total samples/sec:     7.0963 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9155 ms
Total samples/sec:     7.0964 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9153 ms
Total samples/sec:     7.0965 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9256 ms
Total samples/sec:     7.0959 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9270 ms
Total samples/sec:     7.0959 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9396 ms
Total samples/sec:     7.0952 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9469 ms
Total samples/sec:     7.0949 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 799, 0.14446306228637695 sec
Batchsize: 1
Time spent per BATCH:   140.9513 ms
Total samples/sec:     7.0946 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9631 ms
Total samples/sec:     7.0941 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9616 ms
Total samples/sec:     7.0941 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9626 ms
Total samples/sec:     7.0941 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9608 ms
Total samples/sec:     7.0942 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9703 ms
Total samples/sec:     7.0937 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9779 ms
Total samples/sec:     7.0933 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9794 ms
Total samples/sec:     7.0932 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9844 ms
Total samples/sec:     7.0930 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9855 ms
Total samples/sec:     7.0929 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 809, 0.13878297805786133 sec
Batchsize: 1
Time spent per BATCH:   140.9828 ms
Total samples/sec:     7.0931 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9827 ms
Total samples/sec:     7.0931 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9927 ms
Total samples/sec:     7.0926 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9901 ms
Total samples/sec:     7.0927 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9927 ms
Total samples/sec:     7.0926 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9902 ms
Total samples/sec:     7.0927 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9919 ms
Total samples/sec:     7.0926 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9893 ms
Total samples/sec:     7.0927 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   140.9997 ms
Total samples/sec:     7.0922 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0046 ms
Total samples/sec:     7.0920 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 819, 0.14968347549438477 sec
Batchsize: 1
Time spent per BATCH:   141.0152 ms
Total samples/sec:     7.0914 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0205 ms
Total samples/sec:     7.0912 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0248 ms
Total samples/sec:     7.0909 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0220 ms
Total samples/sec:     7.0911 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0192 ms
Total samples/sec:     7.0912 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0188 ms
Total samples/sec:     7.0913 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0262 ms
Total samples/sec:     7.0909 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0294 ms
Total samples/sec:     7.0907 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0284 ms
Total samples/sec:     7.0908 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0283 ms
Total samples/sec:     7.0908 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 829, 0.1417696475982666 sec
Batchsize: 1
Time spent per BATCH:   141.0292 ms
Total samples/sec:     7.0907 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0359 ms
Total samples/sec:     7.0904 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0389 ms
Total samples/sec:     7.0902 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0459 ms
Total samples/sec:     7.0899 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0438 ms
Total samples/sec:     7.0900 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0416 ms
Total samples/sec:     7.0901 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0415 ms
Total samples/sec:     7.0901 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0425 ms
Total samples/sec:     7.0901 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0421 ms
Total samples/sec:     7.0901 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0424 ms
Total samples/sec:     7.0901 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 839, 0.1410839557647705 sec
Batchsize: 1
Time spent per BATCH:   141.0424 ms
Total samples/sec:     7.0901 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0439 ms
Total samples/sec:     7.0900 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0432 ms
Total samples/sec:     7.0900 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0406 ms
Total samples/sec:     7.0902 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0450 ms
Total samples/sec:     7.0899 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0533 ms
Total samples/sec:     7.0895 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0506 ms
Total samples/sec:     7.0897 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0545 ms
Total samples/sec:     7.0895 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0584 ms
Total samples/sec:     7.0893 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0583 ms
Total samples/sec:     7.0893 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 849, 0.14354801177978516 sec
Batchsize: 1
Time spent per BATCH:   141.0612 ms
Total samples/sec:     7.0891 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0583 ms
Total samples/sec:     7.0893 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0572 ms
Total samples/sec:     7.0893 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0543 ms
Total samples/sec:     7.0895 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0523 ms
Total samples/sec:     7.0896 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0596 ms
Total samples/sec:     7.0892 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0643 ms
Total samples/sec:     7.0890 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0637 ms
Total samples/sec:     7.0890 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0611 ms
Total samples/sec:     7.0891 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0595 ms
Total samples/sec:     7.0892 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 859, 0.1469571590423584 sec
Batchsize: 1
Time spent per BATCH:   141.0664 ms
Total samples/sec:     7.0889 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0638 ms
Total samples/sec:     7.0890 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0691 ms
Total samples/sec:     7.0887 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0738 ms
Total samples/sec:     7.0885 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0719 ms
Total samples/sec:     7.0886 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0698 ms
Total samples/sec:     7.0887 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0740 ms
Total samples/sec:     7.0885 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0802 ms
Total samples/sec:     7.0882 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0824 ms
Total samples/sec:     7.0881 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0859 ms
Total samples/sec:     7.0879 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 869, 0.14398837089538574 sec
Batchsize: 1
Time spent per BATCH:   141.0893 ms
Total samples/sec:     7.0877 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0864 ms
Total samples/sec:     7.0879 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0849 ms
Total samples/sec:     7.0879 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0854 ms
Total samples/sec:     7.0879 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0839 ms
Total samples/sec:     7.0880 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0816 ms
Total samples/sec:     7.0881 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0815 ms
Total samples/sec:     7.0881 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0798 ms
Total samples/sec:     7.0882 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0789 ms
Total samples/sec:     7.0882 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0802 ms
Total samples/sec:     7.0882 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 879, 0.14240026473999023 sec
Batchsize: 1
Time spent per BATCH:   141.0817 ms
Total samples/sec:     7.0881 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0817 ms
Total samples/sec:     7.0881 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0820 ms
Total samples/sec:     7.0881 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0866 ms
Total samples/sec:     7.0878 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0845 ms
Total samples/sec:     7.0879 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0866 ms
Total samples/sec:     7.0878 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0897 ms
Total samples/sec:     7.0877 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0943 ms
Total samples/sec:     7.0875 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0920 ms
Total samples/sec:     7.0876 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0917 ms
Total samples/sec:     7.0876 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 889, 0.14151978492736816 sec
Batchsize: 1
Time spent per BATCH:   141.0922 ms
Total samples/sec:     7.0876 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.0955 ms
Total samples/sec:     7.0874 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1061 ms
Total samples/sec:     7.0869 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1141 ms
Total samples/sec:     7.0865 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1115 ms
Total samples/sec:     7.0866 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1119 ms
Total samples/sec:     7.0866 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1102 ms
Total samples/sec:     7.0867 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1140 ms
Total samples/sec:     7.0865 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1110 ms
Total samples/sec:     7.0866 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1088 ms
Total samples/sec:     7.0867 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 899, 0.1438310146331787 sec
Batchsize: 1
Time spent per BATCH:   141.1118 ms
Total samples/sec:     7.0866 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1098 ms
Total samples/sec:     7.0867 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1305 ms
Total samples/sec:     7.0856 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1381 ms
Total samples/sec:     7.0853 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1389 ms
Total samples/sec:     7.0852 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1378 ms
Total samples/sec:     7.0853 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1354 ms
Total samples/sec:     7.0854 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1380 ms
Total samples/sec:     7.0853 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1369 ms
Total samples/sec:     7.0853 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1498 ms
Total samples/sec:     7.0847 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 909, 0.140366792678833 sec
Batchsize: 1
Time spent per BATCH:   141.1489 ms
Total samples/sec:     7.0847 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1482 ms
Total samples/sec:     7.0848 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1657 ms
Total samples/sec:     7.0839 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1660 ms
Total samples/sec:     7.0839 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1717 ms
Total samples/sec:     7.0836 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1718 ms
Total samples/sec:     7.0836 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1767 ms
Total samples/sec:     7.0833 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1764 ms
Total samples/sec:     7.0833 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1737 ms
Total samples/sec:     7.0835 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1775 ms
Total samples/sec:     7.0833 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 919, 0.14721155166625977 sec
Batchsize: 1
Time spent per BATCH:   141.1840 ms
Total samples/sec:     7.0830 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1898 ms
Total samples/sec:     7.0827 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1872 ms
Total samples/sec:     7.0828 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1856 ms
Total samples/sec:     7.0829 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1836 ms
Total samples/sec:     7.0830 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1852 ms
Total samples/sec:     7.0829 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1919 ms
Total samples/sec:     7.0826 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1907 ms
Total samples/sec:     7.0826 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1977 ms
Total samples/sec:     7.0823 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.1966 ms
Total samples/sec:     7.0823 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 929, 0.14335417747497559 sec
Batchsize: 1
Time spent per BATCH:   141.1989 ms
Total samples/sec:     7.0822 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2013 ms
Total samples/sec:     7.0821 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2037 ms
Total samples/sec:     7.0820 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2081 ms
Total samples/sec:     7.0817 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2109 ms
Total samples/sec:     7.0816 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2173 ms
Total samples/sec:     7.0813 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2189 ms
Total samples/sec:     7.0812 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2231 ms
Total samples/sec:     7.0810 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2263 ms
Total samples/sec:     7.0808 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2301 ms
Total samples/sec:     7.0806 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 939, 0.13936209678649902 sec
Batchsize: 1
Time spent per BATCH:   141.2281 ms
Total samples/sec:     7.0807 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2298 ms
Total samples/sec:     7.0807 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2311 ms
Total samples/sec:     7.0806 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2319 ms
Total samples/sec:     7.0806 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2304 ms
Total samples/sec:     7.0806 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2363 ms
Total samples/sec:     7.0803 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2335 ms
Total samples/sec:     7.0805 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2327 ms
Total samples/sec:     7.0805 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2299 ms
Total samples/sec:     7.0807 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2276 ms
Total samples/sec:     7.0808 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 949, 0.13843750953674316 sec
Batchsize: 1
Time spent per BATCH:   141.2246 ms
Total samples/sec:     7.0809 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2262 ms
Total samples/sec:     7.0808 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2268 ms
Total samples/sec:     7.0808 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2254 ms
Total samples/sec:     7.0809 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2240 ms
Total samples/sec:     7.0809 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2307 ms
Total samples/sec:     7.0806 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2327 ms
Total samples/sec:     7.0805 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2315 ms
Total samples/sec:     7.0806 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2287 ms
Total samples/sec:     7.0807 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2335 ms
Total samples/sec:     7.0805 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 959, 0.14458942413330078 sec
Batchsize: 1
Time spent per BATCH:   141.2370 ms
Total samples/sec:     7.0803 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2371 ms
Total samples/sec:     7.0803 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2384 ms
Total samples/sec:     7.0802 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2357 ms
Total samples/sec:     7.0804 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2400 ms
Total samples/sec:     7.0801 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2387 ms
Total samples/sec:     7.0802 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2435 ms
Total samples/sec:     7.0800 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2444 ms
Total samples/sec:     7.0799 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2453 ms
Total samples/sec:     7.0799 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2473 ms
Total samples/sec:     7.0798 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 969, 0.1459517478942871 sec
Batchsize: 1
Time spent per BATCH:   141.2521 ms
Total samples/sec:     7.0795 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2547 ms
Total samples/sec:     7.0794 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2574 ms
Total samples/sec:     7.0793 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2574 ms
Total samples/sec:     7.0793 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2591 ms
Total samples/sec:     7.0792 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2610 ms
Total samples/sec:     7.0791 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2580 ms
Total samples/sec:     7.0792 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2598 ms
Total samples/sec:     7.0792 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2608 ms
Total samples/sec:     7.0791 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2663 ms
Total samples/sec:     7.0788 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 979, 0.14074921607971191 sec
Batchsize: 1
Time spent per BATCH:   141.2658 ms
Total samples/sec:     7.0789 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2643 ms
Total samples/sec:     7.0789 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2619 ms
Total samples/sec:     7.0790 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2772 ms
Total samples/sec:     7.0783 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2813 ms
Total samples/sec:     7.0781 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2787 ms
Total samples/sec:     7.0782 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2799 ms
Total samples/sec:     7.0781 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2788 ms
Total samples/sec:     7.0782 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2820 ms
Total samples/sec:     7.0780 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2801 ms
Total samples/sec:     7.0781 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 989, 0.1406261920928955 sec
Batchsize: 1
Time spent per BATCH:   141.2795 ms
Total samples/sec:     7.0782 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2843 ms
Total samples/sec:     7.0779 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2852 ms
Total samples/sec:     7.0779 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2830 ms
Total samples/sec:     7.0780 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2839 ms
Total samples/sec:     7.0779 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2828 ms
Total samples/sec:     7.0780 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2900 ms
Total samples/sec:     7.0776 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2901 ms
Total samples/sec:     7.0776 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2874 ms
Total samples/sec:     7.0778 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
Batchsize: 1
Time spent per BATCH:   141.2856 ms
Total samples/sec:     7.0779 samples/s
Total labeled samples: 1 person
(1, 3, 224, 224)
steps = 999, 0.13873958587646484 sec
Batchsize: 1
Time spent per BATCH:   141.2831 ms
Total samples/sec:     7.0780 samples/s
Total labeled samples: 1 person
Received these standard args: Namespace(accuracy_only=False, annotations_dir='/home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/caffe2-pose-estimation/annotations/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages', batch_size=1, benchmark_dir='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks', benchmark_only=True, checkpoint=None, data_location='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/pascal_voc_tfrecord/tfrecord-voc.record', data_num_inter_threads=None, data_num_intra_threads=None, disable_tcmalloc=True, framework='caffe', input_graph='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.prototxt', input_weights='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.caffemodel', intelai_models='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/detection_softmax', mode='inference', model_args=[], model_name='detection_softmax', model_source_dir='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet', mpi=None, num_cores=2, num_instances=1, num_inter_threads=1, num_intra_threads=1, num_mpi=1, output_dir='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord', output_results=False, precision='fp32', risk_difference=0.5, socket_id=0, tcmalloc_large_alloc_report_threshold=2147483648, use_case='object_detection', verbose=True)
Received these custom args: []
Current directory: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks
Running: numactl --cpunodebind=0 --membind=0 /usr/bin/python3.6 /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/detection_softmax/inference/fp32/infer_detections.py -i 1000 -w 200 -a 1 -e 1 -g /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.prototxt -weight /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.caffemodel -rd 0.5 -b 1 -bo True -d /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/pascal_voc_tfrecord/tfrecord-voc.record --annotations_dir /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/caffe2-pose-estimation/annotations/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages
PYTHONPATH: :/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/detection_softmax
RUNCMD: /usr/bin/python3.6 common/caffe/run_tf_benchmark.py --framework=caffe --use-case=object_detection --model-name=detection_softmax --precision=fp32 --mode=inference --benchmark-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks --intelai-models=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/detection_softmax --num-cores=2 --batch-size=1 --socket-id=0 --output-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord --annotations_dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/caffe2-pose-estimation/annotations/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages  --benchmark-only  --verbose --model-source-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet --in-graph=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.prototxt --in-weights=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.caffemodel --data-location=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/pascal_voc_tfrecord/tfrecord-voc.record --num-inter-threads=1 --num-intra-threads=1 --disable-tcmalloc=True                   
Batch Size: 1
Ran inference with batch size 1
Log location outside container: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord/benchmark_detection_softmax_inference_fp32_20200518_222334.log
