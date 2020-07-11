2020-05-19 23:37:46.204779: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2993015000 Hz
2020-05-19 23:37:46.205159: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x2ffa1d0 executing computations on platform Host. Devices:
2020-05-19 23:37:46.205179: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
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
WARNING:tensorflow:From /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/ssd_squeezenet/inference/fp32/infer_detections.py:214: string_input_producer (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
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
WARNING:tensorflow:From /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/ssd_squeezenet/inference/fp32/infer_detections.py:219: TFRecordReader.__init__ (from tensorflow.python.ops.io_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.TFRecordDataset`.
WARNING:tensorflow:From /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/ssd_squeezenet/inference/fp32/infer_detections.py:223: start_queue_runners (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
graph has been loaded using caffe..
Inference with real data.
total iteration is 1000
warm up iteration is 0
Batchsize: 1
Time spent per BATCH:    34.4713 ms
Total samples/sec:    29.0097 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:    20.7443 ms
Total samples/sec:    48.2060 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:    15.6508 ms
Total samples/sec:    63.8944 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:    13.6071 ms
Total samples/sec:    73.4908 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:    12.3462 ms
Total samples/sec:    80.9968 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:    11.6786 ms
Total samples/sec:    85.6269 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:    11.0716 ms
Total samples/sec:    90.3208 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:    10.5378 ms
Total samples/sec:    94.8962 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:    10.3779 ms
Total samples/sec:    96.3588 samples/s
Total labeled samples: 1 person
steps = 9, 0.008670330047607422 sec
Batchsize: 1
Time spent per BATCH:    10.2071 ms
Total samples/sec:    97.9707 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     9.9153 ms
Total samples/sec:   100.8544 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     9.5510 ms
Total samples/sec:   104.7014 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     9.3178 ms
Total samples/sec:   107.3212 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     9.0496 ms
Total samples/sec:   110.5023 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     9.0066 ms
Total samples/sec:   111.0299 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     8.9027 ms
Total samples/sec:   112.3251 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     8.7154 ms
Total samples/sec:   114.7394 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     8.5394 ms
Total samples/sec:   117.1043 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     8.3859 ms
Total samples/sec:   119.2471 samples/s
Total labeled samples: 1 person
steps = 19, 0.005570650100708008 sec
Batchsize: 1
Time spent per BATCH:     8.2452 ms
Total samples/sec:   121.2829 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     8.1180 ms
Total samples/sec:   123.1833 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     8.0034 ms
Total samples/sec:   124.9466 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.8974 ms
Total samples/sec:   126.6243 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.8007 ms
Total samples/sec:   128.1941 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.7108 ms
Total samples/sec:   129.6890 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.6317 ms
Total samples/sec:   131.0318 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.5568 ms
Total samples/sec:   132.3310 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.5993 ms
Total samples/sec:   131.5910 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.6240 ms
Total samples/sec:   131.1640 samples/s
Total labeled samples: 1 person
steps = 29, 0.00556492805480957 sec
Batchsize: 1
Time spent per BATCH:     7.5554 ms
Total samples/sec:   132.3556 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.4974 ms
Total samples/sec:   133.3791 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.4404 ms
Total samples/sec:   134.4020 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.3856 ms
Total samples/sec:   135.3983 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.3326 ms
Total samples/sec:   136.3774 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.2850 ms
Total samples/sec:   137.2682 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.2374 ms
Total samples/sec:   138.1714 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.1918 ms
Total samples/sec:   139.0470 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.2556 ms
Total samples/sec:   137.8252 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.2117 ms
Total samples/sec:   138.6640 samples/s
Total labeled samples: 1 person
steps = 39, 0.00561833381652832 sec
Batchsize: 1
Time spent per BATCH:     7.1718 ms
Total samples/sec:   139.4341 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.1322 ms
Total samples/sec:   140.2085 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.0965 ms
Total samples/sec:   140.9147 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.1159 ms
Total samples/sec:   140.5312 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.0815 ms
Total samples/sec:   141.2138 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.0474 ms
Total samples/sec:   141.8954 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     7.0147 ms
Total samples/sec:   142.5568 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.9845 ms
Total samples/sec:   143.1742 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.9545 ms
Total samples/sec:   143.7924 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.9261 ms
Total samples/sec:   144.3822 samples/s
Total labeled samples: 1 person
steps = 49, 0.005593776702880859 sec
Batchsize: 1
Time spent per BATCH:     6.8994 ms
Total samples/sec:   144.9398 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.8732 ms
Total samples/sec:   145.4927 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.9240 ms
Total samples/sec:   144.4249 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.8984 ms
Total samples/sec:   144.9609 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.8742 ms
Total samples/sec:   145.4707 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.8504 ms
Total samples/sec:   145.9777 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.8289 ms
Total samples/sec:   146.4372 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.8067 ms
Total samples/sec:   146.9150 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.7862 ms
Total samples/sec:   147.3575 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.7658 ms
Total samples/sec:   147.8014 samples/s
Total labeled samples: 1 person
steps = 59, 0.005594491958618164 sec
Batchsize: 1
Time spent per BATCH:     6.7463 ms
Total samples/sec:   148.2291 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.9564 ms
Total samples/sec:   143.7533 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.9343 ms
Total samples/sec:   144.2103 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.9127 ms
Total samples/sec:   144.6619 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.8941 ms
Total samples/sec:   145.0515 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.8736 ms
Total samples/sec:   145.4841 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.8538 ms
Total samples/sec:   145.9048 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.8370 ms
Total samples/sec:   146.2620 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.8192 ms
Total samples/sec:   146.6455 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.8409 ms
Total samples/sec:   146.1796 samples/s
Total labeled samples: 1 person
steps = 69, 0.005454063415527344 sec
Batchsize: 1
Time spent per BATCH:     6.8211 ms
Total samples/sec:   146.6042 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.8026 ms
Total samples/sec:   147.0022 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.7848 ms
Total samples/sec:   147.3872 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.7696 ms
Total samples/sec:   147.7183 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.7532 ms
Total samples/sec:   148.0780 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.7381 ms
Total samples/sec:   148.4108 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.7227 ms
Total samples/sec:   148.7493 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.7073 ms
Total samples/sec:   149.0904 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.6926 ms
Total samples/sec:   149.4196 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.7254 ms
Total samples/sec:   148.6903 samples/s
Total labeled samples: 1 person
steps = 79, 0.005575895309448242 sec
Batchsize: 1
Time spent per BATCH:     6.7110 ms
Total samples/sec:   149.0086 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.6971 ms
Total samples/sec:   149.3188 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.6838 ms
Total samples/sec:   149.6166 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.6695 ms
Total samples/sec:   149.9357 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.6564 ms
Total samples/sec:   150.2316 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.6459 ms
Total samples/sec:   150.4689 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.6334 ms
Total samples/sec:   150.7518 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.6217 ms
Total samples/sec:   151.0197 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.6096 ms
Total samples/sec:   151.2955 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.5988 ms
Total samples/sec:   151.5433 samples/s
Total labeled samples: 1 person
steps = 89, 0.005622386932373047 sec
Batchsize: 1
Time spent per BATCH:     6.5879 ms
Total samples/sec:   151.7928 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.5772 ms
Total samples/sec:   152.0415 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.5662 ms
Total samples/sec:   152.2944 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.5556 ms
Total samples/sec:   152.5423 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.5456 ms
Total samples/sec:   152.7739 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.5366 ms
Total samples/sec:   152.9857 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.5266 ms
Total samples/sec:   153.2203 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.5169 ms
Total samples/sec:   153.4466 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.5071 ms
Total samples/sec:   153.6782 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4975 ms
Total samples/sec:   153.9046 samples/s
Total labeled samples: 1 person
steps = 99, 0.005602359771728516 sec
Batchsize: 1
Time spent per BATCH:     6.4886 ms
Total samples/sec:   154.1170 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4997 ms
Total samples/sec:   153.8533 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4911 ms
Total samples/sec:   154.0572 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4824 ms
Total samples/sec:   154.2637 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4737 ms
Total samples/sec:   154.4711 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4651 ms
Total samples/sec:   154.6773 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4892 ms
Total samples/sec:   154.1033 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4805 ms
Total samples/sec:   154.3101 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4719 ms
Total samples/sec:   154.5149 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4633 ms
Total samples/sec:   154.7196 samples/s
Total labeled samples: 1 person
steps = 109, 0.005563020706176758 sec
Batchsize: 1
Time spent per BATCH:     6.4551 ms
Total samples/sec:   154.9158 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4475 ms
Total samples/sec:   155.0980 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4394 ms
Total samples/sec:   155.2938 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4317 ms
Total samples/sec:   155.4799 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4242 ms
Total samples/sec:   155.6611 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4169 ms
Total samples/sec:   155.8391 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4098 ms
Total samples/sec:   156.0114 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4023 ms
Total samples/sec:   156.1936 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3952 ms
Total samples/sec:   156.3684 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3885 ms
Total samples/sec:   156.5324 samples/s
Total labeled samples: 1 person
steps = 119, 0.005595207214355469 sec
Batchsize: 1
Time spent per BATCH:     6.3818 ms
Total samples/sec:   156.6945 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3756 ms
Total samples/sec:   156.8480 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3691 ms
Total samples/sec:   157.0079 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3625 ms
Total samples/sec:   157.1719 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3559 ms
Total samples/sec:   157.3334 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4868 ms
Total samples/sec:   154.1603 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4794 ms
Total samples/sec:   154.3358 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4724 ms
Total samples/sec:   154.5028 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4656 ms
Total samples/sec:   154.6649 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4828 ms
Total samples/sec:   154.2545 samples/s
Total labeled samples: 1 person
steps = 129, 0.005583047866821289 sec
Batchsize: 1
Time spent per BATCH:     6.4759 ms
Total samples/sec:   154.4194 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4679 ms
Total samples/sec:   154.6089 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4606 ms
Total samples/sec:   154.7839 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4542 ms
Total samples/sec:   154.9389 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4474 ms
Total samples/sec:   155.1006 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4406 ms
Total samples/sec:   155.2653 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4345 ms
Total samples/sec:   155.4129 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4281 ms
Total samples/sec:   155.5660 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4221 ms
Total samples/sec:   155.7133 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4158 ms
Total samples/sec:   155.8661 samples/s
Total labeled samples: 1 person
steps = 139, 0.007254838943481445 sec
Batchsize: 1
Time spent per BATCH:     6.4218 ms
Total samples/sec:   155.7207 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4161 ms
Total samples/sec:   155.8587 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4101 ms
Total samples/sec:   156.0031 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.4047 ms
Total samples/sec:   156.1363 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3991 ms
Total samples/sec:   156.2719 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3937 ms
Total samples/sec:   156.4033 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3879 ms
Total samples/sec:   156.5458 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3825 ms
Total samples/sec:   156.6782 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3771 ms
Total samples/sec:   156.8111 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3717 ms
Total samples/sec:   156.9447 samples/s
Total labeled samples: 1 person
steps = 149, 0.0056934356689453125 sec
Batchsize: 1
Time spent per BATCH:     6.3672 ms
Total samples/sec:   157.0561 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3617 ms
Total samples/sec:   157.1914 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3565 ms
Total samples/sec:   157.3196 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3516 ms
Total samples/sec:   157.4397 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3626 ms
Total samples/sec:   157.1682 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3573 ms
Total samples/sec:   157.2986 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3521 ms
Total samples/sec:   157.4293 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3473 ms
Total samples/sec:   157.5462 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3425 ms
Total samples/sec:   157.6673 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3374 ms
Total samples/sec:   157.7928 samples/s
Total labeled samples: 1 person
steps = 159, 0.005584716796875 sec
Batchsize: 1
Time spent per BATCH:     6.3327 ms
Total samples/sec:   157.9100 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3285 ms
Total samples/sec:   158.0148 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3237 ms
Total samples/sec:   158.1356 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3187 ms
Total samples/sec:   158.2592 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3144 ms
Total samples/sec:   158.3673 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3097 ms
Total samples/sec:   158.4871 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3054 ms
Total samples/sec:   158.5936 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.3008 ms
Total samples/sec:   158.7103 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2966 ms
Total samples/sec:   158.8166 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2924 ms
Total samples/sec:   158.9228 samples/s
Total labeled samples: 1 person
steps = 169, 0.005587577819824219 sec
Batchsize: 1
Time spent per BATCH:     6.2882 ms
Total samples/sec:   159.0276 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2842 ms
Total samples/sec:   159.1291 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2801 ms
Total samples/sec:   159.2329 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2939 ms
Total samples/sec:   158.8849 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2900 ms
Total samples/sec:   158.9830 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2859 ms
Total samples/sec:   159.0851 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2818 ms
Total samples/sec:   159.1906 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2779 ms
Total samples/sec:   159.2892 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2739 ms
Total samples/sec:   159.3904 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2699 ms
Total samples/sec:   159.4927 samples/s
Total labeled samples: 1 person
steps = 179, 0.0055501461029052734 sec
Batchsize: 1
Time spent per BATCH:     6.2659 ms
Total samples/sec:   159.5945 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2622 ms
Total samples/sec:   159.6894 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2582 ms
Total samples/sec:   159.7901 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2549 ms
Total samples/sec:   159.8752 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2510 ms
Total samples/sec:   159.9735 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2623 ms
Total samples/sec:   159.6848 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2585 ms
Total samples/sec:   159.7826 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2549 ms
Total samples/sec:   159.8749 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2517 ms
Total samples/sec:   159.9558 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2482 ms
Total samples/sec:   160.0466 samples/s
Total labeled samples: 1 person
steps = 189, 0.008138656616210938 sec
Batchsize: 1
Time spent per BATCH:     6.2581 ms
Total samples/sec:   159.7921 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2544 ms
Total samples/sec:   159.8869 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2509 ms
Total samples/sec:   159.9767 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2472 ms
Total samples/sec:   160.0728 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2437 ms
Total samples/sec:   160.1606 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2401 ms
Total samples/sec:   160.2527 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2370 ms
Total samples/sec:   160.3346 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2334 ms
Total samples/sec:   160.4256 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2301 ms
Total samples/sec:   160.5116 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2267 ms
Total samples/sec:   160.5981 samples/s
Total labeled samples: 1 person
steps = 199, 0.0055425167083740234 sec
Batchsize: 1
Time spent per BATCH:     6.2233 ms
Total samples/sec:   160.6864 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2201 ms
Total samples/sec:   160.7697 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2170 ms
Total samples/sec:   160.8502 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2295 ms
Total samples/sec:   160.5271 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2262 ms
Total samples/sec:   160.6112 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2240 ms
Total samples/sec:   160.6689 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2367 ms
Total samples/sec:   160.3421 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2335 ms
Total samples/sec:   160.4225 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2303 ms
Total samples/sec:   160.5064 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2403 ms
Total samples/sec:   160.2477 samples/s
Total labeled samples: 1 person
steps = 209, 0.005559206008911133 sec
Batchsize: 1
Time spent per BATCH:     6.2371 ms
Total samples/sec:   160.3310 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2341 ms
Total samples/sec:   160.4091 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2310 ms
Total samples/sec:   160.4886 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2279 ms
Total samples/sec:   160.5669 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2247 ms
Total samples/sec:   160.6494 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2217 ms
Total samples/sec:   160.7266 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2187 ms
Total samples/sec:   160.8043 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2161 ms
Total samples/sec:   160.8728 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2131 ms
Total samples/sec:   160.9515 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2103 ms
Total samples/sec:   161.0236 samples/s
Total labeled samples: 1 person
steps = 219, 0.005566835403442383 sec
Batchsize: 1
Time spent per BATCH:     6.2073 ms
Total samples/sec:   161.0995 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2044 ms
Total samples/sec:   161.1766 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2014 ms
Total samples/sec:   161.2534 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1985 ms
Total samples/sec:   161.3297 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1954 ms
Total samples/sec:   161.4092 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1927 ms
Total samples/sec:   161.4799 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1900 ms
Total samples/sec:   161.5518 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1873 ms
Total samples/sec:   161.6203 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1936 ms
Total samples/sec:   161.4563 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1909 ms
Total samples/sec:   161.5286 samples/s
Total labeled samples: 1 person
steps = 229, 0.005564212799072266 sec
Batchsize: 1
Time spent per BATCH:     6.1881 ms
Total samples/sec:   161.5997 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1856 ms
Total samples/sec:   161.6660 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1829 ms
Total samples/sec:   161.7376 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1803 ms
Total samples/sec:   161.8043 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1910 ms
Total samples/sec:   161.5241 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1883 ms
Total samples/sec:   161.5953 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1856 ms
Total samples/sec:   161.6647 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1831 ms
Total samples/sec:   161.7302 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1805 ms
Total samples/sec:   161.7984 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1879 ms
Total samples/sec:   161.6045 samples/s
Total labeled samples: 1 person
steps = 239, 0.005574941635131836 sec
Batchsize: 1
Time spent per BATCH:     6.1854 ms
Total samples/sec:   161.6712 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1828 ms
Total samples/sec:   161.7379 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1803 ms
Total samples/sec:   161.8052 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1779 ms
Total samples/sec:   161.8676 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1755 ms
Total samples/sec:   161.9311 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1729 ms
Total samples/sec:   161.9973 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1707 ms
Total samples/sec:   162.0573 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1683 ms
Total samples/sec:   162.1195 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1659 ms
Total samples/sec:   162.1832 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1634 ms
Total samples/sec:   162.2491 samples/s
Total labeled samples: 1 person
steps = 249, 0.0055620670318603516 sec
Batchsize: 1
Time spent per BATCH:     6.1610 ms
Total samples/sec:   162.3124 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1587 ms
Total samples/sec:   162.3719 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1562 ms
Total samples/sec:   162.4377 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2843 ms
Total samples/sec:   159.1264 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2814 ms
Total samples/sec:   159.2002 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2787 ms
Total samples/sec:   159.2687 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2759 ms
Total samples/sec:   159.3394 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2733 ms
Total samples/sec:   159.4069 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2706 ms
Total samples/sec:   159.4753 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2676 ms
Total samples/sec:   159.5517 samples/s
Total labeled samples: 1 person
steps = 259, 0.005545854568481445 sec
Batchsize: 1
Time spent per BATCH:     6.2648 ms
Total samples/sec:   159.6224 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2698 ms
Total samples/sec:   159.4946 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2673 ms
Total samples/sec:   159.5586 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2644 ms
Total samples/sec:   159.6322 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2617 ms
Total samples/sec:   159.7001 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2588 ms
Total samples/sec:   159.7740 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2562 ms
Total samples/sec:   159.8405 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2537 ms
Total samples/sec:   159.9054 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2510 ms
Total samples/sec:   159.9738 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2609 ms
Total samples/sec:   159.7207 samples/s
Total labeled samples: 1 person
steps = 269, 0.005518436431884766 sec
Batchsize: 1
Time spent per BATCH:     6.2582 ms
Total samples/sec:   159.7909 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2554 ms
Total samples/sec:   159.8607 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2528 ms
Total samples/sec:   159.9276 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2504 ms
Total samples/sec:   159.9909 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2477 ms
Total samples/sec:   160.0582 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2539 ms
Total samples/sec:   159.9001 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2619 ms
Total samples/sec:   159.6963 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2594 ms
Total samples/sec:   159.7608 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2568 ms
Total samples/sec:   159.8258 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2544 ms
Total samples/sec:   159.8871 samples/s
Total labeled samples: 1 person
steps = 279, 0.005591630935668945 sec
Batchsize: 1
Time spent per BATCH:     6.2520 ms
Total samples/sec:   159.9476 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2495 ms
Total samples/sec:   160.0119 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2470 ms
Total samples/sec:   160.0768 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2445 ms
Total samples/sec:   160.1420 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2419 ms
Total samples/sec:   160.2068 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2394 ms
Total samples/sec:   160.2715 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2370 ms
Total samples/sec:   160.3333 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2348 ms
Total samples/sec:   160.3892 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2327 ms
Total samples/sec:   160.4454 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2299 ms
Total samples/sec:   160.5152 samples/s
Total labeled samples: 1 person
steps = 289, 0.0057108402252197266 sec
Batchsize: 1
Time spent per BATCH:     6.2282 ms
Total samples/sec:   160.5613 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2257 ms
Total samples/sec:   160.6251 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2234 ms
Total samples/sec:   160.6838 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2209 ms
Total samples/sec:   160.7484 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2276 ms
Total samples/sec:   160.5764 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2252 ms
Total samples/sec:   160.6364 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2226 ms
Total samples/sec:   160.7055 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2200 ms
Total samples/sec:   160.7711 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2176 ms
Total samples/sec:   160.8339 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2149 ms
Total samples/sec:   160.9029 samples/s
Total labeled samples: 1 person
steps = 299, 0.005470752716064453 sec
Batchsize: 1
Time spent per BATCH:     6.2124 ms
Total samples/sec:   160.9671 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2101 ms
Total samples/sec:   161.0283 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2078 ms
Total samples/sec:   161.0876 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2128 ms
Total samples/sec:   160.9580 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2105 ms
Total samples/sec:   161.0170 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2082 ms
Total samples/sec:   161.0774 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2059 ms
Total samples/sec:   161.1364 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2034 ms
Total samples/sec:   161.2015 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.2010 ms
Total samples/sec:   161.2634 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1988 ms
Total samples/sec:   161.3211 samples/s
Total labeled samples: 1 person
steps = 309, 0.0054285526275634766 sec
Batchsize: 1
Time spent per BATCH:     6.1963 ms
Total samples/sec:   161.3858 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1942 ms
Total samples/sec:   161.4414 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1918 ms
Total samples/sec:   161.5027 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1894 ms
Total samples/sec:   161.5654 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1937 ms
Total samples/sec:   161.4540 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1918 ms
Total samples/sec:   161.5051 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1896 ms
Total samples/sec:   161.5602 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1875 ms
Total samples/sec:   161.6157 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1855 ms
Total samples/sec:   161.6689 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1832 ms
Total samples/sec:   161.7287 samples/s
Total labeled samples: 1 person
steps = 319, 0.005435943603515625 sec
Batchsize: 1
Time spent per BATCH:     6.1809 ms
Total samples/sec:   161.7898 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1785 ms
Total samples/sec:   161.8511 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1765 ms
Total samples/sec:   161.9052 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1742 ms
Total samples/sec:   161.9640 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1720 ms
Total samples/sec:   162.0233 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1697 ms
Total samples/sec:   162.0821 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1674 ms
Total samples/sec:   162.1423 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1651 ms
Total samples/sec:   162.2021 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1630 ms
Total samples/sec:   162.2600 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1608 ms
Total samples/sec:   162.3174 samples/s
Total labeled samples: 1 person
steps = 329, 0.005449056625366211 sec
Batchsize: 1
Time spent per BATCH:     6.1586 ms
Total samples/sec:   162.3742 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1564 ms
Total samples/sec:   162.4337 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1541 ms
Total samples/sec:   162.4921 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1519 ms
Total samples/sec:   162.5514 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1498 ms
Total samples/sec:   162.6079 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1476 ms
Total samples/sec:   162.6659 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1535 ms
Total samples/sec:   162.5089 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1515 ms
Total samples/sec:   162.5631 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1495 ms
Total samples/sec:   162.6151 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1473 ms
Total samples/sec:   162.6730 samples/s
Total labeled samples: 1 person
steps = 339, 0.005444049835205078 sec
Batchsize: 1
Time spent per BATCH:     6.1452 ms
Total samples/sec:   162.7278 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1431 ms
Total samples/sec:   162.7844 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1412 ms
Total samples/sec:   162.8346 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1393 ms
Total samples/sec:   162.8856 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1373 ms
Total samples/sec:   162.9392 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1352 ms
Total samples/sec:   162.9927 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1332 ms
Total samples/sec:   163.0464 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1312 ms
Total samples/sec:   163.1000 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1291 ms
Total samples/sec:   163.1563 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1270 ms
Total samples/sec:   163.2121 samples/s
Total labeled samples: 1 person
steps = 349, 0.005476713180541992 sec
Batchsize: 1
Time spent per BATCH:     6.1251 ms
Total samples/sec:   163.2616 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1264 ms
Total samples/sec:   163.2275 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1325 ms
Total samples/sec:   163.0645 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1304 ms
Total samples/sec:   163.1205 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1284 ms
Total samples/sec:   163.1738 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1265 ms
Total samples/sec:   163.2261 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1326 ms
Total samples/sec:   163.0627 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1307 ms
Total samples/sec:   163.1131 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1287 ms
Total samples/sec:   163.1671 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1268 ms
Total samples/sec:   163.2180 samples/s
Total labeled samples: 1 person
steps = 359, 0.005448818206787109 sec
Batchsize: 1
Time spent per BATCH:     6.1249 ms
Total samples/sec:   163.2682 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1230 ms
Total samples/sec:   163.3185 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1210 ms
Total samples/sec:   163.3723 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1191 ms
Total samples/sec:   163.4216 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1174 ms
Total samples/sec:   163.4671 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1155 ms
Total samples/sec:   163.5191 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1136 ms
Total samples/sec:   163.5693 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1118 ms
Total samples/sec:   163.6190 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1100 ms
Total samples/sec:   163.6664 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1083 ms
Total samples/sec:   163.7125 samples/s
Total labeled samples: 1 person
steps = 369, 0.005417585372924805 sec
Batchsize: 1
Time spent per BATCH:     6.1064 ms
Total samples/sec:   163.7626 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1045 ms
Total samples/sec:   163.8126 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1027 ms
Total samples/sec:   163.8625 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.1010 ms
Total samples/sec:   163.9075 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0991 ms
Total samples/sec:   163.9576 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0974 ms
Total samples/sec:   164.0043 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0957 ms
Total samples/sec:   164.0497 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0939 ms
Total samples/sec:   164.0989 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0921 ms
Total samples/sec:   164.1464 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0905 ms
Total samples/sec:   164.1897 samples/s
Total labeled samples: 1 person
steps = 379, 0.005451679229736328 sec
Batchsize: 1
Time spent per BATCH:     6.0888 ms
Total samples/sec:   164.2351 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0871 ms
Total samples/sec:   164.2825 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0854 ms
Total samples/sec:   164.3269 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0904 ms
Total samples/sec:   164.1921 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0887 ms
Total samples/sec:   164.2381 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0957 ms
Total samples/sec:   164.0505 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0940 ms
Total samples/sec:   164.0969 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0922 ms
Total samples/sec:   164.1443 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0905 ms
Total samples/sec:   164.1891 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0888 ms
Total samples/sec:   164.2357 samples/s
Total labeled samples: 1 person
steps = 389, 0.005393505096435547 sec
Batchsize: 1
Time spent per BATCH:     6.0870 ms
Total samples/sec:   164.2838 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0853 ms
Total samples/sec:   164.3302 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0838 ms
Total samples/sec:   164.3707 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0822 ms
Total samples/sec:   164.4147 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0806 ms
Total samples/sec:   164.4569 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0793 ms
Total samples/sec:   164.4928 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0777 ms
Total samples/sec:   164.5349 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0812 ms
Total samples/sec:   164.4408 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0796 ms
Total samples/sec:   164.4837 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0780 ms
Total samples/sec:   164.5275 samples/s
Total labeled samples: 1 person
steps = 399, 0.005461931228637695 sec
Batchsize: 1
Time spent per BATCH:     6.0765 ms
Total samples/sec:   164.5692 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0748 ms
Total samples/sec:   164.6132 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0732 ms
Total samples/sec:   164.6580 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0716 ms
Total samples/sec:   164.7014 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0762 ms
Total samples/sec:   164.5763 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0745 ms
Total samples/sec:   164.6218 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0729 ms
Total samples/sec:   164.6661 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0712 ms
Total samples/sec:   164.7116 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0697 ms
Total samples/sec:   164.7536 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0681 ms
Total samples/sec:   164.7963 samples/s
Total labeled samples: 1 person
steps = 409, 0.005417823791503906 sec
Batchsize: 1
Time spent per BATCH:     6.0665 ms
Total samples/sec:   164.8394 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0649 ms
Total samples/sec:   164.8827 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0633 ms
Total samples/sec:   164.9266 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0622 ms
Total samples/sec:   164.9569 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0606 ms
Total samples/sec:   164.9992 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0592 ms
Total samples/sec:   165.0370 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0577 ms
Total samples/sec:   165.0786 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0563 ms
Total samples/sec:   165.1160 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0549 ms
Total samples/sec:   165.1568 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0533 ms
Total samples/sec:   165.1978 samples/s
Total labeled samples: 1 person
steps = 419, 0.005434751510620117 sec
Batchsize: 1
Time spent per BATCH:     6.0519 ms
Total samples/sec:   165.2380 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0504 ms
Total samples/sec:   165.2778 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0490 ms
Total samples/sec:   165.3171 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0476 ms
Total samples/sec:   165.3551 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0460 ms
Total samples/sec:   165.3976 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0446 ms
Total samples/sec:   165.4378 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0431 ms
Total samples/sec:   165.4774 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0417 ms
Total samples/sec:   165.5164 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0402 ms
Total samples/sec:   165.5569 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0389 ms
Total samples/sec:   165.5941 samples/s
Total labeled samples: 1 person
steps = 429, 0.0054318904876708984 sec
Batchsize: 1
Time spent per BATCH:     6.0375 ms
Total samples/sec:   165.6328 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0361 ms
Total samples/sec:   165.6694 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0347 ms
Total samples/sec:   165.7083 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0334 ms
Total samples/sec:   165.7447 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0320 ms
Total samples/sec:   165.7822 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0315 ms
Total samples/sec:   165.7960 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0300 ms
Total samples/sec:   165.8365 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0286 ms
Total samples/sec:   165.8768 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0273 ms
Total samples/sec:   165.9131 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0258 ms
Total samples/sec:   165.9519 samples/s
Total labeled samples: 1 person
steps = 439, 0.00552821159362793 sec
Batchsize: 1
Time spent per BATCH:     6.0247 ms
Total samples/sec:   165.9830 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0234 ms
Total samples/sec:   166.0198 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0220 ms
Total samples/sec:   166.0570 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0207 ms
Total samples/sec:   166.0934 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0193 ms
Total samples/sec:   166.1313 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0180 ms
Total samples/sec:   166.1669 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0167 ms
Total samples/sec:   166.2033 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0154 ms
Total samples/sec:   166.2397 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0140 ms
Total samples/sec:   166.2785 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0128 ms
Total samples/sec:   166.3130 samples/s
Total labeled samples: 1 person
steps = 449, 0.005473136901855469 sec
Batchsize: 1
Time spent per BATCH:     6.0116 ms
Total samples/sec:   166.3462 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0103 ms
Total samples/sec:   166.3823 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0089 ms
Total samples/sec:   166.4194 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0076 ms
Total samples/sec:   166.4548 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0064 ms
Total samples/sec:   166.4885 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0051 ms
Total samples/sec:   166.5237 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0039 ms
Total samples/sec:   166.5580 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0027 ms
Total samples/sec:   166.5913 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0014 ms
Total samples/sec:   166.6275 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0001 ms
Total samples/sec:   166.6636 samples/s
Total labeled samples: 1 person
steps = 459, 0.005422353744506836 sec
Batchsize: 1
Time spent per BATCH:     5.9989 ms
Total samples/sec:   166.6985 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9976 ms
Total samples/sec:   166.7333 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9963 ms
Total samples/sec:   166.7688 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9951 ms
Total samples/sec:   166.8027 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9938 ms
Total samples/sec:   166.8383 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9970 ms
Total samples/sec:   166.7508 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9959 ms
Total samples/sec:   166.7820 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9947 ms
Total samples/sec:   166.8128 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9935 ms
Total samples/sec:   166.8469 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9961 ms
Total samples/sec:   166.7753 samples/s
Total labeled samples: 1 person
steps = 469, 0.00541996955871582 sec
Batchsize: 1
Time spent per BATCH:     5.9949 ms
Total samples/sec:   166.8094 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9936 ms
Total samples/sec:   166.8435 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9925 ms
Total samples/sec:   166.8764 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9912 ms
Total samples/sec:   166.9110 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9900 ms
Total samples/sec:   166.9437 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9889 ms
Total samples/sec:   166.9745 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9877 ms
Total samples/sec:   167.0098 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9864 ms
Total samples/sec:   167.0441 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9853 ms
Total samples/sec:   167.0759 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9842 ms
Total samples/sec:   167.1077 samples/s
Total labeled samples: 1 person
steps = 479, 0.005464792251586914 sec
Batchsize: 1
Time spent per BATCH:     5.9831 ms
Total samples/sec:   167.1380 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9820 ms
Total samples/sec:   167.1687 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9808 ms
Total samples/sec:   167.2021 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9797 ms
Total samples/sec:   167.2337 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9786 ms
Total samples/sec:   167.2641 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9776 ms
Total samples/sec:   167.2915 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9766 ms
Total samples/sec:   167.3198 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9756 ms
Total samples/sec:   167.3466 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9746 ms
Total samples/sec:   167.3760 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9736 ms
Total samples/sec:   167.4044 samples/s
Total labeled samples: 1 person
steps = 489, 0.007498025894165039 sec
Batchsize: 1
Time spent per BATCH:     5.9767 ms
Total samples/sec:   167.3172 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9757 ms
Total samples/sec:   167.3457 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9745 ms
Total samples/sec:   167.3785 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9734 ms
Total samples/sec:   167.4075 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9724 ms
Total samples/sec:   167.4379 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9712 ms
Total samples/sec:   167.4695 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9757 ms
Total samples/sec:   167.3445 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9746 ms
Total samples/sec:   167.3758 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9735 ms
Total samples/sec:   167.4047 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9725 ms
Total samples/sec:   167.4333 samples/s
Total labeled samples: 1 person
steps = 499, 0.005456686019897461 sec
Batchsize: 1
Time spent per BATCH:     5.9715 ms
Total samples/sec:   167.4623 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9704 ms
Total samples/sec:   167.4919 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9694 ms
Total samples/sec:   167.5222 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9741 ms
Total samples/sec:   167.3886 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9771 ms
Total samples/sec:   167.3044 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9764 ms
Total samples/sec:   167.3244 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9754 ms
Total samples/sec:   167.3525 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9744 ms
Total samples/sec:   167.3808 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9733 ms
Total samples/sec:   167.4117 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9984 ms
Total samples/sec:   166.7115 samples/s
Total labeled samples: 1 person
steps = 509, 0.0054318904876708984 sec
Batchsize: 1
Time spent per BATCH:     5.9973 ms
Total samples/sec:   166.7424 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0019 ms
Total samples/sec:   166.6130 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     6.0008 ms
Total samples/sec:   166.6443 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9997 ms
Total samples/sec:   166.6748 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9985 ms
Total samples/sec:   166.7073 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9974 ms
Total samples/sec:   166.7400 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9963 ms
Total samples/sec:   166.7696 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9952 ms
Total samples/sec:   166.8008 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9942 ms
Total samples/sec:   166.8292 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9931 ms
Total samples/sec:   166.8581 samples/s
Total labeled samples: 1 person
steps = 519, 0.005483388900756836 sec
Batchsize: 1
Time spent per BATCH:     5.9921 ms
Total samples/sec:   166.8854 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9910 ms
Total samples/sec:   166.9161 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9900 ms
Total samples/sec:   166.9439 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9892 ms
Total samples/sec:   166.9676 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9881 ms
Total samples/sec:   166.9975 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9871 ms
Total samples/sec:   167.0269 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9860 ms
Total samples/sec:   167.0554 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9850 ms
Total samples/sec:   167.0847 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9840 ms
Total samples/sec:   167.1116 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9829 ms
Total samples/sec:   167.1418 samples/s
Total labeled samples: 1 person
steps = 529, 0.0054357051849365234 sec
Batchsize: 1
Time spent per BATCH:     5.9819 ms
Total samples/sec:   167.1706 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9808 ms
Total samples/sec:   167.2008 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9801 ms
Total samples/sec:   167.2214 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9791 ms
Total samples/sec:   167.2500 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9781 ms
Total samples/sec:   167.2783 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9771 ms
Total samples/sec:   167.3047 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9761 ms
Total samples/sec:   167.3325 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9751 ms
Total samples/sec:   167.3605 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9792 ms
Total samples/sec:   167.2472 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9782 ms
Total samples/sec:   167.2758 samples/s
Total labeled samples: 1 person
steps = 539, 0.005416154861450195 sec
Batchsize: 1
Time spent per BATCH:     5.9771 ms
Total samples/sec:   167.3049 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9762 ms
Total samples/sec:   167.3294 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9753 ms
Total samples/sec:   167.3557 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9743 ms
Total samples/sec:   167.3837 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9732 ms
Total samples/sec:   167.4134 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9723 ms
Total samples/sec:   167.4406 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9713 ms
Total samples/sec:   167.4680 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9704 ms
Total samples/sec:   167.4938 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9694 ms
Total samples/sec:   167.5204 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9684 ms
Total samples/sec:   167.5486 samples/s
Total labeled samples: 1 person
steps = 549, 0.005379676818847656 sec
Batchsize: 1
Time spent per BATCH:     5.9673 ms
Total samples/sec:   167.5787 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9700 ms
Total samples/sec:   167.5045 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9695 ms
Total samples/sec:   167.5187 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9685 ms
Total samples/sec:   167.5467 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9675 ms
Total samples/sec:   167.5740 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9666 ms
Total samples/sec:   167.6010 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9656 ms
Total samples/sec:   167.6264 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9647 ms
Total samples/sec:   167.6544 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9638 ms
Total samples/sec:   167.6787 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9628 ms
Total samples/sec:   167.7073 samples/s
Total labeled samples: 1 person
steps = 559, 0.00567317008972168 sec
Batchsize: 1
Time spent per BATCH:     5.9623 ms
Total samples/sec:   167.7218 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9614 ms
Total samples/sec:   167.7450 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9605 ms
Total samples/sec:   167.7700 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9597 ms
Total samples/sec:   167.7946 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9636 ms
Total samples/sec:   167.6846 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9626 ms
Total samples/sec:   167.7122 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9618 ms
Total samples/sec:   167.7336 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9609 ms
Total samples/sec:   167.7595 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9600 ms
Total samples/sec:   167.7862 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9595 ms
Total samples/sec:   167.7981 samples/s
Total labeled samples: 1 person
steps = 569, 0.005494356155395508 sec
Batchsize: 1
Time spent per BATCH:     5.9587 ms
Total samples/sec:   167.8210 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9581 ms
Total samples/sec:   167.8386 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9572 ms
Total samples/sec:   167.8629 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9566 ms
Total samples/sec:   167.8820 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9560 ms
Total samples/sec:   167.8990 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9553 ms
Total samples/sec:   167.9165 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9549 ms
Total samples/sec:   167.9291 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9542 ms
Total samples/sec:   167.9479 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9536 ms
Total samples/sec:   167.9656 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9527 ms
Total samples/sec:   167.9911 samples/s
Total labeled samples: 1 person
steps = 579, 0.005501270294189453 sec
Batchsize: 1
Time spent per BATCH:     5.9519 ms
Total samples/sec:   168.0131 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9511 ms
Total samples/sec:   168.0361 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9504 ms
Total samples/sec:   168.0554 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9496 ms
Total samples/sec:   168.0780 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9488 ms
Total samples/sec:   168.1016 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9524 ms
Total samples/sec:   167.9984 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9517 ms
Total samples/sec:   168.0205 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9508 ms
Total samples/sec:   168.0444 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9503 ms
Total samples/sec:   168.0585 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9495 ms
Total samples/sec:   168.0826 samples/s
Total labeled samples: 1 person
steps = 589, 0.005631208419799805 sec
Batchsize: 1
Time spent per BATCH:     5.9489 ms
Total samples/sec:   168.0978 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9485 ms
Total samples/sec:   168.1098 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9479 ms
Total samples/sec:   168.1260 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9470 ms
Total samples/sec:   168.1508 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9463 ms
Total samples/sec:   168.1720 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9457 ms
Total samples/sec:   168.1888 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9452 ms
Total samples/sec:   168.2031 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9444 ms
Total samples/sec:   168.2262 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9435 ms
Total samples/sec:   168.2502 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9429 ms
Total samples/sec:   168.2687 samples/s
Total labeled samples: 1 person
steps = 599, 0.0055255889892578125 sec
Batchsize: 1
Time spent per BATCH:     5.9422 ms
Total samples/sec:   168.2884 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9414 ms
Total samples/sec:   168.3098 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9406 ms
Total samples/sec:   168.3338 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9398 ms
Total samples/sec:   168.3567 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9390 ms
Total samples/sec:   168.3796 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9382 ms
Total samples/sec:   168.4014 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9374 ms
Total samples/sec:   168.4227 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9367 ms
Total samples/sec:   168.4440 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9359 ms
Total samples/sec:   168.4665 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9352 ms
Total samples/sec:   168.4875 samples/s
Total labeled samples: 1 person
steps = 609, 0.0055577754974365234 sec
Batchsize: 1
Time spent per BATCH:     5.9345 ms
Total samples/sec:   168.5051 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9337 ms
Total samples/sec:   168.5287 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9330 ms
Total samples/sec:   168.5489 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9323 ms
Total samples/sec:   168.5675 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9317 ms
Total samples/sec:   168.5867 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9312 ms
Total samples/sec:   168.6003 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9305 ms
Total samples/sec:   168.6207 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9297 ms
Total samples/sec:   168.6423 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9290 ms
Total samples/sec:   168.6639 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9283 ms
Total samples/sec:   168.6835 samples/s
Total labeled samples: 1 person
steps = 619, 0.0054700374603271484 sec
Batchsize: 1
Time spent per BATCH:     5.9275 ms
Total samples/sec:   168.7046 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9268 ms
Total samples/sec:   168.7261 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9261 ms
Total samples/sec:   168.7453 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9255 ms
Total samples/sec:   168.7635 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9247 ms
Total samples/sec:   168.7861 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9241 ms
Total samples/sec:   168.8032 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9233 ms
Total samples/sec:   168.8245 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9226 ms
Total samples/sec:   168.8453 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9218 ms
Total samples/sec:   168.8665 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9211 ms
Total samples/sec:   168.8870 samples/s
Total labeled samples: 1 person
steps = 629, 0.005445957183837891 sec
Batchsize: 1
Time spent per BATCH:     5.9204 ms
Total samples/sec:   168.9086 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9196 ms
Total samples/sec:   168.9317 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9188 ms
Total samples/sec:   168.9546 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9181 ms
Total samples/sec:   168.9741 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9173 ms
Total samples/sec:   168.9970 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9165 ms
Total samples/sec:   169.0178 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9159 ms
Total samples/sec:   169.0355 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9152 ms
Total samples/sec:   169.0549 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9152 ms
Total samples/sec:   169.0568 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9147 ms
Total samples/sec:   169.0699 samples/s
Total labeled samples: 1 person
steps = 639, 0.005759239196777344 sec
Batchsize: 1
Time spent per BATCH:     5.9145 ms
Total samples/sec:   169.0768 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9138 ms
Total samples/sec:   169.0968 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9133 ms
Total samples/sec:   169.1089 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9127 ms
Total samples/sec:   169.1284 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9121 ms
Total samples/sec:   169.1440 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9119 ms
Total samples/sec:   169.1511 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9112 ms
Total samples/sec:   169.1690 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9107 ms
Total samples/sec:   169.1856 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9134 ms
Total samples/sec:   169.1079 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9126 ms
Total samples/sec:   169.1303 samples/s
Total labeled samples: 1 person
steps = 649, 0.005616426467895508 sec
Batchsize: 1
Time spent per BATCH:     5.9121 ms
Total samples/sec:   169.1434 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9119 ms
Total samples/sec:   169.1500 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9159 ms
Total samples/sec:   169.0366 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9193 ms
Total samples/sec:   168.9393 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9186 ms
Total samples/sec:   168.9583 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9182 ms
Total samples/sec:   168.9699 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9209 ms
Total samples/sec:   168.8935 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9204 ms
Total samples/sec:   168.9070 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9243 ms
Total samples/sec:   168.7958 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9266 ms
Total samples/sec:   168.7307 samples/s
Total labeled samples: 1 person
steps = 659, 0.005482196807861328 sec
Batchsize: 1
Time spent per BATCH:     5.9259 ms
Total samples/sec:   168.7499 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9252 ms
Total samples/sec:   168.7693 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9245 ms
Total samples/sec:   168.7899 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9239 ms
Total samples/sec:   168.8082 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9234 ms
Total samples/sec:   168.8229 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9227 ms
Total samples/sec:   168.8409 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9224 ms
Total samples/sec:   168.8516 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9217 ms
Total samples/sec:   168.8695 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9211 ms
Total samples/sec:   168.8863 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9205 ms
Total samples/sec:   168.9055 samples/s
Total labeled samples: 1 person
steps = 669, 0.00552821159362793 sec
Batchsize: 1
Time spent per BATCH:     5.9199 ms
Total samples/sec:   168.9223 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9197 ms
Total samples/sec:   168.9286 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9234 ms
Total samples/sec:   168.8209 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9229 ms
Total samples/sec:   168.8355 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9222 ms
Total samples/sec:   168.8548 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9215 ms
Total samples/sec:   168.8759 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9238 ms
Total samples/sec:   168.8115 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9231 ms
Total samples/sec:   168.8302 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9224 ms
Total samples/sec:   168.8507 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9218 ms
Total samples/sec:   168.8671 samples/s
Total labeled samples: 1 person
steps = 679, 0.0057201385498046875 sec
Batchsize: 1
Time spent per BATCH:     5.9215 ms
Total samples/sec:   168.8756 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9209 ms
Total samples/sec:   168.8936 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9202 ms
Total samples/sec:   168.9123 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9199 ms
Total samples/sec:   168.9214 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9193 ms
Total samples/sec:   168.9401 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9186 ms
Total samples/sec:   168.9592 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9184 ms
Total samples/sec:   168.9658 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9176 ms
Total samples/sec:   168.9862 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9170 ms
Total samples/sec:   169.0035 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9163 ms
Total samples/sec:   169.0232 samples/s
Total labeled samples: 1 person
steps = 689, 0.005457162857055664 sec
Batchsize: 1
Time spent per BATCH:     5.9157 ms
Total samples/sec:   169.0422 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9150 ms
Total samples/sec:   169.0608 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9177 ms
Total samples/sec:   168.9852 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9172 ms
Total samples/sec:   168.9986 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9167 ms
Total samples/sec:   169.0140 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9161 ms
Total samples/sec:   169.0303 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9155 ms
Total samples/sec:   169.0483 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9149 ms
Total samples/sec:   169.0654 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9144 ms
Total samples/sec:   169.0788 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9140 ms
Total samples/sec:   169.0914 samples/s
Total labeled samples: 1 person
steps = 699, 0.005438804626464844 sec
Batchsize: 1
Time spent per BATCH:     5.9133 ms
Total samples/sec:   169.1108 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9129 ms
Total samples/sec:   169.1217 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9122 ms
Total samples/sec:   169.1413 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9116 ms
Total samples/sec:   169.1602 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9110 ms
Total samples/sec:   169.1756 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9109 ms
Total samples/sec:   169.1777 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9105 ms
Total samples/sec:   169.1915 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9098 ms
Total samples/sec:   169.2098 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9092 ms
Total samples/sec:   169.2282 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9086 ms
Total samples/sec:   169.2442 samples/s
Total labeled samples: 1 person
steps = 709, 0.005429983139038086 sec
Batchsize: 1
Time spent per BATCH:     5.9079 ms
Total samples/sec:   169.2635 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9073 ms
Total samples/sec:   169.2823 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9068 ms
Total samples/sec:   169.2978 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9061 ms
Total samples/sec:   169.3159 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9055 ms
Total samples/sec:   169.3331 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9050 ms
Total samples/sec:   169.3477 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9046 ms
Total samples/sec:   169.3583 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9040 ms
Total samples/sec:   169.3771 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9036 ms
Total samples/sec:   169.3889 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9067 ms
Total samples/sec:   169.2993 samples/s
Total labeled samples: 1 person
steps = 719, 0.00782155990600586 sec
Batchsize: 1
Time spent per BATCH:     5.9094 ms
Total samples/sec:   169.2231 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9087 ms
Total samples/sec:   169.2412 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9082 ms
Total samples/sec:   169.2551 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9076 ms
Total samples/sec:   169.2729 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9070 ms
Total samples/sec:   169.2894 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9064 ms
Total samples/sec:   169.3085 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9059 ms
Total samples/sec:   169.3227 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9056 ms
Total samples/sec:   169.3317 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9052 ms
Total samples/sec:   169.3432 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9049 ms
Total samples/sec:   169.3500 samples/s
Total labeled samples: 1 person
steps = 729, 0.005440235137939453 sec
Batchsize: 1
Time spent per BATCH:     5.9043 ms
Total samples/sec:   169.3683 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9037 ms
Total samples/sec:   169.3859 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9031 ms
Total samples/sec:   169.4020 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9027 ms
Total samples/sec:   169.4133 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9021 ms
Total samples/sec:   169.4317 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9015 ms
Total samples/sec:   169.4498 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9008 ms
Total samples/sec:   169.4691 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.9003 ms
Total samples/sec:   169.4839 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8998 ms
Total samples/sec:   169.4963 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8994 ms
Total samples/sec:   169.5093 samples/s
Total labeled samples: 1 person
steps = 739, 0.005579233169555664 sec
Batchsize: 1
Time spent per BATCH:     5.8989 ms
Total samples/sec:   169.5217 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8987 ms
Total samples/sec:   169.5298 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8981 ms
Total samples/sec:   169.5467 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8975 ms
Total samples/sec:   169.5637 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8972 ms
Total samples/sec:   169.5728 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8966 ms
Total samples/sec:   169.5900 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8960 ms
Total samples/sec:   169.6065 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8956 ms
Total samples/sec:   169.6181 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8952 ms
Total samples/sec:   169.6292 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8946 ms
Total samples/sec:   169.6471 samples/s
Total labeled samples: 1 person
steps = 749, 0.005724191665649414 sec
Batchsize: 1
Time spent per BATCH:     5.8944 ms
Total samples/sec:   169.6536 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8938 ms
Total samples/sec:   169.6704 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8932 ms
Total samples/sec:   169.6868 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8929 ms
Total samples/sec:   169.6946 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8928 ms
Total samples/sec:   169.6991 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8924 ms
Total samples/sec:   169.7098 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8920 ms
Total samples/sec:   169.7205 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8915 ms
Total samples/sec:   169.7359 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8909 ms
Total samples/sec:   169.7523 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8903 ms
Total samples/sec:   169.7698 samples/s
Total labeled samples: 1 person
steps = 759, 0.005509614944458008 sec
Batchsize: 1
Time spent per BATCH:     5.8898 ms
Total samples/sec:   169.7843 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8896 ms
Total samples/sec:   169.7912 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8892 ms
Total samples/sec:   169.8010 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8887 ms
Total samples/sec:   169.8169 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8882 ms
Total samples/sec:   169.8324 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8878 ms
Total samples/sec:   169.8416 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8875 ms
Total samples/sec:   169.8527 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8870 ms
Total samples/sec:   169.8653 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8865 ms
Total samples/sec:   169.8803 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8861 ms
Total samples/sec:   169.8921 samples/s
Total labeled samples: 1 person
steps = 769, 0.005515575408935547 sec
Batchsize: 1
Time spent per BATCH:     5.8856 ms
Total samples/sec:   169.9060 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8853 ms
Total samples/sec:   169.9134 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8848 ms
Total samples/sec:   169.9299 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8878 ms
Total samples/sec:   169.8436 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8874 ms
Total samples/sec:   169.8557 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8868 ms
Total samples/sec:   169.8724 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8862 ms
Total samples/sec:   169.8879 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8860 ms
Total samples/sec:   169.8961 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8855 ms
Total samples/sec:   169.9099 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8850 ms
Total samples/sec:   169.9247 samples/s
Total labeled samples: 1 person
steps = 779, 0.005478382110595703 sec
Batchsize: 1
Time spent per BATCH:     5.8844 ms
Total samples/sec:   169.9398 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8842 ms
Total samples/sec:   169.9473 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8836 ms
Total samples/sec:   169.9647 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8830 ms
Total samples/sec:   169.9822 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8826 ms
Total samples/sec:   169.9939 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8820 ms
Total samples/sec:   170.0109 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8840 ms
Total samples/sec:   169.9524 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8838 ms
Total samples/sec:   169.9584 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8879 ms
Total samples/sec:   169.8403 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8874 ms
Total samples/sec:   169.8546 samples/s
Total labeled samples: 1 person
steps = 789, 0.007943391799926758 sec
Batchsize: 1
Time spent per BATCH:     5.8900 ms
Total samples/sec:   169.7796 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8895 ms
Total samples/sec:   169.7949 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8893 ms
Total samples/sec:   169.7994 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8887 ms
Total samples/sec:   169.8156 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8882 ms
Total samples/sec:   169.8309 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8877 ms
Total samples/sec:   169.8461 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8872 ms
Total samples/sec:   169.8596 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8868 ms
Total samples/sec:   169.8729 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8890 ms
Total samples/sec:   169.8071 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8885 ms
Total samples/sec:   169.8220 samples/s
Total labeled samples: 1 person
steps = 799, 0.005711793899536133 sec
Batchsize: 1
Time spent per BATCH:     5.8883 ms
Total samples/sec:   169.8283 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8878 ms
Total samples/sec:   169.8432 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8872 ms
Total samples/sec:   169.8590 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8868 ms
Total samples/sec:   169.8722 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8864 ms
Total samples/sec:   169.8837 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8858 ms
Total samples/sec:   169.8990 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8853 ms
Total samples/sec:   169.9146 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8849 ms
Total samples/sec:   169.9263 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8847 ms
Total samples/sec:   169.9331 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8843 ms
Total samples/sec:   169.9450 samples/s
Total labeled samples: 1 person
steps = 809, 0.0054666996002197266 sec
Batchsize: 1
Time spent per BATCH:     5.8837 ms
Total samples/sec:   169.9599 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8832 ms
Total samples/sec:   169.9743 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8827 ms
Total samples/sec:   169.9886 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8855 ms
Total samples/sec:   169.9095 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8849 ms
Total samples/sec:   169.9256 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8873 ms
Total samples/sec:   169.8575 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8867 ms
Total samples/sec:   169.8735 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8863 ms
Total samples/sec:   169.8849 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8859 ms
Total samples/sec:   169.8986 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8887 ms
Total samples/sec:   169.8157 samples/s
Total labeled samples: 1 person
steps = 819, 0.005405426025390625 sec
Batchsize: 1
Time spent per BATCH:     5.8881 ms
Total samples/sec:   169.8327 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8876 ms
Total samples/sec:   169.8497 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8870 ms
Total samples/sec:   169.8647 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8867 ms
Total samples/sec:   169.8742 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8886 ms
Total samples/sec:   169.8194 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8882 ms
Total samples/sec:   169.8324 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8878 ms
Total samples/sec:   169.8438 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8874 ms
Total samples/sec:   169.8554 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8869 ms
Total samples/sec:   169.8674 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8866 ms
Total samples/sec:   169.8776 samples/s
Total labeled samples: 1 person
steps = 829, 0.005401611328125 sec
Batchsize: 1
Time spent per BATCH:     5.8860 ms
Total samples/sec:   169.8944 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8855 ms
Total samples/sec:   169.9094 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8850 ms
Total samples/sec:   169.9249 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8879 ms
Total samples/sec:   169.8411 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8874 ms
Total samples/sec:   169.8552 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8869 ms
Total samples/sec:   169.8673 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8864 ms
Total samples/sec:   169.8833 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8892 ms
Total samples/sec:   169.8022 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8887 ms
Total samples/sec:   169.8170 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8915 ms
Total samples/sec:   169.7371 samples/s
Total labeled samples: 1 person
steps = 839, 0.005497932434082031 sec
Batchsize: 1
Time spent per BATCH:     5.8910 ms
Total samples/sec:   169.7506 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8905 ms
Total samples/sec:   169.7659 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8900 ms
Total samples/sec:   169.7796 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8894 ms
Total samples/sec:   169.7962 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8889 ms
Total samples/sec:   169.8100 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8884 ms
Total samples/sec:   169.8243 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8879 ms
Total samples/sec:   169.8399 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8874 ms
Total samples/sec:   169.8542 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8869 ms
Total samples/sec:   169.8678 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8868 ms
Total samples/sec:   169.8717 samples/s
Total labeled samples: 1 person
steps = 849, 0.005499124526977539 sec
Batchsize: 1
Time spent per BATCH:     5.8863 ms
Total samples/sec:   169.8848 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8860 ms
Total samples/sec:   169.8953 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8855 ms
Total samples/sec:   169.9101 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8849 ms
Total samples/sec:   169.9264 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8845 ms
Total samples/sec:   169.9383 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8840 ms
Total samples/sec:   169.9512 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8835 ms
Total samples/sec:   169.9668 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8830 ms
Total samples/sec:   169.9821 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8825 ms
Total samples/sec:   169.9961 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8820 ms
Total samples/sec:   170.0088 samples/s
Total labeled samples: 1 person
steps = 859, 0.005461692810058594 sec
Batchsize: 1
Time spent per BATCH:     5.8816 ms
Total samples/sec:   170.0230 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8810 ms
Total samples/sec:   170.0390 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8805 ms
Total samples/sec:   170.0548 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8799 ms
Total samples/sec:   170.0714 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8827 ms
Total samples/sec:   169.9887 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8823 ms
Total samples/sec:   170.0029 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8817 ms
Total samples/sec:   170.0184 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8839 ms
Total samples/sec:   169.9562 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8834 ms
Total samples/sec:   169.9706 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8829 ms
Total samples/sec:   169.9841 samples/s
Total labeled samples: 1 person
steps = 869, 0.00545501708984375 sec
Batchsize: 1
Time spent per BATCH:     5.8824 ms
Total samples/sec:   169.9983 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8819 ms
Total samples/sec:   170.0133 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8814 ms
Total samples/sec:   170.0267 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8809 ms
Total samples/sec:   170.0410 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8805 ms
Total samples/sec:   170.0548 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8823 ms
Total samples/sec:   170.0028 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8838 ms
Total samples/sec:   169.9569 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8835 ms
Total samples/sec:   169.9673 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8831 ms
Total samples/sec:   169.9793 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8828 ms
Total samples/sec:   169.9874 samples/s
Total labeled samples: 1 person
steps = 879, 0.005467414855957031 sec
Batchsize: 1
Time spent per BATCH:     5.8823 ms
Total samples/sec:   170.0010 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8819 ms
Total samples/sec:   170.0141 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8815 ms
Total samples/sec:   170.0254 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8811 ms
Total samples/sec:   170.0360 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8807 ms
Total samples/sec:   170.0485 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8803 ms
Total samples/sec:   170.0597 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8797 ms
Total samples/sec:   170.0756 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8794 ms
Total samples/sec:   170.0866 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8788 ms
Total samples/sec:   170.1013 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8784 ms
Total samples/sec:   170.1153 samples/s
Total labeled samples: 1 person
steps = 889, 0.005522251129150391 sec
Batchsize: 1
Time spent per BATCH:     5.8780 ms
Total samples/sec:   170.1268 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8775 ms
Total samples/sec:   170.1394 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8796 ms
Total samples/sec:   170.0794 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8813 ms
Total samples/sec:   170.0301 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8810 ms
Total samples/sec:   170.0383 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8806 ms
Total samples/sec:   170.0513 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8802 ms
Total samples/sec:   170.0631 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8797 ms
Total samples/sec:   170.0781 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8792 ms
Total samples/sec:   170.0925 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8787 ms
Total samples/sec:   170.1057 samples/s
Total labeled samples: 1 person
steps = 899, 0.005493879318237305 sec
Batchsize: 1
Time spent per BATCH:     5.8783 ms
Total samples/sec:   170.1181 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8779 ms
Total samples/sec:   170.1281 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8813 ms
Total samples/sec:   170.0310 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8808 ms
Total samples/sec:   170.0453 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8803 ms
Total samples/sec:   170.0598 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8798 ms
Total samples/sec:   170.0730 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8793 ms
Total samples/sec:   170.0878 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8788 ms
Total samples/sec:   170.1023 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8805 ms
Total samples/sec:   170.0525 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8802 ms
Total samples/sec:   170.0636 samples/s
Total labeled samples: 1 person
steps = 909, 0.008220672607421875 sec
Batchsize: 1
Time spent per BATCH:     5.8827 ms
Total samples/sec:   169.9893 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8822 ms
Total samples/sec:   170.0033 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8817 ms
Total samples/sec:   170.0176 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8813 ms
Total samples/sec:   170.0295 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8808 ms
Total samples/sec:   170.0444 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8803 ms
Total samples/sec:   170.0585 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8798 ms
Total samples/sec:   170.0743 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8812 ms
Total samples/sec:   170.0336 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8807 ms
Total samples/sec:   170.0467 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8803 ms
Total samples/sec:   170.0595 samples/s
Total labeled samples: 1 person
steps = 919, 0.00786280632019043 sec
Batchsize: 1
Time spent per BATCH:     5.8825 ms
Total samples/sec:   169.9972 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8820 ms
Total samples/sec:   170.0102 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8815 ms
Total samples/sec:   170.0238 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8810 ms
Total samples/sec:   170.0390 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8805 ms
Total samples/sec:   170.0522 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8826 ms
Total samples/sec:   169.9917 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8822 ms
Total samples/sec:   170.0048 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8817 ms
Total samples/sec:   170.0181 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8812 ms
Total samples/sec:   170.0324 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8840 ms
Total samples/sec:   169.9523 samples/s
Total labeled samples: 1 person
steps = 929, 0.0054168701171875 sec
Batchsize: 1
Time spent per BATCH:     5.8835 ms
Total samples/sec:   169.9668 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8831 ms
Total samples/sec:   169.9797 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8851 ms
Total samples/sec:   169.9199 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8846 ms
Total samples/sec:   169.9344 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8842 ms
Total samples/sec:   169.9459 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8838 ms
Total samples/sec:   169.9579 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8833 ms
Total samples/sec:   169.9721 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8828 ms
Total samples/sec:   169.9859 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8824 ms
Total samples/sec:   169.9989 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8850 ms
Total samples/sec:   169.9226 samples/s
Total labeled samples: 1 person
steps = 939, 0.005432605743408203 sec
Batchsize: 1
Time spent per BATCH:     5.8846 ms
Total samples/sec:   169.9365 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8841 ms
Total samples/sec:   169.9496 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8836 ms
Total samples/sec:   169.9633 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8831 ms
Total samples/sec:   169.9773 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8827 ms
Total samples/sec:   169.9899 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8822 ms
Total samples/sec:   170.0034 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8818 ms
Total samples/sec:   170.0148 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8813 ms
Total samples/sec:   170.0297 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8809 ms
Total samples/sec:   170.0425 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8805 ms
Total samples/sec:   170.0535 samples/s
Total labeled samples: 1 person
steps = 949, 0.005427360534667969 sec
Batchsize: 1
Time spent per BATCH:     5.8800 ms
Total samples/sec:   170.0673 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8825 ms
Total samples/sec:   169.9945 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8821 ms
Total samples/sec:   170.0071 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8847 ms
Total samples/sec:   169.9333 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8842 ms
Total samples/sec:   169.9456 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8838 ms
Total samples/sec:   169.9577 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8833 ms
Total samples/sec:   169.9716 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8829 ms
Total samples/sec:   169.9851 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8846 ms
Total samples/sec:   169.9363 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8841 ms
Total samples/sec:   169.9504 samples/s
Total labeled samples: 1 person
steps = 959, 0.0055010318756103516 sec
Batchsize: 1
Time spent per BATCH:     5.8837 ms
Total samples/sec:   169.9619 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8832 ms
Total samples/sec:   169.9751 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8828 ms
Total samples/sec:   169.9865 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8823 ms
Total samples/sec:   170.0005 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8819 ms
Total samples/sec:   170.0120 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8834 ms
Total samples/sec:   169.9694 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8830 ms
Total samples/sec:   169.9821 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8825 ms
Total samples/sec:   169.9948 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8821 ms
Total samples/sec:   170.0085 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8816 ms
Total samples/sec:   170.0206 samples/s
Total labeled samples: 1 person
steps = 969, 0.005452394485473633 sec
Batchsize: 1
Time spent per BATCH:     5.8812 ms
Total samples/sec:   170.0334 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8808 ms
Total samples/sec:   170.0461 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8803 ms
Total samples/sec:   170.0595 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8798 ms
Total samples/sec:   170.0726 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8794 ms
Total samples/sec:   170.0846 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8790 ms
Total samples/sec:   170.0965 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8786 ms
Total samples/sec:   170.1088 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8782 ms
Total samples/sec:   170.1202 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8778 ms
Total samples/sec:   170.1326 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8773 ms
Total samples/sec:   170.1455 samples/s
Total labeled samples: 1 person
steps = 979, 0.0078012943267822266 sec
Batchsize: 1
Time spent per BATCH:     5.8793 ms
Total samples/sec:   170.0887 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8788 ms
Total samples/sec:   170.1016 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8784 ms
Total samples/sec:   170.1144 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8780 ms
Total samples/sec:   170.1269 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8775 ms
Total samples/sec:   170.1393 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8771 ms
Total samples/sec:   170.1511 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8767 ms
Total samples/sec:   170.1622 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8763 ms
Total samples/sec:   170.1741 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8759 ms
Total samples/sec:   170.1861 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8755 ms
Total samples/sec:   170.1982 samples/s
Total labeled samples: 1 person
steps = 989, 0.005518674850463867 sec
Batchsize: 1
Time spent per BATCH:     5.8751 ms
Total samples/sec:   170.2087 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8747 ms
Total samples/sec:   170.2201 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8743 ms
Total samples/sec:   170.2322 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8739 ms
Total samples/sec:   170.2452 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8734 ms
Total samples/sec:   170.2579 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8730 ms
Total samples/sec:   170.2709 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8726 ms
Total samples/sec:   170.2824 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8722 ms
Total samples/sec:   170.2947 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8749 ms
Total samples/sec:   170.2161 samples/s
Total labeled samples: 1 person
Batchsize: 1
Time spent per BATCH:     5.8744 ms
Total samples/sec:   170.2288 samples/s
Total labeled samples: 1 person
steps = 999, 0.005470752716064453 sec
Batchsize: 1
Time spent per BATCH:     5.8740 ms
Total samples/sec:   170.2405 samples/s
Total labeled samples: 1 person
Received these standard args: Namespace(accuracy_only=False, annotations_dir='/home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/caffe2-pose-estimation/annotations/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages', batch_size=1, benchmark_dir='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks', benchmark_only=True, checkpoint=None, data_location='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/pascal_voc_tfrecord/tfrecord-voc.record', data_num_inter_threads=None, data_num_intra_threads=None, disable_tcmalloc=True, framework='caffe', input_graph='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/Edge-optimized-models/SqueezeNet-5-Class-detection/SqueezeNetSSD-5Class.prototxt', input_weights='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/Edge-optimized-models/SqueezeNet-5-Class-detection/SqueezeNetSSD-5Class.caffemodel', intelai_models='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/ssd_squeezenet', mode='inference', model_args=[], model_name='ssd_squeezenet', model_source_dir='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/Edge-optimized-models/', mpi=None, num_cores=2, num_instances=1, num_inter_threads=1, num_intra_threads=1, num_mpi=1, output_dir='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord', output_results=False, precision='fp32', risk_difference=0.5, socket_id=0, tcmalloc_large_alloc_report_threshold=2147483648, use_case='object_detection', verbose=True)
Received these custom args: []
Current directory: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks
Running: numactl --cpunodebind=0 --membind=0 /usr/bin/python3.6 /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/ssd_squeezenet/inference/fp32/infer_detections.py -i 1000 -w 200 -a 1 -e 1 -g /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/Edge-optimized-models/SqueezeNet-5-Class-detection/SqueezeNetSSD-5Class.prototxt -weight /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/Edge-optimized-models/SqueezeNet-5-Class-detection/SqueezeNetSSD-5Class.caffemodel -rd 0.5 -b 1 -bo True -d /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/pascal_voc_tfrecord/tfrecord-voc.record --annotations_dir /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/caffe2-pose-estimation/annotations/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages
PYTHONPATH: :/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/ssd_squeezenet
RUNCMD: /usr/bin/python3.6 common/caffe/run_tf_benchmark.py --framework=caffe --use-case=object_detection --model-name=ssd_squeezenet --precision=fp32 --mode=inference --benchmark-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks --intelai-models=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/ssd_squeezenet --num-cores=2 --batch-size=1 --socket-id=0 --output-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord --annotations_dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/caffe2-pose-estimation/annotations/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages  --benchmark-only  --verbose --model-source-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/Edge-optimized-models/ --in-graph=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/Edge-optimized-models/SqueezeNet-5-Class-detection/SqueezeNetSSD-5Class.prototxt --in-weights=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/Edge-optimized-models/SqueezeNet-5-Class-detection/SqueezeNetSSD-5Class.caffemodel --data-location=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/pascal_voc_tfrecord/tfrecord-voc.record --num-inter-threads=1 --num-intra-threads=1 --disable-tcmalloc=True                   
Batch Size: 1
Ran inference with batch size 1
Log location outside container: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord/benchmark_ssd_squeezenet_inference_fp32_20200519_233744.log
