	?n???X@?n???X@!?n???X@	?1i
?@?1i
?@!?1i
?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?n???X@?I+???A
ףp=?V@Y?G?z.@*	    ?'?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?n???V@!??o?R	W@)?n???V@1??o?R	W@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?x?&1@!y!5>b]@)?x?&1@1y!5>b]@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism? ?rh@!???ֲf@);?O??n??1?1?q1???:Preprocessing2F
Iterator::Model/?$?@!?'??j@)????Mbp?1L??dH?p?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?1i
?@I??oY?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?I+????I+???!?I+???      ??!       "      ??!       *      ??!       2	
ףp=?V@
ףp=?V@!
ףp=?V@:      ??!       B      ??!       J	?G?z.@?G?z.@!?G?z.@R      ??!       Z	?G?z.@?G?z.@!?G?z.@b      ??!       JCPU_ONLYY?1i
?@b q??oY?W@