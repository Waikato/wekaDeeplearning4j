# DL4JDarkNet19 

[Back to Model Zoo](../../model-zoo.md)

```text
=============================================================================================================================================================
VertexName (VertexType)                       nIn,nOut    TotalParams   ParamsShape                                                  Vertex Inputs
=============================================================================================================================================================
input_1 (InputVertex)                         -,-         -             -                                                            -
conv2d_1 (ConvolutionLayer)                   3,32        864           W:{32,3,3,3}                                                 [input_1]
batch_normalization_1 (BatchNormalization)    32,32       128           gamma:{1,32}, beta:{1,32}, mean:{1,32}, var:{1,32}           [conv2d_1]
leaky_re_lu_1 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_1]
max_pooling2d_1 (SubsamplingLayer)            -,-         0             -                                                            [leaky_re_lu_1]
conv2d_2 (ConvolutionLayer)                   32,64       18,432        W:{64,32,3,3}                                                [max_pooling2d_1]
batch_normalization_2 (BatchNormalization)    64,64       256           gamma:{1,64}, beta:{1,64}, mean:{1,64}, var:{1,64}           [conv2d_2]
leaky_re_lu_2 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_2]
max_pooling2d_2 (SubsamplingLayer)            -,-         0             -                                                            [leaky_re_lu_2]
conv2d_3 (ConvolutionLayer)                   64,128      73,728        W:{128,64,3,3}                                               [max_pooling2d_2]
batch_normalization_3 (BatchNormalization)    128,128     512           gamma:{1,128}, beta:{1,128}, mean:{1,128}, var:{1,128}       [conv2d_3]
leaky_re_lu_3 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_3]
conv2d_4 (ConvolutionLayer)                   128,64      8,192         W:{64,128,1,1}                                               [leaky_re_lu_3]
batch_normalization_4 (BatchNormalization)    64,64       256           gamma:{1,64}, beta:{1,64}, mean:{1,64}, var:{1,64}           [conv2d_4]
leaky_re_lu_4 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_4]
conv2d_5 (ConvolutionLayer)                   64,128      73,728        W:{128,64,3,3}                                               [leaky_re_lu_4]
batch_normalization_5 (BatchNormalization)    128,128     512           gamma:{1,128}, beta:{1,128}, mean:{1,128}, var:{1,128}       [conv2d_5]
leaky_re_lu_5 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_5]
max_pooling2d_3 (SubsamplingLayer)            -,-         0             -                                                            [leaky_re_lu_5]
conv2d_6 (ConvolutionLayer)                   128,256     294,912       W:{256,128,3,3}                                              [max_pooling2d_3]
batch_normalization_6 (BatchNormalization)    256,256     1,024         gamma:{1,256}, beta:{1,256}, mean:{1,256}, var:{1,256}       [conv2d_6]
leaky_re_lu_6 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_6]
conv2d_7 (ConvolutionLayer)                   256,128     32,768        W:{128,256,1,1}                                              [leaky_re_lu_6]
batch_normalization_7 (BatchNormalization)    128,128     512           gamma:{1,128}, beta:{1,128}, mean:{1,128}, var:{1,128}       [conv2d_7]
leaky_re_lu_7 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_7]
conv2d_8 (ConvolutionLayer)                   128,256     294,912       W:{256,128,3,3}                                              [leaky_re_lu_7]
batch_normalization_8 (BatchNormalization)    256,256     1,024         gamma:{1,256}, beta:{1,256}, mean:{1,256}, var:{1,256}       [conv2d_8]
leaky_re_lu_8 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_8]
max_pooling2d_4 (SubsamplingLayer)            -,-         0             -                                                            [leaky_re_lu_8]
conv2d_9 (ConvolutionLayer)                   256,512     1,179,648     W:{512,256,3,3}                                              [max_pooling2d_4]
batch_normalization_9 (BatchNormalization)    512,512     2,048         gamma:{1,512}, beta:{1,512}, mean:{1,512}, var:{1,512}       [conv2d_9]
leaky_re_lu_9 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_9]
conv2d_10 (ConvolutionLayer)                  512,256     131,072       W:{256,512,1,1}                                              [leaky_re_lu_9]
batch_normalization_10 (BatchNormalization)   256,256     1,024         gamma:{1,256}, beta:{1,256}, mean:{1,256}, var:{1,256}       [conv2d_10]
leaky_re_lu_10 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_10]
conv2d_11 (ConvolutionLayer)                  256,512     1,179,648     W:{512,256,3,3}                                              [leaky_re_lu_10]
batch_normalization_11 (BatchNormalization)   512,512     2,048         gamma:{1,512}, beta:{1,512}, mean:{1,512}, var:{1,512}       [conv2d_11]
leaky_re_lu_11 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_11]
conv2d_12 (ConvolutionLayer)                  512,256     131,072       W:{256,512,1,1}                                              [leaky_re_lu_11]
batch_normalization_12 (BatchNormalization)   256,256     1,024         gamma:{1,256}, beta:{1,256}, mean:{1,256}, var:{1,256}       [conv2d_12]
leaky_re_lu_12 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_12]
conv2d_13 (ConvolutionLayer)                  256,512     1,179,648     W:{512,256,3,3}                                              [leaky_re_lu_12]
batch_normalization_13 (BatchNormalization)   512,512     2,048         gamma:{1,512}, beta:{1,512}, mean:{1,512}, var:{1,512}       [conv2d_13]
leaky_re_lu_13 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_13]
max_pooling2d_5 (SubsamplingLayer)            -,-         0             -                                                            [leaky_re_lu_13]
conv2d_14 (ConvolutionLayer)                  512,1024    4,718,592     W:{1024,512,3,3}                                             [max_pooling2d_5]
batch_normalization_14 (BatchNormalization)   1024,1024   4,096         gamma:{1,1024}, beta:{1,1024}, mean:{1,1024}, var:{1,1024}   [conv2d_14]
leaky_re_lu_14 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_14]
conv2d_15 (ConvolutionLayer)                  1024,512    524,288       W:{512,1024,1,1}                                             [leaky_re_lu_14]
batch_normalization_15 (BatchNormalization)   512,512     2,048         gamma:{1,512}, beta:{1,512}, mean:{1,512}, var:{1,512}       [conv2d_15]
leaky_re_lu_15 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_15]
conv2d_16 (ConvolutionLayer)                  512,1024    4,718,592     W:{1024,512,3,3}                                             [leaky_re_lu_15]
batch_normalization_16 (BatchNormalization)   1024,1024   4,096         gamma:{1,1024}, beta:{1,1024}, mean:{1,1024}, var:{1,1024}   [conv2d_16]
leaky_re_lu_16 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_16]
conv2d_17 (ConvolutionLayer)                  1024,512    524,288       W:{512,1024,1,1}                                             [leaky_re_lu_16]
batch_normalization_17 (BatchNormalization)   512,512     2,048         gamma:{1,512}, beta:{1,512}, mean:{1,512}, var:{1,512}       [conv2d_17]
leaky_re_lu_17 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_17]
conv2d_18 (ConvolutionLayer)                  512,1024    4,718,592     W:{1024,512,3,3}                                             [leaky_re_lu_17]
batch_normalization_18 (BatchNormalization)   1024,1024   4,096         gamma:{1,1024}, beta:{1,1024}, mean:{1,1024}, var:{1,1024}   [conv2d_18]
leaky_re_lu_18 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_18]
conv2d_19 (ConvolutionLayer)                  1024,1000   1,025,000     W:{1000,1024,1,1}, b:{1,1000}                                [leaky_re_lu_18]
globalpooling (GlobalPoolingLayer)            -,-         0             -                                                            [conv2d_19]
softmax (ActivationLayer)                     -,-         0             -                                                            [globalpooling]
loss (LossLayer)                              -,-         0             -                                                            [softmax]
-------------------------------------------------------------------------------------------------------------------------------------------------------------
            Total Parameters:  20,856,776
        Trainable Parameters:  20,856,776
           Frozen Parameters:  0
=============================================================================================================================================================
```