# DL4JVGG16

## ImageNet Pretrained

```text
========================================================================================================
VertexName (VertexType)           nIn,nOut     TotalParams   ParamsShape                  Vertex Inputs
========================================================================================================
input_1 (InputVertex)             -,-          -             -                            -
block1_conv1 (ConvolutionLayer)   3,64         1,792         W:{64,3,3,3}, b:{1,64}       [input_1]
block1_conv2 (ConvolutionLayer)   64,64        36,928        W:{64,64,3,3}, b:{1,64}      [block1_conv1]
block1_pool (SubsamplingLayer)    -,-          0             -                            [block1_conv2]
block2_conv1 (ConvolutionLayer)   64,128       73,856        W:{128,64,3,3}, b:{1,128}    [block1_pool]
block2_conv2 (ConvolutionLayer)   128,128      147,584       W:{128,128,3,3}, b:{1,128}   [block2_conv1]
block2_pool (SubsamplingLayer)    -,-          0             -                            [block2_conv2]
block3_conv1 (ConvolutionLayer)   128,256      295,168       W:{256,128,3,3}, b:{1,256}   [block2_pool]
block3_conv2 (ConvolutionLayer)   256,256      590,080       W:{256,256,3,3}, b:{1,256}   [block3_conv1]
block3_conv3 (ConvolutionLayer)   256,256      590,080       W:{256,256,3,3}, b:{1,256}   [block3_conv2]
block3_pool (SubsamplingLayer)    -,-          0             -                            [block3_conv3]
block4_conv1 (ConvolutionLayer)   256,512      1,180,160     W:{512,256,3,3}, b:{1,512}   [block3_pool]
block4_conv2 (ConvolutionLayer)   512,512      2,359,808     W:{512,512,3,3}, b:{1,512}   [block4_conv1]
block4_conv3 (ConvolutionLayer)   512,512      2,359,808     W:{512,512,3,3}, b:{1,512}   [block4_conv2]
block4_pool (SubsamplingLayer)    -,-          0             -                            [block4_conv3]
block5_conv1 (ConvolutionLayer)   512,512      2,359,808     W:{512,512,3,3}, b:{1,512}   [block4_pool]
block5_conv2 (ConvolutionLayer)   512,512      2,359,808     W:{512,512,3,3}, b:{1,512}   [block5_conv1]
block5_conv3 (ConvolutionLayer)   512,512      2,359,808     W:{512,512,3,3}, b:{1,512}   [block5_conv2]
block5_pool (SubsamplingLayer)    -,-          0             -                            [block5_conv3]
flatten (PreprocessorVertex)      -,-          -             -                            [block5_pool]
fc1 (DenseLayer)                  25088,4096   102,764,544   W:{25088,4096}, b:{1,4096}   [flatten]
fc2 (DenseLayer)                  4096,4096    16,781,312    W:{4096,4096}, b:{1,4096}    [fc1]
predictions (DenseLayer)          4096,1000    4,097,000     W:{4096,1000}, b:{1,1000}    [fc2]
--------------------------------------------------------------------------------------------------------
            Total Parameters:  138,357,544
        Trainable Parameters:  138,357,544
           Frozen Parameters:  0
========================================================================================================
```

## VGGFACE Pretrained

```text
====================================================================================================
VertexName (VertexType)        nIn,nOut     TotalParams   ParamsShape                  Vertex Inputs
====================================================================================================
input_1 (InputVertex)          -,-          -             -                            -
conv1_1 (ConvolutionLayer)     3,64         1,792         W:{64,3,3,3}, b:{1,64}       [input_1]
conv1_2 (ConvolutionLayer)     64,64        36,928        W:{64,64,3,3}, b:{1,64}      [conv1_1]
pool1 (SubsamplingLayer)       -,-          0             -                            [conv1_2]
conv2_1 (ConvolutionLayer)     64,128       73,856        W:{128,64,3,3}, b:{1,128}    [pool1]
conv2_2 (ConvolutionLayer)     128,128      147,584       W:{128,128,3,3}, b:{1,128}   [conv2_1]
pool2 (SubsamplingLayer)       -,-          0             -                            [conv2_2]
conv3_1 (ConvolutionLayer)     128,256      295,168       W:{256,128,3,3}, b:{1,256}   [pool2]
conv3_2 (ConvolutionLayer)     256,256      590,080       W:{256,256,3,3}, b:{1,256}   [conv3_1]
conv3_3 (ConvolutionLayer)     256,256      590,080       W:{256,256,3,3}, b:{1,256}   [conv3_2]
pool3 (SubsamplingLayer)       -,-          0             -                            [conv3_3]
conv4_1 (ConvolutionLayer)     256,512      1,180,160     W:{512,256,3,3}, b:{1,512}   [pool3]
conv4_2 (ConvolutionLayer)     512,512      2,359,808     W:{512,512,3,3}, b:{1,512}   [conv4_1]
conv4_3 (ConvolutionLayer)     512,512      2,359,808     W:{512,512,3,3}, b:{1,512}   [conv4_2]
pool4 (SubsamplingLayer)       -,-          0             -                            [conv4_3]
conv5_1 (ConvolutionLayer)     512,512      2,359,808     W:{512,512,3,3}, b:{1,512}   [pool4]
conv5_2 (ConvolutionLayer)     512,512      2,359,808     W:{512,512,3,3}, b:{1,512}   [conv5_1]
conv5_3 (ConvolutionLayer)     512,512      2,359,808     W:{512,512,3,3}, b:{1,512}   [conv5_2]
pool5 (SubsamplingLayer)       -,-          0             -                            [conv5_3]
flatten (PreprocessorVertex)   -,-          -             -                            [pool5]
fc6 (DenseLayer)               25088,4096   102,764,544   W:{25088,4096}, b:{1,4096}   [flatten]
fc7 (DenseLayer)               4096,4096    16,781,312    W:{4096,4096}, b:{1,4096}    [fc6]
fc8 (DenseLayer)               4096,2622    10,742,334    W:{4096,2622}, b:{1,2622}    [fc7]
----------------------------------------------------------------------------------------------------
            Total Parameters:  145,002,878
        Trainable Parameters:  145,002,878
           Frozen Parameters:  0
====================================================================================================
```