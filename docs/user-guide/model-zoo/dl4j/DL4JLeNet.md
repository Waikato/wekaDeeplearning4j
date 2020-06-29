# DL4JLeNet

[Back to Model Zoo](../../model-zoo.md)

```text
==========================================================================================
VertexName (VertexType)   nIn,nOut   TotalParams   ParamsShape               Vertex Inputs
==========================================================================================
in (InputVertex)          -,-        -             -                         -
0 (ConvolutionLayer)      1,20       520           W:{20,1,5,5}, b:{1,20}    [in]
1 (ActivationLayer)       -,-        0             -                         [0]
2 (SubsamplingLayer)      -,-        0             -                         [1]
3 (ConvolutionLayer)      20,50      25,050        W:{50,20,5,5}, b:{1,50}   [2]
4 (ActivationLayer)       -,-        0             -                         [3]
5 (SubsamplingLayer)      -,-        0             -                         [4]
6 (DenseLayer)            2450,500   1,225,500     W:{2450,500}, b:{1,500}   [5]
7 (ActivationLayer)       -,-        0             -                         [6]
8 (DenseLayer)            500,10     5,010         W:{500,10}, b:{1,10}      [7]
9 (ActivationLayer)       -,-        0             -                         [8]
------------------------------------------------------------------------------------------
            Total Parameters:  1,256,080
        Trainable Parameters:  1,256,080
           Frozen Parameters:  0
==========================================================================================
```