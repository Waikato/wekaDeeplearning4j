This section shall provide solutions for issues that  may appear.

---------------------------------------------------------------
## CUDA: GOMP Version 4.0 not found
#### Issue
Starting the `Dl4jMlpClassifier` while using the GPU version of the package results in something similar to:
```
Caused by: java.lang.UnsatisfiedLinkError: 
    /home/user/.javacpp/cache/nd4j-cuda-8.0-0.9.1-linux-x86_64.jar/org/nd4j/nativeblas/linux-x86_64/libjnind4jcuda.so: 
    /usr/lib/x86_64-linux-gnu/libgomp.so.1: 
    version `GOMP_4.0' not found 
    (required by /home/user/.javacpp/cache/nd4j-cuda-8.0-0.9.1-linux-x86_64.jar/org/nd4j/nativeblas/linux-x86_64/libnd4jcuda.so)

```
This happens when your system is using a version below 4.9 of the gcc compiler. You can check this with:
```
$ gcc --version
gcc (Ubuntu 4.8.4-2ubuntu1~14.04.3) 4.8.4
Copyright (C) 2013 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
Therefore the libgomp.so.1 library is still of version 3.0, while the nd4j backend expects version 4.0.

#### Solution
Download the latest version of libgomp for your system and export the following:
```bash
export LD_PRELOAD=<PATH-TO-NEW-LIBGOMP.SO>
```
For Ubuntu you can get the library [here](https://packages.ubuntu.com/xenial/libgomp1), choose your architecture, download and extract the deb-file. For amd64 architectures this would be:
```bash
$ wget http://security.ubuntu.com/ubuntu/pool/main/g/gcc-5/libgomp1_5.4.0-6ubuntu1~16.04.4_amd64.deb
$ ar vx libgomp1_5.4.0-6ubuntu1~16.04.4_amd64.deb
$ tar -xvf data.tar.xz
```
This extracts the library to `./usr/lib/x86_64-linux-gnu/libgomp.so.1`. Afterward set the `LD_PRELOAD` variable to this path as an absolute path and export it as shown above.

---------------------------------------------------------------
## CUDA: Failed to allocate X bytes from DEVICE memory
#### Issue
Your network architecture or your batch size consumes too much memory.

#### Solution
Use a lower batch size, or adjust your Java heap and off-heap limits to your available memory accordingly to the [official Dl4J memory description](https://deeplearning4j.org/memory).