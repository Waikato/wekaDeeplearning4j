# Prerequisites
- Weka 3.8.1 or above ([here](https://sourceforge.net/projects/weka/files/latest/download))
- WekaDeeplearning4j package 1.2 or above ([here](https://github.com/Waikato/wekaDeeplearning4j/releases/latest))

You need to unzip the Weka zip file to a directory of your choice.

#### CPU
For the CPU package no further requisites are necessary.

#### GPU
The GPU package needs the CUDA 8.0 backend to be installed on your system. Nvidia provides some good installation instructions for all platforms:

- [Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Mac OS X](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html)
- [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

# Installing the Weka Package
Weka packages can be easily installed either via the user interface as described [here](https://weka.wikispaces.com/How+do+I+use+the+package+manager%3F#toc2), or simply via the commandline:
```bash
$ java -cp <WEKA-JAR-PATH> weka.core.WekaPackageManager \
       -install-package wekaDeeplearning4j<BACKEND>-dev.zip
```
where `<WEKA-JAR-PATH>` must be replaced by the path pointing to the Weka jar file and `<BACKEND>` must be replaced by either `CPU` or `GPU`, depending on which version you chose.

You can check whether the installation was successful with
```bash
$ java -cp <WEKA-JAR-PATH> weka.core.WekaPackageManager \
       -list-packages installed
```
which results in
```
Installed	Repository	Loaded	Package
=========	==========	======	=======
1.2.0    	-----     	Yes	    wekaDeeplearning4j<BACKEND>-dev: Weka wrappers for Deeplearning4j
```
