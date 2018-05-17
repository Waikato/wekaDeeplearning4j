# Prerequisites
- Weka 3.8.0 or above ([here](https://sourceforge.net/projects/weka/files/latest/download))
- WekaDeeplearning4j package 1.5.0 or above ([here](https://github.com/Waikato/wekaDeeplearning4j/releases/latest))

You need to unzip the Weka zip file to a directory of your choice.

#### CPU
For the CPU package no further requisites are necessary.

#### GPU
The GPU package needs the CUDA 8.0, 9.0 or 9.1 backend with the according cuDNN library to be installed on your system. Nvidia provides some good installation instructions for all platforms:

- [Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Mac OS X](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html)
- [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

# Installing the Weka Package
Weka packages can be easily installed either via the user interface as described [here](https://weka.wikispaces.com/How+do+I+use+the+package+manager%3F#toc2), or simply via the commandline:
```bash
$ java -cp <WEKA-JAR-PATH> weka.core.WekaPackageManager \
       -install-package <PACKAGE-ZIP>
```
where `<WEKA-JAR-PATH>` must be replaced by the path pointing to the Weka jar file, and `<PACKAGE-ZIP>` is the wekaDeeplearning4j package zip file.

You can check whether the installation was successful with
```bash
$ java -cp <WEKA-JAR-PATH> weka.core.WekaPackageManager \
       -list-packages installed
```
which results in
```
Installed	Repository	Loaded	Package
=========	==========	======	=======
1.5.0    	-----     	Yes	    <PACKAGE>: Weka wrappers for Deeplearning4j
```

# Using wekaDeeplearning4j in a Maven Project
It is also possible to include this package as maven project. As of now it is not provided in any maven repository, therefore you need to install this package to your local `.m2` repository:

```bash
$ git clone https://github.com/Waikato/wekaDeeplearning4j.git
$ cd wekaDeeplearning4j/package
$ mvn clean install

```

or, if you want the cuda version:

```bash
$ mvn clean install -P <CUDA-VERSION> # Replace <CUDA-VERSION> with either "8.0", "9.0" or "9.1"
```

Now you can add the maven dependency in your `pom.xml` file 
```xml
<dependency>
    <groupId>nz.ac.waikato.cms.weka</groupId>
    <artifactId>wekaDeeplearning4j</artifactId>
    <version>${wekaDeeplearning4j.version}</version>
</dependency>
```