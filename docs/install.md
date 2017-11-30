# Prerequisites
- Weka 3.8.0 or above ([here](https://sourceforge.net/projects/weka/files/latest/download))
- WekaDeeplearning4j package 1.3.4 or above ([here](https://github.com/Waikato/wekaDeeplearning4j/releases/latest))

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
       -install-package wekaDeeplearning4j-<BACKEND>-<PLATFORM>.zip
```
where `<WEKA-JAR-PATH>` must be replaced by the path pointing to the Weka jar file, `<BACKEND>` must be replaced by either `CPU` or `GPU`, depending on which version you chose and `<PLATFORM>` must be replaced with your operating system (linux, macosx, windows).

You can check whether the installation was successful with
```bash
$ java -cp <WEKA-JAR-PATH> weka.core.WekaPackageManager \
       -list-packages installed
```
which results in
```
Installed	Repository	Loaded	Package
=========	==========	======	=======
1.3.4    	-----     	Yes	    wekaDeeplearning4j-<BACKEND>-<PLATFORM>: Weka wrappers for Deeplearning4j
```

# Using wekaDeeplearning4j in a Maven Project
It is also possible to include this package as maven project. As of now it is not provided in any maven repository, therefore you need to install this package to your local `.m2` repository:

```bash
$ git clone https://github.com/Waikato/wekaDeeplearning4j.git
$ cd wekaDeeplearning4j/package
$ mvn clean install -P <backend> # Replace <backend> with either "CPU" or "GPU"
```

Now you can add the maven dependency in your `pom.xml` file 
```xml
<dependency>
    <groupId>nz.ac.waikato.cms.weka</groupId>
    <artifactId>wekaDeeplearning4j</artifactId>
    <version>${wekaDeeplearning4j.version}</version>
</dependency>
```

When using the CPU the following two dependencies have to be added:
```xml
<!--CPU Specific-->
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-native-platform</artifactId>
    <version>${nd4j.version}</version>
</dependency>
<dependency>
    <groupId>org.bytedeco.javacpp-presets</groupId>
    <artifactId>openblas-platform</artifactId>
    <version>0.2.19-1.3</version>
</dependency>
```
while using the GPU requires to add nd4j GPU dependencies:
```xml
<!--GPU Specific-->
<dependency>
    <groupId>org.bytedeco.javacpp-presets</groupId>
    <artifactId>cuda</artifactId>
    <version>8.0-6.0-1.3</version>
</dependency>
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-cuda-8.0-platform</artifactId>
    <version>${nd4j.version}</version>
</dependency>
```