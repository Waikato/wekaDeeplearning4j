# Contributing

Contributions are welcome and an easy way to get started is to file an issue. Make sure to be as descriptive about your problem as possible. Try to explain what you have tried, what you expected and what the actual outcome was. Give additional information about your java and weka version, as well as platform specific details that could be relevant. 

If you are going to contribute to the codebase, you should fork this repository, create a separate branch on which you commit your changes and file a pull request. A well explained how-to is described [here](https://gist.github.com/Chaser324/ce0505fbed06b947d962).

### Java Code Style
This package mostly follows the official [Google Java Style Guide](https://google.github.io/styleguide/javaguide.html).

### Build Locally
The package uses Gradle to manage dependencies and build the necessary weka package zip file, as well as the additional cuda library zip files. It is either possible to call the specific gradle tasks:
```bash
$ ./gradlew makeMain
$ ./gradlew makeCuda -Dcuda=8.0
$ ./gradlew makeCuda -Dcuda=9.0
$ ./gradlew makeCuda -Dcuda=9.2
```

or to use the provided `build.py` script. The usage is as follows:
```bash
$ ./build.py -h
usage: build.py [-h] [--cuda-version {8.0,9.0,9.2}] [--build-all] [--verbose]

Build the wekaDeeplearning4j packages.

optional arguments:
  -h, --help            show this help message and exit
  --cuda-version {8.0,9.0,9.2}, -c {8.0,9.0,9.2}
                        The cuda version.
  --build-all, -a       Flag to build all platform/cuda packages.
  --verbose, -v         Enable verbose output.
```
