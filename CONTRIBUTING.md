# Contributing

Contributions are welcome and an easy way to get started is to file an issue. Make sure to be as descriptive about your problem as possible. Try to explain what you have tried, what you expected and what the actual outcome was. Give additional information about your java and weka version, as well as platform specific details that could be relevant. 

If you are going to contribute to the codebase, you should fork this repository, create a separate branch on which you commit your changes and file a pull request. A well explained how-to is described [here](https://gist.github.com/Chaser324/ce0505fbed06b947d962).

### Java Code Style
This package mostly follows the official [Google Java Style Guide](https://google.github.io/styleguide/javaguide.html).

### Build Locally
Simply run the `build.sh` script. This assumes:
* Bash
* GNU grep
* GNU sed
* Ant
* Maven
* Weka

```
Usage: build.sh

Optional arguments:
   -v/--verbose            Enable verbose mode
   -i/--install-packages   Install selected packages
   -b/--backend            Select specific backend 
                           Available: ( CPU GPU )
   -c/--clean              Clean up build-environment
   -h/--help               Show this message
```
