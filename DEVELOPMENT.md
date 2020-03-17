# Development

This document provides information that are specific to the development of Wekadeeplearning4j.

## Publish a new Release

See the [RELEASE.md](./RELEASE.md) documentation.

## Build the Weka Package

The `build.py` script handles building (and optionally installing) the Weka
package (calling Gradle underneath). This is also useful for local tests inside
the Weka GUI. Check out `build.py --help` for more information.


## Run JUnit Tests

Run the Gradle test target:
```bash
$ ./gradlew test
```

## Add/Remove Library Dependencies

Dependencies are managed by Gradle and specified in the `gradle.build` file in
the dependencies block:

```groovy
dependencies {
    ...
}
```

## Update CUDA Versions

If a new version of Deeplearning4j updates its supported CUDA version, the
following files need adjustments:

- `build.gradle`: update `valid_cuda_versions` variable
- `build.py`: update `CUDA_VERSIONS` variable
- `cuda-scripts/install-cuda-libs.sh`: update checks against valid cuda versions
- `cuda-scripts/install-cuda-libs.ps1`: update checks against valid cuda versions

## CUDA Library Installation Scripts

There are four scripts in `cuda-scripts`, namely:

- `install-cuda-libs.sh`
- `uninstall-cuda-libs.sh`
- `install-cuda-libs.ps1`
- `uninstall-cuda-libs.ps1`

The install scripts are supposed to download, extract and move all necessary
CUDA libraries (jar files) into the appropriate WekaDeeplearning4j package installation
directory of the host that is running the script, w.r.t. the host's operating
system. 

The uninstall scripts on the other hand will remove (delete) the CUDA libraries
from the local WekaDeeplearning4j installation. 

Linux and MacOS users should use the bash files while Windows users need the
powershell files.


## Update Java Docs

Java docs reside at https://waikato.github.io/wekaDeeplearning4j.
The documentation is automatically generated (extracted from class/method
documentation in the Java files) and pushed to the `gh-pages` branch via:

```bash
$ ./update-javadocs.sh
```
