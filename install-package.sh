#!/usr/bin/env bash
set -e
#set -x

cuda_version=""
version=$(cat version)


function show_usage {
    echo -e "Usage: build.sh"
    echo -e ""
    echo -e "Optional arguments:"
    echo -e "   -c/--cuda-version       Select specific backend "
    echo -e "                           Available: ( 8.0 9.0 9.1 )"
    echo -e "   -h/--help               Show this message"
    exit 0
}

### BEGIN parse arguments ###
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -c|--cuda-version)
    cuda_version="$2"
    shift # past argument
    shift # past value
    ;;
    -h|--help)
    show_usage
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# Remove parameters: Build all packages by default, vebose by default, do not install any packages
echo -e "Parameters:"
echo -e "      cuda-version       = ${cuda_version}"
echo -e ""
### END parse arguments ###

# Get platform
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     platform=linux;;
    Darwin*)    platform=macosx;;
    CYGWIN*)    platform=windows;;
    MINGW*)     platform=windows;;
    *)          platform="UNKNOWN:${unameOut}"
esac

# Check if env var is set and weka.jar could be found
if [[ -z "$WEKA_HOME" ]]; then
    echo -e "WEKA_HOME env variable is not set!" > /dev/stderr
    echo -e "Exiting now..." > /dev/stderr
    exit 1
elif [[ ! -e "$WEKA_HOME/weka.jar" ]]; then
    echo -e "WEKA_HOME=${WEKA_HOME} does not contain weka.jar!" > /dev/stderr
    echo -e "Exiting now..." > /dev/stderr
    exit 1
fi

export CLASSPATH=${WEKA_HOME}/weka.jar
echo -e "Classpath = " ${CLASSPATH}

# Install package from dist dir
rm -r ${WEKA_HOME}/packages/*wekaDeeplearning4j* || true
main_zip_name=wekaDeeplearning4j-${version}.zip
echo -e "Installing package..."
java -cp ${CLASSPATH} weka.core.WekaPackageManager -install-package dist/${main_zip_name}

if [[ ! -z ${cuda_version} ]]; then
  cuda_zip_name=wekaDeeplearning4j-cuda-${cuda_version}-${version}-${platform}-x86_64.zip
  cd ..
  ./cuda-scripts/install-cuda-libs.sh package/dist/$cuda_zip_name
fi

if [ $? -eq 0 ]; then
    echo -e "Installation successful"
else
    echo -e "Installation failed"
fi

cd ..
