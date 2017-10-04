#!/usr/bin/env bash

# Options
install_pack=false
verbose=false
clean=false
out=/dev/null

# Colors
RED='\033[0;31m'
NC='\033[0m' # No Color

# Module prefix
PREFIX=wekaDeeplearning4j

### BEGIN parse arguments ###
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -v|--verbose)
    verbose=true
    shift # past argument
    ;;
    -i|--install-packages)
    install_pack=true
    shift # past argument
    ;;
    -c|--clean)
    clean=true
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo "Parameters:"
echo verbose       = "${verbose}"
echo install_pack  = "${install_pack}"
echo clean         = "${clean}"
echo ""
### END parse arguments ###

# If verbose redirect to stdout, else /dev/null
if [[ "$verbose" = true ]]; then
    out=/dev/stdout
fi

# Check if env var is set and weka.jar could be found
if [[ -z "$WEKA_HOME" ]]; then
    echo -e "${RED}WEKA_HOME env variable is not set!" > /dev/stderr
    echo -e "Exiting now...${NC}" > /dev/stderr
    exit 1
elif [[ ! -e "$WEKA_HOME/weka.jar" ]]; then
    echo -e "${RED}WEKA_HOME=${WEKA_HOME} does not contain weka.jar!" > /dev/stderr
    echo -e "Exiting now...${NC}" > /dev/stderr
    exit 1
fi

export CLASSPATH=${WEKA_HOME}/weka.jar
echo "Classpath = " ${CLASSPATH}

# Available modules
modules=( "Core" "CPU" "GPU" )

# Clean up lib folders and classes
if [[ "$clean" = true ]]; then
    echo "Cleaning up lib-full and lib in each module..."
    for sub in "${modules[@]}"
    do
        dir=${PREFIX}${sub}
        rm ${dir}/lib-full/*
        rm ${dir}/lib/*
        mvn clean > /dev/null # don't clutter with mvn clean output
    done
fi

# Compile source code with maven
echo "Compiling the source code now..."
mvn -DskipTests=true install >  "$out"



function build_module {
    dir=${PREFIX}$1

    # Clean-up
    ant -f ${dir}/build_package.xml clean > /dev/null # don't clutter with ant clean output

    if [[ $1 == "Core" ]]; then
        # Strip that list down by looking up which jars are unnecessary
        jars=(
        "deeplearning"
        "common"
        "datavec"
        "guava"
        "imageio"
        "jackson"
        "jai-imageio"
        "javassist"
        "lombok"
        "nd4j"
        "opencv"
        "reflections"
        "slf4j"
        )
    elif [[ $1 == "CPU" ]]; then
        jars=(
        "openblas" #should already be in core?
        )
    elif [[ $1 == "GPU" ]]; then
        jars=(
        "cuda"
        "nd4j-cuda"
        )
    fi

    # Copy the libs from lib-full to lib
    for name in "${jars[@]}"
    do
        cp ${dir}/lib-full/${name}*.jar ${dir}/lib/
    done

    # Build the package
    ant -f ${dir}/build_package.xml make_package -Dpackage=${dir} > "$out"

    # Install package from dist dir
    if [[ "$install_pack" = true ]]; then
        echo "Installing ${dir} package..."
        java -cp ${CLASSPATH} weka.core.WekaPackageManager -install-package ${dir}/dist/${dir}.zip
    fi
}



for mod in "${modules[@]}"
do
    echo "Starting ant build for" ${mod}
    build_module ${mod}
done
