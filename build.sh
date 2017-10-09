#!/usr/bin/env bash

# Options
install_pack=false
verbose=false
clean=false
out=/dev/null

# Colors
RED='\e[0;31m'
NC='\e[0m' # Reset attributes
BOLD='\e[1m'
GREEN='\e[32m'


# Module prefix
PREFIX=wekaDeeplearning4j

ECHO_PREFIX="${BOLD}[${GREEN}${PREFIX} build.sh${NC}${BOLD}]${NC}: "

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

echo -e ${ECHO_PREFIX}"Parameters:"
echo -e ${ECHO_PREFIX}verbose       = "${verbose}"
echo -e ${ECHO_PREFIX}install_pack  = "${install_pack}"
echo -e ${ECHO_PREFIX}clean         = "${clean}"
echo ""
### END parse arguments ###

# If verbose redirect to stdout, else /dev/null
if [[ "$verbose" = true ]]; then
    out=/dev/stdout
fi

# Check if env var is set and weka.jar could be found
if [[ -z "$WEKA_HOME" ]]; then
    echo -e "${ECHO_PREFIX}${RED}WEKA_HOME env variable is not set!" > /dev/stderr
    echo -e "${ECHO_PREFIX}Exiting now...${NC}" > /dev/stderr
    exit 1
elif [[ ! -e "$WEKA_HOME/weka.jar" ]]; then
    echo -e "$${ECHO_PREFIX}{RED}WEKA_HOME=${WEKA_HOME} does not contain weka.jar!" > /dev/stderr
    echo -e "${ECHO_PREFIX}Exiting now...${NC}" > /dev/stderr
    exit 1
fi

export CLASSPATH=${WEKA_HOME}/weka.jar
echo -e "${ECHO_PREFIX}Classpath = " ${CLASSPATH}

# Available modules
packages=( "Core" "CPU" "GPU" "NLP")

# Clean up lib folders and classes
if [[ "$clean" = true ]]; then
    echo -e "${ECHO_PREFIX}Cleaning up lib in each package..."
    for sub in "${packages[@]}"
    do
        pack=${PREFIX}${sub}
        rm ${pack}/lib/*
        mvn clean > /dev/null # don't clutter with mvn clean output
    done
fi

# Compile source code with maven
echo -e "${ECHO_PREFIX}Pulling dependencies via maven..."
mvn -DskipTests=true install >  "$out"



function build_package {
    pack=${PREFIX}$1

    # Clean-up
    ant -f ${pack}/build_package.xml clean > /dev/null # don't clutter with ant clean output

    # Build the package
    ant -f ${pack}/build_package.xml make_package -Dpackage=${pack} > "$out"

    # Install package from dist dir
    if [[ "$install_pack" = true ]]; then
        # Remove up old packages
        if [[ "$clean" = true ]]; then
            rm -r ${WEKA_HOME}/packages/${pack}
        fi
        echo -e "${ECHO_PREFIX}Installing ${pack} package..."
        java -cp ${CLASSPATH} weka.core.WekaPackageManager -install-package ${pack}/dist/${pack}.zip
    fi
}



for pack in "${packages[@]}"
do
    echo -e "${ECHO_PREFIX}Starting ant build for ${BOLD}"${pack}${NC}
    build_package ${pack}
done
