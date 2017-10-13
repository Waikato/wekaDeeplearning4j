#!/usr/bin/env bash

# Options
install_pack=false
verbose=false
clean=false
out=/dev/null
BACKEND=''

# Colors
RED='\e[0;31m'
NC='\e[0m' # Reset attributes
BOLD='\e[1m'
GREEN='\e[32m'


# Module prefix
PREFIX=wekaDeeplearning4j

cd ${PREFIX}Core

EP="${BOLD}[${GREEN}${PREFIX} build.sh${NC}${BOLD}]${NC}: "

function show_usage {
    echo -e "Usage: build.sh"
    echo -e ""
    echo -e "Optional arguments:"
    echo -e "   -v/--verbose            Enable verbose mode"
    echo -e "   -i/--install-packages   Install selected packages"
    echo -e "   -b/--backend            Select specific backend "
    echo -e "                           Available: ( CPU GPU )"
    echo -e "   -c/--clean              Clean up build-environment"
    echo -e "   -h/--help               Show this message"
    exit 0
}

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
    -p|--package)
    BACKEND="$2"
    shift # past argument
    shift # past value
    ;;
    -c|--clean)
    clean=true
    shift # past argument
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

echo -e ${EP}"Parameters:"
echo -e ${EP}verbose       = "${verbose}"
echo -e ${EP}install_pack  = "${install_pack}"
echo -e ${EP}clean         = "${clean}"
echo -e ${EP}package         = "${BACKEND}"
echo ""
### END parse arguments ###

if [[ ${BACKEND} != 'CPU' && ${BACKEND} != 'GPU' ]]; then
    echo -e "${EP}${RED}Selected package must be either CPU or GPU!" > /dev/stderr
    echo -e "${EP}Exiting now...${NC}" > /dev/stderr
    exit 1
fi

# If verbose redirect to stdout, else /dev/null
if [[ "$verbose" = true ]]; then
    out=/dev/stdout
fi

# Check if env var is set and weka.jar could be found
if [[ -z "$WEKA_HOME" ]]; then
    echo -e "${EP}${RED}WEKA_HOME env variable is not set!" > /dev/stderr
    echo -e "${EP}Exiting now...${NC}" > /dev/stderr
    exit 1
elif [[ ! -e "$WEKA_HOME/weka.jar" ]]; then
    echo -e "${EP}${RED}WEKA_HOME=${WEKA_HOME} does not contain weka.jar!" > /dev/stderr
    echo -e "${EP}Exiting now...${NC}" > /dev/stderr
    exit 1
fi

export CLASSPATH=${WEKA_HOME}/weka.jar
echo -e "${EP}Classpath = " ${CLASSPATH}

BASE=${PREFIX}Core
PACKAGE_NAME=${PREFIX}${BACKEND}"-dev"
# Clean up lib folders and classes
if [[ "$clean" = true ]]; then
    rm lib/*
    mvn clean > /dev/null # don't clutter with mvn clean output
fi

# Compile source code with maven
echo -e "${EP}Pulling dependencies via maven..."
mvn -DskipTests=true -P ${BACKEND} install >  "$out"

echo -e "${EP}Starting ant build for ${BOLD}"${BASE}"-dev"${NC}

# Clean-up
ant -f build_package_${BACKEND}.xml clean > /dev/null # don't clutter with ant clean output

# Build the package
ant -f build_package_${BACKEND}.xml make_package -Dpackage=${PACKAGE_NAME} > "$out"

# Install package from dist dir
if [[ "$install_pack" = true ]]; then
    # Remove up old packages
    if [[ "$clean" = true ]]; then
        rm -r ${WEKA_HOME}/packages/${PREFIX}"CPU-dev"
        rm -r ${WEKA_HOME}/packages/${PREFIX}"GPU-dev"
    fi
    echo -e "${EP}Installing ${PACKAGE_NAME} package..."
    java -cp ${CLASSPATH} weka.core.WekaPackageManager -install-package dist/${PACKAGE_NAME}.zip
fi

# Go back
cd ..
