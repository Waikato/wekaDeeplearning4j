#!/usr/bin/env bash

# Options
install_pack=false
verbose=false
clean=false
out=/dev/null
backend=''

# Colors
red='\e[0;31m'
nc='\e[0m' # Reset attributes
bold='\e[1m'
green='\e[32m'


# Module prefix
prefix=wekaDeeplearning4j


ep="${bold}[${green}${prefix} build.sh${nc}${bold}]${nc}: "

### Check for color support ###
# check if stdout is a terminal...
if test -t 1; then

    # see if it supports colors...
    ncolors=$(tput colors)

    if test -n "$ncolors" && test $ncolors -ge 8; then #Enable colors
        ep="${bold}[${green}${prefix} build.sh${nc}${bold}]${nc}: "
    else #Disable colors
        ep="[${prefix} build.sh]: "
        bold=""
        nc=""
    fi
fi

# Project version (TODO: Fix for non GNU grep versions)
version=`grep -Po 'name="version" value="\K([0-9]+\.[0-9]+\.[0-9]+(-(alpha|beta))?)(?=")' package/build_package.xml`
if echo ${version} | grep -Eq "^[0-9]+\.[0-9]+\.[0-9]+(-(alpha|beta))?$"; then
    echo -e "${ep}Building version: ${version}"
else
    echo -e "${ep}Error finding version. Unknown version: ${version}"
    echo -e "${ep}Exiting now."
    exit 1
fi

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
    -b|--backend)
    backend="$2"
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

echo -e "${ep}Parameters:"
echo -e "${ep}      verbose       = ${verbose}"
echo -e "${ep}      install_pack  = ${install_pack}"
echo -e "${ep}      clean         = ${clean}"
echo -e "${ep}      package       = ${backend}"
echo -e "${ep}"
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


if [[ ${backend} != 'CPU' && ${backend} != 'GPU' ]]; then
    echo -e "${ep}${red}Selected package must be either CPU or GPU!" > /dev/stderr
    echo -e "${ep}Exiting now...${nc}" > /dev/stderr
    exit 1
fi

# If verbose redirect to stdout, else /dev/null
if [[ "$verbose" = true ]]; then
    out=/dev/stdout
fi

# Check if env var is set and weka.jar could be found
if [[ -z "$WEKA_HOME" ]]; then
    echo -e "${ep}${red}WEKA_HOME env variable is not set!" > /dev/stderr
    echo -e "${ep}Exiting now...${nc}" > /dev/stderr
    exit 1
elif [[ ! -e "$WEKA_HOME/weka.jar" ]]; then
    echo -e "${ep}${red}WEKA_HOME=${WEKA_HOME} does not contain weka.jar!" > /dev/stderr
    echo -e "${ep}Exiting now...${nc}" > /dev/stderr
    exit 1
fi

export CLASSPATH=${WEKA_HOME}/weka.jar
echo -e "${ep}Classpath = " ${CLASSPATH}


base="package"
cd ${base}


pack_name=${prefix}-${backend}-${platform}
zip_name=${prefix}-${backend}-${version}-${platform}-x86_64.zip
# Clean up lib folders and classes
if [[ "$clean" = true ]]; then
    [[ -d lib ]] && rm lib/* &> ${out}
    mvn -q clean > /dev/null # don't clutter with mvn clean output
fi

# Compile source code with maven
echo -e "${ep}Pulling dependencies via maven..."
mvn -q -DskipTests=true -P ${backend} install >  ${out}

echo -e "${ep}Starting ant build for ${bold}"${base}${nc}

# Clean-up
ant -f build_package.xml clean > /dev/null # don't clutter with ant clean output

# Build the package
ant -f build_package.xml make_package_${backend} > ${out}

# Install package from dist dir
if [[ "$install_pack" = true ]]; then
    # Remove up old packages
    if [[ "$clean" = true ]]; then
        [[ -d ${WEKA_HOME}/packages/${prefix}-CPU-${platform} ]] && rm -r ${WEKA_HOME}/packages/${prefix}-CPU-${platform} &> ${out}
        [[ -d ${WEKA_HOME}/packages/${prefix}-GPU-${platform} ]] && rm -r ${WEKA_HOME}/packages/${prefix}-GPU-${platform} &> ${out}
    fi
    echo -e "${ep}Installing ${pack_name} package..."
    java -cp ${CLASSPATH} weka.core.WekaPackageManager -install-package dist/${zip_name} > ${out}
    if [ $? -eq 0 ]; then
        echo -e "${ep}Installation successful"
    else
        echo -e "${ep}Installation failed"
    fi
fi

# Go back
cd ..
