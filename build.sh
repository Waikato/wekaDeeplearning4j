#!/usr/bin/env bash

# Options
install_pack=false
verbose=false
clean=false
out=/dev/null
PACK=''

# Colors
RED='\e[0;31m'
NC='\e[0m' # Reset attributes
BOLD='\e[1m'
GREEN='\e[32m'


# Module prefix
PREFIX=wekaDeeplearning4j

EP="${BOLD}[${GREEN}${PREFIX} build.sh${NC}${BOLD}]${NC}: "

function show_usage {
    echo -e "Usage: bash.sh"
    echo -e ""
    echo -e "Optional arguments:"
    echo -e "   -v/--verbose            Enable verbose mode"
    echo -e "   -i/--install-packages   Install selected packages"
    echo -e "   -p/--package            Select specific package (default: all)"
    echo -e "                           Available: ( Core CPU GPU NLP )"
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
    PACK="$2"
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
echo -e ${EP}package         = "${PACK}"
echo ""
### END parse arguments ###

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

# Available modules
if [[ ${PACK} = '' ]]; then
    packages=( "Core" "CPU" "GPU" "NLP")
elif [[ ${PACK} = "Core" || ${PACK} = "CPU" || ${PACK} = "GPU" || ${PACK} = "NLP" ]]; then
    packages=( ${PACK} )
else
    echo -e "${EP}${RED}Invalid package. Exiting now...${NC}"
    exit 1
fi
# Clean up lib folders and classes
if [[ "$clean" = true ]]; then
    echo -e "${EP}Cleaning up lib in each package..."
    for sub in "${packages[@]}"
    do
        pack=${PREFIX}${sub}
        rm ${pack}/lib/*
        mvn clean > /dev/null # don't clutter with mvn clean output
    done
fi

# Compile source code with maven
echo -e "${EP}Pulling dependencies via maven..."
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
        echo -e "${EP}Installing ${pack} package..."
        java -cp ${CLASSPATH} weka.core.WekaPackageManager -install-package ${pack}/dist/${pack}.zip
    fi
}



for pack in "${packages[@]}"
do
    echo -e "${EP}Starting ant build for ${BOLD}"${pack}${NC}
    build_package ${pack}
done
