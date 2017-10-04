#!/usr/bin/env bash

install_pack=false
verbose=false
clean=false
out=/dev/null

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


if [[ "$install_pack" = true ]]; then
	if [[ -z "$WEKA_HOME" ]]; then
    	echo "WEKA_HOME env variable is not set!"
    	echo "Exiting now..."
   		exit 1
	elif [[ ! -e "$WEKA_HOME/weka.jar" ]]; then
	    echo "${WEKA_HOME} does not contain weka.jar!"
    	echo "Exiting now..."
   		exit 1
	fi

	export CLASSPATH=${WEKA_HOME}/weka.jar
	echo "Classpath: " ${CLASSPATH}
fi

arr=( "Core" "CPU" "GPU" )
echo "Cleaning up lib-full and lib in each module..."
for sub in "${arr[@]}"
do
    dir=wekaDeeplearning4j${sub}
    if [[ "$clean" = true ]]; then
        rm ${dir}/lib-full/*
        rm ${dir}/lib/*

        mvn clean > "$out"
    fi
done

echo "Compiling the source code now..."
mvn -DskipTests=true install >  "$out"


function build_ant {
    dir=wekaDeeplearning4j$1

    # clean-up
    ant -f ${dir}/build_package.xml clean > "$out"

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

        for name in "${jars[@]}"
        do
            cp ${dir}/lib-full/${name}*.jar ${dir}/lib/
        done

    elif [[ $1 == "CPU" ]]; then
        jars=(
        "openblas" #should already be in core?
        )

        for name in "${jars[@]}"
        do
            cp ${dir}/lib-full/${name}*.jar ${dir}/lib/
        done
    elif [[ $1 == "GPU" ]]; then
        jars=(
        "cuda"
        "nd4j-cuda"
        )

        for name in "${jars[@]}"
        do
            cp ${dir}/lib-full/${name}*.jar ${dir}/lib/
        done
    fi

    # build the package
    ant -f ${dir}/build_package.xml make_package -Dpackage=${dir} > "$out"

    if [[ "$install_pack" = true ]]; then
        echo "Installing ${dir} package..."
        java -cp ${CLASSPATH} weka.core.WekaPackageManager -install-package ${dir}/dist/${dir}.zip
    fi
}



for sub in "${arr[@]}"
do
    echo "Starting ant build for" ${sub}
    build_ant ${sub}
done
