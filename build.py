#!/usr/bin/env python
import argparse
import os
import subprocess
import sys

CUDA_VERSIONS = ['9.2', '10.0', '10.1', '10.2']
BUILD_PROGRESS = 0.0


class Colors:
    """Shell color codes"""
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printout(out: str = ''):
    """Print normal output"""
    ep = f'{Colors.BOLD}[{Colors.OKGREEN}build.py {BUILD_PROGRESS*100:3.0f}%{Colors.ENDC}{Colors.BOLD}]{Colors.ENDC} '
    for line in out.split('\n'):
        print(f'{ep}{line}')


def printerr(out: str):
    """Print an error in red"""
    ep = f'{Colors.BOLD}[{Colors.OKGREEN}build.py {BUILD_PROGRESS*100:3.0f}%{Colors.ENDC}{Colors.BOLD}]{Colors.ENDC} '
    for line in out.split('\n'):
        print(f'{ep}{Colors.FAIL}{line}{Colors.ENDC}')


def update_progress():
    """Update global build progress"""
    global BUILD_PROGRESS
    steps = get_max_progress_steps()
    BUILD_PROGRESS += (1.0 / steps)
    printout('-' * 64)


def get_max_progress_steps() -> int:
    """Get how many progress steps there are"""
    if opts.build_all:
        # 1 main, 3 cuda, each three times in the build phase
        return 4
    elif opts.cuda_version:  # 1 main, 1 cuda
        return 2
    else:  # 1 main, 1 installation
        return 2


def parse_args() -> argparse.Namespace:
    """Specify build script arguments"""
    parser = argparse.ArgumentParser(
        description='Build the wekaDeeplearning4j packages.')
    parser.add_argument('--cuda-version', '-c', type=str, default=None,
                        help='The cuda version.', choices=CUDA_VERSIONS)
    parser.add_argument('--build-all', '-a', action='store_true',
                        help='Flag to build all platform/cuda packages.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output.')
    return parser.parse_args()


def exec_cmd(cmd: str, print_output=False, exit_on_error=False):
    """Run a given command. If script is in verbose mode then output command"""
    if VERBOSE:
        printout(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    if VERBOSE:
        for line in p.stdout:
            printout(line.decode())

    output, error = p.communicate()


def print_header(opts: argparse.Namespace):
    """Print header about this build script"""
    printout(f'{Colors.HEADER}WekaDeeplearning4j Build Script{Colors.ENDC}')
    printout()
    dct = vars(opts)
    printout(f'{Colors.BOLD}{Colors.UNDERLINE}Parameters:{Colors.ENDC}')
    for k in dct:
        printout(f'     {k:<20} = {dct[k]}')
    printout()


def get_version() -> str:
    """Get the version string"""
    with open('version') as f:
        lines = '\n'.join(f.readlines())
    return lines.replace('\n', '')


def install_main_package():
    """Install the main package in $WEKA_HOME/packages"""
    printout('Installing main package ...')
    user_home = os.environ['HOME']
    weka_home = os.environ.get('WEKA_HOME', f'{user_home}/wekafiles')
    weka_jar = f'{weka_home}/weka.jar'
    if not os.path.isfile(weka_jar):
        printerr(f'{weka_home} did not cointain a weka jar')

    # Install via packagemanager
    exec_cmd(f'rm -r {weka_home}/packages/wekaDeeplearning4j',
             print_output=False,
             exit_on_error=False)

    main_zip = f'wekaDeeplearning4j-{version}.zip'

    exec_cmd(
        f'java -cp {weka_jar} weka.core.WekaPackageManager -install-package dist/{main_zip}')
    update_progress()
    printout('Installation successful')


def build_main_package():
    """Build the main (CPU) package with gradle"""
    update_progress()
    printout('Building main package ...')
    exec_cmd("./gradlew clean makeMain")


def build_cuda_package(cuda_version: str):
    """Build the CUDA package with gradle"""
    update_progress()
    printout(f'Building cuda-{cuda_version} package ...')
    exec_cmd(f"./gradlew makeCuda -Dcuda={cuda_version}")


if __name__ == '__main__':
    # Prepare
    opts = parse_args()
    print_header(opts)
    VERBOSE = opts.verbose
    version = get_version()

    # Clean dist dir first
    exec_cmd('./gradlew cleanDist')

    # Build the main package (CPU, all platforms) in any case
    build_main_package()

    # If opts.cuda_version is set, build the specific cuda package for the
    # current platform
    if opts.cuda_version:
        build_cuda_package(opts.cuda_version)
        sys.exit(0)

    # If opts.build_all flag is set, build all combinations between the cuda
    # versions and the platforms
    if opts.build_all:
        for cuda_version in CUDA_VERSIONS:
            build_cuda_package(cuda_version)
    else:
        install_main_package()
