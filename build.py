#!/usr/bin/env python3.6
import argparse
import os
import shutil
import subprocess
import sys
from collections import Iterable
from datetime import datetime
from distutils.dir_util import copy_tree
from glob import glob
from typing import List

CUDA_VERSIONS = ['8.0', '9.0', '9.1']
PLATFORMS = ['linux', 'macosx', 'windows']
PROGRESS = 0.0


class Colors:
    """Shell color codes"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printout(out: str = ''):
    ep = f'{Colors.BOLD}[{Colors.OKGREEN}build.py {PROGRESS*100:3.0f}%{Colors.ENDC}{Colors.BOLD}]{Colors.ENDC} '
    for line in out.split('\n'):
        print(f'{ep}{line}')


def printerr(out: str):
    ep = f'{Colors.BOLD}[{Colors.OKGREEN}build.py {Colors.ENDC}{Colors.BOLD}]{Colors.ENDC} '
    for line in out.split('\n'):
        print(f'{ep}{Colors.FAIL}{line}{Colors.ENDC}')


def update_progress():
    global PROGRESS
    steps = get_max_progress_steps()
    PROGRESS = PROGRESS + (1.0 / steps)


def get_max_progress_steps() -> int:
    if opts.build_all:
        # 1 main, 3x3 cuda, each three times in the build phase
        return 3 * (1 + 9)
    elif opts.cuda_version:
        return 6 * 1
    else:
        return 4 * 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build the wekaDeeplearning4j packages.')
    parser.add_argument('--cuda-version', '-c', type=str, default=None,
                        help='The cuda version.', choices=CUDA_VERSIONS)
    parser.add_argument('--build-all', '-a', action='store_true',
                        help='Flag to build all platform/cuda packages.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output.')
    return parser.parse_args()


def get_platform() -> str:
    out = exec_cmd('uname -s')

    if 'Linux' in out:
        return 'linux'
    elif 'Darwin' in out:
        return 'macosx'
    elif 'CYGWIN' in out or 'MINGW' in out:
        return 'windows'
    else:
        return f'UNKNOWN:{out}'


def get_os_name(platform: str) -> str:
    return dict(linux='Linux', macosx='Mac', windows='Windows').get(platform)


def get_version() -> str:
    return get_file_content('version').replace('\n', '')


def get_file_content(path: str) -> List[str]:
    with open(path) as f:
        lines = '\n'.join(f.readlines())
        return lines


def rm_dir(path: str):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)


def exec_cmd(cmd: str, print_output=False, exit_on_error=True) -> str:
    if VERBOSE:
        printout(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    output, error = p.communicate()

    if error:
        if print_output or exit_on_error:
            printerr(error.decode('utf-8'))

        if exit_on_error:
            sys.exit(1)

    output_str = output.decode('utf-8')
    if print_output:
        if len(output_str) > 0:
            for line in output_str.split('\n'):
                printout(line)
    return output_str


def clean_dirs():
    rm_dir('lib')
    exec_cmd('mvn -q clean', exit_on_error=False)


def edit_description_props(
    package_name: str,
    build_suffix: str,
    zip_name: str,
    platform: str,
    os_name: str,
    precludes: str,
):
    # Read Description.props
    with open(f'dist/{package_name}/Description.props') as f:
        lines = ''.join(f.readlines())

    date = '{:%Y-%-m-%d}'.format(datetime.now())
    lines = lines.replace('{DATE}', date)
    lines = lines.replace('{BUILD_SUFFIX}', build_suffix)
    lines = lines.replace('{VERSION}', version)
    lines = lines.replace('{ZIP_NAME}', zip_name)
    lines = lines.replace('{PLATFORM}', platform)
    lines = lines.replace('{OS_NAME}', os_name)
    # TODO lines.replace('{PRECLUDES}', precludes)

    # Write Description.props
    with open(f'dist/{package_name}/Description.props', 'w') as f:
        f.write(lines)


def build_main_package():
    printout(
        f'Building main package for {Colors.BOLD}CPU{Colors.ENDC} (all platforms)')
    clean_dirs()
    printout('Pulling dependencies')
    cmd = 'mvn -q -Dmaven.javadoc.skip=true -DskipTests=true dependency:resolve process-sources'
    exec_cmd(cmd)
    update_progress()

    os.makedirs(f'dist/{package_main_name}/lib', exist_ok=True)

    excludes = [
        '*cuda*',
        '*.pom',
        "*android-x86*",
        "*android-arm*",
        "*ios-arm*",
        "*ios-x86*",
        "*x86.jar",
        "*linux-ppc64le*.jar",
        "*linux-armhf*.jar"
        "*linux-i686*.jar"
        "*win-i686*.jar"
    ]

    copy_files_with_exclusions(src='lib/', dst=f'dist/{package_main_name}/lib',
                               excludes=excludes)
    update_progress()

    dst = f'dist/{package_main_name}/'
    shutil.copy2('Description.props', dst)
    shutil.copy2('GenericPropertiesCreator.props', dst)
    shutil.copy2('GUIEditors.props', dst)
    copy_tree('datasets', f'{dst}/datasets/')

    zip_name = f'{basename}-{version}.zip'
    precludes = 'TODO'
    platform_name = ""
    build_suffix = ""
    package_dependencies = ""

    edit_description_props(
        package_name=package_main_name,
        build_suffix=build_suffix,
        zip_name=zip_name,
        platform=platform_name,
        os_name='',
        precludes=precludes
    )

    printout(f'Building {zip_name} with ant')
    cmd = f'ant -f build_package.xml make_package_main \
      -Dpackage_name="{package_main_name}" \
      -Dzip_name="{zip_name}"'
    exec_cmd(cmd)
    update_progress()

    printout('Successfully finished building main package')


def build_cuda_package(cuda_version: str, platform: str):
    printout(
        f'Building package for {Colors.BOLD}CUDA {cuda_version} ({platform}){Colors.ENDC}')
    clean_dirs()
    platform_exclude_1, platform_exclude_2 = get_inverted_platform_selection(
        platform)
    package_name = f'{basename}-cuda-{cuda_version}-{version}-{platform}'

    printout('Pulling dependencies')
    mvn_get_dependencies_cmd = f'mvn -q -Dmaven.javadoc.skip=true -DskipTests=true dependency:resolve process-sources -P cuda-{cuda_version}'
    exec_cmd(mvn_get_dependencies_cmd)
    update_progress()

    os.makedirs(f'dist/{package_name}/lib/', exist_ok=True)
    src = 'lib/'
    dst = f'dist/{package_name}/'

    excludes = [f'*{platform_exclude_1}*',
                f'*{platform_exclude_2}*',
                '*android-x86*',
                '*android-arm*',
                '*ios-arm*',
                '*ios-x86*',
                '*x86.jar',
                "*linux-ppc64le*.jar",
                "*linux-armhf*.jar"
                "*linux-i686*.jar"
                "*win-i686*.jar"]
    includes = ['*cuda-*']
    copy_files_with_inclusions_and_exclusions(src, f'{dst}/lib',
                                              includes=includes,
                                              excludes=excludes)
    update_progress()

    zip_name = f'{basename}-cuda-{cuda_version}-{version}-{platform}-x86_64.zip'
    printout(f'Building {zip_name} with ant')
    ant_build_pkg_cmd = f'ant -f build_package.xml make_package_cuda \
      -Dpackage_name="{package_name}" \
      -Dzip_name="{zip_name}"'
    exec_cmd(ant_build_pkg_cmd)
    update_progress()

    printout(f'Successfully finished building CUDA {cuda_version} ({platform})')


def ensure_dir_ending(path: str) -> str:
    if not path.endswith('/'):
        path += '/'
    return path


def copy_files_with_inclusions_and_exclusions(src: str, dst: str,
    excludes: Iterable,
    includes: Iterable):
    src = ensure_dir_ending(src)
    dst = ensure_dir_ending(dst)

    files = [f'lib/{f}' for f in os.listdir(src)]
    filtered_files = []
    for file in files:
        # Check if any exclude pattern matches the current file
        exclude_pattern_matched = False
        for ex_pat in excludes:
            matches = glob(src + ex_pat)
            if file in matches:
                exclude_pattern_matched = True

        if not exclude_pattern_matched:
            filtered_files.append(file)

    for file in filtered_files:
        for inc_pat in includes:
            # Check if any include pattern matches the current file
            matches = glob(src + inc_pat)
            if file in matches:
                if VERBOSE:
                    printout(f'cp {file} -> {dst}')
                shutil.copy2(file, dst)
                break


def copy_files_with_exclusions(src: str, dst: str, excludes: Iterable):
    src = ensure_dir_ending(src)
    dst = ensure_dir_ending(dst)

    files = [f'lib/{f}' for f in os.listdir(src)]
    for file in files:

        # Check if any exclude pattern matches the current file
        exclude_pattern_matched = False
        for ex_pat in excludes:
            matches = glob(src + ex_pat)
            if file in matches:
                exclude_pattern_matched = True
                break

        if exclude_pattern_matched:
            continue

        if VERBOSE:
            printout(f'cp {file} -> {dst}')

        shutil.copy2(file, dst)


def get_inverted_platform_selection(platform: str) -> List[str]:
    platforms = ['linux', 'windows', 'macosx']
    platforms.remove(platform.lower())
    return platforms


def print_opts(opts: argparse.Namespace):
    dct = vars(opts)
    printout(f'{Colors.BOLD}{Colors.UNDERLINE}Parameters:{Colors.ENDC}')
    for k in dct:
        printout(f'     {k:<20} = {dct[k]}')
    printout()


def install_main_package():
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

    printout('Installing package')
    exec_cmd(
        f'java -cp {weka_jar} weka.core.WekaPackageManager -install-package dist/{main_zip}',
        print_output=False,
        exit_on_error=False)
    update_progress()
    printout('Installation successful')


if __name__ == '__main__':
    opts = parse_args()
    print_opts(opts)
    VERBOSE = opts.verbose
    basename = 'wekaDeeplearning4j'
    version = get_version()
    package_main_name = f'{basename}-{version}'

    exec_cmd('ant -f build_package.xml clean', exit_on_error=False)
    clean_dirs()

    # Build the main package (CPU, all platforms) in any case
    build_main_package()

    # If opts.cuda_version is set, build the specific cuda package for the
    # current platform
    if opts.cuda_version:
        current_platform = get_platform()
        build_cuda_package(opts.cuda_version, current_platform)
        sys.exit(0)

    # If opts.build_all flag is set, build all combinations between the cuda
    # versions and the platforms
    separator = '-' * 60
    printout(separator)
    if opts.build_all:
        for cuda_version in CUDA_VERSIONS:
            for platform in PLATFORMS:
                build_cuda_package(cuda_version, platform)
                printout(separator)
        sys.exit(0)
    else:
        install_main_package()
