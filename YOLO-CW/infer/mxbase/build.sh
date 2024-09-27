#!/bin/bash
path_cur=$(dirname $0)

function check_env()
{
    # set ASCEND_VERSION to ascend-toolkit/latest when it was not specified by user
    if [ ! "${ASCEND_HOME}" ]; then
        export ASCEND_HOME=/usr/local/Ascend/
        echo "Set ASCEND_HOME to the default value: ${ASCEND_HOME}"
    else
        echo "ASCEND_HOME is set to ${ASCEND_HOME} by user"
    fi

    if [ ! "${ASCEND_VERSION}" ]; then
        export ASCEND_VERSION=nnrt/latest
        echo "Set ASCEND_VERSION to the default value: ${ASCEND_VERSION}"
    else
        echo "ASCEND_VERSION is set to ${ASCEND_VERSION} by user"
    fi

    if [ ! "${ARCH_PATTERN}" ]; then
        # set ARCH_PATTERN to ./ when it was not specified by user
        export ARCH_PATTERN=./
        echo "ARCH_PATTERN is set to the default value: ${ARCH_PATTERN}"
    else
        echo "ARCH_PATTERN is set to ${ARCH_PATTERN} by user"
    fi
}

function build_east()
{
    cd $path_cur
    rm -rf build
    mkdir -p build
    cd build
    cmake ..
    make
    ret=$?
    if [ ${ret} -ne 0 ]; then
        echo "Failed to build east."
        exit ${ret}
    fi
    make install
}

check_env
build_east
