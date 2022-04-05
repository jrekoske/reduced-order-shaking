#!/bin/sh
INSTALL_DIR=/Users/jrekoske/my-programs/cvmh-15.1.0
${INSTALL_DIR}/bin/vx_lite -s -z elev -m ${INSTALL_DIR}/model < infile-inner > outfile-inner
${INSTALL_DIR}/bin/vx_lite -s -z elev -m ${INSTALL_DIR}/model < infile-outer > outfile-outer