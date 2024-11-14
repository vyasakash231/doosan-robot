#!/bin/bash
# Kill any existing DRCF64 process
killall DRCF64 2>/dev/null

# Change to DRCF directory and start DRCF64
cd ~/doosan_ws/src/doosan-robot/common/bin/DRCF
./DRCF64 12345 a0509 &

# Wait for DRCF to start properly
sleep 5
