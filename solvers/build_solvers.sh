#!/usr/bin/env bash
# Download and compile the two Secomb-lab solvers at the exact versions used here.
# Tested on Linux with g++ 13. Run from the repository root:  bash solvers/build_solvers.sh
# On Windows, use WSL, or build the same commits with Visual Studio.
set -euo pipefail
cd "$(dirname "$0")"

NETFLOW_COMMIT=9c83a5ebe61b734a92c122fedcf3825649cc2a76   # NetFlowV2, 2019-08-05
GREENS_COMMIT=c5177527e5b3a5762cb9a3bb3afcf70a7a8d7a02    # GreensV4, 2019-07-11

fetch () {   # repo commit
  if [ ! -d "$1" ]; then git clone -q "https://github.com/secomb/$1.git"; fi
  git -C "$1" checkout -q "$2"
}

fetch NetFlowV2 "$NETFLOW_COMMIT"
g++ -O2 -w -std=c++17 -o netflow NetFlowV2/*.cpp -lstdc++fs
echo "built solvers/netflow"

# GreensV4 compiles the tissue consumption law in from tissrate.cpp.dat.
# One-line patch: GreensV4 returns NaN for a cell-free perfusate (haematocrit 0),
# because its first guess divides by the haematocrit-weighted vessel length.
# See solvers/patches/greensv4_cell_free_perfusate.patch.
fetch GreensV4 "$GREENS_COMMIT"
git -C GreensV4 checkout -q -- .
git -C GreensV4 apply ../patches/greensv4_cell_free_perfusate.patch
cp ../configs/greensv4/tissrate.cpp.dat GreensV4/tissrate.cpp.dat
g++ -O3 -w -std=c++17 -o greens GreensV4/*.cpp -lstdc++fs
echo "built solvers/greens"
