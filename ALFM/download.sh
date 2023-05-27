#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Jeffrey J. Nirschl. All rights reserved.
#
# Licensed under the MIT license. See the LICENSE file in the project
# root directory for full license information.
# ======================================================================
set -euo pipefail

# Base URL for downloads
base_url="https://cloud.woerner.eu/s/4ipEqGTBxwWWMWq/download?path=%2F&files="

# Array of file names
files=(
    aml.zip
    bus.zip
    crc.zip
    cxr.zip
    derm.zip
    dr_regular.zip
    dr_uwf.zip
    fundus.zip
    glaucoma.zip
    mammo_calc.zip
    mammo_mass.zip
    oct.zip
    organs_axial.zip
    organs_coronal.zip
    organs_sagittal.zip
    pbc.zip
    pneumonia.zip
)

# Function to download a file if it and the unzipped directory do not exist
download_file() {
    local file=$1
    local dir_name=$(basename "$file" .zip)

    if [[ ! -f "$file" && ! -d "$dir_name" ]]; then
        echo "Downloading $file ..."
        wget "${base_url}${file}" -O "$file"
    else
        echo "$file or directory $dir_name exists. Skipping download."
    fi
}

# Iterate over the files and download them if they do not exist
for file in "${files[@]}"; do
    download_file "$file"
done
