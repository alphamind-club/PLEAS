#!/bin/bash
# Added by bioplease setup
# Remove any old paths first to avoid duplicates
PATH=$(echo $PATH | tr ':' '\n' | grep -v "bioplease_tools/bin" | tr '\n' ':' | sed 's/:$//')
export PATH="/root/mywork/BioPLEASE/bioplease_env/bioplease_tools/bin:$PATH"
