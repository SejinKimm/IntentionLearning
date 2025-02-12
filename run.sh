#!/bin/bash

COMMAND=$1  
REPEAT=$2   

if [ -z "$REPEAT" ]; then
    REPEAT=1
fi

for ((i=1; i<=REPEAT; i++)); do
    eval "$COMMAND"
done