#!/bin/bash

# Change to the script's directory
cd "$(dirname "$0")"

jupyter nbconvert --to markdown blog_post.ipynb