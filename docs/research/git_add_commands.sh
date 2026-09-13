#!/bin/bash
set -e
cd ~/automotive-perception
mkdir -p docs/research/lane docs/research/automotive_perception

# Copy the files from the research bundle after downloading/unzipping it.
# Adjust the bundle path if you saved it elsewhere.
BUNDLE="$HOME/Downloads/automotive_perception_research_bundle"

cp "$BUNDLE/docs/research/lane/Lane_Detection_Stitching_Automotive_Perception_Research.pdf" docs/research/lane/
cp "$BUNDLE/docs/research/lane/build_lane_research_pdf.py" docs/research/lane/
cp "$BUNDLE/docs/research/README.md" docs/research/
cp "$BUNDLE/docs/research/automotive_perception/paper_links.md" docs/research/automotive_perception/
cp "$BUNDLE/docs/research/automotive_perception/dataset_links.md" docs/research/automotive_perception/

# Review before committing.
git status --short
git add docs/research/
git commit -m "add automotive perception and lane research"

git status --short --branch
