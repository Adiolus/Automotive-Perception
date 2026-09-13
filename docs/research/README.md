# Automotive Perception Research

This folder contains the literature and dataset research for the camera-first automotive perception project.

## Lane research
- `lane/Lane_Detection_Stitching_Automotive_Perception_Research.pdf` - consolidated paper review, dataset strategy, scenario coverage, BEV, stitching, topology, confidence, ADAS and general automotive perception references.
- `lane/build_lane_research_pdf.py` - reproducible PDF generator.

## Current project direction
```text
CULane
  + CurveLanes
  + BDD100K
  + OpenLane / OpenLane-V2
  + IDD
  ↓
learned lane detector (CLRerNet first benchmark)
  ↓
Indian-domain adaptation
  ↓
BEV / geometric normalization
  ↓
temporal lane stitching
  ↓
persistent lane IDs
  ↓
lane topology + ego-lane relation
  ↓
object-to-lane association
  ↓
fusion / risk
```

## Recommended research position
Do not claim a new lane detector. The defensible contribution is the integrated, camera-first, asynchronous stack combining learned lane detection, Indian-domain adaptation, BEV-normalized temporal lane stitching, persistent lane IDs, topology-aware handling, calibrated lane confidence, and object-to-lane/risk fusion under a real-time constraint.
