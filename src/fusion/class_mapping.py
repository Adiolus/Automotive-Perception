"""
IDD Segmentation class mapping.

These IDs correspond to the raw IDD label IDs verified from
public-code/helpers/anue_labels.py and the training-mask inspection.
"""

CLASS_NAMES = {
    0: "road",
    1: "parking",
    2: "drivable fallback",
    3: "sidewalk",
    4: "rail track",
    5: "non-drivable fallback",
    6: "person",
    7: "animal",
    8: "rider",
    9: "motorcycle",
    10: "bicycle",
    11: "autorickshaw",
    12: "car",
    13: "truck",
    14: "bus",
    15: "caravan",
    16: "trailer",
    17: "train",
    18: "vehicle fallback",
    19: "curb",
    20: "wall",
    21: "fence",
    22: "guard rail",
    23: "billboard",
    24: "traffic sign",
    25: "traffic light",
    26: "pole",
    27: "polegroup",
    28: "obs-str-bar-fallback",
    29: "building",
    30: "bridge",
    31: "tunnel",
    32: "vegetation",
    33: "sky",
    34: "fallback background",
    35: "unlabeled",
    36: "ego vehicle",
    37: "rectification border",
    38: "out of roi",
    39: "license plate",
    40: "background",
}

CLASS_IDS = {name: idx for idx, name in CLASS_NAMES.items()}

# Classes that are useful for the first ADAS fusion stage.
ROAD_CLASSES = {
    CLASS_IDS["road"],
    CLASS_IDS["drivable fallback"],
}

TRACKABLE_CLASSES = {
    CLASS_IDS["person"],
    CLASS_IDS["rider"],
    CLASS_IDS["motorcycle"],
    CLASS_IDS["bicycle"],
    CLASS_IDS["autorickshaw"],
    CLASS_IDS["car"],
    CLASS_IDS["truck"],
    CLASS_IDS["bus"],
    CLASS_IDS["caravan"],
    CLASS_IDS["trailer"],
    CLASS_IDS["train"],
    CLASS_IDS["vehicle fallback"],
}


def class_name(class_id: int) -> str:
    """Convert a segmentation class ID to its class name."""
    return CLASS_NAMES.get(int(class_id), f"unknown_{class_id}")


def class_id(class_name_value: str) -> int:
    """Convert a class name to its segmentation class ID."""
    return CLASS_IDS[class_name_value]
