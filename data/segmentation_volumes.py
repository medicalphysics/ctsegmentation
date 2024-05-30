VOLUMES = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
    92: "rib_left_1",
    93: "rib_left_2",
    94: "rib_left_3",
    95: "rib_left_4",
    96: "rib_left_5",
    97: "rib_left_6",
    98: "rib_left_7",
    99: "rib_left_8",
    100: "rib_left_9",
    101: "rib_left_10",
    102: "rib_left_11",
    103: "rib_left_12",
    104: "rib_right_1",
    105: "rib_right_2",
    106: "rib_right_3",
    107: "rib_right_4",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages",
}

VOLUMES_DXMC = {
    1: (
        "spleen",
        [
            1,
        ],
    ),
    2: (
        "kidney_right",
        [
            2,
        ],
    ),
    3: (
        "kidney_left",
        [
            3,
        ],
    ),
    4: (
        "gallbladder",
        [
            4,
        ],
    ),
    5: (
        "liver",
        [
            5,
        ],
    ),
    6: (
        "stomach",
        [
            6,
        ],
    ),
    7: (
        "pancreas",
        [
            7,
        ],
    ),
    8: (
        "adrenal_gland_right",
        [
            8,
        ],
    ),
    9: (
        "adrenal_gland_left",
        [
            9,
        ],
    ),
    10: ("lung_left", [10, 11]),
    11: ("lung_right", [12, 13, 14]),
    12: (
        "esophagus",
        [
            15,
        ],
    ),
    13: (
        "trachea",
        [
            16,
        ],
    ),
    14: (
        "thyroid_gland",
        [
            17,
        ],
    ),
    15: (
        "small_bowel",
        [
            18,
        ],
    ),
    16: (
        "duodenum",
        [
            19,
        ],
    ),
    17: (
        "colon",
        [
            20,
        ],
    ),
    18: (
        "urinary_bladder",
        [
            21,
        ],
    ),
    19: (
        "prostate",
        [
            22,
        ],
    ),
    20: (
        "kidney_cyst_left",
        [
            23,
        ],
    ),
    21: (
        "kidney_cyst_right",
        [
            24,
        ],
    ),
    22: (
        "sacrum",
        [
            25,
        ],
    ),
    23: (
        "vertebrae_S1",
        [
            26,
        ],
    ),
    24: ("vertebrae_L", [27, 28, 29, 30, 31]),
    25: ("vertebrae_T", [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]),
    26: ("vertebrae_C", [44, 45, 46, 47, 48, 49, 50]),
    27: (
        "heart",
        [
            51,
        ],
    ),
    28: (
        "aorta",
        [
            52,
        ],
    ),
    29: (
        "pulmonary_vein",
        [
            53,
        ],
    ),
    30: (
        "brachiocephalic_trunk",
        [
            54,
        ],
    ),
    31: (
        "subclavian_artery_right",
        [
            55,
        ],
    ),
    32: (
        "subclavian_artery_left",
        [
            56,
        ],
    ),
    33: (
        "common_carotid_artery_right",
        [
            57,
        ],
    ),
    34: (
        "common_carotid_artery_left",
        [
            58,
        ],
    ),
    35: (
        "brachiocephalic_vein_left",
        [
            59,
        ],
    ),
    36: (
        "brachiocephalic_vein_right",
        [
            60,
        ],
    ),
    37: (
        "atrial_appendage_left",
        [
            61,
        ],
    ),
    38: ("vena_cava", [62, 63]),
    39: (
        "portal_vein_and_splenic_vein",
        [
            64,
        ],
    ),
    40: (
        "iliac_artery_left",
        [
            65,
        ],
    ),
    41: (
        "iliac_artery_right",
        [
            66,
        ],
    ),
    42: (
        "iliac_vena_left",
        [
            67,
        ],
    ),
    43: (
        "iliac_vena_right",
        [
            68,
        ],
    ),
    44: (
        "humerus",
        [
            69,
            70,
        ],
    ),
    45: ("scapula", [71, 72]),
    46: (
        "clavicula",
        [
            73,
            74,
        ],
    ),
    47: (
        "femur",
        [
            75,
            76,
        ],
    ),
    48: (
        "hip_left",
        [
            77,
        ],
    ),
    49: (
        "hip_right",
        [
            78,
        ],
    ),
    50: (
        "spinal_cord",
        [
            79,
        ],
    ),
    51: (
        "gluteus_left",
        [
            80,
            82,
            84,
        ],
    ),
    52: (
        "gluteus_right",
        [
            81,
            83,
            85,
        ],
    ),
    53: (
        "autochthon_left",
        [
            86,
        ],
    ),
    54: (
        "autochthon_right",
        [
            87,
        ],
    ),
    55: (
        "iliopsoas_left",
        [
            88,
        ],
    ),
    56: (
        "iliopsoas_right",
        [
            89,
        ],
    ),
    57: (
        "brain",
        [
            90,
        ],
    ),
    58: (
        "skull",
        [
            91,
        ],
    ),
    59: ("rib_left", [92, 93, 94, 95, 96, 97, 98.0, 99, 100, 101, 102, 103]),
    60: ("rib_right", [104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]),
    61: (
        "sternum",
        [
            116,
        ],
    ),
    62: (
        "costal_cartilages",
        [
            117,
        ],
    ),
}


