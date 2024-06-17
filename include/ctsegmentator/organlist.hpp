
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ctsegmentator {
static const std::map<std::uint8_t, std::string> organ_names = {
    { 1, "spleen" },
    { 2, "kidney_right" },
    { 3, "kidney_left" },
    { 4, "gallbladder" },
    { 5, "liver" },
    { 6, "stomach" },
    { 7, "pancreas" },
    { 8, "adrenal_gland_right" },
    { 9, "adrenal_gland_left" },
    { 10, "lung_left" },
    { 11, "lung_right" },
    { 12, "esophagus" },
    { 13, "trachea" },
    { 14, "thyroid_gland" },
    { 15, "small_bowel" },
    { 16, "duodenum" },
    { 17, "colon" },
    { 18, "urinary_bladder" },
    { 19, "prostate" },
    { 20, "kidney_cyst_left" },
    { 21, "kidney_cyst_right" },
    { 22, "sacrum" },
    { 23, "vertebrae_SL" },
    { 24, "vertebrae_T" },
    { 25, "vertebrae_C" },
    { 26, "heart" },
    { 27, "aorta" },
    { 28, "pulmonary_vein" },
    { 29, "brachiocephalic_trunk" },
    { 30, "subclavian_artery_right" },
    { 31, "subclavian_artery_left" },
    { 32, "common_carotid_artery_right" },
    { 33, "common_carotid_artery_left" },
    { 34, "brachiocephalic_vein_left" },
    { 35, "brachiocephalic_vein_right" },
    { 36, "atrial_appendage_left" },
    { 37, "vena_cava" },
    { 38, "portal_vein_and_splenic_vein" },
    { 39, "iliac_artery_left" },
    { 40, "iliac_artery_right" },
    { 41, "iliac_vena_left" },
    { 42, "iliac_vena_right" },
    { 43, "humerus" },
    { 44, "scapula" },
    { 45, "clavicula" },
    { 46, "femur_left" },
    { 47, "femur_right" },
    { 48, "hip" },
    { 49, "spinal_cord" },
    { 50, "gluteus_left" },
    { 51, "gluteus_right" },
    { 52, "autochthon_left" },
    { 53, "autochthon_right" },
    { 54, "iliopsoas_left" },
    { 55, "iliopsoas_right" },
    { 56, "brain" },
    { 57, "skull" },
    { 58, "rib_left" },
    { 59, "rib_right" },
    { 60, "sternum" }
};
}