# Defines constants for this search space

# PN and MBV3 constants
OFA_PN_STAGE_N_BLOCKS = (4, 4, 4, 4, 4, 1)

OFA_MBV3_STAGE_N_BLOCKS = (4, 4, 4, 4, 4)

OFA_W_PN = 1.3

OFA_W_MBV3 = 1.2

PN_BLOCKS = (
    "mbconv2_e3_k3",
    "mbconv2_e3_k5",
    "mbconv2_e3_k7",
    "mbconv2_e4_k3",
    "mbconv2_e4_k5",
    "mbconv2_e4_k7",
    "mbconv2_e6_k3",
    "mbconv2_e6_k5",
    "mbconv2_e6_k7",
)


MBV3_BLOCKS = (
    "mbconv3_e3_k3",
    "mbconv3_e3_k5",
    "mbconv3_e3_k7",
    "mbconv3_e4_k3",
    "mbconv3_e4_k5",
    "mbconv3_e4_k7",
    "mbconv3_e6_k3",
    "mbconv3_e6_k5",
    "mbconv3_e6_k7",
)

PN_OP2IDX = {}
for __i, __op in enumerate(PN_BLOCKS):
    PN_OP2IDX[__op] = __i
PN_IDX2OP = {v: k for k, v in PN_OP2IDX.items()}
assert len(PN_OP2IDX) == len(PN_IDX2OP)

MBV3_OP2IDX = {}
for __i, __op in enumerate(MBV3_BLOCKS):
    MBV3_OP2IDX[__op] = __i
MBV3_IDX2OP = {v: k for k, v in MBV3_OP2IDX.items()}
assert len(MBV3_OP2IDX) == len(MBV3_IDX2OP)

PN_NET_CONFIGS_EXAMPLE = [
    ["mbconv2_e6_k7", "mbconv2_e6_k7", "mbconv2_e6_k7", "mbconv2_e6_k7",],
    ["mbconv2_e6_k7", "mbconv2_e6_k7", ],
    ["mbconv2_e6_k7", "mbconv2_e6_k7", "mbconv2_e6_k7", "mbconv2_e6_k7",],
    ["mbconv2_e6_k7", "mbconv2_e6_k7", "mbconv2_e6_k7",],
    ["mbconv2_e6_k7", "mbconv2_e6_k7", "mbconv2_e6_k7", "mbconv2_e6_k7",],
    ["mbconv2_e6_k7"],
]

MBV3_NET_CONFIGS_EXAMPLE = [
    ["mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7",],
    ["mbconv3_e6_k7", "mbconv3_e6_k7",],
    ["mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7",],
    ["mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7",],
    ["mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7",],
]

# ResNet constants
OFA_RES_STAGE_MAX_N_BLOCKS = (4, 4, 6, 4)
OFA_RES_STAGE_MIN_N_BLOCKS = (2, 2, 4, 2)
OFA_RES_STAGE_BASE_CHANNELS = (256, 512, 1024, 2048)
OFA_RES_N_STAGES = len(OFA_RES_STAGE_MAX_N_BLOCKS) + 1 # Stem considered a special stage
OFA_RES_WIDTH_MULTIPLIERS = (0.65, 0.8, 1.0)
OFA_RES_EXPANSION_RATIOS = (0.2, 0.25, 0.35)
OFA_RES_ADDED_DEPTH_LIST = (0, 1, 2)
OFA_RES_KERNEL_SIZES = (3,)
OFA_RES_STEM_PREFIXES = ("stem+res", "stem")

# Target networks for cons acc search
PN_CONS_ACC_SEARCH_TARGET_NETS = []
assert len(set(str(__v) for __v in PN_CONS_ACC_SEARCH_TARGET_NETS)) \
       == len(PN_CONS_ACC_SEARCH_TARGET_NETS), "Target nets cannot have duplicates"

MBV3_CONS_ACC_SEARCH_TARGET_NETS = []
assert len(set(str(__v) for __v in MBV3_CONS_ACC_SEARCH_TARGET_NETS)) \
       == len(MBV3_CONS_ACC_SEARCH_TARGET_NETS), "Target nets cannot have duplicates"
