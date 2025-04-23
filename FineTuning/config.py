# Path to contact map files
FOLDERS = {
    'H1': '../ExampleData/RCMC/H1',
    'HCT-116': '../ExampleData/RCMC/HCT-116',
    'GM12878': '../ExampleData/RCMC/GM12878',
    'K562': '../ExampleData/RCMC/K562'
}

# Path to loop files (to add weights on loops during training)
LOOPS = {
    'H1': '../ExampleData/Structures/Loops/H1.txt',
    'HCT-116': '../ExampleData/Structures/Loops/HCT-116.txt',
    'GM12878': '../ExampleData/Structures/Loops/GM12878.txt',
    'K562': '../ExampleData/Structures/Loops/K562.txt'
}

# Path to epigenomic feature files
PATH_epigenomics = '../ExampleData/Epi/'

# Path to pretrained model
PATH_pretrained_model = './temp_model_60.h5'

# Path to OE vectors
PATH_OE = '../ExampleData/MicroC/OE_vectors'

# Regions for training
# key: (chromosome, start_loc, end_loc)
TRAINING_REGIONS = {
    0: ('chr6', 25150000, 28600000),  # high
    # 1: ('chr19', 36050000, 39750000),  # exclude
    2: ('chr5', 157000000, 160150000),  # median
    # 3: ('chr1', 207650000, 210350000),  # median
    4: ('chr4', 61350000, 64450000),  # low
    # 5: ('chr8', 125000000, 129750000),  # median
    6: ('chr6', 29750000, 32250000),  # high
    # 7: ('chrX', 47125000, 49375000),  # median high
    8: ('chr1', 237300000, 240500000),  # low
    9: ('chr7', 10500000, 13500000),  # median low
    # 10: ('chr8', 63000000, 66500000),  # median low
    11: ('chr4', 181300000, 184000000),  # median
    12: ('chr3', 118750000, 121500000),  # median
    # 13: ('chr9', 106625000, 109875000)  # median low
}
