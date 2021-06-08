### These are defaults. Don't touch.

OFA_DEFAULT = {
                    'd': [[2, 3, 4],  # Depth of each stage
                          [2, 3, 4],
                          [2, 3, 4],
                          [2, 3, 4],
                          [2, 3, 4]],
                    'ks': [[3, 5, 7],  # Kernel size in each stage
                           [3, 5, 7],
                           [3, 5, 7],
                           [3, 5, 7],
                           [3, 5, 7]],
                    'e': [[3, 4, 6],  # Expansion ratios for each stage
                          [3, 4, 6],
                          [3, 4, 6],
                          [3, 4, 6],
                          [3, 4, 6]],
                    'r': (160, 176, 192, 208, 224,)  # Valid resolutions
}

PROXYLESS_DEFAULT = {
                    'd': [[2, 3, 4],  # Depth of each stage
                          [2, 3, 4],
                          [2, 3, 4],
                          [2, 3, 4],
                          [2, 3, 4],
                          [1]],
                    'ks': [[3, 5, 7],  # Kernel size in each stage
                           [3, 5, 7],
                           [3, 5, 7],
                           [3, 5, 7],
                           [3, 5, 7],
                           [3, 5, 7]],
                    'e': [[3, 4, 6],  # Expansion ratios for each stage
                          [3, 4, 6],
                          [3, 4, 6],
                          [3, 4, 6],
                          [3, 4, 6],
                          [3, 4, 6]],
                    'r': (160, 176, 192, 208, 224,)  # Valid resolutions
}


# Put your custom search spaces here.
OFA_PRED_1M = {
                    'd': [[2, 3, 4],  # Depth of each stage
                          [2, 3, 4],
                          [2, 3, 4],
                          [2, 3, 4],
                          [2, 3, 4]],
                    'ks': [[3, 5],  # Kernel size in each stage
                           [3, 5, 7],
                           [3, 5, 7],
                           [5, 7],
                           [5, 7]],
                    'e': [[3, 4],  # Expansion ratios for each stage
                          [3, 4, 6],
                          [3, 4, 6],
                          [4, 6],
                          [4, 6]],
                    #'r': (160, 176, 192, 208, 224,)  # Valid resolutions
                    'r': (224,)
}
