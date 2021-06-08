import os
from pathlib import Path
import shutil


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)