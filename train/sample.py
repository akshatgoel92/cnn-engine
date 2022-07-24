import os
import shutil 
import random
import numpy as np



def clear_files(dest):
    try:
        shutil.rmtree(dest)
    except Exception as e:
        print(e)


def ignore_files(src, files):
    return [f for f in files if os.path.isfile(os.path.join(src, f))]


def copy_tree(src, dest, ignore):
    try:
        shutil.copytree(src, dest, ignore=ignore)
    except Exception as e:
        print(e)


def list_files(src):
    try:
        return [os.path.join(src, f) for f in os.listdir(src)]
    except Exception as e:
        print(e)
        return None


def sample_files(files, sample_size):
    if int(sample_size) != sample_size:
        sample_size = int(np.floor(sample_size*len(files)))
    return random.sample(files, sample_size)


def copy_files(src_files, dest_folder):
    for file in src_files:
        shutil.copy(file, dest_folder)


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.remove(path)



if __name__ == '__main__':

    random.seed(1230)
    sample_size = 0.05
    src = 'imagenette2'
    dest = 'imagenette2_sample'
    datasets = ['train', 'val']
    n_classes = None

    copy_tree(src, dest, ignore_files)
    
    for dataset in datasets:
        folders = list_files(os.path.join(src, dataset))

        if n_classes != None:
            folders = sample_files(folder, n_classes)
        
        for folder in folders:
            files = list_files(folder)
            
            if files is not None:
                sampled_files = sample_files(files, sample_size)
                dest_folder = os.path.join(dest, "/".join(sampled_files[0].split('/')[1:-1]))
                copy_files(sampled_files, dest_folder)

    remove_empty_folders(dest)