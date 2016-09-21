import os
import cPickle as pickle
import settings
import shutil
from joblib import Parallel, delayed

tmp_dir = os.path.join(settings.project_root, 'tmp')
raw_pickle = os.path.join(tmp_dir, 'One_Genre.pickle')
final_labels = os.path.join(tmp_dir, 'final_labels.pickle')
delete_files = os.path.join(tmp_dir, 'delete_labels.pickle')

def make_files():
    mapping_dict = {}
    delete_list = []
    with open(raw_pickle, 'r') as raw:
        raw_labels = pickle.load(raw)
        for item in raw_labels:
            if item['genres']:
                mapping_dict[str(item['_id'])] = item['genres'][0]
            else:
                delete_list.append(str(item['_id']))

    with open(final_labels, 'wb') as f:
        pickle.dump(mapping_dict, f)
    with open(delete_files, 'wb') as f:
        pickle.dump(delete_list, f)


with open(delete_files, 'rb') as f:
    delete_list = pickle.load(f)
    delete_list = set(delete_list)

def delete_filter(item):
    global delete_list
    if item in delete_list:
        return True
    else:
        return False

if __name__ == '__main__':
    test_files = os.path.join(tmp_dir, 'test_files')
    file_list = set(os.listdir(test_files))
    # files = Parallel(n_jobs=3)(delayed(delete_filter)(item.split(".")[0]) for item in file_list)

    files = [os.path.join(test_files,item+'.txt') for item in delete_list]
    unlabelled = (os.path.join(tmp_dir, 'unlabelled_files'))
    for f in files:
        shutil.move(f, unlabelled)
