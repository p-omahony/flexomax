import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np

def get_label_from_impath(df, id_col, label_col, impath):
    im_id = int(impath.stem)
    article_type = df[df[id_col]==im_id][label_col].values[0]
    return article_type

def copy_data(df, dataset, images, labels, id_col, label_col):
    errors = 0
    for im in tqdm(images):
        try :
            label = get_label_from_impath(df, id_col, label_col, im)
        except:
            print(f'label not found for im: {im}')
            errors +=1
            continue

        if label not in labels:
            continue

        dst = Path('heavydata') / dataset / 'images' / im.name
        shutil.copy(im, dst)

        label_dst = Path('heavydata') / dataset / 'labels'/ im.stem
        np.save(label_dst, label)
    print(f'{errors}/{len(images)} errors ~ {100*errors/len(images)}%')