import os
import time
import argparse

import json

from annoy import AnnoyIndex

from lib.log import Logger
from lib.merge_index import merge_index


def main(project_name, aux_projext_name):

    tic = time.time()

    logger = Logger('_01_embed_index_{}'.format(project_name))
    logger.info('=' * 50)

    dir_prj = os.path.join('..', project_name[:-7])
    dir_aux = os.path.join('..', aux_projext_name[:-7])

    with open(os.path.join(dir_prj, '_embed_index', 'index_names_{}.json'.format(project_name)), 'r') as f:
        prj_index_names = json.load(f)

    with open(os.path.join(dir_aux, '_embed_index', 'index_names_{}.json'.format(aux_projext_name)), 'r') as f:
        aux_index_names = json.load(f)

    prj_u = AnnoyIndex(512, metric='euclidean')
    prj_u.load(os.path.join(dir_prj, '_embed_index', 'index_features_{}.ann'.format(project_name)))

    aux_u = AnnoyIndex(512, metric='euclidean')
    aux_u.load(os.path.join(dir_aux, '_embed_index', 'index_features_{}.ann'.format(aux_projext_name)))

    logger.info('===> embed index images')

    index_names, t = merge_index(prj_index_names, prj_u, aux_index_names, aux_u)

    dir_index = '_embed_index'
    os.makedirs(dir_index, exist_ok=True)

    new_prj_name = project_name + '_' + aux_projext_name

    with open(os.path.join(dir_index, 'index_names_{}.json'.format(new_prj_name)), 'w') as f:
        json.dump(index_names, f)

    t.build(100)
    t.save(os.path.join(dir_index, 'index_features_{}.ann'.format(new_prj_name)))

    toc = time.time() - tic
    logger.info('Elapsed time: {:.1f} [min]'.format(toc / 60.0))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', default='img_model', help='project name')
    parser.add_argument('--aux_name', '-a', default='delf_model', help='project name')
    params = parser.parse_args()

    main(project_name=params.name, aux_projext_name=params.aux_name)
