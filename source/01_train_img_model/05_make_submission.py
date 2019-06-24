import os
import time
import argparse

import json

import numpy as np
import pandas as pd

from collections import defaultdict
from multiprocessing import Process, Manager

from annoy import AnnoyIndex

from tqdm import tqdm
from logging import getLogger
from lib.log import Logger


def dist_average(indeces, dists, list_dict):

    new_dists = list()
    for i, index in enumerate(indeces):
        ave = (dists[i] + sum([ref_dict[index] for ref_dict in list_dict])) / (1 + len(list_dict))
        new_dists.append(ave)

    ret_indeces, ret_dists = list(zip(*sorted(list(zip(indeces, new_dists)), key=lambda x: x[1])))

    return ret_indeces, ret_dists


def unique_order_preserving(list_input):

    result = list()
    for item in list_input:
        if item not in result:
            result.append(item)

    return result


def process(project_name, target_id_list, procnum, return_dict):

    logger = getLogger('root')

    images = list()

    with open(os.path.join('_embed_index', 'index_names_{}.json'.format(project_name)), 'r') as f:
        index_names = json.load(f)

    with open(os.path.join('_embed_index', 'test_names_{}.json'.format(project_name)), 'r') as f:
        test_names = json.load(f)

    num_index = len(index_names)

    f = 512
    u = AnnoyIndex(f, metric='euclidean')
    u.load(os.path.join('_embed_index', 'index_features_{}.ann'.format(project_name)))
    logger.info('number of index vectors: {}'.format(u.get_n_items()))

    db_test = AnnoyIndex(f, metric='euclidean')
    db_test.load(os.path.join('_embed_index', 'test_features_{}.ann'.format(project_name)))
    logger.info('number of test vectors: {}'.format(db_test.get_n_items()))

    logger.info('===> embed test images and get nearest neighbors')

    search_k = 1_000_00

    for test_id in tqdm(target_id_list):

        # main query

        try:
            db_index = test_names.index(test_id)
            img_feature = db_test.get_item_vector(db_index)

            indeces, dists = u.get_nns_by_vector(img_feature, n=300, search_k=search_k, include_distances=True)
        except ValueError:
            logger.info('{}: ValueError error'.format(test_id))
            indeces, dists = list(range(300)), np.ones(300).tolist()

        indeces_init = list(indeces)
        dists_init = list(dists)

        # sub query

        list_dict = list()
        sub_id_selected = list()

        num_sub_query = 6
        for j in range(num_sub_query):

            sub_id = indeces[0]
            sub_id_selected.append(sub_id)

            # search from index
            indeces_exp, dists_exp = u.get_nns_by_item(sub_id, n=600, search_k=search_k, include_distances=True)

            d = defaultdict(lambda: float(dists_exp[-1]))

            for key, dist_exp in zip(indeces_exp, dists_exp):
                d[key] = dist_exp

            # add result of sub query
            list_dict.append(d)

            # take average by initial query and current sub queries
            indeces, dists = dist_average(indeces_init, dists_init, list_dict)

            # remove selected sub_ids
            indeces, dists = zip(*[(_id, _dist) for _id, _dist, in zip(indeces, dists) if _id not in sub_id_selected])

        # merge selected sub_ids and sorted other sub_ids
        indeces = sub_id_selected + list(indeces)

        indeces = [index % num_index for index in indeces]

        names = [index_names[index] for index in indeces]
        names = unique_order_preserving(names)[:100]

        images.append(' '.join(names))

    return_dict[procnum] = images


def main(project_name):

    tic = time.time()

    logger = Logger('_05_make_submission_{}'.format(project_name))
    logger.info('=' * 50)

    sample_submission = pd.read_csv('../../dataset/sample_submission.csv')

    images = list()

    test_id_list = sample_submission.id

    f = 512
    u = AnnoyIndex(f, metric='euclidean')
    u.load(os.path.join('_embed_index', 'index_features_{}.ann'.format(project_name)))

    db_test = AnnoyIndex(f, metric='euclidean')
    db_test.load(os.path.join('_embed_index', 'test_features_{}.ann'.format(project_name)))

    logger.info('===> embed test images and get nearest neighbors')

    manager = Manager()
    return_dict = manager.dict()

    num_processor = 8

    l = [(len(test_id_list) + i) // num_processor for i in range(num_processor)]
    processor_target = 0

    list_processors = list()

    for p in range(num_processor):

        pr = Process(target=process,
                     args=(project_name, test_id_list[processor_target:processor_target+l[p]], p, return_dict))

        list_processors.append(pr)
        processor_target += l[p]

    for p in range(num_processor):
        list_processors[p].start()

    for p in range(num_processor):
        list_processors[p].join()

    for p in range(num_processor):
        images.extend(return_dict[p])

    assert len(images) == len(test_id_list)

    submission = pd.DataFrame(test_id_list, columns=['id'])
    submission['images'] = images

    output_path = '../../submission/submission_{}.csv'.format(project_name)
    submission.to_csv(output_path, index=False)

    toc = time.time() - tic
    logger.info('Elapsed time: {:.1f} [min]'.format(toc / 60.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', default='', help='project name')
    params = parser.parse_args()

    main(project_name=params.name)
