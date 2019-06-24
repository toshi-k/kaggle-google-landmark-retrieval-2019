from annoy import AnnoyIndex
from tqdm import tqdm

from logging import getLogger


def merge_index(index1, u1, index2, u2):

    logger = getLogger('root')

    f = u1.f + u2.f
    logger.info('Generate new index (dimension: {:d})'.format(f))
    t = AnnoyIndex(f, metric='euclidean')

    num_index = len(index1)

    for i in tqdm(range(num_index)):

        v1 = u1.get_item_vector(i)

        index_target = index2.index(index1[i])
        v2 = u2.get_item_vector(index_target)

        t.add_item(i, v1 + v2)

    return index1, t
