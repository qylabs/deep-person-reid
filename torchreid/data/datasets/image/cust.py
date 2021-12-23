from __future__ import division, print_function, absolute_import

import os.path
import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset


class cust(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'cust_person_reid_reverse'

    # dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='', dy_=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        # data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        # if osp.isdir(data_dir):
        #     self.data_dir = data_dir
        # else:
        #     warnings.warn(
        #         'The current data structure is deprecated. Please '
        #         'put data folders such as "bounding_box_train" under '
        #         '"Market-1501-v15.09.15".'
        #     )

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.dy = dy_
        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        if self.dy:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)
        train = [[], [], []]
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        if self.dy:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)
        # print(gallery)
        super(cust, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):

        # img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')
        #
        # pid_container = set()
        # for img_path in img_paths:
        #     pid, _ = map(int, pattern.search(img_path).groups())
        #     if pid == -1:
        #         continue # junk images are just ignored
        #     pid_container.add(pid)
        # pid2label = {pid: label for label, pid in enumerate(pid_container)}
        imgdir = os.listdir(dir_path)
        imgdir = sorted([i for i in imgdir if os.path.isdir(osp.join(dir_path, i))])
        pid_container = set()
        for i, ii in enumerate(imgdir):
            ids = i
            dir_ = os.path.join(dir_path, ii)
            if len(os.listdir(dir_)) == 0:
                continue
            # pid, _ = map(int, pattern.search(img_path).groups())
            # if pid == -1:
            #     continue # junk images are just ignored
            pid_container.add(ids)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # print(pid2label)
        data = []

        for i, ii in enumerate(imgdir):
            dir_ = os.path.join(dir_path, ii)
            img_paths = glob.glob(osp.join(dir_, '*.jpg'))
            # img_paths = os.listdir(dir_path)
            for img_path in img_paths:
                # ids = int(ii.replace('person', ''))
                ids = i
                # print(img_paths)
                # print(ids)
                # print(img_path)
                # pid, camid = map(int, pattern.search(img_path).groups())
                # if pid == -1:
                #     continue # junk images are just ignored
                # assert 0 <= pid <= 1501 # pid == 0 means background
                # assert 1 <= camid <= 6
                # camid -= 1 # index starts from 0
                if relabel:
                    ids = pid2label[ids]
                data.append((img_path, ids, 0))
        return data
