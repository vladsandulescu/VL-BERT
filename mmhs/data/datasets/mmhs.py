import os
import json
import _pickle as cPickle
from PIL import Image
import re
import random
import base64
import numpy as np
import jsonlines
import sys
import time
import pprint
import logging

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer

from common.utils.create_logger import makedirsExist

class MMHS(Dataset):
    def __init__(self, image_set, root_path, data_path, answer_vocab_file, use_imdb=True,
                 with_precomputed_visual_feat=False, boxes="10-100ada",
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=True, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False, mask_size=(14, 14),
                 aspect_grouping=False, **kwargs):
        """
        Visual Question Answering Dataset

        :param image_set: image folder name
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param mask_size: size of instance mask of each object
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(MMHS, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'

        mmhs_74K = "MMHS16K.json"
        mmhs_img_txt = "img_txt"
        mmhs_img = "img_resized"
        mmhs_imgfeat = "imgfeat/d2_10-100/json/img_resized"
        mmhs_splits = {
            "train": "MMHS16K_splits/train_ids.txt",
            "val": "MMHS16K_splits/val_ids.txt",
            "test": "MMHS16K_splits/test_ids.txt"
        }

        self.boxes = boxes
        self.test_mode = test_mode
        self.with_precomputed_visual_feat = with_precomputed_visual_feat
        self.data_path = data_path
        self.root_path = root_path
        self.image_sets = [iset.strip() for iset in image_set.split('+')]
        self.image_files = os.path.join(data_path, mmhs_img)
        self.ids_files = [os.path.join(data_path, mmhs_splits[iset])
                          for iset in self.image_sets]
        self.img_txt_files = os.path.join(data_path, mmhs_img_txt)
        self.ann_file = os.path.join(data_path, mmhs_74K)
        self.precomputed_box_files = os.path.join(data_path, mmhs_imgfeat)
        self.box_bank = {}
        self.transform = transform
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        self.mask_size = mask_size
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
            'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name,
            cache_dir=self.cache_dir)

        self.database = self.load_annotations(self.ann_file, self.ids_files)
        if self.aspect_grouping:
            self.group_ids = self.group_aspect(self.database)

    @property
    def data_names(self):
        if self.test_mode:
            return ['image', 'boxes', 'im_info', 'text']
        else:
            return ['image', 'boxes', 'im_info', 'text', 'label']

    def __getitem__(self, index):
        idb = self.database[index]

        # image, boxes, im_info
        boxes_data = self._load_json(idb['boxes_fn'])
        if self.with_precomputed_visual_feat:
            image = None
            w0, h0 = idb['width'], idb['height']

            boxes_features = torch.as_tensor(
                np.frombuffer(self.b64_decode(boxes_data['features']), dtype=np.float32).reshape((boxes_data['num_boxes'], -1))
            )
        else:
            image = self._load_image(idb['image_fn'])
            w0, h0 = image.size
        boxes = torch.as_tensor(
            np.frombuffer(self.b64_decode(boxes_data['boxes']), dtype=np.float32).reshape(
                (boxes_data['num_boxes'], -1))
        )

        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1]])
            boxes = torch.cat((image_box, boxes), dim=0)
            if self.with_precomputed_visual_feat:
                if 'image_box_feature' in boxes_data:
                    image_box_feature = torch.as_tensor(
                        np.frombuffer(
                            self.b64_decode(boxes_data['image_box_feature']), dtype=np.float32
                        ).reshape((1, -1))
                    )
                else:
                    image_box_feature = boxes_features.mean(0, keepdim=True)
                boxes_features = torch.cat((image_box_feature, boxes_features), dim=0)
        im_info = torch.tensor([w0, h0, 1.0, 1.0])
        flipped = False
        if self.transform is not None:
            image, boxes, _, im_info, flipped = self.transform(image, boxes, None, im_info, flipped)

        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)

        # flip: 'left' -> 'right', 'right' -> 'left'
        q_tokens = self.tokenizer.tokenize(idb['text'])
        if flipped:
            q_tokens = self.flip_tokens(q_tokens, verbose=False)
        label = torch.as_tensor(idb['label']) if not self.test_mode else None

        # text
        q_retokens = q_tokens
        q_ids = self.tokenizer.convert_tokens_to_ids(q_retokens)

        # concat box feature to box
        if self.with_precomputed_visual_feat:
            boxes = torch.cat((boxes, boxes_features), dim=-1)

        if self.test_mode:
            return image, boxes, im_info, q_ids
        else:
            return image, boxes, im_info, q_ids, label

    @staticmethod
    def flip_tokens(tokens, verbose=True):
        changed = False
        tokens_new = [tok for tok in tokens]
        for i, tok in enumerate(tokens):
            if tok == 'left':
                tokens_new[i] = 'right'
                changed = True
            elif tok == 'right':
                tokens_new[i] = 'left'
                changed = True
        if verbose and changed:
            logging.info('[Tokens Flip] {} -> {}'.format(tokens, tokens_new))
        return tokens_new

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    def load_annotations(self, ann_file, ids_files):
        tic = time.time()
        database = []
        db_cache_name = 'mmhs_boxes{}_{}'.format(self.boxes, '+'.join(self.image_sets))
        if self.test_mode:
            db_cache_name = db_cache_name + '_testmode'
        db_cache_root = os.path.join(self.root_path, 'cache')
        db_cache_path = os.path.join(db_cache_root, '{}.pkl'.format(db_cache_name))

        if os.path.exists(db_cache_path):
            if not self.ignore_db_cache:
                # reading cached database
                print('cached database found in {}.'.format(db_cache_path))
                with open(db_cache_path, 'rb') as f:
                    print('loading cached database from {}...'.format(db_cache_path))
                    tic = time.time()
                    database = cPickle.load(f)
                    print('Done (t={:.2f}s)'.format(time.time() - tic))
                    return database
            else:
                print('cached database ignored.')

        # ignore or not find cached database, reload it from annotation file
        print('loading database of split {}...'.format('+'.join(self.image_sets)))
        tic = time.time()

        filter_ids = []
        for id_file in ids_files:
            with open(id_file, "r") as file:
                lines = [x.strip() for x in file.readlines()]
                filter_ids.extend(lines)

        with jsonlines.open(ann_file) as reader:
            for ann in reader:
                if ann['id'] in filter_ids:
                    db_i = {
                        'id': ann['id'],
                        'image_fn': os.path.join(self.image_files, str(ann['id']) + '.jpg'),
                        'boxes_fn': os.path.join(self.precomputed_box_files, str(ann['id']) + '.json'),
                        'text': ann['text'],
                        'label': ann['label'] if not self.test_mode else None
                    }
                    if os.path.exists(db_i['image_fn']) and os.path.exists(db_i['boxes_fn']):
                        database.append(db_i)
        print('Done (t={:.2f}s)'.format(time.time() - tic))

        # cache database via cPickle
        if self.cache_db:
            print('caching database to {}...'.format(db_cache_path))
            tic = time.time()
            if not os.path.exists(db_cache_root):
                makedirsExist(db_cache_root)
            with open(db_cache_path, 'wb') as f:
                cPickle.dump(database, f)
            print('Done (t={:.2f}s)'.format(time.time() - tic))

        return database
        # return random.sample(database, 256)

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path).convert('RGB')
        else:
            return Image.open(path).convert('RGB')

    def _load_json(self, path):
        if '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            with open(path, 'r') as f:
                return json.load(f)

