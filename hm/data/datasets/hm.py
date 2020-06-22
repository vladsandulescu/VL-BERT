import os
import json
import _pickle as cPickle
from PIL import Image
import re
import base64
import numpy as np
import csv
import sys
import time
import pprint
import logging

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer

from common.utils.create_logger import makedirsExist
from common.nlp.roberta import RobertaTokenizer

csv.field_size_limit(sys.maxsize)
# FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
TRAIN_VAL_FIELDNAMES = ["id", "img", "label", "text", "img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
TEST_FIELDNAMES = ["id", "img", "text", "img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

class HMDataset(Dataset):
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
        super(HMDataset, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'

        precomputed_boxes = {
            'train': "train_batch.tsv",
            "dev": "dev_batch.tsv",
            "test": "test_batch.tsv",
        }
            
        self.boxes = boxes
        self.test_mode = test_mode

        self.data_path = data_path
        self.root_path = root_path
        self.image_set = image_set
        self.precomputed_box_files = os.path.join(data_path, precomputed_boxes[image_set])
        self.box_bank = {}
        self.transform = transform
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        self.mask_size = mask_size
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        if tokenizer is None:
            if pretrained_model_name is None:
                pretrained_model_name = 'bert-base-uncased'
            if 'roberta' in pretrained_model_name:
                tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name, cache_dir=self.cache_dir)
            else:
                tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, cache_dir=self.cache_dir)
        self.tokenizer = tokenizer

        self.database = self.load_precomputed_boxes(self.precomputed_box_files)
        if self.aspect_grouping:
            self.group_ids = self.group_aspect(self.database)

    @property
    def data_names(self):
        if self.test_mode:
            return 'image', 'boxes', 'im_info', 'text'
        else:
            return 'image', 'boxes', 'im_info', 'text', 'label'

    def __getitem__(self, index):
        idb = self.database[index]

        # image, boxes, im_info
        image = self._load_image(os.path.join(self.data_path, idb['img']))
        w0, h0 = idb['img_h'], idb['img_w']
        boxes = torch.as_tensor(idb['boxes'])

        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1]])
            boxes = torch.cat((image_box, boxes), dim=0)

        im_info = torch.tensor([w0, h0, 1.0, 1.0, index])
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

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()
        # get shape of all images
        widths = torch.as_tensor([idb['img_w'] for idb in database])
        heights = torch.as_tensor([idb['img_h'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def load_precomputed_boxes(self, box_file):
        if box_file in self.box_bank:
            return self.box_bank[box_file]
        else:
            in_data = []
            with open(box_file, "r") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=TRAIN_VAL_FIELDNAMES if not self.test_mode else TEST_FIELDNAMES)
                for item in reader:
                    idb = {'img_id': str(item['img_id']),
                            'img': str(item['img']),
                            'text': str(item['text']),
                            'label': int(item['label']) if not self.test_mode else None,
                            'img_h': int(item['img_h']),
                            'img_w': int(item['img_w']),
                            'num_boxes': int(item['num_boxes']),
                            'boxes': np.frombuffer(base64.decodebytes(item['boxes'].encode()),
                                                  dtype=np.float32).reshape((int(item['num_boxes']), -1))
                           }
                    in_data.append(idb)
            self.box_bank[box_file] = in_data

            # if self.image_set == 'train':
            #     return in_data[:16]
            # else:
            return in_data

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        return Image.open(path).convert('RGB')

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

