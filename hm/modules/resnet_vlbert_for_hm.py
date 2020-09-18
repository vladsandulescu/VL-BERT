import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from external.pytorch_pretrained_bert.modeling import BertPredictionHeadTransform
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBert
from common.nlp.roberta import RobertaTokenizer
from common.attention import MultiheadAttention

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class ResNetVLBERT(Module):
    def __init__(self, config):

        super(ResNetVLBERT, self).__init__(config)

        self.image_feature_extractor = FastRCNN(config,
                                                average_pool=True,
                                                final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                enable_cnn_reg_loss=False)
        self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN

        if 'roberta' in config.NETWORK.BERT_MODEL_NAME:
            self.tokenizer = RobertaTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)

        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path
        self.language_pretrained_model_path = language_pretrained_model_path
        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBert(config.NETWORK.VLBERT,
                                         language_pretrained_model_path=language_pretrained_model_path)

        dim = config.NETWORK.VLBERT.hidden_size
        if config.NETWORK.CLASSIFIER_TYPE == "2fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, config.NETWORK.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.NETWORK.CLASSIFIER_HIDDEN_SIZE, 1),
            )
        elif config.NETWORK.CLASSIFIER_TYPE == "1fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, 2)
            )
        else:
            raise ValueError("Not support classifier type: {}!".format(config.NETWORK.CLASSIFIER_TYPE))

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)
        for m in self.final_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(ResNetVLBERT, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        pass

    def train_forward(self,
                      image,
                      boxes,
                      im_info,
                      expression,
                      label,
                      ):
        ###########################################
        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        origin_len = boxes.shape[1]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################
        # prepare text
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states, pooled_rep = self.vlbert(text_input_ids,
                                        text_token_type_ids,
                                        text_visual_embeddings,
                                        text_mask,
                                        object_vl_embeddings,
                                        box_mask,
                                        output_all_encoded_layers=False)

        ###########################################
        outputs = {}

        # classifier
        # logits = self.final_mlp(pooled_rep).squeeze(1)
        logits = self.final_mlp(pooled_rep)

        # loss
        cls_loss = F.cross_entropy(logits, label)
        # cls_loss = F.binary_cross_entropy_with_logits(logits.float(), torch.eye(2)[label].cuda()) * logits.size(1)
        # cls_loss = F.binary_cross_entropy_with_logits(
        #     logits,
        #     label.to(dtype=logits.dtype)) * label.size(0)

        # outputs.update({'label_probs': torch.sigmoid(logits),
        #                 'label': label,
        #                 'cls_loss': cls_loss})
        outputs.update({'label_probs': F.softmax(logits, dim=1)[:, 1],
                        'label': label,
                        'cls_loss': cls_loss})

        loss = cls_loss.mean()

        return outputs, loss

    def inference_forward(self,
                          image,
                          boxes,
                          im_info,
                          expression):

        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        origin_len = boxes.shape[1]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################
        # prepare text
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states, pooled_rep = self.vlbert(text_input_ids,
                                        text_token_type_ids,
                                        text_visual_embeddings,
                                        text_mask,
                                        object_vl_embeddings,
                                        box_mask,
                                        output_all_encoded_layers=False)

        ###########################################
        outputs = {}

        logits = self.final_mlp(pooled_rep)

        # outputs.update({'label_probs': torch.sigmoid(logits)})
        outputs.update({'label_probs': F.softmax(logits, dim=1)[:, 1]})

        return outputs


class ResNetVLBERTPaired(Module):
    def __init__(self, config):

        super(ResNetVLBERTPaired, self).__init__(config)

        self.image_feature_extractor = FastRCNN(config,
                                                average_pool=True,
                                                final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                enable_cnn_reg_loss=False)
        self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN

        if 'roberta' in config.NETWORK.BERT_MODEL_NAME:
            self.tokenizer = RobertaTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)

        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path
        self.language_pretrained_model_path = language_pretrained_model_path
        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBert(config.NETWORK.VLBERT,
                                         language_pretrained_model_path=language_pretrained_model_path)

        dim = config.NETWORK.VLBERT.hidden_size

        if config.NETWORK.CLASSIFIER_TYPE == "2fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, config.NETWORK.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.NETWORK.CLASSIFIER_HIDDEN_SIZE, 1),
            )
        elif config.NETWORK.CLASSIFIER_TYPE == "1fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, 2)
            )
        elif config.NETWORK.CLASSIFIER_TYPE == "1fc-paired":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Linear(dim*2, dim),
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, 2)
            )
        else:
            raise ValueError("Not support classifier type: {}!".format(config.NETWORK.CLASSIFIER_TYPE))

        # init weights
        self.init_weight(config)

        self.fix_params()

    def init_weight(self, config):
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)
        for m in self.final_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(ResNetVLBERTPaired, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        pass

    def train_forward(self,
                      image,
                      boxes,
                      im_info,
                      expression,
                      im2text,
                      label,
                      ):
        ###########################################
        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        origin_len = boxes.shape[1]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################
        # prepare text
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

        # prepare im2text
        im2text_input_ids = im2text.new_zeros((im2text.shape[0], im2text.shape[1] + 2))
        im2text_input_ids[:, 0] = cls_id
        im2text_input_ids[:, 1:-1] = im2text
        _sep_pos_im2text = (im2text_input_ids > 0).sum(1)
        _batch_inds_im2text = torch.arange(im2text.shape[0], device=im2text.device)
        im2text_input_ids[_batch_inds_im2text, _sep_pos_im2text] = sep_id
        im2text_token_type_ids = im2text_input_ids.new_zeros(im2text_input_ids.shape)
        im2text_mask = im2text_input_ids > 0
        im2text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, im2text_input_ids.shape[1], 1))

        ###########################################

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # stacked = torch.stack(list_of_tensors, dim=1)
        # interleaved = torch.flatten(stacked, start_dim=0, end_dim=1)

        sequence_output, pooled_rep = self.vlbert(
            torch.flatten(torch.stack([text_input_ids, im2text_input_ids], dim=1), start_dim=0, end_dim=1),
            torch.flatten(torch.stack([text_token_type_ids, im2text_token_type_ids], dim=1), start_dim=0, end_dim=1),
            torch.flatten(torch.stack([text_visual_embeddings, im2text_visual_embeddings], dim=1), start_dim=0, end_dim=1),
            torch.flatten(torch.stack([text_mask, im2text_mask], dim=1), start_dim=0, end_dim=1),
            torch.cat([object_vl_embeddings, object_vl_embeddings]),
            torch.cat([box_mask, box_mask]),
            output_all_encoded_layers=False)

        ###########################################

        n_pair = pooled_rep.size(0) // 2
        reshaped_output = pooled_rep.contiguous().view(n_pair, -1)

        logits = self.final_mlp(reshaped_output)

        # loss
        cls_loss = F.cross_entropy(logits, label)
        # cls_loss = F.binary_cross_entropy_with_logits(
        #     logits,
        #     label.to(dtype=logits.dtype)) * label.size(0)

        outputs = {}
        # outputs.update({'label_probs': torch.sigmoid(logits),
        #                 'label': label,
        #                 'cls_loss': cls_loss})
        outputs.update({'label_probs': F.softmax(logits, dim=1)[:, 1],
                        'label': label,
                        'cls_loss': cls_loss})

        loss = cls_loss.mean()

        return outputs, loss

    def inference_forward(self,
                          image,
                          boxes,
                          im_info,
                          expression,
                          im2text):
        ###########################################
        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        origin_len = boxes.shape[1]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################
        # prepare text
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

        # prepare im2text
        im2text_input_ids = im2text.new_zeros((im2text.shape[0], im2text.shape[1] + 2))
        im2text_input_ids[:, 0] = cls_id
        im2text_input_ids[:, 1:-1] = im2text
        _sep_pos_im2text = (im2text_input_ids > 0).sum(1)
        _batch_inds_im2text = torch.arange(im2text.shape[0], device=im2text.device)
        im2text_input_ids[_batch_inds_im2text, _sep_pos_im2text] = sep_id
        im2text_token_type_ids = im2text_input_ids.new_zeros(im2text_input_ids.shape)
        im2text_mask = im2text_input_ids > 0
        im2text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, im2text_input_ids.shape[1], 1))

        ###########################################

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # stacked = torch.stack(list_of_tensors, dim=1)
        # interleaved = torch.flatten(stacked, start_dim=0, end_dim=1)

        sequence_output, pooled_rep = self.vlbert(
            torch.flatten(torch.stack([text_input_ids, im2text_input_ids], dim=1), start_dim=0, end_dim=1),
            torch.flatten(torch.stack([text_token_type_ids, im2text_token_type_ids], dim=1), start_dim=0, end_dim=1),
            torch.flatten(torch.stack([text_visual_embeddings, im2text_visual_embeddings], dim=1), start_dim=0,
                          end_dim=1),
            torch.flatten(torch.stack([text_mask, im2text_mask], dim=1), start_dim=0, end_dim=1),
            torch.cat([object_vl_embeddings, object_vl_embeddings]),
            torch.cat([box_mask, box_mask]),
            output_all_encoded_layers=False)
        ###########################################

        n_pair = pooled_rep.size(0) // 2
        reshaped_output = pooled_rep.contiguous().view(n_pair, -1)

        logits = self.final_mlp(reshaped_output)

        outputs = {}
        # outputs.update({'label_probs': torch.sigmoid(logits)})
        outputs.update({'label_probs': F.softmax(logits, dim=1)[:, 1]})

        return outputs


class AttentionPool(nn.Module):
    """ attention pooling layer """
    def __init__(self, hidden_size, drop=0.0):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())
        self.dropout = nn.Dropout(drop)

    def forward(self, input_, mask=None):
        """input: [B, T, D], mask = [B, T]"""
        score = self.fc(input_).squeeze(-1)
        if mask is not None:
            mask = mask.to(dtype=input_.dtype) * -1e4
            score = score + mask
        norm_score = self.dropout(F.softmax(score, dim=1))
        output = norm_score.unsqueeze(1).matmul(input_).squeeze(1)
        return output


class ResNetVLBERTPairedAttn(Module):
    def __init__(self, config):

        super(ResNetVLBERTPairedAttn, self).__init__(config)

        self.image_feature_extractor = FastRCNN(config,
                                                average_pool=True,
                                                final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                enable_cnn_reg_loss=False)
        self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN

        if 'roberta' in config.NETWORK.BERT_MODEL_NAME:
            self.tokenizer = RobertaTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)

        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path
        self.language_pretrained_model_path = language_pretrained_model_path
        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBert(config.NETWORK.VLBERT,
                                         language_pretrained_model_path=language_pretrained_model_path)

        dim = config.NETWORK.VLBERT.hidden_size

        self.attn1 = MultiheadAttention(dim,
                                        config.NETWORK.VLBERT.num_attention_heads,
                                        config.NETWORK.CLASSIFIER_DROPOUT)
        self.attn2 = MultiheadAttention(dim,
                                        config.NETWORK.VLBERT.num_attention_heads,
                                        config.NETWORK.CLASSIFIER_DROPOUT)

        if config.NETWORK.CLASSIFIER_TYPE == "2fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, config.NETWORK.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.NETWORK.CLASSIFIER_HIDDEN_SIZE, 1),
            )
        elif config.NETWORK.CLASSIFIER_TYPE == "1fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, 2)
            )
        elif config.NETWORK.CLASSIFIER_TYPE == "1fc-paired-attn":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Linear(2*dim, dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False)
            )
            self.attn_pool = AttentionPool(dim, config.NETWORK.CLASSIFIER_DROPOUT)
            self.hm_output = torch.nn.Linear(2*dim, 2)
        else:
            raise ValueError("Not support classifier type: {}!".format(config.NETWORK.CLASSIFIER_TYPE))

        # init weights
        self.init_weight(config)

        self.fix_params()

    def init_weight(self, config):
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)
        for m in self.final_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

        if config.NETWORK.CLASSIFIER_TYPE == "1fc-paired-attn":
            for m in self.attn_pool.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    torch.nn.init.constant_(m.bias, 0)

            for m in self.hm_output.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    torch.nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(ResNetVLBERTPairedAttn, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        pass

    def train_forward(self,
                      image,
                      boxes,
                      im_info,
                      expression,
                      im2text,
                      label,
                      ):
        ###########################################
        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        origin_len = boxes.shape[1]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################
        # prepare text
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

        # prepare im2text
        im2text_input_ids = im2text.new_zeros((im2text.shape[0], im2text.shape[1] + 2))
        im2text_input_ids[:, 0] = cls_id
        im2text_input_ids[:, 1:-1] = im2text
        _sep_pos_im2text = (im2text_input_ids > 0).sum(1)
        _batch_inds_im2text = torch.arange(im2text.shape[0], device=im2text.device)
        im2text_input_ids[_batch_inds_im2text, _sep_pos_im2text] = sep_id
        im2text_token_type_ids = im2text_input_ids.new_zeros(im2text_input_ids.shape)
        im2text_mask = im2text_input_ids > 0
        im2text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, im2text_input_ids.shape[1], 1))

        ###########################################

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # stacked = torch.stack(list_of_tensors, dim=1)
        # interleaved = torch.flatten(stacked, start_dim=0, end_dim=1)

        sequence_output, attn_masks = self.vlbert(
            torch.flatten(torch.stack([text_input_ids, im2text_input_ids], dim=1), start_dim=0, end_dim=1),
            torch.flatten(torch.stack([text_token_type_ids, im2text_token_type_ids], dim=1), start_dim=0, end_dim=1),
            torch.flatten(torch.stack([text_visual_embeddings, im2text_visual_embeddings], dim=1), start_dim=0, end_dim=1),
            torch.flatten(torch.stack([text_mask, im2text_mask], dim=1), start_dim=0, end_dim=1),
            torch.cat([object_vl_embeddings, object_vl_embeddings]),
            torch.cat([box_mask, box_mask]),
            output_all_encoded_layers=False,
            output_new_attention_mask=True)

        ###########################################

        # classifier
        bs, tl, d = sequence_output.size()
        left_out, right_out = sequence_output.contiguous().view(
            bs // 2, tl * 2, d).chunk(2, dim=1)
        # bidirectional attention
        mask = attn_masks == 0
        left_mask, right_mask = mask.contiguous().view(bs // 2, tl * 2).chunk(2, dim=1)
        left_out = left_out.transpose(0, 1)
        right_out = right_out.transpose(0, 1)
        l2r_attn, _ = self.attn1(left_out, right_out, right_out, key_padding_mask=right_mask)
        r2l_attn, _ = self.attn2(right_out, left_out, left_out, key_padding_mask=left_mask)
        left_out = self.final_mlp(torch.cat([l2r_attn, left_out], dim=-1)).transpose(0, 1)
        right_out = self.final_mlp(torch.cat([r2l_attn, right_out], dim=-1)).transpose(0, 1)
        # attention pooling and final prediction
        left_out = self.attn_pool(left_out, left_mask)
        right_out = self.attn_pool(right_out, right_mask)
        # logits = self.hm_output(torch.cat([left_out, right_out], dim=-1)).squeeze(1)
        logits = self.hm_output(torch.cat([left_out, right_out], dim=-1))

        # loss
        cls_loss = F.cross_entropy(logits, label)
        # cls_loss = F.binary_cross_entropy_with_logits(
        #     logits,
        #     label.to(dtype=logits.dtype)) * label.size(0)

        outputs = {}
        # outputs.update({'label_probs': torch.sigmoid(logits),
        #                 'label': label,
        #                 'cls_loss': cls_loss})
        outputs.update({'label_probs': F.softmax(logits, dim=1)[:, 1],
                        'label': label,
                        'cls_loss': cls_loss})

        loss = cls_loss.mean()

        return outputs, loss

    def inference_forward(self,
                          image,
                          boxes,
                          im_info,
                          expression,
                          im2text):
        ###########################################
        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        origin_len = boxes.shape[1]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################
        # prepare text
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

        # prepare im2text
        im2text_input_ids = im2text.new_zeros((im2text.shape[0], im2text.shape[1] + 2))
        im2text_input_ids[:, 0] = cls_id
        im2text_input_ids[:, 1:-1] = im2text
        _sep_pos_im2text = (im2text_input_ids > 0).sum(1)
        _batch_inds_im2text = torch.arange(im2text.shape[0], device=im2text.device)
        im2text_input_ids[_batch_inds_im2text, _sep_pos_im2text] = sep_id
        im2text_token_type_ids = im2text_input_ids.new_zeros(im2text_input_ids.shape)
        im2text_mask = im2text_input_ids > 0
        im2text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, im2text_input_ids.shape[1], 1))

        ###########################################

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # stacked = torch.stack(list_of_tensors, dim=1)
        # interleaved = torch.flatten(stacked, start_dim=0, end_dim=1)

        sequence_output, attn_masks = self.vlbert(
            torch.flatten(torch.stack([text_input_ids, im2text_input_ids], dim=1), start_dim=0, end_dim=1),
            torch.flatten(torch.stack([text_token_type_ids, im2text_token_type_ids], dim=1), start_dim=0, end_dim=1),
            torch.flatten(torch.stack([text_visual_embeddings, im2text_visual_embeddings], dim=1), start_dim=0,
                          end_dim=1),
            torch.flatten(torch.stack([text_mask, im2text_mask], dim=1), start_dim=0, end_dim=1),
            torch.cat([object_vl_embeddings, object_vl_embeddings]),
            torch.cat([box_mask, box_mask]),
            output_all_encoded_layers=False,
            output_new_attention_mask=True)
        ###########################################

        # classifier
        bs, tl, d = sequence_output.size()
        left_out, right_out = sequence_output.contiguous().view(
            bs // 2, tl * 2, d).chunk(2, dim=1)
        # bidirectional attention
        mask = attn_masks == 0
        left_mask, right_mask = mask.contiguous().view(bs // 2, tl * 2).chunk(2, dim=1)
        left_out = left_out.transpose(0, 1)
        right_out = right_out.transpose(0, 1)
        l2r_attn, _ = self.attn1(left_out, right_out, right_out, key_padding_mask=right_mask)
        r2l_attn, _ = self.attn2(right_out, left_out, left_out, key_padding_mask=left_mask)
        left_out = self.final_mlp(torch.cat([l2r_attn, left_out], dim=-1)).transpose(0, 1)
        right_out = self.final_mlp(torch.cat([r2l_attn, right_out], dim=-1)).transpose(0, 1)
        # attention pooling and final prediction
        left_out = self.attn_pool(left_out, left_mask)
        right_out = self.attn_pool(right_out, right_mask)
        # logits = self.hm_output(torch.cat([left_out, right_out], dim=-1)).squeeze(1)
        logits = self.hm_output(torch.cat([left_out, right_out], dim=-1))

        outputs = {}
        # outputs.update({'label_probs': torch.sigmoid(logits)})
        outputs.update({'label_probs': F.softmax(logits, dim=1)[:, 1]})

        return outputs
