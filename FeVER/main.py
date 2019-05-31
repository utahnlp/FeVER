# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.5
#
# Date: 2017-03-08 13:10:47
# Last modified: 2019-05-22 13:37:00

"""
The main entrance of the WSABIE model.
"""

import logging
import argparse
# import os

import ExAssist as EA
import torch

import FeVER.utils as utils
from FeVER.model import Model
from FeVER import Logger
from FeVER.learner import Learner
from FeVER.config import Config
from FeVER.dataset import XMLDataset
from FeVER.cv import CV


def train(args):
    """Tran the model.

    Args:
        argparse.Namespace
    """
    logger = logging.getLogger(Logger.project_name)
    modelFile = args.model_file

    dataset = XMLDataset(args)
    args.label_feat_num = dataset.label_feat_num
    args.input_feat_num = dataset.input_feat_num
    args.label_num = dataset.label_num

    logger.info('Initializing the model...')
    model = Model(args)

    learner = Learner(args)
    learner.train(dataset, model)

    # Save the model
    logger.info('Saving the model...')
    torch.save(model,  modelFile)
    # To release the memory
    del dataset

    if args.gen_ctx_embedding is True:
        logger.info('Generating the context embedding...')
        generate_embed(
            args.gen_ctx_file,
            args.gen_ctx_dict_file,
            args.ctx_embed_file,
            model.context_embed_func,
            args)

    if args.gen_label_embedding is True:
        logger.info('Generating the label embedding...')
        generate_embed(
            args.gen_label_file,
            args.gen_label_dict_file,
            args.label_embed_file,
            model.label_embed_func,
            args)

    return model


def generate_embed(featFile, mappingFile,
                   output_path,
                   embed_func, args):
    logger = logging.getLogger(Logger.project_name)

    logger.info('Loading label features...')
    nums, feats, label_index = utils.read_binary(featFile, args)

    tmp = []
    for t in label_index:
        tmp.append(t[0])
    label_index = tmp

    feats, offsets = utils.bag(feats)
    feats = torch.from_numpy(feats).long()
    offsets = torch.from_numpy(offsets).long()
    if args.cuda is True:
        feats = feats.cuda()
        offsets = offsets.cuda()

    logger.info('Loading mapping file...')
    mapping = utils.load_mapping(mappingFile)

    logger.info('Generating embeddings...')
    with torch.no_grad():
        embed = embed_func(feats, offsets, mapping, label_index)

    logger.info('Writing embeddings...')
    utils.write_embed(output_path, embed)


def generation(args):
    logger = logging.getLogger(Logger.project_name)
    # models_directory = os.path.dirname(args.pred_model_file)
    # names = os.listdir(models_directory)
    # names = [s for s in names if
    #          s.startswith('FeVER.model.') and not s.endswith('log')]
    # for name in names:
    # path = os.path.join(models_directory, name)
    output_path = args.pred_output_file
    s = 'Loading the model {a}'.format(a=str(args.pred_model_file))
    logger.info(s)
    model = torch.load(args.pred_model_file,
                       map_location=lambda stroage, loc: stroage)
    if args.embedding == 'label':
        func = model.label_embed_func
    elif args.embedding == 'context':
        func = model.context_embed_func
    generate_embed(
        args.pred_feat_file,
        args.pred_dict_file,
        output_path,
        func,
        args)


def main(config_path):
    assist = EA.getAssist('GEL')
    assist.config_path = config_path
    with EA.start(assist) as assist:
        # print(assist.config['common'])
        config = Config(assist.config)
        Logger.initLogger(config.model_file+'.log')
        if config.mode == 'train':
            train(config)
        elif config.mode == 'generation':
            generation(config)
        elif config.mode == 'CV':
            cv = CV(config)
            cv.runcv()


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('main()',  sort='cumulative')
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    main(args.config_path)
