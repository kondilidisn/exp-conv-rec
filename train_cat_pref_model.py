import os
import shutil
import csv
import random
from tqdm import tqdm, trange
from io import open
import jsonlines
from operator import add
import datetime
import re
import nltk
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
import re
import pickle
import math
from tqdm import tqdm

import transformers

from category_pref_model import Cat_Pref_BERT

from redial_categories_batch_loader import DialogueBatchLoader4Transformers



def main():
	# set up arguments
	parser = argparse.ArgumentParser()
	# model parameters
	parser.add_argument('--output_dir', type=str, default='')

	# dataset parameters
	parser.add_argument('--redial_dataset_path', type=str, default='Datasets/redial')
	parser.add_argument('--train_filename', type=str, default='train_data')
	parser.add_argument('--val_filename', type=str, default='valid_data')
	parser.add_argument('--test_filename', type=str, default='test_data.jsonl')
	parser.add_argument('--movie_details_filename', type=str, default='movie_details.csv')
	parser.add_argument('--conversation_length_limit', type=int, default=40)
	parser.add_argument('--utterance_length_limit', type=int, default=80)
	parser.add_argument("--input_length_limit", default=1024, type=int)


	# training parameters
	parser.add_argument("--learning_rate", default=5e-7, type=float,
						help="The initial learning rate for Adam.")

	parser.add_argument("--adam_epsilon", default=1e-8, type=float,
						help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float,
						help="Max gradient norm.")

	parser.add_argument('--patience', type=int, default=5)
	parser.add_argument("--num_train_epochs", default=100, type=int,
						help="Total number of training epochs to perform.")


	parser.add_argument("--conversations_per_batch", default=1, type=int, help="How many conversations should a batch contain.")
	parser.add_argument("--max_samples_per_gpu", default=1, type=int,
						help="If != -1: Splits a batch into minibatches of size equal to this. Was implemented in order to fit large models in small GPUs. Practically, it does not affect training!")


	parser.add_argument('--seed', type=int, default=42)

	parser.add_argument("--use_cuda", default="True", type=str)

	args = parser.parse_args()

	args.use_cuda = True if args.use_cuda == "True" and torch.cuda.is_available() else False

	output_dir = "Category_Preference_Model"



	if args.output_dir == "":
		args.output_dir = output_dir

	if os.path.isdir(args.output_dir):
		shutil.rmtree(args.output_dir)

	# create output directory
	os.mkdir(args.output_dir)

	# set up logging
	logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt = '%m/%d/%Y %H:%M:%S',
						level = logging.INFO,
						filename = os.path.join(args.output_dir, 'log.txt') )
	logger = logging.getLogger(__name__)

	logging.getLogger(transformers.__name__).setLevel(logging.ERROR)
	logging.getLogger(nltk.__name__).setLevel(logging.ERROR)

	logger.info(args)
	print(args)

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# instanciate batch loader
	batch_loader = DialogueBatchLoader4Transformers(conversations_per_batch=args.conversations_per_batch, \
	 max_input_length = args.input_length_limit, conversation_length_limit = args.conversation_length_limit, \
	 utterance_length_limit = args.utterance_length_limit, data_path=args.redial_dataset_path, \
	 train_path=args.train_filename, valid_path=args.val_filename, test_path=args.test_filename, \
	 movie_details_path=args.movie_details_filename)


	model = Cat_Pref_BERT(vocab_size = batch_loader.vocabulary_size, cat_size = len(batch_loader.categories), n_movies = batch_loader.n_movies, args = args)


	if args.use_cuda:
		model = model.cuda()

	best_eval_loss = 1e10
	patience_count = 0

	for epoch in range(args.num_train_epochs):

		# training
		train_loss = model.train_epoch(batch_loader = batch_loader)

		# validation
		eval_loss = model.evaluate_model(batch_loader = batch_loader, subset = "valid")

		logger.info("Epoch {:0>2d}: | TRAIN | Cat_Loss: {:.4f} RMSE | EVAL | Cat_Loss: {:.4f} RMSE".format(epoch, math.sqrt(train_loss), math.sqrt(eval_loss)))

		if eval_loss < best_eval_loss:
			best_eval_loss = eval_loss
			patience_count = 0
			torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pickle"))
		else:
			patience_count += 1
			if patience_count == args.patience:
				logger.info("Training terminated because maximum patience of {}, was exceeded".format(args.patience))
				break
							
	# evaluate on test set
	model.load_state_dict(torch.load(os.path.join(args.output_dir, "best.pickle")))

	# evaluate model
	eval_loss = model.evaluate_model(batch_loader = batch_loader, subset = "test")

	logger.info(" | TEST | Cat_Loss: {:.4f} RMSE".format(math.sqrt(eval_loss)))



if __name__ == '__main__':
	# with torch.cuda.device(3):
	main()







