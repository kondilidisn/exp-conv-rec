import os
import shutil
import csv
import random
from tqdm import tqdm, trange
from io import open
import jsonlines
from operator import add
# from functools import reduce
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

# print(transformers.__path__)
# exit()


# from transformers import BertTokenizer

from category_pref_model import Cat_Pref_BERT

# from models import HIBERTSemanticTransformer, HIBERTNLGTransformer, HIBERTJointFullTransformer


from redial_categories_batch_loader import DialogueBatchLoader4Transformers



def main():
	# set up arguments
	parser = argparse.ArgumentParser()
	# model parameters
	# parser.add_argument('--exp_dir', type=str, default='experiments')
	parser.add_argument('--output_dir', type=str, default='')
	# dataset parameters
	parser.add_argument('--redial_dataset_path', type=str, default='Datasets/redial')
	parser.add_argument('--train_filename', type=str, default='train_data')
	parser.add_argument('--val_filename', type=str, default='valid_data')
	parser.add_argument('--test_filename', type=str, default='test_data.jsonl')
	parser.add_argument('--movie_details_filename', type=str, default='movie_details.csv')
	parser.add_argument('--conversation_length_limit', type=int, default=40)
	parser.add_argument('--utterance_length_limit', type=int, default=80)



	# training parameters
	# parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
	#                     help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--learning_rate", default=5e-7, type=float,
						help="The initial learning rate for Adam.")
	# parser.add_argument("--weight_decay", default=0.0, type=float,
	#                     help="Weight decay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float,
						help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float,
						help="Max gradient norm.")
	# parser.add_argument("--max_steps", default=-1, type=int,
	#                     help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
	# parser.add_argument("--warmup_steps", default=0, type=int,
	#                     help="Linear warmup over warmup_steps.")


	# parser.add_argument('--max_train_epochs', type=int, default=100)
	parser.add_argument('--patience', type=int, default=5)
	parser.add_argument("--num_train_epochs", default=100, type=int,
						help="Total number of training epochs to perform.")


	parser.add_argument("--conversations_per_batch", default=1, type=int)
	parser.add_argument("--max_samples_per_gpu", default=1, type=int,
						help="If > 0: Splits a conversation into minibatches of size equal to this. Otherwise the batch is an integer of conversations as defined in --conversations_per_batch.")


	# parser.add_argument("--shuffle", default="True", type=str)


	# model parameters
	parser.add_argument("--input_length_limit", default=1024, type=int)


	# parser.add_argument("--hidden_size", default=64, type=int)
	# parser.add_argument("--num_hidden_layers", default=2, type=int)
	# parser.add_argument("--num_attention_heads", default=4, type=int)
	# parser.add_argument("--intermediate_size", default=256, type=int)

	parser.add_argument('--seed', type=int, default=42)


	# parser.add_argument("--cat_sa_alpha", default=1.0, type=float)
	# parser.add_argument("--sem_nlg_alpha", default=1.0, type=float)




	# parser.add_argument("--use_ground_truth", default=False, type=str)



	# parser.add_argument("--per_gpu_train_conversations_per_batch", default=4, type=int,
	#                     help="Batch size per GPU/CPU for training.")
	# parser.add_argument("--per_gpu_eval_conversations_per_batch", default=4, type=int,
	#                     help="Batch size per GPU/CPU for evaluation.")

	parser.add_argument("--use_cuda", default="True", type=str)


	# parser.add_argument("--CLS_mode", default="C_CLS_C_linears", type=str) # CLS_mode = "1_CLS" or "C_CLS_1_linear" or "C_CLS_C_linears"



	# parser.add_argument("--use_CLS_output", default="True", type=str)
	# parser.add_argument("--HIBERT", default="False", type=str)
	# parser.add_argument("--task", default="semantic", type=str)  # task = "semantic" or "nlg" or "pretrained" or "full"
	# parser.add_argument("--use_pretrained", default="True", type=str)


	parser.add_argument("--debug_run", default="False", type=str)


	args = parser.parse_args()

	# args.HIBERT = True if args.HIBERT == "True" else False
	# args.use_pretrained = True if args.use_pretrained == "True" else False
	# args.use_CLS_output = True if args.use_CLS_output == "True" else False
	# args.CLS_mode = True if args.CLS_mode == "True" else False
	args.use_cuda = True if args.use_cuda == "True" and torch.cuda.is_available() else False
	args.debug_run = True if args.debug_run == "True" else False
	# args.use_ground_truth = True if args.use_ground_truth == "True" else False

	# make sure that Experiments directory already exists (args.exp_dir)
	# os.makedirs(args.exp_dir, exist_ok=True)

	# checkpoint_handler, Model_Class, output_dir = init(args)

	# if 


	output_dir = "Category_Preference_Model"



	if args.output_dir == "":
		args.output_dir = output_dir


	if args.debug_run:
		args.output_dir += "_debug_run"
		args.conversations_per_batch = 2
		args.max_samples_per_gpu = 1
		args.num_train_epochs = 2
		# torch.set_num_threads(4)
		# OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 train.py --HIBERT False --debug_run True --use_cuda False
		# CUDA_LAUNCH_BLOCKING=1 

		# args.use_cuda = False


	# final_exp_dir = os.path.join(args.exp_dir , args.output_dir)
	# if os.path.exists(final_exp_dir) and not args.debug_run:
	# 	print("Experiment already exists, with experiment name:\n" + final_exp_dir + "\nSkipping Experiment!")
	# 	exit()



	# args.output_dir = "Developing..." + args.output_dir


	# args.output_dir = os.path.join(args.exp_dir , args.output_dir)


	if os.path.isdir(args.output_dir):
		shutil.rmtree(args.output_dir)

	# create output directory
	os.mkdir(args.output_dir)

	# set up logging
	logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt = '%m/%d/%Y %H:%M:%S',
						level = logging.INFO,
						filename = os.path.join(args.output_dir, 'log.txt') )
						# LOG_FILE_INFO  = os.path.join(args.output_dir, 'log.info'),
						# LOG_FILE_WARNING = os.path.join(args.output_dir, 'log.warning')):
	logger = logging.getLogger(__name__)

	logging.getLogger(transformers.__name__).setLevel(logging.ERROR)
	logging.getLogger(nltk.__name__).setLevel(logging.ERROR)

	# current code has only been implemented for 1 GPU
	args.n_gpu = 1

	logger.info(args)
	print(args)


	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# if we are using BERT, then we set the hidden size of the transformers to be equal to the BERT hidden size
	# if args.use_pretrained and args.HIBERT:
	# 	print("check BERT hidden state size")
	# 	print("You also need to fix the tokenizer !")
	# 	args.hidden_size = 768


	# instanciate batch loader
	# batch_loader = DialogueBatchLoader4Categories(sources="dialogue_to_categories ratings", conversations_per_batch=args.conversations_per_batch, max_input_length = args.input_length_limit)
	batch_loader = DialogueBatchLoader4Transformers(conversations_per_batch=args.conversations_per_batch, \
	 max_input_length = args.input_length_limit, conversation_length_limit = args.conversation_length_limit, \
	 utterance_length_limit = args.utterance_length_limit, data_path=args.redial_dataset_path, \
	 train_path=args.train_filename, valid_path=args.val_filename, test_path=args.test_filename, \
	 movie_details_path=args.movie_details_filename)


	# tokenizer = batch_loader.gpt2_tokenizer

	# enc = batch_loader.encode("and [CLS] Movie_Mentioned EOS three")

	# print(enc)

	# dec = batch_loader.decode(enc)

	# print(dec)
	# exit()

	model = Cat_Pref_BERT(vocab_size = batch_loader.vocabulary_size, cat_size = len(batch_loader.categories), n_movies = batch_loader.n_movies, args = args)

	# model = model.cuda()


	# for j in range(100):

	# 	torch.cuda.empty_cache()
	# 	t = torch.cuda.get_device_properties(3).total_memory
	# 	r = torch.cuda.memory_reserved(3) 
	# 	a = torch.cuda.memory_allocated(3)
	# 	f = r-a  # free inside reserved

	# 	torch.cuda.empty_cache()
	# 	print(f, r, a)



	# 	print("start, ", j)


	# 	model.optimizer.zero_grad()


	# 	torch.cuda.empty_cache()

	# 	contexts = np.full((2, 1000), fill_value = batch_loader.encode("PAD")[0], dtype=np.int64)

	# 	token_types = np.full((2, 1000), fill_value=0, dtype=np.int64)

	# 	attention_masks = np.full((2, args.input_length_limit), fill_value = 1, dtype=np.bool_)

	# 	category_targets = np.zeros((2, 19), dtype=np.float32)
	# 	# batch = []

	# 	contexts = torch.randint(low = 0, high = 30548, size = (2,args.input_length_limit)).cuda()
	# 	token_types = torch.randint(low = 0, high = 1, size = (2,args.input_length_limit)).cuda()
	# 	attention_masks = torch.tensor(attention_masks).cuda()
	# 	category_targets = torch.tensor(category_targets).cuda()


	# 	r = torch.cuda.memory_reserved(3) 
	# 	a = torch.cuda.memory_allocated(3)
	# 	f = r-a
	# 	print(f, r, a)
	# 	last_hidden_state = model.encoder(input_ids = contexts, attention_mask=attention_masks, token_type_ids=token_types)[0]

	# 	r = torch.cuda.memory_reserved(3) 
	# 	a = torch.cuda.memory_allocated(3)
	# 	f = r-a
	# 	print(f, r, a)
	# 	cls_input = last_hidden_state[:, : model.n_cls_tokens, :]
	# 	# pass each CLS hidden activation, through its corresponding trainable linear function
	# 	cat_pred = []
	# 	# cat_pred = torch.zeros_like(category_targets)
	# 	for i in range(model.n_cls_tokens):
	# 		cat_pred.append( model.cat_prediction[i]( cls_input[:,i, :] ) )
	# 	# bring the predicted category vectors to their final form
	# 	cat_pred = torch.stack( cat_pred, dim = 1).view(category_targets.size())



	# 	# pass the activations through softmax activation function
	# 	# cat_pred = torch.nn.functional.softmax( cat_pred, dim= -1)

	# 	# Use sigmoid instead
	# 	cat_pred = torch.sigmoid(cat_pred)


	# 	cat_mask = (category_targets != -1).view(category_targets.size())

	# 	if cat_mask.sum() == 0:
	# 		cat_loss = model.mse_loss(torch.ones(1), torch.ones(1))
	# 		if next(model.parameters()).is_cuda:
	# 			cat_loss = cat_loss.cuda()
	# 	else:
	# 		# cat_loss = self.mse_loss(cat_pred[cat_mask].view(-1), category_targets[cat_mask].view(-1))
	# 		cat_loss = model.mse_loss(cat_pred[cat_mask].view(-1), category_targets[cat_mask].view(-1))

	# 	r = torch.cuda.memory_reserved(3) 
	# 	a = torch.cuda.memory_allocated(3)
	# 	f = r-a
	# 	print(f, r, a)
	# 	torch.cuda.empty_cache()

	# 	r = torch.cuda.memory_reserved(3) 
	# 	a = torch.cuda.memory_allocated(3)
	# 	f = r-a
	# 	print(f, r, a)
	# 	print(cat_loss)
	# 	cat_loss.backward()


	# 	r = torch.cuda.memory_reserved(3) 
	# 	a = torch.cuda.memory_allocated(3)
	# 	f = r-a
	# 	print(f, r, a)
	# 	torch.cuda.empty_cache()

	# 	model.optimizer.step()

	# 	r = torch.cuda.memory_reserved(3) 
	# 	a = torch.cuda.memory_allocated(3)
	# 	f = r-a
	# 	print(f, r, a)
	# 	print("ok, ", j)


	# exit()


	# model.load_state_dict(torch.load( os.path.join("experiments/Developing...FLAT_semantic_1_CLS_CAT_SA_a_1.0_Pretrained_complete_samples_sigmoid_output_binary_cat_target_0.1_0.9", "best_Inter_Loss.pickle") ))
	# exit()

	if args.use_cuda:
		model = model.cuda()

	# model.initialize_optimizer_and_scheduler(n_train_batches = batch_loader.n_batches["train"])

	best_eval_loss = 1e10


	# evaluate model
	# eval_loss = model.evaluate_model(batch_loader = batch_loader, subset = "valid")


	# # exit()

	# if args.task != "semantic":
	# 	perplexity, blue_score = model.evaluate_nlg(batch_loader = batch_loader, subset = "valid")
	# else:
	# 	perplexity, blue_score = None, None


	patience_count = 0

	# train_losses = ( [0] * len(eval_losses[0]),  [0] * len(eval_losses[0]), -1 )

	# checkpoint_handler, train_results_text, criteria_improved = update_ckeckpoint_handler(checkpoint_handler = checkpoint_handler, task = args.task, train_losses = train_losses, eval_losses = eval_losses, epoch = -1, perplexity = perplexity, blue_score = blue_score)




	# for criterion in criteria_improved:
	# 	torch.save(model.state_dict(), os.path.join(args.output_dir, checkpoint_handler[criterion]["filename"]))

	# logger.info(train_results_text)


	for epoch in range(args.num_train_epochs):

		# training
		train_loss = model.train_epoch(batch_loader = batch_loader)

		# validation
		eval_loss = model.evaluate_model(batch_loader = batch_loader, subset = "valid")

		# if args.task != "semantic":
		# 	perplexity, blue_score = model.evaluate_nlg(batch_loader = batch_loader, subset = "valid")
		# else:
		# 	perplexity, blue_score = None, None


		# update checkpoint handler
		# checkpoint_handler, train_results_text, criteria_improved = update_ckeckpoint_handler(checkpoint_handler, args.task, train_losses, eval_losses, epoch, perplexity, blue_score)




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

	# for criterion in checkpoint_handler:
		# load best model for that criterion
	model.load_state_dict(torch.load(os.path.join(args.output_dir, "best.pickle")))

	# evaluate model
	eval_loss = model.evaluate_model(batch_loader = batch_loader, subset = "test")

	# make RMSE from MSE, for category losses
	# eval_loss = 



	logger.info(" | TEST | Cat_Loss: {:.4f} RMSE".format(math.sqrt(eval_loss)))


		# if args.task != "semantic":
		# 	perplexity, blue_score = model.evaluate_nlg(batch_loader = batch_loader, subset = "test")
		# else:
		# 	perplexity, blue_score = None, None



		# if args.task == "semantic":
		# 	checkpoint_handler[criterion]["test_results"] = "Epoch {:0>2d}: | TEST | Inter_Loss: {:.4f}, Norm_Cat_Loss: {:.4f}, Cat_Loss: {:.4f}, Norm_SA_Loss: {:.4f}, SA_Loss: {:.4f}" \
		# 			.format(checkpoint_handler[criterion]["epoch"], eval_interpol_loss, eval_normalized_losses[0], eval_losses[0], eval_normalized_losses[1], eval_losses[1])

		# elif args.task == "nlg":

		# 	checkpoint_handler[criterion]["test_results"] = "Epoch {:0>2d}: | TEST | Test_Loss: {:.4f}, Perplexity: {:.4f}, BLUE: {:.4f}" \
		# 				.format(checkpoint_handler[criterion]["epoch"], eval_losses[0] , perplexity, blue_score)
		# else:

		# 	checkpoint_handler[criterion]["test_results"] = "Epoch {:0>2d}: | TEST | Inter_Loss: {:.4f}, Norm_Cat_Loss: {:.4f}, Cat_Loss: {:.4f}, Norm_SA_Loss: {:.4f}, SA_Loss: {:.4f}, Norm_NLG_Loss: {:.4f}, NLG_Loss: {:.4f}, Perplexity: {:.4f}, BLUE: {:.4f}" \
		# 				.format(checkpoint_handler[criterion]["epoch"], eval_interpol_loss, eval_normalized_losses[0], eval_losses[0], eval_normalized_losses[1], eval_losses[1], eval_normalized_losses[2], eval_losses[2], perplexity, blue_score)


		# higher BLUE means higher performance, so we are trying to minimize the value (1 - BLUE). But we need to print the original value
		# best_value = 1 - checkpoint_handler[criterion]["best_value"] if criterion == "BLUE" else checkpoint_handler[criterion]["best_value"]

		# logger.info("")
		# logger.info("  "  + criterion + "::  Best value: {:.4f}, epoch: {:0>2d}".format(best_value, checkpoint_handler[criterion]["epoch"]))

		# logger.info("Train: " + checkpoint_handler[criterion]["train_results"])
		# logger.info("Test : " + checkpoint_handler[criterion]["test_results"])

		# results = args.output_dir + "\n" + checkpoint_handler[criterion]["train_results"] + "\n" + checkpoint_handler[criterion]["test_results"]

		# if args.use_cuda:
		# 	model = model.cpu()
		# 	torch.save(model.state_dict(), os.path.join(args.output_dir, "best"))
		# 	model = model.cuda()

		# f = open("results_filename.txt", "a")
		# f.write(results)
		# f.close()


	# if the final output directory already exists, then we delete it
	# final_output_dir = os.path.join(args.exp_dir,  args.output_dir[ len(args.exp_dir + "/Developing...") : ] )


	# if os.path.isdir(final_output_dir):
	# 	shutil.rmtree(final_output_dir)


	# os.renames(args.output_dir, final_output_dir)

	# after the program has completed training and evaluation, we transfer


	# source = '/path/to/source_folder'
	# dest1 = '/path/to/dest_folder'


	# files = os.listdir(source)

	# for f in files:
	#         shutil.move(source+f, dest1)

# def init(args):

# 	if args.HIBERT: 

# 		if args.task == "semantic":
# 			Model_Class = HIBERTSemanticTransformer
# 			checkpoint_criteria = ["Cat_Loss", "SA_Loss", "Inter_Loss"]

# 		# elif args.task == "nlg":
# 		# 	Model_Class = HIBERTNLGTransformer
# 		# 	checkpoint_criteria = ["Eval_Loss", "Perplexity", "BLUE"]


# 		else:

# 			checkpoint_criteria = ["Cat_Loss", "SA_Loss", "Inter_Loss", "Perplexity", "BLUE"]

# 			if args.task == "pretrained":
# 				Model_Class = HIBERTPretrainedFullTransformer
# 			else:
# 				Model_Class = HIBERTJointFullTransformer

# 	# Flat models
# 	else:

# 		if args.task == "semantic":
# 			Model_Class = FlatSemanticTransformer


# 			if args.cat_sa_alpha == 1.0 or args.cat_sa_alpha == 0:
# 				checkpoint_criteria = ["Inter_Loss"]
# 			else:
# 				checkpoint_criteria = ["Cat_Loss", "SA_Loss", "Inter_Loss"]

# 		elif args.task == "nlg":
# 			Model_Class = FlatNLGTransformer
# 			checkpoint_criteria = ["Eval_Loss", "Perplexity", "BLUE"]


# 		else:

# 			checkpoint_criteria = ["Cat_Loss", "SA_Loss", "Inter_Loss", "Perplexity", "BLUE"]

# 			if args.task == "pretrained":
# 				Model_Class = FlatPretrainedFullTransformer
# 			else:
# 				Model_Class = FlatJointFullTransformer


# 	checkpoint_handler = {}

# 	for criterion in checkpoint_criteria:
# 		checkpoint_handler[criterion] = {"best_value" : 1e10, "epoch" : 0, "train_results" : "","test_results" : "", "filename": "best_" + criterion + ".pickle"}



# 	# build output_dir, depending on the parameters
# 	# output_dir = "HIBERT" if args.HIBERT else "FLAT"
# 	# output_dir += "_" + args.task

# 	# if args.task == "semantic":
# 	# 	output_dir += "_" + args.CLS_mode
# 	# 	output_dir += "_CAT_SA_a_" + str(args.cat_sa_alpha)

# 	# if args.task == "full" or args.task == "pretrained":
# 	# 	output_dir += "_" + args.CLS_mode
# 	# 	output_dir += "_CAT_SA_a_" + str(args.cat_sa_alpha)
# 	# 	output_dir += "_SEM_NLG_a_" + str(args.sem_nlg_alpha)


# 	# if args.use_pretrained:
# 	# 	output_dir += "_Pretrained"
# 	# else:
# 	# 	output_dir += "_Hidden_" + str(args.hidden_size) + "_Layers_" + str(args.num_hidden_layers) + "_Heads_" + str(args.num_attention_heads) + "_Inter_" + str(args.intermediate_size)

# 	output_dir += "_BS_" + str(args.conversations_per_batch)# + "_miniBS_" + str(args.max_samples_per_gpu)

# 	return checkpoint_handler, Model_Class, output_dir



# def update_ckeckpoint_handler(checkpoint_handler, task, train_losses, eval_losses, epoch, perplexity = None, blue_score = None):


# 	if task == "semantic":

# 		norm_train_cat_loss = train_losses[0][0]
# 		norm_train_sa_loss = train_losses[0][1]
# 		norm_eval_cat_loss = eval_losses[0][0]
# 		norm_eval_sa_loss = eval_losses[0][1]

# 		# train_cat_loss = math.sqrt(train_losses[1][0])
# 		train_cat_loss = train_losses[1][0]
# 		train_sa_loss = train_losses[1][1]
# 		# eval_cat_loss = math.sqrt(eval_losses[1][0])
# 		eval_cat_loss = eval_losses[1][0]
# 		eval_sa_loss = eval_losses[1][1]

# 		train_interpol_loss = train_losses[2]
# 		eval_interpol_loss = eval_losses[2]

# 		checkpoint_criteria_values = {"Cat_Loss": eval_cat_loss, "SA_Loss": eval_sa_loss, "Inter_Loss": eval_interpol_loss}
# 		# checkpoint_criteria_values = {"eval_loss": eval_loss, "Perplexity": Perplexity, "BLUE": - gt_BLUE, "final_distinct_4":- gt_distinct_4_union, "final_distinct_3": - gt_distinct_3_union}

# 		train_results_text = "Epoch {:0>2d}: | TRAIN | Inter_Loss: {:.4f}, Norm_Cat_Loss: {:.4f}, Cat_Loss: {:.4f}, Norm_SA_Loss: {:.4f}, SA_Loss: {:.4f} \n | EVAL | Inter_Loss: {:.4f}, Norm_Cat_Loss: {:.4f}, Cat_Loss: {:.4f}, Norm_SA_Loss: {:.4f}, SA_Loss: {:.4f}" \
# 					.format(epoch, train_interpol_loss, norm_train_cat_loss, train_cat_loss, norm_train_sa_loss, train_sa_loss, eval_interpol_loss, norm_eval_cat_loss, eval_cat_loss, norm_eval_sa_loss, eval_sa_loss)

# 	elif task == "nlg":

# 		train_loss = train_losses[1][0]
# 		eval_loss = eval_losses[1][0]
# 		perplexity = perplexity
# 		blue_score = blue_score


# 		# higher BLUE means higher performance, so we are trying to minimize the value (1 - BLUE)
# 		checkpoint_criteria_values = {"Eval_Loss": eval_loss, "Perplexity": perplexity, "BLUE": 1 - blue_score}

# 		train_results_text = "Epoch {:0>2d}: | TRAIN | Train_Loss: {:.4f} | EVAL | Eval_Loss: {:.4f}, Perplexity: {:.4f}, BLUE: {:.4f}" \
# 					.format(epoch, train_loss, eval_loss, perplexity, blue_score)

# 	else:

# 		norm_train_cat_loss = train_losses[0][0]
# 		norm_train_sa_loss = train_losses[0][1]
# 		norm_eval_cat_loss = eval_losses[0][0]
# 		norm_eval_sa_loss = eval_losses[0][1]

# 		train_cat_loss = math.sqrt(train_losses[1][0])
# 		train_sa_loss = train_losses[1][1]
# 		eval_cat_loss = math.sqrt(eval_losses[1][0])
# 		eval_sa_loss = eval_losses[1][1]

# 		train_interpol_loss = train_losses[2]
# 		eval_interpol_loss = eval_losses[2]


# 		train_nlg_loss = train_losses[0][2]
# 		eval_nlg_loss = eval_losses[0][2]
# 		perplexity = perplexity
# 		blue_score = blue_score


# 		checkpoint_criteria_values = {"Cat_Loss": eval_cat_loss, "SA_Loss": eval_sa_loss, "Inter_Loss": eval_interpol_loss, "NLG_Loss": eval_nlg_loss, "Perplexity": perplexity, "BLUE": 1 - blue_score}
# 		# checkpoint_criteria_values = {"eval_loss": eval_loss, "Perplexity": Perplexity, "BLUE": - gt_BLUE, "final_distinct_4":- gt_distinct_4_union, "final_distinct_3": - gt_distinct_3_union}

# 		train_results_text = "Epoch {:0>2d}: | TRAIN | Inter_Loss: {:.4f}, Norm_Cat_Loss: {:.4f}, Cat_Loss: {:.4f}, Norm_SA_Loss: {:.4f}, SA_Loss: {:.4f}, NLG_Loss: {:.4f} \n | EVAL | Inter_Loss: {:.4f}, Norm_Cat_Loss: {:.4f}, Cat_Loss: {:.4f}, Norm_SA_Loss: {:.4f}, SA_Loss: {:.4f}, NLG_Loss: {:.4f}, Perplexity: {:.4f}, BLUE: {:.4f}" \
# 					.format(epoch, train_interpol_loss, norm_train_cat_loss, train_cat_loss, norm_train_sa_loss, train_sa_loss, train_nlg_loss, eval_interpol_loss, norm_eval_cat_loss, eval_cat_loss, norm_eval_sa_loss, eval_sa_loss, eval_nlg_loss, perplexity, blue_score)


# 	criteria_improved = []

# 	for criterion in checkpoint_handler:
# 		# print(epoch, " Criterion: ", criterion, ", loss: ", checkpoint_criteria_values[criterion], ", best_loss: ", checkpoint_handler[criterion]["best_value"])
# 		if checkpoint_criteria_values[criterion] < checkpoint_handler[criterion]["best_value"]:
# 			# reset universal patience value
# 			criteria_improved.append(criterion)
# 			# update best values for this criterion
# 			checkpoint_handler[criterion]["best_value"] = checkpoint_criteria_values[criterion]
# 			checkpoint_handler[criterion]["epoch"] = epoch
# 			checkpoint_handler[criterion]["train_results"] = train_results_text



# 	return checkpoint_handler, train_results_text, criteria_improved








if __name__ == '__main__':
	# with torch.cuda.device(3):
	main()







