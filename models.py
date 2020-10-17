import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import math
import datetime 
from multiprocessing.pool import ThreadPool
import os
import threading
import nltk

from tqdm import tqdm


import transformers


class TransformerEncoder(nn.Module):
	def __init__(self, vocab_size, cat_size, n_movies, args = None):
		super(TransformerEncoder, self).__init__()
		self.vocab_size = vocab_size
		self.cat_size = cat_size
		self.n_movies = n_movies
		self.CLS_mode = args.CLS_mode
		self.args = args

		self.cat_sa_alpha = args.cat_sa_alpha
		self.sem_nlg_alpha = args.sem_nlg_alpha

		if args.use_ground_truth == False:

			if args.use_pretrained:
				# self.encoder = transformers.GPT2Model.from_pretrained("gpt2") #, output_hidden_states=True

				self.encoder = transformers.BertModel.from_pretrained('bert-base-uncased')

				# if requested, we have to extend the parameters of the positional embeddings, so that the input length limit can be increased.
				if args.input_length_limit > 512:
					# instantiate new extended positional embedding parameters
					new_positional_embeddings = nn.Embedding(num_embeddings=args.input_length_limit, embedding_dim=768)
					# copy pretrained parameters of existing positions (<=512)
					new_positional_embeddings.weight[:512, :] = self.encoder.embeddings.position_embeddings.weight
					# replace paramters in model
					self.encoder.embeddings.position_embeddings.weight = nn.Parameter(new_positional_embeddings.weight)

					self.encoder.embeddings.position_embeddings.num_embeddings = args.input_length_limit

					# update parameter's value on BERT's config
					self.encoder.config.max_position_embeddings = args.input_length_limit

				# extend vocab size of the model accordingly as well
				self.encoder._resize_token_embeddings(new_num_tokens = vocab_size)
			else:
				# print("vocab_size :", vocab_size)
				# print("input_length_limit :", args.input_length_limit)
				# print("hidden_size :", args.hidden_size)
				# config = transformers.GPT2Config(vocab_size_or_config_json_file = vocab_size, n_positions = args.input_length_limit, n_ctx = args.input_length_limit, n_embd = args.hidden_size,
				#                                  n_layer = args.num_hidden_layers, n_head = args.num_attention_heads)


				config = transformers.BertConfig(vocab_size_or_config_json_file = vocab_size, hidden_size = args.hidden_size, num_hidden_layers = args.num_hidden_layers,
												num_attention_heads = args.num_attention_heads, intermediate_size = args.intermediate_size, max_position_embeddings = args.input_length_limit)
				# config.output_attentions = True


				# config.output_past = True
				# config.output_hidden_states=True

				# # self.encoder = transformers.GPT2Model(config)

				# (vocab_size_or_config_json_file=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1,
				#     attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, **kwargs


				self.encoder = transformers.BertModel(config)



			self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=args.adam_epsilon, weight_decay=0, amsgrad=False)




		# set up loss functions
		self.mse_loss = torch.nn.MSELoss(reduction='mean')
		# using same weights with ReDial SA model [1. / 5, 1. / 80, 1. / 15] (Based on label frequency distribution (4.9%, 81%, 14%))
		self.SA_crossE_Loss = torch.nn.CrossEntropyLoss(weight = torch.Tensor([1. / 5, 1. / 80, 1. / 15]), ignore_index=-1, reduction="mean")
		self.CrossE_Loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

		# self.config = self.encoder.config


	# def initialize_optimizer_and_scheduler(self, n_train_batches):

	#     if self.args.max_steps > 0:
	#         t_total = self.args.max_steps
	#         self.args.num_train_epochs = self.args.max_steps // (n_train_batches // self.args.gradient_accumulation_steps) + 1
	#     else:
	#         t_total = n_train_batches // self.args.gradient_accumulation_steps * self.args.num_train_epochs


	#         # optimezer, scheduler# Prepare optimizer and schedule (linear warmup and decay)
	#     no_decay = ['bias', 'LayerNorm.weight']
	#     optimizer_grouped_parameters = [
	#         {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
	#         {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	#         ]
	#     self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
	#     self.scheduler = transformers.WarmupLinearSchedule(self.optimizer, warmup_steps=self.args.warmup_steps, t_total=t_total)





	def get_movie_ratings_vector(self, sa_pred, batch_movie_mentions):
		movie_rating_vector = torch.zeros(sa_pred.size(0), self.n_movies)

		# used the liked probability, so that it is directly comperable with the ReDial baseline
		# liked_pred = sa_pred[:,:,1]
		liked_pred = sa_pred
		# otherwise, you can apply softmax only on the first two dimensions, redistributing the probability of "did not say", to the other two cases (liked/disliked)

		if sa_pred.is_cuda:
			movie_rating_vector = movie_rating_vector.cuda()

		for i in range(len(batch_movie_mentions)):
			for index, movie_id in batch_movie_mentions[i]:
				movie_rating_vector[i, movie_id] = liked_pred[i, index]

		return movie_rating_vector


	# def preprocess_losses(self, losses):
	#     # We preprocess the losses individually
	#     updated_losses = []
	#     for loss in losses:
	#         if self.args.n_gpu > 1:
	#             loss = loss.mean()  # mean() to average on multi-gpu parallel training
	#         if self.args.gradient_accumulation_steps > 1:
	#             loss = loss / self.args.gradient_accumulation_steps
	#         updated_losses.append(loss)

	#     return updated_losses

	def split_batch_to_minibatches(self, batch):

		if self.args.max_samples_per_gpu == -1:
			return [batch]

		# calculate the number of minibatches so that the maximum number of samples per gpu is maintained
		size_of_minibatch = self.args.max_samples_per_gpu * self.args.n_gpu

		# calculate number of minibatches
		number_of_samples_in_batch = batch["contexts"].size(0)
		number_of_minibatches = number_of_samples_in_batch // size_of_minibatch
		# check if there is an incomplete last minibatch
		if number_of_samples_in_batch % size_of_minibatch != 0:
			number_of_minibatches += 1

		# arrange the minibatches
		minibatches = []
		for i in range(number_of_minibatches):
			minibatch = {}
			for key in batch:
				temp = batch[key][ i * size_of_minibatch : (i+1) * size_of_minibatch ]
				minibatch[key] = temp
			minibatches.append(minibatch)

		return minibatches




	def make_batch_multiple_of_GPUs_for_DataParallel(self, batch):

		if self.args.n_gpu == 1:
			return batch

		if self.args.HIBERT:
			return self.HIBERT_make_batch_multiple_of_GPUs_for_DataParallel(batch)
		else:
			return self.FLAT_make_batch_multiple_of_GPUs_for_DataParallel(batch)


	def HIBERT_make_batch_multiple_of_GPUs_for_DataParallel(self, batch):

		number_of_samples = batch["contexts"].size(0)
		samples_to_pad = self.args.n_gpu - number_of_samples % self.args.n_gpu

		# contexts, context_attentions, category_targets, nlg_targets, nlg_gt_inputs, pool_hidden_representations_mask, dialogue_trans_positional_embeddings, \
		# sentiment_analysis_targets, dialogue_trans_token_type_ids, nlg_dialogue_mask_tokens, dialogue_trans_attentions, CLS_pooled_tokens_mask, batch_movie_mentions, complete_sample_movie_targets = batch

		contexts = batch["contexts"]
		context_attentions = batch["context_attentions"]
		category_targets = batch["category_targets"]
		nlg_targets = batch["nlg_targets"]
		nlg_gt_inputs = batch["nlg_gt_inputs"]
		pool_hidden_representations_mask = batch["pool_hidden_representations_mask"]
		dialogue_trans_positional_embeddings = batch["dialogue_trans_positional_embeddings"]
		sentiment_analysis_targets = batch["sentiment_analysis_targets"]
		dialogue_trans_token_type_ids = batch["dialogue_trans_token_type_ids"]
		nlg_dialogue_mask_tokens = batch["nlg_dialogue_mask_tokens"]
		dialogue_trans_attentions = batch["dialogue_trans_attentions"]
		CLS_pooled_tokens_mask = batch["CLS_pooled_tokens_mask"]
		batch_movie_mentions = batch["batch_movie_mentions"]
		complete_sample_movie_targets = batch["complete_sample_movie_targets"]



		# pad with zeros
		pad_with_last = [ contexts, context_attentions, nlg_gt_inputs, pool_hidden_representations_mask, \
					dialogue_trans_positional_embeddings, dialogue_trans_token_type_ids, nlg_dialogue_mask_tokens, dialogue_trans_attentions, CLS_pooled_tokens_mask]
		# pad with -1
		pad_with_minus_1 = [ category_targets, sentiment_analysis_targets, nlg_targets]

		# print(category_targets.size())


		for i, tensor in enumerate(pad_with_last):
			last_sample = tensor[-1].clone().unsqueeze(0)

			number_of_dimentions = len(list(tensor.size()))
			repeat_sizes = [samples_to_pad] + [1] * ( number_of_dimentions -1)
			repeat_sizes = tuple(repeat_sizes)
			pad = last_sample.repeat(repeat_sizes).type(tensor.dtype)

			pad_with_last[i] = torch.cat((tensor, pad), dim = 0)

			# number_of_dimentions = len(list(tensor.size()))
			# repeat_sizes = [samples_to_pad] + [1] * ( number_of_dimentions -1)
			# repeat_sizes = tuple(repeat_sizes)
			# pad = torch.zeros_like(tensor[-1,:]).repeat(repeat_sizes).type(tensor.dtype)
			# pad_with_last[i] = torch.cat((tensor, pad), dim = 0)



		for i, tensor in enumerate(pad_with_minus_1):
			# set 
			number_of_dimentions = len(list(tensor.size()))
			repeat_sizes = [samples_to_pad] + [1] * ( number_of_dimentions -1)
			repeat_sizes = tuple(repeat_sizes)
			pad = - torch.ones_like(tensor[-1,:]).repeat(repeat_sizes).type(tensor.dtype)
			pad_with_minus_1[i] = torch.cat((tensor, pad), dim = 0)



		contexts, context_attentions, nlg_gt_inputs, pool_hidden_representations_mask, \
					dialogue_trans_positional_embeddings, dialogue_trans_token_type_ids, nlg_dialogue_mask_tokens, dialogue_trans_attentions, CLS_pooled_tokens_mask = pad_with_last

		category_targets, sentiment_analysis_targets, nlg_targets = pad_with_minus_1

		# batch  =  contexts, context_attentions, category_targets, nlg_targets, nlg_gt_inputs, pool_hidden_representations_mask, dialogue_trans_positional_embeddings, \
		# sentiment_analysis_targets, dialogue_trans_token_type_ids, nlg_dialogue_mask_tokens, dialogue_trans_attentions, CLS_pooled_tokens_mask, batch_movie_mentions, complete_sample_movie_targets


		batch["contexts"] = contexts 
		batch["context_attentions"] = context_attentions 
		batch["category_targets"] = category_targets 
		batch["nlg_targets"] = nlg_targets 
		batch["nlg_gt_inputs"] = nlg_gt_inputs 
		batch["pool_hidden_representations_mask"] = pool_hidden_representations_mask 
		batch["dialogue_trans_positional_embeddings"] = dialogue_trans_positional_embeddings 
		batch["sentiment_analysis_targets"] = sentiment_analysis_targets 
		batch["dialogue_trans_token_type_ids"] = dialogue_trans_token_type_ids 
		batch["nlg_dialogue_mask_tokens"] = nlg_dialogue_mask_tokens 
		batch["dialogue_trans_attentions"] = dialogue_trans_attentions 
		batch["CLS_pooled_tokens_mask"] = CLS_pooled_tokens_mask 
		batch["batch_movie_mentions"] = batch_movie_mentions 
		batch["complete_sample_movie_targets"] = complete_sample_movie_targets 

		return batch


	def FLAT_make_batch_multiple_of_GPUs_for_DataParallel(self, batch):


		number_of_samples = batch["contexts"].size(0)
		samples_to_pad = self.args.n_gpu - number_of_samples % self.args.n_gpu



		contexts = batch["contexts"]
		token_types = batch["token_types"]
		attention_masks = batch["attention_masks"]
		category_targets = batch["category_targets"]
		sentiment_analysis_targets = batch["sentiment_analysis_targets"]
		nlg_targets = batch["nlg_targets"]
		nlg_gt_inputs = batch["nlg_gt_inputs"]
		batch_movie_mentions = batch["batch_movie_mentions"]
		complete_sample_movie_targets = batch["complete_sample_movie_targets"]


		# contexts, token_types, attention_masks, category_targets, sentiment_analysis_targets, nlg_targets, nlg_gt_inputs, batch_movie_mentions, complete_sample_movie_targets = batch

		# pad with zeros
		pad_with_zeros = [ contexts, token_types, attention_masks, nlg_gt_inputs]
		# pad with -1
		pad_with_minus_1 = [ category_targets, sentiment_analysis_targets, nlg_targets]


		for i, tensor in enumerate(pad_with_zeros):
			number_of_dimentions = len(list(tensor.size()))
			repeat_sizes = [samples_to_pad] + [1] * ( number_of_dimentions -1)
			repeat_sizes = tuple(repeat_sizes)
			pad = torch.zeros_like(tensor[-1,:]).repeat(repeat_sizes).type(tensor.dtype)
			pad_with_zeros[i] = torch.cat((tensor, pad), dim = 0)



		for i, tensor in enumerate(pad_with_minus_1):
			# set 
			number_of_dimentions = len(list(tensor.size()))
			repeat_sizes = [samples_to_pad] + [1] * ( number_of_dimentions -1)
			repeat_sizes = tuple(repeat_sizes)
			pad = - torch.ones_like(tensor[-1,:]).repeat(repeat_sizes).type(tensor.dtype)
			pad_with_minus_1[i] = torch.cat((tensor, pad), dim = 0)


		contexts, token_types, attention_masks, nlg_gt_inputs = pad_with_zeros

		category_targets, sentiment_analysis_targets, nlg_targets = pad_with_minus_1

		# batch = contexts, token_types, attention_masks, category_targets, sentiment_analysis_targets, nlg_targets, nlg_gt_inputs, batch_movie_mentions, complete_sample_movie_targets


		batch["contexts"] = contexts
		batch["token_types"] = token_types
		batch["attention_masks"] = attention_masks
		batch["category_targets"] = category_targets
		batch["sentiment_analysis_targets"] = sentiment_analysis_targets
		batch["nlg_targets"] = nlg_targets
		batch["nlg_gt_inputs"] = nlg_gt_inputs
		batch["batch_movie_mentions"] = batch_movie_mentions
		batch["complete_sample_movie_targets"] = complete_sample_movie_targets


		return batch




	def normalize_losses(self, losses):

		# if we are doing the semantic task, and we only pay attention to 1 of the tasks then we only return the loss of this task NOT NORMALIZED
		if len(losses) == 2 and (self.cat_sa_alpha == 0 or self.cat_sa_alpha == 1):
			return losses

		normalized_losses = []
		for i, loss in enumerate(losses):
			# print(loss)
			normalized_loss = (losses[i] - self.min_max_losses[i][0]) / (self.min_max_losses[i][1] - self.min_max_losses[i][0])
			# print("Normalized :", normalized_loss)
			normalized_losses.append(normalized_loss)

		return normalized_losses

	def interpolate_losses(self, losses):
		# we interpolate the losses

		# if we are doing the nlg task
		if len(losses) == 1:
			return losses[0]
		# if we are doing the semantic task
		elif len(losses) == 2:
			return losses[0] * self.cat_sa_alpha + losses[1] * ( 1 - self.cat_sa_alpha )
		# if we are doing both tasks
		else:
			return ( losses[0] * self.cat_sa_alpha + losses[1] * ( 1 - self.cat_sa_alpha ) ) * self.sem_nlg_alpha + losses[2] * ( 1 - self.sem_nlg_alpha )



	# forward function, splits batch to minibaches, concatentes final outputs, and normalizes losses over batch, w.r.t. number of targets per task
	def forward_batch(self, batch, train = False):
		# will be used for appropriatly averaging the losses, from the minibatches, depending on the number of sampels per minibatch

		batch_normalized_losses = [0, 0, 0]
		batch_losses = [0, 0, 0]
		batch_outputs = [[], [], []]
		batch_interpolated_loss = 0


		batch_SA_targets = (batch["sentiment_analysis_targets"] != -1).sum()

		batch_size = batch["contexts"].size(0)

		minibatches = self.split_batch_to_minibatches(batch)

		for minibatch in minibatches:

			minibatch_size = minibatch["contexts"].size(0)

			minibatch_SA_targets = (minibatch["sentiment_analysis_targets"] != -1).sum()

			minibatch_to_batch_ratio = minibatch_size / batch_size

			if next(self.parameters()).is_cuda and self.args.n_gpu > 1:
				minibatch = self.make_batch_multiple_of_GPUs_for_DataParallel(minibatch)

			outputs, losses = self.forward(minibatch)

			# normalize Category loses so that the represent tha minibatch w.r.t. the complete batch
			losses[0] *= minibatch_to_batch_ratio

			# normalize SA loss of minibatch by the number of SA targets on the minibatch (w.r.t. batch SA targets)
			losses[1] *= minibatch_SA_targets.float() / batch_SA_targets.float()

			# record the unnormalized losses
			for i in range(len(losses)):
				batch_losses[i] += losses[i].item()*minibatch_to_batch_ratio

			# if the task at hand has more than one loss (involves two or more tasks), then we normalize the losses
			if len(losses) != 1 and train:
				losses = self.normalize_losses(losses)

			loss = self.interpolate_losses(losses)

			loss *= minibatch_to_batch_ratio

			batch_interpolated_loss += loss.item()

			for i in range(len(losses)):
				# append the losses of the batch
				batch_normalized_losses[i] += losses[i].item()
				# register outputs
				batch_outputs[i].append(outputs[i])

			# some minibatches might not have any targets (SA mainly)
			if loss != 0 and train:
				# print(loss)
				loss.backward()

		if train:
			self.optimizer.step()

		# we remove the lists that are not being used
		while len(batch_normalized_losses) != len(losses):
			del batch_normalized_losses[-1]
			del batch_outputs[-1]
			del batch_losses[-1]

		for i in range(len(batch_outputs)):
			# concatenate outputs of minibatches
			batch_outputs[i] = torch.cat(batch_outputs[i], dim = 0)

		return batch_outputs, batch_losses, batch_normalized_losses, batch_interpolated_loss




	def evaluate_model(self, batch_loader, subset):

		n_batches = batch_loader.n_batches[subset]

		if self.args.debug_run:
			n_batches = 3

		self.min_max_losses = torch.Tensor([[1e10, 0], [1e10, 0], [1e10, 0]])
		
		if next(self.parameters()).is_cuda:
			self.min_max_losses = self.min_max_losses.cuda()

		# total_losses = [[], [], []]

		with torch.no_grad():

			# for step in range(n_batches):
			for step in tqdm(range(n_batches)):

				torch.cuda.empty_cache()

				# we retrieve a batch
				batch = batch_loader.load_batch(subset = subset, complete = True)

				if batch == None:
					continue

				_, batch_losses, _, _ = self.forward_batch(batch)

				# we keep track of minimum and maximum loss values per task, in oreder to later normalize for joint task training
				for i, loss in enumerate(batch_losses):
					if loss < self.min_max_losses[i][0]:
						self.min_max_losses[i][0] = loss
					if loss > self.min_max_losses[i][1]:
						self.min_max_losses[i][1] = loss

					# total_losses[i].append(loss.item())

		# we remove the total_losses that are not being used
		# while len(total_losses[-1]) == 0:
		#     del total_losses[-1]


		# # we average the losses and normalize them
		# average_losses = []
		# for i, loss in enumerate(total_losses):
		#     average_losses.append(torch.FloatTensor(total_losses[i]).mean())

		normalized_losses = self.normalize_losses(batch_losses)

		# average_losses = [ loss.item() for loss in average_losses]

		interpolated_loss = self.interpolate_losses(normalized_losses)

		return normalized_losses, batch_losses, interpolated_loss



	def train_epoch(self, batch_loader):


		n_batches = batch_loader.n_batches["train"]

		if self.args.debug_run:
			n_batches = 3

		# total_losses = [[], [], []]

		interpolated_losses = []

		# for step in range(n_batches):
		for step in tqdm(range(n_batches)):

			self.optimizer.zero_grad()

			# we retrieve a batch
			batch = batch_loader.load_batch(subset = "train", complete = True)

			if batch == None:
				continue

			# batch_outputs, batch_losses
			# batch_outputs, batch_losses, batch_interpolated_loss
			_, batch_losses, batch_normalized_losses, batch_interpolated_loss = self.forward_batch(batch, train=True)

			# # we store all losses
			# for i, loss in enumerate(batch_losses):
			#     total_losses[i].append(loss.item())

			# # if the task at hand has more than one loss (involves two or more tasks), then we normalize the losses
			# if len(batch_losses) != 1:
			#     batch_losses = self.normalize_losses(batch_losses)

			# batch_loss = self.interpolate_losses(batch_losses)

			# interpolated_losses.append(batch_loss.item())

			# batch_loss.backward()

			# # perform update step
			# torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)
			
			# self.optimizer.step()

			# torch.cuda.empty_cache()

		# we remove the total_losses that are not being used
		# while len(total_losses[-1]) == 0:
		#     del total_losses[-1]

		# # we average the losses and normalize them
		# average_losses = []
		# for i, loss in enumerate(total_losses):
		#     average_losses.append(torch.FloatTensor(total_losses[i]).mean().item())

		return  batch_normalized_losses, batch_losses, batch_interpolated_loss



	def evaluate_nlg(self, batch_loader, subset):

		n_batches = batch_loader.n_batches[subset]

		if self.args.debug_run:
			n_batches = 3

		perplexities = []

		bleu_scores = []

		with torch.no_grad():

			# for step in range(n_batches):
			for step in tqdm(range(n_batches)):

				# retrieve a batch
				batch = batch_loader.load_batch(subset = subset)

				if batch == None:
					continue

				minibatches = self.split_batch_to_minibatches(batch)

				for minibatch in minibatches:

					torch.cuda.empty_cache()

					if next(self.parameters()).is_cuda and self.args.n_gpu > 1:
						minibatch = self.make_batch_multiple_of_GPUs_for_DataParallel(minibatch)

					# we do the forward pass
					batch_perplexities, batch_bleu_scores = self(minibatch, eval_mode = True)


					perplexities += batch_perplexities
					bleu_scores += batch_bleu_scores

		# average the perplexity and the blue scores
		perplexity = np.sum(perplexities) / len(perplexities)
		bleu_score = np.sum(bleu_scores) / len(bleu_scores)

		return perplexity, bleu_score


	def get_blue_score(self, gt_sentence = [], generated_sentence = []):

		max_n_gram = min(len(gt_sentence), len(generated_sentence))
		max_n_gram = min ( max_n_gram, 4)

		weights = [ 1 / max_n_gram for i in range(max_n_gram)]
		weights = tuple(weights)

		sentence_blue_score = nltk.translate.bleu_score.sentence_bleu([gt_sentence], generated_sentence, weights = weights)

		return sentence_blue_score


# the forward functions always returns two lists. A list of outputs, and a list of losses


class FlatSemanticTransformer(TransformerEncoder):
	def __init__(self, vocab_size, cat_size, n_movies, args = None):
		super(FlatSemanticTransformer, self).__init__(vocab_size, cat_size, n_movies, args)

		if self.args.use_ground_truth == False:

			if self.CLS_mode == "1_CLS":
				self.n_cls_tokens = 1
				self.cat_prediction = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=self.cat_size)

			elif self.CLS_mode == "C_CLS_1_linear":
				self.n_cls_tokens = self.cat_size

				# have ONE trainable linear function for ALL CLS token (functioning only as a switch mechanism)
				self.cat_prediction = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1)

			elif self.CLS_mode == "C_CLS_C_linears":
				self.n_cls_tokens = self.cat_size
				# have a trainable linear function for each CLS token
				self.cat_prediction = nn.ModuleList()
				for i in range(self.n_cls_tokens):
					self.cat_prediction.append(nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1))


			# self.sentiment_analysis_prediction = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1)
			self.sentiment_analysis_prediction = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=3)


		if self.args.use_cuda and self.args.n_gpu > 1:
			self.encoder = torch.nn.DataParallel(self.encoder)


	def forward(self, batch):


		contexts = batch["contexts"]
		token_types = batch["token_types"]
		attention_masks = batch["attention_masks"]
		category_targets = batch["category_targets"]
		sentiment_analysis_targets = batch["sentiment_analysis_targets"]
		# nlg_targets = batch["nlg_targets"]
		# nlg_gt_inputs = batch["nlg_gt_inputs"]

		# mask = sentiment_analysis_targets != -1

		# print(sentiment_analysis_targets[mask])

		# contexts, token_types, attention_masks, category_targets, sentiment_analysis_targets, _, _, batch_movie_mentions, _ = batch

		# if the model is on cuda, we transfer all tensors to cuda
		if next(self.parameters()).is_cuda:
			contexts, token_types, attention_masks, category_targets, sentiment_analysis_targets = \
			 contexts.cuda(), token_types.cuda(), attention_masks.cuda(), category_targets.cuda(), sentiment_analysis_targets.cuda()



		# print("Contexts :", contexts.size())
		# print("attention_masks :", attention_masks.size())
		# print("token_types :", token_types.size())
		# print()


		last_hidden_state, _ = self.encoder(input_ids = contexts, attention_mask=attention_masks, token_type_ids=token_types)


		# we only use the hidden states of the last layer
		# hidden_states = last_hidden_state # hidden_states[-1]


		# we only use the first CLS tokens as cat prediction input
		cls_input = last_hidden_state[:, 0: self.n_cls_tokens, :]

		# print("CLS_input:")
		# print(cls_input.size())
		print(cls_input.sum(dim=-1))



		if self.CLS_mode == "1_CLS":
			cat_pred = self.cat_prediction(cls_input).view(category_targets.size())

		elif self.CLS_mode == "C_CLS_1_linear":
			# rearrange size so that all CLS hidden activations go through the same linear
			cls_input = cls_input.reshape(-1, self.encoder.config.hidden_size)
			cat_pred = self.cat_prediction(cls_input)
			# rearrange size so that |C| activtions go for each sample
			cat_pred = cat_pred.view(category_targets.size())

		elif self.CLS_mode == "C_CLS_C_linears":
			# pass each CLS hidden activation, through its corresponding trainable linear function
			cat_pred = []
			# cat_pred = torch.zeros_like(category_targets)
			for i in range(self.n_cls_tokens):
				cat_pred.append( self.cat_prediction[i]( cls_input[:,i, :] ) )
			# bring the predicted category vectors to their final form
			cat_pred = torch.stack( cat_pred, dim = 1).view(category_targets.size())



		# pass the activations through softmax activation function
		# cat_pred = torch.nn.functional.softmax( cat_pred, dim= -1)

		# Use sigmoid instead
		cat_pred = torch.sigmoid(cat_pred)


		# print("cat_pred:")
		# print(cat_pred.size())
		# print("category_targets")
		# print(category_targets.size())
		# exit()

		# print(cat_pred.size())
		# print(category_targets.size())
		# assert cat_pred.size() == category_targets.size()
		# exit()


		# # we get the category prediction
		# if self.CLS_mode:
		#     cat_pred = torch.nn.functional.softmax( self.cat_prediction(cls_input), dim= -1).view(category_targets.size())
		# else:
		#     cat_pred = torch.zeros_like(category_targets)
		#     for i in range(self.n_cls_tokens):
		#         cat_pred[:, i] = self.cat_prediction[i]( cls_input[:,i, :] ).view(cat_pred[:, i].size())
		#     cat_pred = torch.nn.functional.softmax( cat_pred, dim= -1).view(category_targets.size())

		# we get sentiment analysis prediction
		# sa_pred = torch.sigmoid( self.sentiment_analysis_prediction(last_hidden_state).view(sentiment_analysis_targets.size()) )
		sa_pred = self.sentiment_analysis_prediction(last_hidden_state) 

		cat_mask = (category_targets != -1).view(category_targets.size())

		if cat_mask.sum() == 0:
			cat_loss = self.mse_loss(torch.ones(1), torch.ones(1))
			if next(self.parameters()).is_cuda:
				cat_loss = cat_loss.cuda()
		else:
			# cat_loss = self.mse_loss(cat_pred[cat_mask].view(-1), category_targets[cat_mask].view(-1))
			cat_loss = self.mse_loss(cat_pred[cat_mask].view(-1), category_targets[cat_mask].view(-1))

		# if category_targets is not None:
		# if sentiment_analysis_targets is not None:
			# only use the outputs for which there are targets
		sa_mask = (sentiment_analysis_targets != -1).view(sentiment_analysis_targets.size())


		# print(sa_pred.size())
		# print(sentiment_analysis_targets.size())
		# print(sa_mask.sum())
		# print(torch.tensor([0,1]))
		# print()


		# if there are no SA targets, then we do not calculate the SA loss, and set it to 0 (because it returns nan with empty tensors)
		if sa_mask.sum() == 0:
			sa_loss = self.mse_loss(torch.ones(1), torch.ones(1))
			if next(self.parameters()).is_cuda:
				sa_loss = sa_loss.cuda()
		else:
			# print("Predicted :")
			# print(sa_pred[sa_mask])
			# print("Targets :")
			# print(sentiment_analysis_targets[sa_mask])
			# print("Cat loss:", sa_pred[sa_mask].size(), sentiment_analysis_targets[sa_mask].size())
			# sa_loss = self.mse_loss(sa_pred[sa_mask].view(-1), sentiment_analysis_targets[sa_mask].view(-1))
			# print("SA Output :", sa_pred[sa_mask])
			# print("SA Targets :", sentiment_analysis_targets[sa_mask])
			sa_loss = self.SA_crossE_Loss(sa_pred[sa_mask], sentiment_analysis_targets[sa_mask].long())

			# print("SA Loss:", sa_loss)



		# # if there are no SA targets, then we do not calculate the SA loss, and set it to 0 (because it returns nan with empty tensors)
		# if sa_mask.sum() == 0:
		#     sa_loss = self.CrossE_Loss(torch.tensor([0,1]), torch.ones(1))
		#     if next(self.parameters()).is_cuda:
		#         sa_loss = sa_loss.cuda()
		# else:
		# apply softmax on SA prediction
		sa_pred = torch.nn.functional.softmax( sa_pred, dim= -1)

		# Rating prediction is equal to liked probability - the disliked probability in [-1,1]
		rating_pred = sa_pred[:,:,1] - sa_pred[:,:,0]

		return [cat_pred, rating_pred], [cat_loss, sa_loss]

		# || Train this class with interleaving parameters on loss
		# make a train function
		# make an eval function (make them on TransformerEncoder class, and use the appropriate forwards, self.alpha will affect the inter)


	def forward_with_attentions(self, batch):

		contexts = batch["contexts"]
		token_types = batch["token_types"]
		attention_masks = batch["attention_masks"]
		category_targets = batch["category_targets"]
		sentiment_analysis_targets = batch["sentiment_analysis_targets"]
		# nlg_targets = batch["nlg_targets"]
		# nlg_gt_inputs = batch["nlg_gt_inputs"]

		# contexts, token_types, attention_masks, category_targets, sentiment_analysis_targets, _, _, batch_movie_mentions, _ = batch

		# if the model is on cuda, we transfer all tensors to cuda
		if next(self.parameters()).is_cuda:
			contexts, token_types, attention_masks, category_targets, sentiment_analysis_targets = \
			 contexts.cuda(), token_types.cuda(), attention_masks.cuda(), category_targets.cuda(), sentiment_analysis_targets.cuda()

		# print("Contexts :", contexts.size())
		# print("Contexts :", contexts)
		# print("attention_masks :", attention_masks.size())
		# print("token_types :", token_types.size())


		last_hidden_state, past, attentions = self.encoder(input_ids = contexts, attention_mask=attention_masks, token_type_ids=token_types)


		# we only use the hidden states of the last layer
		# hidden_states = last_hidden_state # hidden_states[-1]

		self.mse_loss = torch.nn.MSELoss(reduction='mean')

		# we only use the first CLS tokens as cat prediction input
		cls_input = last_hidden_state[:, 0: self.n_cls_tokens, :]

		# we get the category prediction
		if self.CLS_mode:
			cat_pred = torch.nn.functional.softmax( self.cat_prediction(cls_input), dim= -1).view(category_targets.size())
		else:
			cat_pred = torch.zeros_like(category_targets)
			for i in range(self.n_cls_tokens):
				cat_pred[:, i] = self.cat_prediction[i]( cls_input[:,i, :] ).view(cat_pred[:, i].size())
			cat_pred = torch.nn.functional.softmax( cat_pred, dim= -1).view(category_targets.size())

		# we get sentiment analysis prediction
		sa_pred = torch.sigmoid(self.sentiment_analysis_prediction(last_hidden_state)).view(sentiment_analysis_targets.size())

		cat_mask = (category_targets != -1).view(category_targets.size())

		if cat_mask.sum() == 0:
			cat_loss = self.mse_loss(torch.ones(1), torch.ones(1))
			if next(self.parameters()).is_cuda:
				cat_loss = cat_loss.cuda()
		else:
			cat_loss = self.mse_loss(cat_pred[cat_mask].view(-1), category_targets[cat_mask].view(-1))

		# if category_targets is not None:
		# if sentiment_analysis_targets is not None:
			# only use the outputs for which there are targets
		sa_mask = (sentiment_analysis_targets != -1).view(sentiment_analysis_targets.size())

		# if there are no SA targets, then we do not calculate the SA loss, and set it to 0 (because it returns nan with empty tensors)
		if sa_mask.sum() == 0:
			sa_loss = self.mse_loss(torch.ones(1), torch.ones(1))
			if next(self.parameters()).is_cuda:
				sa_loss = sa_loss.cuda()
		else:
			sa_loss = self.mse_loss(sa_pred[sa_mask].view(-1), sentiment_analysis_targets[sa_mask].view(-1))

		return [cat_pred, sa_pred], [cat_loss, sa_loss], contexts, category_targets, attentions




class FlatNLGTransformer(TransformerEncoder):
	def __init__(self, vocab_size, cat_size, args = None):
		super(FlatNLGTransformer, self).__init__(vocab_size, cat_size, args)


		self.lm_head = nn.Linear(self.encoder.config.n_embd, self.vocab_size, bias=False)

		""" Make sure we are sharing the input and output embeddings.
			Export to TorchScript can't handle parameter sharing so we are cloning them instead.
		"""
		self.encoder._tie_or_clone_weights(self.lm_head, self.encoder.wte)

		if self.args.use_cuda:
			self.encoder = torch.nn.DataParallel(self.encoder)


	def forward(self, batch, eval_mode = False):

		contexts = batch["contexts"]
		token_types = batch["token_types"]
		attention_masks = batch["attention_masks"]
		# category_targets = batch["category_targets"]
		# sentiment_analysis_targets = batch["sentiment_analysis_targets"]
		nlg_targets = batch["nlg_targets"]
		nlg_gt_inputs = batch["nlg_gt_inputs"]

		# contexts, token_types, attention_masks, _, _, nlg_targets, nlg_gt_inputs, batch_movie_mentions, _ = batch

		if eval_mode:
			nlp_input = nlg_gt_inputs
		else:
			nlp_input = contexts


		# if the model is on cuda, we transfer all tensors to cuda
		if next(self.parameters()).is_cuda:
			nlp_input, token_types, attention_masks, nlg_targets = nlp_input.cuda(), token_types.cuda(), attention_masks.cuda(), nlg_targets.cuda()

		last_hidden_state, past = self.encoder(input_ids = nlp_input, attention_mask=attention_masks, token_type_ids=token_types)

		lm_logits = self.lm_head(last_hidden_state)


		self.CrossE_Loss = torch.nn.CrossEntropyLoss(ignore_index=-1)


		if eval_mode:
			# for evaluation mode, we return a 2 lists, containing perplexity and blue score, for each sample, so that they can be averaged over samples / recommenders utterances
			perplexities = []
			blue_scores = []
			# for each sample
			for i in range(contexts.size(0)):
				target_tokens_mask = (nlg_targets[i] != -1).view(nlg_targets[i].size())
				if target_tokens_mask.sum() == 0:
					continue
				model_output = lm_logits[i, target_tokens_mask]
				target = nlg_targets[i, target_tokens_mask]

				perplexities.append( np.exp( self.CrossE_Loss(model_output, target).item() ) )

				generated_sentence = torch.argmax(model_output, dim = -1 )
				blue_scores.append(self.get_blue_score( gt_sentence = target.cpu().tolist(), generated_sentence = generated_sentence.cpu().tolist() ))

			return perplexities, blue_scores

		else:

			nlg_loss = self.CrossE_Loss(lm_logits.view(-1, self.vocab_size), nlg_targets.view(-1))

			return [lm_logits], [nlg_loss]




# class FlatPretrainedFullTransformer(TransformerEncoder):
#     def __init__(self, vocab_size, cat_size, args = None):
#         super(FlatPretrainedFullTransformer, self).__init__(vocab_size, cat_size, args)

#         self.semantic_module = FlatSemanticTransformer(vocab_size = vocab_size, cat_size = cat_size, args = args)
#         self.nlg_module = FlatNLGTransformer(vocab_size = vocab_size, cat_size = cat_size, args = args)
		
#         if self.args.use_cuda:
#             self.semantic_module = torch.nn.DataParallel(self.semantic_module)
#             self.nlg_module = torch.nn.DataParallel(self.nlg_module)


#     def forward(self, batch, use_gt_nlp_input = False):

#         contexts, token_types, attention_masks, category_targets, sentiment_analysis_targets, nlg_targets, nlg_gt_inputs, batch_movie_mentions = batch

#         # if the model is on cuda, we transfer all tensors to cuda
#         if next(self.parameters()).is_cuda:
#             batch = contexts.cuda(), token_types.cuda(), attention_masks.cuda(), category_targets.cuda(), sentiment_analysis_targets.cuda(), nlg_targets.cuda(), nlg_gt_inputs.cuda(), batch_movie_mentions.cuda()
#             # contexts, token_types, attention_masks, nlg_targets = contexts.cuda(), token_types.cuda(), attention_masks.cuda(), nlg_targets.cuda()

#         [cat_pred, sa_pred], [cat_loss, sa_loss] = self.semantic_module(batch)

#         [lm_logits], [nlg_loss] = self.nlg_module(batch, use_gt_nlp_input)

#         return [cat_pred, sa_pred, lm_logits], [cat_loss, sa_loss, nlg_loss]


class FlatJointFullTransformer(TransformerEncoder):
	def __init__(self, vocab_size, cat_size, args = None):
		super(FlatJointFullTransformer, self).__init__(vocab_size, cat_size, args)


		if self.CLS_mode:
			self.n_cls_tokens = 1
			self.cat_prediction = nn.Linear(in_features=self.encoder.config.n_embd, out_features=self.cat_size)
		else:
			self.n_cls_tokens = self.cat_size
			# have a trainable linear function for each CLS token
			self.cat_prediction = nn.ModuleList()
			for i in range(self.n_cls_tokens):
				self.cat_prediction.append(nn.Linear(in_features=self.encoder.config.n_embd, out_features=1))

		self.sentiment_analysis_prediction = nn.Linear(in_features=self.encoder.config.n_embd, out_features=1)


		self.lm_head = nn.Linear(self.encoder.config.n_embd, self.vocab_size, bias=False)

		""" Make sure we are sharing the input and output embeddings.
			Export to TorchScript can't handle parameter sharing so we are cloning them instead.
		"""
		self.encoder._tie_or_clone_weights(self.lm_head, self.encoder.wte)
		
		
		if self.args.use_cuda:
			self.encoder = torch.nn.DataParallel(self.encoder)


	def forward(self, batch, eval_mode = False):


		contexts = batch["contexts"]
		token_types = batch["token_types"]
		attention_masks = batch["attention_masks"]
		category_targets = batch["category_targets"]
		sentiment_analysis_targets = batch["sentiment_analysis_targets"]
		nlg_targets = batch["nlg_targets"]
		nlg_gt_inputs = batch["nlg_gt_inputs"]
		# batch_movie_mentions = batch["batch_movie_mentions"]
		# complete_sample_movie_targets = batch["complete_sample_movie_targets"]



		# if the model is on cuda, we transfer all tensors to cuda
		if next(self.parameters()).is_cuda:
			contexts, token_types, attention_masks, category_targets, sentiment_analysis_targets, nlg_targets, nlg_gt_inputs = \
			 contexts.cuda(), token_types.cuda(), attention_masks.cuda(), category_targets.cuda(), sentiment_analysis_targets.cuda(), nlg_targets.cuda(), nlg_gt_inputs.cuda()

		if eval_mode:
			nlp_input = nlg_gt_inputs
		else:
			nlp_input = contexts


		last_hidden_state, past = self.encoder(input_ids = nlp_input, attention_mask=attention_masks, token_type_ids=token_types)


		# calculate nlg and its loss
		lm_logits = self.lm_head(last_hidden_state)

		self.CrossE_Loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

		if eval_mode:
			# for evaluation mode, we return a 2 lists, containing perplexity and blue score, for each sample, so that they can be averaged over samples / recommenders utterances
			perplexities = []
			blue_scores = []
			# for each sample
			for i in range(contexts.size(0)):
				target_tokens_mask = (nlg_targets[i] != -1).view(nlg_targets[i].size())
				if target_tokens_mask.sum() == 0:
					continue
				model_output = lm_logits[i, target_tokens_mask]
				target = nlg_targets[i, target_tokens_mask]

				perplexities.append( np.exp( self.CrossE_Loss(model_output, target).item() ) )

				generated_sentence = torch.argmax(model_output, dim = -1 )
				blue_scores.append(self.get_blue_score( gt_sentence = target.cpu().tolist(), generated_sentence = generated_sentence.cpu().tolist() ))

			return perplexities, blue_scores

		else:


			self.mse_loss = torch.nn.MSELoss(reduction='mean')


			# we only use the first CLS tokens as cat prediction input
			cls_input = last_hidden_state[:, 0: self.n_cls_tokens, :]
			# we get the category prediction
			if self.CLS_mode:
				cat_pred = torch.nn.functional.softmax( self.cat_prediction(cls_input), dim= -1).view(category_targets.size())
			else:
				cat_pred = torch.zeros_like(category_targets)
				for i in range(self.n_cls_tokens):
					cat_pred[:, i] = self.cat_prediction[i]( cls_input[:,i, :] ).view(cat_pred[:, i].size())
				cat_pred = torch.nn.functional.softmax( cat_pred, dim= -1).view(category_targets.size())

			# we get sentiment analysis prediction
			sa_pred = torch.sigmoid(self.sentiment_analysis_prediction(last_hidden_state)).view(sentiment_analysis_targets.size())

			cat_mask = (category_targets != -1).view(category_targets.size())

			if cat_mask.sum() == 0:
				cat_loss = self.mse_loss(torch.ones(1), torch.ones(1))
				if next(self.parameters()).is_cuda:
					cat_loss = cat_loss.cuda()
			else:
				cat_loss = self.mse_loss(cat_pred[cat_mask].view(-1), category_targets[cat_mask].view(-1))

			# if category_targets is not None:
			# if sentiment_analysis_targets is not None:
				# only use the outputs for which there are targets
			sa_mask = (sentiment_analysis_targets != -1).view(sentiment_analysis_targets.size())

			# if there are no SA targets, then we do not calculate the SA loss, and set it to 0 (because it returns nan with empty tensors)
			if sa_mask.sum() == 0:
				sa_loss = self.mse_loss(torch.ones(1), torch.ones(1))
				if next(self.parameters()).is_cuda:
					sa_loss = sa_loss.cuda()
			else:
				sa_loss = self.mse_loss(sa_pred[sa_mask].view(-1), sentiment_analysis_targets[sa_mask].view(-1))

			nlg_loss = self.CrossE_Loss(lm_logits.view(-1, self.vocab_size), nlg_targets.view(-1))


			return [cat_pred, sa_pred, lm_logits], [cat_loss, sa_loss, nlg_loss]





# we will have one class, that inherits from TransformerEncoder, and uses the encoder as message transformer
#  on top of that we will add the dialogue transformer


# then, all classes-per-task will inherite from this new class

class HIBERT(TransformerEncoder):
	def __init__(self, vocab_size, cat_size, args = None):
		super(HIBERT, self).__init__(vocab_size, cat_size, args)


		# config = transformers.GPT2Config(vocab_size_or_config_json_file = vocab_size, n_positions = args.input_length_limit, n_ctx = args.input_length_limit, n_embd = args.hidden_size,
		#                                              n_layer = args.num_hidden_layers, n_head = args.num_attention_heads)

		# self.dialogue_encoder = transformers.GPT2Model(config)

		# config = transformers.BertConfig(vocab_size_or_config_json_file = vocab_size, hidden_size = args.hidden_size, num_hidden_layers = args.num_hidden_layers,
		#                                 num_attention_heads = args.num_attention_heads, intermediate_size = args.intermediate_size, max_position_embeddings = args.input_length_limit)

		self.dialogue_encoder = transformers.BertModel(self.encoder.config)



class HIBERTSemanticTransformer(HIBERT):
	def __init__(self, vocab_size, cat_size, args = None):
		super(HIBERTSemanticTransformer, self).__init__(vocab_size, cat_size, args)

		self.hidden_size = self.encoder.config.hidden_size

		if self.CLS_mode:
			self.n_cls_tokens = 1
			self.cat_prediction = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=self.cat_size)
		else:
			self.n_cls_tokens = self.cat_size
			# have a trainable linear function for each CLS token
			self.cat_prediction = nn.ModuleList()
			for i in range(self.n_cls_tokens):
				self.cat_prediction.append(nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1))

		self.sentiment_analysis_prediction = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1)


		if self.args.use_cuda and self.args.n_gpu > 1:
			self.encoder = torch.nn.DataParallel(self.encoder)
			self.dialogue_encoder = torch.nn.DataParallel(self.dialogue_encoder)

	def forward(self, batch):

		# contexts, context_attentions, category_targets, nlg_targets, nlg_gt_inputs, pool_hidden_representations_mask, dialogue_trans_positional_embeddings, \
		# sentiment_analysis_targets, dialogue_trans_token_type_ids, nlg_dialogue_mask_tokens, dialogue_trans_attentions, CLS_pooled_tokens_mask, batch_movie_mentions, _ = batch

		contexts = batch["contexts"]
		context_attentions = batch["context_attentions"]
		category_targets = batch["category_targets"]
		nlg_targets = batch["nlg_targets"]
		nlg_gt_inputs = batch["nlg_gt_inputs"]
		pool_hidden_representations_mask = batch["pool_hidden_representations_mask"]
		dialogue_trans_positional_embeddings = batch["dialogue_trans_positional_embeddings"]
		sentiment_analysis_targets = batch["sentiment_analysis_targets"]
		dialogue_trans_token_type_ids = batch["dialogue_trans_token_type_ids"]
		nlg_dialogue_mask_tokens = batch["nlg_dialogue_mask_tokens"]
		dialogue_trans_attentions = batch["dialogue_trans_attentions"]
		CLS_pooled_tokens_mask = batch["CLS_pooled_tokens_mask"]
		batch_movie_mentions = batch["batch_movie_mentions"]

		# if the model is on cuda, we transfer all tensors to cuda
		if next(self.parameters()).is_cuda:
			
			contexts, context_attentions, dialogue_trans_attentions, dialogue_trans_positional_embeddings, dialogue_trans_token_type_ids = \
			 contexts.cuda(), context_attentions.cuda(), dialogue_trans_attentions.cuda(), dialogue_trans_positional_embeddings.cuda(), dialogue_trans_token_type_ids.cuda()

		# print(contexts.size())
		# print(context_attentions.size())
		# print(contexts.view(-1, contexts.size(-1)).size())
		# print(context_attentions.view(-1, context_attentions.size(-1)).size())

		context_sizes = list(contexts.size())

		# reshape contexts and attetnions so that they hierarchical representation will be treated as flat from the message_encoder

		# we get the hidden representations of all tokens
		message_encoder_hidden_representations, _, _ = self.encoder(input_ids = contexts.view(-1, contexts.size(-1)), attention_mask = context_attentions.view(-1, context_attentions.size(-1))) #, token_type_ids=token_types)

		# reshape message_encoder's output so that the output will have the original hierarchical shape

		message_encoder_hidden_representations = message_encoder_hidden_representations.view( tuple(context_sizes + [-1] ) )
		

		temp_dialogue_encoder_input = []
		# max_length = 0
		for i in range(message_encoder_hidden_representations.size(0)):
			# we only keep the hidden representations of the important tokens (CLS, MM, EOS)
			# print(pool_hidden_representations_mask[i])
			temp_dialogue_encoder_input.append((message_encoder_hidden_representations[i, pool_hidden_representations_mask[i]]))
			# saving max length for padding
			# if max_length < temp_dialogue_encoder_input[-1].size(0):
			#     max_length = temp_dialogue_encoder_input[-1].size(0)

		number_of_samples = message_encoder_hidden_representations.size(0)
		sequence_length = dialogue_trans_positional_embeddings.size(-1)

		# pad inputs
		dialogue_encoder_input = torch.zeros(number_of_samples, sequence_length, self.args.hidden_size, dtype = torch.float, device = contexts.device)
		for i, dialogue_input in enumerate(temp_dialogue_encoder_input):
			dialogue_encoder_input[i, : dialogue_input.size(0), : ] = dialogue_input

		# print("dialogue_trans_attentions :", dialogue_trans_attentions.size())
		# print("dialogue_trans_positional_embeddings :", dialogue_trans_positional_embeddings.size())
		# print("dialogue_encoder_input :", dialogue_encoder_input.size())
		# print()

		torch.cuda.empty_cache()


		# we give them as input to the dialogue encoder
		dialogue_encoder_hidden_representations, _, _ = self.dialogue_encoder(attention_mask = dialogue_trans_attentions, position_ids = dialogue_trans_positional_embeddings, token_type_ids = dialogue_trans_token_type_ids, inputs_embeds = dialogue_encoder_input)

		torch.cuda.empty_cache()


		if next(self.parameters()).is_cuda:
			category_targets, sentiment_analysis_targets = category_targets.cuda(), sentiment_analysis_targets.cuda()


		# cls_input = torch.zeros(category_targets.size(0), self.n_cls_tokens, self.hidden_size, dtype=dialogue_encoder_hidden_representations.dtype, device=category_targets.device, requires_grad=True)


		# _like(category_targets).unsqueeze(-1).repeat(1,1, self.hidden_size)

		# cat_pred = []

		# cls_input = dialogue_encoder_hidden_representations[CLS_pooled_tokens_mask].view(dialogue_encoder_hidden_representations.size(0), self.n_cls_tokens , self.hidden_size)

		cls_inputs = []
		# cat_targets = []

		# print("Number of samples: ", dialogue_encoder_hidden_representations.size(0))

		# get_category prediction for every sample in the batch. We force all CLS tokens from each message to predict the dialogue's category information
		for i in range(dialogue_encoder_hidden_representations.size(0)):
			# we get only the CLS tokens' hidden representantion
			# print(torch.sum(CLS_pooled_tokens_mask[i]))
			# print(dialogue_encoder_hidden_representations[i, CLS_pooled_tokens_mask[i]])



			# cat_pooled_tokens = dialogue_encoder_hidden_representations[i, CLS_pooled_tokens_mask[i]].view(-1, self.n_cls_tokens, self.hidden_size).mean(dim = 0)
			# cls_input[i,:] = dialogue_encoder_hidden_representations[i, CLS_pooled_tokens_mask[i]].view(-1, self.n_cls_tokens, self.hidden_size).mean(dim = 0)

			cls_inputs.append(dialogue_encoder_hidden_representations[i, CLS_pooled_tokens_mask[i]].view(-1, self.n_cls_tokens, self.hidden_size).mean(dim = 0))

			# # we get the category prediction
			# if self.CLS_mode:
			#     cat_pred.append(torch.nn.functional.softmax( self.cat_prediction(cat_pooled_tokens), dim= -1).view(category_targets[0].size()) )
			# else:
			#     temp_cat_pred = torch.zeros_like(category_targets[0])
			#     for i in range(self.n_cls_tokens):
			#         temp_cat_pred[i] = self.cat_prediction[i]( cat_pooled_tokens[i, :] ).view(temp_cat_pred[i].size())
			#     cat_pred.append( torch.nn.functional.softmax( temp_cat_pred, dim= -1).view(category_targets[0].size()) )

		# print(cls_input.size())


		cls_input = torch.stack(cls_inputs)

		# print(cls_input.size())
		# we get the category prediction
		if self.CLS_mode:
			cat_pred = torch.nn.functional.softmax( self.cat_prediction(cls_input), dim= -1).view(category_targets.size())
		else:

			cat_pred = [] # = torch.zeros_like(category_targets)
			for i in range(self.n_cls_tokens):
				cat_pred.append(self.cat_prediction[i]( cls_input[:,i, :] ))
			cat_pred = torch.stack(cat_pred)
			cat_pred = torch.nn.functional.softmax( cat_pred, dim= -1).view(category_targets.size())


		# exit()

			# we set the target to each messages CLS token's equal to the dialogue category target
			# cat_targets.append( category_targets[i].unsqueeze(0).repeat(cat_pooled_tokens.size(0), 1) )
		# print(cat_pred.size())

		# for c in cat_pred:
		#     print(c.size())

		# cat_pred = torch.cat(cat_pred, dim = 0).view(category_targets.size())


		# print(cat_pred)

		cat_mask = (category_targets != -1).view(category_targets.size())

		if cat_mask.sum() == 0:
			cat_loss = self.mse_loss(torch.ones(1), torch.ones(1))
			if next(self.parameters()).is_cuda:
				cat_loss = cat_loss.cuda()
		else:
			cat_loss = self.mse_loss(cat_pred[cat_mask].view(-1), category_targets[cat_mask].view(-1))


		# we get sentiment analysis prediction
		sa_pred = torch.sigmoid(self.sentiment_analysis_prediction(dialogue_encoder_hidden_representations)).view(sentiment_analysis_targets.size())

		# if sentiment_analysis_targets is not None:
			# only use the outputs for which there are targets
		sa_mask = (sentiment_analysis_targets != -1).view(sentiment_analysis_targets.size())
		# print(targets_mask)
		# print(sa_pred)

		# if there are no SA targets, then we do not calculate the SA loss, and set it to 0 (because it returns nan with empty tensors)
		if sa_mask.sum() == 0:
			sa_loss = self.mse_loss(torch.ones(1), torch.ones(1))
			if next(self.parameters()).is_cuda:
				sa_loss = sa_loss.cuda()
		else:
			sa_loss = self.mse_loss(sa_pred[sa_mask].view(-1), sentiment_analysis_targets[sa_mask].view(-1))

		# print([cat_pred.size(), sa_pred.size()], [cat_loss, sa_loss])

		return [cat_pred, sa_pred], [cat_loss, sa_loss]


class HIBERTNLGTransformer(HIBERT):
	def __init__(self, vocab_size, cat_size, args = None):
		super(HIBERTNLGTransformer, self).__init__(vocab_size, cat_size, args)

		self.hidden_size = self.dialogue_encoder.config.n_embd

		self.lm_head = nn.Linear(self.encoder.config.n_embd, self.vocab_size, bias=False)

		""" Make sure we are sharing the input and output embeddings.
			Export to TorchScript can't handle parameter sharing so we are cloning them instead.
		"""
		self.encoder._tie_or_clone_weights(self.lm_head, self.encoder.wte)
		if self.args.use_cuda:
			self.encoder = torch.nn.DataParallel(self.encoder)
			self.dialogue_encoder = torch.nn.DataParallel(self.dialogue_encoder)

	def forward(self, batch, eval_mode = False):

		# contexts, context_attentions, category_targets, nlg_targets, nlg_gt_inputs, pool_hidden_representations_mask, dialogue_trans_positional_embeddings, \
		# sentiment_analysis_targets, dialogue_trans_token_type_ids, nlg_dialogue_mask_tokens, dialogue_trans_attentions, CLS_pooled_tokens_mask, batch_movie_mentions, _ = batch

		contexts = batch["contexts"]
		context_attentions = batch["context_attentions"]
		category_targets = batch["category_targets"]
		nlg_targets = batch["nlg_targets"]
		nlg_gt_inputs = batch["nlg_gt_inputs"]
		pool_hidden_representations_mask = batch["pool_hidden_representations_mask"]
		dialogue_trans_positional_embeddings = batch["dialogue_trans_positional_embeddings"]
		sentiment_analysis_targets = batch["sentiment_analysis_targets"]
		dialogue_trans_token_type_ids = batch["dialogue_trans_token_type_ids"]
		nlg_dialogue_mask_tokens = batch["nlg_dialogue_mask_tokens"]
		dialogue_trans_attentions = batch["dialogue_trans_attentions"]
		CLS_pooled_tokens_mask = batch["CLS_pooled_tokens_mask"]
		batch_movie_mentions = batch["batch_movie_mentions"]

		# if we are evaluating, then we do not give masked inputs but the original ones. So we update the last sentence of the context if necessary
		if eval_mode:
			contexts[:, -1, :] = nlg_gt_inputs

		# print(category_targets)


		# if the model is on cuda, we transfer all tensors to cuda
		if next(self.parameters()).is_cuda:
			# contexts, token_types, attention_masks, category_targets, sentiment_analysis_targets = \
			#  contexts.cuda(), token_types.cuda(), attention_masks.cuda(), category_targets.cuda(), sentiment_analysis_targets.cuda()
			contexts, context_attentions, dialogue_trans_positional_embeddings, dialogue_trans_token_type_ids, nlg_dialogue_mask_tokens = \
			 contexts.cuda(), context_attentions.cuda(), dialogue_trans_positional_embeddings.cuda(), dialogue_trans_token_type_ids.cuda(), nlg_dialogue_mask_tokens.cuda()


		# we get the hidden representations of all tokens
		message_encoder_hidden_representations, past = self.encoder(input_ids = contexts, attention_mask=context_attentions) #, token_type_ids=token_types)

		temp_dialogue_encoder_input = []
		for i in range(message_encoder_hidden_representations.size(0)):
			# we only keep the hidden representations of the important tokens (EOSs, MASKs)
			temp_dialogue_encoder_input.append((message_encoder_hidden_representations[i, pool_hidden_representations_mask[i]]))

		number_of_samples = message_encoder_hidden_representations.size(0)

		sequence_length = dialogue_trans_positional_embeddings.size(-1)
		# pad inputs
		dialogue_encoder_input = torch.zeros(number_of_samples, sequence_length, self.args.hidden_size, dtype = torch.float, device = contexts.device)
		for i, dialogue_input in enumerate(temp_dialogue_encoder_input):
			dialogue_encoder_input[i, : dialogue_input.size(0), : ] = dialogue_input


		torch.cuda.empty_cache()
		# we give them as input to the dialogue encoder
		dialogue_encoder_hidden_representations, past = self.dialogue_encoder(attention_mask = nlg_dialogue_mask_tokens, position_ids = dialogue_trans_positional_embeddings, token_type_ids = dialogue_trans_token_type_ids, inputs_embeds = dialogue_encoder_input)


		lm_logits = self.lm_head(dialogue_encoder_hidden_representations)


		self.CrossE_Loss = torch.nn.CrossEntropyLoss(ignore_index=-1)


		torch.cuda.empty_cache()

		if next(self.parameters()).is_cuda:
			nlg_targets = nlg_targets.cuda()


		if eval_mode:
			# for evaluation mode, we return a 2 lists, containing perplexity and blue score, for each sample, so that they can be averaged over samples / recommenders utterances
			perplexities = []
			blue_scores = []
			# for each sample
			for i in range(contexts.size(0)):
				target_tokens_mask = (nlg_targets[i] != -1).view(nlg_targets[i].size())
				if target_tokens_mask.sum() == 0:
					continue
				model_output = lm_logits[i, target_tokens_mask]
				target = nlg_targets[i, target_tokens_mask]

				perplexities.append( np.exp( self.CrossE_Loss(model_output, target).item() ) )

				generated_sentence = torch.argmax(model_output, dim = -1 )
				blue_scores.append(self.get_blue_score( gt_sentence = target.cpu().tolist(), generated_sentence = generated_sentence.cpu().tolist() ))

			return perplexities, blue_scores

		else:

			nlg_loss = self.CrossE_Loss(lm_logits.view(-1, self.vocab_size), nlg_targets.view(-1))

			return [lm_logits], [nlg_loss]






# class HIBERTPretrainedFullTransformer(HIBERT):
#     def __init__(self, vocab_size, cat_size, args = None):
#         super(HIBERTPretrainedFullTransformer, self).__init__(vocab_size, cat_size, args)



class HIBERTJointFullTransformer(HIBERT):
	def __init__(self, vocab_size, cat_size, args = None):
		super(HIBERTJointFullTransformer, self).__init__(vocab_size, cat_size, args)

		self.hidden_size = self.dialogue_encoder.config.n_embd

		if self.CLS_mode:
			self.n_cls_tokens = 1
			self.cat_prediction = nn.Linear(in_features=self.hidden_size, out_features=self.cat_size)
		else:
			self.n_cls_tokens = self.cat_size
			# have a trainable linear function for each CLS token
			self.cat_prediction = nn.ModuleList()
			for i in range(self.n_cls_tokens):
				self.cat_prediction.append(nn.Linear(in_features=self.encoder.config.n_embd, out_features=1))

		self.sentiment_analysis_prediction = nn.Linear(in_features=self.hidden_size, out_features=1)


		self.lm_head = nn.Linear(self.encoder.config.n_embd, self.vocab_size, bias=False)

		""" Make sure we are sharing the input and output embeddings.
			Export to TorchScript can't handle parameter sharing so we are cloning them instead.
		"""
		self.encoder._tie_or_clone_weights(self.lm_head, self.encoder.wte)

		if self.args.use_cuda:
			self.encoder = torch.nn.DataParallel(self.encoder)
			self.dialogue_encoder = torch.nn.DataParallel(self.dialogue_encoder)

	def forward(self, batch, eval_mode = False):

		# contexts, context_attentions, category_targets, nlg_targets, nlg_gt_inputs, pool_hidden_representations_mask, dialogue_trans_positional_embeddings, \
		# sentiment_analysis_targets, dialogue_trans_token_type_ids, nlg_dialogue_mask_tokens, dialogue_trans_attentions, CLS_pooled_tokens_mask, batch_movie_mentions, _ = batch

		contexts = batch["contexts"]
		context_attentions = batch["context_attentions"]
		category_targets = batch["category_targets"]
		nlg_targets = batch["nlg_targets"]
		nlg_gt_inputs = batch["nlg_gt_inputs"]
		pool_hidden_representations_mask = batch["pool_hidden_representations_mask"]
		dialogue_trans_positional_embeddings = batch["dialogue_trans_positional_embeddings"]
		sentiment_analysis_targets = batch["sentiment_analysis_targets"]
		dialogue_trans_token_type_ids = batch["dialogue_trans_token_type_ids"]
		nlg_dialogue_mask_tokens = batch["nlg_dialogue_mask_tokens"]
		dialogue_trans_attentions = batch["dialogue_trans_attentions"]
		CLS_pooled_tokens_mask = batch["CLS_pooled_tokens_mask"]
		batch_movie_mentions = batch["batch_movie_mentions"]



		# if the model is on cuda, we transfer all tensors to cuda
		if next(self.parameters()).is_cuda:
			# contexts, token_types, attention_masks, category_targets, sentiment_analysis_targets = \
			#  contexts.cuda(), token_types.cuda(), attention_masks.cuda(), category_targets.cuda(), sentiment_analysis_targets.cuda()
			contexts, context_attentions, dialogue_trans_attentions, dialogue_trans_positional_embeddings, dialogue_trans_token_type_ids = \
			 contexts.cuda(), context_attentions.cuda(), dialogue_trans_attentions.cuda(), dialogue_trans_positional_embeddings.cuda(), dialogue_trans_token_type_ids.cuda()


		# we get the hidden representations of all tokens
		message_encoder_hidden_representations, past = self.encoder(input_ids = contexts, attention_mask=context_attentions) #, token_type_ids=token_types)

		temp_dialogue_encoder_input = []
		# max_length = 0
		for i in range(message_encoder_hidden_representations.size(0)):
			# we only keep the hidden representations of the important tokens (CLS, MM, EOS)
			# print(pool_hidden_representations_mask[i])
			temp_dialogue_encoder_input.append((message_encoder_hidden_representations[i, pool_hidden_representations_mask[i]]))
			# saving max length for padding
			# if max_length < temp_dialogue_encoder_input[-1].size(0):
			#     max_length = temp_dialogue_encoder_input[-1].size(0)

		number_of_samples = message_encoder_hidden_representations.size(0)
		sequence_length = dialogue_trans_positional_embeddings.size(-1)

		# pad inputs
		dialogue_encoder_input = torch.zeros(number_of_samples, sequence_length, self.args.hidden_size, dtype = torch.float, device = contexts.device)
		for i, dialogue_input in enumerate(temp_dialogue_encoder_input):
			dialogue_encoder_input[i, : dialogue_input.size(0), : ] = dialogue_input

		# print("dialogue_trans_attentions :", dialogue_trans_attentions.size())
		# print("dialogue_trans_positional_embeddings :", dialogue_trans_positional_embeddings.size())
		# print("dialogue_encoder_input :", dialogue_encoder_input.size())
		# print()

		torch.cuda.empty_cache()


		# we give them as input to the dialogue encoder
		dialogue_encoder_hidden_representations, past = self.dialogue_encoder(attention_mask = dialogue_trans_attentions, position_ids = dialogue_trans_positional_embeddings, token_type_ids = dialogue_trans_token_type_ids, inputs_embeds = dialogue_encoder_input)

		lm_logits = self.lm_head(dialogue_encoder_hidden_representations)


		self.CrossE_Loss = torch.nn.CrossEntropyLoss(ignore_index=-1)


		torch.cuda.empty_cache()

		if next(self.parameters()).is_cuda:
			nlg_targets = nlg_targets.cuda()


		if eval_mode:
			# for evaluation mode, we return a 2 lists, containing perplexity and blue score, for each sample, so that they can be averaged over samples / recommenders utterances
			perplexities = []
			blue_scores = []
			# for each sample
			for i in range(contexts.size(0)):
				target_tokens_mask = (nlg_targets[i] != -1).view(nlg_targets[i].size())
				if target_tokens_mask.sum() == 0:
					continue
				model_output = lm_logits[i, target_tokens_mask]
				target = nlg_targets[i, target_tokens_mask]

				perplexities.append( np.exp( self.CrossE_Loss(model_output, target).item() ) )

				generated_sentence = torch.argmax(model_output, dim = -1 )
				blue_scores.append(self.get_blue_score( gt_sentence = target.cpu().tolist(), generated_sentence = generated_sentence.cpu().tolist() ))

			return perplexities, blue_scores

		else:


			# calculate semmantic losses

			self.mse_loss = torch.nn.MSELoss(reduction='mean')

			if next(self.parameters()).is_cuda:
				category_targets, sentiment_analysis_targets = category_targets.cuda(), sentiment_analysis_targets.cuda()



			cls_input = torch.zeros(category_targets.size(0), self.n_cls_tokens, self.hidden_size, dtype=dialogue_encoder_hidden_representations.dtype, device=category_targets.device, requires_grad=True)
			# cls_input = torch.zeros_like(category_targets).unsqueeze(-1).repeat(1,1, self.hidden_size)

			# print(cls_input.size())


			# get_category prediction for every sample in the batch. We force all CLS tokens from each message to predict the dialogue's category information
			for i in range(dialogue_encoder_hidden_representations.size(0)):

				cls_input[i,:] = dialogue_encoder_hidden_representations[i, CLS_pooled_tokens_mask[i]].view(-1, self.n_cls_tokens, self.hidden_size).mean(dim = 0)

			# we get the category prediction
			if self.CLS_mode:
				cat_pred = torch.nn.functional.softmax( self.cat_prediction(cls_input), dim= -1).view(category_targets.size())
			else:
				cat_pred = torch.zeros_like(category_targets)
				for i in range(self.n_cls_tokens):
					cat_pred[:, i] = self.cat_prediction[i]( cls_input[:,i, :] ).view(cat_pred[:, i].size())
				cat_pred = torch.nn.functional.softmax( cat_pred, dim= -1).view(category_targets.size())



			# cat_pred = []

			# cat_targets = []

			# # get_category prediction for every sample in the batch. We force all CLS tokens from each message to predict the dialogue's category information
			# for i in range(dialogue_encoder_hidden_representations.size(0)):
			#     # we get only the CLS tokens' hidden representantion
			#     # print(torch.sum(CLS_pooled_tokens_mask[i]))
			#     # print(dialogue_encoder_hidden_representations[i, CLS_pooled_tokens_mask[i]])
			#     cat_pooled_tokens = dialogue_encoder_hidden_representations[i, CLS_pooled_tokens_mask[i]].view(-1, self.n_cls_tokens, self.hidden_size).mean(dim = 0)
			#     # print(cat_pooled_tokens.size())
			#     # print(torch.sum(cat_pooled_tokens))
			#     # print()
			#     # if there are no CLS tokens in this sample (The recommender just started the conversation), the we skip this sample for category prediction
			#     # if cat_pooled_tokens.size(0) == 0:
			#     #     cat_pred
			#     #     cat_targets
			#     # continue 
			#     # we apply the linear and softmax to each message CLS tokens separately
			#     # print(self.cat_prediction(cat_pooled_tokens).size())
			#     cat_pred.append( torch.nn.functional.softmax( self.cat_prediction(cat_pooled_tokens), dim= -1) )
			#     # we set the target to each messages CLS token's equal to the dialogue category target
			#     # cat_targets.append( category_targets[i].unsqueeze(0).repeat(cat_pooled_tokens.size(0), 1) )
			# # print(cat_pred.size())

			# # for c in cat_pred:
			# #     print(c.size())

			# cat_pred = torch.cat(cat_pred, dim = 0).view(category_targets.size())


			# print(cat_pred)

			cat_mask = (category_targets != -1).view(category_targets.size())

			if cat_mask.sum() == 0:
				cat_loss = self.mse_loss(torch.ones(1), torch.ones(1))
				if next(self.parameters()).is_cuda:
					cat_loss = cat_loss.cuda()
			else:
				cat_loss = self.mse_loss(cat_pred[cat_mask].view(-1), category_targets[cat_mask].view(-1))


			# we get sentiment analysis prediction
			sa_pred = torch.sigmoid(self.sentiment_analysis_prediction(dialogue_encoder_hidden_representations)).view(sentiment_analysis_targets.size())

			# if sentiment_analysis_targets is not None:
				# only use the outputs for which there are targets
			sa_mask = (sentiment_analysis_targets != -1).view(sentiment_analysis_targets.size())
			# print(targets_mask)
			# print(sa_pred)

			# if there are no SA targets, then we do not calculate the SA loss, and set it to 0 (because it returns nan with empty tensors)
			if sa_mask.sum() == 0:
				sa_loss = self.mse_loss(torch.ones(1), torch.ones(1))
				if next(self.parameters()).is_cuda:
					sa_loss = sa_loss.cuda()
			else:
				sa_loss = self.mse_loss(sa_pred[sa_mask].view(-1), sentiment_analysis_targets[sa_mask].view(-1))

			# print([cat_pred.size(), sa_pred.size()], [cat_loss, sa_loss])



			# calculate nlg loss


			nlg_loss = self.CrossE_Loss(lm_logits.view(-1, self.vocab_size), nlg_targets.view(-1))

			return [cat_pred, sa_pred, lm_logits], [cat_loss, sa_loss, nlg_loss]


