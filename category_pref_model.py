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


class Cat_Pref_BERT(nn.Module):
	def __init__(self, vocab_size, cat_size, n_movies, ignore_model = False, args = None):
		super(Cat_Pref_BERT, self).__init__()
		self.vocab_size = vocab_size
		self.cat_size = cat_size
		self.n_movies = n_movies
		# self.CLS_mode = args.CLS_mode
		# self.use_ground_truth = use_ground_truth

		self.args = args

		# self.cat_sa_alpha = args.cat_sa_alpha
		# self.sem_nlg_alpha = args.sem_nlg_alpha

		if ignore_model == False:

			# if args.use_pretrained:
			# 	# self.encoder = transformers.GPT2Model.from_pretrained("gpt2") #, output_hidden_states=True

			self.encoder = transformers.BertModel.from_pretrained('bert-base-uncased')

			# print(self.encoder.embeddings.position_embeddings.weight)
			# exit()

			# if requested, we have to extend the parameters of the positional embeddings, so that the input length limit can be increased.
			if args.input_length_limit > 512:
				# create new positional embeddings of the appropriate new max input length
				pos_emb_weights = torch.Tensor(args.input_length_limit, 768)
				# initialize their values
				torch.nn.init.normal_(pos_emb_weights)	
				# copy first 512 positional embeddings from the pretrained bert
				pos_emb_weights [:512, :] = self.encoder.embeddings.position_embeddings.weight.clone()





				# create Embeddings, out of the parameters
				new_positional_embeddings = nn.Embedding(num_embeddings=args.input_length_limit, embedding_dim=768, _weight= pos_emb_weights)


				# print(new_positional_embeddings.weight.size())
				# print(self.encoder.embeddings.position_embeddings.weight.size())
				# copy pretrained parameters of existing positions (<=512)
				# new_positional_embeddings.weight[:512, :] = torch.ones(512,768).double() #self.encoder.embeddings.position_embeddings.weight.clone()

				# print(new_positional_embeddings.weight.size())
				# replace paramters in model
				# self.encoder.embeddings.position_embeddings.weight = nn.Parameter(new_positional_embeddings)


				# replace the old positional embeddings of the model with the new ones
				self.encoder.embeddings.position_embeddings = new_positional_embeddings

				# print(self.encoder.embeddings.position_embeddings)
				# exit()

				# output = self.encoder.embeddings.position_embeddings(torch.tensor(np.full((2, 1000), fill_value=800, dtype=np.int64)))

				# do necessary changes in the instantiated and pretrained bert model, in order to adjust to the new max input length
				self.encoder.embeddings.register_buffer("position_ids", torch.arange(args.input_length_limit).expand((1, -1)))

				# print(output.size())
				# exit()
				self.encoder.embeddings.position_embeddings.num_embeddings = args.input_length_limit

				self.encoder.embeddings.max_position_embeddings = args.input_length_limit

				# update parameter's value on BERT's config
				self.encoder.config.max_position_embeddings = args.input_length_limit

				# print(self)
				# exit()

			# extend vocab size of the model accordingly as well
			self.encoder._resize_token_embeddings(new_num_tokens = vocab_size)
			# else:
			# 	# print("vocab_size :", vocab_size)
			# 	# print("input_length_limit :", args.input_length_limit)
			# 	# print("hidden_size :", args.hidden_size)
			# 	# config = transformers.GPT2Config(vocab_size_or_config_json_file = vocab_size, n_positions = args.input_length_limit, n_ctx = args.input_length_limit, n_embd = args.hidden_size,
			# 	#                                  n_layer = args.num_hidden_layers, n_head = args.num_attention_heads)


			# 	config = transformers.BertConfig(vocab_size_or_config_json_file = vocab_size, hidden_size = args.hidden_size, num_hidden_layers = args.num_hidden_layers,
			# 									num_attention_heads = args.num_attention_heads, intermediate_size = args.intermediate_size, max_position_embeddings = args.input_length_limit)
			# 	# config.output_attentions = True


			# 	# config.output_past = True
			# 	# config.output_hidden_states=True

			# 	# # self.encoder = transformers.GPT2Model(config)

			# 	# (vocab_size_or_config_json_file=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1,
			# 	#     attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, **kwargs


			# 	self.encoder = transformers.BertModel(config)



			self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=args.adam_epsilon, weight_decay=0, amsgrad=False)




		
		# if self.args.use_ground_truth == False:

			# if self.CLS_mode == "1_CLS":
			# 	self.n_cls_tokens = 1
			# 	self.cat_prediction = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=self.cat_size)

			# elif self.CLS_mode == "C_CLS_1_linear":
			# 	self.n_cls_tokens = self.cat_size

			# 	# have ONE trainable linear function for ALL CLS token (functioning only as a switch mechanism)
			# 	self.cat_prediction = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1)

			# elif self.CLS_mode == "C_CLS_C_linears":
			self.n_cls_tokens = self.cat_size
			# have a trainable linear function for each CLS token
			self.cat_prediction = nn.ModuleList()
			for i in range(self.n_cls_tokens):
				self.cat_prediction.append(nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1))


			# self.sentiment_analysis_prediction = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1)
			# self.sentiment_analysis_prediction = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=3)

		# set up loss functions
			self.mse_loss = torch.nn.MSELoss(reduction='mean')
		# using same weights with ReDial SA model [1. / 5, 1. / 80, 1. / 15] (Based on label frequency distribution (4.9%, 81%, 14%))
		# self.SA_crossE_Loss = torch.nn.CrossEntropyLoss(weight = torch.Tensor([1. / 5, 1. / 80, 1. / 15]), ignore_index=-1, reduction="mean")


		# if self.args.use_cuda and self.args.n_gpu > 1:
		# 	self.encoder = torch.nn.DataParallel(self.encoder)

		# self.CrossE_Loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")


	# def get_movie_ratings_vector(self, sa_pred, batch_movie_mentions):
	# 	movie_rating_vector = torch.zeros(sa_pred.size(0), self.n_movies)

	# 	# used the liked probability, so that it is directly comperable with the ReDial baseline
	# 	# liked_pred = sa_pred[:,:,1]
	# 	liked_pred = sa_pred
	# 	# otherwise, you can apply softmax only on the first two dimensions, redistributing the probability of "did not say", to the other two cases (liked/disliked)

	# 	if sa_pred.is_cuda:
	# 		movie_rating_vector = movie_rating_vector.cuda()

	# 	for i in range(len(batch_movie_mentions)):
	# 		for index, movie_id in batch_movie_mentions[i]:
	# 			movie_rating_vector[i, movie_id] = liked_pred[i, index]

	# 	return movie_rating_vector


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
		size_of_minibatch = self.args.max_samples_per_gpu

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


	# def normalize_losses(self, losses):

	# 	# if we are doing the semantic task, and we only pay attention to 1 of the tasks then we only return the loss of this task NOT NORMALIZED
	# 	if len(losses) == 2 and (self.cat_sa_alpha == 0 or self.cat_sa_alpha == 1):
	# 		return losses

	# 	normalized_losses = []
	# 	for i, loss in enumerate(losses):
	# 		# print(loss)
	# 		normalized_loss = (losses[i] - self.min_max_losses[i][0]) / (self.min_max_losses[i][1] - self.min_max_losses[i][0])
	# 		# print("Normalized :", normalized_loss)
	# 		normalized_losses.append(normalized_loss)

	# 	return normalized_losses

	# def interpolate_losses(self, losses):
	# 	# we interpolate the losses

	# 	# if we are doing the nlg task
	# 	if len(losses) == 1:
	# 		return losses[0]
	# 	# if we are doing the semantic task
	# 	elif len(losses) == 2:
	# 		return losses[0] * self.cat_sa_alpha + losses[1] * ( 1 - self.cat_sa_alpha )
	# 	# if we are doing both tasks
	# 	else:
	# 		return ( losses[0] * self.cat_sa_alpha + losses[1] * ( 1 - self.cat_sa_alpha ) ) * self.sem_nlg_alpha + losses[2] * ( 1 - self.sem_nlg_alpha )



	# forward function, splits batch to minibaches, concatentes final outputs, and normalizes losses over batch, w.r.t. number of targets per task
	def forward_batch(self, batch, train = False):
		# will be used for appropriatly averaging the losses, from the minibatches, depending on the number of sampels per minibatch

		# batch_normalized_losses = [0, 0, 0]
		batch_loss = 0
		batch_output = []
		# batch_interpolated_loss = 0

		# batch_SA_targets = (batch["sentiment_analysis_targets"] != -1).sum()

		batch_size = batch["contexts"].size(0)

		# print("Batch size:", batch_size)

		minibatches = self.split_batch_to_minibatches(batch)

		for minibatch in minibatches:

			minibatch_size = minibatch["contexts"].size(0)
			# print("minibatch_size :", minibatch_size)

			# minibatch_SA_targets = (minibatch["sentiment_analysis_targets"] != -1).sum()

			minibatch_to_batch_ratio = minibatch_size / batch_size
			# print("minibatch ratio :", minibatch_to_batch_ratio)
			# exit()

			# if next(self.parameters()).is_cuda and self.args.n_gpu > 1:
			# 	minibatch = self.make_batch_multiple_of_GPUs_for_DataParallel(minibatch)

			cat_pred, cat_loss = self.forward(minibatch)


			# normalize Category loses so that the represent tha minibatch w.r.t. the complete batch
			cat_loss *= minibatch_to_batch_ratio
			# print(cat_loss)
			# exit()

			# normalize SA loss of minibatch by the number of SA targets on the minibatch (w.r.t. batch SA targets)
			# losses[1] *= minibatch_SA_targets.float() / batch_SA_targets.float()

			# record the unnormalized losses
			# for i in range(len(losses)):
			# 	batch_loss += cat_loss.item()*minibatch_to_batch_ratio

			# if the task at hand has more than one loss (involves two or more tasks), then we normalize the losses
			# if len(losses) != 1 and train:
			# 	losses = self.normalize_losses(losses)

			# loss = self.interpolate_losses(losses)

			# cat_loss *= minibatch_to_batch_ratio

			# batch_interpolated_loss += loss.item()

			batch_output.append(cat_pred)


			# for i in range(len(losses)):
			# 	# append the losses of the batch
			# 	batch_normalized_losses[i] += losses[i].item()
			# 	# register outputs
			# 	batch_outputs[i].append(outputs[i])

			# some minibatches might not have any targets (SA mainly)
			if cat_loss != 0 and train:
				# print(loss)
				cat_loss.backward()

		if train:
			self.optimizer.step()

		# # we remove the lists that are not being used
		# while len(batch_normalized_losses) != len(losses):
		# 	del batch_normalized_losses[-1]
		# 	del batch_outputs[-1]
		# 	del batch_losses[-1]

		# for i in range(len(batch_outputs)):
		# 	# concatenate outputs of minibatches
		batch_output = torch.cat(batch_output, dim = 0)

		return batch_output, cat_loss



	def evaluate_model(self, batch_loader, subset):

		n_batches = batch_loader.n_batches[subset]

		if self.args.debug_run:
			n_batches = 3

		# self.min_max_losses = torch.Tensor([[1e10, 0], [1e10, 0], [1e10, 0]])
		
		# if next(self.parameters()).is_cuda:
		# 	self.min_max_losses = self.min_max_losses.cuda()

		# total_losses = [[], [], []]


		total_loss = 0
		num_of_samples = 0

		with torch.no_grad():

			# for step in range(n_batches):
			for step in tqdm(range(n_batches)):

				# torch.cuda.empty_cache()

				# we retrieve a batch
				batch = batch_loader.load_batch(subset = subset)

				batch_size = batch["contexts"].size(0)

				num_of_samples += batch_size


				if batch == None:
					continue

				_, cat_loss = self.forward_batch(batch)

				total_loss += cat_loss.item() * batch_size

				# we keep track of minimum and maximum loss values per task, in oreder to later normalize for joint task training
				# for i, loss in enumerate(batch_losses):
				# 	if loss < self.min_max_losses[i][0]:
				# 		self.min_max_losses[i][0] = loss
				# 	if loss > self.min_max_losses[i][1]:
				# 		self.min_max_losses[i][1] = loss

					# total_losses[i].append(loss.item())

		# we remove the total_losses that are not being used
		# while len(total_losses[-1]) == 0:
		#     del total_losses[-1]


		# # we average the losses and normalize them
		# average_losses = []
		# for i, loss in enumerate(total_losses):
		#     average_losses.append(torch.FloatTensor(total_losses[i]).mean())

		# normalized_losses = self.normalize_losses(batch_losses)

		# average_losses = [ loss.item() for loss in average_losses]

		# interpolated_loss = self.interpolate_losses(normalized_losses)

		total_loss /= num_of_samples

		return  total_loss



	def train_epoch(self, batch_loader):


		n_batches = batch_loader.n_batches["train"]

		if self.args.debug_run:
			n_batches = 3

		total_loss = 0
		num_of_samples = 0

		# interpolated_losses = []

		# for step in range(n_batches):
		for step in tqdm(range(n_batches)):

			self.optimizer.zero_grad()

			# we retrieve a batch
			batch = batch_loader.load_batch(subset = "train")

			if batch == None:
				continue

			batch_size = batch["contexts"].size(0)

			num_of_samples += batch_size


			# batch_outputs, batch_losses
			# batch_outputs, batch_losses, batch_interpolated_loss
			_, cat_loss= self.forward_batch(batch, train=True)

			total_loss += cat_loss*batch_size

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

		total_loss /= num_of_samples

		return  total_loss

# class FlatSemanticTransformer(TransformerEncoder):
# 	def __init__(self, vocab_size, cat_size, n_movies, args = None):
# 		super(FlatSemanticTransformer, self).__init__(vocab_size, cat_size, n_movies, args)

# 		if self.args.use_ground_truth == False:

# 			if self.CLS_mode == "1_CLS":
# 				self.n_cls_tokens = 1
# 				self.cat_prediction = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=self.cat_size)

# 			elif self.CLS_mode == "C_CLS_1_linear":
# 				self.n_cls_tokens = self.cat_size

# 				# have ONE trainable linear function for ALL CLS token (functioning only as a switch mechanism)
# 				self.cat_prediction = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1)

# 			elif self.CLS_mode == "C_CLS_C_linears":
# 				self.n_cls_tokens = self.cat_size
# 				# have a trainable linear function for each CLS token
# 				self.cat_prediction = nn.ModuleList()
# 				for i in range(self.n_cls_tokens):
# 					self.cat_prediction.append(nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1))


# 			# self.sentiment_analysis_prediction = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1)
# 			self.sentiment_analysis_prediction = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=3)


# 		if self.args.use_cuda and self.args.n_gpu > 1:
# 			self.encoder = torch.nn.DataParallel(self.encoder)


	def forward(self, batch):


		contexts = batch["contexts"]
		token_types = batch["token_types"]
		attention_masks = batch["attention_masks"]
		category_targets = batch["category_targets"]
		# sentiment_analysis_targets = batch["sentiment_analysis_targets"]
		# nlg_targets = batch["nlg_targets"]
		# nlg_gt_inputs = batch["nlg_gt_inputs"]

		# mask = sentiment_analysis_targets != -1

		# print(sentiment_analysis_targets[mask])

		# contexts, token_types, attention_masks, category_targets, sentiment_analysis_targets, _, _, batch_movie_mentions, _ = batch

		# if the model is on cuda, we transfer all tensors to cuda
		if next(self.parameters()).is_cuda:
			contexts, token_types, attention_masks, category_targets = \
			 contexts.cuda(), token_types.cuda(), attention_masks.cuda(), category_targets.cuda()



		# print("Contexts :", contexts.size())
		# print("attention_masks :", attention_masks.size())
		# print("token_types :", token_types.size())
		# print()

		# print(contexts)


		# last_hidden_state, _ = self.encoder(input_ids = contexts, attention_mask=attention_masks, token_type_ids=token_types)
		last_hidden_state = self.encoder(input_ids = contexts, attention_mask=attention_masks, token_type_ids=token_types)[0]
		# last_hidden_state = x[0]
		# print(last_hidden_state)
		# exit()

		# for i in x:
		# 	print(i)
		# 	print(type(i))
		# exit()

		# print(last_hidden_state.size())
		# print(last_hidden_state)

		# we only use the hidden states of the last layer
		# # hidden_states = last_hidden_state # hidden_states[-1]

		# print(last_hidden_state)
		# print(self.n_cls_tokens + 1)
		# print()
		# print(last_hidden_state.size())
		# we only use the first CLS tokens as cat prediction input

		cls_input = last_hidden_state[:, : self.n_cls_tokens, :]

		# print("CLS_input:")
		# print(cls_input.size())
		# print(cls_input.sum(dim=-1))



		# if self.CLS_mode == "1_CLS":
		# 	cat_pred = self.cat_prediction(cls_input).view(category_targets.size())

		# elif self.CLS_mode == "C_CLS_1_linear":
		# 	# rearrange size so that all CLS hidden activations go through the same linear
		# 	cls_input = cls_input.reshape(-1, self.encoder.config.hidden_size)
		# 	cat_pred = self.cat_prediction(cls_input)
		# 	# rearrange size so that |C| activtions go for each sample
		# 	cat_pred = cat_pred.view(category_targets.size())

		# elif self.CLS_mode == "C_CLS_C_linears":

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
		# print(cat_pred)
		# exit()
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
		# sa_pred = self.sentiment_analysis_prediction(last_hidden_state) 

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
		# sa_mask = (sentiment_analysis_targets != -1).view(sentiment_analysis_targets.size())


		# print(sa_pred.size())
		# print(sentiment_analysis_targets.size())
		# print(sa_mask.sum())
		# print(torch.tensor([0,1]))
		# print()


		# # if there are no SA targets, then we do not calculate the SA loss, and set it to 0 (because it returns nan with empty tensors)
		# if sa_mask.sum() == 0:
		# 	sa_loss = self.mse_loss(torch.ones(1), torch.ones(1))
		# 	if next(self.parameters()).is_cuda:
		# 		sa_loss = sa_loss.cuda()
		# else:
		# 	# print("Predicted :")
		# 	# print(sa_pred[sa_mask])
		# 	# print("Targets :")
		# 	# print(sentiment_analysis_targets[sa_mask])
		# 	# print("Cat loss:", sa_pred[sa_mask].size(), sentiment_analysis_targets[sa_mask].size())
		# 	# sa_loss = self.mse_loss(sa_pred[sa_mask].view(-1), sentiment_analysis_targets[sa_mask].view(-1))
		# 	# print("SA Output :", sa_pred[sa_mask])
		# 	# print("SA Targets :", sentiment_analysis_targets[sa_mask])
		# 	sa_loss = self.SA_crossE_Loss(sa_pred[sa_mask], sentiment_analysis_targets[sa_mask].long())

		# 	# print("SA Loss:", sa_loss)



		# # # if there are no SA targets, then we do not calculate the SA loss, and set it to 0 (because it returns nan with empty tensors)
		# # if sa_mask.sum() == 0:
		# #     sa_loss = self.CrossE_Loss(torch.tensor([0,1]), torch.ones(1))
		# #     if next(self.parameters()).is_cuda:
		# #         sa_loss = sa_loss.cuda()
		# # else:
		# # apply softmax on SA prediction
		# sa_pred = torch.nn.functional.softmax( sa_pred, dim= -1)

		# # Rating prediction is equal to liked probability - the disliked probability in [-1,1]
		# rating_pred = sa_pred[:,:,1] - sa_pred[:,:,0]

		return cat_pred, cat_loss

		# || Train this class with interleaving parameters on loss
		# make a train function
		# make an eval function (make them on TransformerEncoder class, and use the appropriate forwards, self.alpha will affect the inter)

