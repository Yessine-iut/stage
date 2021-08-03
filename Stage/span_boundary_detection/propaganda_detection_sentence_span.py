# from transformers import BertTokenizer, BertForTokenClassification, AdamW, BertModel, BertForSequenceClassification
from nltk.tokenize.util import string_span_tokenize
# from unidecode import unidecode
import torch, os, ast, string, pickle, itertools, pandas as pd, numpy as np, math, json, random
import torch.nn.functional as F
import torch.nn as nn
# from toolz import unique
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.linear_model import LogisticRegression
from collections import Counter

from custom_transformers.src.transformers.models.bert import BertTokenizer, BertForTokenClassification, BertForTokenClassification_CRF
from custom_transformers.src.transformers.models.roberta import RobertaTokenizer, RobertaForSequenceClassification_joint_loss
from custom_transformers.src.transformers.optimization import AdamW

from extractPropagandaFeatures import get_propagandaFeatures

def getFiles_in_folder(path):
	return os.listdir(path)

#### rearrange data to multi-class
def arrangeDataFrame(path_articles,tokenizer,dict_labels,task):
	# read dir
	path_articles = path_articles+'/'
	file_type = ''
	if task == 'nlp4if': file_type = '.labels.tsv'
	elif task == 'semeval': file_type = '.task-flc-tc.labels'
	files = getFiles_in_folder(path_articles)#[:5]
	article_names, span_names = [],[]

	for file in files:
		if '.txt' in file: article_names.append(file)
		elif file_type in file: span_names.append(file)
	saved_list = []

	# get file names - each file
	for idx,article in enumerate(article_names):
		article_name = article.split('.')[0]
		# print(article_name)
		# tmp_propaganda_text, tmp_none_propaganda_text = [], []

		#find spans on each article
		with open(path_articles+article_name+file_type) as s:
			propaganda_spans = s.readlines()
			text = open(path_articles+article_name+'.txt','r').read()

			sentences = text.split('\n')
			sentence_spans = list(string_span_tokenize(text,'\n'))
			# print('=====each file:',sentences,'\n',sentence_spans)
			collect_propaganda_spans, collect_duplicated_propaganda_spans, all_sentences_span = [], [], []

			for i,sentence_span in enumerate(sentence_spans):
				token_start = sentence_spans[i][0]
				token_end = sentence_spans[i][1]
				# print('each sentence_span: (',token_start,token_end,')')
				all_sentence_span_range = range(token_start,token_end)
				# all_sentences_span.all_sentence_span_range
				# print('====all_sentence_span_range',all_sentence_span_range)
				# exit()

				# ONLY READ LABELED_FRAGMENT - convert spans to num_label, and collect spans
				for span in propaganda_spans:
					span = span.split('\t')
					article_id, label, span_begin, span_end = span[0],span[1],int(span[2]),int(span[3])
					label = dict_labels[label] 	# make dict to match labels

					# if propaganda_span IN current sentence_span : write dict_label
					if span_begin in all_sentence_span_range and span_end in all_sentence_span_range:  
						# collect all spans in numeric - compare later
						collect_duplicated_propaganda_spans.append([article_id, label, span_begin, span_end,(token_start,token_end)])
						collect_propaganda_spans.append((token_start,token_end))
				# print('-----collect_propaganda_spans',len(collect_propaganda_spans)) # propaganda_spans for each article

			
			# print('=====collect_propaganda_spans',collect_propaganda_spans,len(collect_propaganda_spans))
			# print('=====sentence_spans',sentence_spans,len(sentence_spans))
			# exit()
			for i,sentence_span in enumerate(sentence_spans):
				# print('>>> NOW sentence_span:',sentence_span)
				# print('collect_propaganda_spans',collect_propaganda_spans)
				### WRITE THE NON-PROPAGANDA
				if sentence_span not in collect_propaganda_spans:
					# print(i,'sentence_span not in',sentence_span, sentence_span[0],sentence_span[1])
					none_propaganda_span = text[sentence_span[0]:sentence_span[1]]
					non_propaganda_token_labels = []
					encoded = tokenizer.encode(none_propaganda_span)
					# print('Encoded:',encoded,len(encoded))
					for i,e in enumerate(encoded):
						token_decoded = tokenizer.decode(e).replace(' ','')
						# print(i,'Decoded:',token_decoded,len(token_decoded))
						# if token_decoded in unidecode(propaganda_span).lower() or '##' in token_decoded:  # to match BERT tokenizer
						# 	token_label = label
						# else: 
						token_label = 0
						non_propaganda_token_labels.append(token_label)

					saved_list.append([none_propaganda_span,non_propaganda_token_labels]) 
					# print('none_propaganda_span:\t\t',[none_propaganda_span,non_propaganda_token_labels],'\n============\n') ## completed
					# exit()
				### WRITE THE PROPAGANDA
				else:
					# check how many times the tuple duplicates
					# print('----collect_duplicated_propaganda_spans',collect_duplicated_propaganda_spans)
					all_tokenTuples = []
					for propaganda_span in collect_duplicated_propaganda_spans:
						article_id, label, span_begin, span_end,(token_start,token_end) = propaganda_span
						# print(article_id, label, span_begin, span_end,(token_start,token_end))
						all_tokenTuples.append((token_start,token_end))
					unique_propaganda_tuples = list(set(all_tokenTuples))
					# print(len(all_tokenTuples),len(list(set(all_tokenTuples))))
					# print(all_tokenTuples,list(set(all_tokenTuples)))
					# print('1111111sentence_span',sentence_span)
					
					for unique_span in unique_propaganda_tuples:
						# print('----unique_span',unique_span)
						count = 0
						whole_spans = []

						if unique_span == sentence_span:   # focus only propaganda spans
							# print(unique_span, sentence_span)

							# check duplication
							# get propaganda_span_values
							for propaganda_span in collect_duplicated_propaganda_spans: 
								article_id, label, span_begin, span_end,(token_start,token_end) = propaganda_span
								if unique_span == (token_start,token_end):
									# print(2,unique_span,(token_start,token_end))
									count += 1
									# whole_spans.extend(propaganda_span)
									# whole_spans_duplicate.append(propaganda_span)
									whole_spans.append(propaganda_span)
							# print('>>>>whole_spans,count',whole_spans,count)
							if count == 1: # NO duplicate
								article_id, label, span_begin, span_end,(token_start,token_end) = whole_spans[0]
								propaganda_sentence = text[sentence_span[0]:sentence_span[1]]
								propaganda_span = text[span_begin:span_end]
								# print('-------------propaganda_span', propaganda_span)
								# exit()

								propaganda_token_labels = []
								encoded = tokenizer.encode(propaganda_sentence)

								for i,e in enumerate(encoded):
									token_decoded = tokenizer.decode(e).replace(' ','')
									# print(i,'Decoded:',token_decoded,len(token_decoded))
									if token_decoded in unidecode(propaganda_span).lower() or '##' in token_decoded:  # to match BERT tokenizer
										token_label = label
										# print('check',token_decoded,'\t//////\t',propaganda_span)
									else: # WITH duplicate
										token_label = 0
									propaganda_token_labels.append(token_label)
									# print('==============',[propaganda_sentence,propaganda_token_labels])
									# exit()							
								saved_list.append([propaganda_sentence,propaganda_token_labels])
							elif count > 1: 
								# print('++++found duplicated',unique_span,whole_spans)

								# get all sets of begin_offset and enf_offset to match
								# print('___whole_spans',whole_spans, len(whole_spans))
								# exit()

								article_id_list, label_list, span_begin_list, span_end_list,token_start_list,token_end_list = [],[],[],[],[],[]
								for whole_span in whole_spans:
									article_id, label, span_begin, span_end,(token_start,token_end) = whole_span
									label_list.append(label)
									span_begin_list.append(span_begin)
									span_end_list.append(span_end)

								propaganda_spans = []	
								propaganda_sentence = text[sentence_span[0]:sentence_span[1]]
								for label, span_begin, span_end in zip(label_list,span_begin_list,span_end_list):
									# print(0,span_begin_list,span_end_list)
									propaganda_span = unidecode(text[span_begin:span_end]).lower()
									propaganda_spans.append(propaganda_span)

								propaganda_token_labels = []
								encoded = tokenizer.encode(propaganda_sentence)
								for e in encoded:
									token_decoded = tokenizer.decode(e).replace(' ','')
									# print(token_decoded,'----propaganda_spans:',propaganda_spans)
									# exit()
									# print(i,'Decoded:',token_decoded,len(token_decoded))
									# print('--token_decoded',token_decoded)
									if any(token_decoded.replace('##','') in span for span in propaganda_spans):  # to match BERT tokenizer # or '##' in token_decoded
										for label,propaganda_span in zip(label_list,propaganda_spans):
										# 	if token_decoded in 
										# exit()
											if token_decoded.replace('##','') in propaganda_span:
												token_label = label
										# print('check',token_decoded,'\t//////\t',propaganda_span)
									else: # WITH duplicate
										token_label = 0
									propaganda_token_labels.append(token_label)
								# print('==============',[propaganda_sentence,propaganda_token_labels])						
								saved_list.append([propaganda_sentence,propaganda_token_labels])
								# exit()
							# saved_list.append([propaganda_span,propaganda_token_labels]) 

						# reset parameters
						whole_spans = [] 
						count = 0
	# put in df, remove duplicate rows
	# print('saved_list----',saved_list)
	df = pd.DataFrame(saved_list,columns=['text','labels'])
	df = df[~df.astype(str).duplicated()]
	return df 

def arrangeDataFrame_nlp4if_testWithArticleNum(path_articles,tokenizer,dict_labels,task='nlp4if'):
	# read dir
	path_articles = path_articles+'/'
	file_type = ''
	if task == 'nlp4if': file_type = '.labels.tsv'
	files = getFiles_in_folder(path_articles)#[:5]
	article_names, span_names = [],[]

	for file in files:
		if '.txt' in file: article_names.append(file)
		elif file_type in file: span_names.append(file)
	saved_list = []

	# get file names - each file
	for idx,article in enumerate(article_names):
		article_name = article.split('.')[0]
		article_id = article_name.split('e')[1]

		#find spans on each article
		with open(path_articles+article_name+file_type) as s:
			propaganda_spans = s.readlines()
			text = open(path_articles+article_name+'.txt','r').read()

			sentences = text.split('\n')
			sentence_spans = list(string_span_tokenize(text,'\n'))
			# print('=====each file:',sentences,'\n',sentence_spans)

			collect_propaganda_spans, collect_duplicated_propaganda_spans, all_sentences_span = [], [], []

			for i,sentence_span in enumerate(sentence_spans):
				token_start = sentence_spans[i][0]
				token_end = sentence_spans[i][1]
				# print('each sentence_span: (',token_start,token_end,')')
				all_sentence_span_range = range(token_start,token_end)
				# all_sentences_span.all_sentence_span_range
				# print('====all_sentence_span_range',all_sentence_span_range)
				# exit()

				# ONLY READ LABELED_FRAGMENT - convert spans to num_label, and collect spans
				for span in propaganda_spans:
					span = span.split('\t')
					article_id, label, span_begin, span_end = span[0],span[1],int(span[2]),int(span[3])
					label = dict_labels[label] 	# make dict to match labels

					# if propaganda_span IN current sentence_span : write dict_label
					if span_begin in all_sentence_span_range and span_end in all_sentence_span_range:  
						# collect all spans in numeric - compare later
						collect_duplicated_propaganda_spans.append([article_id, label, span_begin, span_end,(token_start,token_end)])
						collect_propaganda_spans.append((token_start,token_end))
				# print('-----collect_propaganda_spans',len(collect_propaganda_spans)) # propaganda_spans for each article

			
			# print('=====collect_propaganda_spans',collect_propaganda_spans,len(collect_propaganda_spans))
			# print('=====sentence_spans',sentence_spans,len(sentence_spans))
			# exit()
			for i,sentence_span in enumerate(sentence_spans):
				# print('>>> NOW sentence_span:',sentence_span)
				# print('collect_propaganda_spans',collect_propaganda_spans)
				### WRITE THE NON-PROPAGANDA
				if sentence_span not in collect_propaganda_spans:
					# print(i,'sentence_span not in',sentence_span, sentence_span[0],sentence_span[1])
					none_propaganda_span = text[sentence_span[0]:sentence_span[1]]
					non_propaganda_token_labels = []
					encoded = tokenizer.encode(none_propaganda_span)
					# print('Encoded:',encoded,len(encoded))
					for i,e in enumerate(encoded):
						token_decoded = tokenizer.decode(e).replace(' ','')
						# print(i,'Decoded:',token_decoded,len(token_decoded))
						# if token_decoded in unidecode(propaganda_span).lower() or '##' in token_decoded:  # to match BERT tokenizer
						# 	token_label = label
						# else: 
						token_label = 0
						non_propaganda_token_labels.append(token_label)

					saved_list.append([article_id,none_propaganda_span,non_propaganda_token_labels]) 
					# print('none_propaganda_span:\t\t',[none_propaganda_span,non_propaganda_token_labels],'\n============\n') ## completed
					# exit()
				### WRITE THE PROPAGANDA
				else:
					# check how many times the tuple duplicates
					# print('----collect_duplicated_propaganda_spans',collect_duplicated_propaganda_spans)
					all_tokenTuples = []
					for propaganda_span in collect_duplicated_propaganda_spans:
						article_id, label, span_begin, span_end,(token_start,token_end) = propaganda_span
						# print(article_id, label, span_begin, span_end,(token_start,token_end))
						all_tokenTuples.append((token_start,token_end))
					unique_propaganda_tuples = list(set(all_tokenTuples))
					# print(len(all_tokenTuples),len(list(set(all_tokenTuples))))
					# print(all_tokenTuples,list(set(all_tokenTuples)))
					# print('1111111sentence_span',sentence_span)
					
					for unique_span in unique_propaganda_tuples:
						# print('----unique_span',unique_span)
						count = 0
						whole_spans = []

						if unique_span == sentence_span:   # focus only propaganda spans
							# print(unique_span, sentence_span)

							# check duplication
							# get propaganda_span_values
							for propaganda_span in collect_duplicated_propaganda_spans: 
								article_id, label, span_begin, span_end,(token_start,token_end) = propaganda_span
								if unique_span == (token_start,token_end):
									# print(2,unique_span,(token_start,token_end))
									count += 1
									# whole_spans.extend(propaganda_span)
									# whole_spans_duplicate.append(propaganda_span)
									whole_spans.append(propaganda_span)
							# print('>>>>whole_spans,count',whole_spans,count)
							if count == 1: # NO duplicate
								article_id, label, span_begin, span_end,(token_start,token_end) = whole_spans[0]
								propaganda_sentence = text[sentence_span[0]:sentence_span[1]]
								propaganda_span = text[span_begin:span_end]
								# print('-------------propaganda_span', propaganda_span)
								# exit()

								propaganda_token_labels = []
								encoded = tokenizer.encode(propaganda_sentence)

								for i,e in enumerate(encoded):
									token_decoded = tokenizer.decode(e).replace(' ','')
									# print(i,'Decoded:',token_decoded,len(token_decoded))
									if token_decoded in unidecode(propaganda_span).lower() or '##' in token_decoded:  # to match BERT tokenizer
										token_label = label
										# print('check',token_decoded,'\t//////\t',propaganda_span)
									else: # WITH duplicate
										token_label = 0
									propaganda_token_labels.append(token_label)
									# print('==============',[propaganda_sentence,propaganda_token_labels])
									# exit()							
								saved_list.append([article_id,propaganda_sentence,propaganda_token_labels])
							elif count > 1: 
								# print('++++found duplicated',unique_span,whole_spans)

								# get all sets of begin_offset and enf_offset to match
								# print('___whole_spans',whole_spans, len(whole_spans))
								# exit()

								article_id_list, label_list, span_begin_list, span_end_list,token_start_list,token_end_list = [],[],[],[],[],[]
								for whole_span in whole_spans:
									article_id, label, span_begin, span_end,(token_start,token_end) = whole_span
									label_list.append(label)
									span_begin_list.append(span_begin)
									span_end_list.append(span_end)

								propaganda_spans = []	
								propaganda_sentence = text[sentence_span[0]:sentence_span[1]]
								for label, span_begin, span_end in zip(label_list,span_begin_list,span_end_list):
									# print(0,span_begin_list,span_end_list)
									propaganda_span = unidecode(text[span_begin:span_end]).lower()
									propaganda_spans.append(propaganda_span)

								propaganda_token_labels = []
								encoded = tokenizer.encode(propaganda_sentence)
								for e in encoded:
									token_decoded = tokenizer.decode(e).replace(' ','')
									# print(token_decoded,'----propaganda_spans:',propaganda_spans)
									# exit()
									# print(i,'Decoded:',token_decoded,len(token_decoded))
									# print('--token_decoded',token_decoded)
									if any(token_decoded.replace('##','') in span for span in propaganda_spans):  # to match BERT tokenizer # or '##' in token_decoded
										for label,propaganda_span in zip(label_list,propaganda_spans):
										# 	if token_decoded in 
										# exit()
											if token_decoded.replace('##','') in propaganda_span:
												token_label = label
										# print('check',token_decoded,'\t//////\t',propaganda_span)
									else: # WITH duplicate
										token_label = 0
									propaganda_token_labels.append(token_label)
								# print('==============',[propaganda_sentence,propaganda_token_labels])						
								saved_list.append([article_id,propaganda_sentence,propaganda_token_labels])
								# exit()
							# saved_list.append([propaganda_span,propaganda_token_labels]) 

						# reset parameters
						whole_spans = [] 
						count = 0
	# put in df, remove duplicate rows
	# print('saved_list----',saved_list)
	df = pd.DataFrame(saved_list,columns=['article_id','text','labels'])
	df = df[~df.astype(str).duplicated()]
	return df 
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def combine_predictions(preds):
	rearraged_sentence_list = []
	for i,sentence in enumerate(preds): # each sentence (128,19)
		label_by_token = []
		# print('sentence:',i,sentence,sentence.shape)
		for j,token in enumerate(sentence):  # each token (19)
			# print('token:',j,token,token.shape)
			max_score = max(token) # find max of list
			j = 0
			minmax_token = [1 if i==max_score else j for i in token] # if max of token==1, then map with dict
			# print('minmax_token:',j,minmax_token)
			index_position = minmax_token.index(1)  # get position of predicted label
			# print(index_position) 
			label_by_token.extend([index_position])
		rearraged_sentence_list.append(label_by_token)
	# print('rearraged_sentence_list:',rearraged_sentence_list,len(rearraged_sentence_list))
	return rearraged_sentence_list
def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys
def range_subset(range1, range2): # (sub_range,whole_range)
    """Whether range1 is a subset of range2."""
    if not range1:
        return True  # empty range is subset of anything
    if not range2:
        return False  # non-empty range can't be subset of empty range
    if len(range1) > 1 and range1.step % range2.step:
        return False  # must have a single value or integer multiple step
    return range1.start in range2 and range1[-1] in range2


####################### JOINT-LOSS
def train_jointLoss_sentence_span(train_df,completeFeatures_vectors_train,model_name,train_limit,feature_type,num_batch=8,epochs=3,task='semeval'):
	device = "cuda:0" if torch.cuda.is_available() else "cpu"

	if task == 'semeval': num_labels = 15
	MAX_LEN = 128

	# if model_name == 'BERT':
	# 	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	# 	model = cBertForSequenceClassification_joint_loss.from_pretrained('bert-base-uncased',num_labels=num_labels)
	# 	lr=5e-5 ## (original = 3e-5)  1.0e-6
	# 	print('Using cBertForSequenceClassification_sentence_to_span')
	# elif model_name == 'RoBERTa':
	tokenizer = RobertaTokenizer.from_pretrained('roberta-base') #'roberta-base'
	model = RobertaForSequenceClassification_joint_loss.from_pretrained('roberta-base',num_labels=num_labels)
	lr = 2e-5 ## (original = 2e-5 )  default = 6e-4 works but not with extraFeatures|| doesn't learn =1.9e-6
	# prospoed 2e-3 works well with semantic -- lets see argumentation features -- can try lower lr?
	# the BEST 2e-2 or 2e-1 ===== 2e-1 for no_extraFeatures
	print('Using cRoBERTaForSequenceClassification_sentence_to_span')

	if completeFeatures_vectors_train is not None:	completeFeatures_vectors = completeFeatures_vectors_train
		# completeFeatures_vectors = [[0]] * train_df.shape[0]
	# else: completeFeatures_vectors = completeFeatures_vectors_train

	print('Learning rate:',lr)
	# completeFeatures_vectors = completeFeatures_vectors[0]

	# print('----completeFeatures_vectors',completeFeatures_vectors[:train_limit],len(completeFeatures_vectors[:train_limit]))
	# exit()


	# do batches for sentence-level 
	train_input_batches, train_label_batches, train_extraFeature_batches, train_propaganda_text_batches = [],[],[],[]
	for x in batch(train_df.text[:train_limit], num_batch):
		train_input_batches.append(list(x.values))
	for x in batch(train_df.labels[:train_limit], num_batch):
		train_label_batches.append(x)
	for x in batch(completeFeatures_vectors[:train_limit], num_batch):
		train_extraFeature_batches.append(x)
	for x in batch(train_df.propaganda_text[:train_limit], num_batch):
		train_propaganda_text_batches.append(x)

	# print('train_input_batches',train_input_batches[0],len(train_input_batches))
	# print('train_extraFeature_batches',train_extraFeature_batches[0],len(train_extraFeature_batches))
	# exit()

	# do batches for span-level 
	model = model.to(device)
	model.train()
	optim = AdamW(model.parameters(), lr=lr)

	for epoch in range(epochs):
		print('At epoch:',epoch+1,'/',epochs)
		for input_batch,label_batch, extraFeature_batch,span_batch in zip(train_input_batches,train_label_batches,train_extraFeature_batches,train_propaganda_text_batches):
		# for input_batch,label_batch,span_batch in zip(train_input_batches,train_label_batches,train_propaganda_text_batches):
			input_batch = [str(x) for x in input_batch] # in case of 'nan'
			inputs = tokenizer(input_batch, return_tensors="pt",padding='max_length', truncation=True, max_length=MAX_LEN) #
			span_batch = [str(x) for x in span_batch] # in case of 'nan'
			span_inputs = tokenizer(span_batch, return_tensors="pt",padding='max_length', truncation=True, max_length=20)
			# print('extraFeature_batch 1:', extraFeature_batch)
			
			if completeFeatures_vectors_train is None:
				extraFeature_batch = None
				# print('extraFeature_batch as -None-:',extraFeature_batch)
			else:
				extraFeature_batch = torch.tensor([extraFeature_batch])#.unsqueeze(0) ## try max_seq
				# print('extraFeatures',extraFeatures.size())
				# extraFeatures = torch.reshape(extraFeatures, (extraFeatures.size()[1],extraFeatures.size()[2],1)) # [batch_size, seq_len, input_size]
				extraFeature_batch = torch.reshape(extraFeature_batch, (extraFeature_batch.size()[1],1,extraFeature_batch.size()[2])) # [batch_size, seq_len, input_size]
			# print('reshaped',extraFeature_batch,extraFeature_batch.size())
			# exit()
			label_batch = torch.tensor(list(label_batch))
			# print('label_batch:',label_batch)
			
			if device != 'cpu':
				# print('putting into cuda')
				inputs = inputs.to(device)
				span_inputs = span_inputs.to(device)
				# if completeFeatures_vectors_train is not None:
				extraFeature_batch = extraFeature_batch.to(device)
				label_batch = label_batch.to(device)
				# print('done')

			# print('start training')
			# print('extraFeature_batch 3:', extraFeature_batch,type(extraFeature_batch))
			# exit()

			optim.zero_grad()
			outputs = model(**inputs, labels=label_batch, span=span_inputs, extraFeature=extraFeature_batch)
			# outputs = model(**inputs, labels=label_batch, span=span_inputs) #extraFeature=extraFeature_batch,
			# loss, logits = outputs[:2]   #loss, logits, attentions
			joint_loss,sen_logits,span_logits = outputs
			print('At epoch:',epoch+1,'/',epochs,'\tJoint_loss:',joint_loss)

			joint_loss.backward()
			# print('done 1')
			optim.step()
			# print('done 2')

	# if model_name == 'BERT':
	# 	model.save_pretrained('./fine_tuned_model_sentence_span/BertForSequenceClassification_jointLoss_sentence_span/')
	# elif model_name == 'RoBERTa':
	model.save_pretrained('./fine_tuned_model_sentence_span/RoBertaForSequenceClassification_jointLoss_sentence_span_'+feature_type)

def predict_jointLoss_sentence_span(test_df,completeFeatures_vectors_test,limit,dict_labels,epochs,runtime,oversmapled,args_dict,feature_type,additional_outName,model_name='BERT',num_batch=8):
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	num_labels = 15
	# file_type = ''
	# if task == 'semeval':
	# 	file_type = '.task-flc-tc.labels'
	#
	# article_names, span_names = [],[]

	# test_limit = 1

	# for file in files:
	# 	if '.txt' in file: article_names.append(file)
	# 	# elif file_type in file: span_names.append(file)
	# saved_list,collect_article = [],[]

	# if model_name == 'BERT':
	# 	bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	# 	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	# 	model = cBertForSequenceClassification_joint_loss.from_pretrained('./fine_tuned_model/BertForSequenceClassification_jointLoss_sentence_span/',num_labels=num_labels)
	# 	model_name = 'BertForSequenceClassification_jointLoss_sentence_span'
	# 	print('Using BertForSequenceClassification_jointLoss_sentence_span')
	# elif model_name == 'RoBERTa':
	bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	tokenizer = RobertaTokenizer.from_pretrained('roberta-base') #'roberta-base'
	model = RobertaForSequenceClassification_joint_loss.from_pretrained('./fine_tuned_model_sentence_span/RoBertaForSequenceClassification_jointLoss_sentence_span_'+feature_type,num_labels=num_labels)
	model_name = 'RobertaForSequenceClassification_joint_loss'
	print('Using RoBERTa')

	model = model.to(device)
	model.eval()

	if completeFeatures_vectors_test is not None: completeFeatures_vectors = completeFeatures_vectors_test
	# 	completeFeatures_vectors = [[0]] * test_df.shape[0]
	# else: completeFeatures_vectors = completeFeatures_vectors_test


	# do batches for sentence-level
	test_input_batches, test_propaganda_text_batches, test_article_num_batches, test_extraFeature_batches = [],[],[],[]
	for x in batch(test_df.text[:limit], num_batch):
		test_input_batches.append(list(x.values))
	for x in batch(completeFeatures_vectors[:limit], num_batch):
		test_extraFeature_batches.append(x)
	for x in batch(test_df.propaganda_text[:limit], num_batch):
		test_propaganda_text_batches.append(list(x.values))
	for x in batch(test_df.article_name[:limit], num_batch):
		test_article_num_batches.append(x)

	save_prediction = []

	for articleNum_batch, input_batch, span_batch, extraFeature_batch in zip(test_article_num_batches, test_input_batches,test_propaganda_text_batches,test_extraFeature_batches):
		# print('input_batch',input_batch,type(input_batch))
		tokenized_sentence = tokenizer(input_batch, return_tensors="pt", padding='max_length', truncation=True,max_length=128)
		# print('tokenized_sentence',tokenized_sentence)
		# exit()
		tokenized_span = tokenizer(span_batch, return_tensors="pt", padding='max_length', truncation=True,max_length=20)
		# label_batch = torch.tensor(list(label_batch))

		if completeFeatures_vectors_test is None:	extraFeature_batch = None
		else:
			extraFeature_batch = torch.tensor([extraFeature_batch])  # .unsqueeze(0) ## try max_seq
			# print('extraFeatures',extraFeatures.size())
			# extraFeatures = torch.reshape(extraFeatures, (extraFeatures.size()[1],extraFeatures.size()[2],1)) # [batch_size, seq_len, input_size]
			extraFeature_batch = torch.reshape(extraFeature_batch,(extraFeature_batch.size()[1], 1, extraFeature_batch.size()[2]))

		if device != 'cpu':
			tokenized_sentence = tokenized_sentence.to(device)
			tokenized_span = tokenized_span.to(device)
			# if completeFeatures_vectors_test is not None:
			extraFeature_batch = extraFeature_batch.to(device)
			print('done putting into cuda')

		_,_,pred_logits = model(**tokenized_sentence, span=tokenized_span, extraFeature=extraFeature_batch)
		# print('pred_logits', pred_logits, len(pred_logits))
		output_softmax = F.softmax(pred_logits)
		# print('output_softmax', output_softmax, len(output_softmax))
		for e in output_softmax:	##### add text, span, offsets, here
			pred_value, pred_idx = e.max(0)
			save_prediction.append([pred_value.item(), pred_idx.item()])

	df_save_prediction = pd.DataFrame(save_prediction,columns=['pred_value', 'pred_idx_label'])

	return df_save_prediction

	#
	# print('arrange_prediction:',arrange_prediction)
	# # write predicted list to tab-seperated file
	# # saving_path = "./other code/SemEval20-datasets/_prediction/"+'sameval20_'+model_name+"_CRF_oversampled_"+str(epochs)+"_"+task+".txt"
	# saving_path = "./_prediction_outputs/"+task+"/"+model_name+'_'+feature_type+'_'+oversmapled+'_epochs_'+str(epochs)+'_runtime-'+str(runtime)+'_'+additional_outName+".txt"
	# with open(saving_path, "w") as outfile:
	# 	for row in arrange_prediction:
	# 		row = "\t".join(row)+'\n'
	# 		outfile.write(row)
	# print('DONE writing output:',saving_path)

def eval_jointLoss_sentence_span(test_df,pred_df,train_limit,dict_labels):
	labels = list(dict_labels.values())
	true = test_df.labels[:train_limit]
	pred = pred_df.pred_idx_label[:train_limit]
	print('Evaluation\tprecision_recall_fscore_support:')
	precision, recall, fscore, support = precision_recall_fscore_support(true, pred, average='macro')
	print('macro:', precision, recall, fscore, support)
	precision, recall, fscore, support = precision_recall_fscore_support(true, pred, average='micro')
	print('micro:', precision, recall, fscore, support)
	precision, recall, fscore, support = precision_recall_fscore_support(true, pred, average='weighted')
	print('weighted:', precision, recall, fscore, support)
	precision, recall, fscore, support = precision_recall_fscore_support(true, pred, average=None, labels=labels)
	print('None:', precision, recall, fscore, support)

	print('\nEvaluation\tclassification_report:')
	print(classification_report(true, pred, labels=labels))



def load_data(workshop_name,tokenizer):
	if workshop_name == 'nlp4if':
		nlp4if_dict_labels = {'Appeal_to_Authority':1,
			'Appeal_to_fear-prejudice':2,
			'Bandwagon':3,
			'Reductio_ad_hitlerum': 3,
			'Black-and-White_Fallacy':4,
			'Causal_Oversimplification':5,
			'Doubt':6,
			'Exaggeration,Minimisation':7,
			'Flag-Waving':8,
			'Loaded_Language':9,
			'Name_Calling,Labeling':10,
			'Repetition':11,
			'Slogans':12,
			'Thought-terminating_Cliches': 13,
			'Straw_Men':14,
			'Red_Herring': 14,
			'Whataboutism':14,
			'Obfuscation,Intentional_Vagueness,Confusion':0,}
		# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		# train_dir = './other code/protechn_corpus_eval/train'
		# train_df = arrangeDataFrame(train_dir,tokenizer,nlp4if_dict_labels,task='nlp4if')
		# dev_dir = './other code/protechn_corpus_eval/dev' ## arrange dataset
		# dev_df = arrangeDataFrame(dev_dir,tokenizer,nlp4if_dict_labels)
		train_dir = './dataset/NLP4IF19/train'
		dev_dir = './dataset/NLP4IF19/dev'
		test_dir = './dataset/NLP4IF19/test'
		train_df = rearrangeData_for_sentence_to_span(train_dir, tokenizer, nlp4if_dict_labels, task='nlp4if')
		dev_df = rearrangeData_for_sentence_to_span(dev_dir, tokenizer, nlp4if_dict_labels, task='nlp4if')
		test_df = rearrangeData_for_sentence_to_span(test_dir, tokenizer, nlp4if_dict_labels, task='nlp4if')
		combine_train_dev_df = pd.concat([train_df,dev_df], ignore_index=True)
		test_span_template = '_'
		return nlp4if_dict_labels,combine_train_dev_df,test_df,test_dir, test_span_template
	elif workshop_name == 'semeval':
		semeval_dict_labels = {'Appeal_to_Authority':1,
			'Appeal_to_fear-prejudice':2,
			'Bandwagon,Reductio_ad_hitlerum':3,
			'Black-and-White_Fallacy':4,
			'Causal_Oversimplification':5,
			'Doubt':6,
			'Exaggeration,Minimisation':7,
			'Flag-Waving':8,
			'Loaded_Language':9,
			'Name_Calling,Labeling':10,
			'Repetition':11,
			'Slogans':12,
			'Thought-terminating_Cliches':13,
			'Whataboutism,Straw_Men,Red_Herring':14 }
		train_dir = './dataset/SemEval20/custom_train_dev'
		# train_df = arrangeDataFrame(train_dir,tokenizer,semeval_dict_labels,task='semeval')
		train_df = rearrangeData_for_sentence_to_span(train_dir,tokenizer,semeval_dict_labels,task='semeval')
		# print('train_df',train_df,train_df.shape) # shape (22519,3)
		# exit()

		test_dir = './dataset/SemEval20/test-articles'
		test_df = pd.DataFrame({'A' : []}) # retunn empty dataFrame since no test_labels
		test_span_template = './dataset/SemEval20/test-task-tc-template.out'
		return semeval_dict_labels,train_df,test_df,test_dir, test_span_template

def rearrangeData_for_sentence_to_span(path_articles,tokenizer,dict_labels,task):
	# read dir
	path_articles = path_articles+'/'
	file_type = ''
	if task == 'nlp4if': file_type = '.labels.tsv'
	elif task == 'semeval': file_type = '.task-flc-tc.labels'
	files = getFiles_in_folder(path_articles)#[:5]
	article_names, span_names = [],[]

	for file in files:
		if '.txt' in file: article_names.append(file)
		elif file_type in file: span_names.append(file)
	saved_list = []

	# get file names - each file
	for idx,article in enumerate(article_names):
		article_name = article.split('.')[0]
		# tmp_propaganda_text, tmp_none_propaganda_text = [], []

		#find spans on each article
		with open(path_articles+article_name+file_type) as s:
			propaganda_spans = s.readlines()
			text = open(path_articles+article_name+'.txt','r').read()

			# sentences = text.split('\n')
			sentence_spans = list(string_span_tokenize(text,'\n'))
			# print('=====each file:',sentences,'\n',sentence_spans)
			collect_propaganda_spans, collect_duplicated_propaganda_spans, all_sentences_span = [], [], []

			for i,sentence_span in enumerate(sentence_spans):
				token_start = sentence_spans[i][0]
				token_end = sentence_spans[i][1]
				# print('each sentence_span: (',token_start,token_end,')')
				all_sentence_span_range = range(token_start,token_end)
				# all_sentences_span.all_sentence_span_range
				# print('====all_sentence_span_range',all_sentence_span_range)
				# exit()
				marked_sentence = 0

				# ONLY READ LABELED_FRAGMENT - convert spans to num_label, and collect spans
				for span in propaganda_spans:
					span = span.split('\t')
					article_id, label, span_begin, span_end = span[0],span[1],int(span[2]),int(span[3])
					label = dict_labels[label] 	# make dict to match labels

					# if propaganda_span IN current sentence_span : write dict_label
					if span_begin in all_sentence_span_range and span_end in all_sentence_span_range:  
						# collect all spans in numeric - compare later
						collect_duplicated_propaganda_spans.append([article_id, label, span_begin, span_end,(token_start,token_end)])
						collect_propaganda_spans.append((token_start,token_end))
						# print('test',article_id, label, span_begin, span_end,text[span_begin:span_end],text[token_start:token_end])
						propaganda_text = text[span_begin:span_end]
						full_sentence = text[token_start:token_end]

						length_full_sentence = token_end-token_start
						# print('\nspan_begin:span_end:',span_begin,span_end)
						# print('token_start:token_end:',token_start,token_end) # the offsets of the whole sentence
						reset_token_start = 0
						reset_token_end = length_full_sentence
						# print('reset_token_start,reset_token_end:',reset_token_start,reset_token_end)
						distance_from_startidx_to_startpanidx = abs(span_begin-token_start)
						reset_span_startOffset = distance_from_startidx_to_startpanidx
						reset_span_endOffset = distance_from_startidx_to_startpanidx+(span_end-span_begin)
						# print('reset_span_startOffset,reset_span_endOffset:', reset_span_startOffset,reset_span_endOffset)


						saved_list.append([article_name,full_sentence,label,propaganda_text,reset_span_startOffset,reset_span_endOffset])
						marked_sentence = 1
				if marked_sentence == 0:
					full_sentence = text[token_start:token_end]
					saved_list.append([article_name,full_sentence,0,'',0,0])
					marked_sentence = 0
	df = pd.DataFrame(saved_list,columns=['article_name','text','labels','propaganda_text','start_index','end_index'])
	return df 


def main():
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	workshop_name = 'semeval'
	dict_labels, semeval_train_df, semeval_test_df, semeval_test_dir, semeval_path_test_span_template = load_data(workshop_name, tokenizer)
	workshop_name = 'nlp4if'
	_, nlp4if_train_df, test_df, nlp4if_test_dir, nlp4if_path_test_span_template = load_data(workshop_name, tokenizer)
	#
	# train_df = pd.concat([nlp4if_train_df, semeval_train_df], ignore_index=True)

	path_train_df = './fine_tuned_model_sentence_span/pickle_train_df.pickle'
	# pickle_train_df = open(path_train_df, 'wb')
	# pickle.dump(train_df, pickle_train_df)
	# pickle_train_df.close()
	train_df = pickle.load(open(path_train_df, 'rb'))

	path_test_df = './fine_tuned_model_sentence_span/pickle_test_df.pickle'
	# pickle_test_df = open(path_test_df, 'wb')
	# pickle.dump(test_df, pickle_test_df)
	# pickle_test_df.close()
	test_df = pickle.load(open(path_test_df, 'rb'))

	# print('train_df_from_pickle:',train_df)
	# print('test_df_from_pickle:', test_df,test_df.shape)
	# print('columns',train_df.columns)
	# exit()



	############## MOCK JSON OUTPUT
	# train_df = train_df[train_df.article_name == 'article701225819'][:100]
	# train_df = train_df.rename(columns={"propaganda_text": "span","labels":"label"})

	# print('mock train_df',train_df,train_df.columns)
	# train_df = train_df.drop(columns=['article_name'])

	# train_df = train_df.set_index(['text'])

	# print('train_df',train_df,train_df.columns)

	# result = train_df.to_json(orient="table")
	# parsed = json.loads(result)
	# # json.dumps(parsed, indent=4)
	# print('parsed', parsed)
	# exit()
	#
	# mock_outputdict = {}
	# count_repeated_span = 0
	# for article_name, text, label, span, start_index, end_index in zip(train_df.article_name,train_df.text,train_df.label,train_df.span,train_df.start_index,train_df.end_index):
	#
	# 	# # ========= version 2
	# 	# # mock_outputdict['article_name'] = {}
	# 	# mock_outputdict[article_name] = article_name
	# 	# # mock_outputdict['article_name']['text'] = {}
	# 	# # mock_outputdict[article_name]['text'] = {} # check here if 'text' exsits
	# 	# # mock_outputdict[article_name]['text'] = {}
	# 	# mock_outputdict[article_name][text] = text
	# 	# print('mock_outputdict',mock_outputdict)
	# 	# exit()
	# #
	# #
	# 	#========= version 1
	# 	# mock_outputdict[article_name] = {}
	#
	# 	# try:
	# 	# 	if mock_outputdict[article_name][text]:
	# 	# 		print('check if valid:',mock_outputdict[article_name][text])
	# 	# 	# exit()
	# 	# 	# mock_outputdict[article_name][text] = {}
	# 	# except KeyError:
	# 	# 	mock_outputdict[article_name] = {}
	# 	# 	mock_outputdict[article_name][text] = {}
	#
	# 	# mock_outputdict[article_name]['text'] = text
	#
	# 	# mock_outputdict[article_name][text]['span'] = {}
	# 	# mock_outputdict[article_name][text]['label'] = {}
	# 	# mock_outputdict[article_name][text]['start_char'] = {}
	# 	# mock_outputdict[article_name][text]['end_char'] = {}
	# 	# mock_outputdict[article_name][text]['probability'] = {}
	#
	# 	dict_addElements = {}
	#
	# 	if label != 0:
	# 		dict_addElements = {'span':span,
	# 							'label':label,
	# 							'start_char':start_index,
	# 							'end_index':end_index,
	# 							'probability':round(random.uniform(0.3, 1.0),2)}
	# 		# exit()
	# 		# print('\ntext:', text)
	# 		# print('span:', span)
	# 		# print('original_index:',start_index, end_index)  # -- based on original
	# 		# print('----text checking:',text[start_index:end_index])
	# 		# mock_outputdict[article_name][text]['span'] = span
	# 		# mock_outputdict[article_name][text]['label'] = label
	# 		# mock_outputdict[article_name][text]['start_char'] = start_index
	# 		# mock_outputdict[article_name][text]['end_char'] = end_index
	# 		# mock_outputdict[article_name][text]['probability'] = round(random.uniform(0.3, 1.0),2)
	# 		# print('mock_outputdict', mock_outputdict)
	# 		# print('\n')
	# 		# # exit()
	# 	else:
	# 		# mock_outputdict[article_name][text]['span'] = ''
	# 		# mock_outputdict[article_name][text]['label'] = 0
	# 		# mock_outputdict[article_name][text]['start_char'] = None
	# 		# mock_outputdict[article_name][text]['end_char'] = None
	# 		# mock_outputdict[article_name][text]['probability'] = None
	# 		dict_addElements = {'span':'',
	# 							'label':0,
	# 							'start_char':0,
	# 							'end_index':0,
	# 							'probability':0}
	# 	print('dict_addElements',dict_addElements)
	#
	# 	if article_name in mock_outputdict:
	# 		print('in 1----')
	# 		if text == "": pass
	# 		elif text in mock_outputdict[article_name]:
	# 			# mock_outputdict[article_name][text].update(dict_addElements) ## not update(), it will just replace the old values
	# 			count_repeated_span += 1
	# 			print('======== check count_repeat',count_repeated_span)
	# 			mock_outputdict[article_name][text]['span_'+str(count_repeated_span)] = dict_addElements
	# 			print('stop 1',mock_outputdict)
	# 			# exit()
	# 		else:
	# 			count_repeated_span = 0
	# 			mock_outputdict[article_name][text] = {}
	# 			# mock_outputdict[article_name][text] = dict_addElements
	# 			mock_outputdict[article_name][text]['span_' + str(count_repeated_span)] = dict_addElements
	# 			# print('mock_outputdict in 2',mock_outputdict)
	# 			print('stop 2',mock_outputdict)
	# 			# exit()
	# 	else:
	# 		mock_outputdict[article_name] = {}
	# 		mock_outputdict[article_name][text] = {}
	# 		# mock_outputdict[article_name][text].update(dict_addElements)
	# 		mock_outputdict[article_name][text]['span_' + str(count_repeated_span)] = dict_addElements
	# 		print('stop 3',mock_outputdict)
	#
	#
	# # print('final mock_outputdict', mock_outputdict)
	# # exit()
	#
	# # mock_outputdict[article_name][text] = dict_addElements
	# # print(mock_outputdict)
	# # exit()
	#
	# 	# elements needed
	# 	# article_name
	# 		# text
	# 			# span
	# 			# label
	# 			# start_char
	# 			# end_char
	# 			# probability
	#
	# json_object = json.dumps(mock_outputdict, indent=4)
	# print(json_object)
	# exit()
	#
	# # mock_result = train_df.to_json(orient="records") # records
	# # print('mock_result',mock_result)
	# parsed = json.loads(mock_outputdict)
	# print('parsed',parsed)
	# exit()
	# with open('mock_propagandaOutput_one_document_v2.json', 'w') as json_file:
	# 	json.dump(parsed, json_file)
	# exit()


	##############

	# ################ MODEL 4 SENTENCE-to-SPAN
	# args_dict = dict(
	#     text_list=['',''],
	#     persuasion_speechStyle_=True,
	#     persuasion_lexicalComplexity_=True,
	# 	persuasion_concreteness_=True,
	# 	persuasion_subjectivity_=True,
	#
	# 	sentiment_SentiWordnet_=True,
	# 	sentiment_warriner_=True,
	# 	sentiment_depechemood_=True,
	# 	sentiment_connotation_=True,
	# 	sentiment_politeness_=True,
	#
	# 	messageSimplicity_imageability_=True,
	# 	messageSimplicity_lexicalLength_=True,
	# 	messageSimplicity_lexicalEncoding_=True,
	# 	messageSimplicity_pronouns_=True,
	# )
	# feature_type, additional_outName = 'semantic', ''
	# #################
	#
	# args_dict = dict(
	#     text_list=['',''],
	# 	argumentDectection_=True,
	# 	# argumentationComponents_claim_premise_=True,
	# )
	# feature_type,additional_outName = 'argumentation', ''
	# # ##############
	#
	# feature_type = 'no_feature_type'
	# additional_outName = ''	#'_2ndLossFunction_'
	# completeFeatures_vectors,args_dict = None,None
	# #################

	args_dict = dict(
	    text_list=['',''],
	    persuasion_speechStyle_=True,
	    persuasion_lexicalComplexity_=True,
		persuasion_concreteness_=True,
		persuasion_subjectivity_=True,

		sentiment_SentiWordnet_=True,
		sentiment_warriner_=True,
		sentiment_depechemood_=True,
		sentiment_connotation_=True,
		sentiment_politeness_=True,

		messageSimplicity_imageability_=True,
		messageSimplicity_lexicalLength_=True,
		messageSimplicity_lexicalEncoding_=True,
		messageSimplicity_pronouns_=True,

		argumentDectection_=True,
	)
	feature_type,additional_outName = 'semantic_and_argumentation', ''

	# model_name = 'BERT'
	model_name = 'RoBERTa' # by default
	MAX_LEN = 128
	epochs = 10
	train_limit = 10009999999
	# test_article_limit = 199999999
	runtimes=1
	oversmapled = ''
	num_batch = 8 ## originally 16
	# test_dir = nlp4if_test_dir
	# path_test_span_template = _
	completeFeatures_vectors_train, completeFeatures_vectors_test = None,None

	###############
	if feature_type != 'no_feature_type':
		print('start extracting features:',feature_type)
		text_inputs = [str(x) for x in train_df.text[:train_limit]]
		args_dict.update({'text_list': text_inputs})

		if feature_type == 'semantic': path_save_pickle_vectors_train = './fine_tuned_model_sentence_span/semantic_completeFeatures_vectors_combine_train_dev.pickle'
		elif feature_type == 'argumentation': path_save_pickle_vectors_train = './fine_tuned_model_sentence_span/argumentation_completeFeatures_vectors_combine_train_dev.pickle'
		elif feature_type == 'semantic_and_argumentation': path_save_pickle_vectors_train = './fine_tuned_model_sentence_span/semantic_and_argumentation_completeFeatures_vectors_combine_train_dev.pickle'
		# # ---- save files
		# completeFeatures_vectors = get_propagandaFeatures(**args_dict)
		# pickle_vectors_train = open(path_save_pickle_vectors_train, 'wb')
		# pickle.dump(completeFeatures_vectors, pickle_vectors_train)
		# pickle_vectors_train.close()
		##---- load file
		read_pickle = open(path_save_pickle_vectors_train, 'rb')
		completeFeatures_vectors_train = pickle.load(read_pickle)
		# print('done extracting extraFeatures of train-set',completeFeatures_vectors,'\nLen:',len(completeFeatures_vectors),'\nAt:',path_save_pickle_vectors_train)

		##### read test here!
		if feature_type == 'semantic':
			path_save_pickle_vectors_test = './fine_tuned_model_sentence_span/semantic_completeFeatures_vectors_combine_test.pickle'
		elif feature_type == 'argumentation':
			path_save_pickle_vectors_test = './fine_tuned_model_sentence_span/argumentation_completeFeatures_vectors_combine_test.pickle'
		elif feature_type == 'semantic_and_argumentation':
			path_save_pickle_vectors_test = './fine_tuned_model_sentence_span/semantic_and_argumentation_completeFeatures_vectors_test.pickle'
		# # ---- save files
		# completeFeatures_vectors = get_propagandaFeatures(**args_dict)
		# pickle_vectors_train = open(path_save_pickle_vectors_test, 'wb')
		# pickle.dump(completeFeatures_vectors, pickle_vectors_train)
		# pickle_vectors_train.close()
		##---- load file
		read_pickle = open(path_save_pickle_vectors_test, 'rb')
		completeFeatures_vectors_test = pickle.load(read_pickle)
		# print('done extracting extraFeatures of test-set',completeFeatures_vectors,'\nLen:',len(completeFeatures_vectors),'\nAt:',path_save_pickle_vectors_train)
		# exit()


	# print('====feature_type:',feature_type)
	# print('train_df shape:',train_df[['labels', 'propaganda_text']][:10],train_df.shape) #22519
	# print('completeFeatures_vectors',completeFeatures_vectors[0],type(completeFeatures_vectors),'\nlen per each element:',len(completeFeatures_vectors[0]))
	# exit()	
	###############

	for runtime in range(runtimes):
		print('\nRuntime at:',runtime+1)
		## completeFeatures_vectors is just to TEST for now
		# completeFeatures_vectors = completeFeatures_vectors_test[:train_limit]
		print('======training started:')
		train_jointLoss_sentence_span(train_df,completeFeatures_vectors_train,model_name,train_limit,feature_type,num_batch=num_batch,epochs=epochs)
		pred_df = predict_jointLoss_sentence_span(test_df,completeFeatures_vectors_test,train_limit,dict_labels,epochs,runtime,oversmapled,args_dict,feature_type,additional_outName,model_name=model_name)
		eval_jointLoss_sentence_span(test_df,pred_df,train_limit,dict_labels)

		# proposed_architecture_sentence_to_span(train_df,completeFeatures_vectors,model_name,train_limit,workshop_name,num_batch=num_batch,epochs=epochs) #completeFeatures_vectors
		# print('=========== DONE TRAINING\n\n')
		# predict_sentence_to_span(test_dir,test_article_limit,dict_labels,path_test_span_template,epochs,runtime,oversmapled,args_dict,feature_type,additional_outName,task=workshop_name,model_name=model_name)
		# ##### predict_finetune_BertForTokenClassification_CRF(test_df,train_limit,dict_labels,model_name=model_name,task=task,num_batch=16,MAX_LEN=MAX_LEN)
		# # predict_finetune_BertForTokenClassification_CRF_withspan(test_dir,test_article_limit,dict_labels,runtime,oversmapled,model_name=model_name,task=workshop_name,num_batch=16,MAX_LEN=MAX_LEN,epochs=epochs)
		
		# predict_finetune_BertForTokenClassification_CRF_withspan_semeval(test_dir,test_article_limit,dict_labels,path_test_span_template,epochs,runtime,oversmapled,task=workshop_name,model_name=model_name)
		print("epochs:",epochs,"MAX_LEN:",MAX_LEN,model_name,workshop_name)






	# ideas for ENSAMBLE
	# # train_ensamble_sentencelevel_NER(combine_train_dev_df,train_limit,completeFeatures_vectors_train)
	# pred_ensamble_sentencelevel_NER(test_df,train_limit,completeFeatures_vectors_test,task='nlp4if',num_batch=16)
main()
#
# text_list = ['South Florida Muslim Leader Sofian Zakkout\u2019s David Duke Day',
# 				 'David Duke, the white supremacist icon and former Grand Wizard of the Ku Klux Klan, has been denounced by many as a hatemonger, and rightfully so.',
# 				 'However, one individual who represents the Muslim community of South Florida, Sofian Zakkout, is enamored with Duke and has been promoting Duke\u2019s bigoted work for many years.',
# 				 'At this same rally, a white nationalist plowed his car into a group of people who were protesting the rally, killing one.',
# 				 'David Duke Exposes the Real Racist Jewish Supremacists Who Orchestrate the Destruction of European Mankind\u2026\u2019 Above the posting, Zakkout wrote of Duke, \u201cI respect him for his honesty!\u201d',
# 				 'In October 2015, Zakkout posted to Facebook a Duke video, within which Duke makes the wild claim that there has been a \u201ccomplete takeover of American foreign policy and\u2026 American politics by Jewish extremists.\u201d Above the video on Facebook, Zakkout praised Duke, exclaiming \u201cDavid Duke, a man to believe in!\u201d',
# 				 'On the video, Farrakhan repeatedly refers to Jews as \u201cSatan.\u201d He states to his audience: \u201cReally, they\u2019re not Jews.']
# mock_outputdict = {}
# mock_outputdict['article701225819'] = {}
#
# dict_addElements = {'span': '',
# 					'label': 0,
# 					'start_char': 0,
# 					'end_index': 0,
# 					'probability':0}
# mock_outputdict['article701225819'][text_list[0]] = {}
# mock_outputdict['article701225819'][text_list[0]]['span_1'] = {}
# mock_outputdict['article701225819'][text_list[0]]['span_1'] = dict_addElements
#
# mock_outputdict['article701225819'][text_list[1]] = {}
# dict_addElements = {'span': 'hatemonger',
# 					'label': 10,
# 					'start_char': 116,
# 					'end_index': 126,
# 					'probability':0.63}
# mock_outputdict['article701225819'][text_list[1]]['span_1'] = {}
# mock_outputdict['article701225819'][text_list[1]]['span_1'] = dict_addElements
#
# dict_addElements = {'span': 'white supremacist icon',
# 					'label': 10,
# 					'start_char': 16,
# 					'end_index': 38,
# 					'probability':0.48}
# mock_outputdict['article701225819'][text_list[1]]['span_2'] = {}
# mock_outputdict['article701225819'][text_list[1]]['span_2'] = dict_addElements
#
# dict_addElements = {'span': 'Ku Klux Klan',
# 					'label': 2,
# 					'start_char': 70,
# 					'end_index': 82,
# 					'probability':0.52}
# mock_outputdict['article701225819'][text_list[1]]['span_3'] = {}
# mock_outputdict['article701225819'][text_list[1]]['span_3'] = dict_addElements
#
# dict_addElements = {'span': 'Grand Wizard of the Ku Klux Klan',
# 					'label': 10,
# 					'start_char': 50,
# 					'end_index': 82,
# 					'probability':0.43}
# mock_outputdict['article701225819'][text_list[1]]['span_4'] = {}
# mock_outputdict['article701225819'][text_list[1]]['span_4'] = dict_addElements
#
# mock_outputdict['article701225819'][text_list[2]] = {}
# dict_addElements = {'span': 'enamored',
# 					'label': 9,
# 					'start_char': 97,
# 					'end_index': 105,
# 					'probability': 0.53}
# mock_outputdict['article701225819'][text_list[2]]['span_1'] = {}
# mock_outputdict['article701225819'][text_list[2]]['span_1'] = dict_addElements
#
# mock_outputdict['article701225819'][text_list[3]] = {}
# dict_addElements = {'span': 'white nationalist plowed his car into a group of people who were protesting the rally, killing one',
# 					'label': 2,
# 					'start_char': 22,
# 					'end_index': 120,
# 					'probability': 0.35}
# mock_outputdict['article701225819'][text_list[3]]['span_1'] = {}
# mock_outputdict['article701225819'][text_list[3]]['span_1'] = dict_addElements
#
# mock_outputdict['article701225819'][text_list[4]] = {}
# dict_addElements = {
# 	'span': 'Racist Jewish Supremacists',
# 	'label': 10,
# 	'start_char': 28,
# 	'end_index': 54,
# 	'probability': 0.70}
# mock_outputdict['article701225819'][text_list[4]]['span_1'] = {}
# mock_outputdict['article701225819'][text_list[4]]['span_1'] = dict_addElements
#
# dict_addElements = {
# 	'span': 'Destruction',
# 	'label': 9,
# 	'start_char': 75,
# 	'end_index': 86,
# 	'probability': 0.67}
# mock_outputdict['article701225819'][text_list[4]]['span_2'] = {}
# mock_outputdict['article701225819'][text_list[4]]['span_2'] = dict_addElements
#
# mock_outputdict['article701225819'][text_list[5]] = {}
# dict_addElements = {
# 	'span': 'wild',
# 	'label': 9,
# 	'start_char': 86,
# 	'end_index': 90,
# 	'probability': 0.50}
# mock_outputdict['article701225819'][text_list[5]]['span_1'] = {}
# mock_outputdict['article701225819'][text_list[5]]['span_1'] = dict_addElements
#
# dict_addElements = {
# 	'span': 'Jewish extremists',
# 	'label': 10,
# 	'start_char': 191,
# 	'end_index': 208,
# 	'probability': 0.50}
# mock_outputdict['article701225819'][text_list[5]]['span_2'] = {}
# mock_outputdict['article701225819'][text_list[5]]['span_2'] = dict_addElements
#
# dict_addElements = {
# 	'span': 'a man to believe in!',
# 	'label': 12,
# 	'start_char': 286,
# 	'end_index': 306,
# 	'probability': 0.70}
# mock_outputdict['article701225819'][text_list[5]]['span_3'] = {}
# mock_outputdict['article701225819'][text_list[5]]['span_3'] = dict_addElements
#
# mock_outputdict['article701225819'][text_list[6]] = {}
# dict_addElements = {
# 	'span': 'Satan',
# 	'label': 9,
# 	'start_char': 54,
# 	'end_index': 59,
# 	'probability': 1.0}
# mock_outputdict['article701225819'][text_list[6]]['span_1'] = {}
# mock_outputdict['article701225819'][text_list[6]]['span_1'] = dict_addElements
# dict_addElements = {
# 	'span': 'Satan',
# 	'label': 10,
# 	'start_char': 54,
# 	'end_index': 59,
# 	'probability': 0.73}
# mock_outputdict['article701225819'][text_list[6]]['span_2'] = {}
# mock_outputdict['article701225819'][text_list[6]]['span_2'] = dict_addElements
# dict_addElements = {
# 	'span': 'Satan',
# 	'label': 2,
# 	'start_char': 54,
# 	'end_index': 59,
# 	'probability': 0.49}
# mock_outputdict['article701225819'][text_list[6]]['span_3'] = {}
# mock_outputdict['article701225819'][text_list[6]]['span_3'] = dict_addElements
# dict_addElements = {
# 	'span': 'Satan',
# 	'label': 11,
# 	'start_char': 54,
# 	'end_index': 59,
# 	'probability': 0.65}
# mock_outputdict['article701225819'][text_list[6]]['span_4'] = {}
# mock_outputdict['article701225819'][text_list[6]]['span_4'] = dict_addElements
#
# json_object = json.dumps(mock_outputdict, indent=4)
# parsed = json.loads(json_object)
# print(json_object)
# with open('mock_propagandaOutput_one_document_v2.json', 'w') as json_file:
# 	# json.dump(json_object, json_file)
# 	json.dump(parsed, json_file)
# 	# print('done')
# exit()





