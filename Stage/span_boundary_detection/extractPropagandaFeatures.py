'''
Extract features listed in prodaganda detection techiques.
'''
import re, os, nltk, json, string, pickle, math, torch
import pandas as pd
# nltk.download('wordnet')
# nltk.download('sentiwordnet')
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np

# from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from custom_transformers.src.transformers.models.bert import BertTokenizer, BertModel, BertForSequenceClassification
import torch.nn.functional as F
# from custom_transformers.src.transformers.models.bert import cBertForSequenceClassification_withOutput

'''
PERSUASION
'''
def persuasion_speechStyle(sentence,read_dict=None,ps=None):
	if read_dict == None and ps == None:
		ps = PorterStemmer()
		sentence = sentence.lower()
		path_speechStyle = './Lexicon/inquirerbasic.csv'
		with open(path_speechStyle,'r') as f:
			read_df = pd.read_csv(f, delimiter=',',index_col=False)
			read_df['Entry'] = [row.lower().split('#')[0] for row in read_df.Entry]
			read_df.set_index('Entry', inplace=True)
			# print(read_df.shape)
			# read_df.index.duplicated(keep='first')
			read_df = read_df[~read_df.index.duplicated(keep='first')]
		read_dict = read_df.transpose().to_dict()
		serial_speechStyle = pickle.dump(read_dict,open('./__dumps_objs/persuasion_speechStyle.dump','wb'))
	# print(read_df)
	# exit()

	# create empty dict with keys
	keys = ['Positiv','Negativ','Pstv','Affil','Ngtv','Hostile','Strong','Power','Weak',
	'Submit','Active','Passive','Pleasur','Pain','Feel','Arousal','EMOT','Virtue','Vice',
	'Ovrst','Undrst','Academ','Doctrin','Econ@','Exch','ECON','Exprsv','Legal','Milit',
	'Polit@','POLIT','Relig','Role','COLL','Work','Ritual','SocRel','Race','Kin@','MALE',
	'Female','Nonadlt','HU','ANI','PLACE','Social','Region','Route','Aquatic','Land','Sky',
	'Object','Tool','Food','Vehicle','BldgPt','ComnObj','NatObj','BodyPt','ComForm','COM',
	'Say','Need','Goal','Try','Means','Persist','Complet','Fail','NatrPro','Begin','Vary',
	'Increas','Decreas','Finish','Stay','Rise','Exert','Fetch','Travel','Fall','Think','Know',
	'Causal','Ought','Perceiv','Compare','Eval@','EVAL','Solve','Abs@','ABS','Quality','Quan',
	'NUMB','ORD','CARD','FREQ','DIST','Time@','TIME','Space','POS','DIM','Rel','COLOR','Self',
	'Our','You','Name','Yes','No','Negate','Intrj','IAV','DAV','SV','IPadj','IndAdj','PowGain',
	'PowLoss','PowEnds','PowAren','PowCon','PowCoop','PowAuPt','PowPt','PowDoct','PowAuth',
	'PowOth','PowTot','RcEthic','RcRelig','RcGain','RcLoss','RcEnds','RcTot','RspGain',
	'RspLoss','RspOth','RspTot','AffGain','AffLoss','AffPt','AffOth','AffTot','WltPt',
	'WltTran','WltOth','WltTot','WlbGain','WlbLoss','WlbPhys','WlbPsyc','WlbPt','WlbTot',
	'EnlGain','EnlLoss','EnlEnds','EnlPt','EnlOth','EnlTot','SklAsth','SklPt','SklOth','SklTot',
	'TrnGain','TrnLoss','TranLw','MeansLw','EndsLw','ArenaLw','PtLw','Nation','Anomie','NegAff',
	'PosAff','SureLw','If','NotLw','TimeSpc','FormLw']

	# count found values
	count_values = []
	for token in nltk.word_tokenize(sentence):
		token = ps.stem(token)
		try:
			# print('found,',token)
			count_values.extend(list(read_dict[token].values()))
		except KeyError:
			# print(token,': not found')
			pass

	# list_count_value = list(Counter(count_values))
	dict_counter = dict(Counter(count_values))

	# update only found values
	final_dict = {}
	for key in keys:
		try:
			update_value = dict_counter[key]
			final_dict[key] = update_value
		except KeyError:
			final_dict[key] = 0
	return final_dict

# persuasion_speechStyle("hello i feel great about the election that's coming yonder zero zone ABANDON ABANDON ABANDON ABANDON")

def persuasion_lexicalComplexity(sentence,tokenizer,model=None):
	'''
	Try to obtain sentence embedding layer from BERT transformer
	'''

	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	# device = "cpu"
	if model == None:
		model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True,) # Whether the model returns all hidden-states.
		model.eval() # Put the model in "evaluation" mode, meaning feed-forward operation.
		model = model.to(device)

	marked_text = "[CLS] " + sentence + " [SEP]"
	tokenized_text = tokenizer.tokenize(marked_text)
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
	segments_ids = [1] * len(tokenized_text)
	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])

	if device != 'cpu':
		tokens_tensor = tokens_tensor.to(device)
		segments_tensors = segments_tensors.to(device)

	with torch.no_grad():
		outputs = model(tokens_tensor, segments_tensors)
		hidden_states = outputs[2]
	## default BERT embedding 
	# token_embeddings = torch.stack(hidden_states, dim=0)
	# token_embeddings = torch.squeeze(token_embeddings, dim=1)
	# token_embeddings = token_embeddings.permute(1,0,2)
	# print(token_embeddings,token_embeddings.size())
	## `hidden_states` has shape [13 x 1 x 22 x 768]

	## average the second to last hiden layer of each token producing a single 768 length vector.
	## to get sentence embedding 
	# `token_vecs` is a tensor with shape [22 x 768]
	token_vecs = hidden_states[-2][0]
	# Calculate the average of all 22 token vectors.
	sentence_embedding = torch.mean(token_vecs, dim=0)
	# print(sentence_embedding,sentence_embedding.size())
	# sentence_embedding = sentence_embedding.FeaturesDict
	
	sentence_embedding = [float(x) for x in sentence_embedding]
	keys = []
	for i in range(0,len(sentence_embedding)): 
		keys.append('BERT_sentenceVector_'+str(i+1))
	dict_sentence_embedding = dict(zip(keys, sentence_embedding))

	return dict_sentence_embedding

# def persuasion_personalityTraits(sentence,read_agreeableness,read_conscientiousness,read_extraversion,read_neuroticism,read_openness):
# 	sentence = sentence.lower()
# 	# path_agreeableness = './Lexicon/personality_word_and_phrase_correlations-master/csv/A.top100.1to3grams.gender_age_controlled.rmatrix.csv'
# 	# path_conscientiousness = './Lexicon/personality_word_and_phrase_correlations-master/csv/C.top100.1to3grams.gender_age_controlled.rmatrix.csv'
# 	# path_extraversion = './Lexicon/personality_word_and_phrase_correlations-master/csv/E.top100.1to3grams.gender_age_controlled.rmatrix.csv'
# 	# path_neuroticism = './Lexicon/personality_word_and_phrase_correlations-master/csv/N.top100.1to3grams.gender_age_controlled.rmatrix.csv'
# 	# path_openness = './Lexicon/personality_word_and_phrase_correlations-master/csv/O.top100.1to3grams.gender_age_controlled.rmatrix.csv'
# 	# read_agreeableness = pd.read_csv(path_agreeableness,index_col=0).transpose().to_dict()
# 	# read_conscientiousness = pd.read_csv(path_conscientiousness,index_col=0).transpose().to_dict()
# 	# read_extraversion = pd.read_csv(path_extraversion,index_col=0).transpose().to_dict()
# 	# read_neuroticism = pd.read_csv(path_neuroticism,index_col=0).transpose().to_dict()
# 	# read_openness = pd.read_csv(path_openness,index_col=0).transpose().to_dict()

# 	# serial_read_agreeableness = pickle.dump(read_agreeableness,open('./__dumps_objs/serial_personalityTraits_agreeableness.dump','wb'))
# 	# serial_read_conscientiousness = pickle.dump(read_conscientiousness,open('./__dumps_objs/serial_personalityTraits_conscientiousness.dump','wb'))
# 	# serial_read_extraversion = pickle.dump(read_extraversion,open('./__dumps_objs/serial_personalityTraits_extraversion.dump','wb'))
# 	# serial_read_neuroticism = pickle.dump(read_neuroticism,open('./__dumps_objs/serial_personalityTraits_neuroticism.dump','wb'))
# 	# serial_read_openness = pickle.dump(read_openness,open('./__dumps_objs/serial_personalityTraits_openness.dump','wb'))


# 	agreeableness,conscientiousness,extraversion,neuroticism,openness=[],[],[],[],[]
# 	for arg in read_agreeableness.keys():
# 		if str(arg) in sentence:
# 			try:	token_found = read_agreeableness[arg]; agreeableness.append(token_found['agr'])
# 			except KeyError:	pass
# 	for arg in read_conscientiousness.keys():
# 		if str(arg) in sentence:
# 			try:	token_found = read_conscientiousness[arg]; conscientiousness.append(token_found['con'])
# 			except KeyError:	pass
# 	for arg in read_extraversion.keys():
# 		if str(arg) in sentence:
# 			try:	token_found = read_extraversion[arg]; extraversion.append(token_found['ext'])
# 			except KeyError:	pass
# 	for arg in read_neuroticism.keys():
# 		if str(arg) in sentence:
# 			try:	token_found = read_neuroticism[arg]; neuroticism.append(token_found['neu'])
# 			except KeyError:	pass
# 	for arg in read_openness.keys():
# 		if str(arg) in sentence:
# 			try:	token_found = read_openness[arg]; openness.append(token_found['ope'])
# 			except KeyError:	pass
# 	result_vectors = {}
# 	result_vectors['personalityTraits_agreeableness'] = sum(agreeableness)
# 	result_vectors['personalityTraits_conscientiousness'] = sum(conscientiousness)
# 	result_vectors['personalityTraits_extraversion'] = sum(extraversion)
# 	result_vectors['personalityTraits_neuroticism'] = sum(neuroticism)
# 	result_vectors['personalityTraits_openness'] = sum(openness)
# 	return result_vectors
	
# '''
# ARGUMENT
# '''
# def argumentMining_arguingType(sentence,read_arguingType):
# 	sentence = sentence.lower()
# 	# defult_path = './Lexicon/arglex_Somasundaran07/'
# 	# allFiles_in_defult_path = os.listdir(defult_path)
# 	# labels = []
# 	# for label in allFiles_in_defult_path: 
# 	# 	if '.tff' in label and '_' not in label: labels.append(label)

# 	# result_vectors = {}
# 	# # search regex patterns in each label
# 	# for label in labels:
# 	# 	colName = 'arguingType_'+label.replace('.tff','') # create key
# 	# 	# print(label)
# 	# 	full_path = defult_path+label #'assessments.tff' 
# 	# 	count_found = 0
# 	# 	# print(full_path)
# 	# 	with open(full_path,'r') as f:
# 	# 		full_regex = f.readlines()[1:] # list of patterns
# 	# 		# serial_read_agreeableness = pickle.dump(full_regex,open('./__dumps_objs/arguingType_'+label+'.dump','wb'))

# 	# 		# for each pattern in each type, count freqency
# 	# 		for regex in full_regex:
# 	# 			regex = regex.replace('\n','')
# 	# 			if re.search(regex,sentence):
# 	# 				count_found+=1
# 	# 	# append value to named key
# 	# 	result_vectors[colName] = count_found


	# ########
	# result_vectors = {}
	# col_names = ['arguingType_inconsistency','arguingType_difficulty','arguingType_rhetoricalquestion','arguingType_generalization',
	# 'arguingType_necessity','arguingType_structure','arguingType_inyourshoes','arguingType_wants','arguingType_authority',
	# 'arguingType_contrast','arguingType_causation','arguingType_possibility','arguingType_doubt','arguingType_assessments',
	# 'arguingType_conditionals','arguingType_priority','arguingType_emphasis']
	# for full_regex,col_name in zip(read_arguingType,col_names):
	# 	count_found = 0
	# 	for regex in full_regex:
	# 		regex = regex.replace('\n','')
	# 		if re.search(regex,sentence):
	# 			count_found+=1
	# 	result_vectors[col_name] = count_found

	# return result_vectors

def persuasion_concreteness(sentence,dict_lexicon):
	# defult_path = './Lexicon/Concreteness_ratings_Brysbaert_et_al_BRM.txt'
	# read_lexicon = pd.read_csv(defult_path, delimiter='\t')#.to_dict()
	# read_lexicon = read_lexicon.set_index('Word')
	# dict_lexicon = read_lexicon.transpose().to_dict()

	# serial_argumentMining_concreteness= pickle.dump(dict_lexicon,open('./__dumps_objs/serial_argumentMining_concreteness.dump','wb'))

	sentence = sentence.lower()
	count_score = []
	result_vectors = {}
	for token in nltk.word_tokenize(sentence):
		try:
			standadized_score = dict_lexicon[token]
			# print(token,standadized_score['Conc.M'])
			count_score.append(standadized_score['Conc.M'])
		except KeyError:
			pass
	result_vectors['concreteness'] = sum(count_score)
	return result_vectors

def persuasion_subjectivity(sentence,dict_subj):
	sentence = sentence.lower()
	# defult_path = './Lexicon/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
	# read_lexicon = open(defult_path,'r').readlines()
	# dict_subj = {}
	# for line in read_lexicon:
	# 	line = line.split()
	# 	word = line[2].split('=')[1]
	# 	subj_value = line[0].split('=')[1]
	# 	if subj_value == 'weaksubj': 
	# 		dict_subj[word] = 0
	# 	else: dict_subj[word] = 1

	# serial_argumentMining_subjectivity= pickle.dump(dict_subj,open('./__dumps_objs/argumentMining_subjectivity.dump','wb'))


	# count # of weaksubj and strongsubj
	sentence = sentence.lower()
	strongsubj,weaksubj = [],[]
	for token in nltk.word_tokenize(sentence):
		try:
			label = dict_subj[token]
			if label == 0: weaksubj.append(1)
			else: strongsubj.append(1)
		except KeyError:
			pass

	result_vectors = {}
	result_vectors['subjectivity_strongsubjectivity'] = len(strongsubj)
	result_vectors['subjectivity_weaksubjectivity'] = len(weaksubj)
	return result_vectors

'''
SENTIMENT
'''
def penn_to_wn(tag):
    """
    Convert vbetween the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
def get_sentiment(word,tag):
    """ returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """
    lemmatizer = WordNetLemmatizer()
    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())
    return (swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score())

def sentiment_SentiWordnet(sentence):
	sentence = sentence.lower()
	# ps = PorterStemmer()
	sentence = nltk.word_tokenize(sentence)
	pos_val = nltk.pos_tag(sentence)
	pos,neg,obj = [],[],[]
	for word,tag in pos_val:
		senti_val = get_sentiment(word,tag)
		if senti_val:
			pos.append(senti_val[0])
			neg.append(senti_val[1])
			obj.append(senti_val[2])
	result_vectors = {}
	result_vectors['sentiwordnet_pos'] = sum(list(pos))
	result_vectors['sentiwordnet_neg'] = sum(list(neg))
	result_vectors['sentiwordnet_obj'] = sum(list(obj))
	return result_vectors

def sentiment_warriner(sentence,dict_lexicon):
	sentence = sentence.lower()
	# defult_path = './Lexicon/Ratings_Warriner_et_al.csv'
	# read_lexicon = pd.read_csv(defult_path, delimiter=',',index_col=0)
	# read_lexicon = read_lexicon[['Word','V.Mean.Sum','A.Mean.Sum','D.Mean.Sum']].set_index('Word')
	# dict_lexicon = read_lexicon.transpose().to_dict()

	# serial_sentiment_warriner= pickle.dump(dict_lexicon,open('./__dumps_objs/sentiment_warriner.dump','wb'))

	V,A,D = [],[],[]
	result_vectors = {}
	for token in nltk.word_tokenize(sentence):
		try:
			standadized_score = dict_lexicon[token]
			V.append(standadized_score['V.Mean.Sum'])
			A.append(standadized_score['A.Mean.Sum'])
			D.append(standadized_score['D.Mean.Sum'])
			pass
		except KeyError:
			pass
	result_vectors['warriner_valence'] = sum(V)
	result_vectors['warriner_arousal'] = sum(A)
	result_vectors['warriner_dominance'] = sum(D)
	return result_vectors

def sentiment_depechemood(sentence,dict_lexicon):
	sentence = sentence.lower()
	# defult_path = './Lexicon/depecheMood_english_token_full.tsv'
	# read_lexicon = pd.read_csv(defult_path, delimiter='\t',index_col=0)
	# dict_lexicon = read_lexicon.transpose().to_dict()

	# serial_sentiment_depechemood= pickle.dump(dict_lexicon,open('./__dumps_objs/sentiment_depechemood.dump','wb'))
	
	AFRAID,AMUSED,ANGRY,ANNOYED,DONT_CARE,HAPPY,INSPIRED,SAD = [],[],[],[],[],[],[],[]
	result_vectors = {}
	for token in nltk.word_tokenize(sentence):
		try:
			standadized_score = dict_lexicon[token]
			AFRAID.append(standadized_score['AFRAID'])
			AMUSED.append(standadized_score['AMUSED'])
			ANGRY.append(standadized_score['ANGRY'])
			ANNOYED.append(standadized_score['ANNOYED'])
			DONT_CARE.append(standadized_score['DONT_CARE'])
			HAPPY.append(standadized_score['HAPPY'])
			INSPIRED.append(standadized_score['INSPIRED'])
			SAD.append(standadized_score['SAD'])
			pass
		except KeyError:
			pass
	result_vectors['depechemood_AFRAID'] = sum(AFRAID)
	result_vectors['depechemood_AMUSED'] = sum(AMUSED)
	result_vectors['depechemood_ANGRY'] = sum(ANGRY)
	result_vectors['depechemood_ANNOYED'] = sum(ANNOYED)
	result_vectors['depechemood_DONT_CARE'] = sum(DONT_CARE)
	result_vectors['depechemood_HAPPY'] = sum(HAPPY)
	result_vectors['depechemood_INSPIRED'] = sum(INSPIRED)
	result_vectors['depechemood_SAD'] = sum(SAD)
	return result_vectors

def pos_matching_connotation(tag):
	'''
	have to check how to match type of pos correctly
	'''
	adjective = ['JJ','JJR','JJS']
	noun = ['NN','NNS','FW']
	propernoun = ['NNP','NNPS','PRP']
	verb = ['MD','VB','VBD','VBG','VBN','VBP','VBZ','RB','RBR','RBS','RP']
	if tag in adjective: return 'adjective'
	elif tag in noun: return 'noun'
	elif tag in propernoun: return 'propernoun'
	elif tag in verb: return 'verb'
	else: return 'noun'

def sentiment_connotation(sentence,dict_lexicon):
	sentence = sentence.lower()
	# defult_path = './Lexicon/connotation_lexicon_a.0.1.csv'
	# lexiconlist = []
	# with open(defult_path,'r') as f:
	# 	for data in f.readlines():
	# 		line = data.replace('\n','').split('_')
	# 		word = line[0]
	# 		pos = line[1].split(',')[0]
	# 		sentiment = line[1].split(',')[1]
	# 		lexiconlist.append([word,pos,sentiment])
	# 		# dict_lexicon[word] = sentiment
	# df_lexicon = pd.DataFrame(lexiconlist,columns=['word','pos','sentiment']).set_index('word')
	# dict_lexicon = df_lexicon.transpose().to_dict()

	# serial_sentiment_connotation= pickle.dump(dict_lexicon,open('./__dumps_objs/sentiment_connotation.dump','wb'))

	pos_val = nltk.pos_tag(nltk.word_tokenize(sentence))
	pos,neg,neu = [],[],[]
	for pos_token in pos_val:
		word,tag = pos_token
		true_tag = pos_matching_connotation(tag)
		try:
			#### perform matching pos and true_tag
			senti_val = dict_lexicon[word]
			if senti_val['sentiment'] =='positive': pos.append(1)
			if senti_val['sentiment'] =='negative': neg.append(1)
			if senti_val['sentiment'] =='neutral': neu.append(1)
		except KeyError:
			# print('not found',word,true_tag)
			pass
	result_vectors = {}
	result_vectors['connotation_pos'] = sum(list(pos))
	result_vectors['connotation_neg'] = sum(list(neg))
	result_vectors['connotation_neu'] = sum(list(neu))
	return result_vectors

def sentiment_politeness(sentence,pos_words,neg_words):
	sentence = sentence.lower()
	# path_pos = './Lexicon/politeness-master/liu-positive-words.txt'
	# path_neg = './Lexicon/politeness-master/liu-negative-words.txt'
	# with open(path_pos,'r') as f:
	# 	pos_words = list(filter(None, f.read().split('\n')))
	# with open(path_neg,'r') as f:
	# 	neg_words = list(filter(None, f.read().split('\n')))

	# serial_sentiment_politeness_pos= pickle.dump(pos_words,open('./__dumps_objs/sentiment_politeness_pos.dump','wb'))
	# serial_sentiment_politeness_neg= pickle.dump(neg_words,open('./__dumps_objs/sentiment_connotation_neg.dump','wb'))

	count_pos, count_neg = [],[]
	for token in nltk.word_tokenize(sentence):
		if token in pos_words: count_pos.append(1)
		if token in neg_words: count_neg.append(1)
	# print(count_pos, count_neg)
	result_vectors = {}
	result_vectors['politeness_pos'] = sum(count_pos)
	result_vectors['politeness_neg'] = sum(count_neg)
	# print(result_vectors)
	return result_vectors

'''
SIMPLICITY OF MESSAGE
'''
def messageSimplicity_imageability(sentence,dict_imageability_lexicon):
	sentence = sentence.lower()
	# defult_path = './Lexicon/imageability_resource_abstract-concrete_labels.txt'
	# imageability_lexicon = []
	# with open(defult_path,'r') as f:
	# 	word_tuples = list(filter(None, f.read().split('\n')))
	# 	for line in word_tuples:
	# 		word,bi_label,tuple_scores = tuple(line.split('\t'))
	# 		scores = json.loads(tuple_scores)
	# 		abstract_score = scores['A']
	# 		concreteness_score = scores['C']
	# 		imageability_lexicon.append([word,bi_label,abstract_score,concreteness_score])		

	# df_imageability_lexicon = pd.DataFrame(imageability_lexicon,columns=['word','label','abstract_score','concreteness_score']).set_index('word')
	# dict_imageability_lexicon = df_imageability_lexicon.transpose().to_dict()
	
	# serial_messageSimplicity_imageability= pickle.dump(dict_imageability_lexicon,open('./__dumps_objs/messageSimplicity_imageability.dump','wb'))

	abstract,concreteness = [],[]
	for token in nltk.word_tokenize(sentence):
		try:
			token_found = dict_imageability_lexicon[token]
			# print(token,token_found['abstract_score'],token_found['concreteness_score'])
			abstract.append(token_found['abstract_score'])
			concreteness.append(token_found['concreteness_score'])
		except KeyError:
			pass
	result_vectors = {}
	result_vectors['imageability_abstract'] = sum(abstract)
	result_vectors['imageability_concreteness'] = sum(concreteness)
	return result_vectors

def messageSimplicity_lexicalLength(sentence):
	actual_charLength, word_length, punctuation_frequency, capital_case_frequency = 0,0,0,0

	actual_charLength = len(sentence)
	word_length = len(nltk.word_tokenize(sentence))
	for s in sentence:
		if s in string.punctuation: punctuation_frequency+=1 
	for s in sentence:
		if s.isupper(): capital_case_frequency+=1 
	# print(actual_charLength, word_length, punctuation_frequency, capital_case_frequency)
	result_vectors = {}
	result_vectors['lexicalLength_actual_charLength'] = actual_charLength
	result_vectors['lexicalLength_word_length'] = word_length
	result_vectors['lexicalLength_punctuation_frequency'] = punctuation_frequency
	result_vectors['lexicalLength_capital_case_frequency'] = capital_case_frequency
	return result_vectors

def messageSimplicity_lexicalEncoding(sentence):
	# print('Started at lexicalEncoding:',sentence)
	sentence = str(sentence).lower()
	dictChars = dict.fromkeys(string.ascii_lowercase, 0)
	dictChars['non-alphabet'] = 0

	counted_char = Counter(sentence)
	count_only_alphabet = {}
	count_non_alphabet = 0
	for key in counted_char.keys():
		if key not in string.ascii_lowercase: 
			# print('====',key,counted_char[key])
			count_non_alphabet+=counted_char[key]
		else: 
			count_only_alphabet[key] = counted_char[key]
	dictChars.update({'non-alphabet':count_non_alphabet})

	count_allchars = {**dictChars,**count_only_alphabet}
	return count_allchars

def messageSimplicity_pronouns(sentence):
	sentence = sentence.lower()
	defult_path = './Lexicon/list_of_pronouns.txt'
	with open(defult_path,'r') as f:
		pronoun_list = list(filter(None, f.read().split('\n')))

	# serial_messageSimplicity_pronouns= pickle.dump(pronoun_list,open('./__dumps_objs/messageSimplicity_pronouns.dump','wb'))

	tokenized_sent = nltk.word_tokenize(sentence)
	dict_countedPronouns = {i:tokenized_sent.count(i) for i in pronoun_list}
	return dict_countedPronouns

def argumentationComponents_claim_premise(sentence,tokenizer,model_claim_premise):
	### '0' = premise, '1' = claim
	# read BERT fine-tuned model as eval() to predict
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	# device = "cpu"
	model_claim_premise = model_claim_premise.to(device)

	model_claim_premise.eval()
	input_sentence = tokenizer(list([sentence]), return_tensors="pt",padding=True,truncation=True)
	outputs, bert_pooled_output = model_claim_premise(**input_sentence.to(device))
	logits = outputs[0]
	# predict by activation function
	# output_softmax = F.softmax(logits)
	output_softmax = torch.nn.Softmax(logits).dim.to("cpu")
	# print('output_softmax argumentationComponents_claim_premise:',output_softmax)

	threshold = 0.5
	preds = np.where(output_softmax[:, 1] > threshold, 1, 0)
	# print('preds argumentationComponents_claim_premise:',preds)
	# exit()
	bert_pooled_output = bert_pooled_output.to("cpu")


	# if claim [1], then all premise [0] vectors = [0,..,0] -- vice versa
	claim_premise = {}
	if preds == 1: 	# '1' = claim
		claim_premise['claim'] = list(bert_pooled_output.detach().numpy()[0]) # original
		claim_premise['premise'] = list(np.zeros(bert_pooled_output.size()[1])) # original
		# claim_premise['claim'] = [1] # original
		# claim_premise['premise'] = [0] # original
	else: 		# '0' = premise
		claim_premise['claim'] = list(np.zeros(bert_pooled_output.size()[1])) # original
		claim_premise['premise'] = list(bert_pooled_output.detach().numpy()[0]) # original
		# claim_premise['claim'] = [0] # original
		# claim_premise['premise'] = [1] # original
	# print('claim_premise:',claim_premise)
	return claim_premise

def argumentDectection(sentence,tokenizer,model_argumentDectection):
	### '0' = non-argumetative, '1' = argumetative
	# read BERT fine-tuned model as eval() to predict
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	# device = "cpu"
	model_argumentDectection = model_argumentDectection.to(device)

	model_argumentDectection.eval()
	input_sentence = tokenizer(list([sentence]), return_tensors="pt",padding=True,truncation=True)
	outputs, bert_pooled_output = model_argumentDectection(**input_sentence.to(device))
	logits = outputs[0]
	# predict by activation function
	# output_softmax = F.softmax(logits)
	output_softmax = torch.nn.Softmax(logits).dim.to("cpu")
	# print('>>>>output_softmax argumentDectection:', output_softmax)
	threshold = 0.5
	preds = np.where(output_softmax[:, 1] > threshold, 1, 0)
	# print('preds argumentDectection: ',preds)
	# exit()
	bert_pooled_output = bert_pooled_output.to("cpu")


	# if claim [1], then all premise [0] vectors = [0,..,0] -- vice versa
	argumentDectection = {}
	if preds == 1: 	# '1' = argumetative
		argumentDectection['argumentative'] = list(bert_pooled_output.detach().numpy()[0]) # original
		argumentDectection['non-argumentative'] = list(np.zeros(bert_pooled_output.size()[1])) # original
		# argumentDectection['argumentative'] = [1] # original
		# argumentDectection['non-argumentative'] = [0] # original
	else: 		# '0' = non-argumetative
		argumentDectection['argumentative'] = list(np.zeros(bert_pooled_output.size()[1])) # original
		argumentDectection['non-argumentative'] = list(bert_pooled_output.detach().numpy()[0]) # original
		# argumentDectection['argumentative'] = [0] # original
		# argumentDectection['non-argumentative'] = [1] # original
	# print('argumentDectection:',argumentDectection)
	return argumentDectection


def argumentDectection_binary(sentence,tokenizer,model_argumentDectection):
	### '0' = non-argumetative, '1' = argumetative
	# read BERT fine-tuned model as eval() to predict
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	# device = "cpu"
	model_argumentDectection = model_argumentDectection.to(device)
	model_argumentDectection.eval()
	input_sentence = tokenizer(list([sentence]), return_tensors="pt",padding=True,truncation=True)
	outputs, bert_pooled_output = model_argumentDectection(**input_sentence.to(device))
	logits = outputs[0]
	# predict by activation function
	output_softmax = F.softmax(logits).to(device)
	threshold = 0.5
	preds = np.where(output_softmax[:, 1] > threshold, 1, 0)
	# print(preds)


	# if claim [1], then all premise [0] vectors = [0,..,0] -- vice versa
	argumentDectection = {}
	if preds == 1: 	# '1' = argumetative
		# argumentDectection['argumentative'] = list(bert_pooled_output.detach().numpy()[0]) # original
		# argumentDectection['non-argumentative'] = list(np.zeros(bert_pooled_output.size()[1])) # original
		argumentDectection['argumentative'] = [1] # original
		# argumentDectection['non-argumentative'] = [0] # original
	else: 		# '0' = non-argumetative
		# argumentDectection['argumentative'] = list(np.zeros(bert_pooled_output.size()[1])) # original
		# argumentDectection['non-argumentative'] = list(bert_pooled_output.detach().numpy()[0]) # original
		argumentDectection['argumentative'] = [0] # original
		# argumentDectection['non-argumentative'] = [1] # original
	return argumentDectection

def conditional_argumentANDclaimpremise(sentence,tokenizer,model_argumentDectection, model_claim_premise):
	### '0' = non-argumetative, '1' = argumetative
	# read BERT fine-tuned model as eval() to predict
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	# device = "cpu"
	model_argumentDectection = model_argumentDectection.to(device)
	model_argumentDectection.eval()
	input_sentence = tokenizer(list([sentence]), return_tensors="pt",padding=True,truncation=True)
	outputs, bert_pooled_output = model_argumentDectection(**input_sentence.to(device))
	logits = outputs[0]
	# predict by activation function
	output_softmax = F.softmax(logits)
	threshold = 0.5
	preds = np.where(output_softmax[:, 1] > threshold, 1, 0)
	# print(preds)


	# if claim [1], then all premise [0] vectors = [0,..,0] -- vice versa
	argumentDectection = {}
	if preds == 1: 	# '1' = argumetative
		argumentDectection['argumentative'] = list(bert_pooled_output.detach().numpy()[0]) # original
		argumentDectection['non-argumentative'] = list(np.zeros(bert_pooled_output.size()[1])) # original
		model_claim_premise = model_claim_premise.to(device)
		model_claim_premise.eval()
		# input_sentence = tokenizer(list([sentence]), return_tensors="pt",padding=True,truncation=True)
		outputs_claim_premise, bert_pooled_output_claim_premise = model_claim_premise(**input_sentence.to(device))
		logits_claim_premise = outputs_claim_premise[0]
		# predict by activation function
		output_softmax_claim_premise = F.softmax(logits_claim_premise)
		threshold_claim_premise = 0.5
		preds_claim_premise = np.where(output_softmax_claim_premise[:, 1] > threshold_claim_premise, 1, 0)
		if preds_claim_premise == 1: 	# '1' = claim
			argumentDectection['claim'] = list(bert_pooled_output_claim_premise.detach().numpy()[0]) # original
			argumentDectection['premise'] = list(np.zeros(bert_pooled_output_claim_premise.size()[1])) # original
		else: 		# '0' = premise
			argumentDectection['claim'] = list(np.zeros(bert_pooled_output_claim_premise.size()[1])) # original
			argumentDectection['premise'] = list(bert_pooled_output_claim_premise.detach().numpy()[0]) # original


	else: 		# '0' = non-argumetative
		argumentDectection['argumentative'] = list(np.zeros(bert_pooled_output.size()[1])) # original
		argumentDectection['non-argumentative'] = list(bert_pooled_output.detach().numpy()[0]) # original
		argumentDectection['claim'] = list(np.zeros(bert_pooled_output.size()[1])) # original
		argumentDectection['premise'] = list(np.zeros(bert_pooled_output.size()[1])) # original
	return argumentDectection
	
# # sentence = "They have the life of a person at their discretion." # 0
# sentence = "Artists' freedom to express ideas" # 1
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model_argumentDectection = cBertForSequenceClassification_withOutput.from_pretrained('./__dumps_objs/argumentDetection')
# result = argumentationComponents_claim_premise(sentence,tokenizer,model_argumentDectection)
# print(result)

def get_propagandaFeatures(text_list,
							tokenizer, model, ps, read_speechStyle, read_agreeableness,
						   read_conscientiousness, read_extraversion, read_neuroticism, read_openness,
						   read_persuasion_concreteness,
						   read_persuasion_subjectivity, read_sentiment_warriner,
						   read_sentiment_depechemood, read_sentiment_connotation,
						   read_sentiment_politeness_pos, read_sentiment_politeness_neg,
						   read_messageSimplicity_imageability, model_claim_premise,
						   model_argumentDectection,
						   persuasion_speechStyle_=False,persuasion_lexicalComplexity_=False,
						   # persuasion_personalityTraits_=False,argumentMining_arguingType_=False,
						   persuasion_concreteness_=False,persuasion_subjectivity_=False,
						   sentiment_SentiWordnet_=False,sentiment_warriner_=False,sentiment_depechemood_=False,sentiment_connotation_=False,sentiment_politeness_=False,
						   messageSimplicity_imageability_=False,messageSimplicity_lexicalLength_=False,messageSimplicity_lexicalEncoding_=False,messageSimplicity_pronouns_=False,
						   argumentationComponents_claim_premise_=False,argumentDectection_=False,conditional_argumentANDclaimpremise_=False,
						   argumentDectection_binary_=False,
						   ):
	
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	# device = "cpu"

	completeFeatures_vectors = []
	count_for_writing = 0
	for text in text_list:
		speechStyle_vectors = persuasion_speechStyle(text,read_speechStyle,ps) if persuasion_speechStyle_ == True else {}
		lexicalComplexity_vectors = persuasion_lexicalComplexity(text,tokenizer,model=model) if persuasion_lexicalComplexity_ == True else {}
		# personalityTraits_vectors = persuasion_personalityTraits(text,read_agreeableness,read_conscientiousness,read_extraversion,read_neuroticism,read_openness) if persuasion_personalityTraits_ == True else {}

		# arguingType_vectors = argumentMining_arguingType(text,read_arguingType) if argumentMining_arguingType_ == True else {}
		concreteness_vector = persuasion_concreteness(text,read_persuasion_concreteness) if persuasion_concreteness_ == True else {}
		subjectivity_vectors = persuasion_subjectivity(text,read_persuasion_subjectivity) if persuasion_subjectivity_ == True else {}

		sentiwordnet_vectors = sentiment_SentiWordnet(text) if sentiment_SentiWordnet_==True	else {}
		warrinerVAD_vectors = sentiment_warriner(text,read_sentiment_warriner) if sentiment_warriner_==True	else {}
		depechemood_vectors = sentiment_depechemood(text,read_sentiment_depechemood) if sentiment_depechemood_==True	else {}
		connotation_vectors = sentiment_connotation(text,read_sentiment_connotation) if sentiment_connotation_==True	else {}
		politeness_vectors = sentiment_politeness(text,read_sentiment_politeness_pos,read_sentiment_politeness_neg) if sentiment_politeness_==True else {}

		imageability_vectors = messageSimplicity_imageability(text,read_messageSimplicity_imageability) if messageSimplicity_imageability_==True	else {}
		lexicalLength_vectors = messageSimplicity_lexicalLength(text) if messageSimplicity_lexicalLength_==True else {}
		lexicalEncoding_vectors = messageSimplicity_lexicalEncoding(text) if messageSimplicity_lexicalEncoding_==True else {}
		pronouns_vectors = messageSimplicity_pronouns(text) if messageSimplicity_pronouns_==True	else {}

		claim_premise_vectors = argumentationComponents_claim_premise(text,tokenizer,model_claim_premise) if argumentationComponents_claim_premise_==True else {}
		argumentDectection_vectors = argumentDectection(text,tokenizer,model_argumentDectection) if argumentDectection_==True else {}
		conditional_argumentANDclaimpremise_vectors = conditional_argumentANDclaimpremise(text,tokenizer,model_argumentDectection, model_claim_premise) if conditional_argumentANDclaimpremise_==True else {}
		##
		argumentDectection_binary_vectors = argumentDectection_binary(text,tokenizer,model_argumentDectection) if argumentDectection_binary_==True else {}

		text_dict = {'text':text}
		## with BERT sentence embedding
		allFeature_vectors = {**text_dict,**speechStyle_vectors,**lexicalComplexity_vectors,**concreteness_vector,**subjectivity_vectors,**sentiwordnet_vectors,**warrinerVAD_vectors,
		**depechemood_vectors,**connotation_vectors,**politeness_vectors,**imageability_vectors, **lexicalLength_vectors,**lexicalEncoding_vectors,**pronouns_vectors,
		**claim_premise_vectors,**argumentDectection_vectors,**conditional_argumentANDclaimpremise_vectors,**argumentDectection_binary_vectors}
		# ## with BERT sentence embedding_old version_no personalityTrait and AM
		# allFeature_vectors = {**text_dict,**speechStyle_vectors,**lexicalComplexity_vectors,**personalityTraits_vectors, **arguingType_vectors,**concreteness_vector,**subjectivity_vectors,**sentiwordnet_vectors,**warrinerVAD_vectors,
		# **depechemood_vectors,**connotation_vectors,**politeness_vectors,**imageability_vectors, **lexicalLength_vectors,**lexicalEncoding_vectors,**pronouns_vectors}
		# ## usable (no BERT)
		# allFeature_vectors = {**text_dict,**personalityTraits_vectors, **arguingType_vectors,**concreteness_vector,**subjectivity_vectors,**sentiwordnet_vectors,**warrinerVAD_vectors,
		# **depechemood_vectors,**connotation_vectors,**politeness_vectors,**imageability_vectors, **lexicalLength_vectors,**lexicalEncoding_vectors,**pronouns_vectors}


		
		# print('========',allFeature_vectors.values()) ## return
		reformated_allFeature_vectors = []
		for element in allFeature_vectors.values():
			if isinstance(element, list) == False:	reformated_allFeature_vectors.append(element)
			elif isinstance(element, list):
				for item in element: 
					reformated_allFeature_vectors.append(item)
		# print('reformated_allFeature_vectors',reformated_allFeature_vectors)

		# write file to check
		allFeature_vectors_noTextCol = reformated_allFeature_vectors[1:]
		# print('allFeature_vectors_noTextCol',allFeature_vectors_noTextCol)
		allFeature_vectors_noTextCol = [0 if math.isnan(x) else x for x in allFeature_vectors_noTextCol]
		completeFeatures_vectors.append(allFeature_vectors_noTextCol)
		# print(allFeature_vectors_noTextCol, len(allFeature_vectors_noTextCol))

		if count_for_writing == 0:
			df_allFeature_vectors = pd.DataFrame.from_dict(data=allFeature_vectors, orient='index').transpose()
			df_allFeature_vectors.to_csv('propagandaSample_allFeatures.csv', header=True,index=False,mode='w')
			count_for_writing+=1
		else:
			df_allFeature_vectors = pd.DataFrame.from_dict(data=allFeature_vectors, orient='index').transpose()
			df_allFeature_vectors.to_csv('propagandaSample_allFeatures.csv', header=False,index=False,mode='a')

	# print('\n==================completeFeatures_vectors:\n',completeFeatures_vectors)
	# print('list(completeFeatures_vectors):',len(completeFeatures_vectors))
	# print('done')
	# # exit()

	return completeFeatures_vectors
# print('STARTED !')
# sentence = ["the heart doesn't have to work as hard as it does with weak muscles"] # 0
# print('sentence:',sentence)
# x_result = get_propagandaFeatures(sentence,persuasion_speechStyle_=True,persuasion_lexicalComplexity_=True,sentiment_depechemood_=True,argumentDectection_=True,argumentationComponents_claim_premise_=True)
# print(x_result)
# print('DONE Processing !')

# result = get_propagandaFeatures(text_list='hey hey can you read this messgae clearly',
# 							persuasion_speechStyle_=True,persuasion_lexicalComplexity_=True,
# 							persuasion_concreteness_=True,persuasion_subjectivity_=True,
# 							sentiment_SentiWordnet_=True,sentiment_warriner_=True,sentiment_depechemood_=True,sentiment_connotation_=True,sentiment_politeness_=True,
# 							messageSimplicity_imageability_=True,messageSimplicity_lexicalLength_=True,messageSimplicity_lexicalEncoding_=True,messageSimplicity_pronouns_=True)
# print('result:',result)
