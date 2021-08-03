# This python script implements the pipeline of propaganda detection on spans and its tecniques applied.
# Our pipeline has 2 models
# 1) MODEL1:    span boundary detection
# 2) MODEL2:    propaganda techique detection (at sentence-span level)
import torch, pandas as pd, numpy as np, json, pickle
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
import torch.nn.functional as F
from tokenizers import decoders
from custom_transformers.src.transformers.models.bert import BertTokenizer, BertForTokenClassification_CRF, BertModel, cBertForSequenceClassification_withOutput
from custom_transformers.src.transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from custom_transformers.src.transformers.models.roberta import RobertaTokenizer, RobertaForSequenceClassification_joint_loss
from extractPropagandaFeatures import get_propagandaFeatures
from flask import Flask,request,jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def combine_predictions(preds):
	rearraged_sentence_list = []
	for i,sentence in enumerate(preds): # each sentence (128,19)
		label_by_token = []
		for j,token in enumerate(sentence):  # each token (19)
			max_score = max(token) # find max of list
			j = 0
			minmax_token = [1 if i==max_score else j for i in token] # if max of token==1, then map with dict
			index_position = minmax_token.index(1)  # get position of predicted label
			label_by_token.extend([index_position])
		rearraged_sentence_list.append(label_by_token)
	return rearraged_sentence_list

# sudo
# ==== MODEL 1: Span boundary detection
# read the trained model
# read text, make batches based on sentence-level
# predict per batch
# option1:
# - count 'S' label, then collect indices where 'S' indicates
# - from 'S' indices, get the tokens, then convert tokenzied to original text
# - save text per batch - with the converted span
def span_boundary_detection(sentences,device,span_boundary_detection_tokenizer,span_boundary_detection_model,num_batch=8, MAX_LEN=128,num_labels=2):
    tokenizer = span_boundary_detection_tokenizer;  tokenizer.pad_token = 0
    model = span_boundary_detection_model
    model = model.to(device)
    model = model.eval()

    input_batches = []
    for x in batch(sentences, num_batch):
        input_batches.append(list(x))

    for input_batch in input_batches:
        input_batch = [str(x) for x in input_batch]  # in case of 'nan'
        inputs = tokenizer(input_batch, return_tensors="pt", padding='max_length', truncation=True,max_length=MAX_LEN, return_offsets_mapping=True)
        inputs_offset_mapping = inputs.offset_mapping.tolist()

        ### find index of [SEP] --- per sentence
        ids2token_batch = []
        for element in inputs.input_ids:
            span_id2token = tokenizer.convert_ids_to_tokens(element)#[1:sep_index] # to ignore [CLS]
            sep_index = span_id2token.index('[SEP]')
            ids2token_batch.append(sep_index) # start_sentence from index: 1, end_sentence from index: sep_index

        if device != 'cpu':
            inputs = inputs.to(device)

        ###### perform softmax
        # if CRF != True:
        output = model(**inputs)
        output = output[0].logits
        output_activation = F.softmax(output, dim=1)
        pred = combine_predictions(output_activation)
        # print('pred', pred, len(pred[0]))
        # exit()

        ###### perform CRF
        # output,output_activation = model(**inputs,return_crf=True)
        # pred = output_activation
        # # print('performing CRF output_activation')
        # print('\npred',pred,len(pred))
        # # exit()

        save_predicted_token_indices = []   # template: [sentence, [ [span_tokens1,start_idx,end_idx],[span_tokens2,start_idx,end_idx] ]]
        ### match the tokens and its labels
        for idx,(sent_text,pred_sentence,offsets) in enumerate(zip(input_batch,pred,inputs_offset_mapping)):

            sentence_ids = inputs.input_ids[idx][1:ids2token_batch[idx]]
            tokenized_sent = tokenizer.convert_ids_to_tokens(inputs.input_ids[idx][1:ids2token_batch[idx]])
            pred_sentence = pred_sentence[1:ids2token_batch[idx]]#.tolist()
            offsets = offsets[1:ids2token_batch[idx]]

            # searching for 'S' individual token, with its offsets -- for search/combining later
            propaganda_idxtoken,propaganda_idxList,propaganda_idxPair, propaganda_idxoriginal= [],[], [], []
            for idx2,(token,pred,offset) in enumerate(zip(tokenized_sent,pred_sentence,offsets)):
                if pred == 1:
                    propaganda_idxtoken.append(idx2)
                    propaganda_idxoriginal.append(offset) # have to get offsets of MULTIPLE tokens

            # forming connected spans as one span
            temp_start_idx = 0
            for i,(element,tokenidx_original) in enumerate(zip(propaganda_idxtoken,propaganda_idxoriginal)):
                # print(i,element,tokenidx_original)
                # exit()
                if i == 0:  temp_start_idx = i
                try:
                    if element+1 != propaganda_idxtoken[i+1]:   # if next num is not consecutive, then make it a span
                        propaganda_idxList.append([propaganda_idxtoken[temp_start_idx:i + 1],propaganda_idxoriginal[temp_start_idx:i + 1]])
                        temp_start_idx = i+1    # reset parameter when each span is processed
                except IndexError: # last index occurs 'IndexError'
                    propaganda_idxList.append([propaganda_idxtoken[temp_start_idx:],propaganda_idxoriginal[temp_start_idx:i+1]]) # if the last index is a propaganda span

            for idx_list in propaganda_idxList:
                idx_list, original_offset = idx_list
                start_original_offset = original_offset[0][0]
                end_original_offset = original_offset[-1][1]

                if len(idx_list) == 1:
                    start_idx, end_idx = idx_list[0], idx_list[0]
                    tokenzied_span = tokenized_sent[start_idx:end_idx + 1]
                    span_ids_to_words = tokenizer.decode(sentence_ids[start_idx:start_idx + 1]) # using [start_idx:start_idx + 1] for only len(idx_list) == 1
                    bert_ids = sentence_ids[start_idx:start_idx + 1]

                    propaganda_idxPair.append([tokenzied_span,bert_ids,start_idx, start_idx+1,start_original_offset,end_original_offset])  # per sentence [tokenzied_span,start_idx, start_idx+1]
                else:
                    start_idx, end_idx = idx_list[0], idx_list[-1] #idx_list[-1]
                    tokenzied_span = tokenized_sent[start_idx:end_idx+1]
                    span_ids_to_words = tokenizer.decode(sentence_ids[start_idx:end_idx+1])
                    bert_ids = sentence_ids[start_idx:end_idx+1]
                    propaganda_idxPair.append([tokenzied_span,bert_ids,start_idx,end_idx,start_original_offset,end_original_offset]) # per sentence
            save_predicted_token_indices.append([sent_text, propaganda_idxPair])  # collect spans
        print("span_boundary_detection")
        return save_predicted_token_indices

# ==== MODEL 2: Propaganda techique detection
# read the trained model
# read text output from MODEL1 as input batches
# predict per span of a sentence
# option1: count on output_predictions ***
# - as it predicts per sentence+span, get the the highest score
# - if the 2nd highest label score, if more than thershold of 0.5? then we consider there are more than 1 technique in this span.
# form the JSONV4 to send to webdemo
#
# *** ISSUES:
# case 1: one sentence can have multiple spans, then model 2 needs to predict the same sentence more than one time wrt spans

def sentence_span_classification(sentence_spans,device,id_name,
                                       tokenizer, model, ps, read_speechStyle, read_agreeableness,
                                       read_conscientiousness, read_extraversion, read_neuroticism, read_openness,
                                       read_persuasion_concreteness,
                                       read_persuasion_subjectivity, read_sentiment_warriner,
                                       read_sentiment_depechemood, read_sentiment_connotation,
                                       read_sentiment_politeness_pos, read_sentiment_politeness_neg,
                                       read_messageSimplicity_imageability, model_claim_premise,
                                       model_argumentDectection,
                                       roberta_tokenizer,bert_tokenizer,model2,threshold = 0.1):
    # read the trained model
    num_labels = 15 # 0+14 labels
    bert_tokenizer.decoder = decoders.WordPiece()
    model2 = model2.to(device)
    model2.eval()

    all_spans_per_doc = []

    ### process per sentence (that has spanS in the loop)
    for sentence in sentence_spans:
        sent, spans = sentence
        if len(spans) == 0:
            final_span, label, start_original_offset, end_original_offset, probability = '', 0, 0 ,0 ,0
            span_elements_sentence = [final_span, label, start_original_offset, end_original_offset, probability]
            all_spans_per_doc.append([sent, [span_elements_sentence]])
            continue
        sentence_inputs = roberta_tokenizer(sent, return_tensors="pt", padding='max_length', truncation=True,max_length=128)
        # extract sentence features
        args_dict = dict(
            text_list=sent,#['', ''],
            tokenizer=tokenizer, model=model, ps=ps,
            read_speechStyle=read_speechStyle, read_agreeableness=read_agreeableness,
            read_conscientiousness=read_conscientiousness, read_extraversion=read_extraversion,
            read_neuroticism=read_neuroticism, read_openness=read_openness,
            read_persuasion_concreteness=read_persuasion_concreteness,
            read_persuasion_subjectivity=read_persuasion_subjectivity,
            read_sentiment_warriner=read_sentiment_warriner,
            read_sentiment_depechemood=read_sentiment_depechemood,
            read_sentiment_connotation=read_sentiment_connotation,
            read_sentiment_politeness_pos=read_sentiment_politeness_pos,
            read_sentiment_politeness_neg=read_sentiment_politeness_neg,
            read_messageSimplicity_imageability=read_messageSimplicity_imageability,
            model_claim_premise=model_claim_premise, model_argumentDectection=model_argumentDectection,

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

        sentence_extraFeatures = get_propagandaFeatures(**args_dict)
        sentence_extraFeatures = torch.tensor([sentence_extraFeatures])  # .unsqueeze(0) ## try max_seq
        sentence_extraFeatures = torch.reshape(sentence_extraFeatures,(sentence_extraFeatures.size()[1], 1, sentence_extraFeatures.size()[2]))

        span_elements_sentence = []
        for span in spans:
            tokens, bert_ids, token_start_idx, token_end_idx, start_original_offset, end_original_offset = span

            # make prediction per span -- use the bert_tokenzied words to run on roberta model
            span_inputs = roberta_tokenizer(tokens, return_tensors="pt", padding='max_length', truncation=True,max_length=20)

            if device != 'cpu':
                sentence_inputs = sentence_inputs.to(device)
                span_inputs = span_inputs.to(device)
                sentence_extraFeatures = sentence_extraFeatures.to(device)

            _, _, pred_logits = model2(**sentence_inputs, span=span_inputs, extraFeature=sentence_extraFeatures)     # prediction per span
            output_softmax = F.softmax(pred_logits)[0].tolist()[1:]     # get only the label 1 to 14 (ignore 0)
            output_softmax = [round(num, 2) for num in output_softmax]

            # get index of labels that has theshold more than 0.5?
            selected_indics = np.argwhere(np.array(output_softmax) > threshold)+1
            if len(selected_indics) == 0:   selected_indics = [[0]] # in case of a given span has NO Propaganda_technique detected
            for label in selected_indics:
                label = label[0]
                probability = output_softmax[label-1]
                final_span = sent[start_original_offset: end_original_offset]
                span_elements = [final_span, label, start_original_offset, end_original_offset, probability]
                span_elements_sentence.append(span_elements)
        all_spans_per_doc.append([sent,span_elements_sentence])

    dict_to_json = {}
    dict_to_json['article_'+str(id_name)] = {}
    for idx,element in enumerate(all_spans_per_doc):
        idx+=1
        sentence, spans_list = element
        print('sentence, spans_list', sentence, spans_list)
        dict_to_json['article_' + str(id_name)]['propaganda_' + str(idx)] = {}
        dict_to_json['article_'+str(id_name)]['propaganda_' + str(idx)]['text'] = sentence

        # count the repeated offsets, if there is -- get/save offsets, if no -- write as a single number
        temp_idx_offsets = []
        for idx2,span in enumerate(spans_list):     # get only the offset(s) that duplicate
            final_span, label, start_original_offset, end_original_offset, probability = span
            temp_idx_offsets.append((start_original_offset, end_original_offset))
        duplicate_offsets = {x for x in temp_idx_offsets if temp_idx_offsets.count(x) > 1}

        # check if write as a single number , or combined as a list
        temp_duplicated_offsets = []
        count_repeated_idx = 0
        for idx2,span in enumerate(spans_list):
            final_span, label, start_original_offset, end_original_offset, probability = span
            duplicate_offsets = list(duplicate_offsets)

            current_offsets = (start_original_offset, end_original_offset)  # check here if the spans are shared (multiple labels of the same span)
            if current_offsets in duplicate_offsets:
                elements = [str(idx),str(idx2),final_span,label,start_original_offset,end_original_offset,(start_original_offset,end_original_offset),probability]  # save all duplicated values
                temp_duplicated_offsets.append(elements)
                count_repeated_idx += 1
            else:   #   write after skiping
                idx2 += 1
                dict_to_json['article_' + id_name]['propaganda_' + str(idx)]['span_' + str(idx2-count_repeated_idx)] = {}
                dict_to_json['article_' + id_name]['propaganda_' + str(idx)]['span_' + str(idx2-count_repeated_idx)]['span'] = {}
                dict_to_json['article_' + id_name]['propaganda_' + str(idx)]['span_' + str(idx2-count_repeated_idx)]['label'] = {}
                dict_to_json['article_' + id_name]['propaganda_' + str(idx)]['span_' + str(idx2-count_repeated_idx)]['start_char'] = {}
                dict_to_json['article_' + id_name]['propaganda_' + str(idx)]['span_' + str(idx2-count_repeated_idx)]['end_index'] = {}
                dict_to_json['article_' + id_name]['propaganda_' + str(idx)]['span_' + str(idx2-count_repeated_idx)]['probability'] = {}
                dict_to_json['article_' + id_name]['propaganda_' + str(idx)]['span_' + str(idx2-count_repeated_idx)]['span'] = str(final_span)
                dict_to_json['article_' + id_name]['propaganda_' + str(idx)]['span_' + str(idx2-count_repeated_idx)]['label'] = int(label)
                dict_to_json['article_' + id_name]['propaganda_' + str(idx)]['span_' + str(idx2-count_repeated_idx)]['start_char'] = int(start_original_offset)
                dict_to_json['article_' + id_name]['propaganda_' + str(idx)]['span_' + str(idx2-count_repeated_idx)]['end_index'] = int(end_original_offset)
                dict_to_json['article_' + id_name]['propaganda_' + str(idx)]['span_' + str(idx2-count_repeated_idx)]['probability'] = float(probability)
            last_span_idx = idx2-count_repeated_idx

        # write as dataframe -> list - > dict
        df_duplicated_offsets = pd.DataFrame(temp_duplicated_offsets,columns =['propganda_idx','span_idx','final_span','label','start_original_offset','end_original_offset','offsets','probability'])
        for propganda_idx in set(df_duplicated_offsets.propganda_idx.tolist()):
            for offset in set(df_duplicated_offsets.offsets.tolist()):
                last_span_idx += 1  # continue running index
                dict_to_json['article_' + id_name]['propaganda_' + propganda_idx]['span_' + str(last_span_idx)] = {}   # create dict to put values later
                dict_to_json['article_' + id_name]['propaganda_' + propganda_idx]['span_' + str(last_span_idx)]['span'] = {}
                dict_to_json['article_' + id_name]['propaganda_' + propganda_idx]['span_' + str(last_span_idx)]['label'] = {}
                dict_to_json['article_' + id_name]['propaganda_' + propganda_idx]['span_' + str(last_span_idx)]['start_char'] = {}
                dict_to_json['article_' + id_name]['propaganda_' + propganda_idx]['span_' + str(last_span_idx)]['end_index'] = {}
                dict_to_json['article_' + id_name]['propaganda_' + propganda_idx]['span_' + str(last_span_idx)]['probability'] = {}

                span = str(df_duplicated_offsets[(df_duplicated_offsets['propganda_idx'] == propganda_idx) & (df_duplicated_offsets['offsets'] == offset)].final_span.values[0])
                label_list = df_duplicated_offsets[(df_duplicated_offsets['propganda_idx'] == propganda_idx) & (df_duplicated_offsets['offsets'] == offset)].label.values
                label_list = [int(x) for x in label_list]
                start_original_offset_list = df_duplicated_offsets[(df_duplicated_offsets['propganda_idx'] == propganda_idx) & (df_duplicated_offsets['start_original_offset'] == list(offset)[0])].start_original_offset.values
                start_original_offset_list = [int(x) for x in start_original_offset_list]
                end_original_offset_list = df_duplicated_offsets[(df_duplicated_offsets['propganda_idx'] == propganda_idx) & (df_duplicated_offsets['end_original_offset'] == list(offset)[1])].end_original_offset.values
                end_original_offset_list = [int(x) for x in end_original_offset_list]
                probability_list = list(df_duplicated_offsets[(df_duplicated_offsets['propganda_idx'] == propganda_idx) & (df_duplicated_offsets['offsets'] == offset)].probability.values)
                probability_list = [float(x) for x in probability_list]

                dict_to_json['article_' + id_name]['propaganda_' + propganda_idx]['span_' + str(last_span_idx)]['span'] = span
                dict_to_json['article_' + id_name]['propaganda_' + propganda_idx]['span_' + str(last_span_idx)]['label'] = label_list
                dict_to_json['article_' + id_name]['propaganda_' + propganda_idx]['span_' + str(last_span_idx)]['start_char'] = start_original_offset_list
                dict_to_json['article_' + id_name]['propaganda_' + propganda_idx]['span_' + str(last_span_idx)]['end_index'] = end_original_offset_list
                dict_to_json['article_' + id_name]['propaganda_' + propganda_idx]['span_' + str(last_span_idx)]['probability'] = probability_list

    print('dict_to_json', dict_to_json)
    with open('output_'+str(id_name)+".json", "w") as outfile:
        json.dump(dict_to_json, outfile, indent=4)
    print("le type de json")
    print(type(dict_to_json))
    print("sentance_span_classification")
    return dict_to_json

############### PRE-LOAD MODULES ###############
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer_bert_cased = PreTrainedTokenizerFast.from_pretrained('bert-base-cased')
model1 = BertForTokenClassification_CRF.from_pretrained('./fine_tuned_model/BERTfinetune_tokenclassification_CRF/', num_labels=2)
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model2 = RobertaForSequenceClassification_joint_loss.from_pretrained('./fine_tuned_model_sentence_span/RoBertaForSequenceClassification_jointLoss_sentence_span_' + 'semantic_and_argumentation',num_labels=15)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True,)    # Whether the model returns all hidden-states.
model = model.to(device)
model.eval()   # Put the model in "evaluation" mode, meaning feed-forward operation.

ps = PorterStemmer()
read_speechStyle = pickle.load(open('./__dumps_objs/persuasion_speechStyle.dump','rb'))

read_agreeableness = pickle.load(open('./__dumps_objs/serial_personalityTraits_agreeableness.dump','rb'))
read_conscientiousness = pickle.load(open('./__dumps_objs/serial_personalityTraits_conscientiousness.dump','rb'))
read_extraversion = pickle.load(open('./__dumps_objs/serial_personalityTraits_extraversion.dump','rb'))
read_neuroticism = pickle.load(open('./__dumps_objs/serial_personalityTraits_neuroticism.dump','rb'))
read_openness = pickle.load(open('./__dumps_objs/serial_personalityTraits_openness.dump','rb'))

read_persuasion_concreteness = pickle.load(open('./__dumps_objs/serial_argumentMining_concreteness.dump','rb'))
read_persuasion_subjectivity = pickle.load(open('./__dumps_objs/argumentMining_subjectivity.dump','rb'))

# PorterStemmer_obj = PorterStemmer() #sentiment_SentiWordnet
read_sentiment_warriner = pickle.load(open('./__dumps_objs/sentiment_warriner.dump','rb'))
read_sentiment_depechemood= pickle.load(open('./__dumps_objs/sentiment_depechemood.dump','rb'))
read_sentiment_connotation= pickle.load(open('./__dumps_objs/sentiment_connotation.dump','rb'))
read_sentiment_politeness_pos= pickle.load(open('./__dumps_objs/sentiment_politeness_pos.dump','rb'))
read_sentiment_politeness_neg= pickle.load(open('./__dumps_objs/sentiment_connotation_neg.dump','rb'))

read_messageSimplicity_imageability= pickle.load(open('./__dumps_objs/messageSimplicity_imageability.dump','rb'))

model_claim_premise = cBertForSequenceClassification_withOutput.from_pretrained('./__dumps_objs/arugmentationComponent_detection',from_tf=False)
model_claim_premise = model_claim_premise.to(device)
model_argumentDectection = cBertForSequenceClassification_withOutput.from_pretrained('./__dumps_objs/argumentDetection',from_tf=False)
model_argumentDectection = model_argumentDectection.to(device)

def pipeline_propaganda_span_detection(text_paragraph,id_name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sentences = sent_tokenize(text_paragraph)
    sentence_spans = span_boundary_detection(sentences,device,span_boundary_detection_tokenizer=tokenizer_bert_cased,span_boundary_detection_model=model1)
    res = sentence_span_classification(sentence_spans,device,id_name,
                                       tokenizer, model, ps, read_speechStyle, read_agreeableness,
                                       read_conscientiousness, read_extraversion, read_neuroticism, read_openness,
                                       read_persuasion_concreteness,
                                       read_persuasion_subjectivity, read_sentiment_warriner,
                                       read_sentiment_depechemood, read_sentiment_connotation,
                                       read_sentiment_politeness_pos, read_sentiment_politeness_neg,
                                       read_messageSimplicity_imageability, model_claim_premise,
                                       model_argumentDectection,
                                       roberta_tokenizer=roberta_tokenizer, bert_tokenizer=bert_tokenizer,
                                       model2=model2, threshold=0.1
                                       )
    return res

############### Do_POST from web ###############
# text as a paragraph (multiple sentences)
@app.route("/post",methods=['POST'])
def sendData():
    if request.method=='POST':
       request_data = request.get_json()
       test_text = request_data['text']
       json_sample=request_data
       json_sample = json.dumps(json_sample) # JSON file ---> read json from command line from *here (this line)*
       json_sample = json.loads(json_sample) # from Json to dict
       res=pipeline_propaganda_span_detection(text_paragraph=json_sample['text'], id_name=str(json_sample['id']))
       data="success"
       print("success")
       return  jsonify(res)

if __name__ == "__main__":
    app.run(debug=True,host=app.config.get("HOST", "127.0.0.2"),
    )


# ############## TESTING PYTHON Functions #################
# test_text = 'In a glaring sign of just how stupid and petty things have become in Washington these days, Manchin was invited on Fox News Tuesday morning to discuss how he was one of the only Democrats in the chamber for the State of the Union speech not looking as though Trump killed his grandma.' \
#            ' Here is the second sentence.' \
#            ' The results have been unmitigated disaster after disaster.' \
#            ' Yet, if we do that, we wont have to worry about mass shooters, right?'
#
# json_sample = {'id': 0, 'text': test_text, 'headers': {'Access-Control-Allow-Origin': '*'}}
# json_sample = json.dumps(json_sample) # JSON file ---> read json from command line from *here (this line)*
# ####################
# json_sample = json.loads(json_sample) # from Json to dict
# res = pipeline_propaganda_span_detection(text_paragraph=json_sample['text'], id_name=str(json_sample['id']))
# print('DONE\n',res)