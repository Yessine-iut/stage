import torch, os
import pandas as pd, numpy as np
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from nltk.tokenize.util import string_span_tokenize
from sklearn.metrics import precision_recall_fscore_support, classification_report
from custom_transformers.src.transformers.models.bert import BertTokenizer, BertForTokenClassification, BertForTokenClassification_CRF
from custom_transformers.src.transformers.models.roberta import RobertaTokenizer, RobertaForTokenClassification_CRF
from custom_transformers.src.transformers.optimization import AdamW
from early_stopping import EarlyStopping
print('done importing')

def getFiles_in_folder(path):
	return os.listdir(path)

def arrangeNERdataset(path, label_tag,custom_tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')):
    # the model returns dataFrame of normal_text, and bert_tokenized_label
    files = getFiles_in_folder(path)
    # print('len_files', len(files))
    collect_tokens_labels = []
    propagandaLabeled_tokenizedSentence, nonePropagandaLabeled_tokenizedSentence = [], []
    complete_taggings = []
    # files = [x for x in files if x.]
    unique_files = []
    for file in files:
        file = file.split('.')[0]
        unique_files.extend([file])
    unique_files = set(unique_files) # files contain both article_text and annotated_label

    for file in unique_files:
        # print(file)
        article_name = file.split('.')[0]
        # read each .txt file, and its labels (of the same artcile at a time)
        try:
            with open(path+file+'.txt','r') as text_string, open(path+article_name+label_tag,'r') as labels:
                text = text_string.read()
                text_spans = list(string_span_tokenize(text, '\n')) # segmented as spans
                label_spans = labels.read().split('\n')
                text_spans_range, propaganda_spans_range = [],[]
                # test_text = []
                # put all span into range
                for text_span in text_spans:
                    start_offset, end_offset = text_span[0],text_span[1]
                    original_text_range = range(int(start_offset),int(end_offset)) ### main variable -- write file according to this to avoid duplicate rows
                    original_text = text[text_span[0]:text_span[1]]
                    # print('=====current text range == original_text:',original_text_range,original_text)
                    text_spans_range.append(original_text_range)

                    # proceed only sentence with propaganda_span
                    for label_span in label_spans:
                        try:
                            article_id, technique, label_start_offset, label_end_offset = label_span.split('\t')
                            # print('original_text_range,label_start_offset, label_end_offset',original_text_range,label_start_offset, label_end_offset)
                            # propaganda_spans_range.append(range(int(start_offset),int(end_offset)))
                            propaganda_spans_range.append(range(int(label_start_offset), int(label_end_offset)))

                            # print('.......current comparison:',int(label_start_offset),int(label_end_offset),original_text_range)
                            # if int(label_start_offset) in original_text_range and int(label_end_offset) in original_text_range: # propaganda span
                            if int(label_start_offset) in original_text_range and int(label_end_offset) <= int(end_offset):  # propaganda span -- only what we consider
                                ### perform tokenization, put 'BIES' labels
                                span = text[int(label_start_offset):int(label_end_offset)]
                                # print('=====span text',span)

                                ## write normal text, then replace the span tokens with the BIES tagging - output as a sentence, but tokenzied elements in a list[]
                                # put all tokens in dict as key, assign value 0
                                tokens_span = word_tokenize(span)
                                # tokens_original_text = word_tokenize(original_text)

                                custom_tokens_span = custom_tokenizer.convert_ids_to_tokens(custom_tokenizer(span).input_ids)[1:-1]
                                custom_tokens_original_text = custom_tokenizer.convert_ids_to_tokens(custom_tokenizer(original_text).input_ids)[1:-1]
                                # print('custom_tokens_original_text',custom_tokens_original_text)
                                # print('tokens_span',custom_tokens_span)
                                # exit()

                                ###### BIEOS implementation
                                # if len(tokens_span) == 1: # 1 word - labeled as propaganda
                                #     inner_arrange = []
                                #     for custom_t in custom_tokens_original_text:
                                #         if custom_t in custom_tokens_span: inner_arrange.append('S')
                                #         else:  inner_arrange.append('O')
                                #     propagandaLabeled_tokenizedSentence.append([original_text,inner_arrange])
                                #     # break  # break to process next sentence
                                # else: # >=2 words (span) - labeled as propaganda
                                #     inner_arrange = []
                                #     count_repetitive_token = 0
                                #     max_count_reprtitive_token = custom_tokens_span.count(custom_tokens_span[-1])
                                #     print('custom_tokens_span',custom_tokens_span)
                                #     for idx,custom_t in enumerate(custom_tokens_original_text):
                                #         # check if 'B' and 'E' both inside already, if so, the rest of sentence should be 'O'
                                #         # if inner_arrange.count('B') == 0 and inner_arrange.count('E') == 0:
                                #         if custom_t == custom_tokens_span[0]:
                                #             if inner_arrange.count('B') == 0 and custom_tokens_original_text[idx+1] in custom_tokens_span[1:]:  # check the next token if 'I', if so, then write 'B'
                                #                 inner_arrange.append('B')
                                #                 print("FIRST 'B' in inner_arrange, inner_arrange:",inner_arrange,custom_t)
                                #             else: inner_arrange.append('O') # inner_arrange.count('B') != 0
                                #             # print('custom_tokens_original_text',custom_tokens_original_text)
                                #             # print('custom_t == custom_tokens_span[0]',custom_t,custom_tokens_span[0])
                                #             # print('checking: inner_arrange[-1] == B', inner_arrange[-1])
                                #         elif custom_t in custom_tokens_span[1:] and inner_arrange.count('B') == 1: # check if token before is 'B' --- and custom_t != custom_tokens_span[-1]
                                #             # try:
                                #             if inner_arrange[-1] == 'B':   # check if 'B' is in span label already and has the only one 'B' --- the true 'I'
                                #                 inner_arrange.append('I')
                                #                 print('>>>>> I1', inner_arrange,custom_t)
                                #             else:
                                #                 inner_arrange.append('O')
                                #                 print('>>>>> O2', inner_arrange, custom_t)
                                #             # elif inner_arrange[-1] == 'I':
                                #             #     inner_arrange.append('I')
                                #             #     print('>>>>> I2', inner_arrange,custom_t)
                                #             #     exit()
                                #             # except IndexError:
                                #             #     # continue
                                #             #     inner_arrange.append('O')
                                #             #
                                #             #     exit()
                                #         elif custom_t == custom_tokens_span[-1]:
                                #             print('"E" custom_t == custom_tokens_span[-1]', '-start-',custom_t,'|',custom_tokens_span[-1],'-end-')
                                #             print('before inner_arrange', inner_arrange)
                                #             try:
                                #                 if inner_arrange[-1] == 'I' and custom_tokens_span.count(custom_tokens_span[-1]) >= 2: # example: '"' has 2 or more times in span
                                #                     inner_arrange.append('I')
                                #                     count_repetitive_token += 1
                                #                     print('>>>>> I3', inner_arrange,custom_t)
                                #                 elif inner_arrange[-1] == 'I' and custom_tokens_span.count(custom_tokens_span[-1]) == 1:
                                #                     inner_arrange.append('E')
                                #                     print('>>>>> E1', inner_arrange,custom_t)
                                #                 elif inner_arrange[-1] == 'I' and custom_tokens_span.count(custom_tokens_span[-1]) > 1 and count_repetitive_token == max_count_reprtitive_token-1:
                                #                     inner_arrange.append('E', inner_arrange,custom_t)
                                #                     print('>>>>> E2, inner_arrange', inner_arrange,custom_t)
                                #                 elif inner_arrange[-1] == 'B' and len(tokens_span) == 2:
                                #                     inner_arrange.append('E')
                                #                     print('>>>>> E3', inner_arrange,custom_t)
                                #                 else:
                                #                     print("other cases:",'-start-',custom_t,'|',custom_tokens_span[-1],'-end-')
                                #                     # print('E inner_arrange after',inner_arrange,'custom_t',custom_t)
                                #                     exit()
                                #             except IndexError:
                                #                 inner_arrange.append('O')
                                #                 print('custom_t == custom_tokens_span[-1]',custom_t,'and', custom_tokens_span[-1])
                                #                 print('E0 IndexError error, inner_arrange:', inner_arrange,custom_t)
                                #                 # exit()
                                #             #     continue
                                #
                                #                 # else:   inner_arrange.append('O')
                                #                 # print('custom_t == custom_tokens_span[-1]', custom_t,custom_tokens_span[-1])
                                #
                                #         # else:   inner_arrange.append('O')
                                #         else:   inner_arrange.append('O')
                                #         # else: # TESTING
                                #         #     print('inner_arrange', inner_arrange)
                                #         #
                                #         #     print('custom_t in custom_tokens_span[1:-1]', custom_t, 'AND',custom_tokens_span[1:-1])
                                #         #     exit()
                                #     propagandaLabeled_tokenizedSentence.append([original_text,inner_arrange])
                                #     # print('original_text',original_text)
                                #     # print('inner_arrange',inner_arrange)
                                #     print('\n\n\n end of for loop')
                                #     # exit()
                                #
                                # break # break to process next span

                                ###### SO implementation (single label)
                                if len(tokens_span) == 1: # 1 word - labeled as propaganda
                                    inner_arrange = []
                                    for custom_t in custom_tokens_original_text:
                                        if custom_t in custom_tokens_span:    inner_arrange.append('S')
                                        else:  inner_arrange.append('O')
                                        # print('inner_arrange',inner_arrange)
                                    propagandaLabeled_tokenizedSentence.append([original_text,inner_arrange])
                                    # print('inner_arrange >>>',inner_arrange)
                                else: # >=2 words (span) - labeled as propaganda
                                    inner_arrange = []
                                    for custom_t in custom_tokens_original_text:
                                        if custom_t == custom_tokens_span[0]:    inner_arrange.append('S')
                                        elif custom_t in custom_tokens_span[1:-1]:    inner_arrange.append('S')
                                        elif custom_t == custom_tokens_span[-1]: inner_arrange.append('S')
                                        else:   inner_arrange.append('O')
                                    propagandaLabeled_tokenizedSentence.append([original_text,inner_arrange])
                                    # print('inner_arrange >>>', inner_arrange)
                                # continue
                        except ValueError:
                            # print('ValueError: label_span',label_span) # usually error occures due to a token of empty string
                            pass
        except FileNotFoundError:
            # print('FileNotFoundError',file) # end of file - found '\n'
            pass

        # for element in propagandaLabeled_tokenizedSentence[:10]:
        #     labels_in_sent = []
        #     print(element)
        #     exit()
        #     for sent,label in element:
        #         tokens_in_sent.append(token)
        #         labels_in_sent.append(label)
        #     complete_taggings.append([tokens_in_sent,labels_in_sent])

    # print('complete_taggings',complete_taggings)
    df_complete_taggings = pd.DataFrame(propagandaLabeled_tokenizedSentence,columns=['text','labels'])
    return df_complete_taggings

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def covertBIOES_to_labels(labels_list,label_num):
    if label_num == 2: BIOES_dict = {'O': 0, 'S': 1}
    else: BIOES_dict = {'O': 0, 'B': 1, 'I': 2, 'E':3, 'S':4}
    converted_labels = []
    for labels in labels_list:
        label_per_row = []
        for l in labels:
            label_per_row.extend([BIOES_dict[l]])
        converted_labels.append(label_per_row)
    return converted_labels

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

def covertLabels_to_BIOES(labels_list,label_num):
    if label_num == 2: BIOES_dict = {'O': 0, 'S': 1}
    else: BIOES_dict = {'O': 0, 'B': 1, 'I': 2, 'E':3, 'S':4}
    converted_labels = []
    for labels in labels_list:
        label_per_row = []
        for l in labels:
            label_per_row.extend(getKeysByValue(BIOES_dict,l))
        converted_labels.append(label_per_row)
    return converted_labels


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


def trainBIOES(model_name, train, labels, label_num,limit=999999999,num_batch=8,MAX_LEN=128,epochs=1,early_stopping=None):
    if model_name == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=label_num)
        print('Training --- Using BertForTokenClassification')
    elif model_name == 'BERT CRF':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification_CRF.from_pretrained('bert-base-uncased', num_labels=label_num)
        print('Training --- Using BertForTokenClassification_CRF')
    elif model_name == 'ROBERTA CRF':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification_CRF.from_pretrained('roberta-base', num_labels=label_num)
        print('Training --- Using RobertaForTokenClassification_CRF')


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model = model.train()
    optim = AdamW(model.parameters(), lr=5e-5)

    train_input_batches, train_label_batches = [], []
    for x in batch(train[:limit], num_batch):
        train_input_batches.append(list(x.values))
    for x in batch(labels[:limit], num_batch):
        train_label_batches.append(list(x.values))

    valid_running_loss = 0
    total_loss = []
    average_valid_loss = None

    for epoch in range(epochs):
        print('At epoch:', epoch + 1, '/', epochs)
        for input_batch, label_batch in zip(train_input_batches, train_label_batches):
            input_batch = [str(x) for x in input_batch]  # in case of 'nan'
            inputs = tokenizer(input_batch, return_tensors="pt", padding='max_length', truncation=True,max_length=MAX_LEN)
            # inputs = tokenizer(input_batch, return_tensors="pt",truncation=True)
            label_batch = covertBIOES_to_labels(label_batch,label_num)
            # labels = torch.tensor(label_batch)
            # print('labels',labels)
            # exit()

            ### in case of token-classification
            label = []
            for e in label_batch:
                int_list = e
                # int_list = ast.literal_eval(e) # e = e.replace('[','').replace(']','').replace(',','').split()
                int_list = (int_list + [-100] * MAX_LEN)[:MAX_LEN] # padding zero at the end, (int_list + [0] * MAX_LEN)[:MAX_LEN]
                label.append(int_list)
            labels = torch.tensor(label)#.unsqueeze(0) ## try max_seq
            # print('test torch.tensor(label)',torch.tensor(label))
            # exit()

            if device != 'cpu':
                inputs = inputs.to(device)
                labels = labels.to(device)
                print('done putting into cuda')
            optim.zero_grad()
            outputs, _ = model(**inputs, labels=labels)
            loss, _ = outputs.loss, outputs.logits #outputs[:2]  # loss, logits, attentions
            print('At epoch:', epoch + 1, '/', epochs, '\tLoss:', loss)
            loss.backward()
            optim.step()

            if early_stopping != None:
                ### perform early stopping
                # check if valid_running_loss is decresing, if it increases more than patience, then break
                valid_running_loss += loss.item()
                total_loss.append(loss.item())
                # print('\tvalid_running_loss', valid_running_loss)
                average_valid_loss = valid_running_loss / len(total_loss)
                print('\taverage_valid_loss', average_valid_loss)
        if average_valid_loss != None and early_stopping.step(average_valid_loss):
            print('-----> performed early_stopping')
            print('-----> average_valid_loss:',average_valid_loss)
            break


    # save pre-trained model
    if model_name == 'BERT':
        path = './fine_tuned_model/BERTfinetune_tokenclassification_softmax/'
        model.save_pretrained(path)
        print('Saved find-tuned model to',path,'\n')
    elif model_name == 'BERT CRF':
        path = './fine_tuned_model/BERTfinetune_tokenclassification_CRF/'
        model.save_pretrained(path)
        print('Saved find-tuned model to', path, '\n')
    elif model_name == 'ROBERTA CRF':
        path = './fine_tuned_model/RoBERTafinetune_tokenclassification_CRF/'
        model.save_pretrained(path)
        print('Saved find-tuned model to', path, '\n')
    return model

def predictBIOES(model_name, test, label_num,limit=999999999, num_batch=8, MAX_LEN=128, model=None,CRF=False):
    if CRF==False:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('./fine_tuned_model/BERTfinetune_tokenclassification_softmax/', num_labels=label_num)
        print('Prediction --- Using fine-tuned BertForTokenClassification')
        softmax_output = True
    elif model_name=='BERT CRF' and CRF==True:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification_CRF.from_pretrained('./fine_tuned_model/BERTfinetune_tokenclassification_CRF/', num_labels=label_num)
        print('Prediction --- Using fine-tuned BertForTokenClassification_CRF')
        CRF_output = True
    elif model_name=='ROBERTA CRF' and CRF==True:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification_CRF.from_pretrained('./fine_tuned_model/RoBERTafinetune_tokenclassification_CRF/', num_labels=label_num)
        print('Prediction --- Using fine-tuned RoBERTafinetune_tokenclassification_CRF')
        CRF_output = True
    # elif model == None:
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #     model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=label_num)
    #     print('Prediction --- Using BertForTokenClassification')

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model = model.eval()

    test_input_batches = []
    for x in batch(test[:limit], num_batch):
        test_input_batches.append(list(x.values))

    collect_predictions = []
    for input_batch in test_input_batches:
        input_batch = [str(x) for x in input_batch]  # in case of 'nan'
        inputs = tokenizer(input_batch, return_tensors="pt", padding='max_length', truncation=True, max_length=MAX_LEN)

        if device != 'cpu':
            inputs = inputs.to(device)
            print('done putting into cuda')

        if CRF != True: # perform softmax
            output = model(**inputs)
            output = output[0].detach().numpy()
            output = torch.tensor(output)
            output_activation = F.softmax(output, dim=1)
            pred = combine_predictions(output_activation)
        else:
            output,output_activation = model(**inputs,return_crf=True) # perform CRF
            pred = output_activation
            print('performing CRF output_activation')
            # for t,p in zip(inputs.input_ids,pred):
            #     text = tokenizer.convert_ids_to_tokens(t)
                # print(text,p)
        collect_predictions.extend(pred)
    # print('collect_predictions',collect_predictions,len(collect_predictions))
    return collect_predictions

def evalBIOES(true,pred,label_num, limit,MAX_LEN=128):
    padded_true = []
    for e in true[:limit]:
        int_list = e
        int_list = (int_list + ['O'] * MAX_LEN)[:MAX_LEN]  # padding zero at the end,
        padded_true.append(int_list)
    true = list(np.concatenate(padded_true).flat)
    pred = covertLabels_to_BIOES(pred[:limit],label_num)
    pred = list(np.concatenate(pred).flat)
    if label_num == 2:
        labels = ['O', 'S']
    else:
        labels = ['B', 'I', 'O', 'E', 'S']
    print('Labels are', labels)
    print('Evaluation\tprecision_recall_fscore_support:')
    precision,recall,fscore,support = precision_recall_fscore_support(true,pred, average='macro')
    print('macro:',precision,recall,fscore,support)
    precision, recall, fscore, support = precision_recall_fscore_support(true,pred, average='micro')
    print('micro:', precision,recall,fscore,support)
    precision, recall, fscore, support = precision_recall_fscore_support(true,pred, average='weighted')
    print('weighted:', precision, recall, fscore, support)
    precision, recall, fscore, support = precision_recall_fscore_support(true,pred, average=None, labels=labels)
    print('None:', precision, recall, fscore, support)

    print('\nEvaluation\tclassification_report:')
    print(classification_report(true,pred, labels=labels))


def main():
    ### tokenize each instance: pre-processing: get token to match their label BIOE -- arrange into table/file: get text files as segmented sentences/paragraphs
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    path_nlp4if_train = './dataset/NLP4IF19/train/'
    path_nlp4if_dev = './dataset/NLP4IF19/dev/'
    path_nlp4if_test = './dataset/NLP4IF19/test/'
    nlp4if_label_tag = '.labels.tsv' #article111111112.labels.tsv

    path_semeval_train = './dataset/SemEval20/custom_train_dev/'
    semeval_label_tag = '.task-flc-tc.labels' #article111111111.task-flc-tc.labels


    ### pre-processing into trainable-dataset
    # # NLP4IF'19
    nlp4if_train = arrangeNERdataset(path_nlp4if_train,nlp4if_label_tag,custom_tokenizer=bert_tokenizer)
    nlp4if_dev = arrangeNERdataset(path_nlp4if_dev, nlp4if_label_tag, custom_tokenizer=bert_tokenizer)
    # # print('nlp4if_train+dev',nlp4if_train,nlp4if_dev)
    # nlp4if_combined_train = pd.concat([nlp4if_train,nlp4if_dev])
    # # print('nlp4if_combined_train',nlp4if_combined_train)


    #SEMEVAL'20T11
    semeval_train_dev = arrangeNERdataset(path_semeval_train, semeval_label_tag, custom_tokenizer=bert_tokenizer)
    # print(semeval_train_dev)
    # for t,l in zip(semeval_train_dev.text,semeval_train_dev.labels):
    #     print(t,l)
    # exit()

    combine_train = pd.concat([nlp4if_train,nlp4if_dev,semeval_train_dev])

    ### TEST only NLP4IF
    nlp4if_test = arrangeNERdataset(path_nlp4if_test, nlp4if_label_tag, custom_tokenizer=bert_tokenizer)

    print('combine_train:',combine_train)
    for text in combine_train.text:
        print(text)
    exit()



    # print('Done pre-processing files: len(nlp4if_train),len(nlp4if_test):',len(nlp4if_train),len(nlp4if_test))

    ### train
    ####### hyperparameters #######
    trained_model = None
    # model_name = 'BERT'
    model_name = 'BERT CRF'
    # model_name = 'ROBERTA CRF'
    limit = 5000000
    num_batch = 8
    MAX_LEN = 128
    epochs = 5
    label_num = 2
    # early_stopping = EarlyStopping(patience=3)


    # ### nlp4if
    # # train_text = nlp4if_combined_train.text
    # # train_labels = nlp4if_train.labels
    # ### semeval
    # train_text = semeval_train_dev.text
    # train_labels = semeval_train_dev.labels
    ### combine dataset NLP4IF+SemEval
    train_text = combine_train.text
    train_labels = combine_train.labels

    trained_model = trainBIOES(model_name, train_text, train_labels, label_num, limit, num_batch, MAX_LEN, epochs)#,early_stopping)

    ### predict
    pred = predictBIOES(model_name, nlp4if_test.text, label_num, limit, num_batch, MAX_LEN, model=trained_model,CRF=True) # model=None (for using pre-trained model without fine-tuning)

    ### evaluate
    evalBIOES(true=nlp4if_test.labels, pred=pred, label_num=label_num,limit=limit,MAX_LEN=MAX_LEN)
    print('Model summary:\n',model_name,'limit:',limit,'num_batch:',num_batch,'MAX_LEN:',MAX_LEN,'epochs:',epochs,'label_num:',label_num)



main()

