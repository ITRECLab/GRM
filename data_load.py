import numpy as np 
from torch.utils import data 
import torch 
from transformers import BertTokenizer
from transformers.data.processors.utils import InputExample
import re
import scipy
from utils import Annotation

sigma = 5
mu = 0.0

norm_0=scipy.stats.norm(mu,sigma).cdf(1)-scipy.stats.norm(mu,sigma).cdf(0)#norm高斯概率分布函数#目标实体旁边第一个单词的概率
norm_1=scipy.stats.norm(mu,sigma).cdf(2)-scipy.stats.norm(mu,sigma).cdf(1)#scipy.stats.norm(mu,sigma)高斯概率密度函数cdf:Cumulative distribution function.#目标实体旁边第二个单词的概率
norm_2=scipy.stats.norm(mu,sigma).cdf(3)-scipy.stats.norm(mu,sigma).cdf(2)
norm_3=scipy.stats.norm(mu,sigma).cdf(4)-scipy.stats.norm(mu,sigma).cdf(3)
norm_4=scipy.stats.norm(mu,sigma).cdf(5)-scipy.stats.norm(mu,sigma).cdf(4)
norm_5=scipy.stats.norm(mu,sigma).cdf(6)-scipy.stats.norm(mu,sigma).cdf(5)
norm_6=scipy.stats.norm(mu,sigma).cdf(7)-scipy.stats.norm(mu,sigma).cdf(6)


class NerDataset(data.Dataset):
    def __init__(self, path, task_name, pretrained_dir):
        self.VOCAB_DICT = {
            'bc5cdr': ('<PAD>', 'B-Chemical', 'O', 'B-Disease' , 'I-Disease', 'I-Chemical'),
            'bc2gm': ('<PAD>', 'B', 'I', 'O'),
            'bc6pm': ('<PAD>', 'B', 'I', 'O'),
            'bionlp3g' : ('<PAD>', 'B-Amino_acid', 'B-Anatomical_system', 'B-Cancer', 'B-Cell', 
                        'B-Cellular_component', 'B-Developing_anatomical_structure', 'B-Gene_or_gene_product', 
                        'B-Immaterial_anatomical_entity', 'B-Multi-tissue_structure', 'B-Organ', 'B-Organism', 
                        'B-Organism_subdivision', 'B-Organism_substance', 'B-Pathological_formation', 
                        'B-Simple_chemical', 'B-Tissue', 'I-Amino_acid', 'I-Anatomical_system', 'I-Cancer', 
                        'I-Cell', 'I-Cellular_component', 'I-Developing_anatomical_structure', 'I-Gene_or_gene_product', 
                        'I-Immaterial_anatomical_entity', 'I-Multi-tissue_structure', 'I-Organ', 'I-Organism', 
                        'I-Organism_subdivision', 'I-Organism_substance', 'I-Pathological_formation', 'I-Simple_chemical', 
                        'I-Tissue', 'O')
        }
        self.VOCAB = self.VOCAB_DICT[task_name.lower()]
        self.tag2idx = {v:k for k,v in enumerate(self.VOCAB)}
        self.idx2tag = {k:v for k,v in enumerate(self.VOCAB)}

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=False)
        instances = open(path).read().strip().split('\n\n')
        sents = []
        tags_li = []
        for entry in instances:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<PAD>"] + tags + ["<PAD>"])
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)


    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = self.tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [self.tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


    def pad(self, batch):
        '''Pads to the longest sample'''
        f = lambda x: [sample[x] for sample in batch]
        words = f(0)
        is_heads = f(2)
        tags = f(3)
        seqlens = f(-1)
        maxlen = np.array(seqlens).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
        x = f(1, maxlen)
        y = f(-2, maxlen)


        f = torch.LongTensor

        return words, f(x), is_heads, tags, f(y), seqlens


class RCDataSet(data.Dataset):
    label_map = {'0': 0, '1': 1}
    def __init__(self, documents, pretrained_dir, testData = False):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=False)
        self.examples = []
        self.label_1_count = 0
        self.label_0_count = 0
        self.testData = testData
        for i, document in enumerate(documents):
            self.examples.extend(self.__create_examples(
                *self.__train_document_process(document)))
        print("Examples_total_num {}.\tLabel_0_num: {}\tLabel_1_num: {}".format(
            len(self.examples), self.label_0_count, self.label_1_count))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __train_document_process(self, document):
        pmid = document['id']
        relations = set()
        genes = set()
        text_li = []
        passages = document['passages']
        split_word = re.compile(r"\w+|\S")
        for r in document['relations']:
            relations.add((r['infons']['Gene1'], r['infons']['Gene2']))

        for passage in passages:
            anns = passage['annotations']
            text = passage['text']
            offset_p = passage['offset']
            index = 0
            if len(anns) == 0:
                text_li.extend(split_word.findall(text))
            else:
                anns = Annotation.sortAnns(anns)
                for ann in anns:
                    if Annotation.gettype(ann) == 'Gene':
                        for infon_type in ann['infons']:
                            if infon_type.lower() == 'ncbi gene':
                                genes.add(ann['infons'][infon_type])
                    else:
                        continue
                    for i, location in enumerate(ann['locations']):
                        offset = location['offset']
                        length = location['length']
                        text_li.extend(split_word.findall(
                            text[index:offset-offset_p]))
                        if i == len(ann['locations']) - 1:
                            ncbi_gene_id = Annotation.getNCBIID(ann)
                            text_li.append("Gene_{}".format(ncbi_gene_id))
                        index = max(offset - offset_p + length, index)
                text_li.extend(split_word.findall(text[index:]))
        return pmid, text_li, genes, relations


    def __create_examples(self, pmid, text_li, genes, relations):
        examples = []
        text_li_ori = text_li
        guids = set()
        for g1 in genes:
            for g2 in genes:
                guid = f"{pmid}_{g1}_{g2}"
                if self.testData and f"{pmid}_{g2}_{g1}" in guids:
                    continue
                text_li = text_li_ori.copy()
                if (g1, g2) in relations or (g2, g1) in relations:
                    label = "1"
                    self.label_1_count += 1
                else:
                    label = "0"
                    self.label_0_count += 1
                g1_l = "Gene_S" if g1 == g2 else "Gene_A"
                g2_l = "Gene_S" if g1 == g2 else "Gene_B"
                for i, word in enumerate(text_li):
                    if word[:5] == "Gene_":
                        if word[5:] == g1:
                            text_li[i] = g1_l
                        elif word[5:] == g2:
                            text_li[i] = g2_l
                        else:
                            text_li[i] = "Gene_N"
                text_a = " ".join(text_li)
                if self.testData:
                    guids.add(guid)

                entity1_index = []
                entity2_index = []

                if "Gene_S" in text_li:
                    for i, word in enumerate(text_li):
                        if word == "Gene_S":
                            entity1_index.append(i)
                    P_array = np.zeros(512)
                    for i in entity1_index:
                        if i < 512:
                            P_array[i] = norm_0
                            if i + 1 < 512:
                                P_array[i + 1] = norm_0
                            if i + 2 < 512:
                                P_array[i + 2] = norm_1
                            if i + 3 < 512:
                                P_array[i + 3] = norm_2
                            if i + 4 < 512:
                                P_array[i + 4] = norm_3
                            if i + 5 < 512:
                                P_array[i + 5] = norm_4
                            if i + 6 < 512:
                                P_array[i + 6] = norm_5
                            if i + 7 < 512:
                                P_array[i + 7] = norm_6
                            if i - 1 >= 0:
                                P_array[i - 1] = norm_1
                            if i - 2 >= 0:
                                P_array[i - 2] = norm_2
                            if i - 3 >= 0:
                                P_array[i - 3] = norm_3
                            if i - 4 >= 0 and P_array[i - 4] == 0:
                                P_array[i - 4] = norm_4
                            if i - 5 >= 0 and P_array[i - 5] == 0:
                                P_array[i - 5] = norm_5
                            if i - 6 >= 0 and P_array[i - 6] == 0:
                                P_array[i - 6] = norm_6
                    P_gauss1_array = P_array
                    P_gauss2_array = P_array
                else:
                    for i, word in enumerate(text_li):
                        if word == "Gene_A":
                            entity1_index.append(i)
                        if word == "Gene_B":
                            entity2_index.append(i)
                    P_array1 = np.zeros(512)
                    for i in entity1_index:
                        if i < 512:
                            P_array1[i] = norm_0
                            if i + 1 < 512:
                                P_array1[i + 1] = norm_0
                            if i + 2 < 512:
                                P_array1[i + 2] = norm_1
                            if i + 3 < 512:
                                P_array1[i + 3] = norm_2
                            if i + 4 < 512:
                                P_array1[i + 4] = norm_3
                            if i + 5 < 512:
                                P_array1[i + 5] = norm_4
                            if i + 6 < 512:
                                P_array1[i + 6] = norm_5
                            if i + 7 < 512:
                                P_array1[i + 7] = norm_6
                            if i - 1 >= 0:
                                P_array1[i - 1] = norm_1
                            if i - 2 >= 0:
                                P_array1[i - 2] = norm_2
                            if i - 3 >= 0:
                                P_array1[i - 3] = norm_3
                            if i - 4 >= 0 and P_array1[i - 4] == 0:
                                P_array1[i - 4] = norm_4
                            if i - 5 >= 0 and P_array1[i - 5] == 0:
                                P_array1[i - 5] = norm_5
                            if i - 6 >= 0 and P_array1[i - 6] == 0:
                                P_array1[i - 6] = norm_6
                    P_array2 = np.zeros(512)
                    for i in entity2_index:
                        if i < 512:
                            P_array2[i] = norm_0
                            if i + 1 < 512:
                                P_array2[i + 1] = norm_0
                            if i + 2 < 512:
                                P_array2[i + 2] = norm_1
                            if i + 3 < 512:
                                P_array2[i + 3] = norm_2
                            if i + 4 < 512:
                                P_array2[i + 4] = norm_3
                            if i + 5 < 512:
                                P_array2[i + 5] = norm_4
                            if i + 6 < 512:
                                P_array2[i + 6] = norm_5
                            if i + 7 < 512:
                                P_array2[i + 7] = norm_6
                            if i - 1 >= 0:
                                P_array2[i - 1] = norm_1
                            if i - 2 >= 0:
                                P_array2[i - 2] = norm_2
                            if i - 3 >= 0:
                                P_array2[i - 3] = norm_3
                            if i - 4 >= 0 and P_array2[i - 4] == 0:
                                P_array2[i - 4] = norm_4
                            if i - 5 >= 0 and P_array2[i - 5] == 0:
                                P_array2[i - 5] = norm_5
                            if i - 6 >= 0 and P_array2[i - 6] == 0:
                                P_array2[i - 6] = norm_6
                    P_gauss1_array = P_array1
                    P_gauss2_array = P_array2
                P_gauss1_list=list(P_gauss1_array)
                P_gauss2_list=list(P_gauss2_array)
                examples.append(
                    InputExample(guid= guid, text_a=text_a, text_b=None, label=label, P_gauss1_list=P_gauss1_list, P_gauss2_list=P_gauss2_list))#
                print(examples)
        return examples 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203
21712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284_6446_27254', text_a='Identification of Gene_N ( Gene_N ) a
s a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extracts , and was purified and identified as Gene_N ( calcium
- regulated heat - stable protein of apparent molecula
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203
21712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284_6446_27254', text_a='Identification of Gene_N ( Gene_N ) a
s a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extracts , and was purified and identified as Gene_N ( calcium
- regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometr
ically at Ser52 in vitro and its brain - specific isoform Gene_B at the equivalent residue ( Ser58 ) .
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203
21712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284_6446_27254', text_a='Identification of Gene_N ( Gene_N ) a
s a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extracts , and was purified and identified as Gene_N ( calcium
- regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometr
ically at Ser52 in vitro and its brain - specific isoform Gene_B at the equivalent residue ( Ser58 ) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) ce
lls were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but not by rapamycin [ an inhi
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203
21712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284_6446_27254', text_a='Identification of Gene_N ( Gene_N ) a
s a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extracts , and was purified and identified as Gene_N ( calcium
- regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometr
ically at Ser52 in vitro and its brain - specific isoform Gene_B at the equivalent residue ( Ser58 ) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) ce
lls were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or P
D 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in E
S ( embryonic stem ) cells from
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cell
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase )
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - acti
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L15
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in res
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK casc
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gen
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.053200144651
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.079259709439
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.03
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623
[InputExample(guid='15910284_6446_6446', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N )
 was detected in liver extracts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_S
( Gene_S ) and Gene_N ( p90 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58
) . Gene_N became phosphorylated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoin
ositide 3 - kinase ) , but not by rapamycin [ an inhibitor of Gene_N ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and he
nce the activation of Gene_N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant
( Gene_N [ L155E ] ) that activates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was
 prevented by blocking activation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylate
d Gene_N at Ser30 , Ser32 and Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by
 Gene_N in response to Gene_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein ki
nase ( s ) .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.0532001446
5193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514
063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943
910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='15910284
_6446_21977', text_a='Identification of Gene_N ( Gene_N ) as a physiological substrate for Gene_N and Gene_N using KESTREL . A substrate for Gene_N ( Gene_N ) was detected in liver extr
acts , and was purified and identified as Gene_N ( calcium - regulated heat - stable protein of apparent molecular mass 24 kDa ) . Gene_N , as well as Gene_A ( Gene_A ) and Gene_N ( p90
 ribosomal S6 kinase ) , phosphorylated Gene_N stoichiometrically at Ser52 in vitro and its brain - specific isoform Gene_N at the equivalent residue ( Ser58 ) . Gene_N became phosphory
lated at Ser52 when HEK - 293 ( human embryonic kidney ) cells were stimulated with Gene_N ( Gene_N ) and this was prevented by inhibitors of PI3K ( phosphoinositide 3 - kinase ) , but
not by rapamycin [ an inhibitor of Gene_B ( Gene_N ) ] or PD 184352 , an inhibitor of the classical MAPK ( mitogen - activated protein kinase ) cascade and hence the activation of Gene_
N . Gene_N induced a similar phosphorylation of Gene_N in ES ( embryonic stem ) cells from wild - type mice or mice that express the Gene_N ( Gene_N ) mutant ( Gene_N [ L155E ] ) that a
ctivates Gene_N normally , but cannot activate Gene_N . Gene_N also became phosphorylated at Ser52 in response to EGF ( epidermal growth factor ) and this was prevented by blocking acti
vation of both the classical MAPK cascade and the activation of Gene_N , but not if just one of these pathways was inhibited . Gene_N ( Gene_N ) phosphorylated Gene_N at Ser30 , Ser32 a
nd Ser41 in vitro , and Ser41 was identified as a site phosphorylated in cells . These and other results demonstrate that Gene_N is phosphorylated at Ser52 by Gene_N in response to Gene
_N , at Ser52 by Gene_N and Gene_N in response to EGF , and at Ser41 in the absence of Gene_N / EGF by a DYRK isoform or another proline - directed protein kinase ( s ) .', text_b=None,
 label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667
691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203
21712212, 0.
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='9188101_3683_3687', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - as
sociated antigen 1 . Gene_N ( Gene_N , Gene_N ) is a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_A ( Gene_A , Gene_A / Gene_N ) , complement re
ceptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 , 95 ( Gene_B / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory respo
nses . In this report , we describe a cell - free assay using purified recombinant extracellular domains of Gene_A and a dimeric immunoadhesin of Gene_N . The binding of recombinant sec
reted Gene_A to Gene_N is divalent cation dependent ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_A - mediated cell adhesion , indicating
 that its conformation mimics that of Gene_A on activated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five po
int mutants of the Gene_N immunoadhesin were generated and residues important for binding of monoclonal antibodies and purified Gene_A were identified . Nineteen of these mutants bind r
ecombinant Gene_A equivalently to wild type . Sixteen mutants show a 66 - 2500 - fold decrease in Gene_A binding yet , with few exceptions , retain binding to the monoclonal antibodies
. These mutants , along with modeling studies , define the Gene_A binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG bet
a - sheet of the Ig fold . The mutant G32A also abrogates binding to Gene_A while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue w
ith Gene_A . These data have allowed the generation of a highly refined model of the Gene_A binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854
, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0
761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974
8854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227,
0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130109879371
54, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792597094391
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='9188101_3683_3687', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - as
sociated antigen 1 . Gene_N ( Gene_N , Gene_N ) is a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_A ( Gene_A , Gene_A / Gene_N ) , complement re
ceptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 , 95 ( Gene_B / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory respo
nses . In this report , we describe a cell - free assay using purified recombinant extracellular domains of Gene_A and a dimeric immunoadhesin of Gene_N . The binding of recombinant sec
reted Gene_A to Gene_N is divalent cation dependent ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_A - mediated cell adhesion , indicating
 that its conformation mimics that of Gene_A on activated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five po
int mutants of the Gene_N immunoadhesin were gene
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='9188101_3683_3687', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - as
sociated antigen 1 . Gene_N ( Gene_N , Gene_N ) is a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_A ( Gene_A , Gene_A / Gene_N ) , complement re
ceptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 , 95 ( Gene_B / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory respo
nses . In this report , we describe a cell - free assay using purified recombinant extracellular domains of Gene_A and a dimeric immunoadhesin of Gene_N . The binding of recombinant sec
reted Gene_A to Gene_N is divalent cation dependent ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_A - mediated cell adhesion , indicating
 that its conformation mimics that of Gene_A on activated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five po
int mutants of the Gene_N immunoadhesin were generated and residues important for binding of monoclonal antibodies and purified Gene_A were identified . Nineteen of these mutants bind r
ecombinant Gene_A equivalently to wild type . Sixteen mutants show a 66 - 2500 - fold decrease in Gene_A binding yet , with few exceptions , retain binding to the monoclonal antibodies
. These mutants , along with modeling studies , define the Gene_A binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG bet
a - sheet of the Ig fold . The mutant G32A also abrogates binding to Gene_A while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue w
ith Gene_A . These data have allowed the generation of a highly refined model of the Gene_A binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854
, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0
761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974
8854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227,
0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053200144651939
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='9188101_3683_3687', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - as
sociated antigen 1 . Gene_N ( Gene_N , Gene_N ) is a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_A ( Gene_A , Gene_A / Gene_N ) , complement re
ceptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 , 95 ( Gene_B / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory respo
nses . In this report , we describe a cell - free assay using purified recombinant extracellular domains of Gene_A and a dimeric immunoadhesin of Gene_N . The binding of recombinant sec
reted Gene_A to Gene_N is divalent cation dependent ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_A - mediated cell adhesion , indic
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='9188101_3683_3687', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - as
sociated antigen 1 . Gene_N ( Gene_N , Gene_N ) is a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_A ( Gene_A , Gene_A / Gene_N ) , complement re
ceptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 , 95 ( Gene_B / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory respo
nses . In this report , we describe a cell - free assay using purified recombinant extracellular domains of Gene_A and a dimeric immunoadhesin of Gene_N . The binding of recombinant sec
reted Gene_A to Gene_N is divalent cation dependent ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_A - mediated cell adhesion , indicating
 that its conformation mimics that of Gene_A on activated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five po
int mutants of the Gene_N immunoadhesin were generated and residues important for binding of monoclonal antibodies and purified Gene_A were identified . Nineteen of these mutants bind r
ecombinant Gene_A equivalently to wild type . Sixteen mutants show a 66 - 2500 - fold decrease in Gene_A binding yet , with few exceptions , retain binding to the monoclonal antibodies
. These mutants , along with modeling studies , define the Gene_A binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG bet
a - sheet of the Ig fold . The mutant G32A also abrogates binding to Gene_A while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue w
ith Gene_A .
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='9188101_3683_3687', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - as
sociated antigen 1 . Gene_N ( Gene_N , Gene_N ) is a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_A ( Gene_A , Gene_A / Gene_N ) , complement re
ceptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 , 95 ( Gene_B / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory respo
nses . In this report , we describe a cell - free assay using purified recombinant extracellular domains of Gene_A and a dimeric immunoadhesin of Gene_N . The binding of recombinant sec
reted Gene_A to Gene_N is divalent cation dependent ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_A - mediated cell adhesion , indicating
 that its conformation mimics that of Gene_A on activated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five po
int mutants of the Gene_N immunoadhesin were generated and residues important for binding of monoclonal antibodies and purified Gene_A were identified . Nineteen of these mutants bind r
ecombinant Gene_A equivalently to wild type . Sixteen mutants show a 66 - 2500 - fold decrease in Gene_A binding yet , with few exceptions , retain binding to the monoclonal antibodies
. These mutants , along with modeling studies , define the Gene_A binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG bet
a - sheet of the Ig fold . The mutant G32A also abrogates binding to Gene_A while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue w
ith Gene_A . These data have allowed the generation of a highly refined model of the Gene_A binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854
, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='9188101_3683_3687', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - as
sociated antigen 1 . Gene_N ( Gene_N , Gene_N ) is a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_A ( Gene_A , Gene_A / Gene_N ) , complement re
ceptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 , 95 ( Gene_B / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory respo
nses . In this report , we describe a cell - free assay using purified recombinant extracellular domains of Gene_A and a dimeric immunoadhesin of Gene_N . The binding of recombinant sec
reted Gene_A to Gene_N is divalent cation dependent ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_A - mediated cell adhesion , indicating
 that its conformation mimics that of Gene_A on activated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five po
int mutants of the Gene_N immunoadhesin were generated and residues important for binding of monoclonal antibodies and purified Gene_A were identified . Nineteen of these mutants bind r
ecombinant Gene_A equivalently to wild type . Sixteen mutants show a 66 - 2500 - fold decrease in Gene_A binding yet , with few exceptions , retain binding to the monoclonal antibodies
. These mutants , along with modeling studies , define the Gene_A binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG bet
a - sheet of the Ig fold . The mutant G32A also abrogates binding to Gene_A while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue w
ith Gene_A . These data have allowed the generation of a highly refined model of the Gene_A binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854
, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0
761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0623977191666769
1, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974
8854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227,
0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130109879371
54, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.062
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='9188101_3683_3687', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - as
sociated antigen 1 . Gene_N ( Gene_N , Gene_N ) is a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_A ( Gene_A , Gene_A / Gene_N ) , complement re
ceptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 , 95 ( Gene_B / Ge
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='9188101_3683_3687', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - as
sociated antigen 1 . Gene_N ( Gene_N , Gene_N ) is a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_A ( Gene_A , Gene_A / Gene_N ) , complement re
ceptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 , 95 ( Gene_B / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory respo
nses . In this report , we describe a cell - free assay using purified recombinant extracellular domains of Gene_A and a dimeric immunoadhesin of Gene_N . The binding of recombinant sec
reted Gene_A to Gene_N is divalent cation dependent ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_A - mediated cell adhesion , indicating
 that its conformation mimics that of Gene_A on activated lymphocytes . We describe six novel
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='9188101_3683_3687', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - as
sociated antigen 1 . Gene_N ( Gene_N , Gene_N ) is a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_A ( Gene_A , Gene_A / Gene_N ) , complement re
ceptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 , 95 ( Gene_B / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory respo
nses . In this report , we describe a cell - free assay using purified recombinant extracellular domains of Gene_A and a dimeric immunoadhesin of Gene_N . The binding of recombinant sec
reted Gene_A to Gene_N is divalent cation dependent ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_A - mediated cell adhesion , indicating
 that its conformation mimics that of Gene_A on activated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five po
int mutants of the Gene_N immunoadhesin were generated and residues important for binding of monoclonal antibodies and purified Gene_A were identified . Nineteen of these mutants bind r
ecombinant Gene_A equivalently to wild type . Sixteen mutants show a 66 - 2500 - fold decrease in Gene_A binding yet , with few exceptions , retain binding to the monoclonal antibodies
. These mutants , along with modeling studies , define the Gene_A binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q7
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='9188101_3683_3687', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - as
sociated antigen 1 . Gene_N ( Gene_N , Gene_N ) is a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_A ( Gene_A , Gene_A / Gene_N ) , complement re
ceptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 , 95 ( Gene_B / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory respo
nses . In this report , we describe a cell - free assay using purified recombinant extracellular domains of Gene_A and a dimeric immunoadhesin of Gene_N . The binding of recombinant sec
reted Gene_A to Gene_N is divalent cation dependent ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_A - mediated cell adhesion , indicating
 that its conformation mimics that of Gene_A on activated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five po
int mutants of the Gene_N immunoadhesin were generated and residues important for binding of monoclonal antibodies and purified Gene_A were identified . Nineteen of these mutants bind r
ecombinant Gene_A equivalently to wild type . Sixteen mutants show a 66 - 2500 - fold decrease in Gene_A binding yet , with few exceptions , retain binding to the monoclonal antibodies
. These mutants , along with modeling studies , define the Gene_A binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG bet
a - sheet of the Ig fold . The mutant G32A also abrogates binding to Gene_A while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue w
ith Gene_A . These data have allowed the generation of a highly refined model of the Gene_A binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='9188101_3683_3687', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - as
sociated antigen 1 . Gene_N ( Gene_N , Gene_N ) is a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_A ( Gene_A , Gene_A / Gene_N ) , complement re
ceptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 , 95 ( Gene_B / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory respo
nses . In this report , we describe a cell - free assay using purified recombinant extracellular domains of Gene_A and a dimeric immunoadhesin of Gene_N . The binding of recombinant sec
reted Gene_A to Gene_N is divalent cation dependent ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_A - mediated cell adhesion , indicating
 that its conformation mimics that of Gene_A on activated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five po
int mutants of the Gene_N immunoadhesin were generated and residues important for binding of monoclonal antibodies and purified Gene_A were identified . Nineteen of these mutants bind r
ecombinant Gene_A equivalently to wild type . Sixteen mutants show a 66 - 2500 - fold decrease in Gene_A binding yet , with few exceptions , retain binding to the monoclonal antibodies
. These mutants , along with modeling studies , define the Gene_A binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG bet
a - sheet of the Ig fold . The mutant G32A also abrogates binding to Gene_A while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue w
ith Gene_A . These data have allowed the generation of a highly refined model of the Gene_A binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854
, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
[InputExample(guid='9188101_3683_3683', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - associated antigen 1 . Gene_N ( Gene_N , Gene_N ) is
 a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_S ( Gene_S , Gene_S / Gene_N ) , complement receptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 ,
 95 ( Gene_N / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory responses . In this report , we describe a cell - free
assay using purified recombinant extracellular domains of Gene_S and a dimeric immunoadhesin of Gene_N . The binding of recombinant secreted Gene_S to Gene_N is divalent cation dependen
t ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_S - mediated cell adhesion , indicating that its conformation mimics that of Gene_S on ac
tivated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five point mutants of the Gene_N immunoadhesin were gener
ated and residues important for binding of monoclonal antibodies and purified Gene_S were identified . Nineteen of these mutants bind recombinant Gene_S equivalently to wild type . Sixt
een mutants show a 66 - 2500 - fold decrease in Gene_S binding yet , with few exceptions , retain binding to the monoclonal antibodies . These mutants , along with modeling studies , de
fine the Gene_S binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG beta - sheet of the Ig fold . The mutant G32A also ab
rogates binding to Gene_S while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue with Gene_S . These data have allowed the generatio
n of a highly refined model of the Gene_S binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703
2514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07
032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.0435855837097
48854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792
5970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.0
6239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0
.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970
943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320
014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053
20014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358
5583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06
239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0703251406
3960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.0343130
10987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154,
 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0
7925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.0532001446519395
6, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193
956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691,
0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974885
4, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203217
12212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07
61620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.06239771916667691, 0
.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854
, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.079259709439102
99, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325
14063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034
313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.034313010987937154, 0.043
585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.
06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='9188101_3683_3687', text_a='Identification of the binding site in Gene_N for its receptor , leukocyte function - as
sociated antigen 1 . Gene_N ( Gene_N , Gene_N ) is a member of the Ig superfamily and is a counterreceptor for the Gene_N integrins : Gene_A ( Gene_A , Gene_A / Gene_N ) , complement re
ceptor 1 ( Gene_N , Gene_N / Gene_N ) , and p150 , 95 ( Gene_B / Gene_N ) . Binding of Gene_N to these receptors mediates leukocyte - adhesive functions in immune and inflammatory respo
nses . In this report , we describe a cell - free assay using purified recombinant extracellular domains of Gene_A and a dimeric immunoadhesin of Gene_N . The binding of recombinant sec
reted Gene_A to Gene_N is divalent cation dependent ( Mg2 + and Mn2 + promote binding ) and sensitive to inhibition by antibodies that block Gene_A - mediated cell adhesion , indicating
 that its conformation mimics that of Gene_A on activated lymphocytes . We describe six novel anti - Gene_N monoclonal antibodies , two of which are function blocking . Thirty - five po
int mutants of the Gene_N immunoadhesin were generated and residues important for binding of monoclonal antibodies and purified Gene_A were identified . Nineteen of these mutants bind r
ecombinant Gene_A equivalently to wild type . Sixteen mutants show a 66 - 2500 - fold decrease in Gene_A binding yet , with few exceptions , retain binding to the monoclonal antibodies
. These mutants , along with modeling studies , define the Gene_A binding site on Gene_N as residues E34 , Gene_N , M64 , Y66 , N68 , and Q73 , that are predicted to lie on the CDFG bet
a - sheet of the Ig fold . The mutant G32A also abrogates binding to Gene_A while retaining binding to all of the antibodies , possibly indicating a direct interaction of this residue w
ith Gene_A . These data have allowed the generation of a highly refined model of the Gene_A binding site of Gene_N .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854
, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910
299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0
761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709
[InputExample(guid='14517323_5566_5566', text_a='The retrieval function of the Gene_N requires PKA phosphorylation of its C - terminus . The Gene_N is a Golgi / intermediate compartment
 - located integral membrane protein that carries out the retrieval of escaped ER proteins bearing a C - terminal KDEL sequence . This occurs throughout retrograde traffic mediated by C
OPI - coated transport carriers . The role of the C - terminal cytoplasmic domain of the Gene_N in this process has been investigated . Deletion of this domain did not affect receptor s
ubcellular localization although cells expressing this truncated form of the receptor failed to retain KDEL ligands intracellularly . Permeabilized cells incubated with ATP and GTP exhi
bited tubular processes - mediated redistribution from the Golgi area to the ER of the wild - type receptor , whereas the truncated form lacking the C - terminal domain remained concent
rated in the Golgi . As revealed with a peptide - binding assay , this domain did not interact with both coatomer and ARF - GAP unless serine 209 was mutated to aspartic acid . In contr
ast , alanine replacement of serine 209 inhibited coatomer / ARF - GAP recruitment , receptor redistribution into the ER , and intracellular retention of KDEL ligands . Serine 209 was p
hosphorylated by both cytosolic and recombinant Gene_S . Inhibition of endogenous PKA activity with H89 blocked Golgi - ER transport of the native receptor but did not affect redistribu
tion to the ER of a mutated form bearing aspartic acid at position 209 . We conclude that PKA phosphorylation of serine 209 is required for the retrograde transport of the Gene_N from t
he Golgi complex to the ER from which the retrieval of proteins bearing the KDEL signal depends .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620
321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0
7032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854,
0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
[InputExample(guid='14517323_5566_5566', text_a='The retrieval function of the Gene_N requires PKA phosphorylation of its C - terminus . The Gene_N is a Golgi / intermediate compartment
 - located integral membrane protein that carries out the retrieval of escaped ER proteins bearing a C - terminal KDEL sequence . This occurs throughout retrograde traffic mediated by C
OPI - coated transport carriers . The role of the C - terminal cytoplasmic domain of the Gene_N in this process has been investigated . Deletion of this domain did not affect receptor s
ubcellular localization although cells expressing this truncated form of the receptor failed to retain KDEL ligands intracellularly . Permeabilized cells incubated with ATP and GTP exhi
bited tubular processes - mediated redistribution from the Golgi area to the ER of the wild - type receptor , whereas the truncated form lacking the C - terminal domain remained concent
rated in the Golgi . As revealed with a peptide - binding assay , this domain did not interact with both coatomer and ARF - GAP unless serine 209 was mutated to aspartic acid . In contr
ast , alanine replacement of serine 209 inhibited coatomer / ARF - GAP recruitment , receptor redistribution into the ER , and intracellular retention of KDEL ligands . Serine 209 was p
hosphorylated by both cytosolic and recombinant Gene_S . Inhibition of endogenous PKA activity with H89 blocked Golgi - ER transport of the native receptor but did not affect redistribu
tion to the ER of a mutated form bearing aspartic acid at position 209 . We conclude that PKA phosphorylation of serine 209 is required for the retrograde transport of the Gene_N from t
he Golgi complex to the ER from which the retrieval of proteins bearing the KDEL signal depends .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620
321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0
7032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854,
0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputE
xample(guid='14517323_5566_10945', text_a='The retrieval function of the Gene_B requires PKA phosphorylation of its C - terminus . The Gene_B is a Golgi / intermediate compartment - loc
ated integral membrane protein that carries out the retrieval of escaped ER proteins bearing a C - terminal KDEL sequence . This occurs throughout retrograde traffic mediated by COPI -
coated transport carriers . The role of the C - terminal cytoplasmic domain of the Gene_B in this process has been investigated . Deletion of this domain did not affect receptor subcell
ular localization although cells expressing this truncated form of the receptor failed to retain KDEL ligands intracellularly . Permeabilized cells incubated with ATP and GTP exhibited
tubular processes - mediated redistribution from the Golgi area to the ER of the wild - type receptor , whereas the truncated form lacking the C - terminal domain remained concentrated
in the Golgi . As revealed with a peptide - binding assay , this domain did not interact with both coatomer and ARF - GAP unless serine 209 was mutated to aspartic acid . In contrast ,
alanine replacement of serine 209 inhibited coatomer / ARF - GAP recruitment , receptor redistribution into the ER , and intracellular retention of KDEL ligands . Serine 209 was phospho
rylated by both cytosolic and recombinant Gene_A . Inhibition of endogenous PKA activity with H89 blocked Golgi - ER transport of the native receptor but did not affect redistribution t
o the ER of a mutated form bearing aspartic acid at position 209 . We conclude that PKA phosphorylation of serine 209 is required for the retrograde transport of the Gene_B from the Gol
gi complex to the ER from which the retrieval of proteins bearing the KDEL signal depends .', text_b=None, label='1', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.043585583709748854, 0.0
5320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0
.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.076162032171221
2, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956,
0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956
, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.079259709439102
99, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
[InputExample(guid='14517323_5566_5566', text_a='The retrieval function of the Gene_N requires PKA phosphorylation of its C - terminus . The Gene_N is a Golgi / intermediate compartment
 - located integral membrane protein that carries out the retrieval of escaped ER proteins bearing a C - terminal KDEL sequence . This occurs throughout retrograde traffic mediated by C
OPI - coated transport carriers . The role of the C - terminal cytoplasmic domain of the Gene_N in this process has been investigated . Deletion of this domain did not affect receptor s
ubcellular localization although cells expressing this truncated form of the receptor failed to retain KDEL ligands intracellularly . Permeabilized cells incubated with ATP and GTP exhi
bited tubular processes - mediated redistribution from the Golgi area to the ER of the wild - type receptor , whereas the truncated form lacking the C - terminal domain remained concent
rated in the Golgi . As revealed with a peptide - binding assay , this domain did not interact with both coatomer and ARF - GAP unless serine 209 was mutated to aspartic acid . In contr
ast , alanine replacement of serine 209 inhibited coatomer / ARF - GAP recruitment , receptor redistribution into the ER , and intracellular retention of KDEL ligands . Serine 209 was p
hosphorylated by both cytosolic and recombinant Gene_S . Inhibition of endogenous PKA activity with H89 blocked Golgi - ER transport of the native receptor but did not affect redistribu
tion to the ER of a mutated form bearing aspartic acid at position 209 . We conclude that PKA phosphorylation of serine 209 is required for the retrograde transport of the Gene_N from t
he Golgi complex to the ER from which the retrieval of proteins bearing the KDEL signal depends .', text_b=None, label='0', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620
321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.0
7032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854,
0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputE
xample(guid='14517323_5566_10945', text_a='The retrieval function of the Gene_B requires PKA phosphorylation of its C - terminus . The Gene_B is a Golgi / intermediate compartment - loc
ated integral membrane protein that carries out the retrieval of escaped ER proteins bearing a C - terminal KDEL sequence . This occurs throughout retrograde traffic mediated by COPI -
coated transport carriers . The role of the C - terminal cytoplasmic domain of the Gene_B in this process has been investigated . Deletion of this domain did not affect receptor subcell
ular localization although cells expressing this truncated form of the receptor failed to retain KDEL ligands intracellularly . Permeabilized cells incubated with ATP and GTP exhibited
tubular processes - mediated redistribution from the Golgi area to the ER of the wild - type receptor , whereas the truncated form lacking the C - terminal domain remained concentrated
in the Golgi . As revealed with a peptide - binding assay , this domain did not interact with both coatomer and ARF - GAP unless serine 209 was mutated to aspartic acid . In contrast ,
alanine replacement of serine 209 inhibited coatomer / ARF - GAP recruitment , receptor redistribution into the ER , and intracellular retention of KDEL ligands . Serine 209 was phospho
rylated by both cytosolic and recombinant Gene_A . Inhibition of endogenous PKA activity with H89 blocked Golgi - ER transport of the native receptor but did not affect redistribution t
o the ER of a mutated form bearing aspartic acid at position 209 . We conclude that PKA phosphorylation of serine 209 is required for the retrograde transport of the Gene_B from the Gol
gi complex to the ER from which the retrieval of proteins bearing the KDEL signal depends .', text_b=None, label='1', P_gauss1_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712
212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.04358
[InputExample(guid='27906179_65018_65018', text_a="Gene_S - dependent phosphorylation of Gene_S and Parkin is essential for mitochondrial quality control . Mitochondrial dysfunction has
 been linked to the pathogenesis of a large number of inherited diseases in humans , including Parkinson ' s disease , the second most common neurodegenerative disorder . The Parkinson
' s disease genes Gene_S and parkin , which encode a mitochondrially targeted protein kinase , and an E3 ubiquitin ligase , respectively , participate in a key mitochondrial quality - c
ontrol pathway that eliminates damaged mitochondria . In the current study , we established an in vivo Gene_S / Parkin - induced photoreceptor neuron degeneration model in Drosophila wi
th the aim of dissecting the Gene_S / Parkin pathway in detail . Using LC - MS / MS analysis , we identified Serine 346 as the sole autophosphorylation site of Drosophila Gene_S and fou
nd that substitution of Serine 346 to Alanine completely abolished the Gene_S autophosphorylation . Disruption of either Gene_S or Parkin phosphorylation impaired the Gene_S / Parkin pa
thway , and the degeneration phenotype of photoreceptor neurons was obviously alleviated . Phosphorylation of Gene_S is not only required for the Gene_S - mediated mitochondrial recruit
ment of Parkin but also induces its kinase activity toward Parkin . In contrast , phosphorylation of Parkin by Gene_S is dispensable for its translocation but required for its activatio
n . Moreover , substitution with autophosphorylation - deficient Gene_S failed to rescue Gene_S null mutant phenotypes . Taken together , our findings suggest that autophosphorylation o
f Gene_S is essential for the mitochondrial translocation of Parkin and for subsequent phosphorylation and activation of Parkin .", text_b=None, label='0', P_gauss1_list=[0.079259709439
10299, 0.07925970943910299, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.0623977191
6667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07
925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.053200144
65193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.0532001
4465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07616
20321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0
.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748
854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.06239771916667691, 0.070325140639
60227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943
910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585
583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.062
39771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.0
43585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063
960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.03431301
0987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299,
0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956
, 0.043585583709748854, 0.034313010987937154, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.0792597094
3910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.07925970943910299, 0.079259709
43910299, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.0532001
4465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0
.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.062397
71916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.0435
85583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.079
25970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299
, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.043585583709748854, 0.05320014465
193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.06239771916667691, 0.07032514063960227, 0.076162032
1712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07616203
21712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05
320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.
07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854,
 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.07616203
21712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0623977191666769
1, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358558370974
8854, 0.034313010987937154, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597
0943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
[InputExample(guid='11821392_2822_2822', text_a="Gene_N interacts with Gene_S isozymes and inhibits pervanadate - induced Gene_S activation in human embryonic kidney - 293 cells . Gene_
N has been implicated in the pathogenesis of many neurodegenerative diseases , including Parkinson ' s disease and Alzheimer ' s disease . Although the function of Gene_N remains largel
y unknown , recent studies have demonstrated that this protein can interact with phospholipids . To address the role of Gene_N in neurodegenerative disease , we have investigated whethe
r it binds Gene_S ( Gene_S ) and affects Gene_S activity in human embryonic kidney ( HEK ) - 293 cells overexpressing wild type Gene_N or the mutant forms of Gene_N ( A53T , A30P ) asso
ciated with Parkinson ' s disease . Tyrosine phosphorylation of Gene_N appears to play a modulatory role in the inhibition of Gene_S , because mutation of Tyr ( 125 ) to Phe slightly in
creases inhibitory effect of Gene_N on Gene_S activity . Treatment with pervanadate or phorbol myristate acetate inhibits Gene_S more in HEK 293 cells overexpressing Gene_N than in cont
rol cells . Binding of Gene_N to Gene_S requires phox and pleckstrin homology domain of Gene_S and the amphipathic repeat region and non - Abeta component of Gene_N . Although biologica
lly important , co - transfection studies indicate that the interaction of Gene_N with Gene_S does not influence the tendency of Gene_N to form pathological inclusions . These results s
uggest that the association of Gene_N with Gene_S , and modulation of Gene_S activity , is biologically important , but Gene_S does not appear to play an essential role in the pathophys
iology of Gene_N .", text_b=None, label='0', P_gauss1_list=[0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0
.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691,
 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.
043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.06239771916667691
, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748
854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07
61620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854
, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.062397719166676
91, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321
712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.053200144
65193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.0623977
1916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358
5583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.062397
71916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.0435
85583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227,
0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212,
 0.07032514063960227, 0.06239771916667691, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.0703251406396022
7, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.076162032
1712212, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771
916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987
937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0623977
1916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358
5583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597094391
0299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.
0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0
.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691,
 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.0532001446519395
6, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956
, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053200144651939
56, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325140
63960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620
321712212, 0.07032514063960227, 0.06239771916667691, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.070325
14063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
[InputExample(guid='11821392_2822_2822', text_a="Gene_N interacts with Gene_S isozymes and inhibits pervanadate - induced Gene_S activation in human embryonic kidney - 293 cells . Gene_
N has been implicated in the pathogenesis of many neurodegenerative diseases , including Parkinson ' s disease and Alzheimer ' s disease . Although the function of Gene_N remains largel
y unknown , recent studies have demonstrated that this protein can interact with phospholipids . To address the role of Gene_N in neurodegenerative disease , we have investigated whethe
r it binds Gene_S ( Gene_S ) and affects Gene_S activity in human embryonic kidney ( HEK ) - 293 cells overexpressing wild type Gene_N or the mutant forms of Gene_N ( A53T , A30P ) asso
ciated with Parkinson ' s disease . Tyrosine phosphorylation of Gene_N appears to play a modulatory role in the inhibition of Gene_S , because mutation of Tyr ( 125 ) to Phe slightly in
creases inhibitory effect of Gene_N on Gene_S activity . Treatment with pervanadate or phorbol myristate acetate inhibits Gene_S more in HEK 293 cells overexpressing Gene_N than in cont
rol cells . Binding of Gene_N to Gene_S requires phox and pleckstrin homology domain of Gene_S and the amphipathic repeat region and non - Abeta component of Gene_N . Although biologica
lly important , co - transfection studies indicate that the interaction of Gene_N with Gene_S does not influence the tendency of Gene_N to form pathological inclusions . These results s
uggest that the association of Gene_N with Gene_S , and modulation of Gene_S activity , is biologically important , but Gene_S does not appear to play an essential role in the pathophys
iology of Gene_N .", text_b=None, label='0', P_gauss1_list=[0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0
.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691,
 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.
043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.06239771916667691
, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748
854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07
61620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854
, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.062397719166676
91, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321
712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.053200144
65193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.0623977
1916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358
5583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.062397
71916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.0435
85583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227,
0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212,
 0.07032514063960227, 0.06239771916667691, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.0703251406396022
7, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.076162032
1712212, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771
916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987
937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0623977
1916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358
5583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597094391
0299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.
0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0
.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691,
 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.0532001446519395
6, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956
, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.053200144651939
56, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.070325140
63960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620
321712212, 0.07032514063960227, 0.06239771916667691, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.070325
14063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), InputExample(guid='11821392_2822_6622', text_a="Gene_B interacts with Gene_A isozymes and inhibits pervanadate - induce
d Gene_A activation in human embryonic kidney - 293 cells . Gene_B has been implicated in the pathogenesis of many neurodegenerative diseases , including Parkinson ' s disease and Alzhe
imer ' s disease . Although the function of Gene_B re
[InputExample(guid='11821392_2822_2822', text_a="Gene_N interacts with Gene_S isozymes and inhibits pervanadate - induced Gene_S activation in human embryonic kidney - 293 cells . Gene_
N has been implicated in the pathogenesis of many neurodegenerative diseases , including Parkinson ' s disease and Alzheimer ' s disease . Although the function of Gene_N remains largel
y unknown , recent studies have demonstrated that this protein can interact with phospholipids . To address the role of Gene_N in neurodegenerative disease , we have investigated whethe
r it binds Gene_S ( Gene_S ) and affects Gene_S activity in human embryonic kidney ( HEK ) - 293 cells overexpressing wild type Gene_N or the mutant forms of Gene_N ( A53T , A30P ) asso
ciated with Parkinson ' s disease . Tyrosine phosphorylation of Gene_N appears to play a modulatory role in the inhibition of Gene_S , because mutation of Tyr ( 125 ) to Phe slightly in
creases inhibitory effect of Gene_N on Gene_S activity . Treatment with pervanadate or phorbol myristate acetate inhibits Gene_S more in HEK 293 cells overexpressing Gene_N than in cont
rol cells . Binding of Gene_N to Gene_S requires phox and pleckstrin homology domain of Gene_S and the amphipathic repeat region and non - Abeta component of Gene_N . Although biologica
lly important , co - transfection studies indicate that the interaction of Gene_N with Gene_S does not influence the tendency of Gene_N to form pathological inclusions . These results s
uggest that the association of Gene_N with Gene_S , and modulation of Gene_S activity , is biologically important , but Gene_S does not appear to play an essential role in the pathophys
iology of Gene_N .", text_b=None, label='0', P_gauss1_list=[0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0
.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691,
 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.
043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.06239771916667691
, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748
854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.07
61620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854
, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.062397719166676
91, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321
712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.053200144
65193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.0623977
1916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358
5583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.062397
71916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.0435
85583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227,
0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212,
 0.07032514063960227, 0.06239771916667691, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.0703251406396022
7, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.076162032
1712212, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771
916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987
937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0623977
1916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.04358
5583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.0792597094391
0299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.04358558
3709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239
771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.
0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0
.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691,
 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.
[InputExample(guid='14512552_4758_4758', text_a='Interacting domains of the HN and F proteins of newcastle disease virus . The activation of most paramyxovirus fusion proteins ( F prote
ins ) requires not only cleavage of F ( 0 ) to F ( 1 ) and F ( 2 ) but also coexpression of the homologous attachment protein , hemagglutinin - Gene_S ( HN ) or hemagglutinin ( H ) . Th
e type specificity requirement for HN or H protein coexpression strongly suggests that an interaction between HN and F proteins is required for fusion , and studies of chimeric HN prote
ins have implicated the membrane - proximal ectodomain in this interaction . Using biotin - labeled peptides with sequences of the Newcastle disease virus ( NDV ) F protein heptad repea
t 2 ( HR2 ) domain , we detected a specific interaction with amino acids 124 to 152 from the NDV HN protein . Biotin - labeled HR2 peptides bound to glutathione S - transferase ( GST )
fusion proteins containing these HN protein sequences but not to GST or to GST containing HN protein sequences corresponding to amino acids 49 to 118 . To verify the functional signific
ance of the interaction , two point mutations in the HN protein gene , I133L and L140A , were made individually by site - specific mutagenesis to produce two mutant proteins . These mut
ations inhibited the fusion promotion activities of the proteins without significantly affecting their surface expression , attachment activities , or Gene_S activities . Furthermore ,
these changes in the sequence of amino acids 124 to 152 in the GST - HN fusion protein that bound HR2 peptides affected the binding of the peptides . These results are consistent with t
he hypothesis that HN protein binds to the F protein HR2 domain , an interaction important for the fusion promotion activity of the HN protein .', text_b=None, label='0', P_gauss1_list=
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.
07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.053200144651
93956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.0532001446
5193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], P_gauss2_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.053200144651939
56, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761620321712212, 0.07032514063960227, 0.06239771916667691, 0.0532001446519
3956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.034313010987937154, 0.043585583709748854, 0.05320014465193956, 0.06239771916667691, 0.07032514063960227, 0.0761620321712212, 0.07925970943910299, 0.07925970943910299, 0.0761
620321712212, 0.07032514063960227, 0.06239771916667691, 0.05320014465193956, 0.043585583709748854, 0.034313010987937154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
Examples_total_num 6734.        Label_0_num: 6247       Label_1_num: 487
iterator:  <torch.utils.data.dataloader.DataLoader object at 0x000002545AA88A08>
Train RC Fold: 0, Epoch: 1, loss: 0.2607:  56%|████████████████████▌                | 1171/2101 [02:40<02:05,  7.39it/s]Traceback (most recent call last):
  File "train.py", line 830, in <module>
    train_dataset_rc, epoch)
  File "train.py", line 235, in train_RC
    loss.backward()
  File "F:\software\ana\envs\pytorch-gpu\lib\site-packages\torch\tensor.py", line 245, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
  File "F:\software\ana\envs\pytorch-gpu\lib\site-packages\torch\autograd\__init__.py", line 147, in backward
    allow_unreachable=True, accumulate_grad=True)  # allow_unreachable flag
KeyboardInterrupt
Train RC Fold: 0, Epoch: 1, loss: 0.2607:  56%|████████████████████▌                | 1171/2101 [02:40<02:07,  7.27it/s]

Administrator@BF-202011111649 MINGW64 /d/code/ppim_withGauss (master)
$ ^C

Administrator@BF-202011111649 MINGW64 /d/code/ppim_withGauss (master)
$



    def filterLongDocument(self, documents):
        pass

    def collate_fn(self, batch_examples):
        batch_text_a = [e.text_a for e in batch_examples]
        batch_guid = [e.guid for e in batch_examples]
        batch_P_gauss1_list = [e.P_gauss1_list for e in batch_examples]  # 填入高斯
        batch_P_gauss2_list = [e.P_gauss2_list for e in batch_examples]  # 填入高斯
        batch_P_gauss1_list = torch.tensor(batch_P_gauss1_list, dtype=torch.float)
        batch_P_gauss2_list = torch.tensor(batch_P_gauss2_list, dtype=torch.float)
        x = self.tokenizer(
            batch_text_a, return_tensors='pt', pad_to_max_length=True, max_length = 512, truncation = True)
        batch_label = torch.tensor([self.label_map[e.label] for e in batch_examples])
        return x, batch_label, batch_guid, batch_P_gauss1_list, batch_P_gauss2_list#

class TriageDataSet(data.Dataset):
    label_map = {'0': 0, '1': 1}
    def __init__(self, documents, pretrained_dir):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=False)
        self.examples = []
        self.label_1_count = 0
        self.label_0_count = 0
        for i, document in enumerate(documents):
            self.examples.append(self.__create_examples(document))
        print("Examples_total_num {}.\tLabel_0_num: {}\tLabel_1_num: {}".format(
            len(self.examples), self.label_0_count, self.label_1_count))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __create_examples(self, document):
        pmid = document['id']
        
        text_l = []
        for passage in document['passages']:
            text_l.append(passage['text'])
        text = ' '.join(text_l)

        label = '1' if document['infons']['relevant'] == 'yes' else '0'
        if label == '1':
            self.label_1_count += 1
        else:
            self.label_0_count += 1
        example = InputExample(guid=f"{pmid}", text_a=text, text_b=None, label=label)
        return example

    def collate_fn(self, batch_examples):
        batch_text_a = [e.text_a for e in batch_examples]
        batch_guid = [e.guid for e in batch_examples]

        x = self.tokenizer(
            batch_text_a, return_tensors='pt', pad_to_max_length=True, max_length = 512, truncation = True)
        batch_label = torch.tensor([self.label_map[e.label] for e in batch_examples])
        return x, batch_label, batch_guid
