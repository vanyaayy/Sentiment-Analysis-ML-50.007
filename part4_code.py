import pandas as pd
import numpy as np
from collections import defaultdict
import random
from itertools import product

class HMM_3:
    def __init__(self,file,k=2):
        self.words = None #set of unique words
        self.tag_word_count = None # dict((tag,word),count)
        self.transmissions = None # dict((tag_u,tag_v),count)
        self.tag2_ls = None
        self.count = None # dict(tag,count)
        self.uvw_dic=None
        self.read_file(file,k)
        self.tags = set(self.count.keys())
        
        for key in self.tag_word_count:
            self.tag_word_count[key] = self.tag_word_count[key]/self.count[key[0]]
        self.word_ls = tuple(self.words)#tuple cause immutable
        self._tag_ls = tuple(self.tags)
        #move #START# to first row and #END# to last row
        ls = list(self._tag_ls)
        ls.remove('#START#')
        ls.remove('#END#')
        self.tag3_ls = tuple(ls)
        ls.insert(0,'#None#')
        ls.insert(0,'#START#')
        ls.append('#END#')
        self.tag_ls = tuple(ls)
        self.make_matrix()
    
    def make_matrix(self):
        tag_length = len(self.tag_ls)
        transition_matrix = np.zeros((tag_length,tag_length))
        self.transition_matrix = pd.DataFrame(transition_matrix,index=self.tag_ls,columns=self.tag_ls)
        
        transition_matrix2 = np.zeros((len(self.tag2_ls),tag_length))
        
        for i in range(len(self.tag2_ls)):
            for j in range(tag_length):
                tag_uv = self.tag2_ls[i]
                tag_w = self.tag_ls[j]
                
                transition_matrix2[i][j] = self.uvw_dic[tag_uv[0],tag_uv[1],tag_w]
                
        self.transition_matrix2 = pd.DataFrame(transition_matrix2,index=self.tag2_ls,columns=self.tag_ls)
        
        word_length = len(self.word_ls)
        em_matrix = np.zeros((tag_length,word_length))
        self.em_matrix = pd.DataFrame(em_matrix,index=self.tag_ls,columns=self.word_ls)

        em_matrix = np.zeros((tag_length,word_length))
        for i in range(tag_length):
            for j in range(word_length):
                tag = self.tag_ls[i]
                word = self.word_ls[j]
                em_matrix[i][j] = self.tag_word_count[tag,word]
        self.em_matrix_proba = pd.DataFrame(em_matrix,index=self.tag_ls,columns=self.word_ls)
        pass
    
    def read_file(self,file,k):
        from collections import defaultdict
        seq = ['#START#']
        f = open(file,'r',encoding='utf-8')
        tag_word_ls = []
        word_count = defaultdict(int)
        for line in f:
            split = line.split(' ')
            if len(split)>2:
                continue
            elif len(split)<2:
                #this is a line break
                seq.append('#END#')
                seq.append('#START#')
                continue
            word,tag = split
            word = word.strip()
            tag = tag.strip()
            tag_word_ls.append([tag,word])
            word_count[word]+=1
            seq.append(tag)
        f.close()
        
        #Emissions
        for i in range(len(tag_word_ls)):
            tag,word = tag_word_ls[i]
            if word_count[word]<k:
                tag_word_ls[i] = [tag,'#UNK#']
        tag_word_count = defaultdict(int)
        
        words = []
        for tag,word in tag_word_ls:
            tag_word_count[tag,word]+=1
            words.append(word)
        self.words = set(words)
        self.tag_word_count= tag_word_count
        
        #Transitions
        del seq[-1] #delete last item from the list
         #print(seq)
        trans_dict = defaultdict(int)
        count_u = defaultdict(int)
        for i in range(len(seq)-1):
            tag_u = seq[i]
            count_u[tag_u] += 1 # need to count #END# too
            if tag_u == "#END#":
                continue
            #if u is not #END# we count the transmission 
            tag_v = seq[i+1]
            if (tag_u =="#START#" and tag_v == "#END#"):
                #check for empty blank lines at the end and dont count them
                print('these are blank lines')
                count_u["#START#"] -= 1 #remove additional start
                break
            trans_dict[(tag_u,tag_v)] += 1
        self.transmissions = trans_dict
        self.count = count_u
        
        uvw_dic = defaultdict(int)
        uv_dic = defaultdict(int)
        uvw_dic[("#None#",seq[0],seq[1])] += 1
        for i in range(2,len(seq),1):
            if seq[i-2] == '#END#':
                tag_uvw = ("#None#",seq[i-1],seq[i])
                uvw_dic[tag_uvw] += 1
                tag_uv = ("#None#",seq[i-1])
                uv_dic[tag_uv] += 1
            if seq[i-1] != '#END#' and seq[i-2] != '#END#':
                tag_uvw = (seq[i-2],seq[i-1],seq[i])
                uvw_dic[tag_uvw] += 1 
            if seq[i-2] != '#END#':
                tag_uv = (seq[i-2],seq[i-1])
                uv_dic[tag_uv] += 1
        
        for key in uvw_dic:
            uvw_dic[key] = uvw_dic[key]/uv_dic[(key[0],key[1])]
        
        self.uvw_dic = uvw_dic
            
        uvw_dic = defaultdict(int)
        uvw_dic["#None#",'#START#']=0
        tag_pair = set(self.count.keys())
        tag_pair.add('#None#')
        for comb in product(tag_pair, repeat=2):
            uvw_dic[comb[0],comb[1]]=0
        self.tag2_ls = list(uvw_dic)

def viterbi(word_arr,Hmm):

    states = Hmm.tag_ls[:-1] #set of all possible tags remove #START# and #STOP#
    S = Hmm.tag2_ls
    tag_u ,tag_v = zip(*S)
    A = Hmm.transition_matrix2 # A(tag_uv_vector,tag_w)
    B = Hmm.em_matrix_proba # B(tag_u->word)
    T = len(S) # Total number unique tags
    N = len(word_arr)+2 # Length of sentence make sure no #START# and #STOP#
    
    
    T1 = pd.DataFrame(index=S, columns=range(N)).fillna(float('-inf')) # score 
    T2 = pd.DataFrame(index=S, columns=range(N)) # backpointer
    T1.at[("#None#", '#START#'), 0] = 1 # initialization
    
    #iterate through-each word
    for j in range(1,N-1):
        word = word_arr[j-1]
        if word not in Hmm.words:
            word = '#UNK#'
        epsilon = 1e-10
        x = (np.log(A.multiply(B[word], axis='columns') + epsilon)).add(T1[j-1], axis='index').astype('float64')
        
        #iterate through each possible tag except #END#
        for curr_tag in states:
            uv=S[np.argmax(x[curr_tag].values)] #Top u,v -> w
            score = np.max(x[curr_tag].values)  #get score from top u,v
            T1.loc[[(uv[1],curr_tag)],j] = score #store score in T1(v,w)
            T2.loc[[(uv[1],curr_tag)],j] = uv[0] #store u in T2(v,w)
    
    #handle #END#
    j = N-1
    epsilon = 1e-10
    x = np.log(A['#END#'] + epsilon).add(T1[j-1],axis='index').astype('float64')
    best_pair = x.idxmax() #u,v -> end
    T1.at[(best_pair[1], '#END#'), j] = x.max()
    T2.at[(best_pair[1], '#END#'), j] = best_pair[0]
    
    #backtrack
    pair = T1[N-1].astype('float64').idxmax()
    ans = []
    for i in range(N-1,0,-1):
        next_state = T2.loc[[pair],i][0]
        ans.append(pair[1])
        pair = (next_state, pair[0])
    ans = ans[::-1][:-1] #reverse and remove #END#
    return ans

def test(test_file, output_file, hmm):
    file_object = open(test_file , "r", encoding="utf-8")
    ls = [[]]
    index = 0
    test = []
    for line in file_object:
        test.append(line.strip())
        if (line.strip() == ""):
            ls.append([])
            index += 1
        else:
            ls[index].append(line.strip())
    ls.pop(-1)
    test_df = pd.DataFrame(test, columns=['Word'])
    predict = []

    for i in ls:  
        out = viterbi(i, hmm)
        for j in out:
            predict.append(j)
        predict.append("")

    test_df['Tag'] = predict

    # Writing the output to the file in the desired format
    with open(output_file, "w", encoding="utf-8") as file:
        for word, tag in zip(test_df['Word'], test_df['Tag']):
            if word == "":
                file.write("\n")
            else:
                file.write(f"{word} {tag}\n")

    file_object.close()

###ES####
hmm_ES = HMM_3("Data/ES/train")
test("Data/ES/dev.in", "Data/ES/dev.p4.out", hmm_ES)

###RU####
hmm_RU = HMM_3("Data/RU/train")
test("Data/RU/dev.in", "Data/RU/dev.p4.out", hmm_RU)

###ES TEST DATA####
hmm_ES_Test = HMM_3("Data/ES/train")
test("Data/ES/test.in", "Data/ES/test.p4.out", hmm_ES_Test)

###RU TEST DATA####
hmm_RU_Test = HMM_3("Data/RU/train")
test("Data/RU/test.in", "Data/RU/test.p4.out", hmm_RU_Test)
