import nltk
import numpy as np 
from collections import defaultdict
import math

# O QUE FALTA: 
# 1 implementar vetorização TF-IDF [x]

# 2 trocar o retorno da funcao 'preprocessamento' pra retornar uma 
# lista strings ao invés de uma lista de listas (pode botar em uma lista usando for) [x]

# 3 funcao principal pra segmentar as sentencas em 'subtopicos' usando as funcoes 
# 'preprocessamento', 'calculo TF-IDF', 'similaridade_cosseno' []

# 4 funcao pra escolha de topicos com base em alguma condição (so substantivo e verbo) pra compor o output final []

# 5 destrinchar todas as funcoes pra entender o funcionamento completo []

# 1. 'tokenizando' sentenças e normalizando
def pre_processamento(doc):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    sentencas = nltk.tokenize.sent_tokenize(documento.lower()) # lista de sentencas com .lower() para 'polir' comparacoes
    print("total de sentencas:", len(sentencas))

    sentencas_processadas = []
    for sentenca in sentencas:
        palavras = nltk.word_tokenize(sentenca) # tokenizando cada sentenca em sentencas
        palavras = [
            palavra 
            for palavra in palavras 
            if palavra.isalnum() and palavra not in stopwords # filtrando palavras que nao estao nas stopwords e pegando numeros tambem
        ]
        sentencas_processadas.append(palavras)

    return sentencas_processadas

# 2. implementação de TF-IDF (para vetorizar cada sentenca)
# 2.1 Calculo de Frequência de Termo 
# TF(w,d) = numero de vezes que w(palavra) aparece em d (documento) / numero total de palavras no documento d
def calculoTF(sentenca): # opera em uma unica sentenca
    tf_dict = {} # dicionario pra armazenar a frequência de cada palavra no dicionário (chaves = palavras, valores = frequẽncia)
    total_palavras = len(sentenca)
    # contando a frequência de cada palavra
    for palavra in sentenca:
        tf_dict[palavra] = tf_dict.get(palavra, 0) + 1 # pegando a contagem atual da palavra, se ela estiver no dicionario.
    
    for palavra in tf_dict: # calculando 
        tf_dict[palavra] = tf_dict[palavra] / total_palavras # 
    
    return tf_dict # retornando a frequência dos termos 

# 2.2 Calculo de Frequẽncia Inversa do Documento 
# IDF(w) = log(N/1+df(w)) N = total de sentencas, df(w) é o numero de sentencas contendo a palavra w 
def calculoIDF(doc): # opera em todas as sentencas no documento
    idf_dict = defaultdict(lambda: 0) # defaultdict de collections é usado aqui para definir uma dicionario cujo o valor de chave padrao
    # inicial é 0. (lambda: 0) é uma funcao anonima que sempre retorna 0
    total_sentencas = len(doc)

    for sentenca in doc: 
        palavras_unicas = set(sentenca) # inserindo palavras unicas da sentenca na variavel 
        for palavra in palavras_unicas:
            idf_dict[palavra] += 1 # incrementando a chave "palavra" em 1 pra cada vez que ela aparecer em diferentes sentencas
    
    for palavra in idf_dict:
        idf_dict[palavra] = math.log(total_sentencas) / (1 + idf_dict[palavra]) # adicionando 1 para evitar divisao por 0 
    
    return idf_dict 

# 2.3 junção das funcoes calculoTF e calculoIDF
def calculoTF_IDF(doc):
    idf_dict = calculoIDF(doc) # chamando a funcao calculoIFD para o documento
    tf_idf_doc = [] 

    for sentenca in doc: # iterando por cada sentenca no documento
        tf_dict = calculoTF(sentenca) # aplicando a funcao calculoTF para a sentenca atual 
        tf_idf_dict = {
            palavra: 
            tf_dict[palavra] * idf_dict[palavra] # calculando a pontuacao TF-IDF para cada palavra multiplicando a pontuacao de TF e IDF 
            for palavra in tf_dict
        }
        tf_idf_doc.append(tf_idf_dict) # pegando a pontuacao atual da sentenca para a lista td_idf_doc
    
    return tf_idf_doc, idf_dict

# 3 vetorização do documento para ser analisado na segmentação 
def vetorizacao_doc(doc, idf_dict):
    vocab = sorted(idf_dict.keys()) # as chaves do dicionario IDF sao as palavras e seus valores sao a pontuacao IDF 
    # sorted() para ordernar as palavras alfabeticamente 
    vetores_tf_idf = [] # armazenando vetores TF-IDF pra cada sentenca no documento
    # cada vetor é uma lista de valores numericos representando a pontuacao TF-IDF para as palavras na sentenca

    for sentenca in doc: 
        vetortemp = [
            sentenca.get(palavra, 0) # pega a pontuacao TF-IDF atual da palavra na sentenca atual e se a palavra nao tiver no vocabulario
            # retorna 0 (valor default)
            for palavra in vocab
        ]
        vetores_tf_idf.append(vetortemp) # o vetor pra sentenca atual vai ser adicionado a lista vetores_td_idf
    
    return vetores_tf_idf, vocab

# 4. função pra calcular similaridade de cossenos usando numpy
def similaridade_cosseno(v1, v2):
    produto_escalar = np.dot(v1,v2) # calculando o produto escalar entre os vetores (soma dos produtos de seus elementos):
    # a1 * b1 + a2 * b2 + a3 * b3

    norma1, norma2 = np.linalg.norm(v1), np.linalg.norm(v2) 
    # calculando as normas euclidianas dos vetores (a raiz quadrada da soma ao quadrado de todos os elementos):
    # A = [a1,a2,a3] que é calculado como ||A|| = raiz de (a1^2+a2^2+a3^2)  
    if norma1 == 0 or norma2 == 0:
        return 0 # evitando divisao por 0 

    return produto_escalar / (norma1 * norma2) # similaridade de cossenos = (v1 * v2) / ||v1||*||v2||

with open ("nj1.txt", "r") as doc:
    documento = doc.read()
