import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer

from collections import defaultdict
import numpy as np 

# 1. 'tokenizando' sentencas + normalizacao 

def pre_processamento(texto):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    sentencas = nltk.tokenize.sent_tokenize(texto.lower()) # lista de sentencas
    print("total de sentencas:", len(sentencas))
    
    tokens = [
        WhitespaceTokenizer().tokenize(s) 
            for s in sentencas
    ]
    filtro = [
        [
        t 
            for t in s # iterando sobre cada palavra(t) na sentenca (s) 
                if t not in stopwords and t.isalnum() # condicao para palavra(t) nao ser stopword mas pode ser numero
        ]
        
        for s in tokens # iterando sobre cada sentenca em tokens
    ]
    
    return filtro

# 2. funcao pra calcular similaridade de cosseno 
def similaridade_cosseno(vetor1,vetor2):
    produto_escalar = np.dot(vetor1,vetor2) # calculando o produto escalar entre os vetores:
    # a1 * b1 + a2 * b2 + a3 * b3
    norma1 = np.linalg.norm(vetor1) # o calculo da norma de um vetor seria algo como:
    # A = [a1,a2,a3] que é calculado como ||A|| = raiz de (a1^2+a2^2+a3^2)  
    norma2 = np.linalg.norm(vetor2)
    return produto_escalar / (norma1 * norma2)

# 3. funcao pra vetorizar as sentencas (usando contagem de palavras)
def vetorizar(sentencas):
    # criando um vocabulario de todas as palavras unicas
    vocab = set() # set garante que cada palavra apareça só uma vez ao criar o conjunto vocab
    for s in sentencas: 
        vocab.update(s) # dividindo sentencas em palavras usando split() 
    vocab = list(vocab) # tranformando o conj. vocab em uma lista 
    
    # criando vetores pra cada sentenca
    vetores = []
    for s in sentencas:
        vetor = [s.count(palavra) for palavra in vocab] # pra cada palavra no vocabulario (vocab),
        # conta quantas vezes ela aparece na sentenca 
        vetores.append(vetor)

    return vetores

# 4. funcao principal pra segmentar o documento
def segmentacao_doc(doc, limiar_similaridade):
    # pre processamento de sentencas 
    sentencas_processadas = pre_processamento(doc)

    # vetorizando sentencas 
    vetores = vetorizar(sentencas_processadas)
    print(vetores)
    # agrupando sentencas com base na similaridade 
    grupos = []
    grupo_atual = [sentencas_processadas[0]]

    for i in range(1, len(sentencas_processadas)):
        # calculando a similaridade entre as sentenca atual e anterior
        similaridade = similaridade_cosseno(vetores[i], vetores[i-1])
        print(f"Similaridade entre sentença {i} e {i+1}: {similaridade:.2f}")  # Debug: Mostra a similaridade
        if similaridade >= limiar_similaridade:
            # se a similaridade for alta adiciona no grupo atual
            grupo_atual.append(sentencas_processadas[i])
        else:
            # caso contrario finaliza o grupo_atual e passa pro próximo
            grupos.append(grupo_atual)
            grupo_atual = [sentencas_processadas[i]]
    # adicionando o ultimo grupo 
    grupos.append(grupo_atual)

    return grupos

# abrindo o arquivo para ler todo seu conteúdo 
with open('nj1.txt', 'r', encoding='utf-8') as arq:
    doc = arq.read()  # lendo o arquivo inteiro como uma string e armazenando em 'texto'

limiar_similaridade = 0.5
grupos = segmentacao_doc(doc, limiar_similaridade)

for i, grupo in enumerate(grupos):
    print(f"subtopico {i+1}:")
    for sentenca in grupo:
        print(f"-> {' '.join(sentenca)}")