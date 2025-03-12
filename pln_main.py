import nltk
import numpy as np 
import re

# O QUE FALTA: 
# 1 implementar vetorização TF-IDF 

# 2 trocar o retorno da funcao 'preprocessamento' pra retornar uma 
# lista strings ao invés de uma lista de listas (pode botar em uma lista usando for) 

# 3 funcao principal pra segmentar as sentencas em 'subtopicos' usando as funcoes 
# 'preprocessamento', 'calculo TF-IDF', 'similaridade_cosseno' 

# 4 funcao pra escolha de topicos com base em alguma condição (so substantivo e verbo) pra compor o output final

# extra ler sobre RE (expressoes regulares) e sobre dicionarios kkkk + tentar 'enxugar' mais esse codigo 

# 1. 'tokenizando' sentenças e normalizando
def preprocessamento(texto):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    sentencas = nltk.tokenize.sent_tokenize(texto.lower()) # lista de sentencas
    print("total de sentencas:", len(sentencas))

    tokens = [
        nltk.word_tokenize(s)
        for s in sentencas
    ]
    filtragem = [
        [
        t
        for t in s # iterando sobre cada palavra(t) na sentenca (s) 
        if t not in stopwords and re.match(r'^\w+$', t) # condicao para palavra(t) nao ser stopword mas pode ser numero
        ]
        
        for s in tokens # iterando sobre cada sentenca em tokens
    ]
    
    return filtragem

# 2. implementação de TF-IDF 
# 2.1 Calculo de Frequência de Termo 
# TF(w,d) = numero de vezes que w(palavra) aparece no documento d / numero total de palavras no documento d
def calculoTF(doc):
    tf_dict = {} # dicionario pra armazenar a frequência de cada palavra no dicionário (chaves = palavras, valores = frequẽncia)
    total_palavras = len(doc)
    # contando a frequência de cada palavra
    for palavra in doc:
        tf_dict[palavra] = tf_dict.get(palavra, 0) + 1 # pegando a contagem atual da palavra, se ela estiver no dicionario.
    
    for palavra in tf_dict: # calculando 
        tf_dict[palavra] = tf_dict[palavra] / total_palavras # 

    return tf_dict # retornando a frequência dos termos (implementado em dicionario pra debuggar melhor)

# 3. função pra calcular similaridade de cossenos usando Numpy
def similaridade_cosseno(v1, v2):
    produto_escalar = np.dot(v1,v2) # calculando o produto escalar entre os vetores (soma dos produtos de seus elementos):
    # a1 * b1 + a2 * b2 + a3 * b3

    norma1, norma2 = np.linalg.norm(v1), np.linalg.norm(v2) 
    # calculando as normas euclidianas dos vetores (a raiz quadrada da soma ao quadrado de todos os elementos):
    # A = [a1,a2,a3] que é calculado como ||A|| = raiz de (a1^2+a2^2+a3^2)  

    return produto_escalar / (norma1 * norma2) # similaridade de cossenos = (v1 * v2) / ||v1||*||v2||

with open ("nj1.txt", "r") as doc:
    documento = doc.read()

sent_normalizadas = preprocessamento(documento)
print(sent_normalizadas)
tf_dict = calculoTF(sent_normalizadas)
print(f"{tf_dict}.2f")