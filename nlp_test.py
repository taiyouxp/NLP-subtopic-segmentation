import nltk

from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from collections import defaultdict
import numpy as np 

from nltk.stem import RSLPStemmer

# terminar:
# 1 melhorar os rotulos (pegar nomes proprios)
# 2 comentar tudo 
# 3 melhorar a saida
# 4 (talvez) usar mais alguma tecnica no preprocessamento

# 1. 'tokenizando' sentencas + normalizacao 
def pre_processamento(texto):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    stemmer = RSLPStemmer()
    sentencas_p_processamento, sentencas = nltk.tokenize.sent_tokenize(texto.lower()), nltk.tokenize.sent_tokenize(texto) # lista de sentencas
    print("total de sentencas:", len(sentencas))
    
    tokens = [
        WhitespaceTokenizer().tokenize(s) 
            for s in sentencas_p_processamento
    ]

    sentencas_processadas = [
        [
        stemmer.stem(t) 
            for t in s # iterando sobre cada palavra(t) na sentenca (s) 
                if t not in stopwords and t.isalnum() # condicao para palavra(t) nao ser stopword mas pode ser numero
        ]
        
        for s in tokens # iterando sobre cada sentenca em tokens
    ]
    
    return sentencas_processadas, sentencas

# 2. funcao pra calcular similaridade de cosseno 
def similaridade_cosseno(vetor1,vetor2):
    produto_escalar = np.dot(vetor1,vetor2) # calculando o produto escalar entre os vetores:
    # a1 * b1 + a2 * b2 + a3 * b3
    norma1, norma2 = np.linalg.norm(vetor1), np.linalg.norm(vetor2) # o calculo da norma de um vetor seria algo como:
    # A = [a1,a2,a3] que é calculado como ||A|| = raiz de (a1^2+a2^2+a3^2)  

    if norma1 == 0 or norma2 == 0:
        return 0  # Evita divisão por zero
    return produto_escalar / (norma1 * norma2)

# 3. Vetorização com TF-IDF
def vetorizar_tfidf(sentencas):
    sentencas_texto = [" ".join(s) for s in sentencas]
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.8)
    vetores_tfidf = vectorizer.fit_transform(sentencas_texto)
    
    return vetores_tfidf.toarray()

# 4. Segmentação do Documento
def segmentacao_doc(doc, limiar_similaridade):
    # pre processamento de sentencas 
    sentencas_processadas, sentencas_originais = pre_processamento(doc)

    # vetorizando sentencas 
    vetores = vetorizar_tfidf(sentencas_processadas)
    print(vetores)

    # agrupando sentencas com base na similaridade 
    grupos = []
    grupo_atual = [sentencas_originais[0]]

    for i in range(1, len(sentencas_processadas)):
        # calculando a similaridade entre as sentenca atual e anterior
        similaridade = similaridade_cosseno(vetores[i], vetores[i-1])
        print(f"Similaridade entre sentença {i} e {i+1}: {similaridade:.2f}")  # Debug: Mostra a similaridade
        if similaridade >= limiar_similaridade:
            # se a similaridade for alta adiciona no grupo atual
            grupo_atual.append(sentencas_originais[i])
        else:
            # caso contrario finaliza o grupo_atual e passa pro próximo
            grupos.append(grupo_atual)
            grupo_atual = [sentencas_originais[i]]
    # adicionando o ultimo grupo 
    grupos.append(grupo_atual)

    return grupos

# 4. Extrair Palavras-Chave
def extrair_palavras_chave(sentenca):
    # Lista de sufixos para substantivos e verbos em português
    sufixos_substantivos = ['al','ção', 'dade', 'mente', 'agem', 'ismo', 'ente', 'icos', 'ico', 'ulo', 'ento']
    sufixos_verbos = ['ar', 'er', 'ir'] 

    # Filtra apenas substantivos (N) e verbos (V)
    palavras_chave = [
        palavra
        for palavra in sentenca
        if any(palavra.endswith(sufixo) for sufixo in sufixos_substantivos + sufixos_verbos)
    ]
    
    return palavras_chave[:5]  # Limita a 5 palavras

# 5. Gerar Rótulo para Subtópico
def gerar_rotulo_subtopico(subtopico):
    # Junta todas as sentenças do subtópico em um único texto
    texto_subtopico = " ".join(subtopico)
    # Extrai palavras-chave a partir desse texto
    return extrair_palavras_chave(texto_subtopico.split())

# 6. Segmentação do Documento
def segmentacao_doc(doc, limiar_similaridade):
    # pre processamento de sentencas 
    sentencas_processadas, sentencas_originais = pre_processamento(doc)

    # vetorizando sentencas 
    vetores = vetorizar_tfidf(sentencas_processadas)
    print(vetores)

    # agrupando sentencas com base na similaridade 
    grupos = []
    grupo_atual = [sentencas_originais[0]]

    for i in range(1, len(sentencas_processadas)):
        # calculando a similaridade entre as sentenca atual e anterior
        similaridade = similaridade_cosseno(vetores[i], vetores[i-1])
        print(f"Similaridade entre sentença {i} e {i+1}: {similaridade:.2f}")  # Debug: Mostra a similaridade
        if similaridade >= limiar_similaridade:
            # se a similaridade for alta adiciona no grupo atual
            grupo_atual.append(sentencas_originais[i])
        else:
            # caso contrario finaliza o grupo_atual e passa pro próximo
            grupos.append(grupo_atual)
            grupo_atual = [sentencas_originais[i]]
    # adicionando o ultimo grupo 
    grupos.append(grupo_atual)

    return grupos

# 6. Execução
with open('vj7.txt', 'r') as arq:
    doc = arq.read()

limiar_similaridade = 0.1  # Ajuste conforme necessário
grupos = segmentacao_doc(doc, limiar_similaridade)

print(grupos)
# Exibir resultados com rótulos
for i, grupo in enumerate(grupos):
    rotulo = gerar_rotulo_subtopico(grupo)
    print(f"subtópico {i+1} - rótulo: {rotulo}")
    for sentenca in grupo:
        print(f"-> {sentenca}")