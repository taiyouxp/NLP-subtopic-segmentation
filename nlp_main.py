import nltk
import numpy as np
from collections import defaultdict
import math
from sklearn.cluster import AgglomerativeClustering

# Baixar recursos do NLTK (se necessário)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# 1. 'tokenizando' sentenças e normalizando
with open("sample.txt", "r", encoding='utf-8') as doc:
    documento = doc.read()

def pre_processamento(doc):
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    sentencas = nltk.tokenize.sent_tokenize(doc.lower())  # lista de sentenças com .lower() para 'polir' comparações

    print("Total de sentenças:", len(sentencas))

    sentencas_processadas = []
    for sentenca in sentencas:
        palavras = nltk.word_tokenize(sentenca, language='portuguese')  # tokenizando cada sentença em palavras
        palavras = [
            palavra
            for palavra in palavras
            if palavra.isalnum() and palavra not in stopwords  # filtrando palavras que não estão nas stopwords e pegando números também
        ]
        if palavras:
            sentencas_processadas.append(palavras)
    
    print("Sentenças processadas:")
    for i, sentenca in enumerate(sentencas_processadas):
        print(f"Sentence {i + 1}: {' '.join(sentenca)}")

    return sentencas_processadas

# 2. implementação de TF-IDF (para vetorizar cada sentença)
# 2.1 Cálculo de Frequência de Termo 
def calculoTF(sentenca):  # opera em uma única sentença
    tf_dict = {}  # dicionário para armazenar a frequência de cada palavra
    total_palavras = len(sentenca)
    
    # Contando a frequência de cada palavra
    for palavra in sentenca:
        tf_dict[palavra] = tf_dict.get(palavra, 0) + 1
    
    # Normalizando a frequência
    for palavra in tf_dict:
        tf_dict[palavra] = tf_dict[palavra] / total_palavras
    
    return tf_dict

# 2.2 Cálculo de Frequência Inversa do Documento 
def calculoIDF(doc):  # opera em todas as sentenças no documento
    idf_dict = defaultdict(lambda: 0)  # defaultdict para armazenar a frequência inversa
    total_sentencas = len(doc)

    for sentenca in doc:
        palavras_unicas = set(sentenca)  # palavras únicas na sentença
        for palavra in palavras_unicas:
            idf_dict[palavra] += 1
    
    # Calculando o IDF
    for palavra in idf_dict:
        idf_dict[palavra] = math.log(total_sentencas / (1 + idf_dict[palavra]))  # adicionando 1 para evitar divisão por zero
    
    return idf_dict

# 2.3 Junção das funções calculoTF e calculoIDF
def calculoTF_IDF(doc):
    idf_dict = calculoIDF(doc)  # calculando o IDF para o documento
    tf_idf_doc = []

    for sentenca in doc:
        tf_dict = calculoTF(sentenca)  # calculando o TF para a sentença atual
        tf_idf_dict = {
            palavra: tf_dict[palavra] * idf_dict[palavra]  # calculando o TF-IDF para cada palavra
            for palavra in tf_dict
        }
        tf_idf_doc.append(tf_idf_dict)
    
    print("TF-IDF Scores:")
    for i, tf_idf_dict in enumerate(tf_idf_doc):
        print(f"Sentence {i + 1}: {tf_idf_dict}")

    return tf_idf_doc, idf_dict

# 3. Vetorização do documento para ser analisado na segmentação
def vetorizacao_doc(tf_idf_doc, idf_dict):
    vocab = sorted(idf_dict.keys())  # vocabulário ordenado
    vetores_tf_idf = []

    for sentenca in tf_idf_doc:
        vetortemp = [
            sentenca.get(palavra, 0)  # obtendo o valor TF-IDF da palavra ou 0 se não existir
            for palavra in vocab
        ]
        vetores_tf_idf.append(vetortemp)
    
    print("Vetores TF-IDF:")
    for i, vetor in enumerate(vetores_tf_idf):
        print(f"Sentenca {i + 1}: {vetor}")

    return vetores_tf_idf, vocab

# 4. Aplicação de similaridade dos cossenos
def similaridade_cosseno(v1, v2):
    produto_escalar = np.dot(v1, v2)
    norma1, norma2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norma1 == 0 or norma2 == 0:
        return 0  # evitando divisão por zero
    return produto_escalar / (norma1 * norma2)

def smlcss_matriz(vetores_tf_idf):
    n = len(vetores_tf_idf)
    matriz_sml = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            matriz_sml[i, j] = similaridade_cosseno(vetores_tf_idf[i], vetores_tf_idf[j])
            if i != j:
                matriz_sml[j, i] = matriz_sml[i, j]  # matriz simétrica
    
    print("Matriz de Similaridade de Cosseno:")
    print(matriz_sml)
    return matriz_sml

# 5. Segmentação do documento em subtópicos usando similaridade 'threshold'
def segmentacao_cluster(matriz_sml, limiar=0.7):
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=1 - limiar
    )
    clusters = clustering.fit_predict(1 - matriz_sml)
    return clusters

# 6. Extração de palavras-chave
def extrair_palavras_chave(sentenca):
    sufixos_substantivos = ['ção', 'dade', 'mente', 'agem', 'ismo', 'ente']
    sufixos_verbos = ['ar', 'er', 'ir', 'ando', 'endo', 'indo']
    palavras_chave = [
        palavra
        for palavra in sentenca
        if any(palavra.endswith(sufixo) for sufixo in sufixos_substantivos + sufixos_verbos)
    ]
    return palavras_chave[:5]

# 7. Geração de rótulo para subtópico
def gerar_rotulo_subtopico(subtopico):
    texto_subtopico = " ".join(" ".join(sentenca) for sentenca in subtopico)
    return extrair_palavras_chave(texto_subtopico.split())

# 8. Função principal para segmentar e exibir subtópicos
def subtopicos(clusters, sentencas_processadas):
    subtopicos = []
    
    # Agrupa as sentenças com base nos clusters
    for cluster_id in set(clusters):
        indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        subtopico = [sentencas_processadas[i] for i in indices]
        subtopicos.append(subtopico)
    
    # Gera rótulos para cada subtópico
    for i, subtopico in enumerate(subtopicos):
        rotulo = gerar_rotulo_subtopico(subtopico)
        print(f"Subtópico {i + 1} - Rótulo: {rotulo}")
        for sentenca in subtopico:
            print(f"-> {' '.join(sentenca)}")
        print()  # Espaço entre subtópicos

# Execução do pipeline
sentencas_processadas = pre_processamento(documento)
tf_idf_doc, idf_dict = calculoTF_IDF(sentencas_processadas)
vetores_tf_idf, vocab = vetorizacao_doc(tf_idf_doc, idf_dict)
matriz_sml = smlcss_matriz(vetores_tf_idf)

clusters = segmentacao_cluster(matriz_sml, limiar=0.7)
print('\nClusters:', clusters)

# Exibir subtópicos
subtopicos(clusters, sentencas_processadas)