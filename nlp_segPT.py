import spacy

# Carregando o modelo de linguagem em português
nlp = spacy.load("pt_core_news_lg")

# 1. Pré-processamento com spacy
def pre_processamento(texto):
    doc_spacy = nlp(texto)  # Processando o texto ao criar um objeto 'doc' que vai possuir tokens e informações linguisticas (POS, tagging, lema)

    sentencas_processadas = []  # Lista para armazenar sentenças processadas
    sentencas_originais = []  # Lista para armazenar sentenças originais
    
    for sentenca in doc_spacy.sents:  # .sents é para Iterar sobre cada sentença no documento
        sentencas_originais.append(sentenca.text)  # .text armazena a sentença original
        palavras = [
            token.text  # Mantém o texto original (não lematizado)
            for token in sentenca  # Itera sobre cada token na sentença
            if not token.is_stop  # Remove stopwords 
            and token.is_alpha  # Mantém apenas palavras (remove números e pontuação)
            and token.pos_ in ["NOUN", "VERB", "PROPN"]  # Usa .pos_ para filtrar substantivos, verbos e nomes próprios
        ]
        if palavras:  # Se a sentença tiver palavras válidas
            sentencas_processadas.append(palavras)  # Adiciona à lista de sentenças processadas
    
    print("Total de sentenças:", len(sentencas_processadas))

    return sentencas_processadas, sentencas_originais

# 2. Função de similaridade entre sentenças
def similaridade(lista_sentencas):
    docs = [nlp(sent) for sent in lista_sentencas]  # Nlp para processa cada sentença com spacy (a gente vai usar isso para calcular similaridade)
    similaridades = [] 
    for i in range(len(docs) - 1): # Itera sobre os índices da lista docs, exceto o último, pois estamos comparando cada sentença com a próxima
        sim = docs[i].similarity(docs[i + 1])  # Calcula a similaridade de cosseno entre sentenças consecutivas usando embeddings 
         # (vetores que representam o significado das sentenças)
        similaridades.append(sim)  
        print(f"Similaridade entre sentença {i + 1} e {i + 2}: {sim:.2f}") 

    return similaridades

# 3. Extração de palavras-chave
def extrair_palavras_chave(sentenca_processada):
    palavras_chave = []
    for palavra in sentenca_processada:
        # Prioriza nomes próprios (PROPN)
        if palavra[0].isupper():  # Se a palavra começa com letra maiúscula (nomes próprios)
            palavras_chave.append(palavra)
    # Se não houver nomes próprios, adiciona outras palavras relevantes (ja pegamos substantivos, verbos e nomes próprios)
    if not palavras_chave:
        palavras_chave = sentenca_processada

    return list(set(palavras_chave))[:5]  # Remove duplicatas e limita a 5 palavras-chave

# 4. Geração de rótulos
def gerar_rotulo_subtopico(subtopico_processado):
    # Aqui, subtopico_processado é uma lista de sentenças processadas (listas de palavras)
    palavras_chave = []
    for sentenca_processada in subtopico_processado:  # Itera sobre cada sentença processada no subtópico
        palavras_chave.extend(extrair_palavras_chave(sentenca_processada))  # Adiciona as palavras-chave extraídas à lista palavras_chave. 
        # O método extend é usado para adicionar todos os elementos de uma lista a outra lista.

    return list(set(palavras_chave))[:5]  # Remove duplicatas com set(), limita a 5 palavras-chave com fatiamento '[:5]' e 
    # Transforma o conjunto set de volta para lista com list()

# 5. Segmentação do documento
def segmentacao_doc(doc, limiar_similaridade):
    sentencas_processadas, sentencas_originais = pre_processamento(doc)  # Pré-processa o documento
    similaridades = similaridade(sentencas_originais)  # Calcula a similaridade entre sentenças
    
    grupos = []  # Lista para armazenar os grupos de sentenças
    grupo_atual = [sentencas_originais[0]]  # Inicia o primeiro grupo com a primeira sentença original
    grupo_processado_atual = [sentencas_processadas[0]]  # Inicia o primeiro grupo com a primeira sentença processada

    for i in range(1, len(sentencas_originais)):  # Itera sobre as sentenças restantes
        if similaridades[i - 1] >= limiar_similaridade:  # Se a similaridade for alta
            grupo_atual.append(sentencas_originais[i])  # Adiciona a sentença original ao grupo atual
            grupo_processado_atual.append(sentencas_processadas[i])  # Adiciona a sentença processada ao grupo atual
        else:
            grupos.append((grupo_atual, grupo_processado_atual))  # Finaliza o grupo atual e o adiciona à lista de grupos
            grupo_atual = [sentencas_originais[i]]  # Inicia um novo grupo com a sentença original
            grupo_processado_atual = [sentencas_processadas[i]]  # Inicia um novo grupo com a sentença processada
    
    grupos.append((grupo_atual, grupo_processado_atual))  # Adiciona o último grupo à lista

    return grupos

# 6. Execução principal
with open('sample.txt', 'r', encoding='utf-8') as f:  # Abre o arquivo de texto
    texto = f.read().replace('\n', ' ')  # Lê o conteúdo do arquivo e substitui quebras de linhas por espaços, se houver
    
limiar = 0.7  # Define o limiar de similaridade (ajuste conforme necessário)
grupos = segmentacao_doc(texto, limiar)  # Segmenta o documento em subtópicos
    
for i, (grupo_original, grupo_processado) in enumerate(grupos):  # Itera sobre os grupos de sentenças
    rotulo = gerar_rotulo_subtopico(grupo_processado)  # Gera um rótulo para o subtópico com base nas sentenças processadas
    print(f"\nSubtópico {i + 1} - Rótulo: {', '.join(rotulo)}")  # Exibe o rótulo
    for sentenca in grupo_original:  # Itera sobre as sentenças originais do subtópico
        print(f"{sentenca}")  # Exibe cada sentença