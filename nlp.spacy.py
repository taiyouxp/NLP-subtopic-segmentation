import spacy
nlp = spacy.load("pt_core_news_lg") #O spacy já possui recursos como tokenização, lemmataização, indetificação da classe gramatical e detectação de stowpords


def pre_processamento(texto):
    texto = texto.lower()  # Converte a string para minúsculas

    doc_spacy = nlp(texto)  #cria um objeto doc com o texto processado com as anotações linguisticas 
    sentencas_processadas = [] #receberá a lista de palavras totalmente filtradas
    
    for sentenca in doc_spacy.sents: #itera sobre cada senteça que o spacy gerou em doc_spacy
        palavras = [
            token.text.lower()
            #token.lemma_.lower() # função para que a palavra volte para sua forma "padrão": andar -> ando
            for token in sentenca 
            if not token.is_stop #remove as stopwords que não servem para o entendimento 
            and token.is_alpha #mantem apenas texto, remove numeros e pontuação
            and token.pos_ in ["NOUN", "VERB","PROPN"] #filtra apenas os substativos e os verbos 
        ]
        if palavras:
            sentencas_processadas.append(palavras)
    
    print("Total de sentenças:", len(sentencas_processadas))
    return sentencas_processadas

def lista_de_senteças(texto):
    lista_de_sentecas = []
    documento = nlp(texto)
    senteças = documento.sents
    for sent in senteças:
        lista_de_sentecas.append(sent.text)
    return lista_de_sentecas


def ler_texto(texto):
    with open(texto, "r", encoding="utf-8") as arquivo:
        conteudo = arquivo.read()
    return conteudo


def similaridade(lista):
    valores_de_similaridade = []
    sentecas_processadas = []
    for texto in lista:
        senteca = nlp(texto)
        sentecas_processadas.append(senteca)
    for i in range(len(sentecas_processadas)- 1):
        score_similarity = sentecas_processadas[i].similarity(sentecas_processadas[i+1])
        valores_de_similaridade.append(score_similarity)
    return valores_de_similaridade


texto = "rede.txt"
conteudo = ler_texto(texto)
sentenças_processadas = pre_processamento(conteudo)
lista = lista_de_senteças(conteudo)
similaridade_sentecas = similaridade(lista)
print(similaridade_sentecas)
print(lista)
print(sentenças_processadas)
