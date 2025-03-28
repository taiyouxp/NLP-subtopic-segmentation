# Subtopic Segmentation and Labeling with spaCy / Segmentação Subtopical e Rotulação com spaCy
### A simple NLP project for college / Um projeto simples de PLN para faculdade

## <span style="color: #737373">✱</span> Overview / Visão geral
### EN: this project aims to automatically segment a document into subtopics and assign meaningful labels to each segment based on semantic similarity between sentences. <br><br>  PT: Este projeto visa segmentar automaticamente um documento em subtópicos e atribuir rótulos significativos a cada segmento com base na similaridade semântica entre frases.

## <span style="color: #737373">✱</span> Key Features / Características principais
#### * Document Segmentation – Divides text into coherent subtopics / Segmentação de Documento – Divide o texto em subtópicos coerentes
#### * Automatic Labeling – Generates descriptive labels using keywords /  Rotulação Automática – Gera rótulos descritivos usando palavras-chave
#### * SpaCy NLP – Uses en_core_web_lg for high-quality english text processing / usa pt_core_news_lg para processamento de alta qualidade de texto em português
#### * Cosine Similarity – Measures sentence similarity for segmentation / Similaridade de Cosseno – Mede similaridade entre frases para segmentação

## <span style="color: #737373">✱</span> Installation / Instalação
### EN: Before runnings the code, install spaCy and the language model (*if you're using ubuntu/debian, you will need either a virtual environment[<code style="color: #f0e008">'python3 -m venv'</code>] or pipx for isolated installation) <br><br> PT: Antes de executar o código, instale o spaCy e o modelo de linguagem (*se você estiver usando distros baseadas em ubuntu/debian, você precisará usar um ambiente virtual[<code style="color: #f0e008">'python3 -m venv'</code>] ou pipx para instalação isolada)

#### English model: <pre style="color: #f0e008">pip install -U spacy <br>python -m spacy download en_core_web_lg</pre>
#### Modelo português:<pre style="color: #f0e008">pip install -U spacy <br>python -m spacy download pt_core_news_lg</pre>
## <span style="color: #737373">✱</span> How it works / Como funciona 

### <span style="color: #1897d6">[1]</span> Preprocessing / Preprocessamento (<code style="color: #f0084a">pre_processing</code>)
#### <li> Tokenizes sentences / Tokeniza sentença <br> <li> Removes stopwords, numbers, and punctuation / Remove stopwords, números e pontuações <br> <li> Keeps only nouns, verbs, and proper nouns / Armazena somente substantivos, verbos e substantivos próprios 

### <span style="color: #1897d6">[2]</span> Similarity Calculation / Calculo de similaridade (<code style="color: #f0084a">similarity</code>)
#### <li> Uses spaCy embeddings to compute cosine similarity between sentences / Usa spacy embeddings pra computar similaridade de cossenos entre as sentenças <br> <li> Helps determine where subtopic boundaries should be / Ajuda a determinar onde o limite dos subtópico devem estar

### <span style="color: #1897d6">[3]</span> Keyword Extraction / Extração de palavras chave (<code style="color: #f0084a">extract_keywords</code>)
#### <li> Prioritizes proper nouns (PROPN) for labeling / prioriza substativos próprios (PROPN) para rotulação<br> <li> Falls back to other keywords if needed / Volta para as outras palavras chave se preciso

### <span style="color: #1897d6">[4]</span> Subtopic Labeling / Rotulação de subtópicos (<code style="color: #f0084a">generate_subtopic_label</code>)
#### <li> Combines keywords from sentences in a subtopic (PROPN) for labeling / Combina palavras-chave das sentencas em um subtópico <br> <li> Selects the top 5 most relevant words as the label / Seleciona as 5 palavras mais relevantes para compor o rótulo

### <span style="color: #1897d6">[5]</span> Document Segmentation / Segmentação do documento (<code style="color: #f0084a">document_segmentation</code>)
#### <li> Groups sentences with high similarity (≥ threshold) / Agrupa sentenças com alta similaridade (> limiar) <br> <li> Splits text into subtopics when similarity drops / Divide o texto em subtópicos quando a similaridade é menor que o limiar

## <span style="color: #737373">✱</span> Customization / Customização 
### <li> Adjust threshold (0.7-0.9) for more/less granular segments / ajuste de limiar (0.7-0.9) para mais ou menos segmentos
### <li> Modify extract_keywords to prioritize different POS tags / Modificar extracao_palavra_chave para priorizar difererentes marcações POS 
### <li> Try different models (e.g., en_core_web_trf for transformers) / Tente diferentes modelos
---
## <span style="color: #737373">✱</span> Modifies / Modificaçoes
### As i progress in NLP/ML stuff i will maybe update this code to apply more functionalities  but thats it for now / Com a progressão que eu tiver em conhecimentos de NPL/AM talvez eu atualize esse código para aplicar mais funcionalidades.
<h4> © 2025 tyou</h4>
