#Construção do chatbot com Deep NLP

#Importação das bibliotecas
import numpy as np, tensorflow as tf, time, re, contractions
#from pycontractions import Contractions

#1 Parte pré-processamento dos dados

#Importação da base de dados
linhas = open(".\\recursos\\movie_lines.txt",encoding='utf-8',errors="ignore").read().lower().split("\n")
conversas = open(".\\recursos\\movie_conversations.txt",encoding='utf-8',errors="ignore").read().lower().split("\n")
#cont = Contractions('GoogleNews-vectors-negative300.bin')
# Criação de um dicionário para mapear cada linha com seu ID
id_para_linha = dict()

for l in linhas:
    _l = l.split(" +++$+++ ")
    if len(_l) == 5:
        id_para_linha[_l[0]] = _l[4]
        
# Criação e uma lista com todas as conversas
conversas_id = []
for conversa in conversas[:-1]:
    _conversa = conversa.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(" ", "")
    conversas_id.append(_conversa.split(","))

# Libera memoria
linhas = None
conversas = None

# Separação das perguntas e respostas
perguntas = []
respostas = []
for conversa in conversas_id:
    for i in range(len(conversa)-1):
        perguntas.append(id_para_linha[conversa[i]])
        respostas.append(id_para_linha[conversa[i + 1]])

respostas_limpas = [contractions.fix(re.sub("[\W\_]",lambda w:"" if w.group(0) not in "! " else w.group(0),r)) for r in respostas]
perguntas_limpas = [contractions.fix(re.sub("[\W\_]",lambda w:"" if w.group(0) not in "! " else w.group(0),r)) for r in perguntas]

perguntas_dict = dict()
respostas_dict = dict()
palavras_contagem = {}
for frase in perguntas_limpas:
    for w in frase.split(' '):
        palavras_contagem[w] = int(w not in palavras_contagem) or (palavras_contagem[w] + 1)
        
for frase in respostas_limpas:
    for w in frase.split(' '):
        palavras_contagem[w] = int(w not in palavras_contagem) or (palavras_contagem[w] + 1)
        
# Remoção de palavras não frequentes e tokenização (dois dicionários)
limite = 20
perguntas_palavras_int = {}
numero_palavra = 0
for palavra, contagem in palavras_contagem.items():
    if contagem >= limite:
        perguntas_palavras_int[palavra] = numero_palavra
        numero_palavra += 1
        
respostas_palavras_int = perguntas_palavras_int.copy()

# Adição de tokens ao dicionário
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
for k, index in zip(tokens,range(1,len(tokens)+1)):
    perguntas_palavras_int[k] = len(perguntas_palavras_int) + index
    respostas_palavras_int[k] = len(respostas_palavras_int) + index
    
# Criação do dicinário inverso com o dicionário de respostas
respostas_int_palavras = {p_i:p for p, p_i in respostas_palavras_int.items()}

# Adição do token final de string <EOS> para o final de cada resposta
for i in range(len(respostas_limpas)):
    respostas_limpas[i] += "<EOS>"
    
# Tradução de todas as perguntas e respostas para inteiros
# Substituição da palavras menos frequentes para <OUT>
perguntas_para_int = []
for pergunta in perguntas_limpas:
    ints = []
    for palavra in pergunta.split():
        if palavra not in perguntas_palavras_int:
            ints.append(perguntas_palavras_int['<OUT>'])
        else:
            ints.append(perguntas_palavras_int[palavra])
    perguntas_para_int.append(ints)
    
respostas_para_int = []
for resposta in respostas_limpas:
    ints = []
    for palavra in resposta.split():
        if palavra not in respostas_palavras_int:
            ints.append(respostas_palavras_int['<OUT>'])
        else:
            ints.append(respostas_palavras_int[palavra])
    respostas_para_int.append(ints)