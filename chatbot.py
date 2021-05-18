#Construção do chatbot com Deep NLP

#Importação das bibliotecas
import numpy as np, tensorflow as tf, time, re, contractions
#from pycontractions import Contractions

#1 Parte pré-processamento dos dados

#Importação da base de dados
linhas = open(".\\recursos\\movie_lines.txt",encoding='utf-8',errors="ignore").read().split("\n")
conversas = open(".\\recursos\\movie_conversations.txt",encoding='utf-8',errors="ignore").read().split("\n")
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
    
# Separação das perguntas e respostas
perguntas = []
respostas = []
for conversa in conversas_id:
    for i in range(len(conversa)-1):
        perguntas.append(id_para_linha[conversa[i]])
        respostas.append(id_para_linha[conversa[i + 1]])

respostas_limpas = [re.sub("[\W]",lambda w:"" if w.group(0) not in "! " else w.group(0),contractions.fix(w)) for w in respostas]
perguntas_limpas = [re.sub("[\W]",lambda w:"" if w.group(0) not in "! " else w.group(0),contractions.fix(w)) for w in perguntas]


