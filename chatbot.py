#Construção do chatbot com Deep NLP

#Importação das bibliotecas
import numpy as np, tensorflow as tf, time, re

#1 Parte pré-processamento dos dados

#Importação da base de dados
linhas = open(".\\recursos\\movie_lines.txt",encoding='utf-8',errors="ignore").read().split("\n")

