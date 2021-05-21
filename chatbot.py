#Construção do chatbot com Deep NLP
#pip install tensorflow-addons
#pip instrall contractions==0.0.18
#conda install python=3.6
#Importação das bibliotecas
import numpy as np, time, re, contractions, tensorflow as tf
#from pycontractions import Contractions

#1 Parte pré-processamento dos dados

#Importação da base de dados
linhas = open('.\\recursos\\movie_lines.txt',encoding='utf-8',errors="ignore").read().lower().split("\n")
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
for index, k in enumerate(tokens,1):
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
    
    
# Ordernação das perguntas e respostas pelo tamanho das perguntas
perguntas_limpas_ordenadas = []
respostas_limpas_ordenadas = []
for tamanho in range(1, 25 + 1):
    for i in enumerate(perguntas_para_int):
        if len(i[1]) == tamanho:
            perguntas_limpas_ordenadas.append(perguntas_para_int[i[0]])
            respostas_limpas_ordenadas.append(respostas_para_int[i[0]])
            
# --- Parte 2 - Construção do modelo Seq2Seq ---
# Criação de placeholders para as entradas e saídas
# [64, 25]
def entradas_modelo():
    entradas = tf.placeholder(tf.int32, [None, None], name = 'entradas')
    saidas = tf.placeholder(tf.int32, [None, None], name = 'saidas')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return entradas, saidas, lr, keep_prob

# Pré-processamento das saídas (alvos)
# [batch_size, 1] = [64, 1]
# 0 - SOS (8825)
# 1 - SOS (8825)
def preprocessamento_saidas(saidas, palavra_para_int, batch_size):
    esquerda = tf.fill([batch_size, 1], palavra_para_int['<SOS>'])
    direita = tf.strided_slice(saidas, [0,0], [batch_size, -1], strides = [1,1])
    saidas_preprocessadas = tf.concat([esquerda, direita], 1)
    return saidas_preprocessadas

# Criação da RNN do codificador
#tf.VERSION    
def rnn_codificador(rnn_entradas, rnn_tamanho, numero_camadas, keep_prob, tamanho_sequencia):
    lstm = tf.contrib.rnn.LSTMCell(rnn_tamanho)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_celula = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * numero_camadas)
    _, encoder_estado = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_celula,
                                                     cell_bw = encoder_celula,
                                                     sequence_length = tamanho_sequencia,
                                                     inputs = rnn_entradas,
                                                     dtype = tf.float32)
    return encoder_estado

# Decodificação da base de treinamento
def decodifica_base_treinamento(encoder_estado, decodificador_celula, 
                                decodificador_embedded_entrada, tamanho_sequencia,
                                decodificador_escopo, funcao_saida,
                                keep_prob, batch_size):
    estados_atencao = tf.zeros([batch_size, 1, decodificador_celula.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(estados_atencao,
                                                                                                                                    attention_option = 'bahdanau',
                                                                                                                                    num_units = decodificador_celula.output_size)
    funcao_decodificador_treinamento = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_estado[0],
                                                                                     attention_keys, 
                                                                                     attention_values, 
                                                                                     attention_score_function, 
                                                                                     attention_construct_function,
                                                                                     name = 'attn_dec_train')
    decodificador_saida, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decodificador_celula,
                                                                       funcao_decodificador_treinamento,
                                                                       decodificador_embedded_entrada,
                                                                       tamanho_sequencia,
                                                                       scope = decodificador_escopo)
    decodificador_saida_dropout = tf.nn.dropout(decodificador_saida, keep_prob)
    return funcao_saida(decodificador_saida_dropout)  
    
# Decodificação da base de teste/validação
def decodifica_base_teste(encoder_estado, decodificador_celula, 
                          decodificador_embedding_matrix,sos_id, eos_id, tamanho_maximo,
                          numero_palavras, decodificador_escopo, funcao_saida,
                          keep_prob, batch_size):                          
    estados_atencao = tf.zeros([batch_size, 1, decodificador_celula.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(estados_atencao,
                                                                                                                                    attention_option = 'bahdanau',
                                                                                                                                    num_units = decodificador_celula.output_size)
    funcao_decodificador_teste = tf.contrib.seq2seq.attention_decoder_fn_inference(funcao_saida,
                                                                                   encoder_estado[0],
                                                                                   attention_keys, 
                                                                                  attention_values, 
                                                                                   attention_score_function, 
                                                                                   attention_construct_function,
                                                                                   decodificador_embedding_matrix,
                                                                                   sos_id,
                                                                                   eos_id,
                                                                                   tamanho_maximo,
                                                                                   numero_palavras,
                                                                                   name = 'attn_dec_inf')
    previsoes_teste, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decodificador_celula,
                                                                   funcao_decodificador_teste,
                                                                   scope = decodificador_escopo)
    return previsoes_teste  

# Criação da RNN do decodificador
def rnn_decodificador(decodificador_embedded_entrada, decodificador_embeddings_matrix,
                      codificador_estado, numero_palavras, tamanho_sequencia, rnn_tamanho,
                      numero_camadas, palavra_para_int, keep_prob, batch_size):
    with tf.variable_scope("decodificador") as decodificador_escopo:
        lstm = tf.contrib.rnn.LSTMCell(rnn_tamanho)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decodificador_celula = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * numero_camadas)
        pesos = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        funcao_saida = lambda x: tf.contrib.layers.fully_connected(x, numero_palavras,
                                                                   None,
                                                                   scope = decodificador_escopo,
                                                                   weights_initializer = pesos,
                                                                   biases_initializer = biases)
        previsoes_treinamento = decodifica_base_treinamento(codificador_estado,
                                                            decodificador_celula,
                                                            decodificador_embedded_entrada,
                                                            tamanho_sequencia,
                                                            decodificador_escopo,
                                                            funcao_saida,
                                                            keep_prob,
                                                            batch_size)
        decodificador_escopo.reuse_variables()
        previsoes_teste = decodifica_base_teste(codificador_estado,
                                                decodificador_celula,
                                                decodificador_embeddings_matrix,
                                                palavra_para_int['<SOS>'],
                                                palavra_para_int['<EOS>'],
                                                tamanho_sequencia - 1,
                                                numero_palavras,
                                                decodificador_escopo,
                                                funcao_saida,
                                                keep_prob,
                                                batch_size)
        return previsoes_treinamento, previsoes_teste
    
# Criação do modelo Seq2Seq
def modelo_seq2seq(entradas, saidas, keep_prob, batch_size, tamanho_sequencia,
                   numero_palavras_respostas, numero_palavras_perguntas,
                   tamanho_codificador_embeddings, tamanho_decodificador_embeddings,
                   rnn_tamanho, numero_camadas, perguntas_palavras_int):
    codificador_embedded_entrada = tf.contrib.layers.embed_sequence(entradas,
                                                                    numero_palavras_respostas + 1,
                                                                    tamanho_codificador_embeddings,
                                                                    initializer = tf.random_uniform_initializer(0,1))
    codificador_estado = rnn_codificador(codificador_embedded_entrada,
                                         rnn_tamanho, numero_camadas,
                                         keep_prob, tamanho_sequencia)
    saidas_preprocessadas = preprocessamento_saidas(saidas, perguntas_palavras_int, batch_size)
    decodificador_embeddings_matrix = tf.Variable(tf.random_uniform([numero_palavras_perguntas + 1,
                                                                     tamanho_decodificador_embeddings], 0, 1))
    decodificador_embedded_entradas = tf.nn.embedding_lookup(decodificador_embeddings_matrix,
                                                             saidas_preprocessadas)
    previsoes_treinamento, previsoes_teste = rnn_decodificador(decodificador_embedded_entradas,
                                                               decodificador_embeddings_matrix,
                                                               codificador_estado,
                                                               numero_palavras_perguntas,
                                                               tamanho_sequencia,
                                                               rnn_tamanho,
                                                               numero_camadas,
                                                               perguntas_palavras_int,
                                                               keep_prob,
                                                               batch_size)
    return previsoes_treinamento, previsoes_teste
#Configuração dos hiperparâmetros
epocas = 100
batch_size = 64
rnn_tamanho = 512
numero_camadas = 3
tamanho_codificador_embeddings = 512
tamanho_decodificador_embeddings = 512
learning_rate = .01
learning_rate_decaimento = .9
min_learning_rate = .0001
probability_dropout = .5

# Definição da seção
tf.reset_default_graph()
session = tf.InteractiveSession()

# Carregamento do modelo
entradas, saidas, lr, keep_prob = entradas_modelo()

# Configuração do tamanho da sequência
tamanho_sequencia = tf.placeholder_with_default(25,None,name="tamanho_sequencia")

# Obtenção das dimensões dos tensores de entrada
dimensao_entrada = tf.shape(entradas)

# Obtenção das previsões de treinamento e teste
previsoes_treinamento, previsoes_teste = modelo_seq2seq(tf.reverse(entradas,[-1]),
                                                        saidas,
                                                        keep_prob,
                                                        batch_size,
                                                        tamanho_sequencia,
                                                        len(respostas_palavras_int),
                                                        len(perguntas_palavras_int),
                                                        tamanho_codificador_embeddings,
                                                        tamanho_codificador_embeddings,
                                                        rnn_tamanho,
                                                        numero_camadas,
                                                        perguntas_palavras_int)

