---
layout: tutorial
comments: true
title: TensorFlow Essencial
subtitle: "Aprenda TensorFlow, o pacote criado pelo Google para treinar redes neurais. "
date: 2017-05-12
true-dt: 2017-05-12
tags: [Tutorial]
author: "Matheus Facure"
header-img: "img/dark-ann.jpg"
---

O [TensorFlow](https://www.tensorflow.org/) é uma biblioteca de computação numérica desenvolvida pelo Goolge e feita *open source* em 2015, tornando-se rapidamente a mais famosa e recomendada para *Deep Learning*. No início de 2017, o TensorFlow saiu da sua versão beta (de testes) e ganhou suporte para o Windows. Dentre as companhias que usam o TensorFlow estão Airbnb, Airbus, Dropbox, ebay, IBM, Intel, Snapchat, Twitter e Uber. Algumas das vantagens que fazem o TensorFlow se destacar das outras bibliotecas de redes neurais são;

* API principal em Python, com implementação de computações extremamente eficientes em C++. Também suporta utilização de GPUs para maior paralelização dos processamentos;
  
* Muitas APIs de alto nível, como [Keras](https://keras.io/) e [TFLearn](http://tflearn.org/), que tornam a implementação de redes neurais extremamente simples;
  
* Permite treinar redes neurais colossais com bilhões de dados. Isso não é nenhuma surpresa, considerando que o TensorFlow foi desenvolvido pelo Google justamente para esse propósito;
  
* Grande flexibilidade (ao custo de maior complexidade) no API original de Python, permitindo criar qualquer arquitetura de rede neural;
  
* Várias funções de otimização com implementação de diferenciação automática. Assim, não precisamos nos preocupar com os gradientes e podemos deixar essa tarefa para o TensorFlow;
  
* Uma ótima ferramenta de visualização chamada *TensorBoard*.

## Pré-requisitos

<p>Vou pressupor que você tenha os conhecimentos especificados no tutorial sobre <a href="https://matheusfacure.github.io/2017/01/15/pre-req-ml/">matemática e programação para aprendizado de máquina</a>, isto é, que sabe cálculo (derivadas), o básico de álgebra linear, de estatística e de programação. Eu também vou pressupor que você viu os tutoriais anteriores a esse. <a href="https://matheusfacure.github.io/tutorials/">Meus tutoriais </a> são ordenados de maneira lógica e sugiro fortemente que você se atenha à ordem deles para maior compreensão.</p>

## Conteúdo

<ol>
	<li><a href="#insta">Instalação</a></li>
	<li><a href="#tensor">Tensores</a></li>
	<li><a href="#fluxo">Fluxograma TensorFlow</a>
  <ol>
	  <li><a href="#construcao">Fase de Construção</a></li>
	  <li><a href="#execucao">Fase de Execução</a></li>
  </ol>
  </li>
	<li><a href="#consideracoes">Considerações Finais</a>
  <ol>
    <li><a href="#grafo">Criação do Grafo</a></li>
    <li><a href="#var">Criação de Variáveis</a></li>
    <li><a href="#sess">Interagindo com o TensorFlow na Sessão</a></li>
  </ol>
  </li>
	<li><a href="#ref">Referências</a></li>
</ol>


## Instalação <a name="insta"></a>

Se você está usando Linux, a instalação do TensorFlow deve ser tão simples como `pip3 install tensorflow`. Contudo, isso pode variar de sistema operacional para sistema operacional. Além disso, a instalação pode ser mais complicada se deseja-se utilizar suporte para GPU. Assim, eu prefiro recomendar a [referência oficial do TensorFlow para instalação](https://www.tensorflow.org/install/). Aqui, vamos utilizar a versão 1.1.0 do TensorFlow e não garanto que o que será desenvolvido se estenderá às outras versões sem necessidade de adaptações.

<img src="/img/tutorial/tf.jpg" alt="tf" class="img-responsive thumbnail pull-right" style="margin-left:3%; width: 40%;">

## Tensores <a name="tensor"></a>
Qualquer forma de dados no TensorFlow é representada pela estrutura de tensores, que são simplesmente *arrays* ou listas com \\(n\\) eixos. Em termos de álgebra linear, tensores são generalizações de matrizes. Por exemplo, um vetor é um tensor com uma dimensão ou um eixo. Uma matriz é um tensor com duas dimensões ou dois eixos. Uma pilha de matrizes é um tensor com  3 dimensões ou três eixos e assim por diante. É preciso ter cuidado para não confundir dimensões de um tensor – o número de eixos da estrutura de dados – com dimensão dos dados – o número de colunas ou variáveis em uma tabela de dados.

Os tensores tem tipos estáticos e todos os elementos devem ser do mesmo tipo. Eles também tem formato dinâmico, o que quer dizer que podemos reformatá-lo com operações de `reshape`, `transpose`, etc. Para representar o formato do tensor, usamos a notação entre colchetes `[ ]`. Assim, um tensor sem dimensão (um escalar) é representado pela forma `[]`, um tensor 1D (vetor) é representado pela forma `[k]`, um tensor 2D é representado pela forma `[k,m]` e assim por diante. Como um exemplo mais concreto, considera um lote de 100 imagens coloridas (RGB) e de tamanho 28x28 px. O tensor que armazenaria esses dados seria do formato `[n_lote, n_altura, n_largura, n_cores]`, ou, em números, `[100, 28, 28, 3]`.

## Fluxograma TensorFlow <a name="fluxo"></a>

Um programa TensorFlow é dividido em duas partes: 

1) Montagem do modelo ou grafo.  
2) Execução do grafo

Na primeira parte, nenhuma computação de fato acontece. Nós só estruturamos uma série de operações que serão rodadas na segunda parte. A execução do grafo, por sua vez, é tipicamente um loop de treinamento, no qual, a cada iteração, rodamos o grafo montado na primeira parte. Para entender melhor como isso acontece, vamos considerar como exemplo a construção do algoritmo de regressão linear. Em primeiro lugar, vamos baixar os dados para usar no nosso exemplo. Essa base diz respeito ao preço de casas em bairros da Califórnia. Nossa variável de interesse é o preço médio do bairro (em 100 mil dólares).


{% highlight python %}

import numpy as np # para computação numérica menos intensivas
import os # para criar pastas
from sklearn.datasets import fetch_california_housing # base de dados
from sklearn.metrics import r2_score # para avalização

import tensorflow as tf

X = fetch_california_housing().data # aqui pegamos as variáveis preditoras
y = fetch_california_housing().target # aqui pegamos a variável, independente

{% endhighlight %}

### Fase de Construção <a name="construcao"></a>

Agora que temos os dados, vamos para **construção do grafo** no TensorFlow. Um grafo, aqui, deve ser entendido como nada mais do que uma sequência de operações computacionais. Para montar o grafo, nós vamos criar uma instância `tf.Graph()` e dentro dela vamos colocar as operações e variáveis que quereremos executar mais para frente. O que criamos dentro do grafo é normalmente chamado de *nodes* (nós) do grafo. Esses nós podem ser operações (*ops*), como subtração e adição, ou variáveis.

Além disso, antes de montar o grafo eu gosto de definir algumas constantes que serão utilizadas tanto na montagem quanto na execução do grafo. Geralmente essas constantes são os hiper-parâmetros do modelo que montaremos e tê-los todos em um só lugar pode ajudar bastante no momento de ajustá-los. Particularmente, nesse caso vamos implementar uma **regressão linear com gradiente descendente**. Assim, precisamos definir o número de iterações de treino e a taxa de aprendizado. Para conveniência, eu também gosto de definir o formato dos dados. 


{% highlight python %}
# definindo constantes
lr = 1e-2 # taxa de aprendizado
n_iter = 2501 # número de iterações de treino
n_inputs = X.shape[1] # número de variáveis independentes
n_outputs = 1 # número de variáveis dependentes

graph = tf.Graph() # isso cria um grafo
with graph.as_default(): # isso abre o grafo para que possamos colocar operações e variáveis
    
    # adiciona as variáveis ao grafo
    W = tf.Variable(tf.truncated_normal([n_inputs, n_outputs], stddev=.1), name='Weight')
    b = tf.Variable(tf.zeros([n_outputs]), name='bias')

    ######################################
    # Monta o modelo de regressão linear #
    ######################################

    # Camadas de Inputs
    x_input = tf.placeholder(tf.float32, [None, n_inputs], name='X_input')
    y_input = tf.placeholder(tf.float32, [None, n_outputs], name='y_input')

    # Camada Linear
    y_pred = tf.add(tf.matmul(x_input, W), b, name='y_pred')

    # Camada de custo ou função objetivo
    EQM = tf.reduce_mean(tf.square(y_pred - y_input), name="EQM")

    # otimizador
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(EQM)

    # inicializador
    init = tf.global_variables_initializer()

    # para salvar o modelo treinado
    saver = tf.train.Saver()

{% endhighlight %}

Vamos agora destrinchar esse código para entender o que está acontecendo. Primeiro, nós definimos algumas constantes. Não há nenhum segredo aqui e sequer utilizamos o TensorFlow para isso. Em seguida, criamos um grafo e abrimos ele com o comando `with`. Dentro do grafo, nós começamos definindo as variáveis do modelo de regressão linear. Se você conhece o modelo, deve se lembrar de que ele é \\(\pmb{y}=\pmb{X}\pmb{w} + \pmb{b} + \pmb{\epsilon}\\) (se não, eu tenho [um tutorial](https://matheusfacure.github.io/2017/02/15/MQO-formula-analitica/) explicando isso também). Assim, as nossas variáveis são \\(\pmb{w}\\) e \\(\pmb{b}\\). Na nomenclatura de redes neurais, elas recebem os nomes de pesos e vieses, respectivamente. Aqui, nós vamos iniciar os pesos com pequenos valores aleatórios, segunda uma distribuição normal com média zero (padrão) e desvio padrão 0,1; o viés será inicializado em zero. Note como utilizamos a função `tf.Variable()` para definir as variáveis. Esse comando autoriza o TensorFlow a atualizar o valor numérico das variáveis durante o treinamento. Note também como passamos o formato das variáveis para sua construção. O argumento `name` não é obrigatório em nenhum momento aqui, mas  considerado boa prática nomear variáveis e operações no TensorFlow.

Uma vez que tenhamos as variáveis, partimos para a construção do modelo. Em primeiro lugar, definimos a camada de *input* com a função `tf.placeholder()`. Esses *placeholders* são espaços que serão alimentados com os nossos dados. Por hora, eles não contêm nada, mas nós já definimos o formato deles. Por exemplo, o *placeholder* para as variáveis independentes têm o formato `[None, n_inputs]`, em que `n_inputs` é a quantidade de variáveis independentes e usamos `None` para dizer que a dimensão de quantidade de observações poderá variar. Além disso, como primeiro argumento, passamos o tipo de variável que o *placeholders* irá receber e novamente nomeamos esses nós do grafo.

A seguir, definimos a camada linear \\(\pmb{X}\pmb{w} + \pmb{b} \\). Usamos `tf.matmul()` para multiplicação de matriz e `tf.add()` para adição. O resultado dessa operação será um vetor de previsões. Na camada de custo, nós vamos subtrair os \\(y\\)s reais dos \\(y\\)s previstos, elevar isso ao quadrado com `tf.square()` e tirar a média para produzir o erro quadrático médio. Isso finaliza o nosso modelo. Um detalhe é que podemos usar funções como `tf.add()` e o sinal `+` de maneira equivalente. Aqui, nós optamos por utilizar `tf.add()` para conseguir nomear a operação, o que é uma boa prática de organização do código.

Por fim, nós definimos um otimizador. Aqui, vamos  utilizar o otimizador Adam, que é uma versão melhorada do gradiente descendente; ele incorpora momento e taxa de aprendizado adaptativa. Independentemente do otimizador utilizado, ele toma conta tanto do cálculo do gradiente quanto de atualizar as variáveis. Note como passamos para ele tanto o que queremos minimizar (`EQM`) quanto a taxa de aprendizado.

As últimas operações do nosso grafo são um inicializador e um *saver*. O inicializador será utilizado para inicializar as variáveis que definimos mais acima. É importante ressaltar que até esse momento nenhuma computação foi realizada. Nem mesmo as variáveis foram inicializadas. Por fim, o *saver* faz o que se espera: salva **as variáveis** do modelo treinado. Isso é útil caso queiramos continuar o treinamento mais tarde ou realizar previsões depois. Termina assim nossa fase de construção do grafo.

### Fase de Execução <a name="execucao"></a>

Na fase de execução, tipicamente realizamos iterações de execuções do grafo e reportamos alguma métrica de progresso, como o valor da função custo. Nós não executamos as computações do grafo em Python, mas em alguma linguagem mais eficiente. Pense no TensorFlow como sendo uma linguagem em si, que é executada em baixo nível, fora do Python. Assim, para interagir com o grafo TensorFlow, precisamos abrir de uma sessão TensorFlow. A execução do grafo será feita por meio dela.


{% highlight python %}
# criamos uma pasta para salvar o modelo
if not os.path.exists('tmp'):
    os.makedirs('tmp')

# abrimos a sessão tf
with tf.Session(graph=graph) as sess:
    sess.run(init) # iniciamos as variáveis
    
    # cria um feed_dict
    feed_dict = {x_input: X, y_input: y.reshape(-1,1)}
    
    # realizamos as iterações de treino
    for step in range(n_iter + 1):
        
        # executa algumas operações do grafo
        _, l = sess.run([optimizer, EQM], feed_dict=feed_dict)
        
        if (step % 500) == 0:
            print('Custo na iteração %d: %.2f \r' % (step, l), end='')
            saver.save(sess, "./tmp/my_model.ckpt")

{% endhighlight %}


Em primeiro lugar, nós criamos uma pasta `tmp/`, onde salvaremos as variáveis do nosso modelo. Depois, abrimos a sessão para conseguir interagir com o grafo TensorFlow. Note como passamos como argumento para a sessão o grafo ciado anteriormente: `tf.Session(graph=graph)`. Dentro da sessão, a primeira coisa que fazemos é inicializar as variáveis. A função `sess.run()` executa um nó do grafo e nesse caso estamos executando o nó de inicialização das variáveis. Em seguida, nós criamos um `feed_dict`, um dicionário que referencia quais dados serão passados para quais *placeholders*. Lembre-se de que temos dois: um para colocar as variáveis independentes (`x_input`) e outro para colocar a variável dependente (`y_input`). Nosso *placeholder* da variável \\(y\\) é do formato `[None, 1]`, mas a nossa variável \\(y\\) é um array numpy de uma única dimensão. Por isso, usamos `y.reshape(-1,1)` para transformá-lo em um array de duas dimensões `[n_obs, 1]`.

Depois disso, iniciamos o loop de treinamento. Essa é a parte mais importante da fase de execução. Nesse loop, vamos rodar as iterações de treinamento do modelo. Para isso, novamente usamos `sess.run()`, só que dessa vez o primeiro argumento é uma lista de nós do grafo para rodarmos. Dentre esses argumentos, passamos a operação de treino `optimizer`. Essa operação minimiza a função `EQM`, e por isso depende dela, assim, mesmo se só tivessemos passado `optimizer` como argumento para `sess.run()`, todas as operações necessárias para calcular `EQM` são computadas também, isto é, o próprio nó `EQM` e a multiplicação de matriz na camada linear. Como essa última operação depende dos valores nos *placeholders*, nós precisamos passar como argumento de `sess.run()` o `feed_dict` que mapeia quais dados serão colocados no *placeholder*. Um detalhe importante é que passamos para `sess.run()`, na lista de nós para rodar, `EQM`. Isso não é necessário para o treinamento, já que ao rodar `optimizer` a operação `EQM` já será rodada pelo fato de `optimizer` depender dela. No entanto, nós queremos trazer o valor numérico de `EQM` para o Python e reportá-lo a cada 500 iterações de treino. O fato de você não ver o que está sendo executado nos bastidores pelo TensorFlow é algo que torna essa biblioteca particularmente contra-intuitiva. Para entender o que está acontecendo, lembre-se que **ao usar `sess.run()` em um nó do grafo, todos os nós dos quais ele depende também serão executados**. No caso do nó `optimizer`, mais duas coisas acontecem: (1) os gradientes são calculados e (2) as variáveis são atualizada conforme a regra definida pelo otimizador.

Por fim, a última etapa da iteração de treino é um condicional que executa seu conteúdo a cada 500 iterações. Dentro dele, nós primeiro reportamos o custo daquela iteração e depois salvamos o modelo com o nome `my_model.ckpt` na pasta `tmp/`. Uma dúvida que pode surgir aqui é porque estamos calculando o valor da função custo a cada iteração se apenas o reportamos a cada 500 iterações? Isso não está deixando nosso treinamento ineficiente? A resposta é não, pois, ao executar `optimizer`, já estamos incorrendo no custo computacional de calcular `EQM` e o custo de alocá-lo para uma variável é mínimo. Um outro detalhe é que salvar o modelo no disco geralmente é uma operação demorada, então fazer isso com muita frequência tornará o treinamento mais lento. 

Após treinado, nós podemos restaurar o modelo salvo para fazer previsões. Novamente, isso tem que ser feito dentro de uma sessão TensorFlow. Agora, no entanto, em vez de inicializarmos as variáveis como tínhamos feito antes, nós restauramos o modelo treinado utilizando o *saver*.


{% highlight python %}
# novamente, abrimos a sessão tf
with tf.Session(graph=graph) as sess:
    
    # restauramos o valor das variáveis 
    saver.restore(sess, "./tmp/my_model.ckpt", )
    
    # rodamos o nó de previsão no grafo
    y_hat = sess.run(y_pred, feed_dict={x_input: X})
    
    print('\nR2: %.3f' % r2_score(y_pred=y_hat, y_true=y))

{% endhighlight %}
    
{% highlight bash %}
>>> R2: 0.499
{% endhighlight %}


Acima, após restaurar o modelo, nós rodamos o nó `y_pred`. Note como esse nó depende apenas do *placeholder* das variáveis independentes, por isso criamos um `feed_dict` mapeando apenas o que passaremos para o *placeholder* `x_input`. Por fim, nós reportamos o \\(R^2\\) do modelo. Note que esse \\(R^2\\) diz respeito aos dados de treinamento, mas como esse é um tutorial de TensorFlow e não de aprendizado de máquina não me dei ao trabalho de separar os dados em duas subamostras. Na prática, isso terá que ser feito. Sobre a restauração do modelo, é preciso fazer um **aviso importante**. O comportamento padrão de `saver.restore()` é restaurar todas as variáveis com os seus **nomes originais**. Isso significa que antes de restaurar o modelo, nós precisamos construir o grafo **mantendo os nomes das variáveis do modelo salvo!**. Por exemplo, rodar o código acima sem ter antes feito a fase de construção como fizemos resultaria em um `NotFoundError`.

## Considerações Finais <a name="consideracoes"></a>

Antes de prosseguir, vamos relembrar a estrutura desenvolvida aqui para um programa TensorFlow


<ol>
  <li>Definir constantes (e.g. hiper-parâmetros)</li>
  <li>Fase de construção do grafo
    <ol>
      <li>Definir variáveis</li>
      <li>Definir <em>placeholders</em> e montar o modelo</li>
      <li>Definir as operações de treino (e.g. <code class="highlighter-rouge">optimizer</code>)</li>
      <li>Definir auxiliares (e.g. <code class="highlighter-rouge">init</code> e <code class="highlighter-rouge">saver</code>)</li>
    </ol>
  </li>
  <li>Fase de execução do grafo
    <ol>
      <li>Inicializar ou restaurar variáveis</li>
      <li>Loop de treinamento
        <ol>
        <li>Montar definir quais dados passar ao grafo</li>
        <li>Rodar uma iteração de treino</li>
        <li>Mostrar alguma métrica de progresso e salvar o modelo (opcional)</li>
        </ol>
        </li>
      <li>Fazer previsões e reportar métrica de performance</li>
    </ol>
    </li>
</ol>  

O que vimos até aqui aborda todo o conteúdo essencial do TensorFlow e com pequenas mudanças no código acima já será possível treinar redes neurais. Particularmente, nós vimos que o TensorFlow mesmo sendo feito principalmente para Python é quase uma linguagem de computação em si, com regras próprias e bem diferentes das do Python. E como toda linguagem de computação, no TensorFlow há varias formas de realizar a mesma tarefa. Acima, nós só abordamos uma, mas vale a pena destacar alguns comandos equivalentes bem comuns, principalmente para facilitar na leitura de códigos de outras pessoas. Começando pela criação do grafo.

### Criação do Grafo <a name="grafo"></a>

Eu disse que para criar um grafo TensorFlow era preciso utilizar os comandos


{% highlight python %}
# fase de construção
some_graph = tf.Graph() # cria o grafo
with some_graph.as_default():
    [...] # adiciona nós ao grafo

# fase de execução
with tf.Session(graph=some_graph) as sess:
    [...] # executa o grafo
{% endhighlight %}

Isso não é de todo verdade. Embora recomendado, você pode criar o grafo sem usar `tf.Graph()` e o TensorFlow adicionará as operações/nós a um grafo padrão. Assim, as fases de construção e execução seriam da seguinte forma


{% highlight python %}
# fase de construção
[...] # criação o grafo

with tf.Session() as sess:
    [...]  # executa o grafo
{% endhighlight %}

A primeira forma é mais organizada, pois deixa bem clara a divisão entre as duas fases de um típico programa TensorFlow. Além disso, a primeira opção é útil quando estamos trabalhando com mais de um grafo. Mesmo assim, a segunda opção é menos formal e eu mesmo acabo optando por ela muitas vezes.

### Criação de Variáveis <a name="var"></a>

Ao criar variáveis, nós usamos `tf.Variable()`. Outra opção seria utilizar `tf.get_variable()`, que cria uma variável nova, caso ela ainda não exista, ou reusa a variável, caso ela já exista. A funcionalidade reutilizar uma variável é mais complexa e não será discutida aqui, já que nosso objetivo é apenas criar variáveis. No entanto, é preciso dizer que simplesmente usar `tf.get_variable()` com uma variável que já existe resultará em um erro; isso é feito para evitar reutilizar variáveis sem querer.

A vantagem de utilizar `tf.get_variable()` é que podemos passar como argumento um inicializador. Isso pode não parecer importante agora, mas um dos fatores que possibilitou o recente sucesso das redes neurais é saber como inicializar os seus parâmetros. No futuro, veremos que fazer isso na mão pode ser entendiante. Assim, utilizamos `tf.get_variable()` para criar variáveis da seguinte forma.

{% highlight python %}
tf.reset_default_graph() # limpa o grafo de todas as variáveis
W = tf.get_variable('Weight', shape=[n_inputs, n_outputs],
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
{% endhighlight %}

Se você costuma usa notebooks Jupyter, sabe que uma célula costuma ser executada mais de uma vez. Rodar duas vezes uma célula com o código acima **sem `tf.reset_default_graph()`** resultaria em erro por chamar `tf.get_variable()` com variáveis já adicionada ao grafo. Por isso, nós primeiro limpamos o grafo de todas as variáveis com `tf.reset_default_graph()`. Note que `tf.reset_default_graph()` excluirá do grafo todos os nós adicionados ao grafo até aquele ponto, então tome cuidado para não utilizá-lo no meio da criação de um grafo! Note também como passamos para `tf.get_variable()` um inicializador. Nesse caso, usamos simplesmente `tf.truncated_normal_initializer` para iniciar a variável com pequenos valores aleatórios.

### Interagindo com o TensorFlow na Sessão <a name="sess"></a>

Para interagir com o TensorFlow, utilizamos dentro da sessão `sess.run()` nas operações que queremos executar. Esse método é o mais geral e funciona tanto para acessar o valor de variáveis quanto para executar operações como a de treino. No entanto, além dele, temos outra forma de trazer o valor de variáveis do TensorFlow para o Python. Essa outra opção é usar o método `.eval()` das próprias variáveis. Ainda assim, esse método **deve ser usado dentro de uma sessão** e não funciona com operações. Lembre-se de que `.eval()` funciona com tensores, mas não funciona operações. Para rodar operações, temos uma terceira opção de interação que é usar o método `.run()` de uma operação ou nó do grafo. Esse método não funciona com **tensores** (como matrizes e vetores). Por exemplo, suponha que queiramos restaurar o nosso modelo treinado, executar mais 100 iterações de treino e então saber qual  é valor do viés `b` do modelo após esse treinamento adicional. Ou então, que queremos saber qual foi o valor final da função custo. Isso pode ser feito da seguinte forma:


{% highlight python %}
# novamente, abrimos a sessão tf
with tf.Session(graph=graph) as sess:
    
    # restauramos o modelo salvo
    saver.restore(sess, "./tmp/my_model.ckpt", )
    
    # treinamos o modelo por mais 100 iterações
    for i in range(101):
        optimizer.run(feed_dict={x_input: X, y_input: y.reshape(-1,1)})
    
    # duas formas de acessar o valor de variáveis
    print(sess.run(b))
    print(b.eval())
    
    # acessa o valor final da função custo
    print(EQM.eval(feed_dict={x_input: X, y_input: y.reshape(-1,1)}))
{% endhighlight %}

{% highlight bash %}
>>> INFO:tensorflow:Restoring parameters from ./tmp/my_model.ckpt
>>> [ 0.01486106]
>>> [ 0.01486106]
>>> 0.664753
{% endhighlight %}
    
Uma dúvida que geralmente surge é quando usar `sees.run()` e quando usar `.run()` ou `.eval()`. 

No caso de rodar `EQM` com `.eval()`, nós também precisamos passar um `feed_dict`, já que esse nó do grafo depende dos *placeholders*. O mesmo se aplica a usar `.run()` na operação `optimizer`. Com todas essas opções, qual delas devemos usar? Uma diferença fundamental é que podemos passar vários nós para `sess.run()`, ao passo que os métodos `.run()` e `.eval()` só executam o nó em questão. Assim, eu recomendo usar sempre `sees.run()`. Não apenas por motivos de padronização, mas porque quase sempre é mais eficiente. Por exemplo, se quisermos executar duas operações encadeadas no grafo, como a função custo `EQM` e a operação de treino `optimizer`, podemos fazê-lo usando `.eval()` e `.run()`, uma para cada nó, respectivamente. Mas isso resulta em duas execuções idênticas do grafo, até o ponto de `EQM`, pelo menos. Usar `sees.run()` nessas duas operações, por outro lado, resulta em uma única execução do grafo. Primeiro, o grafo será executado até `EQM` e o valor numérico dessa variável será trazido ao Python. Depois, o grafo continua a rodar a partir de `EQM`, executando a operação `optimizer`, que minimiza `EQM`.

Com isso finalizamos esse tutorial de TensorFlow essencial. Espero que não tenha sido complicado e que eu não tenha deixado coisas por explicar. Qualquer dúvida ou sugestão, não exite em entrar em contato. Você também pode comentar suas dúvidas e sugestões no final desse post.

## Referências <a name="ref"></a>

O TensorFlow é extremamente recente e mesmo a documentação original do pacote não é muito boa. Mesmo assim, ela pode ser útil em alguns casos, principalmente os [tutoriais mais básicos](https://www.tensorflow.org/get_started/). Fora isso, eu recomendo a segunda parte do livro [Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291), que foi o conteúdo mais claro que li sobre o TensorFlow. 

Uma outra opção para tutoriais é procurar vídeos no YouTube. Quanto a isso, tenho duas playlists para recomendar:

1) [TensorFlow, por Hvass Laboratories](https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ)  
2) [TensorFlow, por 周莫烦](https://www.youtube.com/playlist?list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f)

Eu também coloquei esse tutorial em um [notebook Jupyter no GitHub](https://github.com/matheusfacure/Tutoriais-de-AM/blob/master/Redes%20Neurais%20Artificiais/Tutorial%20de%20TensorFlow.ipynb). Eu recomendo que você baixe o código e execute as células, colocando `print()` nos lugares que não entender. No final, também coloquei um exercício para fixar o que expliquei aqui. A tarefa é adaptar o código para implementar regressão logística. Dica: será preciso mudar a função custo e adicionar uma camada logística. 
