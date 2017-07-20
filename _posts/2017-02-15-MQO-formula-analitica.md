---
layout: tutorial
comments: true
title: Regressão Linear MQO
subtitle: "Aprenda a implementar o modelo estatístico mais utilizado nas ciências sociais."
date: 2017-02-15
update: 2017-07-19
tags: [Tutorial]
author: "Matheus Facure"
header-img: "img/dark-ann.jpg"
---

<div class="row">
<ul class="nav nav-tabs navbar-left">
    <li class="active"><a href="">Numpy</a></li>
    <li><a href="/2017/02/15/MQO-sklearn/">Scikit-Learn</a></li>
</ul>
</div>
 
<h2 id="Regressão-Linear">Conteúdo</h2>
<ul>
	<li><a href="#pre_requisitos">Pré-requisitos</a></li>
	<li><a href="#intuicao">Intuição</a></li>
	<li><a href="#justificativa_matematica">Justificativa matemática</a></li>
	<li><a href="#desenhando_e_testando_o_algoritmo">Desenhando e testando o algoritmo</a></li>
	<li><a href="#consideracoes_finais">Considerações finais</a></li>
</ul>

<h2 id="pre_requisitos">Pré-requisitos</h2>

<p>Vou pressupor que você tenha os conhecimentos especificados no tutorial sobre <a href="https://matheusfacure.github.io/2017/01/15/pre-req-ml/">matemática e programação para aprendizado de máquina</a>, isto é, que sabe cálculo (derivadas), o básico de álgebra linear, de estatística e de programação.</p>

<h2 id="intuicao">Intuição</h2>
<p>Vamos supor que tenhamos dados em tabela sobre duas variáveis: x e y. Se colocarmos cada par (x,y) em um gráfico, teremos uma figura como a seguinte:</p>

<img class="img-responsive center-block thumbnail" src="/img/tutorial/regr_lin_mqo/scatter.png" alt="scatter" width="388" height="291" />

<p>O que o algoritmo de regressão linear faz é simplesmente achar a reta que melhor se encaixa entre os pontos:</p>
<p>Assim, podemos prever (com erro) um valor de y dado um valor de x. Por exemplo, nós não temos uma observação em que \(x=1\), mas gostaríamos de prever qual seria o valor de y caso x fosse 1. Basta então olhar na linha qual valor de y quando x assume o valor 1. Na imagem acima, y seria aproximadamente 2.5 (ponto amarelo).</p>

<img class="img-responsive center-block thumbnail" src="/img/tutorial/regr_lin_mqo/scatter_line.png" alt="figure_1" width="388" height="291" />

<p>Ok. Esse exemplo é bem simples e meramente ilustrativo. Suponha agora que y não dependa mais apenas de x, mas de x e z. Bem, nesse caso, teríamos um gráfico em 3D e a regressão linear acharia o plano que melhor se encaixa nos dados. E para mais dimensões? Digamos que y dependa de 100 variáveis. Nós não podemos mais visualizar esse caso, mas sabemos que não é muito diferente dos casos 2D ou 3D, só que agora a regressão linear acha o hiperplano que melhor se encaixa nos dados. Se isso está um pouco abstrato e difícil de visualizar, pense  sempre em 3D quando trabalhando com muitas dimensões. É um truque muito útil que eu aprendi em um vídeo do <a href="https://en.wikipedia.org/wiki/Geoffrey_Hinton">professor de Geoffrey Hinton</a> e, segundo ele, todo mundo faz isso: quando tentar visualizar 100 dimensões, por exemplo, pense em 3D e grite para si mesmo "100D" e você conseguirá abstrair grandes dimensionalidades.</p>

<h2 id="justificativa_matematica">Justificativa matemática</h2>
<p>Imagine que temos dados em tabela, sendo que cada linha é uma observação e cada coluna uma variável. Então escolhemos uma das colunas para ser nossa variável dependente y (aquela que queremos prever) e as outras serão as variáveis independentes (X). Nosso objetivo é aprender como chegar das variáveis independentes na variável dependente, ou, em outras palavras, prever y a partir de X. Note que X é uma matriz nxd, em que n é o número de observações e d o número de dimensões; y é um vetor coluna nx1. Podemos definir o problema como um sistema de equações em que cada equação é uma observação:</p>

$$\begin{cases}
w_0 + w_1 x_1 + ... + w_d x_1 = y_1 \\
w_0 + w_1 x_2 + ... + w_d x_2 = y_2 \\
... \\
w_0 + w_1 x_n + ... + w_d x_n = y_n \\
\end{cases}$$

<p>Normalmente \( n > d\), isto é, temos mais observações que dimensões. Sistemas assim costumam não ter solução, pois há muitas equações e poucas variáveis para ajustar. Intuitivamente, pense que, na prática, muitas coisas afetam a variável y, principalmente se ela for algo de interesse das ciências humanas como, por exemplo, preço, desemprego, felicidade etc. Muitas das coisas que afetam y não podem ser coletadas como dados; desse modo, as equações acima não teriam solução porque não teríamos todos os fatores que afetam y.</p>
<p>Para lidar com esse problema, vamos adicionar nas equações um termo erro \( \varepsilon\) que representará os fatores que não conseguimos observar, erros de medição, etc.</p>

$$\begin{cases}
w_0 + w_1 x_{11} + ... + w_d x_{1d} + \varepsilon_1 = y_1 \\
w_0 + w_1 x_{21} + ... + w_d x_{2d} + \varepsilon_2 = y_2 \\
... \\
w_0 + w_1 x_{n1} + ... + w_d x_{nd} + \varepsilon_3 = y_n \\
\end{cases}$$

<p>Ou, em forma de matriz:</p>

$$\begin{bmatrix}
1 & x_{11} & ... & x_{1d} \\
1 & x_{21} & ... & x_{2d} \\
\vdots & \vdots& \vdots & \vdots \\
1 & x_{n1} & ... & x_{nd} \\
\end{bmatrix}
\times
\begin{bmatrix}
w_0 \\
w_1 \\
\vdots \\
w_d \\
\end{bmatrix}
+
\begin{bmatrix}
\varepsilon_0 \\
\varepsilon_1 \\
\vdots \\
\varepsilon_n \\
\end{bmatrix}
=
\begin{bmatrix}
y_0 \\
y_1 \\
\vdots \\
y_n \\
\end{bmatrix}$$

$$X_{nd} \pmb{w}_{d1} + \pmb{\epsilon}_{n1} = \pmb{y}_{n1}$$

<p>Para estimar a equação acima, usaremos a técnica de Mínimos Quadrados Ordinários (MQO): queremos achar os \( \pmb{\hat{w}}\) que minimizam os \( n\) \( \varepsilon^2 \), ou, na forma de vetor, \( \pmb{\epsilon}^T \pmb{\epsilon}\). Por que minimizar os erros quadrados? Assim como todo algoritmo de Aprendizado de Máquina, regressão linear também pode ser encarada como problemas de minimização de função custo. Então, nesse caso, nossa função custo é \( L = \pmb{\epsilon}^T \pmb{\epsilon}\). Um nome comum dessa função é o custo quadrático L2, pois nesse caso o custo é o quadrado da norma L2 do vetor \( \pmb{\epsilon}\). Note que nós poderíamos usar também a norma L1 do mesmo vetor como função custo. Ou ainda, poderíamos usar outras funções que adicionam uma penalidade também para o tamanho de \( \pmb{\hat{w}}\), como acontece nos algoritmos de regressão <a href="https://en.wikipedia.org/wiki/Tikhonov_regularization">Ridge</a> ou <a href="https://en.wikipedia.org/wiki/Lasso_(statistics)">Lasso</a>, mas isso terá que ficar para outro tutorial. Por hora, a soma dos mínimos quadrados bastará como função custo, até porque ela tem a vantagem de deixar a matemática muito mais simples:</p>

$$ \pmb{\epsilon}^T  \pmb{\epsilon} = (\pmb{y} - \pmb{\hat{w}}X)^T(\pmb{y} - \pmb{\hat{w}} X) \\= \pmb{y}^T \pmb{y} - \pmb{\hat{w}}^T X^T \pmb{y} - \pmb{y}^T X \pmb{\hat{w}} + \pmb{\hat{w}} X^T X \pmb{\hat{w}} \\= \pmb{y}^T \pmb{y} - 2\pmb{\hat{w}}^T X^T \pmb{y} + \pmb{\hat{w}} X^T X \pmb{\hat{w}} $$

<p>Aqui, usamos o fato de que \( \pmb{\hat{w}}^T X^T \pmb{y}\) e \( \pmb{y}^T X \pmb{\hat{w}}\) são simplesmente escalares \( 1x1\) e a transposta de um escalar é o mesmo escalar: \( \pmb{\hat{w}}^T X^T \pmb{y} = (\pmb{\hat{w}}^T X^T \pmb{y})^T = \pmb{y}^T X \pmb{\hat{w}}\). Derivando em \( \pmb{\hat{w}}\) e achando a CPO:</p>

$$ \frac{\partial \pmb{\epsilon}^T \pmb{\epsilon}}{\partial \pmb{\hat{w}}} = -2X^T\pmb{y} + 2X^T X \pmb{\hat{w}} = 0 $$

<p>Derivando mais uma vez para checar a CSO chegamos em \( 2X^TX\), que é positiva definida se as colunas de\( \pmb{X}\) forem independentes. Temos então um ponto de mínimo quando:</p>

$$ \pmb{\hat{w}} = (X^T X)^{-1} X^T \pmb{y} $$

<p>Bom, parece que chegamos em algo interessante. Nos nossos dados temos \( \pmb{X}\) e \( \pmb{y}\), então podemos achar \( \hat{\pmb{w}}\) facilmente: basta substituir os valores na fórmula. O próximo passo é desenhar o algoritmo e ver como ele se sai em dados reais.</p>
<p>OBS:</p>
<p>1) A maioria dessas informações eu tirei de várias fontes na internet, mas uma que abrange tudo e mais muitas extensões são os vídeos da gravação do curso de <a href="https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/">Aprendizado de Máquina da Universidade de Oxford (2014)</a>.</p>
<p>2) Para mais detalhes sobre a matemática de regressão linear como MQO, veja este <a href="https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf">passo a passo da Universidade de Stanford.</a></p>
<p>3) É possível chegar em uma fórmula para os vários \( \hat{w_i}\) apenas com cálculo multivariado, sem usar álgebra linear. Para os que tem dificuldade com álgebra linear (como é o meu caso), esse método parece mais atraente. No entanto, ele tem uma notação muito mais carregada de somatórios. Da minha experiência aprendendo isso, acho que vale muito a pena investir tempo para entender bem álgebra linear, pois todos os algoritmos de Aprendizado de Máquina ficam muito mais fáceis depois. Caso queira ver como resolver a equação de otimização do algoritmo de MQO sem usar álgebra linear, eu sugiro o livro do <a href="https://www.amazon.com.br/Introdu%C3%A7%C3%A3o-Econometria-Uma-Abordagem-Moderna/dp/8522104468/ref=sr_1_1/156-3678212-7923752?ie=UTF8&qid=1486067848&sr=8-1&keywords=econometria">Wooldrige</a>.</p>

<h2 id="desenhando_e_testando_o_algoritmo">Desenhando e testando o algoritmo</h2>
<p>Resumindo o que vimos acima de forma intuitiva, podemos dizer que o algoritmo de regressão linear por MQO acha uma linha, plano ou hiperplano que passa entre os dados de forma que a distância quadrada entre os pontos e a linha, plano ou hiperplano seja minimizada. Vamos agora implementar esse algoritmo em Python. Para realmente entender a mecânica do algoritmo, vamos implementá-lo usando apenas Numpy, um pacote de computação numérica. Isso será feito apenas por motivos pedagógicos e, na prática, recomendo aplicar regressão linear partindo de uma implementação já existente. Para isso, veja o <a href="/2017/02/15/MQO-sklearn/">tutorial de regressão linear com Scikit-Learn</a>.</p>

{% highlight python %}

import pandas as pd # para ler os dados em tabela
import numpy as np # para álgebra linear
from sklearn import linear_model, model_selection, datasets
import matplotlib.pyplot as plt # para fazer gráficos
from matplotlib import style
from time import time # para ver quanto tempo demora
np.random.seed(1) # para resultados consistentes 

class linear_regr(object):

    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        # adiciona coluna de 1 nos dados
        X = np.insert(X_train, 0, 1, 1)

        # estima os w_hat
        # (X^T * X)^-1 * X^T * y
        w_hat = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y_train)

        self.w_hat = w_hat
        self.coef = self.w_hat[1:]
        self.intercept = self.w_hat[0]

    def predict(self, X_test):
        X = np.insert(X_test, 0, 1, 1) # adiciona coluna de 1 nos dados
        y_pred = np.dot(X, self.w_hat) # X * w_hat = y_hat
        return y_pred


{% endhighlight %}

<p>Ok! Teoria justificada e algoritmo pronto. Vamos agora ver se ele consegue aprender os \( \hat{w_i}\) de dados reais. OBS: Os dados podem ser encontrados <a href="http://www.cengage.com/aise/economics/wooldridge_3e_datasets/">aqui</a>.</p>

{% highlight python %}

data = pd.read_csv('../data/hprice.csv', sep=',').ix[:, :6] # lendo os dados
data.fillna(-99999, inplace = True) # preenchendo valores vazios
X = np.array(data.drop(['price'], 1)) # escolhendo as variáveis independentes
y = np.array(data['price']) # escolhendo a variável dependente

# separa em bases de treino e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state = 1)
data.head(5)
{% endhighlight %}

<table class="table table-bordered table-striped table-hover" border="1">
<thead>
<tr style="text-align:right;">
<th>price</th>
<th>assess</th>
<th>bdrms</th>
<th>lotsize</th>
<th>sqrft</th>
<th>colonial</th>
</tr>
</thead>
<tbody>
<tr>
<td>300.0</td>
<td>349.1</td>
<td>4</td>
<td>6126</td>
<td>2438</td>
<td>1</td>
</tr>
<tr>
<td>370.0</td>
<td>351.5</td>
<td>3</td>
<td>9903</td>
<td>2076</td>
<td>1</td>
</tr>
<tr>
<td>191.0</td>
<td>217.7</td>
<td>3</td>
<td>5200</td>
<td>1374</td>
<td>0</td>
</tr>
<tr>
<td>195.0</td>
<td>231.8</td>
<td>3</td>
<td>4600</td>
<td>1448</td>
<td>1</td>
</tr>
<tr>
<td>373.0</td>
<td>319.1</td>
<td>4</td>
<td>6095</td>
<td>2514</td>
<td>1</td>
</tr>
</tbody>
</table>

<p>Treinando, testando e comparando o regressor:</p>

{% highlight python %}
t0 = time()
regr = linear_regr()
regr.fit(X_train, y_train)
print("Tempo do criado manualmente:", round(time()-t0, 3), "s")

# medindo os erros
y_hat = regr.predict(X_test) # prevendo os preços

print('Média do erro absoluto: ', np.absolute((y_hat - y_test)).mean())
print('Média do erro relativo: ', np.absolute(((y_hat - y_test) / y_test)).mean())

# comparando com o de mercado
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print("\n\nTempo do de mercado:", round(time()-t0, 3), "s")

# medindo os erros
y_hat = regr.predict(X_test) # prevendo os preços
w_hat = regr.intercept_
w_hat = np.append(w_hat, regr.coef_)

print('Média do erro absoluto: ', np.absolute((y_hat - y_test)).mean())
print('Média do erro relativo: ', np.absolute(((y_hat - y_test) / y_test)).mean())
{% endhighlight %}

Tempo do criado manualmente: 0.097 s  
Média do erro absoluto: 34.9234990043  
Média do erro relativo: 0.122915533711  
  
Tempo do de mercado: 0.254 s  
Média do erro absoluto: 34.9234990043  
Média do erro relativo: 0.122915533711  

<p>Nada mal... O preço previsto é, na média, apenas 12,2% diferente do preço real/observado. Note que o algoritmo aprendeu os parâmetros \( \hat{\pmb{w}}\) com uma parte dos dados e usou-os para prever dados que nunca tinha visto, mostrando uma boa capacidade de generalização.</p>

<p> Para comparação, utilizamos um modelo de regressão linear implementada no módulo  <a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">Scikit-Learn</a>, ao qual chamamos “de mercado”. O nosso algoritmo produz os mesmos resultados do de mercado, então podemos saber que não erramos nada. Além disso, o nosso algoritmo é mais rápido, mas essa diferença é insignificante, em termos práticos.

<h2 id="consideracoes_finais">Considerações Finais</h2>
<p>Eu considero o algoritmo de regressão linear como a base da ciência de modelagem estatística. Embora muito simples, regressão linear normalmente já te leva bem longe em termos de qualidade de previsão, enquanto os algoritmos de Aprendizado de Máquina mais complexos só fornecem uma melhora marginal em cima da qualidade adquirida com regressão linear. Assim sendo, termino com alguns conselhos práticos para usá-la.</p>

<ol>
	<li>Quando tentar prever um valor contínuo - como preço, demanda, ou um índice qualquer - sempre comece usando regressão linear antes de tentar outros algoritmos de AM mais complexos.</li>
	<li>Regressão linear possibilita uma excelente interpretação dos parâmetros \( \hat{w_i}\) encontrados e pode ser por isso considerado um modelo caixa branca. Infelizmente, a capacidade interpretativa depende de um aprofundamento que não é a intenção desse tutorial. Caso queira se aprofundar no algoritmo, veja <a href="https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf">1</a> ou <a href="https://www.coursera.org/learn/erasmus-econometrics">2</a>.</li>
	<li>Regressão linear por MQO tem um processo de treinamento muuuuuito rápido. Mesmo com milhões de dados, é possível estimar os parâmetros em menos de um segundo. Além disso, uma vez treinado, o regressor ocupa muito pouco espaço, pois só armazena o vetor \( \pmb{\hat{w}}\). Há uma exceção, no entanto. Regressão linear da forma que vimos depende da inversão de uma matriz, o que pode começar a demorar muito quando temos muitas dimensões nos nossos dados - digamos mais de 10000. Nesse caso, recomenda-se algum tipo de redução de dimensionalidade antes de treinar o algoritmo, ou otimizar a função custo de maneira iterativa ao invés de analiticamente.</li>
</ol>
<p>Vale uma nota de atenção: aqui só podemos abordar regressão linear brevemente. Ainda há problemas de inferência (saber se os coeficientes são estatisticamente significantes), de interpretação em outras escalas, de hipóteses assumidas e do que fazer quando tais hipóteses são violadas. Tenha isso em mente na hora de usá-lo! Muita coisa ficou incompleta aqui.</p>

<hr>

<ul class="pager">
  <li class="previous"><a href="https://matheusfacure.github.io/2017/01/15/pre-req-ml/">Anterior</a></li>
  <li class="next"><a href="https://matheusfacure.github.io/2017/02/20/MQO-Gradiente-Descendente/">Próximo</a></li>
</ul>

