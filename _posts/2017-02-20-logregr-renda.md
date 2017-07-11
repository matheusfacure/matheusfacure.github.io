---
layout: post
comments: true
title: O Poder de Algoritmos Lineares para Prever Renda
subtitle: "Muitas vezes a simplicidade vence. Essa é uma delas."
date: 2017-02-20
tags: [Post]
author: "Matheus Facure"
header-img: "/img/fundo_main.png"
modal-id: 3
thumbnail: /img/portfolio/renda-logregr/modelcompare.png
description: Modelos complexos de aprendizado de máquina são ótimos, mas, às vezes, modelos lineares simples bastam e apresentam inúmeras vantagens. Neste post, mostro como a técnica de regressão logística se compara com ferramentas mais poderosas, como o algoritmo de k-vizinhos  mais próximos e um <em>Ensemble AdaBoost</em> de árvores de decisão.
---

<h2>Conteúdo</h2>
<ul>
	<li><a href="#intro">Introdução</a></li>
	<li><a href="#Dados">Os Dados</a></li>
	<li><a href="#Explo">Análise Exploratória</a></li>
	<li><a href="#Inferencial">Análise Inferencial</a>
<ul>
	<li><a href="#Aval">Métricas de Avaliação</a></li>
	<li><a href="#Treinamento">Treinamento e Avaliação dos Modelos</a></li>
</ul>
</li>
	<li><a href="#Resultados">Resultados</a></li>
	<li><a href="#ref">Referências</a></li>
</ul>
<h2 id="intro">Introdução</h2>
<p>Utilizando dados do censo, vamos prever quais indivíduos tem renda maior do que 50 mil dólares anuais. No processo, nós vamos mostrar também o poder e a simplicidade de algoritmos lineares, comparando-o com outros algoritmos de aprendizado de máquina mais complexos. Particularmente, nós veremos que o algoritmo de regressão logística, mesmo sendo extremamente simples, consegue um performance igual ou apenas levemente inferior a de algoritmos não lineares e mais complexos.</p>
<p><strong>Atenção!</strong> Este trabalho é uma adaptação do segundo projeto do <a href="https://br.udacity.com/course/machine-learning-engineer-nanodegree--nd009/">Nanodegree Engenheiro de Machine Learning</a> da Udacity. Se você está fazendo esse Nanodegree, saiba que uma série de sanções lhe podem ser aplicadas por submeter trabalho que não é seu. Assim, eu recomendo fortemente que não leia adiante antes de terminar o seu próprio projeto.</p>

<h2 id="Dados">Os Dados</h2>
<p>Os dados desse projeto são do censo americano de 1994. A base de dados foi retiradas do <em><a href="https://archive.ics.uci.edu/ml/datasets/Census+Income">UCI Machine Learning Repository</a> </em>e foi compilada por Ronny Kohavi e Barry Becker. Após retirar alguns dados mal formatados ou com entradas ausentes, restaram 45222 observações. A variável que queremos prever é 'income' (renda), que assumo 1 se a renda do indivíduo for maior do que 50 mil dólares anuais e 0 caso contrário. Para informações detalhadas sobre as variáveis independentes utilizadas como preditoras, olhe a referência da base de dados.</p>

<h2 id="Explo">Análise Exploratória</h2>
<p>Antes de utilizar os modelos preditivos, conduzimos uma série exploração dos dados para melhor entender os padrões e relações neles presentes. Em primeiro lugar, nós descobrimos que apenas 24,78% dos indivíduos tem renda superior aos 50 mil dólares anuais, o que significa um modelo ingênuo que preveja sempre renda menor do que isso conseguirá uma acurácia de 75,22%.</p>
<p>Em seguida, nós procuramos entender a distribuição das variáveis contínuas. Nos nossos dados, existem apenas duas delas, 'capital-gain' (ganhos de capital) e 'capital-loss' (perdas de capital). Nós então montamos os histogramas de cada uma delas.</p>

<img class="img-responsive center-block thumbnail" src="/img/portfolio/renda-logregr/capgainloss.png" alt="capgainloss.png"/>

<p>Nós notamos que as distribuições de ambas são bastante assimétricas e que maioria das pessoas não tem fonte de renda advinda de investimentos em capital (pois o valor com maior frequência tanto para ganhos quanto para perdas de capital é zero). Como algoritmos de aprendizado de máquina podem ser sensíveis a esse tipo de distribuição, nós aplicamos a transformação logarítmica nessas duas variáveis. Para evitar logaritmos de zeros (que são indefinidos), nós adicionamos 1 antes de aplicar a transformação. De forma resumida, nós substituímos as variáveis contínuas originais pelas suas versões transformadas da seguinte forma:</p>

$$ tranf (x) = log(x + 1) $$

<p>Após a transformação, os histogramas dessas variáveis assumiram o seguinte formato:</p>

<img class="img-responsive center-block thumbnail" src="/img/portfolio/renda-logregr/capgainlosstransf.png" alt="capgainlosstransf.png"/>

Em seguida, nós aplicamos <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">escalonamento </a><em><a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">min-max</a> </em>em todas as variáveis numéricas, para que todas elas assumissem a mesma escala de valores, sendo assim tratadas igualmente pelos algoritmos de aprendizado de máquina, no início do treinamento.  Por fim, nós codificamos as variáveis categóricas em <em>dummies </em> e separamos aleatoriamente um <em>set </em>de treino com 80% das amostras e o <em>set</em> de teste com o resto delas. Essa codificação final nos deixou com 103 variáveis, mas muitas delas esparsas, devido a codificação <em>one-hot </em>característica da transformação em <em>dummies</em>.</p>

<h2 id="Inferencial">Análise Inferencial</h2>
<h3 id="Aval">Métricas de Avaliação</h3>
<p>Como já discutimos acima, mais de 75% dos indivíduos nos nossos dados recebem menos do que 50 mil dólares anuais. Em outras palavras, podemos dizer que as <strong>classes dos nossos dados estão desbalanceadas</strong>. Assim, utilizar apenas métrica de acurácia pode resultar em má interpretação da qualidade do modelo, uma vez que simplesmente prever sempre uma renda abaixo de 50 mil resultará em mais do que 75% de acurácia. Assim, nós vamos utilizar também uma outra métrica de avaliação, chamada <em>F-score.</em></p>
<p>Para entender o <em>F-score</em> precisamos antes entender as métricas de <strong>precisão</strong> e <strong>revocação</strong>. A primeira delas nos dirá quanto dos indivíduos que prevemos como tendo renda maior do que 50 mil de fato tem renda maior do que 50 mil; a segunda métrica nos diz quando quanto dos indivíduos com renda superior a 50 mil nós conseguimos identificar como tendo renda superior a 50 mil. Formalmente, nós podemos definir essas métricas como sendo</p>

$$ P=\frac{T_p}{T_p + F_p} \quad \quad R=\frac{T_p}{T_p + F_n} $$

<p>Em que \( T_p\) são os verdadeiros positivos, \( F_p\) são os falsos positivos e \( F_n\) são os falsos negativos. De maneira intuitiva, a precisão nos penaliza caso classifiquemos muitas pessoas como alta renda, mas que são na verdade de baixa renda. A revocação, por outro lado, nos penaliza caso falhemos em identificar as pessoas de alta renda. Por exemplo, um classificador ingênuo que previsse baixa renda para todos os indivíduos obteria uma acurácia maior do que 75%, mas uma revocação de zero, pois não identificaria nenhuma pessoa de alta renda.</p>
<p>Por fim, o <em>F-score</em> é uma combinação dessas duas medidas e pode ser definido da seguinte maneira:</p>

$$ F_\beta = (1+\beta^2) \frac{P * R}{(\beta^2 * P) + R} $$

<p>Nesse caso particular, vamos utilizar \( \beta=0,5\) para dar mais ênfase à precisão.</p>

<h3 id="Treinamento">Treinamento e Avaliação dos Modelos</h3>
<p>Inicialmente, vamos treinar 3 classes de modelos <em>off-the-shelf </em>(prontos, com hiper-parâmetros padrões). O primeiro tipo de modelo será uma regressão logística, que é simplesmente um modelo linear para classificação. Explicações detalhadas sobre regressão logística e linearidade podem ser encontradas na seção de algoritmos deste site, <a href="https://matheusfacure.github.io/2017/02/25/regr-log/">nesta</a> e <a href="https://matheusfacure.github.io/2017/03/01/regr-poli/">nesta postagem</a>, respectivamente. O segundo modelo considerado será uma <a href="http://scikit-learn.org/stable/modules/ensemble.html#adaboost">árvore de decisão impulsionada</a>. Por fim, consideraremos um modelo de <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">k-vizinhos mais próximos</a>. A grande diferença entre a regressão logística e os outros dois modelos considerados é que os últimos não estão restritos a aprender apenas relações lineares entre os dados. Em outras palavras, os dois últimos algoritmos considerados são aproximadores universais de qualquer função, o que significa que eles podem aprender as mais variadas formas de superfícies de separação entre os indivíduos de alta renda e os de baixa renda.</p>
  
Nós treinamos os três modelos acima propostos e resumimos os resultados na seguinte imagem.  

<img class="img-responsive center-block thumbnail" src="/img/portfolio/renda-logregr/modelcompare.png" alt="modelcompare.png"/>

<p>No primeiro quadro (superior, à esquerda), podemos ver o tempo de treinamento de cada modelo e logo notamos que a regressão logística é superior aos outros dois independente da quantidade de dados utilizada. Nos dois quadros seguintes (superior, ao centro e à direita), vemos que o modelo linear é apenas um pouco pior do que os outros, tanto em termos de acurácia quanto em termos de <em>F-score, </em>ambas as métricas medidas no <em>set</em> de treinamento. Nos quadros de baixo, primeiro podemos ver que o tempo para efetivar as previsões é negligenciável, tanto para o modelo de regressão logística quanto para o de árvore de decisão impulsionada. Em se tratando das performances avaliadas no <em>set</em> de teste, podemos ver que o modelo de árvore de decisão impulsionada é apenas marginalmente superior ao modelo linear e que este é, por sua vez, superior ao modelo de k-vizinhos mais próximos.</p>

<h2 id="Resultados">Resultados</h2>
<p>Por fim, nós otimizamos o modelo de regressão logística com respeito ao seu hiper-parâmetro de regularização e os resultados foram resumidos na seguinte tabela.</p>

<img class="img-responsive center-block thumbnail" src="/img/portfolio/renda-logregr/resultslinmod.png" alt="resultslinmod" width="50%" />

<p>Podemos ver que a otimização resulta em pouca melhora. Isso acontece pelo fato de regressão logística já ser um modelo extremamente simples, no qual não há muito o que ajustar.</p>

<h2 id="ref">Referências</h2>
<p>O projeto original pode ser conferido (em inglês) no <a href="https://github.com/matheusfacure/Tutoriais-de-AM/tree/master/Exemplos">meu GitHub</a>. Além disso, na documentação do <em>TensorFlow</em> há <a href="https://www.tensorflow.org/tutorials/wide">um tutorial e uma discussão excelente</a> sobre a resolução do mesmo problema abordado aqui.</p>
<p></p>
