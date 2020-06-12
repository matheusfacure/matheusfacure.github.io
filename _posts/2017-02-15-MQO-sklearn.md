---
layout: tutorial
comments: true
title: Aplicando Regressão Linear com <em>Scikit-Learn</em> 
subtitle: "Aprenda como aplicar esse poderoso modelo estatístico com menos de 15 linhas de código."
date: 2017-07-19
true-dt: 2017-07-19
tags: [Tutorial-alt]
author: "Matheus Facure"
---

<div class="row">
<ul class="nav nav-tabs navbar-left">
    <li><a href="/2017/02/15/MQO-formula-analitica/">Numpy</a></li>
    <li class="active"><a href="#">Scikit-Learn</a></li>
</ul>
</div>

## Pré-requisitos

Neste tutorial veremos como aplicar o modelo de regressão linear em dados reais para fazer previsões. Esta é uma versão alternativa do [tutorial de Regressão Linear MQO](/2017/02/15/MQO-formula-analitica/). Aqui, focaremos mais na aplicação e menos na teoria, vendo simplesmente como utilizar o modelo de regressão linear. Embora isso não seja um pré-requisito, recomendo fortemente que você veja o tutorial em que implementamos regressão linear sem ajuda de pacotes, apenas com Numpy. Isso te dará um excelente entendimento da técnica e te destacará daqueles que só a aplicam sem saber como ela funciona.

## Os Dados

Vamos tentar prever o preço de imóveis em Boston. Nossa tarefa será prever o preço mediano de uma vizinhança (\\(\pmb{y}\\)) a partir de variáveis \\(\pmb{X}\\), tais como taxa de criminalidade na região, número médio de quartos, índice de pobreza... Para mais informações sobre os dados, elas estão [na documentação do Scikit-Learn](http://scikit-learn.org/stable/datasets/index.html#boston-house-prices-dataset).

## Aplicando Regressão Linear

Em primeiro lugar, vamos importar algumas ferramentas do sklearn. `load_boston` será utilizado para carregar os dados, `train_test_split` será útil para separar os dados em um set de treino e um set de teste e `LinearRegression` implementará o algoritmo de regressão linear.

{% highlight python %}
from sklearn.datasets import load_boston # para carregar os dados
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # importa o modelo

# carrega os dados
house_data = load_boston()
X = house_data['data']
y = house_data['target']
{% endhighlight %}

Carregamos os dados invocando a função `load_boston()`. Isso retorna um dicionário contento as variáveis independentes (X) sob o nome `data` e as variáveis dependentes sob o nome `target`. Vamos acessar essas variáveis e alocá-las para `X` e `y` no Python.

{% highlight python %}
from sklearn.datasets import load_boston # para carregar os dados
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # importa o modelo

# carrega os dados
house_data = load_boston()
X = house_data['data']
y = house_data['target']
{% endhighlight %}

A próxima linha separa os dados em set de teste e de treino. [Lembre-se](http://127.0.0.1:4000/AM-Essencial/#Validação-cruzada) de que precisamos treinar e testar o modelo em dados diferentes para obter uma avaliação consistente do seu desempenho. 

{% highlight python %}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
{% endhighlight %}

Em seguida, criamos um regressor que implementa o algoritmo de regressão linear e o treinamos nos dados de treino.

{% highlight python %}
regr = LinearRegression() # cria o modelo
regr.fit(X_train, y_train) # treina o modelo
{% endhighlight %}

Finalmente, com o modelo treinado, podemos realizar previsões e ver quão próximas elas estão da realidade. Para isso, vamos utilizar a métrica \\(R^2\\), que nos diz o quanto da variação nos preços (\\(y\\)) é explicada pelo nosso modelo. Vamos computar o \\(R^2\\) para os dados de treino e de teste. 

{% highlight python %}
r2_train = regr.score(X_train, y_train)
r2_test = regr.score(X_test, y_test)
print('R2 no set de treino: %.2f' % r2_train)
print('R2 no set de teste: %.2f' % r2_test)
{% endhighlight %}
```
R2 no set de treino: 0.74
R2 no set de teste: 0.73
```
Nada mal! Nosso modelo explica 73% da variação nos preços dos imóveis! Além disso, podemos ver que seu desempenho é um pouco melhor no set de treino do que no de teste. Isso indica que ele está sofrendo um pouco com sobre-ajustamento. Esses dados contém apenas 506 observações, o que explica parte desse sobre-ajustamento. Para resolver esse problema, poderíamos coletar mais observações, retirar algumas variáveis pouco relevantes ou utilizar alguma técnica mais avançada de regularização (algo que ainda não vimos nos tutoriais passados). Você pode baixar o [código completo no meu GitHub](https://github.com/matheusfacure/Tutoriais-de-AM/blob/master/Regress%C3%A3o%20Linear/sk_linregr.py).
