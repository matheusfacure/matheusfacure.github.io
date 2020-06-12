---
layout: tutorial
tags: [Tutorial]
comments: true
title: Tudo Mais Constante em de MQO
subtitle: "Um aprofundamento teórico sobre a estimação da Regressão Linear"
date: 2017-02-16
true-dt: 2019-01-15
author: "Matheus Facure"
---

## Conteúdo
1. [Introdução](#intro)
2. [Causalidade e Ceteris Paribus](#ceterisparibus)
3. [Intuição](#intuicao)
4. [Fundamentos Matemáticos](#fundamentos)
5. [Parcialização e o Teorema Frisch-Waugh-Lovell](#fwl)
6. [Implementação](#implementacao)
7. [Referências](#ref)

## Pré-requisitos

Vou pressupor que você tenha os conhecimentos especificados no tutorial sobre <a href="https://matheusfacure.github.io/2017/01/15/pre-req-ml/">matemática e programação para aprendizado de máquina</a>, isto é, que sabe cálculo (derivadas), o básico de álgebra linear, de estatística e de programação. Eu também vou pressupor que você viu os tutoriais anteriores a esse. <a href="https://matheusfacure.github.io/tutorials/">Meus tutoriais </a> são ordenados de maneira lógica e sugiro fortemente que você se atenha à ordem deles para maior compreensão.

<a name="intro"></a>
## Introdução

O modelo de regressão linear é de longe o mais utilizado em econometria. Assim, acho que vale um aprofundamento em como esse modelo funciona. A ideia deste tutorial é justamente promover esse aprofundamento, desenvolvendo um entendimento algébrico e intuitivo de como interpretá-lo. A ideia é mostrar como MQO pode ir além das previsões, provendo anlálises de como as variáveis independentes \\(x\\) se relacionam com a variável dependente que queremos modelar \\(y\\).

<a name="ceterisparibus"></a>
## Causalidade e Ceteris Paribus

Na maioria dos estudos estatísticos aplicados às ciências sociais se está interessado no **efeito causal** que uma variável (por exemplo, salário mínimo) tem em outra (por exemplo, desemprego). Esse tipo de relação é extremamente difícil de descobrir. Muitas vezes, o que se consegue encontrar é uma associação (correlação) entre duas variáveis. Infelizmente, sem o estabelecimento de uma relação causal, apenas correlação não nos fornece uma base solida para tomada de decisão (por exemplo, subir ou baixar o salário mínimo para diminuir o desemprego).

Um conceito bastante relevante para a análise causal é o de *ceteris paribus*, que significa `todos outros fatores (relevantes) mantidos constantes`. A maioria das questões econometricas são de natureza *ceteris paribus*. Por exemplo, quando se deseja saber o efeito da educação no salário, queremos manter inalteradas outras variáveis relevantes, como por exemplo a renda familiar. O problema é que raramente é possível manter literalmente "tudo mais constante". A grande questão em estudos sociais empíricos sempre é então se há suficientes fatores relevantes sendo controlados (mantidos constantes) para possibilitar a inferência causal.

Obviamente, a maioria dos dados que um pesquisador das ciências sociais encontra não está na forma de testes experimentais, onde pode-se artificialmente controlar variáveis. Muito mais comuns são os dados onde tudo varia em conjunto: renda e educação sobem juntas com salário; investimentos policiais sobem juntos com a taxa de criminalidade. Felizmente, veremos como o modelo de regressão linear pode simular uma situação *ceteris paribus* e possibilitar inferência causal.


<a name="intuicao"></a>
## Intuição

A melhor forma de conseguir uma intuição da interpretação de um modelo MQO é com um exemplo. Vamos tentar entender o papel da educação no salário usando os dados [cps09mar](https://www.ssc.wisc.edu/~bhansen/econometrics/) de pesquisas do *Bureau of Labor Statistics*, um departamento do censo americano. Nossas variáveis de interesse são salário (y), medido em dólares por horas, e educação (x) medida em anos de estudo. Discrições mais detalhadas podem ser encontradas na [documentação dos dados](https://www.ssc.wisc.edu/~bhansen/econometrics/cps09mar_description.pdf).

O primeiro passo é ler os dados. Além disso, vamos converter o salário de dólares por ano em dólares por hora. Depois, vamos adicionar uma variável de anos no mercado de trabalho (tenure), que é definida como a idade, menos anos de educação menos 6 (com isso, estamos assumindo que uma pessoa começa a estudar aos 6 anos e é contratada logo após parar de estudar). Por fim, vamos excluir pessoas que não tem salário (`earnings==0`)

```python
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import inv

variables = {0: "age",
             1: "female",
             3: "education",
             4: "earnings",
             5: "hours",
             6: "week"}


dataset = (pd.read_csv("./data/cps09mar.txt", sep = "\t", header=None)
            .rename(index=str, columns=variables)
            .assign(earnings=lambda df: df["earnings"] / (df["hours"] * df["week"]),
                    tenure=lambda df: df["age"] - df["education"] - 6)
            .query("earnings>0")
            .loc[:, list(variables.values()) + ["tenure"]])

dataset.head()
```
<table class="table table-striped table-bordered table-hover">
   <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>female</th>
      <th>education</th>
      <th>earnings</th>
      <th>hours</th>
      <th>week</th>
      <th>tenure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52</td>
      <td>0</td>
      <td>12</td>
      <td>62.393162</td>
      <td>45</td>
      <td>52</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>0</td>
      <td>18</td>
      <td>21.367521</td>
      <td>45</td>
      <td>52</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>0</td>
      <td>14</td>
      <td>15.686275</td>
      <td>40</td>
      <td>51</td>
      <td>18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41</td>
      <td>1</td>
      <td>13</td>
      <td>22.596154</td>
      <td>40</td>
      <td>52</td>
      <td>22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>0</td>
      <td>13</td>
      <td>62.125000</td>
      <td>50</td>
      <td>52</td>
      <td>23</td>
    </tr>
  </tbody>
</table>

Agora, vamos regredir o logaritmo do salário em educação, idade e sexo. Usar o logaritmo é comum quando queremos interpretar o impacto de educação num aumento **percentual** no salário. O logaritmo nos dá essa interpretação.


```python
import statsmodels.formula.api as smf
results = smf.ols('np.log(earnings) ~ education + age + female', data=dataset).fit()
results.summary().tables[1]
```

<table class="table table-striped table-bordered table-hover">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    1.1501</td> <td>    0.016</td> <td>   71.436</td> <td> 0.000</td> <td>    1.119</td> <td>    1.182</td>
</tr>
<tr>
  <th>education</th> <td>    0.1082</td> <td>    0.001</td> <td>  114.390</td> <td> 0.000</td> <td>    0.106</td> <td>    0.110</td>
</tr>
<tr>
  <th>age</th>       <td>    0.0095</td> <td>    0.000</td> <td>   42.276</td> <td> 0.000</td> <td>    0.009</td> <td>    0.010</td>
</tr>
<tr>
  <th>female</th>    <td>   -0.2629</td> <td>    0.005</td> <td>  -50.165</td> <td> 0.000</td> <td>   -0.273</td> <td>   -0.253</td>
</tr></table>

Esqueça as outras colunas por hora e vamos focar na `coef`. Podemos ver que o coeficiente de educação tem um valor de `0.1082`. Isso que dizer que, segundo esse modelo, devemos esperar um aumento de, **em média**, 10% no salário para cada ano *a mais* de educação. Mais do que isso, essa análise **mantém idade e sexo constantes**, isto é, para pessoas da mesma idade e do mesmo sexo, devemos esperar um aumento de 10% no salário para cada ano *a mais* de educação. Também podemos analisar os outros coeficientes de maneira análoga: esse modelo mostra que mulheres ganham, em média, 26% menos que homens na mesma faixa etária em com o mesmo nível de escolaridade. 

O exemplo acima mostra como regressão linear pode providenciar uma interpretação *ceteris paribus* (tudo mais constante) mesmo quando os dados não foram coletados de maneira *ceteris paribus*. Na explicação acima, pode parecer que coletamos dados de pessoas com diferentes salários mais mantendo idade, educação e sexo constante. **Isso não é o caso!** Em estudos sociais a maioria dos dados são puramente observacionais e raramente temos o luxo de coletar dados de maneira experimental, mantendo outros fatores relevantes contates. Por isso recorremos à regressão linear. Ela nos dá o poder de fazer com dados observacionais o que outras ciências fazem em laboratório: manter variáveis constantes.


<a name="fundamentos"></a>
## Fundamentos Matemáticos

Embora a intuição da análise *ceteris paribus* em regressão linear seja simples, a teoria matemática por trás dessa intuição é um pouco densa. Por isso, antes de irmos a ela, é preciso desenvolver alguns fundamentos sobre a mecânica algébrica do modelo de regressão linear. Embora esses fundamentos serão utilizados aqui para explicar a intuição *ceteris paribus*, eles também te darão uma ótima base para entender outras propriedades bem interessantes da regressão linear que infelizmente não cabe aqui. Assim, espero que o que segue, embora mais complicado, seja extremamente útil para expandir seu entendimento de regressão linear para outras aplicações.

Em primeiro lugar, vamos relembrar algumas equações do modelo linear.

$$\pmb{y}=\pmb{X}\pmb{\beta}+\pmb{e}$$

$$\pmb{\hat{\beta}}=(\pmb{X}^T\pmb{X})^{-1}\pmb{X}^T\pmb{y}$$

$$\pmb{\hat{e}}=\pmb{y}-\pmb{X}\pmb{\hat{\beta}}$$

$$\pmb{X}^T\pmb{\hat{e}}=\pmb{0}$$

A primeira equação acima explicita nossa escolha de modelar \\(\pmb{y}\\) como uma combinação linear das colunas de \\(\pmb{X}\\) (variáveis) mais um erro \\(\pmb{e}\\) com média condicional zero \\(E(\pmb{e} \| \pmb{x})=0\\). É importante ressaltar que essa equação representa um **modelo teórico** e que nem \\(\pmb{\beta}\\), nem \\(\pmb{e}\\) são observáveis. A segunda equação mostra o estimador MQO (Mínimos Quadrados Ordinários) \\(\pmb{\hat{\beta}}\\) de \\(\pmb{\beta}\\). Como um subproduto da estimação de \\(\pmb{\hat{\beta}}\\), obtemos os **resíduos**, representados na terceira equação pela diferença entre os valores reais e estimados de \\(y\\). Os resíduos \\(\pmb{\hat{e}}\\) não devem ser confundidos com o erro \\(\pmb{e}\\), que são **não observáveis**. Os resíduos tem média zero e são não correlacionados com \\(\pmb{X}\\), fato que é dado pela quarta equação acima. Note que isso é um **resultado algébrico** que não depende de hipótese alguma e é valido para qualquer estimador linear.

<details><summary>Prova</summary>
$$
\begin{align}
\pmb{X}^T\pmb{\hat{e}} & = \pmb{X}^T(\pmb{y}-\pmb{X}\pmb{\hat{\beta}}) \\
& = \pmb{X}^T(\pmb{y}-\pmb{X}(\pmb{X}^T\pmb{X})^{-1}\pmb{X}^T\pmb{y}) \\
& = \pmb{X}^T\pmb{y}-\pmb{X}^T\pmb{X}(\pmb{X}^T\pmb{X})^{-1}\pmb{X}^T\pmb{y}) \\
& = \pmb{X}^T\pmb{y}-\pmb{I}\pmb{X}^T\pmb{y} \\
& =\pmb{0}
\end{align}
$$
</details>


### Matriz de Projeção

Defina uma matriz \\(\pmb{P}\\) tal que 

$$ \pmb{P}=\pmb{X}(\pmb{X}^T\pmb{X})^{-1}\pmb{X}^T$$

\\(\pmb{P}\\) tem a propriedade de mapear \\(\pmb{y}\\) em \\(\pmb{\hat{y}}\\), por isso também leva o nome de matriz chapéu.

$$\pmb{P}\pmb{y}=\pmb{\hat{y}}$$

Outras propriedades úteis de \\(\pmb{P}\\) é idempotencia (\\(\pmb{P}\pmb{P}=\pmb{P}\\)) e simetria \\(\pmb{P}^T=\pmb{P}\\). Além disso, note que \\(\pmb{P}\pmb{X}=\pmb{X}\\)

<details><summary>Provas</summary>
$$
\begin{align}
\pmb{P}\pmb{y} & = \pmb{X}(\pmb{X}^T\pmb{X})^{-1}\pmb{X}^T\pmb{y} \\
& = \pmb{X}\pmb{\hat{\beta}} \\
& = \pmb{\hat{y}}\\
\end{align}
$$

$$
\begin{align}
\pmb{P}\pmb{P} & = \pmb{P}\pmb{X}(\pmb{X}^T\pmb{X})^{-1}\pmb{X}^T \\
& = \pmb{X}(\pmb{X}^T\pmb{X})^{-1}\pmb{X}^T \\
& = \pmb{P}\\
\end{align}
$$

$$
\begin{align}
\pmb{M}\pmb{X} & = \pmb{X}(\pmb{X}^T\pmb{X})^{-1}\pmb{X}^T\pmb{X} \\
& = \pmb{X}\pmb{I} \\
& = \pmb{X}\\
\end{align}
$$

$$
\begin{align}
\pmb{P}^T & = (\pmb{X}(\pmb{X}^T\pmb{X})^{-1}\pmb{X}^T)^T \\
& = (\pmb{X}^T)^T\bigg((\pmb{X}^T\pmb{X})^{-1}\bigg)^T(\pmb{X})^T \\
& = \pmb{X}\bigg((\pmb{X}^T\pmb{X})^T\bigg)^{-1}\pmb{X}^T \\
& = \pmb{X}\bigg((\pmb{X})^T(\pmb{X}^T)^T\bigg)^{-1}\pmb{X}^T \\
& = \pmb{X}(\pmb{X}^T\pmb{X})^{-1}\pmb{X}^T\\
& = \pmb{P}\\
\end{align}
$$
</details>


### Projeção Ortogonal

Defina uma matriz \\(\pmb{M}\\) tal que 

$$\pmb{M}=\pmb{I}-\pmb{P}$$

\\(\pmb{M}\\) é uma matriz de projeção ortogonal ou **matriz aniquiladora**, uma vez que zera \\(\pmb{X}\\).

$$\pmb{M}\pmb{X}=\pmb{0}$$


\\(\pmb{M}\\) tem propriedades similares a \\(\pmb{P}\\), isto é, idempotência (\\(\pmb{M}\pmb{M}=\pmb{M}\\)) e simetria (\\(\pmb{M}^T=\pmb{M}\\)). Enquanto que \\(\pmb{P}\\) cria os valores ajustados \\(\pmb{\hat{y}}\\), \\(\pmb{M}\\) cria os resíduos \\(\pmb{\hat{e}}\\).

$$\pmb{M}\pmb{y}=\pmb{\hat{e}}$$

<details><summary>Provas</summary>
$$
\begin{align}
\pmb{M}\pmb{y} & = (\pmb{I}-\pmb{P})\pmb{y} \\
& = \pmb{I}\pmb{y}-\pmb{P}\pmb{y} \\
& = \pmb{y}-\pmb{\hat{y}} \\
& = \pmb{\hat{e}} \\
\end{align}
$$

$$
\begin{align}
\pmb{M}\pmb{X} & = (\pmb{I}-\pmb{P})\pmb{X} \\
& = \pmb{I}\pmb{X}-\pmb{P}\pmb{X} \\
& = \pmb{X}-\pmb{X} \\
& = \pmb{0} \\
\end{align}
$$

$$
\begin{align}
\pmb{M}^T & = (\pmb{I} - \pmb{P})^T \\
& = \pmb{I}^T - \pmb{P}^T \\
& = \pmb{I} - \pmb{P} \\
& = \pmb{M} \\
\end{align}
$$

$$
\begin{align}
\pmb{M}\pmb{M} & = (\pmb{I}-\pmb{P})(\pmb{I}-\pmb{P}) \\
& = \pmb{I}\pmb{I} - \pmb{P}\pmb{I} - \pmb{I}\pmb{P} + \pmb{P}\pmb{P} \\
& = \pmb{I} - 2\pmb{P} + \pmb{P} \\
& = \pmb{I} - \pmb{P}\\
& = \pmb{M}\\
\end{align}
$$
</details>

### Regressão por Componentes

Tendo definido \\(\pmb{P}\\) e \\(\pmb{M}\\) podemos partir para entender como funciona o modelo de regressão linear por partes. Podemos particionar as variáveis de um modelo de regressão linear da seguinte forma

$$\pmb{X} = \big[\pmb{X}_1 \ \pmb{X}_2\big]$$

$$\pmb{\hat{\beta}} = \binom{\pmb{\hat{\beta}}_1}{\pmb{\hat{\beta}}_2}$$

$$\pmb{\hat{y}} = \pmb{X}_1 \pmb{\hat{\beta}}_1 + \pmb{X}_2 \pmb{\hat{\beta}}_2 + \pmb{\hat{e}}$$


Estamos interessados nas expressões algébricas que definem \\(\pmb{\hat{\beta}}_1\\) e \\(\pmb{\hat{\beta}}_2\\). Para isso, defina as seguintes matrizes:


$$
\hat{\pmb{Q}_{xx}}=
\pmb{X}^T\pmb{X}=
\begin{bmatrix}
    \frac{1}{n} \pmb{X}_1^T\pmb{X}_1 & \frac{1}{n} \pmb{X}_1^T\pmb{X}_2 \\
    \frac{1}{n} \pmb{X}_2^T\pmb{X}_1 & \frac{1}{n} \pmb{X}_2^T\pmb{X}_2
\end{bmatrix}
$$

$$
\hat{\pmb{Q}_{xy}}=
\pmb{X}^T\pmb{y}=
\begin{bmatrix}
    \frac{1}{n} \pmb{X}_1^T\pmb{y}\\
    \frac{1}{n} \pmb{X}_2^T\pmb{y}
\end{bmatrix}
$$

Pela [fórmula de inversão de matriz por blocos](https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion), temos que

$$
\hat{\pmb{Q}_{xx}}^{-1}=
\begin{bmatrix}
    \hat{\pmb{Q}_{11}}^{-1} & - \hat{\pmb{Q}_{11}}^{-1} \pmb{X}_1^T\pmb{X}_2 (\pmb{X}_2^T\pmb{X}_2)^{-1} \\
    -\hat{\pmb{Q}_{22}}^{-1} \pmb{X}_2^T\pmb{X}_1 (\pmb{X}_1^T\pmb{X}_1)^{-1} & \hat{\pmb{Q}_{22}}^{-1}
\end{bmatrix}
$$

Em que 

$$
\begin{align}
\pmb{Q}_{11} & = \frac{1}{n}\pmb{X}_1^T\pmb{X}_1 - \frac{1}{n} \pmb{X}_1^T\pmb{X}_2 (\pmb{X}_2^T\pmb{X}_2)^{-1}\pmb{X}_2^T\pmb{X}_1 \\
& = \frac{1}{n}\pmb{X}_1^T \pmb{M}_2 \pmb{X}_1\\
\end{align}
$$

e

$$
\begin{align}\pmb{Q}_{22} & =\frac{1}{n}\pmb{X}_2^T\pmb{X}_2 - \frac{1}{n} \pmb{X}_2^T\pmb{X}_1 (\pmb{X}_1^T\pmb{X}_1)^{-1}\pmb{X}_1^T\pmb{X}_2 \\
& = \frac{1}{n}\pmb{X}_2^T \pmb{M}_1 \pmb{X}_2\\
\end{align}
$$

<details><summary>Provas</summary>

$$
\begin{align}
\pmb{Q}_{11} & = \frac{1}{n}\pmb{X}_1^T\pmb{X}_1 - \frac{1}{n} \pmb{X}_1^T\pmb{X}_2 (\pmb{X}_2^T\pmb{X}_2)^{-1}\pmb{X}_2^T\pmb{X}_1 \\
& = \frac{1}{n}\pmb{X}_1^T(\pmb{X}_1 - \pmb{X}_2 (\pmb{X}_2^T\pmb{X}_2)^{-1}\pmb{X}_2^T\pmb{X}_1)\\
& = \frac{1}{n}\pmb{X}_1^T(\pmb{I} - \pmb{X}_2 (\pmb{X}_2^T\pmb{X}_2)^{-1}\pmb{X}_2^T)\pmb{X}_1\\
& = \frac{1}{n}\pmb{X}_1^T \pmb{M}_2 \pmb{X}_1\\
\end{align}
$$
</details>

Com isso, temos que

$$
\binom{\pmb{\hat{\beta}}_1}{\pmb{\hat{\beta}}_2} = 
\begin{bmatrix}
    (\pmb{X}_1^T \pmb{M}_2 \pmb{X}_1)^{-1} \pmb{X}_1^T \pmb{M}_2 \pmb{y}\\
    (\pmb{X}_2^T \pmb{M}_1 \pmb{X}_2)^{-1} \pmb{X}_2^T \pmb{M}_1 \pmb{y} \\
\end{bmatrix}\\
$$

<details><summary>Provas</summary>

$$
\begin{align}
\binom{\pmb{\hat{\beta}}_1}{\pmb{\hat{\beta}}_2} = &
\begin{bmatrix}
    \hat{\pmb{Q}_{11}}^{-1} & - \hat{\pmb{Q}_{11}}^{-1} \pmb{X}_1^T\pmb{X}_2 (\pmb{X}_2^T\pmb{X}_2)^{-1} \\
    -\hat{\pmb{Q}_{22}}^{-1} \pmb{X}_2^T\pmb{X}_1 (\pmb{X}_1^T\pmb{X}_1)^{-1} & \hat{\pmb{Q}_{22}}^{-1}
\end{bmatrix}
\begin{bmatrix}
    \frac{1}{n} \pmb{X}_1^T\pmb{y}\\
    \frac{1}{n} \pmb{X}_2^T\pmb{y}\\
\end{bmatrix} \\
\\
= & 
\begin{bmatrix}
    \frac{1}{n} (\hat{\pmb{Q}_{11}}^{-1} \pmb{X}_1^T\pmb{y} - \hat{\pmb{Q}_{11}}^{-1} \pmb{X}_1^T\pmb{X}_2 (\pmb{X}_2^T\pmb{X}_2)^{-1} \pmb{X}_2^T\pmb{y})\\
    \frac{1}{n} (-\hat{\pmb{Q}_{22}}^{-1} \pmb{X}_2^T\pmb{X}_1 (\pmb{X}_1^T\pmb{X}_1)^{-1} \pmb{X}_1^T\pmb{y} +  \hat{\pmb{Q}_{22}}^{-1} \pmb{X}_2^T\pmb{y}) \\
\end{bmatrix}\\
\\
= & 
\begin{bmatrix}
    \frac{1}{n} \hat{\pmb{Q}_{11}}^{-1} (\pmb{X}_1^T\pmb{y} - \pmb{X}_1^T\pmb{X}_2 (\pmb{X}_2^T\pmb{X}_2)^{-1} \pmb{X}_2^T\pmb{y})\\
    \frac{1}{n} \hat{\pmb{Q}_{22}}^{-1} (\pmb{X}_2^T\pmb{y} - \pmb{X}_2^T\pmb{X}_1 (\pmb{X}_1^T\pmb{X}_1)^{-1} \pmb{X}_1^T\pmb{y}) \\
\end{bmatrix}\\
\\
= & 
\begin{bmatrix}
    \frac{1}{n} \hat{\pmb{Q}_{11}}^{-1} \pmb{X}_1^T (\pmb{I} - \pmb{X}_2 (\pmb{X}_2^T\pmb{X}_2)^{-1} \pmb{X}_2^T)\pmb{y}\\
    \frac{1}{n} \hat{\pmb{Q}_{22}}^{-1} \pmb{X}_2^T (\pmb{I} - \pmb{X}_1 (\pmb{X}_1^T\pmb{X}_1)^{-1} \pmb{X}_1^T)\pmb{y} \\
\end{bmatrix}\\
\\
= & 
\begin{bmatrix}
    \frac{1}{n} \hat{\pmb{Q}_{11}}^{-1} \pmb{X}_1^T \pmb{M}_2 \pmb{y}\\
    \frac{1}{n} \hat{\pmb{Q}_{22}}^{-1} \pmb{X}_2^T \pmb{M}_1 \pmb{y} \\
\end{bmatrix}\\
\\
= & 
\begin{bmatrix}
    (\pmb{X}_1^T \pmb{M}_2 \pmb{X}_1)^{-1} \pmb{X}_1^T \pmb{M}_2 \pmb{y}\\
    (\pmb{X}_2^T \pmb{M}_1 \pmb{X}_2)^{-1} \pmb{X}_2^T \pmb{M}_1 \pmb{y} \\
\end{bmatrix}\\
\end{align}
$$
</details>

<a name="fwl"></a>
## Parcialização e o Teorema Frisch-Waugh-Lovell

OK. A partir desses fundamentos estamos prontos para entender porque o modelo de regressão linear possibilita uma interpretação *ceteris paribus*. Para começar, note como a fórmula acima, tanto para \\(\pmb{\hat{\beta}}_1\\) quanto para \\(\pmb{\hat{\beta}}_2\\), tem um formato de um estimador MQO, isto é, \\(\pmb{\hat{\beta}}=(\pmb{X}^T\pmb{X})^{-1}\pmb{X}^T\pmb{y}\\). Se isso não está claro de imediato, considere as seguintes definições:

Seja 

$$\tilde{\pmb{X}}_2=\pmb{M}_1 \pmb{X}_2$$

e

$$\tilde{\pmb{e}}_1=\pmb{M}_1 \pmb{y}$$

temos que

$$ \pmb{\hat{\beta}}_2 = (\tilde{\pmb{X}}_2^T \tilde{\pmb{X}}_2)^{-1} (\tilde{\pmb{X}}_2^T \tilde{\pmb{e}}_1)$$


Para isso, usamos o fato de que \\(\pmb{M}=\pmb{M}\pmb{M}\\)

<details><summary>Provas</summary>

$$
\begin{align}
\pmb{\hat{\beta}}_2 & = (\pmb{X}_2^T\pmb{M}_1\pmb{X}_2)^{-1} \pmb{X}_2^T \pmb{M}_1 \pmb{y} \\
& = (\pmb{X}_2^T\pmb{M}_1 \pmb{M}_1 \pmb{X}_2)^{-1} \pmb{X}_2^T \pmb{M}_1 \pmb{M}_1 \pmb{y} \\
& = (\pmb{X}_2^T\pmb{M}_1^T \pmb{M}_1 \pmb{X}_2)^{-1}  \pmb{X}_2^T \pmb{M}_1^T \pmb{M}_1 \pmb{y} \\
& = ((\pmb{M}_1\pmb{X}_2)^T \pmb{M}_1 \pmb{X}_2)^{-1}  (\pmb{M}_1 \pmb{X}_2)^T \pmb{M}_1 \pmb{y} \\
& = (\tilde{\pmb{X}}_2^T \tilde{\pmb{X}}_2)^{-1} (\tilde{\pmb{X}}_2^T \tilde{\pmb{e}}_1) \\
\end{align}
$$
</details>

De maneira análoga, temos

$$ \pmb{\hat{\beta}}_1 = (\tilde{\pmb{X}}_1^T \tilde{\pmb{X}}_1)^{-1} (\tilde{\pmb{X}}_1^T \tilde{\pmb{e}}_2)$$

Assim, o coeficiente estimado \\(\pmb{\hat{\beta}}_1\\) é algebricamente idêntico a regressão de \\(\tilde{\pmb{e}}_2\\) em \\(\tilde{\pmb{X}}_1\\). Agora, note como esses dois são simplesmente \\(\pmb{y}\\) e \\(\pmb{X}_1\\), respectivamente, multiplicados por \\(\pmb{M}_2\\). Dos fundamentos, sabemos que a pré-multiplicação por uma matriz de projeção ortogonal \\(\pmb{M}\\) cria resíduos. Assim, \\(\tilde{\pmb{e}}_2\\) é simplesmente o resíduo da regressão de \\(\pmb{y}\\) em \\(\pmb{X}_2\\) e as colunas de \\(\tilde{\pmb{X}}_1\\) são os resíduos da regressão das colunas de \\(\pmb{X}_1\\) em \\(\pmb{X}_2\\). Assim, \\(\pmb{\hat{\beta}}_1\\) pode ser entendido como a regressão de \\(\pmb{y}\\) em \\(\pmb{X}_1\\) **após os termos considerados (projetado) a informação em \\(\pmb{X}_2\\)**.

<figure class="figure center-block" style="width: 100%;">
  <img src="/img/tutorial/mindblown.gif" class="img-responsive center-block" alt="Mind Blown">
</figure>

Note que fizemos a demonstração genérica com matrizes, mas isso também vale pare um único coeficiente \\(\beta\\): o valor estimado dele é equivalente a regressão de \\(\pmb{y}\\) nele após termos já considerado todos os outros fatores. Em outras palavras, esse coeficiente representa o impacto de uma variável \\(x\\) em \\(y\\) após todos os outros fatores incluídos no modelo terem sido controlados, isto é, mantidos constantes.

<a name="implementacao"></a>
## Implementação

Se você prefere ler código do que linhas matemáticas, podemos aplicar o teorema acima com algumas poucas linhas de Python. Em primeiro lugar, vamos definir a fórmula que regride uma variável na outra. Em seguida, vamos separar os dados em X e y. Repetiremos o estudo de ver como educação impacta na renda, dada a idade e o sexo. Vamos particionar X em dois: uma parte terá só a variável educação e a outra terá as outras variáveis e a constante.

```python
def regress(y, X):
    return inv(X.T.dot(X)).dot(X.T).dot(y)

X = (dataset
     .loc[:, ["education", "age", "female"]]
     .assign(intercept=1))

y = (dataset
     .assign(log_earnings=lambda df: np.log(df["earnings"]))
     .loc[:, ["log_earnings"]]
     .values)

X1 = X.loc[:, ["intercept", "age", "female"]].values
X2 = X.loc[:, ["education"]].values
```

Agora, vamos seguir o teorema acima e rodar as duas regressões parciais. E como era de se esperar, o coeficiente de educação é o mesmo `0.108` que obtivemos antes. 

```python
beta1_tilde = regress(y, X1)
y1_hat_tilde = X1.dot(beta1_tilde)
e1_tilde = y - y1_hat_tilde

beta2_tilde = regress(X2, X1)
X2_hat_tilde = X1.dot(beta2_tilde)
X2e_tilde = X2 - X2_hat_tilde

beta2_hat = regress(e1_tilde, X2e_tilde)

beta2_hat # coef para education
```
```
array([[0.10815617]])
```

Apenas para confirmar, vamos rodar a regressão de y em todas as variáveis de X. E, de fato, podemos ver que o valor `0.10815617` se repete para o coeficiente da educação.

```python
regress(y, X.values)
```
```
array([[ 0.10815617],
       [ 0.00954044],
       [-0.26289372],
       [ 1.15011621]])
```

<a name="ref"></a>
## Referências
Este tutorial é baseado no capítulo 3, *The Algebra of Least Squares*, do livro [Econometrics](https://www.ssc.wisc.edu/~bhansen/econometrics/), de Bruce E. Hansen. Além disso, tirei uma coisa ou outra na da parte de intuição *ceteris paribus* do livro [Introductory Econometrics: A Modern Approach](https://www.amazon.com/Introductory-Econometrics-Modern-Approach-Economics/dp/1111531048/ref=sr_1_2?s=books&ie=UTF8&qid=1547503795&sr=1-2&refinements=p_lbr_one_browse-bin%3AJeffrey+M.+Wooldridge), de Jeffrey Wooldridge. Por fim, como sempre, todo o código desenvolvido aqui está no [meu GitHub](https://github.com/matheusfacure/Tutoriais-de-AM/blob/master/MQO%20Ceteris%20Paribus.ipynb). Nesse link, coloquei também código referente a parte de Fundamentos Matemáticos, que pode te ajudar a entender melhor a matemática que desenvolvemos.

