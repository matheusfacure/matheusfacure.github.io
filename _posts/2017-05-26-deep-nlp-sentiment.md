---
layout: post
comments: true
title: Análise de Sentimentos do Jeito Certo
subtitle: "Porque é uma boa ideia usar Deep PNL para análise de sentimentos"
date: 2017-05-26
tags: [Post]
author: "Matheus Facure"
modal-id: 9
thumbnail: /img/portfolio/deep-nlp/thumbnail.jpg
description: Análise de sentimento é um termo geral para se referir à tarefa de perceber emoções a partir de dados gerados por pessoas. Dentre esses dados, os mais comuns estão na forma de texto. A partir de comentários nas mídias sociais, podemos ver como as pessoas reagem a uma postagem ou notícia, como estão avaliando um restaurante ou se aprovam ou não um político. Neste post, faço um breve resumo sobre as formas de realizar análise de sentimentos. Em seguida, mostro como Deep Learning (particularmente, redes neurais recorrentes) pode ajudar nessa tarefa.
---

## Conteúdo
1. [Deep PLN para Análise de Sentimentos](#deep-pnl) 
2. [O Modelo](#modelo)
3. [O experimento](#experimento)
4. [Referências](#ref)

## Deep PLN para Análise de Sentimentos <a name="deep-pnl"></a>

Análise de sentimento é um termo geral para se referir a tarefa de perceber emoções a partir de dados gerados por pessoas. Dentre esses dados, o mais comum para analise de sentimento é texto. A partir de comentários  nas mídias sociais, podemos ver como as pessoas reagem a uma postagem ou notícia, como estão avaliando um restaurante ou como avaliam um político. A forma mais precisa, porém mais ingênua, de fazer isso é pagar algumas pessoas para monitorar os comentários das mídias sociais. No entanto, milhares de comentários e avaliações são geradas por dia e usar pessoas para monitorá-los seria ou muito caro ou simplesmente impraticável. Uma alternativa mais razoável é usar processamento de linguagem natural (PNL) para automatizar o monitoramento dos sentimentos nos textos. Mas, mesmo em PNL, há várias formas de abordar esse tipo de tarefa.

A forma mais simples consiste em primeiro construir um vocabulário de palavras que estão associadas a sentimentos positivos e negativos, tais como "excelente" e "péssimo", respectivamente. Geralmente, é preciso pagar algum bom linguista para essa primeira etapa. Em seguida, ao analisar um texto, nós simplesmente contamos o número de palavras negativas e positivas para conseguir uma pontuação. Se essa pontuação ultrapassar um limiar preestabelecido, nós então classificamos o texto como tendo um sentimento positivo. Essa metodologia funciona bem, mesmo sendo extremamente simples. No entanto, ela é extremamente dependente do domínio da linguagem analisada, uma vez que palavras que expressam sentimento positivo em um contexto podem corresponder a sentimentos negativos em outros. Por exemplo, a palavra “denso” pode indicar uma avaliação positiva de um livro, mas também pode indicar uma reação negativa a uma massa italiana.

Uma segunda forma de realizar analise de sentimento é usar aprendizado de máquina. Em vez de pagar um linguista para determinar quais são as palavras negativas e positivas, nós coletamos milhares de exemplos de texto positivo e texto negativo e treinamos uma máquina para aprender por conta própria as palavras positivas e negativas. Achar milhares de texto com sentimento negativo e positivo é fácil. Basta baixar [bases de avaliações de produtos da Amazon](http://jmcauley.ucsd.edu/data/amazon/), e supor que avaliações com mais de 3 estrelas são positivas e com menos de 3 estrelas, negativas (as com 3 estrelas nós descartamos como sendo neutras). Em seguida, para cada palavra no nosso vocabulário, nós contamos quantas vezes ela aparece nos textos com sentimentos positivos e negativos; então dividimos o número de ocorrências nos textos positivos pelo número de ocorrências nos textos negativos para conseguir uma pontuação para cada palavra. Essa pontuação será proporcional a "positividade" da palavra. Com isso, é fácil conseguir uma pontuação para o texto: basta tirar a média das pontuações de cada palavra nele. Essa abordagem é extremamente eficiente, fácil de implementar e funciona super bem. Ela resolve a primeira etapa do processo de PNL, isto é, não é precismo mais pagar um linguista para construir léxicos negativos e positivos. No entanto, cabe ressaltar alguns problemas que esse método não resolve.

Em primeiro lugar, ainda estamos desconsiderando informação do contexto, que pode dar um sentimento diferente para a mesma palavra. Em segundo lugar, ao simplesmente somar a pontuação das palavras para conseguir uma pontuação para o texto, estamos ignorando a estrutura sequencial do texto. Por exemplo, uma avaliação como "Eu estava esperando algo realmente desastroso, mas no final acabei me surpreendendo" tem duas orações, sendo que a primeira é negativa, mas a segunda é positiva. No português, o sentimento presente na oração coordenada adversativa predomina, então a frase como um todo é positiva. Isso é extremamente difícil de notar com um algoritmo de aprendizado que ignora a estrutura sequencial do texto. Uma outra dificuldade desse tipo de método é lidar com expressões idiomáticas, como "amigo da onça". Um algoritmo que analisasse cada palavra individualmente provavelmente atribuiria um sentimento positivo a essa expressão (por conta da palavra amigo), o que estaria equivocado.

Para lidar com essas dificuldades, podemos usar redes neurais recorrentes. Elas funcionam de forma iterativa, lendo a o texto em sequência, palavra por palavra. No início da frase, a rede neural codifica a primeira palavra em uma representação interna; na próxima iteração, a rede neural observa tanto a segunda palavra quanto a representação interna da iteração anterior e assim produz uma nova representação interna. Esse processo é repetido até o final da sequência de palavras. Nossa esperança é que, palavra por palavra, a rede vá incorporando na sua representação interna a informação presente no texto. Então, por fim, a partir da representação interna na última iteração, a rede neural produz uma probabilidade da frase ter um sentimento positivo. Isso resolve os dois problemas citados acima. Primeiro, as redes neurais recorrentes são feitas para representar sequências, então o texto é facilmente interpretado dessa forma. Em segundo lugar, a capacidade de representar sequências dá a rede neural poder para entender expressões idiomáticas; ela pode simplesmente codificar em sua representação interna a sequência de palavras da expressão de forma distinta de como codificaria cada palavra separadamente. Muito bem, isso soa promissor. Vamos tentar!

## O Modelo <a name="modelo"></a>
Em primeiro lugar, baixei as [avaliações da Amazon](http://jmcauley.ucsd.edu/data/amazon/). Lá, cada comentário/avaliação era associado a uma classificação de 1 a 5 estrelas. Eu supus que todas as avaliações com mais de 3 estrelas eram positivas e todas com menos de 3, negativas. Depois, usando 2 milhões de avaliações (1m de positivas e 1m de negativas), treinei o seguinte modelo de rede neural recorrente.

<table class="table table-striped table-bordered table-hover">
  <thead>
    <tr>
      <th>Camada</th>
      <th>Formato</th>
      <th>Especificação</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Entrada</td>
      <td>Sequência, 1 variável</td>
      <td>Indicies das palavras</td>
    </tr>
    <tr>
      <td>Representação</td>
      <td>Sequência, 64 variáveis</td>
      <td>Representação das palavras</td>
    </tr>
    <tr>
      <td>Recorrente</td>
      <td>Sequência, 32 variáveis</td>
      <td>LSTM bi-direcional</td>
    </tr>
    <tr>
      <td>Dropout</td>
      <td>Sequência, 32 variáveis</td>
      <td>Destruição de 70%</td>
    </tr>
    <tr>
      <td>Recorrente</td>
      <td>Sequência, 32 variáveis</td>
      <td>LSTM bi-direcional</td>
    </tr>
    <tr>
      <td>Dropout</td>
      <td>Sequência, 32 variáveis</td>
      <td>Destruição de 70%</td>
    </tr>
    <tr>
      <td>Saída</td>
      <td>Escalar, 1 variável</td>
      <td>Função logística</td>
    </tr>
  </tbody>
</table>

Nessa rede neural, as duas primeiras camadas tomam conta de codificar as palavras do texto em uma representação matemática. Para mais informações sobre esse processo, sugiro um [post que fiz](https://matheusfacure.github.io/2017/03/20/word2vec/) há algum tempo. Em seguida, as duas camadas recorrentes leem o texto palavra por palavra (com as palavras já codificadas), construindo uma representação interna da sequência observada. Você pode pensar na rede neural codificando a sequência em uma representação interna abstrata. Por fim, nós pegamos essa codificação abstrata e passamos à uma camada de saída, cujo trabalho é decodificar a representação interna da rede neural em uma probabilidade - a probabilidade do texto conter um sentimento positivo. Em termos técnicos, esse modelo é o que chamamos de rede neural recorrende profunda bi-direcional. Eu pessoalmente acho esse modelo bem complexo e só posso dar uma explicação bem resumida e intuitiva sobre ele. No final do post, vou colocar referências para mais informações sobre ele.

<a href="http://colah.github.io/posts/2015-09-NN-Types-FP/">
<img class="img-responsive center-block thumbnail" src="/img/portfolio/deep-nlp/bi-rnn.png" alt="bi-rnn" />
</a>

## O experimento <a name="experimento"></a>

Muito bonito o modelo acima, mas funciona? Para ver isso, eu reservei 5% das 2mi avaliações e treinei essa rede neural nos outros 95% dos dados. Depois de pouco mais de uma hora, o modelo conseguiu prever corretamente mais de 92% do sentimento das avaliações não utilizadas para o treino. Podemos dizer que esse resultado é satisfatório. Não é nada fenomenal, mas já algo que pode ser usado industrialmente. Um detalhe importante, é que testamos o modelo treinado com avaliações do site da Amazon e essas avaliações**já tem uma pontuação de sentimento atrelado a elas**. Em outras palavras, não tem sentido prático prever avaliações da Amazon, porque já podemos saber o sentimento delas. 

Para avaliar o modelo com exemplos mais reais, eu fui ao Facebook, em páginas como a do Prêmio Nobel, The Economist, The New York Times e TED Talks, e peguei alguns exemplos de comentários. Então, utilizei o modelo treinado para conseguir a probabilidade de sentimento positivo de cada um deles.

Dos onze comentários com sentimento positivo que coletei, dois foram classificados de forma errada. O primeiro deles é o seguinte: 
```
The far reaching extent of people in the world are inherently good. This man could have ran off, and done nothing. Instead, he stepped up. We need more people to step up in all phases of life. This isn't an issue we can kill our way out of.
```
A rede neural deu uma probabilidade de 0.35 desse comentário ser positivo. O outro comentário classificado de forma errada recebeu uma probabilidade de 0.18 de ser positivo:

```
Just because I'm homeless doesn't mean I haven't got a heart. You, sir, have more heart than many very extremely wealthy men and women whose jobs are to care for their fellow citizens... I'm from the USA so yes, I am referring to our cruel leadership in DC. I hope they read your story and take inspiration from your actions. Bless and thank you
```
Esse comentário diz respeito a atuação heroica de um sem teto no atentado de Manchester (05/2017). Podemos ver que ele carrega um sentimento misto de louvor à ação do homem, mas também expressa uma indignidade com como as lideranças dos Estados Unidos. Eu mesmo tive minhas dúvidas antes de julgar esse comentário como tendo um sentimento positivo; não é surpresa então que a rede neural também tenha se confundido.

Alguns outros exemplos de comentários positivos, corretamente classificados (com mais de 95% de certeza), são os seguintes:

```
Congratulations Taiwan! This is a great day for human rights and equality before the law. As per usual, the homophobic religious nutjobs spewing their hateful bile on here are ALL closeted gays themselves who are trying to divert attention away from their own deeply repressed same sex attractions. It is classic Freudian psychology!
```

```
When I told everyone I wanted to be a stand up comic they all laughed. Ten years later I finally made it and nobody's laughing now
```

```
Congratulations Taiwan! This is a great day for human rights and equality before the law. As per usual, the homophobic religious nutjobs spewing their hateful bile on here are ALL closeted gays themselves who are trying to divert attention away from their own deeply repressed same sex attractions. It is classic Freudian psychology!
```

```
Although I am an Atheist and although the Catholic Church has been implicit in horrible crimes against humanity and specifically the vulnerable, I believe that Pope Francis is good, honest and empathetic man and an excellent religious leader for these times.
```

Esse último exemplo tem uma particularidade. Primeiro, a pessoa expressa um sentimento negativo e em seguida, contrapõe o que tinha dito com um sentimento positivo, que predomina no comentário. Esse caso é análogo ao uso de orações coordenadas adversativas que discutimos acima e é particularmente difícil de ser classificado corretamente por modelos que desconsideram a ordem das palavras.

Quanto aos exemplos de comentários negativos, dois dos doze que coletei foram incorretamente classificados.

```
I can't believe how this coward would choose a concert venue for a pop singer whose fan base demographic are teenage girls. Make no mistake about it, children were the deliberate targets for this latest horrific, terrorist attack. Parents unsuspectingly took their innocent teenage girls to an Ariana Grande concert for a seemingly fun night of entertainment and were victimized by a deadly suicide bombing. Even those who were not killed or injured will be mentally traumatized for life. Thank God the terrorist never made it inside of the venue and the bomb detonated outside the main entrance in a public space, or else the fatalities would have been much worse. Acts of violence against innocent children in the name of a religion or ideology? The dehumanization continues. God help us. Praying for the victims and their families at this time!
```

```
My deepest condolences to the families. As a Muslim I feel ashamed that my religion is being used to commit crimes like these. I wish all those Muslims who have issues with Western values would just leave and pick a country who shares their values and has no secularism. I wonder how long they would make it there.
```

Ambos os comentários foram classificados como positivos, com uma probabilidade maior do que 95%. Eu não sei porque isso aconteceu. Particularmente, não consigo ver nada que possa causar dúvidas quanto ao sentimento negativo. Outros exemplos de comentários negativos, esses classificados corretamente, são os seguintes:


```
The entire argument is flawed, because in your analogy, the owners of big companies are the Elizabeth I of our times. They are the only ones who gain net benefit from breakthroughs of industrial engineering. Everyone else breaks even or sees inflation erode their relative earnings.
```

```
have we learned nothing from the terminator? Must we all die so some psychos in white coats play god? It's bad enough we have d wave. Can we jus chill out with the self destruction for like 5 minutes?
```

```
Absolutely boring. I know GGM has his fanboys and lackeys out there but the fact that Joyce and Proust never got this award, it proves or shows that you Swedes cannot read well. Ugh, that first paragraph or even the first sentence of Cien años de soledad, ugh, how hideously composed! I wonder if Flaubert would have dug this Márquez dude. Why read García Márquez when one can enjoy Borges, Casares, Filloy, Di Benedetto or Mujica Lainez?

```

```
How can any army which occupies a people be considered remotely moral? If that was the case, the British Empire would have been the most moral and virtuous entity the world has ever known. It is an absolute travesty that this is even considered a debate, and shows how little man has actually progressed.
```

Bem, claramente existem vantagens em usar redes neurais recorrentes em vez de outro modelo, mas isso é sempre o caso? Obviamente não. Em primeiro lugar, redes neurais são extremamente complicadas e difíceis de treinar; em algumas situações, elas só oferecem benefícios marginais sobre modelos mais simples. Em segundo lugar, redes neurais recorrentes são ótimas para lidar com sequências e oferecem diversas vantagens de efe ciência computacional quando essas sequências são longas. No entanto, quando o texto consiste em apenas algumas palavras, como Tweets, por exemplo, usar uma rede neural recorrente seria um pouco exagerado. Novamente, é provável que modelos mais simples funcionem bem. Em suma, a mensagem final é: não complique além do necessário. Redes neurais são ótimas, mas muitas vezes modelos bem menos complicados simplesmente funcionam.


## Referências <a name="ref"></a>

O conteúdo desse post é uma aplicação do que estudei no curso de [*Deep Learning for Natural Language Processing* (2016-2017), da universidade de Oxford](http://www.cs.ox.ac.uk/teaching/courses/2016-2017/dl/), particularmente os conteúdos da aula 5, sobre classificação de texto.  
Se lhe interessa apenas saber mais sobre o modelo de redes neurais recorrentes, sugiro que leia [este post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). Outra opção é [este post](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/), que também aborda redes neurais recorrentes bi-direcionais.
Como de costume, o código do projeto está disponível no [meu GitHub](https://github.com/matheusfacure/nlp-oxford).


