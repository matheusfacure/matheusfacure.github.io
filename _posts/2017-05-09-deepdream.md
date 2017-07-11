---
layout: post
comments: true
title: Deep Dream para Interpretar Redes Neurais
subtitle: "A real utilidade daquelas imagens loucas que você vê na internet."
date: 2017-05-09
tags: [Post]
author: "Matheus Facure"
header-img: "/img/portfolio/deepdream/dd_50_after_com_festa_da_lanterna.jpg"
modal-id: 7
thumbnail: /img/portfolio/deepdream/thumbnail.jpg
description: Sabe aquelas imagens psicodélicas que vira e meche aparecem na internet? Elas são na verdade uma técnica chamada Deep Dream, desenvolvida para entender os que as redes neurais aprendem.
---

## O algoritmo de Deep Dream

Quando os pesquisadores do Google lançaram o algoritmo de Deep Dream a internet foi a loucura com milhares de imagens e vídeos que pareciam alucinações. O que a maioria das pessoas não sabe é que essa técnica foi desenvolvida com um propósito além do de criar obras de arte psicodélicas. Originalmente, o Deep Dream foi pensado como uma forma para auxiliar pesquisadores a entender o que aprendiam as redes neurais que eles estavam treinando. 

Uma rede neural pode ser uma ferramenta poderosíssima de reconhecimento de imagem. Em 2014, elas já ultrapassavam a performance humana na tarefa de enxergar, conforme ficou demonstrado pela [derrota de Andrej Karpathy na competição de reconhecimento de imagem ILSVRC (ImageNet Large Scale Visual Recognition Challenge)](http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/). Mas com todo todo esse poder vem grandes <s>responsabilidades</s> complicações. Essas redes neurais conseguiam performance no estado da arte em reconhecimento de imagem, mas como elas faziam isso? O que elas estavam aprendendo? Como elas conseguem ver? Essas são algumas perguntas que ainda estão em aberto e que atraem uma grande parte da comunidade de inteligência artificial. 

Embora nós não possamos respondê-las de forma certeira, há certamente métodos que nos permitem dar pelo menor uma olhada de relance dentro das redes neurais. Talvez assim possamos entender melhor o que elas estão aprendendo. Um desses métodos é o algoritmo de Deep Dream. Para melhor entendê-lo, eu reimplementei o algoritmo em [TensorFlow](https://www.tensorflow.org/), uma biblioteca de computação numérica desenvolvida pelo Google e perfeita para lidar com redes neurais.

Em primeiro lugar, baixei uma rede neural já treinada pelo Google. Essa rede neural é chamada carinhosamente de [Inception](https://arxiv.org/abs/1512.00567) e é gigantesca, com mais de 50 camadas de neurônios. Ela foi treinada para classificar imagens em 1000 classes, dente as quais estão diversos tipos de objetos, peças de roupa, veículos e até várias raças de cachorro. Para fazer isso, nós podemos imaginar que a rede neural tenha que aprenda algo sobre a visão, por exemplo, sobre o que diferencia um bull terrier de um dachshund. Mais ainda, nós sabemos que tudo o que a rede neural tem acesso são os valores numéricos dos pixeis, representando níveis de vermelho, verde e azul. Perceber objetos e animais a partir desses dados tão brutos é extremamente difícil, então a rede neural precisa antes convertê-los em algo mais abstrato. Isso é similar ao que nós fazemos ao enxergar. O que nosso corpo percebe são espectros de luz refletidos pelos objetos. Nosso cérebro então toma conta do trabalho de representar esses sinais de baixa abstração em algo mais complexo, como um violão ou um sorvete.

Partindo dessa noção biológica, nós podemos pensar em cada camada da rede neural artificial como captando um nível de abstração da imagem. Por exemplo, as primeiras camadas provavelmente detectam padrões bem simples, como traços, diferenças de luz e contraste, etc. Esses padrões então vão para as camadas mais do meio, que provavelmente aprendem a juntar esses padrões simples em abstrações mais complexas, como o olho de uma pessoa, a roda de um carro ou a janela de uma casa. Por fim, as camadas mais ao fim juntam os padrões aprendidos pelas camadas do meio para representar abstrações ainda mais complexas, como uma pessoa, uma casa ou um cachorro em uma prancha de surf.

Uma vez que tenhamos essa rede neural já treinada, o algoritmo de Deep Dream é extremamente simples: nós primeiro passamos uma imagem pela rede neural; em seguida selecionamos alguma camada de neurônios dessa rede neural e perguntamos que padrão essa camada gosta de ver. Seja lá o que for, nós adicionamos esse padrão à imagem. Então, apresentamos essa nova imagem mais uma vês a rede neural e repetimos o processo por quantas interações quisermos (aqui eu usei apenas 15). O resultado será uma imagem com os padrões enxergados pela rede neural fortemente destacados. 

Se você não está interessado nos detalhes técnicos, pule este parágrafo. Talvez esteja se perguntando, como sabemos qual padrão a camada da rede neural gosta de ver? Para isso, basta calcular a **o gradiente da camada com respeito à imagem**. Isso nós dará o quanto que a atividade na camada muda de acordo com a imagem passada. Nós usamos isso para ver o que cada camada gosta de detectar. Graças a forma como o TensorFlow está implementado, nós sequer precisamos calcular essas derivadas na mão, uma vez que a biblioteca faz isso de [forma automática](https://en.wikipedia.org/wiki/Automatic_differentiation).

Mas chega de teoria e vamos aos resultados. Para analisar o que as camadas "enxergam", considere essa foto que a Elis tirou de nós em um elevador velho de Buenos Aires. 

<img class="img-responsive center-block thumbnail" src="/img/portfolio/deepdream/dd_matelis_norm.jpg" alt="mat_elis_bue" style="width:30%">

Se utilizarmos o Deep Dream nas camadas 1, 5, 18 e 44 temos os seguintes resultados.

<div class="row">

<div class="col-md-6">
<img class="img-responsive thumbnail" src="/img/portfolio/deepdream/dd_1_after_com_matelis2.jpg" alt="mat_elis_1" style="width:100"/>
<div class="caption">
<p>Camada 1</p>
</div>
</div>

<div class="col-md-6">
<img class="img-responsive thumbnail" src="/img/portfolio/deepdream/dd_5_after_com_matelis2.jpg" alt="mat_elis_2" style="width:100"/>
<div class="caption">
<p>Camada 5</p>
</div>
</div>
</div>

<div class="row">
<div class="col-md-6">
<img class="img-responsive thumbnail" src="/img/portfolio/deepdream/dd_18_after_com_matelis2.jpg" alt="mat_elis_18" style="width:100"/>
<div class="caption">
<p>Camada 18</p>
</div>
</div>

<div class="col-md-6">
<img class="img-responsive thumbnail" src="/img/portfolio/deepdream/dd_44_after_com_matelis2.jpg" alt="mat_elis_44" style="width:100"/>
<div class="caption">
<p>Camada 44</p>
</div>
</div>
</div>

Como era de se esperar, a camada 1 gosta de ver padrões simples, como traços. A camada 5, por sua vez, parece organizar esses traços em padrões quadriculados. Se agora saltamos para as camadas 18 e 44, padrões mais intrincados começam a surgir. Na camada 18, podemos ver a presença de curvas, assim como padrões dentro de padrões. Quando chegamos à camada 44, os padrões são tão abstratos que sequer conseguimos vê-los todos. Ainda assim, alguns deles se destacam, por exemplo, no canto direito inferior surgiu algo que lembra uma raposa; no meio, na cintura da Elis, surgiram padrões que lembram características de um pavão. Particularmente interessante é como o chão do elevador se transformou em algo que lembra muito um carro. Tudo isso indica que na camada 44 a rede neural já está abstraindo noções de objetos e animais inteiros!

Essa rede neural tem mais de 50 camadas e mostrar como o algoritmo de Deep Dream se comporta com apenas 4 delas me faria sentir mal por privar você de toda a diversão. Então vamos considerar todas as camadas. Para isso, vamos usar esta imagem da festa da lanterna na minha escola primária. 

<img class="img-responsive center-block thumbnail" src="/img/portfolio/deepdream/dd_festa_da_lanterna.jpg" alt="lanternas" style="width:60%">

E agora vamos aplicar o algoritmo de Deep Dream com cada uma das camadas nessa imagem. Eis o resultado (se você quiser ver os GIF maiores, é só clicar neles com o segundo botão e selecionar "Abrir em Nova Aba").

<div class="row">
<div class="col-md-6">
<img class="img-responsive thumbnail" src="/img/portfolio/deepdream/dd_lanterna_1.gif" alt="lenterna_gif1" style="width:100"/>
<div class="caption">
<p>Camadas 0 à 25</p>
</div>
</div>

<div class="col-md-6">
<img class="img-responsive thumbnail" src="/img/portfolio/deepdream/dd_laterna_2.gif" alt="lenterna_gif2" style="width:100"/>
<div class="caption">
<p>Camadas 26 à 55</p>
</div>
</div>
</div>

Ok. Isso talvez seja rápido demais. Abaixo, vamos focar nas imagens mais interessantes. A primeira imagem diz respeito a camada 31. Ao que parece, essa cama é ótima em perceber cachorros (???). Se pularmos para a camada 35, podemos ver que o algoritmo de Deep Dream produz no horizonte algo que lembra uma cidade, com telhados abobadados. Na camada 38, podemos ver vários padrões que lembram um carro e algumas pessoas mas as abstrações já parecem muito complexas para entendermos. Na camada 50 os padrões são realmente complicados. Se você olhar bem, pode notar partes da imagem se assemelhando a cabeças de pássaros, pessoas, casas... Novamente, para ver as imagens ampliadas, basta abri-las em uma nova aba.

<div class="row">
<div class="col-md-6">
<img class="img-responsive thumbnail" src="/img/portfolio/deepdream/dd_31_after_com_festa_da_lanterna.jpg" alt="lanterna_31" style="width:100"/>
<div class="caption">
<p>Camada 31</p>
</div>
</div>

<div class="col-md-6">
<img class="img-responsive thumbnail" src="/img/portfolio/deepdream/dd_35_after_com_festa_da_lanterna.jpg" alt="lanterna_35" style="width:100"/>
<div class="caption">
<p>Camada 35</p>
</div>
</div>
</div>

<div class="row">
<div class="col-md-6">
<img class="img-responsive thumbnail" src="/img/portfolio/deepdream/dd_38_after_com_festa_da_lanterna.jpg" alt="lanterna_38" style="width:100"/>
<div class="caption">
<p>Camada 38</p>
</div>
</div>

<div class="col-md-6">
<img class="img-responsive thumbnail" src="/img/portfolio/deepdream/dd_50_after_com_festa_da_lanterna.jpg" alt="lanterna_50" style="width:100"/>
<div class="caption">
<p>Camada 50</p>
</div>
</div>
</div>

Apesar de termos conseguido várias imagens interessantes, não está claro como esses padrões complexos ajudam no reconhecimento de imagem realizado pela rede neural. Devemos então lembrar que estamos considerando o algoritmo de Deep Dream como uma análise do que **cada camada** da rede neural percebe. No entanto, uma camada tem **vários detectores de características** e pode ser simplista tratá-las como uma unidade uniforme. Talvez seja necessário olhar nos detectores dentro de cada camada para ver os padrões que a rede neural está aprendendo. Para essa tarefa, achei que seria interessante utilizar um (pedaço de um) quadro do Bosch. Para cada uma das camadas, nós selecionamos 3 captadores e aplicamos o Deep Dream tendo eles em consideração.

<img class="img-responsive center-block thumbnail" src="/img/portfolio/deepdream/dd_bosch.gif" alt="bosch" style="width:80%">

Bem, eu ainda não consigo ver exatamente como esses padrões auxiliam no reconhecimento de imagem, mas ficou claro como uma camada pode ter detectores que captam padrões bastante heterogêneos. Considere a camada 10, por exemplo. Nas imagens abaixo, fica claro como essa uma camada tem detectores que percebem coisas bem distintas. Além disso, não há uma hierarquia de abstração aprendida entre os detectores de uma mesma camada.

<div class="row">
<div class="col-md-6">
<img class="img-responsive thumbnail" src="/img/portfolio/deepdream/dd_bosch.jpg" alt="bosch_original" style="width:100"/>
<div class="caption">
<p>Quadro Original</p>
</div>
</div>

<div class="col-md-6">
<img class="img-responsive thumbnail" src="/img/portfolio/deepdream/dd_10_35_after_com_boshc.jpg" alt="bosch_dd_1" style="width:100"/>
<div class="caption">
<p>Camada 10, detector 35</p>
</div>
</div>
</div>

<div class="row">
<div class="col-md-6">
<img class="img-responsive thumbnail" src="/img/portfolio/deepdream/dd_10_44_after_com_boshc.jpg" alt="bosch_dd_2" style="width:100"/>
<div class="caption">
<p>Camada 10, detector 44</p>
</div>
</div>

<div class="col-md-6">
<img class="img-responsive thumbnail" src="/img/portfolio/deepdream/dd_10_57_after_com_boshc.jpg" alt="bosch_dd_2" style="width:100"/>
<div class="caption">
<p>Camada 10, detector 57</p>
</div>
</div>
</div>

Com tudo isso, a conclusão que cheguei é que o algoritmo de Deep Dream pode nos dar alguma luz sobre o que as redes neurais aprendem, mas nada muito certo. Além disso, ele tem uma clara limitação que é funcionar apenas com redes neurais para reconhecimento de imagem. Por outro lado, ele é muito divertido e consegue criar imagens realmente impressionantes. Para terminar, vamos aplicar o algoritmo de Deep Dream iterativamente na mesma imagem dando zoom a cada iteração para produzir uma espécie de viagem psicodélica.

<img class="img-responsive center-block thumbnail" src="/img/portfolio/deepdream/trip.gif" alt="trip" style="width:80%">

## Referências

Existem muitos tutorias na internet mostrando como implementar o algoritmo de Deep Dream, mas a maioria deles está muito mal documentada. Os próprios desenvolvedores do Google disponibilizaram um [tutorial usando o TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb), mas boa sorte tentado entendê-lo. Felizmente, eu achei um [tutorial muito bem explicado e documentado](https://www.youtube.com/watch?v=ws-ZbiFV1Ms) no YouTube. Partindo desse tutorial, eu fiz apenas algumas modificações para construir o código desse projeto. Eu também coloquei o código usado nesse projeto em um [repositório no GitHub](https://github.com/matheusfacure/DeepArt). Acredito que ele está bem documentado e espero que não gere dúvidas. 

Para entender melhor a motivação teórica por traz do Deep Dream eu sugiro [este post](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) do Google Research Blog. Você também pode conferir a [galeria de Deep Dream do Google](https://photos.google.com/share/AF1QipPX0SCl7OzWilt9LnuQliattX4OUCj_8EP65_cTVnBmS1jnYgsGQAieQUc1VQWdgQ?key=aVBxWjhwSzg2RjJWLWRuVFBBZEN1d205bUdEMnhB).

