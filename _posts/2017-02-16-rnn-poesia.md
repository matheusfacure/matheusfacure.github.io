---
layout: post
comments: true
title: Escrevendo Poesia com Redes Neurais
subtitle: "Pode uma máquina ser criativa? Mas é Claro!"
date: 2017-02-16
tags: [Post]
author: "Matheus Facure"
header-img: "/img/fundo_main.png"
modal-id: 2
thumbnail: /img/portfolio/rnn-poet/thumbnail.png
description: Inteligência artificial já chegou em um estágio em que consegue escrever pequenas frases poéticas. Embora ainda seja extremamente difícil gerar um contexto longo e coeso, neste posto mostro como podemos usar redes neurais recorrentes para simular o estilo de uma escrita e até escrever pequenos poemas.
---

<h2>Resumo</h2>
<p>Na minha opinião, redes neurais recorrentes (RNRs) são os modelos mais divertidos de todo Aprendizado de Máquina. Para não acabar com a diversão, nesse estudo eu não gastarei muito tempo com a teoria das RNRs e vou me concentrar mais no que elas são capazes de fazer. Especificamente, vou mostrar como é possível <strong>gerar texto poético, caractere por caractere, utilizando uma rede neural recorrente.</strong></p>

<h2>Conteúdo</h2>
<ul>
	<li><a href="#RNNs">Redes neurais recorrentes</a></li>
	<li><a href="#Modelando-texto">Modelando texto a nível de caracteres</a>
<ul>
	<li><a href="#Funcionamento">Funcionamento das RNRs</a></li>
</ul>
</li>
	<li><a href="#Experimentos">Experimentos</a>
<ul>
	<li><a href="#Shakespeare">Shakespeare</a></li>
	<li><a href="#Textos-pt">Textos em português</a></li>
	<li><a href="#Drummond">Drummond</a></li>
	<li><a href="#Camoes">Camões</a></li>
	<li><a href="#leis">Gerando leis</a></li>
</ul>
</li>
	<li><a href="#Trabalhos-relacionados">Trabalhos relacionados</a></li>
	<li><a href="#Conclusao">Conclusão</a></li>
	<li><a href="#Referencias">Referências</a></li>
</ul>
<h2 id="RNNs">Redes neurais recorrentes</h2>
<p>O que torna as RNRs tão poderosas é o fato delas operarem em sequências, podendo recebê-las como <em>inputs </em>é produzi-las como <em>outputs.</em> Há uma grande flexibilidade sobre como essas sequências podem ser representadas. Por exemplo, podemos observar uma sequência e prever um único resultado ou toda uma sequência de resultados:</p>

<figure class="figure center-block thumbnail">
  <img class="img-responsive center-block thumbnail" src="/img/portfolio/rnn-poet/diags.jpeg" alt="diags" width="1329" height="416" />
  <figcaption class="figure-caption text-center">Fonte: <em><a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/">The Unreasonable Effectiveness of Recurrent 
Neural Networks</a>.</em></figcaption>
</figure>


<p>Intuitivamente, essa capacidade de modelar sequências dá à RNR o poder de incorporar na sua representação interna toda a dinâmica de evolução da sequência. As redes neurais sem recorrência estão fadadas a perceber cada observação dos dados como independentes, não sendo capazes de notar como uma amostra influencia diretamente na outra. RNRs não tem tal limitação.</p>

<h2 id="Modelando-texto">Modelando texto a nível de caracteres</h2>
<p>O objetivo aqui é construir uma rede neural recorrente que consiga gerar texto poético. Para isso, vou criar uma RNR que observe uma sequência de caracteres e tente prever qual será o próximo caractere, dado a sequência observada. Uma vez treinada, poderemos utilizar essa RNR para gerar texto da seguinte forma. Primeiro, apresentamos a ela uma sequência de caracteres (isto é, um pouco de texto) e pediremos que ela preveja o próximo caractere da sequência; então tomamos essa previsão como verdadeira, incorporando o caractere previsto à sequência. Em seguida apresentamos a nova sequência - com o caractere previsto no final - e pediremos para a RNR prever o próximo caractere novamente. Repetindo esses passos várias vezes, poderemos gerar a quantidade de texto que quisermos.</p>
<p>Algumas coisas importantes que devemos notar: (1) para realizar a tarefa descrita acima, a rede neural não só terá que aprender toda a ortografia e sintaxe, mas também as relações semânticas entre as palavras. Tudo isso observando apenas caracteres individuais! (2) Como nossos recursos computacionais são um tanto limitados, nós teremos que restringir o tamanho da sequência observada para no máximo 100 caracteres. Isso significa que, para produzir texto, a rede neural só poderá observar os últimas 100 caracteres da sequência de letras; (3) A RNR gerará texto um caractere de cada vez; se ela conseguir gerar algo com pelo menos um pouco se sentido, isso já será verdadeiramente impressionante!</p>

<h3 id="Funcionamento">Funcionamento das RNRs</h3>
<p>Essa será uma pequena seção teórica sobre RNRs. Você pode pular essa seção sem problemas. Eu só vou colocá-la aqui para os mais curiosos com as formalidades técnicas.</p>
<p>A intuição do funcionamento de uma RNR é bastante simples: a cada passo na sequência a RNR observa um caractere do texto e atualiza seu estado interno com base no caractere observado e no estado interno do passo anterior; ao final de 100 passos, isto é, depois de ter observado 100 caracteres, a rede neural fará uma previsão do caractere seguinte da sequência.</p>
<p>Formalmente, dada uma sequência de \( n\) caracteres \( (c_0,c_1, ...,c_n)\), a RNR computa uma sequência de estados internos \( (h_0, h_1, ..., h_n)\) e de previsões \( (\hat{y}_0,\hat{y}_1, ...,\hat{y}_n)\) iterando as seguintes equações:</p>

$$ h_t = tanh(W_{hc}c_t + W_{hh} h_{t-1} + b_h) $$

$$ \hat{y}_t= W_{yh}h_t + b_y $$

<p>Nós descartamos todas as previsões \( \hat{y}_t\) que não a última, isto é, a produzida após observar toda a sequência. Nós otmizamos esse modelo atualizando os parâmetros \( W\) e \( b\) de forma iterativa, com gradiente descendente através do tempo (Rumelhart et al., 1986; Werbos, 1990).</p>
<p>Ao final do aprendizado, o que teremos aprendido será a distribuição multinomial do próximo caractere, dados os 100 caracteres anteriores. Para representar essa distribuição multinomial, vamos utilizar a função <a href="https://en.wikipedia.org/wiki/Softmax_function">softmax</a>. Além de várias propriedade interessantes para o aprendizado, essa função tem uma característica que nós ajudará a <strong>gerar texto com menos ou mais criatividade</strong>. Nós podemos escalar o <em>input </em>à função <em>softmax </em>dividindo-o por uma temperatura (vamos usar algo entre 0 e 1). Quanto maior a temperatura, mais próximo a distribuição uniforme será o nosso <em>output</em>, ou seja, a rede terá menos confiança ao prever o próximo caractere, sendo assim mais criativa.</p>

<h2 id="Experimentos">Experimentos</h2>
<h3 id="Shakespeare">Shakespeare</h3>
<p>O primeiro experimento que realizamos foi com Shakespeare. Nós baixamos a obra completa do autor e separamos o texto por caracteres. Esse processo gerou uma base de dados com impressionantes 4.573.338 caracteres, 67 caracteres distintos e nos restringimos a 1.524.430 sequências de texto por limitações computacionais. Nós então treinamos uma rede neural recorrente nessa base de dados. Após ler apenas uma vez a obra completa de Shakespeare, a rede neural foi capaz de gerar texto como o seguinte:</p>

*counsel: O, what learning is!  
My lord, I'll tell mane me down and make brith,  
for the dore of do as in shere neven in the sen  
Thou love the gromentle and brether the dartion.*  
  
*BERANDE:
Nother the ferther and for her the monter.*
  
*CLOUMI:  
Sear the more some so with and sight and forsing  
Of not the come the hers the would be my seep,*

<p>Ok... Isso não está tão bom, mas já podemos ter uma noção do que a rede aprendeu.  Em primeiro lugar, podemos ver que ela consegue gerar várias palavras e relacioná-las com alguma sintaxe. Por exemplo o frase "My lord, I'll tell mane" está correta gramatica e sintaticamente. Contudo, a palavra "mane" no final da frase não faz muito sentido. Em segundo lugar, podemos ver que a rede neural aprendeu a estruturar o texto em versos e falas de personagens. Lembre-se de que a rede produziu esse texto caractere por caractere, o que é realmente impressionante!</p>
<p>E abaixo está o texto gerado depois da RNR ler a obra completa de Shakespeare 18 vezes:</p>

*SIR ANDREW:  
I am not a soldier with my honour'd death.*  
  
*DUKE ORSINO:  
Thou wise a thing of them with the armourence,  
To endure the such a thing in Antony.*  
  
*DUKE VINCENTIO:  
Let him have a means that strike him at the bear.*  
  
*LAERTES:  
That say you all the same, sir, so long, and the heart  
I have in the strange lies of watery instruction,  
Or else the dear profit of words shall be  
No tale of the dear father of the saw of men,  
To do him hence, do not hang all for his chamber,  
And have of the life in his since man's mind.*  
  
*GLOUCESTER:  
Here were he was come to the enemies:  
The cardinal here is the perfect of them.*  
  
*PRINCE HENRY:  
I have heard me that are the part of the breath of  
his honourable.*  
  
*CLOTEN:  
What is the street?*  
  
*BENVOLIO:  
This is the speeder of my masters;  
And the companion and my wife hath not a love  
That will speak to thee as more in this good ducats.*  
  
*EDMUND:  
O, sir, I am a book hart in me as the master.*  

<p>Nós podemos ver que a rede aprendeu a estruturar o texto de maneira ainda mais convincente e comete menos erros gramaticais. No entanto, há uma dificuldade em considerar um contexto maior, isto é, a rede neural consegue produzir frases individuais com sentido, mas não consegue juntá-las em uma parágrafo com conteúdo semântico interessante.</p>

<h3 id="Textos-pt">Textos em português</h3>
<p>Modelar texto em inglês é bem mais fácil do que modelar texto em português. Em primeiro lugar, isso acontece porque há muito menos caracteres distintos na língua inglesa. Por exemplo, no nosso experimento de Shakespeare havia apenas 67 caracteres distintos. Além disso, a língua inglesa têm bem menos conjugações verbais. Assim, para facilitar o trabalho da rede neural (e poupar a memória do nosso computador) retiramos todos os acentos e cedilhas dos textos em português. Com isso, ficamos com mais ou menos 84 caracteres distintos (dependendo do texto utilizado).</p>

<h3 id="Drummond">Drummond</h3>
<p>Juntamos vários livros de poesia de Drummond e treinamos uma pequena rede neural recorrente com essas poesias. Após algumas horas de treino, a rede foi capaz de gerar texto como o seguinte:</p>

*15. CAMA  
A noite, se fez  
mais forte dos meninos.  
Pousando a paz do sono e somo e meu fogo  
que compor um sonho e fazer  
na barta seu corpo, esse escravo,  
escolhe e teu sapato de ser, a fala e e brilha  
e latar-me com o vidro de crianca  
pairando para o beijo  
e perdidos nao gritas.  
Toda historia do ar.  
Sob consolar-se a sombra  
de carne a mao, no rio.*  

<p>Eu achei esse texto extremamente interessante pois a rede gerou até o nome do poema e começou ele de forma coerente com título!</p>

<h3 id="Camoes">Camões</h3>
<p>Gerar texto a partir da leitura de Camões foi mais complicado. A rede aprendeu bem a estrutura do texto, incluindo a numeração dos cantos e a utilização de falas dos personagens. No entanto, as frases parecem ter pouco ou nenhum sentido, quase não se relacionando entre si.</p>

*LXXII.  
Qu'esperos passeia da negro estremos,  
 Bem versos que a vivia pretendia  
 Presencia que não vesta profundamente.  
 Aquelle inda com ser desaffonto,  
 Mas quando gozar me doa a pobreza,  
 Vi que não o esquece, porque se vós  
 Serás frescas corpestades imagina.*  

*CCXXXVI.
Na esperança duvidosa cuidando  
 Até por para vida de diamante.  
 Assi o tal! e pena); que se levantão  
 Esse anno meu passados eterno.  
 DELIO.  
 Que diz o soffrio, ágoa, emfim, vem, difanos,  
 Qu'eu não torne o mar n'alma verde Monto.*  

<p>Mesmo assim, se procurarmos bem no texto gerado, conseguimos achar um ou dois versos mais interessantes:</p>

   *Mas não ficou brando outro amor o sente,  
 Com a alma descuidado de seu sentido,  
 E o Pundo a vida consentida e minha,  
 E a morte de meus amores lembranças;*  
  
ou  
  
   *Em quanto a vida está de tudo esconde,  
 Que a minha morte a minha confiança,  
 E a parte de teu descontente e dia.*  

ou ainda

 *A qualquer o rio, o desejo accento.  
 Da vida a sombra e o seu tormento;  
 A dor de tua tão alto e a mais creio.  
 Aquelle no mundo a morte se enreda,  
 Por a esperança mais he de soltar assi;  
 Agora em teus versos vivia e a me deseja.*  

<h3 id="leis">Gerando Leis</h3>
<p>Mas nós não precisamos nos restringir à literatura. Nós podemos, por exemplo, gerar texto de leis se treinarmos a rede neural com vários <a href="http://www.planalto.gov.br/ccivil_03/">códigos brasileiros</a>:</p>
<p><em>Art. 29 (Revogado pelo Decreto-Lei nº 406, de 1968).</em>
<em>§ 4º Os candide o pamento ror na seção de produtos de registro de susposição de assembléia, demais estabe e pela instituii da competência do interessado ou prova de cempresse para os juros do cônjuge, a que se refere o termo do disposto no art. 1.027 a 1.104, o de ensino ou de interpinha e fica contra a prestação do domicílio do período de pelo tempo de decisão na data da lei.</em></p>
<p><em>Art. 269. As contrárias com as do presideito de tranomissão do prazo e os processos de favo da autoridade contra ele o trabalho seja instituirá ao limite da condição ou alter de serviço, as respondentes, poderá ser recurso anterior, pelo prédio superior de candidato e a legar, a responsabilidade de trabalho, ou para a conta a que se refere o art. 1.530, se o empregado se inferior de suterior a sua prestação, em relação de execução de maneira intedar a pena de competência de 10 (dez) dias a que se res mercados na utisição de seus seguintes em caracter os demais substitutos de tempo de sua prestação.</em></p>
<p><em>Art. 138. O casamento de dois dias, a conto sobre a presumir o contrato de exposição de cargo público ou a providar o processo de serviços de descendentes ou qualquer crames forem de cada um dos livros para a coisa compradade, quando a concessão de seus curriculares e as concernentes de direito de concorrer o exercício do disposto no art. 1.070 e erro e compromisso ou de trabá de trabalho contratado por meio a não se constitui a concessão de de direito à construção distituição do trabalho este não pode sesse o prazo de conterá em caso de participação e acomento de comprovação dos comodados.</em></p>
<p>Quase nada acima faz sentido, mas é interessante ver como a rede foi capaz de estruturar as leis em artigos e parágrafos, gerar datas, abrir e fechar parênteses, etc.</p>

<h2 id="Trabalhos-relacionados">Trabalhos relacionados</h2>
<p>Primeiramente, nós imaginamos que a incapacidade da nossa rede neural em estruturar contextos mais longo vinha das nossas limitações computacionais, que nos fez utilizar um modelo relativamente pequeno. Nós então exploramos outros trabalhos para ver se outras pessoas com mais recursos conseguiam resultados melhores.</p>
<p>Particularmente, olhamos os resultados obtidos por Sutskever et al, no artigo <em>Generating Text with Recurrent Neural Networks</em>, no qual os autores utilizam a maior rede neural treinada até então. O modelo deles, treinado com texto do <em>New York Times</em>, produziu o seguinte texto:</p>
<p><em>while he was giving attention to the second advantage of school building a 2-for-2 stool killed by the Cultures saddled with a halfsuit defending the Bharatiya Fernall ’s office . Ms. Claire Parters will also have a history temple for him to raise jobs until naked Prodiena to paint baseball partners , provided people to ride both of Manhattan in 1978 , but what was largely directed to China in 1946 , focusing on the trademark period is the sailboat yesterday and comments on whom they obtain overheard within the 120th anniversary , where many civil rights defined , officials said early that forms , ” said Bernard J. Marco Jr. of Pennsylvania , was monitoring New York.</em></p>
<p>Nós podemos ver que nesse caso também há uma dificuldade em formar contextos mais longos, assim como com nossos modelos.</p>

<h2 id="Conclusao">Conclusão</h2>
<p>Por muito tempo RNRs foram consideradas quase impossíveis de treinar, mas (muito) recentemente já é possível fazer isso mesmo em computadores não industriais. Embora o sucesso das RNRs em modelar texto seja ainda bastante duvidoso, eu sou bastante otimista e acho que os desenvolvimentos nessa área só irão crescer nos próximos anos. Mesmo assim, acho que há ainda um longo caminho até que consigamos gerar máquinas com inteligência suficiente para realmente entender e gerar texto humano. Por hora, sequer podemos dizer se as limitações atuais nesse campo só são devido às limitações computacionais ou a classe do algoritmo de inteligência artificial que estamos utilizando.</p>

<h2 id="Referencias">Referências</h2>
<ul>
	<li><em><a href="http://bigthink.com/natalie-shoemaker/a-japanese-ai-wrote-a-novel-almost-wins-literary-award">Japanese AI Writes a Novel, Nearly Wins Literary Award</a></em></li>
	<li><em><a href="https://www.researchgate.net/publication/221345823_Generating_Text_with_Recurrent_Neural_Networks">Generating Text with Recurrent Neural Networks</a></em>, Sutskever et al.</li>
	<li><em><a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/">The Unreasonable Effectiveness of Recurrent Neural Networks</a></em>,  Andrej Karpathy.</li>
</ul>
Uma pequena versão do projeto está disponível no <a href="https://github.com/matheusfacure/LiterNet">meu GitHub</a>
