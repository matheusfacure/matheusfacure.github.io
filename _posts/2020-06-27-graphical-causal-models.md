---
layout: tutorial
tags: [Econ]
comments: true
title: 4 - Graphical Causal Models
subtitle: Reasoning About Causality 
date: 2020-06-27
true-dt: 2020-06-27
author: "Matheus Facure"
---

## Thinking About Causality

One of the main assumptions that we make when doing causal inference is that the treatment is at least conditionally independent of the potential outcomes.

$$
(Y_0, Y_1) \perp T | X
$$

This means that we are able to measure an effect on the outcome that is solely due to the treatment, and not any other variable lurking around. The classic example of this is the effect of a medicine on an ill patient. If only severely ill patients get the drug, it might even look like giving the drug decreases the patient's health. That is because the effect of the severity is getting mixed up with the effect of the drug. If, however, we break down the patients by severe and not severe cases and analyse the drug impact in each subgroup, we will get a more clear picture of the true effect. This breaking down the population by its features is what we call controlling for or conditioning on X. By conditioning on the severe cases, the treatment mechanism becomes as good as random. Patientes whiting the severe group may or may not receive the drug only due to chance, not due a high severity anymore, since all patients are the same on this dimension. And if treatment is as if randomly assigned within groups, the treatment becomes conditionally independent of the potential outcomes. 

Independence and conditional independence are central to causal inference. Yet, it can be quite challenging to wrap our head around them. To help us, we use the framework of **causal graphical models**. A causal graphical model is a way to represent how causality works in terms of what causes what. As we will see, it is also a powerful way to think about causality and understand what variables we should condition on in order to make the potential outcomes independent of the treatment.

A graphical model looks like this

![svg](/img/econ/graphs/output_2_0.svg)


Each node is a random variable. We use arrows, or edges, to show if a variable causes another. In the first graphical model above we are saying that Z causes X and that U causes X and Y. To give a more concrete example, we can translate our thoughts about the impact of the medicine on patient survival as the second graph above. Severeness causes both medicine and survival and medicine also causes survival. As we will see, causal graphical models help us make our thinking about causality more clear, as it makes explicit our beliefs about how the world works. 

## Crash Course in Graphical Models

There are [whole semesters on graphical models](https://www.coursera.org/specializations/probabilistic-graphical-models). But, for our purpose, it is just (very) important that we understand what kind of independence and conditional independence assumptions a graphical model entails. As we shall see, independence flows through a graphical model like water flows through a stream. We can stop this flow or we can enable it, depending on how we treat the variables in it. To understand this, let's examine some common graphical structures and examples. They will be quite simple, but they are the sufficient building blocks to understand everything about independence and conditional independence on graphical models.

First, look at this very simple graph. A causes B which causes C. Or X causes Y which causes Z.


![svg](/img/econ/graphs/output_4_0.svg)


In the first graph, dependence flows in the direction of the arrows. To give a more concrete example, let's say that knowing about causal inference is the only way to solve business problems and solving those problems is the only way to get a job promotion. So causal knowledge causes problem solving which causes job promotion. We can say here that job promotion is dependent on causal knowledge. The greater the causal knowledge, the greater your chances of getting a promotion. Notice that dependence is symmetric, although it is a little less intuitive. The greater your chances of promotion, the greater the chance you have causal knowledge, otherwise it would be difficult to get a promotion. 

Now, let's say I condition on the intermediary variable. In this case, the dependence is blocked. So, X and Z are independent given Y. By the same token, in our exemple, if I know you are good at solving problems, knowing that you know causal inference doesn't give any further information about your chances of getting a job promotion. In mathematical terms, \\(E[Promotion\|Solve \ problems, Causal \ knowledge]=E[Promotion\|Solve \ problems]\\). The inverse is also true, once I know how good you are at solving problems, knowing about your job promotion status gives me no further information about how likely you are to know causal inference. 

As a general rule, the dependence flow in the direct path from A to B is blocked when we condition on an intermediary variable C. Or,

$$A \not\!\perp\!\!\!\perp B$$

and

$$
A \!\perp\!\!\!\perp B | C
$$

Now, let's consider a fork structure in the graph. In this case, the same variable causes two other variables down the graph. In this case, the dependence flows backward through the arrows and we have what it is called a **backdoor path**. We can close the backdoor path and shut down dependence by conditioning on the common cause.


<img class="img-responsive center-block" src="/img/econ/graphs/output_6_0.svg" style="width: 100%;" alt=""/>

As an example, let's say your knowledge of statistics causes you to know more of causal inference and machine learning. If I don't know your level of statistical knowledge, then knowing that you are good at causal inference makes it more likely that you are also good at machine learning. That is because even if I don't know your level of statistical knowledge, I can infer it from your causal inference knowledge: if you are good at causal inference you are probably good at statistics, which also makes it more likely that you are good at machine learning. 

Now, if I condition on your knowledge about statistics, then how much you know about machine learning becomes independent of how much you know about causal inference. You see, knowing your level of statistics already gives me all the information I need to infer the level of your machine learning skills. Knowing your level of causal inference will give no further information in this case. 

As a general rule, two variables that share a common cause are dependent, but independent when we condition on the common cause. Or

$$A \not\!\perp\!\!\!\perp B$$

and

$$
A \!\perp\!\!\!\perp B | C
$$

The only structure that is missing is the collider. A collider is when two arrows collide on a single variable. We can say that in this case both variables share a common effect. 

<img class="img-responsive center-block" src="/img/econ/graphs/output_8_0.svg" style="width: 100%;" alt=""/>


As an example, consider that there are two ways to get a job promotion. You can either be good at statistics or flatter your boss. If I don't condition on your job promotion, that is, I know nothing if you will or won't get it, then your level of statistics and flattering are independent. In other words, knowing how good you are at statistics tells me nothing about how good you are at flattering your boss. On the other hand, if you did get a job promotion, suddenly, knowing your level of statistics tells me about your level of flattering. If you are bad at statistics and you did get a promotion, it becomes more likely that you know how to flatter, otherwise you wouldn't get a promotion. Conversely, if you are bad at flattering, it must be the case that you are good at statistics. This phenomenon is sometimes called **explaining away**, because one cause already explains the effect, making the other cause less likely.

As a general rule, conditioning on a collider opens the dependence path. Not conditioning on it leaves it closed. Or

$$A \!\perp\!\!\!\perp B$$

and

$$
A \not\!\perp\!\!\!\perp B | C
$$

Knowing the three structures, we can derive an even more general rule. A path is blocked if and only if:
1. It contains a non collider that has been conditioned on
2. It contains a collider that has not been conditioned on and has no descendants that have been conditioned on.

Here is a cheat sheet about how dependence flows in a graph. I've taken from a [Stanford presentation](http://ai.stanford.edu/~paskin/gm-short-course/lec2.pdf) by Mark Paskin.

<img class="img-responsive center-block" src="/img/econ/graphs/graph-flow.png" style="width: 75%;" alt=""/>

As a final example, try to figure out some independence and dependence relationship in the following causal graph.
1. Is \\(D \perp C\\)?
2. Is \\(D \perp C\| A \\) ?
3. Is \\(D \perp C\| G \\) ?
4. Is \\(A \perp F \\) ?
5. Is \\(A \perp F\|E \\) ?
6. Is \\(A \perp F\|E,C \\) ?


![svg](/img/econ/graphs/output_10_0.svg)


**Answers**:
1. \\(D \perp C\\). It contains a colider that it has **not** been conditioned on.
2. \\(D \not \perp C\| A \\). It contains a collider that it has  been conditioned on.
3. \\(D \not\perp C\| G \\). It contains the descendant of a collider that has  been conditioned on. You can see G as some kind of proxy for A here.
4. \\(A \perp F \\). It contains a collider, B->E<-F, that it has **not** been conditioned on.
5. \\(A \not\perp F\|E \\). It contains a collider, B->E<-F, that it has been conditioned on.
6. \\(A \perp F\|E, C \\). It contains a collider, B->E<-F, that it has been conditioned on, but it contains a non collider that has been conditioned on. Conditioning on E opens the path, but conditioning on C closes it again.

Knowing about causal graphical models enables us to understand the problems that arise in causal inference. As we've seen, the problem always boils down to bias. 

$$
E[Y|T=1] - E[Y|T=0] = \underbrace{E[Y_1 - Y_0|T=1]}_{ATET} + \underbrace{\{ E[Y_0|T=1] - E[Y_0|T=0] \}}_{BIAS}
$$

Graphical models allow us to diagnose which bias we are dealing with and what are the tools we need to correct for them.

## Confounding Bias

The first big cause of bias is confounding bias. It happens when the treatment and the outcome shares a common cause. For example, let's say that the treatment is education and the outcome is income. It is hard to know the causal effect of education on the wage because both share a common cause: intelligence. So we could make the argument that more educated people earn more money simply because they are more intelligent, not because they have more education. In order to identify the causal effect, we need to close all backdoor paths between the treatment and the outcome. If we do so, the only effect that will be left is the direct effect T->Y. In our example, if we control for intelligence, that is, we compare people with the same level of intelligence but different levels of education, the difference in the outcome will be only due to the difference in education, since intelligence will be the same for everyone. So, in order to fix confounding bias, we need to control all common causes of the treatment and the outcome.


![svg](/img/econ/graphs/output_12_0.svg)


Unfortunately, it is not always possible to control for all common causes. Sometimes, there are unknown causes or known causes that we can't measureThe case of intelligence is one of the latter. Despite all the effort, scientists haven't yet figured out how to measure intelligence well. I'll use U to denure unmeasured variables here. Now, assume for a moment that intelligence can't affect your education directly. It just affects how well you do on the SATs, but it is the SATs that determine your level of education, since it opens the possibility of a good college. Even if we can't control for the unmeasurable intelligence, we  can control for SAT and close that backdoor path.


![svg](/img/econ/graphs/output_14_0.svg)


In the following graph, conditioning on X1 and X2, or, SAT and family income, is sufficient to close all backdoor paths between the treatment and the outcome. In other words, \\((Y_0, Y_1) \perp T \| X1, X2\\). So even if we can measure all common causes, we can still attain conditional independence if we control for measurable variables that mediate the effect of the unmeasured on the treatment.

But what if that is not the case? What if the unmeasured variable causes the treatment and the outcome directly? In the following example, intelligence causes education and income directly. So there is confounding on the relation between the treatment education and the outcome wage. In this case, we can't control the confounder, because it is unmeasurable. However, we have other measured variables that can act as a proxy for the confounder. Those variables are not in the backdoor path, but controlling for them will help lower the bias (but it won't eliminate it). Those variables are sometimes referred to as surrogate confounders.

In our example, we can't measure intelligence, but we can measure some causes of it, like father education and mother education, and some effects of it, like IQ or SAT score. Controlling for those surrogate variables is not sufficient to eliminate bias, but it sure helps.


![svg](/img/econ/graphs/output_16_0.svg)


## Selection Bias

Now you might think that it is a good idea to add everything you can measure to your model just to be sure you don't have confounding bias. Well, think again.

![image.png](/img/econ/graphs/selection_bias.png)

The second big source of bias is what we will call selection bias. If confounding bias happens when we don't control for a common cause, selection bias is more related to effects. One word of caution here, economists tend to refer to all sorts of bias as selection bias. Here, I think the distinction between it and confounding bias is very helpful, so I'll stick to it. 

More often than not, selection bias arises when we control for more variables that we should. It might be the case that treatment and the potential outcome are marginally independent, but become dependent once we control on a collider. 

Imagine that with the help of some miracle you are finally able to randomize education in order to measure its effect on wage. But just to be sure you won't have confounding, you control for a lot of variables. Among them, you control for investments. But investment is not a common cause of education and wage. Instead, it is a consequence of both. More educated people both earn more and invest more. Also, those who earn more invest more. Since investment is a collider, by conditioning on it, you are opening a second path between the treatment and the outcome, which will make it harder to measure the direct effect. One way to think about this is that by controlling investments, you are looking at small groups of the population where investment is the same and then finding the effect of education on those groups. But by doing so, you are also indirectly and inadvertently not allowing wages to change much. As a result, you won't be able to see how education changes wage, because you are not allowing wages to change. 



![svg](/img/econ/graphs/output_18_0.svg)


To demonstrate why this is the case if investments and education takes only 2 values. Either people invest or not. They are either educated or not. Initially, when we don't control for investments, the bias term is zero \\(E[Y_0\|T=1] - E[Y_0\|T=0] = 0\\) because the education was randomised. This means that the wage people would have in the case they didn't receive education \\(Wage_0\\) is the same if they do or don't receive the education treatment. But what happens if we condition on investments?

Looking at those that invest, we probably have the case that \\(E[Y_0\|T=0, I=1] > E[Y_0\|T=1, I=1]\\). In words, among those that invest, those that manage to do so even without education are more independent of education to achieve high earnings. For this reason, the wage those people have, \\(Wage_0\|T=0\\), is probably higher than the wage the educated group would have in the case that they didn't had education, \\(Wage_0\|T=1\\). A similar reasoning can be applied to those that don't invest, where we also probably have \\(E[Y_0\|T=0, I=0] > E[Y_0\|T=1, I=0]\\). Those that don't invest even with education, probably would have a lower wage, had they not got the education, than those that didn't invest but also didn't have education. 

To use a purely graphical argument, if someone invests, knowing that they have high education explains away the second cause which is wage. Conditioned on investing, higher education is associated with low wages and we have a negative bias \\(E[Y_0\|T=0, I=i] > E[Y_0\|T=1, I=i]\\). 

Just as a side note, all of this we've discussed is also true if we condition on any descendent of a common effect.

<img class="img-responsive center-block" src="/img/econ/graphs/output_20_0.svg" style="width: 15%;" alt=""/>

A similar thing happens when we condition on a mediator of the treatment. A mediator is a variable between the treatment and the outcome. It, well, mediates the causal effect. For example, suppose again you are able to randomize education. But, just to be sure, you decide to control whether or not the person had a white collar job. Once again, this conditioning biasses the causal effect estimation. This time, not because it opens a front door path with a collider, but because it closes one of the channels through which the treatment operates. In our example, getting a white collar job is one way that more education leads to higher pay. By controlling it, we close this channel and leave open only the direct effect of education on wage.


![svg](/img/econ/graphs/output_22_0.svg)


To give a potential outcome argument, we know that, due to randomisation, the bias is zero \\(E[Y_0\|T=1] - E[Y_0\|T=0] = 0\\). However, if we condition on the white collar individuals, we have that \\(E[Y_0\|T=0, WC=1] > E[Y_0\|T=1, WC=1]\\). That is because those that manage to get a white collar job even without education are probably more hard working than those that required the help of education to get the same job. With the same reasoning, \\(E[Y_0\|T=0, WC=0] > E[Y_0\|T=1, WC=0]\\) because those that didn't get a white collar job even with education are probably less hard working than those that didn't, but also didn't have any education. 

In our case, conditioning on the mediator induces a negative bias. It makes the effect of education seem lower than it actually is. This is the case because the causal effect is positive. In the effect where negative, conditioning on a mediator would have a positive bias. In all cases, this sort of conditioning makes the effect look weaker than it is. 

To put it in a more prosaic way, suppose that you have to choose between two candidates for a job at your company. Both have equally impressive professional achievements, but one does not have a higher education degree. Which one should you choose? Of course, you should go with the one without the higher education, because he managed to achieve the same things as the other one but had the odds stacked against him.

![image.png](/img/econ/graphs/diploma.png)

## References

I like to think of this entire series as a tribute to Joshua Angrist, Alberto Abadie and Christopher Walters for their amazing Econometrics class. Most of the ideas here are taken from their classes at the American Economic Association. Watching them is what is keeping me sane during this tough year of 2020.
* [Cross-Section Econometrics](https://www.aeaweb.org/conference/cont-ed/2017-webcasts)
* [Mastering Mostly Harmless Econometrics](https://www.aeaweb.org/conference/cont-ed/2020-webcasts)

I'll also like to reference the amazing books from Angrist. They have shown me that Econometrics, or 'Metrics as they call it, is not only extremely useful but also profoundly fun.

* [Mostly Harmless Econometrics](https://www.mostlyharmlesseconometrics.com/)
* [Mastering 'Metrics](https://www.masteringmetrics.com/)

My final reference is Miguel Hernan and Jamie Robins' book. It has been my trustworthy companion in the most thorny causal questions I had to answer.

* [Causal Inference Book](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)

The code for this and other tutorials about causality can by found at my [GitHub page](https://github.com/matheusfacure/python-causality-handbook). If you like it, don't forget to leave me a star! 

![img](/img/econ/poetry.png)
