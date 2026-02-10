#import "@preview/touying:0.6.1": *

#let main_title = [Supervised Learning: a tutorial example]
#let main_author = [#link("https://www.bpesquet.fr")[Baptiste Pesquet]]

#import themes.simple: *
#show: simple-theme.with(
  aspect-ratio: "16-9",
  footer: [#main_title],
  // Uncomment the following line to obtain an animation-free version
  config-common(handout: true),
)

#set math.equation(numbering: "(1)")
#show figure.caption: set text(size: 16pt)
#show table: set align(right)

#title-slide[
  // Institution logos
  #place(
    bottom + left,
    dx: 10mm,
    link("https://ensc.bordeaux-inp.fr")[#image("../../images/ensc_logo.jpg", width: 15%)],
  )
  #place(
    bottom + right,
    dx: -10mm,
    link("https://www.institutoptique.fr")[#image("../../images/iogs_logo.jpg", width: 15%)],
  )

  #title[
    #main_title
  ]

  #v(1em)

  #main_author

  #datetime.today().display("[month repr:long] [year repr:full]")
  // Hard value for event date
  // #datetime(year: 2026, month: 01, day: 06).display("[month repr:long] [day], [year repr:full]")
]

== Outline <touying:hidden>

#show outline: set text(size: 20pt)

#components.adaptive-columns(outline(title: none, depth: 2, indent: 1em))

= Components

== Data

- Input variable: $x$.
- Target variable: $t$.
- $N$ observations of $x$, denoted $x_1,x_2,...,x_N$.
- $N$ corresponding values of $t$, denoted $t_1,t_2,...,t_N$.
- A couple ($x_i, t_i$) is called a _data sample_.

=== Example: a noisy sinusoidal training set

#figure(
  image("images/Figure_1.png", width: 51%),
  caption: [Training set of $N=10$ samples. Inputs $x_i$ are scalars in the $[0,1]$ range. Targets values $t_i$ are generated using the function $t = sin(2 pi x) + "noise"$.],
)

== Model

Learned relationship between inputs and targets.

Example: @model defines a _linear model_.

$ y(x,bold(w)) = w_0 + w_1 x + w_2 x^2 ... + w_M x^M = sum_(j=0)^M w_j x^j $ <model>

- $bold(w)$: vector of polynomial coefficients, also called _weights_ or _parameters_.
- $M$: order of the polynomial function.

=== What the model "sees"

#figure(
  image("images/Figure_2.png", width: 51%),
)

== Loss function

Quantifies the misfit between model output and expected results for any given value of $bold(w)$.

Also called _error_ or _cost function_.

Example: @sum_of_squares defines the _sum of squares_ loss function.

$ L(bold(w)) = 1/2 sum_(n=1)^N (y(x_n, bold(w)) - t_n)^2 $ <sum_of_squares>

=== Example: errors for some training samples

#figure(
  image("images/Figure_3.png", width: 49%),
  caption: [Adapted from @bishop2023learning],
)

== Optimization algorithm

Goal: find the set of weights $bold(w^*)$ that minimizes the loss function $L(bold(w))$.

Can be done either by _analytical_ or _numerical_ methods.

Example: minimization of $L(bold(w))$ defines a system of $M$ equations with $M$ unknowns $w_i$ (@error_deriv).

$ partial(L(bold(w)))/partial(w_i) = 0 <=> sum_(j=0)^M A_(i j) w_j = T_i $ <error_deriv>

#align(center)[With $A_(i j) = sum_(n=1)^N x_n^(i+j)$ and $T_i = sum_(n=1)^N x_n^î t_n$]

= Factors

== Model complexity

=== Example: impact of model complexity on fitting

#figure(
  image("images/Figure_4.png", width: 125%),
)

=== Generalization

Ability of a trained model to perform well with new, unseen data.

Measured through a dedicated _test set_ and a performance metric.

Example: @rms defines the _Root Mean Square_ error function.

$ E_("RMS") = sqrt(1/N sum_(n=1)^N (y(x_n,bold(w^*)) - t_n)^2) $ <rms>

=== Example: impact of model complexity on generalization error

#figure(
  image("images/Figure_5.png", width: 52%),
)

=== Example: impact of model complexity on weights

#figure(
  table(
    columns: 5,
    stroke: none,

    table.header[][$M = 0$][$M = 1$][$M = 3$][$M = 9$],
    [$w_0^*$], [$-0.0375$], [$0.3519$], [$-0.4385$], [$-0.5625$],
    [$w_1^*$], [], [$-0.7788$], [$10.2930$], [$-6.3815$],
    [$w_2^*$], [], [], [$26.9737$], [$232.25$],
    [$w_3^*$], [], [], [$17.0830$], [$-1038.7$],
    [$w_4^*$], [], [], [$$], [$841.95$],
    [$w_5^*$], [], [], [$$], [$3568.1$],
    [$w_6^*$], [], [], [$$], [$-8702.6$],
    [$w_7^*$], [], [], [$$], [$6689$],
    [$w_8^*$], [], [], [$$], [$-1040.7$],
    [$w_9^*$], [], [], [$$], [$-542.7$],
  ),
)

== Dataset size

=== Example: impact of dataset size on overfitting

#figure(
  image("images/Figure_6.png", width: 69%),
)

== Regularization

Add a penalty term to the loss function in order to lower the magnitude of weights.

Example: @reg_sum_of_squares uses _L2 regularization_.

$ L(bold(w)) = 1/2 sum_(n=1)^N (y(x_n, bold(w)) - t_n)^2 + lambda/2 norm(bold(w))^2 $ <reg_sum_of_squares>

- $lambda$: regularization rate.
- $norm(bold(w))^2 = sum_(j=1)^M w_j^2$

=== Example: impact of regularization on overfitting

#figure(
  image("images/Figure_7.png", width: 125%),
)

=== Example: impact of regularization on generalization error

#figure(
  image("images/Figure_8.png", width: 52%),
)

=== Example: impact of regularization on model weights

#figure(
  table(
    columns: 4,
    stroke: none,

    table.header[][$ln lambda = -infinity$][$ln lambda = -10$][$ln lambda = 0$],
    [$w_0^*$], [$-0.5625$], [$-0.5515$], [$0.1955$],
    [$w_1^*$], [$-6.3815$], [$12.6206$], [$-0.1786$],
    [$w_2^*$], [$232.25$], [$-29.8103$], [$-0.2508$],
    [$w_3^*$], [$-1038.7$], [$4.0986$], [$-0.1764$],
    [$w_4^*$], [$841.95$], [$13.9172$], [$-0.0975$],
    [$w_5^*$], [$3568.1$], [$7.9275$], [$-0.0375$],
    [$w_6^*$], [$-8702.6$], [$0.8214$], [$0.0045$],
    [$w_7^*$], [$6689$], [$-2.6544$], [$0.0329$],
    [$w_8^*$], [$-1040.7$], [$-3.3195$], [$0.0518$],
    [$w_9^*$], [$-542.7$], [$-3.3804$], [$0.0640$],
  ),
)

== Model selection

=== Hyperparameters

- $M$ and $lambda$ are examples of _hyperparameters_: configuration properties chosen before training.
- Exploring combinations of hyperparameters can be costly and even untractable.

=== Training, validation and test sets

#figure(
  image("images/Figure_9.png", width: 70%),
)

=== K-fold cross-validation

#figure(
  image("images/Figure_10.png", width: 50%),
  caption: [Adapted from @bishop2023learning.],
)

== From example to reality

- Dataset size.
- Several input values (_features_) for each sample.
- Number of model parameters ($>1"B"$ in some cases).
- Iterative optimization techniques for loss minimization.
- Computational cost.

= References

#slide[
  #set text(size: 20pt)

  #bibliography("./bibliography.bib", title: none)
]
