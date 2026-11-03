#import "@preview/touying:0.7.4": *

#let main_title = [From text to vectors]
#let main_author = [#link("https://www.bpesquet.fr")[Baptiste Pesquet]]

// Link to interactive demos
#let demos_url = "https://huggingface.co/spaces/dlouapre/nlp-before-llms"
#let fig_caption = [Image credits: #link(demos_url)[David Louapre]]

// Static picture of an interactive demo from the article, linked to the live version
#let demo(file, section, height: 9cm) = figure(
  link(demos_url, image("images/" + file + ".png", height: height)),
  caption: [Interactive demo from section #section of the #link(demos_url)[article] (click to open).],
)

// Small helper for the pipeline diagram
#let stage(body) = box(
  stroke: 1pt + gray,
  radius: 6pt,
  inset: 10pt,
  width: 100%,
  align(center, body),
)

#import themes.simple: *
#show: simple-theme.with(
  aspect-ratio: "16-9",
  footer: [#main_title],
  // Uncomment the following line to obtain an animation-free version
  // config-common(handout: true),
)

#set quote(block: true)
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

#components.adaptive-columns(outline(title: none, depth: 1, indent: 1em))

= Overview of Natural Language Processing

== Four main tasks

#figure(
  image("images/nlp_tasks.png", width: 51%),
  caption: fig_caption,
)

== A common objective

#align(horizon, [Learning _conditional probability distributions given an input text_, seen as a sequence of words.])

== Why language is hard

#item-by-item-fn("past-faded")[
  - _Ambiguity_: "I saw the man with the telescope", "La petite brise la glace".
  - _Variability_: “I can’t make it”, “won’t be able to come, sorry”, “cant come sry 😕”.
  - _Long-range dependencies_: "The keys that I left on the kitchen table yesterday _are_...".
  - _Zipf's law_: in a corpus, $"word frequency" prop 1/"word rank" ==>$ most words are rare.
]

== Classification

- Assign a text to one of a predefined list of classes: spam detection, sentiment analysis, topic classification...
- The classifier learns the conditional probability that a certain sequence of words $x_1 dots x_L$ belongs to a class $C$:
$ p(C | x_1 dots x_L) $
- The length $L$ is not fixed a priori: it varies from one text to another.

---

=== Sentiment can be subtle 😊

#quote(attribution: [Bag-of-words model: 76% positive])[
  "Wow, what a masterpiece. Two hours of pure brilliance,
  and I only checked my watch twenty times.
  The acting was so convincing that I truly believed
  the actors wanted to be somewhere else.
  A perfect film if you love naps."
]

== Language modeling

- In principle: learn the probability $p(x_1 dots x_L)$ of any word sequence.
- In practice, learn the _next-word probability_:
$ p(x_(L+1) | x_1 dots x_L) $
- The _chain rule_ recovers the probability of a whole sequence:
$ p(x_1 x_2 x_3 x_4) = p(x_1) thin p(x_2 | x_1) thin p(x_3 | x_1 x_2) thin p(x_4 | x_1 x_2 x_3) $

---

- Main modern use: _autoregressive generation_.
- Other uses: scoring candidate sentences ("by Albert Einstein" vs. "by Alfred Einstein").
- Reduces to a classification problem, with classes the words of a fixed vocabulary.
  - Big difference: number of possible classes.

== Translation

- In principle: learn the probability of a candidate translation $y_1 dots y_K$ given a source sentence $x_1 dots x_L$.
- In practice, estimate the next word of the translation:
$ p(y_(i+1) | y_1 dots y_i ; x_1 dots x_L) $
- Not trivial at all:
  - the two sentences don't have the same length;
  - translation is not a word-to-word correspondence;
  - word order can differ completely between languages.

---

=== Word order example

#figure(
  link(demos_url + "#13-translation")[
    #image("images/word_alignment.png", width: 95%)],
  caption: fig_caption,
)

== Tagging

- Assign a class (_tag_) to each word or group of words:
  - grammatical error detection;
  - grammatical category of each word in a sentence (noun, verb...);
  - _Named Entity Recognition (NER)_: names of people, places...
- Can be seen as a classifier applied to each word, taking its context into account:
$ p(t_i | x_1 dots x_L) $

= Counting words: n-gram models

== Markov model

- Simplest/dumbest possible language model: the _unigram_ (1-gram) model draws words randomly according to their frequency.
$ p(x_(L+1) | x_1 dots x_L) = p(x_i) = (C(x_i)) / N $
- $C(x_i)$: count of word $x_i$ in the training corpus; $N = sum_i C(x_i)$: corpus size.

---

=== Simple Markov chain

- In a _bigram_ model, the next word only depends on the previous one:
$ p(x_(L+1) | x_1 dots x_L) = p(x_(L+1) | x_L) $
- Poor assumption for human language, but very simple to model with a _transition matrix_.
- Method: count each word $C(v)$ and each pair of consecutive words $C(v, w)$, then estimate $p(w | v) = C(v,w) \/ C(v)$.

👩🏻‍💻 Activity 1: build a Markov chain model

---

=== n-gram models

- An _n-gram_ model predicts the next word from the $n-1$ previous ones:
$ p(x_(L+1) | x_1 dots x_L) = p(x_(L+1) | x_(L-n+2) dots x_L) " for "n>1 $
- The transition matrix has size $V^(n-1) times V$: for $n = 3$ and $V = 30,000$, about $3 times 10^13$ entries!
- In practice, only the n-grams that actually occur are stored in a count dictionary.

---

=== Impact of $n$ on next word probabilities

#figure(
  link(demos_url + "#n-gram-models")[
    #image("images/n_gram_explorer.png", width: 64%)],
  caption: [Dataset: WikiText-2. #fig_caption],
)

== Comparing probability distributions

=== Problem statement

- Goal: approximate the "true" next-word distribution $p(x_(L+1) | x_1 dots x_L)$.
- Any candidate model predicts a distribution $q(x_(L+1) | x_1 dots x_L)$.

How do we measure how close $q$ is to $p$, which is a priori unknown?

---

=== KL divergence and cross-entropy

- Standard tool: the _Kullback–Leibler_ (KL) divergence.
$ D_"KL" (p parallel q) = sum_x p(x) log (p(x)) / (q(x)) $
- It splits into two terms; the first is minus the _entropy_ of $p$, which doesn't depend on $q$.
$ D_"KL" (p parallel q) = sum_x p(x) log p(x) - sum_x p(x) log q(x) $

---

=== Cross-entropy

The second term is the _cross-entropy_:
$ H(p, q) = -sum_x p(x) log q(x) $
$==>$ Finding $q$ that minimizes $D_"KL" (p parallel q)$ is equivalent to minimizing the cross-entropy $H(p, q)$.

---

=== Observing entropy, cross-entropy and KL divergence

#figure(
  link(demos_url + "#kl-divergence-and-cross-entropy-loss")[
    #image("images/kl_divergence.png", width: 71%)],
  caption: fig_caption,
)

---

=== Expectation over distribution probability

$ bb(E)_(x tilde p) [f(x)] = sum_x p(x)f(x) $

With a context $c = (x_1, dots, x_L)$ and a model $q_theta$ depending on parameters $theta$, we seek to minimize:

$ -sum_x p(x | c) log q_theta (x | c) = -bb(E)_(x tilde p(dot | c)) [log q_theta (x | c)] $

$p$ is unknown, but our text corpus is a sample drawn from $p$.

---

=== Law of large numbers

As $N$ grows, the frequencies converge to the true probabilities.

$ 1/N sum_(i=1)^N f(x_i) stretch(->)^(N -> oo) bb(E)_(x ~ p) [f(x)] $

$==>$ Averaging over the training samples estimates the cross-entropy.

$
  hat(H) = -1/N sum_(i=1)^N log q_theta (x_i | x_(<i)) stretch(->)^(N -> oo) bb(E)_(x tilde p(dot | c)) [log q_theta (x | c)]
$

---

=== Likelihood

$L(theta)$: probability the model gives to the whole training set. Assuming independance between samples:

$ L(theta) = product_(i=1)^N q_theta (x_i | x_(<i)) $
$
  op("argmax", limits: #true)_theta 1/N log L(theta)
  = op("argmin", limits: #true)_theta (
    - 1/N sum_(i=1)^N log q_theta (x_i mid(|) x_(<i))
  )
  = op("argmin", limits: #true)_theta hat(H)
$

---

=== KL divergence, cross-entropy and likelihood

- $hat(H)$ is the _negative log-likelihood_ of the text divided by its number of words.
- Maximizing the likelihood of the corpus $<==>$ Minimizing cross-entropy $<==>$ Minimizing the KL divergence.

---

=== Perplexity

Cross-entropy is hard to read: we usually report its exponential, the _perplexity_ $cal(P)$:

$ cal(P) = exp(-1/N sum_(i=1)^N log q(x_i | x_(<i))) = (product_(i=1)^N 1/(q(x_i | x_(<i))))^(1/N) $

---

=== Interpreting perplexity

Perplexity can be seen as the number of words the model hesitates between.

- If it guesses uniformly among $k$ words, each next word gets probability $1/k$, so perplexity is $k$.
- A perplexity of 20 means the model is about as uncertain as if it were picking uniformly among 20 words each time.

---

=== Validation and test sets

- Never measure perplexity on the training text: a model that memorizes it would look great!
- The _validation set_ is used for every model choice (e.g. the value of $n$).
- The _test set_ is used only once, at the end, to report the official number.

👨🏼‍💻 Activity 2: evaluate the perplexity of the unigram and bigram models.

= Classification with bag of words

== Architecture of a NLP classifier

- Input texts have variable length, and the key information can be anywhere in a review.
- Classic ML methods (logistic regression, MLPs, trees...) need fixed-size inputs.
- A classification model for NLP can be split into two steps:
  - an _encoder_ builds a fixed-size (_latent_) representation of the text;
  - a _classification head_ computes class probabilities from it.
- n-gram models follow this pattern: they keep the last $n-1$ words, and the head is a count table.

---

#figure(
  image("images/encoder_head.png", width: 65%),
  caption: fig_caption,
)

== Bag of words

=== Encoder

- Choose a vocabulary of $V$ words (for instance, the most frequent ones in the training set).
- Each sample becomes a _sparse_ count vector of fixed size $V$, whose entry number $v$ is the count $C_v$ of the $v$-th word of the vocabulary.
- Consequences:
  - Words outside the vocabulary are ignored.
  - Word order is lost.

---

=== Logistic regression head for binary classification

- One weight $w_v$ per vocabulary word $v$:

$ p("positive class" | x_1 dots x_L) = sigma(b + sum_(v=1)^V w_v C_v) " with" sigma(z) = 1/(1 + e^(-z)) $

- Each occurrence of a word pushes the score up or down by its weight.
- Weights are learned from labeled samples by minimizing the cross-entropy loss.

---

=== Testing a bag of words model for movie reviews

#figure(
  link(demos_url + "#32-bag-of-words")[
    #image("images/bag_of_words_movies.png", width: 66%)],
  caption: fig_caption,
)

== TF-IDF weighting

- Words like _the_, _a_, _and_ appear very frequently but are uninformative.
- _Inverse Document Frequency_ (IDF) downweights frequent words:

$ "idf"(v) = -log N_v / N = log N / N_v $

- $N$: number of documents; $N_v$: number of documents containing $v$.
- Each entry becomes $C_v times "idf"(v)$: _Term Frequency $times$ Inverse Document Frequency_.

---

=== Example

- $d_1$ = "the movie is great", $d_2$ = "the movie is bad".

#figure(
  table(
    columns: 6,
    align: center,
    table.header([], [*the*], [*movie*], [*is*], [*great*], [*bad*]),
    [$d_1$], [1], [1], [1], [1], [0],
    [$d_2$], [1], [1], [1], [0], [1],
    [idf], [0], [0], [0], [$log_2 2$], [$log_2 2$],
  ),
)

- Frequent words weight nothing: only _great_ and _bad_ tell reviews apart.
- On IMDB dataset with $log_2$: _the_ $->$ 0.01, _boring_ $->$ 4, _flabbergasting_ $->$ 13.6.

== Putting bag of words to the test

🧑🏻‍💻 Activity 3: classify movie reviews with various bag of words variants.

- Pros:
  - 87.6% accuracy without knowing the order of the words!
  - Weights are interpretable.
- Limits:
  - “superb” is as different from “excellent” as it is from “boring”.
  - "I expected it to be terrible, but it was great." identical to "I expected it to be great, but it was terrible." for the model.
  - Negation ("Not bad at all") and sarcasm fool it too.

= Dense word embeddings

== The embedding matrix

- Idea: represent each word by a dense vector of real numbers in a space of dimension $D << V$ (e.g. $D = 300$).
- These vectors are called _embeddings_.
- All embeddings are stored in an _embedding matrix_ $W_E in cal(M)_(D times V)(RR)$ with one column per word.
- Looking up an embedding = multiplying $W_E$ by the word's one-hot vector.

---

=== Example of embedding matrix

#figure(
  link(demos_url + "#41-the-embedding-matrix")[
    #image("images/embedding_matrix.png", width: 73%)],
  caption: fig_caption,
)

---

=== Cosine similarity

- A benefit of dense embeddings over sparse representations is the possibility of comparisons.
- Example: _cosine similarity_:

$ cos(u, v) = (u dot v) / (norm(u) norm(v)) in [-1,1] $

- 1 $==>$ same direction, 0 $==>$ orthogonal, $-1$ $==>$ opposite directions.

---

=== Reflecting semantics

Similarity between the vectors should reflect the similarity between the meanings of the words.

$ bold(v)_"king" - bold(v)_"man" + bold(v)_"woman" approx bold(v)_"queen" $

#figure(
  image("images/word2vec_analogy.svg", width: 25%),
  caption: [Idealized 2D projection: real embeddings have hundreds of dimensions.],
)

---

=== Aggregating word vectors

- Replacing $L$ words by $L$ vectors of size $D$ still doesn't give a fixed-size representation.
- An aggregation (pooling) step is needed:
  - average of the $L$ vectors $->$ size $D$;
  - weighted average to favor important words $->$ size $D$;
  - concatenation of the last $n - 1$ vectors $->$ size $(n-1) times D$.
- Concatenation keeps the order of the last words: a good fit for language modeling.

---

=== Aggregation methods

#figure(
  link(demos_url + "#41-the-embedding-matrix")[
    #image("images/embedding_aggregation.png", width: 39%)],
  caption: fig_caption,
)

== Bengio's neural language model

- Bengio et al. (2003): the first neural language model!
- Keep the Markov assumption, but replace the count table with a neural network reading the concatenated embeddings of the $n - 1$ previous words.
- Look up embeddings $e_(L-n+2), dots, e_L$, concatenate them into a context vector $c$, feed $c$ to a one-hidden-layer  network (MLP).
- Word embeddings and network weights are jointly learned by minimizing the cross-entropy loss with backpropagation.
- The model generalizes from seen contexts to new, unseen ones.

== Averaging the word vectors of a text

- For classification, concatenating the last words is useless.
- Simplest option: *average* the embeddings. With the bag of words count vector $C$ and $f=C/L$ the vector of the frequencies of the words in the sample.

$ 1/L sum_(i=1)^L e_i = 1/L W_E C = W_E f $

- The average of the embeddings is a bag of words, linearly compressed by $W_E$ from $V$ to $D$ dimensions.

= Pre-training embeddings

== Two-stage training

- What if we could learn embeddings once, on a task with abundant free data, and reuse them everywhere?
- _Pre-training_: learn the embedding matrix $W_E$ on a task with abundant, free data.
- _Distributional hypothesis_: words that appear in similar contexts tend to have similar meanings.

#quote(attribution: [J.R. Firth, 1957])[
  "You shall know a word by the company it keeps!"
]

== word2vec

- Two architectures by Mikolov et al. (2013), using a window around a middle word:
  - CBOW (_Continuous Bag of Words_): predict the middle word from the average of the surrounding embeddings;
  - _Skip-gram_: predict each surrounding word from the middle word's embedding.
- Neither have a hidden layer: going from embeddings to word scores costs a single matrix–vector product.
- Light and fast enough to learn from billions of words.

---

=== Architecture comparison

#figure(
  link(demos_url + "#41-the-embedding-matrix")[
    #image("images/word2vec_models.png", width: 37%)],
  caption: fig_caption,
)

== Putting word2vec to the test

🧑🏻‍💻 Activity 4: train word2vec on a Wikipedia sample.

- Pros:
  - Related words end up close together.
  - Works fine for common words (which depends on dataset).
- Limits:
  - Captures _relatedness of usage_, not equivalence of meaning.
  - One word $<==>$ one vector: different meanings get mixed.
    - "We had a picnic on the river _bank_."
    - "My _bank_ closed my account."

== Transfer learning

#figure(
  link(demos_url + "#55-transfer-learning")[
    #image("images/transfer.png", width: 55%)],
  caption: fig_caption,
)

== Learned biases

Embeddings also learn the stereotypes present in their training corpus.

#align(center)[
  man : computer programmer $approx$ woman : *homemaker*
]

#text(size: 16pt)[Analogy produced by word2vec embeddings trained on news articles (Bolukbasi et al., 2016).]

- Origin: the model faithfully reproduces corpus co-occurrences, stereotypes included.
- Consequence: biases spread to applications (résumé screening, search, recommendation).
- What to do: measure (association tests), mitigate, document data and models.

= Tokenization

== What is a token?

- So far, we have treated words as the basic units of language.
- A model needs a _finite vocabulary_ of discrete symbols.
- A _tokenizer_ converts raw text into a sequence of tokens:
  - words, parts of words, characters, or bytes;
  - each token is mapped to an integer ID.
- The model processes these IDs, not raw text.

== Three granularities

#figure(
  table(
    columns: (1fr, 1.6fr, 1fr, 1fr, 1.2fr),
    align: left,
    table.header([*Unit*], [*"deconstruction"*], [*Vocabulary*], [*Sequences*], [*Unknown word*]),
    [Word], [`[deconstruction]`], [huge], [short], [frequent],
    [Character], [`[d] [e] [c] [o] …`], [tiny], [very long], [impossible],
    [Subword], [`[de] [construct] [ion]`], [fixed in advance], [medium], [split into pieces],
  ),
  caption: [Subwords: frequent pieces stay whole, rare words break apart. No word is ever _out of vocabulary_. The example split is illustrative.],
)

== Byte Pair Encoding (BPE)

=== Algorithm

+ Start from the characters of each word.
+ Count adjacent pairs, weighted by word frequency.
+ Merge the most frequent pair into a new symbol.
+ Repeat until the target vocabulary size is reached.

To tokenize new text, replay the learned merges in the same order.

---

=== Example

Corpus: _cat_ ×5, _cats_ ×2, _catch_ ×3, _rat_ ×4.

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: left,
    table.header([*Step*], [*Merge*], [*Segmented corpus*]),
    [0], [—], [`c a t · c a t s · c a t c h · r a t`],
    [1], [`a+t` (14)], [`c at · c at s · c at c h · r at`],
    [2], [`c+at` (10)], [`cat · cat s · cat c h · r at`],
    [3], [`r+at` (4)], [`cat · cat s · cat c h · rat`],
  ),
)

- Learned vocabulary: characters + `at`, `cat`, `rat`.
- A new word such as _rats_ splits into known pieces.

#speaker-note[
  Step 1: (a, t) occurs 5 + 2 + 3 + 4 = 14 times. Step 2: (c, at) occurs 5 + 2 + 3 = 10 times. Step 3: (r, at) wins with 4, ahead of (cat, c) and (c, h) with 3. Ties are broken by a fixed rule.
]

== Tokenization algorithms

#figure(
  table(
    columns: (auto, 2fr, 1.2fr),
    align: left,
    table.header([*Algorithm*], [*How it picks the units*], [*Example model*]),
    [BPE], [merges the most frequent pair], [GPT-2 (byte-level)],
    [WordPiece], [merges the pair that most increases corpus likelihood], [BERT],
    [Unigram], [starts from a large vocabulary and prunes the least useful pieces], [T5],
  ),
)

_SentencePiece_ applies BPE or Unigram to raw text, spaces included (written ▁): no language-specific preprocessing.

== Practical consequences

- Other languages cost more: a tokenizer trained mostly on English splits French more finely. More tokens $==>$ shorter usable context and higher cost.
- Numbers get fragmented: "12345" may become "123" + "45". Digits no longer line up and arithmetic gets harder.
- Code too: indentation and variable names fragment if the tokenizer saw no code during training.

#speaker-note[
  The number split is illustrative: the exact split depends on the tokenizer.
]
