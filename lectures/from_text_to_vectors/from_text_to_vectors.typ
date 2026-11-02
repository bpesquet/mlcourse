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

#import themes.simple: *
#show: simple-theme.with(
  aspect-ratio: "16-9",
  footer: [#main_title],
  // Uncomment the following line to obtain an animation-free version
  // config-common(handout: true),
)

#set math.equation(numbering: "(1)")
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

#components.adaptive-columns(outline(title: none, depth: 2, indent: 1em))

= Overview of NLP

== Four main tasks

// #demo("nlp_tasks", "1", height: 9.0cm)

#figure(
  image("images/nlp_tasks.png", width: 51%),
  caption: fig_caption,
)

== Why language is hard

#item-by-item-fn("past-faded")[
  - _Ambiguity_: "I saw the man with the telescope", "La petite brise la glace".
  - _Variability_: “I can’t make it”, “won’t be able to come, sorry”, “cant come sry 😕”.
  - _Long-range dependencies_: "The keys that I left on the kitchen table yesterday _are_...".
  - _Zipf's law_: in a corpus, $"word frequency" prop 1/"word rank" =>$ most words are rare.
]

== A common objective

Learning _conditional probability distributions given an input text_, seen as a sequence of words.

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

== Markov models

- Simplest possible language model: the *unigram* (1-gram) model draws words according to their frequency.

$ p(x_i | x_1 dots x_(i-1)) = p(x_i) = (C(x_i)) / N $

- $C(x_i)$: count of $x_i$ in the training corpus; $N = sum C(x_i)$: corpus size.
- Historical note: Markov applied his chains to text in 1913, computing vowel/consonant transitions in Pushkin's _Eugene Onegin_.

---

=== Simple Markov chain

- In a *bigram* model, a word only depends on the previous one:

$ p(x_i | x_1 dots x_(i-1)) = p(x_i | x_(i-1)) $

- Wrong for human language, but very simple to model with a _transition matrix_.

---

- Count each word $C(v)$ and each pair of consecutive words $C(v, w)$, then estimate $p(w | v) = C(v,w) \/ C(v)$.
- Exercise 2.1: generated text has no global meaning, but local word order is mostly grammatical.

---

=== n-gram models

- An *n-gram* model predicts the next word from the $n-1$ previous words.
- Bigram: $n = 2$. Unigram: $n = 1$ (no context).
- The transition matrix has size $V^(n-1) times V$: for $n = 3$ and $V = 30,000$, about $3 times 10^13$ entries!
- In practice, *only the n-grams that actually occur are stored* in a count dictionary.
- Shannon (1948) built such approximations of English... by sampling them by hand from books.

---

#demo("ngram_explorer", "2.1", height: 7.3cm)

== Comparing probability distributions

- Goal: approximate the "true" next-word distribution $p(x_(L+1) | x_1 dots x_L)$.
- Any candidate model predicts a distribution $q(x_(L+1) | x_1 dots x_L)$.
  - For an n-gram model, it only depends on $x_(L-n+2) dots x_L$.
- *How do we measure how close $q$ is to $p$?*

---

=== KL divergence and cross-entropy

- Standard tool: the *Kullback–Leibler (KL) divergence*.

$ D_"KL" (p parallel q) = sum_x p(x) log (p(x)) / (q(x)) $

---

- It splits into two terms:

$ D_"KL" (p parallel q) = sum_x p(x) log p(x) - sum_x p(x) log q(x) $

- The first term is minus the *entropy of $p$*, which doesn't depend on $q$.

---

- The second term is the *cross-entropy*:

$ H(p, q) = -sum_x p(x) log q(x) $

- *Finding $q$ that minimizes $D_"KL" (p parallel q)$ is equivalent to minimizing the cross-entropy $H(p, q)$.*

---

#demo("kl_divergence", "2.2", height: 9.0cm)

---

=== Likelihood

- With a context $c = (x_1, dots, x_L)$ and a model $q_theta$ with parameters $theta$, we minimize:

$ -sum_x p(x | c) log q_theta (x | c) = -bb(E)_(x tilde p(dot | c)) [log q_theta (x | c)] $

---

- $p$ is unknown, but *our text corpus is a sample drawn from $p$*.
- Averaging over the $N$ positions of the text estimates the cross-entropy:

$ hat(H) = -1/N sum_(i=1)^N log q_theta (x_i | x_(<i)) $

---

- $hat(H)$ is the average *negative log-likelihood (NLL)* per word of the text.
- *Minimizing cross-entropy is equivalent to maximizing the likelihood of the corpus.*

---

- Each term is the cross-entropy with a one-hot distribution on the observed word: the *cross-entropy loss*, used to train every language model (LLMs included).

$ cal(L)_"CE" = -sum_x bb(1)[x = x_i] log q_theta (x | x_(<i)) = -log q_theta (x_i | x_(<i)) $

- For n-grams, no gradient needed: the count ratio $C(v,w) \/ C(v)$ is the maximum-likelihood solution.

---

#align(center + horizon)[
  *Maximizing likelihood = Minimizing cross-entropy loss = Minimizing KL divergence*
]

---

=== Perplexity

- Cross-entropy is hard to read: we usually report its exponential, the *perplexity*:

$ cal(P) = exp(-1/N sum_(i=1)^N log q(x_i | x_(<i))) = (product_(i=1)^N 1/(q(x_i | x_(<i))))^(1/N) $

---

- A uniform model over $V$ words has perplexity $V$.
- *Perplexity = number of words the model hesitates between at each step.*
- Shannon (1951) used humans as language models: \~1 bit per letter with long contexts, i.e. a perplexity of \~2.

---

=== Validation and test sets

- Never measure perplexity on the training text: a model that memorizes it would look great.
- The *validation set* is used for every model choice (e.g. the value of $n$).
- The *test set* is used only once, at the end, to report the official number.

---

- Exercise 2.2:
  - the unigram model hesitates between \~1,000 words instead of 33,277;
  - the bigram model has *infinite* perplexity: one validation bigram out of five never occurs in the training set!

== The curse of dimensionality

- 33,000 words $==>$ \~1.1 billion possible pairs, but a 2-million-word text contains at most 2 million pairs (only \~0.06% of them in practice).
- Unseen combinations appear in test texts as early as $n = 2$ or $n = 3$.
- *The number of possible contexts grows exponentially as $V^(n-1)$, while the training text size stays fixed.*
- Exercise 2.3a: 5-gram samples look like real sentences... because 95% of validation 5-grams never occur in training. *The 5-gram model writes well only because it recites its training text.*

---

=== Fixing zero probabilities

- *Smoothing*: give unseen sequences a small non-zero probability.
  - _Add-one (Laplace)_ smoothing works badly on words: too much mass goes to billions of unseen pairs.
- *Interpolation*: mix the n-gram estimate with the $(n-1)$-gram one, recursively down to unigrams.

---

- *Kneser-Ney* smoothing: long the state of the art among counting methods.
- Exercise 2.3b: going from $n = 2$ to $n = 5$ only lowers interpolated perplexity from 346 to 296. *Counting barely exploits longer contexts.*

= Classification with bag of words

== Encoder → representation → head

- Input texts have variable length, and the key information can be anywhere in a review.
- Classic ML methods (logistic regression, MLPs, trees...) need *fixed-size inputs*.
- A classification model splits into two steps:
  - an *encoder* builds a fixed-size (latent) representation of the text;
  - a *classification head* computes class probabilities from it.
- n-gram models follow this pattern too: keep the last $n-1$ words, then look up a count table.

---

#demo("encoder_head", "3.1", height: 9.0cm)

== Bag of words

- Example review: _"Great acting, great lighting, and a flabbergasting soundtrack, but not even Zendaya could save this boring, boring plot."_
- Count word occurrences: _great_ 2, _boring_ 2, _acting_ 1...
- *Choose a vocabulary of $V$ words* (e.g. the most frequent ones): each review becomes a vector of size $V$ whose entry $v$ is the count $C_v$.
- A *sparse vector*: a median IMDB review has \~178 words for thousands of vocabulary entries.
- Words outside the vocabulary (_Zendaya_, _flabbergasting_) are ignored.
- *Word order is completely lost.*

---

=== Logistic regression head

- One weight $w_v$ per vocabulary word:

$ p("positive" | x_1 dots x_L) = sigma(b + sum_(v=1)^V w_v C_v), quad sigma(z) = 1/(1 + e^(-z)) $

- Each occurrence of a word pushes the score up or down by its weight.

---

- Weights are learned from labeled reviews by minimizing the cross-entropy loss (two classes).
- With 10,000 weights and 20,000 reviews, *regularization* is needed to avoid overfitting.

---

#demo("bag_of_words", "3.2", height: 7.3cm)

---

=== Results

- Exercise 3.2: *87.6% test accuracy*, while being completely blind to word order!
- Weights are easy to interpret: the model extracts its own list of positive and negative words.
- Top positive "words": _7_ and _8_... from reviews ending with "7/10" or "8/10". A shortcut!
- Pang et al. (2002): humans picking words by hand reached 58–64%, learned classifiers up to 82.9%.
- *Learning word weights from data beats hand-picked word lists.*

== TF-IDF weighting

- Words like _the_, _a_, _and_ appear in almost every review (97–99%), unlike _boring_ (6%).
- *Inverse document frequency (IDF)* downweights frequent words:

$ "idf"(v) = -log N_v / N = log N / N_v $

- $N$: number of documents; $N_v$: number of documents containing $v$.
- Each entry becomes $C_v times "idf"(v)$: *term frequency × inverse document frequency*.
- With $log_2$: _the_ $->$ 0.01, _boring_ $->$ 4, _flabbergasting_ $->$ 13.6.

---

- Idea due to Karen Sparck Jones (1972) for document retrieval. Still used by search engines.
- Exercise 3.3 (counts vs. binary presence vs. TF-IDF): TF-IDF wins, but by less than 1 point.
  - Logistic regression already learns one weight per word, so a fixed per-word factor changes little.
- *Whatever the weighting, our bag of words stays below 89%.*
- Wang & Manning (2012) reached 91.2% by adding word pairs (bigrams) like "not good".

== Limits of bag of words

- _"Excellent movie, awesome soundtrack."_ vs. _"Superb film, wonderful score."_
  - Same meaning, but *orthogonal* vectors: no shared non-zero entry.
  - Rare words get weights learned from a handful of examples, or no weight at all.
- *Word order is lost*:
  - _"I expected it to be terrible, but it was great."_
  - _"I expected it to be great, but it was terrible."_
  - Identical vectors, hence identical predictions!
- Negation (_"not bad at all"_) and sarcasm also fool the model.

---

=== Hard reviews

- A small set of short reviews, grouped by challenge: negation, contrast, sarcasm, word order, synonyms, rare words.

---

- Exercise 3.4 (bag of words):
  - sarcasm about sleeping in the theater gets 76% positive (_perfect_, _masterpiece_);
  - _"Not bad at all, actually."_ only gets 34% positive;
  - synonyms are handled fine: each word appears in hundreds of labeled reviews.
- *The bag of words knows which words are positive, but not what the review actually says.*

= Dense word embeddings

== The embedding matrix

- Represent each word by a *dense vector* of real numbers in a space of dimension $D << V$ (e.g. $D = 300$): its *embedding*.
- Hinton (1986) called them _distributed representations_.
- All embeddings are stored in an *embedding matrix* with one column per word:

$ W in cal(M)_(D times V)(RR) $

- Looking up an embedding = multiplying $W$ by the word's one-hot vector.

---

#demo("embedding_matrix", "4.1", height: 9.0cm)

---

=== Cosine similarity

- Dense embeddings can be compared directly using *cosine similarity*:

$ cos(u, v) = (u dot v) / (norm(u) norm(v)) $

- 1 for same direction, 0 for orthogonal, $-1$ for opposite directions.
- Different words always have similarity 0 in bag-of-words space.

---

- *Good embeddings: geometric similarity reflects semantic similarity.* What a model learns about _excellent_ partly transfers to _superb_.
- Another route to dense vectors: _Latent Semantic Analysis_ (truncated SVD of a word-document count matrix).

---

=== Aggregating word vectors

- Replacing $L$ words by $L$ vectors of size $D$ still doesn't give a fixed-size representation!
- An *aggregation (pooling)* step is needed:
  - *average* of the $L$ vectors $->$ size $D$;
  - *weighted average* to favor important words $->$ size $D$;
  - *concatenation* of the last $n - 1$ vectors $->$ size $(n-1) times D$.
- Concatenation keeps the order of the last words but forgets everything before: a good fit for language modeling.

---

#demo("embedding_aggregation", "4.1", height: 9.0cm)

== Concatenation → Bengio's neural language model

- Keep the Markov assumption, but replace the count table with *a neural network reading the concatenated embeddings* of the $n - 1$ previous words.
- Bengio et al. (2003): *the first neural language model*.
  - Look up embeddings $e_(L-n+2), dots, e_L$, concatenate them into a context vector $c$, feed $c$ to a one-hidden-layer network:

$ q(x_(L+1) = v | x_1 dots x_L) = "softmax"(U tanh(A c + b) + d)_v $

---

- $A, b$: hidden layer; $U, d$: output layer computing one score (*logit*) $z_v$ per word.
- Softmax turns the $V$ logits into a probability distribution:

$ "softmax"(z)_v = e^(z_v) / (sum_w e^(z_w)) $

---

- *Word embeddings and network weights are jointly learned* by minimizing the cross-entropy loss with backpropagation.
- Similar words converge to close embeddings because they appear in similar contexts.
- *The model generalizes from seen contexts to new, unseen ones*: a remedy to the curse of dimensionality.

---

=== Results

- Exercise 4.2 (3 context words, $D = 100$, 200 hidden units): *test perplexity 208.5*.
  - Better than the interpolated 5-gram (277.7) and even than Kneser-Ney 5-gram (219.3)!

---

- Nearest neighbors: _three_ $->$ numbers, _he_ $->$ _she_, _film_ $->$ _book_, _novel_, _album_.
- Neighbors of _excellent_ and _terrible_ are noise: these words are rare on Wikipedia.
- Almost all training time goes to the output layer: $200 times 33,277$ multiplications per word.

== Aggregating the words of a text

- For classification, sentiment can be anywhere: concatenating the last words is useless.
- Simplest option: *average* the embeddings. With the bag-of-words count vector $C$:

$ 1/L sum_(i=1)^L e_i = 1/L W C = W f $

---

- $f = C \/ L$: relative frequencies of the words in the review.
- *Averaging embeddings = a bag of words linearly compressed by $W$ from $V$ to $D$ dimensions.* Word order is still lost.

---

- With a logistic regression head of weights $h$:

$ h dot W f = sum_(v=1)^V (h dot w_v) f_v $

- Just a logistic regression on word frequencies, with weight $h dot w_v$ for word $v$: no gain if $W$ is learned from scratch!

---

- Two ways to improve:
  - add *non-linear hidden layers* after averaging: the *Deep Averaging Network (DAN)*;
  - start from a *good, fixed embedding matrix*: synonyms get similar weights, even unseen ones (section 5).

---

=== Results

- Exercise 4.3 (DAN trained from scratch): *88.1%*, no better than bag of words (87.6–88.4%).
- Same mistakes on hard reviews, but more confident: _"Not bad at all, actually."_ gets 0.000 instead of 0.335.
- *Averaging embeddings learned from scratch is a bag of words in disguise.*
- fastText (2017): average of word (and bigram) embeddings + linear classifier. Trains on a billion words in minutes on a CPU.

== Embeddings are still word representations

- Exercise 4.4: compare the nearest neighbors in the DAN and in the language model.
  - The DAN only learns sentiment: _excellent_ $->$ _wonderfully_, _terrible_ $->$ _awful_, but _three_, _he_, _film_ are noise.
  - The language model does the opposite.
- *Task-specific embeddings only capture what the task needs, within the reach of its data.*
- Word meaning doesn't depend on the task! *What if we could learn embeddings once, on a task with abundant free data, and reuse them everywhere?*

= Pre-training embeddings: word2vec and GloVe

== Two-stage training

- *Pre-training*: learn the embedding matrix $W$ on a task with abundant, free data (raw text: billions of words).
- *Downstream task*: copy $W$ into the model of the task we care about and train it on its (expensive) labeled data.
- We need a pretext task whose loss can be computed from raw text only.
- *Distributional hypothesis*: words that appear in similar contexts tend to have similar meanings.
  - _"You shall know a word by the company it keeps!"_ (Firth, 1957)

== word2vec

- Two architectures by Mikolov et al. (2013), using a window around a middle word:
  - *CBOW* (_Continuous Bag of Words_): predict the middle word from the average of the surrounding embeddings;
  - *Skip-gram*: predict each surrounding word from the middle word's embedding.
- *No non-linear hidden layer at all*: going from embeddings to word scores costs a single matrix–vector product.

---

#demo("word2vec_models", "5.2", height: 9.0cm)

---

=== Skip-gram

- Two matrices: $W$ ($D times V$) gives a center vector $v_w$ per word; $U$ ($V times D$) gives a context vector $u_c$ per output word.

$ q(c | w) = e^(u_c dot v_w) / (sum_(c'=1)^V e^(u_(c') dot v_w)) $

- Trained with the cross-entropy loss on all (center, context) pairs of the text.
- Problem: the denominator still sums over all $V$ words!

---

=== Negative sampling

- Turn the question into a binary one: *is $(w, c)$ a real pair from the text, or a fake one?*
- For each real pair, draw $k$ noise words $n_1 dots n_k$ and train a logistic regression with dot products as logits:

$ cal(L) = -log sigma(u_c dot v_w) - sum_(i=1)^k log sigma(-u_(n_i) dot v_w) $

---

- With $k = 5$: 6 dot products per pair instead of 33,277.
- Noise words are drawn from counts raised to the power $3\/4$.
- *Subsampling*: frequent words like _the_ are randomly discarded with probability $1 - sqrt(t \/ f)$.

---

=== Results

- Exercise 5.2: word2vec trains on WikiText-2 in \~15 seconds vs. \~6 minutes for Bengio's model: \~30× faster per pair.
- Similar neighbors: numbers with numbers, _he_ with _him_ and _she_.
- *word2vec is light and fast enough to learn from billions of words.*

== GloVe

- Pennington, Socher & Manning (2014) went back to counting.
- Build a $V times V$ *co-occurrence matrix* $X$ (counts of word $j$ in the window of word $i$), then fit:

$ w_i dot tilde(w)_j + b_i + tilde(b)_j approx log X_(i j) $

- Weighted least squares: noisy rare pairs get less weight, and the weight is capped for very frequent pairs.
- *GloVe compresses a huge count table into dense vectors of a few hundred dimensions.*

---

=== Predicting vs. counting

- Levy & Goldberg (2014): at its optimum, skip-gram with $k$ negative samples satisfies:

$ v_w dot u_c = log (p(w, c)) / (p(w) thin p(c)) - log k $

- The log term is the *Pointwise Mutual Information (PMI)*: high for _ice_/_cold_, \~0 for _ice_/_the_.

---

- word2vec implicitly factorizes a co-occurrence statistics matrix, like GloVe and LSA.
- *Predicting local contexts and counting global ones lead to nearly the same mathematical problem.*
- Pre-trained GloVe vectors used next: 6 billion words, 400,000 words, 300 dimensions.

== What do embeddings capture?

- Semantic similarity (nearest neighbors)... and a surprising geometric structure!
- $v_"paris" - v_"france" + v_"spain"$ is closest to _madrid_: a direction means "capital of".
- $v_"king" - v_"man" + v_"woman"$ is closest to _queen_ (once _king_ itself is excluded).
- Analogies "$a$ is to $b$ as $c$ is to ?": find the word closest to $v_b - v_a + v_c$.
- Exercise 5.4: GloVe answers \~3 analogy questions out of 4, without being trained for it.

---

#demo("analogy_directions", "5.4", height: 7.3cm)

---

=== Two caveats

- *Analogy search must exclude the words of the question*, otherwise it often returns one of them.
  - _man : doctor :: woman : ?_ returns _physician_ with the exclusion rule, _doctor_ without it.
  - Keep this in mind when reading famous claims about biased analogies.
- *Meaning lies in directions, not in individual coordinates.*
  - word2vec and GloVe losses only depend on dot products, which are invariant under rotation of the whole space.

== Transfer learning

- Initialize the DAN of section 4.3 with GloVe vectors: knowledge learned on one task is reused on another.
- Two strategies:
  - *freeze* the embeddings: keep $W$ fixed, train only the hidden layers and the head;
  - *fine-tune* the embeddings: start from GloVe, but keep updating $W$ with the rest of the network.
- Frozen embeddings keep everything learned from 6 billion words, including words absent from the labeled reviews.

---

#demo("transfer", "5.5", height: 9.0cm)

---

=== Results

- Exercise 5.5a (20,000 labeled reviews): fine-tuned GloVe +0.5 point over scratch, frozen GloVe \~4 points *worse*.
  - Iyyer et al. found the same: large gains on short sentences (SST: 86.3% vs. 83.2%), tiny ones on IMDB (89.4% vs. 88.8%).

---

- Exercise 5.5b (100 labeled reviews): frozen GloVe 75.2% vs. 69.2% from scratch.
  - 100 random reviews only contain \~3,686 of the 10,000 vocabulary words.
- *Pre-training is most valuable when labeled data is scarce.*
- Still, only the first layer is transferred, averaging ignores word order, and each word has a single vector.

= Interlude: where do we stand?

== Scoreboard

#figure(
  text(size: 16pt)[
    #table(
      columns: (auto, auto, 1fr, auto, auto, auto),
      align: left,
      table.header(
        [*Model*], [*Sec.*], [*Text → fixed-size vector*], [*Reused knowledge*], [*WikiText-2 ppl.*], [*IMDB acc.*]
      ),
      [Interpolated 5-gram], [2], [keep the last 4 words], [none], [277.7], [–],
      [Bag of words, TF-IDF], [3], [word counts], [none], [–], [88.4%],
      [Bengio's language model], [4.2], [concatenate the last 3 embeddings], [none], [208.5], [–],
      [Deep averaging network], [4.3], [average the embeddings], [none], [–], [88.1%],
      [DAN + GloVe], [5.5], [average the embeddings], [embeddings (6B words)], [–], [88.1%],
    )
  ],
)

---

- *Language modeling*: steady progress, from \~1,000 (unigram) to \~200 perplexity.
  - But *our language models see at most four words into the past.*
- *Classification*: stuck between 87.6% and 88.4%. Nothing beats the plain bag of words.
  - *On IMDB, word pairs brought more than embeddings* (91.2% for Wang & Manning).
- Hard reviews: no model gets more than 4 out of 10 negation, contrast, sarcasm and word-order reviews right.

== Polysemy

- Each word has a single, fixed vector: _score_ of a film = _score_ of a football match.
- Exercise 6 (GloVe): *a static embedding mixes all the senses of a word, and the most frequent one dominates.*
- Telling a river _bank_ from a money _bank_ requires looking at the surrounding words: *contextual embeddings* (sections 8.2 and 10.2).

---

=== Back to the two questions

- *Variable-length input?* So far, either keep the last few words in order (good for language modeling), or keep all words but lose their order (good for classification). *No model reads the whole text in order*, like humans do.
- *Transfer?* Only the embedding matrix is transferred. It helps when labels are scarce, not with the full labeled set.
  - The part above the embeddings still learns from scratch, and each word still has a fixed vector.

= Recurrent networks

== Reading text in order

- To separate _"...terrible, but it was great"_ from _"...great, but it was terrible"_, the encoder must aggregate embeddings *in an order-dependent way*.
- *Convolution*: slide a window over the text, compute features for each window, then keep each feature's maximum over all windows (*max pooling*).
  - Sees order inside each window, not beyond it.
- *Recurrent network*: read words one by one and update a _state_ vector after each word. The final state represents the whole text.

---

#demo("order_encoders", "7", height: 9.0cm)

== Recurrent neural network (RNN)

- An RNN reads the embeddings $x_1 dots x_L$ one at a time and updates a *hidden state* $h_t$ of size $H$:

$ h_t = tanh(W_h h_(t-1) + W_x x_t + b) $

- Starts from $h_0 = 0$. $W_h$, $W_x$ and $b$ are shared across time steps: a fixed number of parameters for any length.
- $h_t$ depends on all words read so far *and on their order*.

---

- *The final state $h_L$ is the fixed-size representation of the whole text*, read by a head:

$ p("positive" | x_1 dots x_L) = sigma(w dot h_L + c) $

- Each $h_t$ is also a vector per word: tagging can apply a classifier to each $h_t$.
  - A second RNN reading right-to-left gives a *bidirectional RNN*.
- Keep the sequence $h_1 dots h_L$ in mind: attention will read all of it (section 9.2).

== Training and vanishing gradients

- Training *unrolls* the RNN in time: a deep feedforward network with $L$ layers sharing the same weights.
- *Backpropagation through time (BPTT)*. By the chain rule:

$
  (partial h_L) / (partial h_t) = (partial h_L) / (partial h_(L-1)) dots.c (partial h_(t+1)) / (partial h_t), quad (partial h_s) / (partial h_(s-1)) = "diag"(1 - h_s^2) thin W_h
$

- A long product of matrices shrinks or grows exponentially with $L - t$: *vanishing* or *exploding gradients*.
- Words at the start of a long review barely receive any gradient.

---

- Bengio, Simard & Frasconi (1994): storing information robustly *requires* operating where gradients vanish. A fundamental trade-off!
- Exploding gradients: *gradient clipping* rescales the gradient when its norm exceeds a threshold.
  - PyTorch: `nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- Vanishing gradients require changing the architecture.

---

=== LSTM

- *Long short-term memory* adds a *cell state* $c_t$, updated by addition rather than matrix multiplication:

$ c_t = f_t dot.o c_(t-1) + i_t dot.o g_t, quad h_t = o_t dot.o tanh(c_t) $

- $g_t = tanh(W_g [h_(t-1), x_t] + b_g)$: candidate update.

---

- Three *gates* with values in $[0, 1]$, e.g. $f_t = sigma(W_f [h_(t-1), x_t] + b_f)$:
  - *forget* gate $f_t$, *input* gate $i_t$, *output* gate $o_t$.
- *When the forget gate is close to 1, the gradient flows back through many steps almost unchanged.*
- *GRU*: a lighter variant with two gates and no separate cell.

---

=== Results

- Exercise 7.2a: gradient magnitude is divided by \~2.2 per step back for the RNN, \~1.6 for the LSTM. *Forget gates control how far back the gradient flows.*
- Exercise 7.2b (LSTM on IMDB, last 400 words kept):
  - first classifier to separate the two "same words" reviews: 0.63 vs. 0.11;
  - *but trained on 20,000 labeled reviews only, it doesn't beat the bag of words.*
- Dai & Le's remedy: pre-train the LSTM on raw text first (section 8).

---

#demo("lstm_reading", "7.2", height: 7.3cm)

== RNNs as language models and generators

- A head reads $h_t$ and outputs the next-word distribution:

$ q(x_(t+1) | x_1 dots x_t) = "softmax"(U h_t + d) $

- *The context is no longer limited to the last $n - 1$ words.*
- Mikolov et al. (2010): RNN ensembles nearly halved the perplexity of Kneser-Ney 5-grams on news data.
- Training on 2 million words: *truncated BPTT* with chunks of 35 words; the hidden state carries over, gradients stop at chunk boundaries.

---

- Exercise 7.3: after a single epoch, the LSTM beats every model of sections 2 and 4 (validation perplexity 204.9 vs. 223.9 for Bengio's model).
- Best 2016 LSTMs on WikiText-2: 2 layers × 650 units, *dropout*, *variational dropout* $->$ test perplexity 96.3, and 80.8 with a copy mechanism.
- RNNs made *character-level generation* famous: Karpathy's 2015 blog post on Shakespeare, Wikipedia, LaTeX and Linux code.

---

#demo("char_samples", "7.3", height: 8.2cm)

---

=== Words, characters or subwords?

- *Words*: fixed vocabulary, out-of-vocabulary words become `<unk>`, _film_ and _films_ are unrelated.
- *Characters*: no unknown words (65 symbols in TinyShakespeare), but \~5× longer sequences and spelling must be learned.
- Perplexities on different units can't be compared directly: 2.0 bits/char $approx$ 11 bits/word $approx$ perplexity 2,000 per word!
- *Subwords*: frequent words stay whole, rare words are split. _Byte-Pair Encoding (BPE)_ merges the most frequent pairs of symbols.
  - This is how modern *tokenizers* work.

= Pre-training the whole encoder

== From embeddings to the whole encoder

- The LSTM can read in order, but it learns everything from 20,000 labeled reviews.
- A language model *is* an LSTM trained on raw text, with the same embeddings + LSTM structure as the classifier's encoder.
- *Pre-train the whole encoder as a language model, copy it into the classifier, then fine-tune on labeled reviews.*
  - Only the head is learned from scratch; the language model's output layer is discarded.

---

#demo("pretrain_encoder", "8", height: 9.0cm)

== Pre-train, then fine-tune

- Dai & Le (2015): LSTM with 1,024 units pre-trained as a language model.
  - IMDB: 86.5% from scratch $->$ *92.4%* pre-trained (90.0% with word2vec embeddings only).
  - _Sequence autoencoder_ pre-training: 92.8%. Pre-trained LSTMs are also more stable to train.

---

- Exercise 8.1 (128 units, 50,000 unlabeled IMDB reviews): *a small language model only transfers its embeddings, not its LSTM.*
- Scale matters: Radford et al. (2017) trained a 4,096-unit character LSTM on 82 million Amazon reviews $->$ 92.9% with a logistic regression on its final state.
  - *A single "sentiment neuron" alone reaches 92.3%!*

---

=== ULMFiT

- Howard & Ruder (2018): _Universal Language Model Fine-tuning_, built on AWD-LSTM.
  1. Pre-train a language model on a large general corpus (WikiText-103).
  2. Fine-tune it as a language model on the target corpus (IMDB reviews).
  3. Add a head and fine-tune the whole classifier on labeled reviews.

---

- Avoid *catastrophic forgetting*: *gradual unfreezing* from the top layer down, and smaller learning rates for lower layers.
- *95.4%* on IMDB with forward + backward models (94.7% with one direction).

== Contextual embeddings

- The LSTM state $h_t$ depends on $x_t$ and on all previous words: a vector for $x_t$ that *changes with the sentence*.
- These *contextual embeddings* solve the polysemy problem: each occurrence of _bank_ gets its own vector.

---

#demo("contextual_vectors", "8.2", height: 9.0cm)

---

=== ELMo

- _Embeddings from Language Models_ (Peters et al., 2018):
  - two LSTM language models (left-to-right and right-to-left), 2 layers × 4,096 units, \~800 million words of news;
  - character-level convolutional input: no out-of-vocabulary words;
  - final vector = task-specific weighted sum of all layers.

---

- Language models stay *frozen*: ELMo vectors are concatenated to static embeddings in existing models.
- New state of the art on six NLP tasks.
- Two schools, both inherited by Transformers: *feature extraction* (ELMo) vs. *fine-tuning* (ULMFiT).

= Sequence-to-sequence and attention

== Encoder–decoder (seq2seq)

- Until 2014, the best translation systems were *statistical machine translation (SMT)* systems built from many components.
- Combine two RNNs:
  - the *encoder* reads the source sentence; its final state $h_L$ is a fixed-size representation;
  - the *decoder* is a language model whose initial state is $h_L$:

$ p(y_(N+1) | y_1 dots y_N ; h_L) $

---

- Training uses *teacher forcing* (read the ground-truth previous words). At test time, the decoder reads its own outputs, e.g. with *greedy decoding*.

---

#demo("seq2seq", "9.1", height: 8.5cm)

---

=== Sequence to sequence

- Cho et al. (June 2014) introduced the encoder–decoder (and the GRU) to score phrase pairs inside an SMT system.
- Sutskever, Vinyals & Le (Sept. 2014): 4-layer LSTMs, 1,000 units, 12 million sentence pairs.
  - Ensemble of 5 models: BLEU 34.8 vs. 33.3 for the SMT baseline. First pure neural system to beat phrase-based SMT on a large task.

---

- *Bottleneck*: all information about the source must fit in $h_L$, whatever its length. Quality drops quickly on long sentences.
- Exercise 9.1 (reversing digit sequences): perfect on length 10, degrades as sequences grow.

== Attention

- Bahdanau, Cho & Bengio (2014): *keep all the encoder states*:

$ x_1 dots x_L --> h_1 dots h_L $

$ p(y_(N+1) | y_1 dots y_N ; h_1 dots h_L) $

- For each target word, compute a *context vector*: a weighted average of the encoder states.

$ c_N = sum_(j=1)^L alpha_(N,j) h_j $

---

- The weights come from a softmax over compatibility scores $e_(N,j)$ between the decoder state $s_N$ and each $h_j$:

$ alpha_(N,j) = exp(e_(N,j)) / (sum_(k=1)^L exp(e_(N,k))) $

- Simplest score: the dot product $e_(N,j) = s_N dot h_j$ (Luong, Pham & Manning, 2015).

---

- The head reads both $s_N$ and $c_N$. Weights are learned end-to-end, without alignment supervision.
- *With attention, each target word can directly read the states of all source words.*

---

#demo("seq2seq_attention", "9.2", height: 9.0cm)

---

=== Results

- Same size, same data: encoder–decoder BLEU 17.8 vs. *26.8* with attention (28.5 with longer training).
- Attention shows no degradation on sentences of 50+ words.
- Attention weights form a learned *word alignment* matrix (e.g. _zone économique européenne_ vs. _European Economic Area_).
- Exercise 9.2 (sequences of 30 digits): 82% fully reversed instead of 0%, 95% correct digits instead of 54%.

== Measuring translation quality

- *BLEU* (Papineni et al., 2002): compare a candidate with human reference translations by counting shared n-grams ($n = 1$ to 4).
- Reference: _"I loved this great movie"_; candidate: _"I loved this great film"_.

#figure(
  text(size: 18pt)[
    #table(
      columns: 4,
      align: left,
      table.header([*$n$*], [*Candidate n-grams*], [*In reference*], [*Precision $p_n$*]),
      [1], [I, loved, this, great, film], [4 of 5], [0.80],
      [2], [I loved, loved this, this great, great film], [3 of 4], [0.75],
      [3], [I loved this, loved this great, this great film], [2 of 3], [0.67],
      [4], [I loved this great, loved this great film], [1 of 2], [0.50],
    )
  ],
)

---

- Geometric mean of precisions: $(0.80 times 0.75 times 0.67 times 0.50)^(1\/4) = 0.67$, i.e. *66.9*.
- Counts are *clipped* to the reference counts, and short candidates get a *brevity penalty*:

$ "BLEU" = "BP" dot (p_1 p_2 p_3 p_4)^(1\/4), quad "BP" = min(1, e^(1 - r \/ c)) $

---

- $c$: candidate length; $r$: reference length.
- Limits: surface matching only (_film_ vs. _movie_ costs a third of the score). Reliable at corpus level (0.99 correlation with human judgments), not per sentence.

---

=== Statistical vs. neural translation

- IBM models (Brown et al.): find $e$ maximizing $Pr(e) Pr(f | e)$, i.e. a language model × a translation model learned through word *alignment*.
- *Phrase-based SMT* (Koehn, Och & Marcu, 2003): phrase tables + n-gram language model + reordering model. *Moses* became the industry standard.

---

- WMT'14 English → French: Moses 33.3, best system 37.0, Sutskever et al. 34.8 (36.5 reranking Moses), Bahdanau et al. 36.2 vs. 35.6 on sentences without unknown words.
- Google's GNMT (2016): 8-layer LSTMs + attention + subwords $->$ *41.2* BLEU. Replaced phrase-based Google Translate.

= Epilogue: the Transformer

== Transformers replace RNNs

- June 2017: *a Google team removed the RNNs entirely and kept only attention*. The *Transformer* is the basis of all today's LLMs.
- It was not designed for chatbots, but as a better tool for classification and translation!
- Two costs of recurrence:
  - *gradient flow*: information and gradients go through every intermediate step;
  - *sequential computation*: $h_t$ must wait for $h_(t-1)$, which prevents parallelism on GPUs.

---

=== Self-attention

- Each word computes compatibility scores (dot products after learned linear maps) with *all the words of the same text*, itself included.
- Its new vector is the softmax-weighted average of all word vectors.
- Any two words are connected in a single step, and all new vectors are computed at once with a few matrix products.
- Several such layers are stacked.
- A weighted average ignores order: *positional encodings/embeddings* are added to word embeddings at the input.

---

#demo("self_attention", "10.1", height: 8.8cm)

---

=== The original Transformer

- An encoder–decoder architecture for translation, like section 9:
  - the *encoder* is a stack of self-attention layers reading the source;
  - the *decoder* attends to the encoder vectors and applies self-attention to the translation written so far.
- Decoder self-attention uses a *mask*: each position only attends to itself and previous positions.
- Today, "Transformer" is often equated with "autoregressive LLM"... but it was born for machine translation!

== BERT

- Devlin et al. (Oct. 2018): pre-train a *Transformer encoder* on raw text, then fine-tune it (like ULMFiT).
- _Bidirectional Encoder Representations from Transformers._ The goal was a better text encoder, not a text generator.
- A standard language model is unidirectional, and a Transformer encoder would see the word it must predict.
- *Masked Language Model (MLM)*: hide 15% of tokens (80% `[MASK]`, 10% random, 10% unchanged) and predict them from both sides.
  - Inspired by the Cloze test (1953). It is CBOW with a whole-text context and a deep Transformer!

---

- Pre-trained on 3.3 billion words (BooksCorpus + English Wikipedia), 30,000 _WordPiece_ subwords.
  - *BERT base*: 12 layers, 768 dims, 12 heads, 110M parameters.
  - *BERT large*: 24 layers, 1,024 dims, 16 heads, 340M parameters.

---

- *Fixed-size representation?* A special `[CLS]` token is prepended and processed like any other token:

$ x_"CLS" x_1 dots x_L --> h_"CLS" h_1 dots h_L $

- A second pre-training task (*Next Sentence Prediction*) teaches $h_"CLS"$ to summarize the input.
- *The `[CLS]` vector is the fixed-size representation of the text, read by the classification head.*

---

#demo("bert", "10.2", height: 9.0cm)

---

=== Results

- Fine-tuning: add a head on $h_"CLS"$, train 2–4 epochs with a small learning rate (2e-5 to 5e-5).
- New state of the art on 11 NLP tasks; GLUE average from 72.8 (GPT) to *80.5* (BERT large).
- Also usable frozen, like ELMo: only 0.3 F1 below fine-tuning on NER.

---

- Exercise 10.2a: *the vector of a word depends on all the other words*, so _bank_ gets different vectors for different senses.
- Exercise 10.2b: *89.2%* with only 1,000 labeled reviews (LSTMs: 65–70%, DAN: 82.5%).
- Full IMDB: 94.6% (BERT base), 95.6% with further MLM pre-training on IMDB.

== From GPT-1 to GPT-3, then InstructGPT

- June 2018: *GPT-1* uses the other half, the *Transformer decoder*.
  - Thanks to the mask, it is a language model trained on raw text, with all positions computed in parallel.
  - 12 layers, 768 dims, 12 heads, 512-token context, trained on BooksCorpus.

---

- Fine-tuned with a linear head on an *extract token* appended at the end (the only token that sees the whole text):

$ x_1 dots x_L x_"ext" --> h_1 dots h_L h_"ext" $

- State of the art on 9 of 12 tasks. Even without fine-tuning, the raw language model could solve some tasks (e.g. appending _"very"_ and comparing _positive_/_negative_).

---

=== GPT-2

- February 2019: same architecture, scaled up. 48 layers, *1.5 billion parameters*, 1,024-token context.
- Trained on WebText: 40 GB from 8 million web pages linked from Reddit.

---

- *No fine-tuning at all*: _"language models begin to learn these tasks without any explicit supervision"_.
  - Tasks written as text: `TL;DR:` for summarization, `english sentence = french sentence` for translation.
- Zero-shot WikiText-2 perplexity: *29.41* (smallest) and *18.34* (largest), vs. 39.14 for the previous best.

---

=== GPT-3 and InstructGPT

- May 2020: *GPT-3*, GPT-2 scaled ×100: *175 billion parameters*, 96 layers, 2,048-token context, 300 billion tokens.
- *In-context learning*: the task is described in the input text (the *prompt*), with a few examples.
- End of the transfer story: sections 2–4 transfer nothing, section 5 the embeddings, section 8 the encoder, BERT/GPT-1 everything but the head... and GPT-3 not even a head.

---

- *The task is written in the prompt, and the pre-trained language model does everything.*
- InstructGPT (2022): SFT + reward model + RLHF. Outputs of a 1.3B model preferred to those of 175B GPT-3: *post-training* turns a language model into an assistant.

== Key takeaways

- *Variable-length input*: last words (n-grams, Bengio) $->$ orderless aggregation (bag of words, averaging) $->$ ordered reading (RNNs) $->$ attention over all states $->$ self-attention (Transformer).
- *Transfer*: nothing $->$ embeddings (word2vec, GloVe) $->$ whole encoder (ULMFiT, ELMo) $->$ everything but the head (BERT, GPT-1) $->$ nothing to retrain (GPT-3 prompting).
- Training objective throughout: *cross-entropy = negative log-likelihood = KL divergence* up to a constant.
- *Pre-training shines when labeled data is scarce, and scale makes it work.*

