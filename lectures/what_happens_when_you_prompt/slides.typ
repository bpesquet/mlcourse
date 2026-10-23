#import "@preview/touying:0.6.1": *

#let main_title = [What happens when you prompt]
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

#components.adaptive-columns(outline(title: none, depth: 3, indent: 1em))

= The big picture

== Key takeaways

- An LLM service is a _networked software system_ built around one or more _generative models_.
- Your prompt is a small part of the exchange.
- The _harness_ orchestrates the interaction: it prepares and updates the context, manages tools and controls the response flow.
- The _model_ processes a finite context and generates outputs autoregressively.
- The model does not inherently retain interaction history between turns. Conversational continuity is provided by the surrounding system.

== Information flow

#align(center, image("images/prompt_information_flow.svg", width: 68%))

= Step-by-step process

== From input to client request

- Typing is normally local only.
- The client-side software (web browser, desktop or mobile app) sends your prompt as a web request when you submit it, augmenting it with info such as:
  - conversation identifier;
  - selected model;
  - references to attached files, if any;
  - relevant metadata (time zone, etc.).
- The user interface is updated to indicate that processing has started.

== Network exchange

- The request travels from the client to the provider's servers over an encrypted HTTPS (TLS) connection.
- The response is not a single reply but a long-lived stream, typically using Server-Sent Events (SSE).
- Through this open stream, the server sends back response fragments as soon as they are generated.

== Server intake

Upon request arrival, several server-side operations may take place:

- User authentication and authorization.
- Usage and rate limits checkups.
- Conversation storage.
- Safety screenings.
- Request routing to an appropriate processing service.

== Harness

Orchestration layer coordinating the model
and the surrounding software services. Implemented by potentially distributed components. Roles:

- Assemble and manage the model's input context.
- Process files and prepare multimodal inputs.
- Invoke the model and interpret its structured output.
- Run tools and integrate their results.
- Apply validation, safety and permission checks throughout.
- Handle response delivery.

== Context assembly

The harness constructs a model input from relevant
instructions, conversation state and available data, typically including:
- The system prompt.
- Session metadata: current date, time zone, approximate location.
- Relevant saved preferences.
- Definitions and usage guidance for every tool the model may use (web search, code execution, etc.).
- Extracted text, visual representations or other file content.
- Your current prompt and selected conversation history.

---

=== The system prompt

Set of instructions supplied to the model by the service.

Typically contains:
- Model identity and role.
- Behavioral guidelines and boundaries.
- Tool usage instructions and output format requirements.

Examples: #link("https://raw.githubusercontent.com/x1xhlol/system-prompts-and-models-of-ai-tools/refs/heads/main/Anthropic/Claude%20Sonnet%205.txt")[Claude Sonnet 5], #link("https://raw.githubusercontent.com/x1xhlol/system-prompts-and-models-of-ai-tools/refs/heads/main/Perplexity/Prompt.txt")[Perplexity]

---

=== The context window

- Model's _working memory_: maximum amount of text (measured in tokens) that it can take into account at once.
- Larger context window  ability to process longer inputs.
- Longer inputs $==>$ #sym.arrow.tr cost and #sym.arrow.br speed.

#figure(
  table(
    columns: (1fr, 1fr),
    align: left,
    table.header([*Context window*], [*Approx. English words*]),
    [4,000 tokens], [3,000 words],
    [128,000 tokens], [96,000 words],
    [1 million tokens], [750,000 words],
  ),
)

== Model

=== Input representations

- A _tokenizer_ splits the raw textual input into frequent sub-word pieces called _tokens_. A common word is one, a rare word is several.
- Each token is mapped to an integer ID.
- Images, audio, and other modalities require additional preprocessing to be encoded into representations appropriate to the model.

---

=== Multimodality

- The ability to process non-textual formats may be provided by:
  - the model, using modality-specific encoders or representations;
  - external tools orchestrated by the harness;
  - both.
- Images may be represented by patches, visual tokens or encoded features.
- Audio may use audio tokens, encoded features or transcription.
- Video may be represented as sequences of frames and associated temporal information.

---

=== Embeddings

- Each token ID is looked up in a learned embedding table and becomes a vector representing the token in a high-dimensional space.
- Conceptually, a text-only input becomes a matrix $X in RR^(n times d)$, with $n$ the number of input positions and $d$ the dimension of the hidden representation space.
- Positional information is incorporated according to the model architecture, either through embeddings or within subsequent computations.

---

=== Inference

- The model processes the input representations through multiple neural network layers. These computations typically combine:
  - _Self-attention_, which allows representations to incorporate information from other positions in the input.
  - _ Feed-forward_ (MLP) blocks, which transform representations at each position.
  - Residual connections and normalization operations.
- Across layers, the representations are progressively
  transformed into features used to predict the next token.

---

=== Sampling and decoding

- The model produces $RR$-valued scores (_logits_) for possible next tokens.
- A probability distribution is obtained from these scores, typically using softmax.
- Conceptually, the model estimates $P(y_t | x, y_1, ..., y_(t-1))$, with $x$ the supplied context and $y_i$ the tokens generated so far.
- A decoding strategy then picks the next token. Examples include _greedy decoding_ or _temperature sampling_.
- The same prompt does not necessarily produce identical output on every run!

---

=== Autoregressive generation

- Each new token becomes part of the context used to predict subsequent tokens.
- Generation continues until a limit or stopping condition is reached.

#figure(
  image("images/autoregressive_generation.gif", width: 56%),
  caption: [Image credits: #link("https://www.youtube.com/watch?v=0VLAoVGf_74")[Welch Labs].],
)

---

=== Infrastructure and optimizations

- Model-serving systems use GPUs or other specialized accelerators to perform operations.
- The workload may be distributed across multiple devices.
- Many other optimizations take place (prefill, caching, request batching, etc.).

== Tools and context update

- Tools extend the model beyond producing text: searching the web, fetching a page, running code, creating files, etc.
- Model output may contain user-facing text, structured data, control flow signals or tool calls.
- The harness determines whether to continue generation, execute a tool and update the context, return a response or stop processing.
- A complex task may cycle through this loop many times.
- Tools may run on the provider's servers, on the client machine or on external services.

== Response streaming and rendering

- As tokens are generated, they are decoded back into text fragments and sent down the open stream as events.
- The client software interprets the received data and renders it.
- Completion may result from an end-of-turn signal, a length limit, cancellation or an error.
- Each subsequent submission starts a fresh cycle.
