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

// ---------- Palette (three zones) ----------
#let zones = (
  device: (
    fill: rgb("#FAC775"),
    stroke: rgb("#633806"),
    title: rgb("#412402"),
    sub: rgb("#633806"),
    label: "Your device and network",
  ),
  server: (
    fill: rgb("#CECBF6"),
    stroke: rgb("#3C3489"),
    title: rgb("#26215C"),
    sub: rgb("#3C3489"),
    label: "Service servers",
  ),
  model: (
    fill: rgb("#9FE1CB"),
    stroke: rgb("#085041"),
    title: rgb("#04342C"),
    sub: rgb("#085041"),
    label: "Model and tools",
  ),
)

// Small coloured tag shown under each stage heading
#let zone-tag(z) = {
  let c = zones.at(z)
  box(fill: c.fill, stroke: 0.9pt + c.stroke, radius: 2pt, inset: (x: 5pt, y: 2.5pt), text(
    size: 8pt,
    fill: c.title,
    c.label,
  ))
  v(0.3em)
}

// ---------- Diagram primitives ----------
#let u = 16cm / 680   // one unit of the original 680 × 425 schema
#let ink = rgb("#5F5E5A")

#let node(x, y, title, sub, z, w: 136, h: 56) = {
  let c = zones.at(z)
  place(top + left, dx: x * u, dy: y * u, rect(
    width: w * u,
    height: h * u,
    radius: 4pt,
    fill: c.fill,
    stroke: 0.9pt + c.stroke,
    inset: 0pt,
    align(center + horizon, stack(spacing: 4pt, text(size: 9pt, weight: "bold", fill: c.title, title), text(
      size: 7.5pt,
      fill: c.sub,
      sub,
    ))),
  ))
}

#let arrow(x1, y1, x2, y2, dash: none) = {
  let dx = x2 - x1
  let dy = y2 - y1
  let len = calc.sqrt(dx * dx + dy * dy)
  let (ux, uy) = (dx / len, dy / len)
  let (hl, hw) = (7, 3.5) // head length / half-width, in schema units
  let (bx, by) = (x2 - ux * hl, y2 - uy * hl)
  place(top + left, line(start: (x1 * u, y1 * u), end: (bx * u, by * u), stroke: (
    paint: ink,
    thickness: 0.7pt,
    dash: dash,
  )))
  place(top + left, polygon(
    fill: ink,
    stroke: none,
    (x2 * u, y2 * u),
    ((bx - uy * hw) * u, (by + ux * hw) * u),
    ((bx + uy * hw) * u, (by - ux * hw) * u),
  ))
}

#let legend-item(x, z) = {
  let c = zones.at(z)
  place(top + left, dx: x * u, dy: 395 * u, stack(
    dir: ltr,
    spacing: 4pt,
    box(width: 12 * u, height: 12 * u, fill: c.fill, stroke: 0.9pt + c.stroke, radius: 1.5pt),
    text(size: 7.5pt, fill: ink, c.label),
  ))
}

#let schema = box(width: 680 * u, height: 375 * u, place(top + left, dy: -40 * u, box(width: 680 * u, height: 420 * u, {
  // Top row: request travels left to right
  node(29, 50, "Input box", "Keys, Enter", "device")
  node(191, 50, "Client request", "Message, context", "device")
  node(353, 50, "Network", "DNS, TLS, HTTP", "device")
  node(515, 50, "Server intake", "Auth, limits", "server")
  arrow(167, 78, 189, 78)
  arrow(329, 78, 351, 78)
  arrow(491, 78, 513, 78)
  arrow(583, 108, 583, 178)

  // Bottom row: reply travels right to left
  node(515, 180, "Prompt assembly", "Tools, history", "server")
  node(353, 180, "Model inference", "Tokens, sampling", "model")
  node(191, 180, "Stream back", "Server-sent events", "device")
  node(29, 180, "Render", "Markdown, paint", "device")
  arrow(513, 208, 491, 208)
  arrow(351, 208, 329, 208)
  arrow(189, 208, 167, 208)

  // Tool loop
  node(353, 290, "Tool calls", "Search, fetch, code", "model")
  arrow(405, 238, 405, 288)
  arrow(437, 288, 437, 238)

  // Next turn
  arrow(97, 178, 97, 108, dash: "dashed")
  place(top + left, dx: 107 * u, dy: 137 * u, text(size: 7.5pt, fill: ink, [↻ next turn]))

  legend-item(110, "device")
  legend-item(300, "server")
  legend-item(450, "model")
})))

== Outline <touying:hidden>

#show outline: set text(size: 20pt)

#components.adaptive-columns(outline(title: none, depth: 2, indent: 1em))

= The big picture

== Key takeaways

- An LLM-based service is a _networked software system_ wrapped around a _generative language model_.
- Your prompt is a _small part_ of the exchange.
- The interaction is orchestrated by the software surrounding the model, called the _harness_.
- Responses are generated _autoregressively_, one token at a time.
- Much of the journey is _ordinary web technology_.

== Information flow

#figure(
  schema,
  caption: [The general information flow of a prompt. Generated with Claude.],
) <fig-flow>

= Step-by-step process

== From input to client request

- Typing is _local only_.
- The client-side software (web browser, desktop or mobile app) sends your prompt as a _web request_ when you submit it, augmenting it with info such as:
  - conversation ID
  - selected model
  - references to attached files, if any
  - client context (time zone, etc.)
- the user interface is updated to show that a response is pending.

== Network exchange

- The request travels from the client to an LLM-based service's server over an encrypted HTTPS/TLS connection.
- The response is not merely a single page, but a _long-lived stream_.
- Through this open stream, the server can send back _response fragments_ as they become available.

== Server intake

Upon request arrival, several server-side operations take place:

- User authentication and authorization.
- Usage and rate limits checkups.
- Conversation storage.
- Safety screenings.

== Prompt assembly

The model receives one long document containing (in this order):
- The _system prompt_.
- The _entire conservation history_, including your current prompt at the very end.

---

=== The system prompt

---

=== The context window

- Model's _working memory_: maximum amount of text (measured in tokens) that it can take into account at once.

== Model inference

== Tool calls

== Response streaming and rendering

- Eventually, the model samples a special "stop" token that triggers:
  - network stream closing;
  - user interface updating.
- Between prompts, the client/server interaction is _stateless_: each subsequent submission starts a new cycle.
