// notes_to_pdf.py -i papers/Noesis/paper.slides.md -o papers/Noesis/paper.slides.pdf --type slides --slides_engine typst

# Noesis: A Protocol for Trading Machine Intelligence

## Investor Pitch: Problem and Solution

* Problem

- Today, applications and agents buy LLM inference

- AI tokens are treated like a commodity, but ... the market has none of a
  commodity market's mechanisms

  1. **No price discovery**: a small number of providers post static, unilateral
     price lists
  2. **No bundled quality guarantee**: a per-token price says nothing about
     the capability, latency, or reliability
  3. **No verified fulfillment**: nothing checks a response against what was
     promised, (e.g., _model nerfing_ or degraded service goes undetected)
  4. **No accountability**: a provider that under-delivers pays no
     reputational or financial cost and can keep selling at the same price

* Consequence

- Buyers cannot
  - compare offers
  - prove they were shortchanged
  - overpay by routing every request to a model that might be too expensive
    for their needs

- Each provider's pricing scheme is incompatible with every other's,
  fragmenting the market instead of helping buyers compare
  - No mechanism exists today to correct this

* Solution: Noesis

- **Noesis** is a market for AI intelligence where buyers and sellers
  automatically trade AI tokens
  - Think of Nasdaq for AI tokens

- Noesis treats machine intelligence as a fungible commodity and clears it
  through two cooperating components
  - **NoesisMarket**: a periodic call auction that matches buyers bidding for
    a (capability, latency, reliability) bundle against sellers asking a
    price, clearing at a single uniform price per capability tier
  - **NoesisServer**: an API gateway that executes the matched contract
    against real providers, meters whether it was honored, and reports
    shortfalls back to the market as a reputation and pricing signal

* Why it matters
- **Verified, not trusted**: a statistical fulfillment test catches
  under-delivery instead of taking a provider's word for it
- **Enforceable accountability**: violations lower a seller's reputation
  score and gate its eligibility, mirroring capacity-market penalties in
  electricity markets, with an on-chain stake-slashing variant for a
  trust-minimized version
- **Exchange economics**: liquidity pooling across many sellers behind one
  API, monetized the way a financial exchange is, a small fee on cleared
  volume
- **Compounding data moat**: every logged request funds difficulty-aware
  routing, answer fusion, and an opt-in distillation corpus, cutting cost
  further as volume grows

- @Key idea@: buyers pay for compute consumed, not functionality delivered,
  and have no standard way to verify or compare what they bought

* Competitive Analysis

// TODO(ai_gp): Add examples of services
// TODO(ai_gp): Add Noesis with all yes
\begingroup \scriptsize
```{=typst}
#styled-table(
  headers: ("Property buyers need", "Direct API", "Gateway", "Spot compute"),
  rows: (
    ("Price discovery via open auction", "No", "No", "Yes"),
    ("Bundled quality guarantee", "Partial", "Partial", "No"),
    ("Verified fulfillment / SLA monitoring", "No", "No", "Partial"),
    ("Reputation-based accountability", "No", "No", "Partial"),
    ("Difficulty-aware cost routing", "No", "Partial", "No"),
  ),
  caption: "No existing option covers the properties buyers need",
)
```
\endgroup

* Market

// TODO(ai_gp): Describe the TAM and the projections

* Noesis: Internals

// TODO(ai_gp): Add Fig 1 pic from the paper

* Noesis: Two Implementations

- Can be implemented as centralized exchange or decentralized one

