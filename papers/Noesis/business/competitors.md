# Competitor Matrix

Data as of 2026-11. Most figures are third-party estimates (Sacra, Latka, ZoomInfo,
Dealroom, CB Insights, CoinGecko), not company disclosures. Treat revenue,
headcount, and price as rough. `?` = not found or not verified

## Capabilities

Legend: Yes = core feature. Partial = exists but limited, manual, or not
guaranteed. No = absent

- **Auction price**: buyers bid, sellers ask, and a clearing price results
- **Bundled quality**: a buyer can buy a (capability, latency, reliability)
  contract, or an SLA with a numeric uptime or latency promise
- **Metered fulfillment**: delivery is measured against the bought contract. A
  generic uptime SLA does not count
- **Reputation feeds price**: a shortfall lowers a seller's standing, pay, or
  eligibility
- **Pooled sellers**: many sellers sit behind one API or one market

| Competitor                 | Auction price | Bundled quality | Metered fulfillment | Reputation feeds price | Pooled sellers | Customer pays for                 |
| :------------------------- | :------------ | :-------------- | :------------------ | :--------------------- | :------------- | :-------------------------------- |
| OpenAI API                 | No            | Partial         | No                  | No                     | No             | Tokens, list price                |
| AWS Bedrock                | No            | Partial         | No                  | No                     | No             | Tokens, or provisioned hours      |
| OpenRouter                 | No            | Partial         | No                  | No                     | Yes            | Tokens, plus 5.5% fee             |
| Portkey                    | No            | Partial         | No                  | No                     | Yes            | Plan by log volume                |
| LiteLLM                    | No            | No              | No                  | No                     | Yes            | Enterprise license                |
| Requesty                   | No            | No              | No                  | No                     | Yes            | Tokens, plus 5% markup            |
| Martian                    | No            | ?               | No                  | No                     | Partial        | ?                                 |
| Not Diamond                | No            | Partial         | No                  | No                     | Partial        | Tokens routed                     |
| Unify                      | No            | No              | No                  | No                     | ?              | ?                                 |
| Akash Network              | Yes           | No              | No                  | No                     | Yes            | Compute lease                     |
| Vast.ai                    | Partial       | No              | Partial             | Partial                | Yes            | GPU-hours                         |
| RunPod                     | No            | No              | No                  | No                     | Yes            | GPU-hours                         |
| io.net                     | No            | No              | Partial             | No                     | Yes            | GPU-hours                         |
| SF Compute                 | Partial       | No              | Partial             | No                     | Yes            | GPU-hours, plus fee               |
| Compute Exchange           | Partial       | No              | No                  | No                     | Yes            | Capacity contract                 |
| Andromeda                  | No            | No              | Partial             | Partial                | Yes            | Cluster contract, plus spread     |
| Ornn                       | No            | No              | No                  | No                     | Yes            | Compute contract                  |
| Prime Intellect            | No            | No              | No                  | No                     | Yes            | GPU-hours, tokens                 |
| Bittensor                  | No            | No              | Partial             | Yes                    | Partial        | Tokens (seller paid by emissions) |
| Gensyn                     | No            | No              | Partial             | No                     | No             | ?                                 |
| Ritual                     | No            | No              | Partial             | Partial                | Yes            | Per-model rate, in RITUAL token   |
| Nosana                     | No            | No              | No                  | Partial                | Yes            | GPU-hours                         |
| Fetch.ai / ASI Alliance    | No            | No              | No                  | Partial                | Partial        | ?                                 |
| x402 / AP2 (payment rails) | No            | No              | No                  | No                     | No             | Settlement fee                    |
| **Noesis**                 | Yes           | Yes             | Yes                 | Yes                    | Yes            | **Cleared volume**                |

## What They Do

### OpenAI API

- Direct first-party model API for developers and enterprises
- Per-token list price, with Batch and Flex at 50% off and Priority at about 2x
- Enterprise Scale Tier sells capacity by the day, with an uptime and latency SLA
  and a credit if both are missed (read on an unofficial mirror, not verified)
- Substitute for Noesis, not an exchange: one seller, one posted price

### AWS Bedrock

- Managed multi-vendor model API billed through AWS
- On-demand per-token price, Batch 50% off, Priority tier +75%, Provisioned
  Throughput by the hour
- Availability SLA per region, with credits of 10-100% of the bill
- Many models, but one seller and fixed prices. Checks uptime only

### OpenRouter

- One API to 400+ models from 80+ providers, with automatic fallback
- Passes provider list price through and charges 5.5% on credit purchases
- Company-reported 8M+ users and 25T tokens a week (May 2026)
- Stripe agreed to acquire it on 2026-08-19, price undisclosed (press reports
  $7.5B-8B)
- Closest incumbent: same single-API pitch, but providers set the price and no
  one checks quality

### Portkey

- AI gateway and control plane: routing, guardrails, observability, budgets
- Does not resell tokens. Charges by plan and log volume
- Company-reported 24,000+ orgs and 120M requests a day (Feb 2026)
- Palo Alto Networks closed its acquisition on 2026-05-29

### LiteLLM

- MIT-licensed SDK and proxy to 100+ providers, self-hosted by platform teams
- Virtual keys, budgets, load balancing, and rate limits
- Revenue is enterprise licenses. No cut of token spend
- Retries and logging only. No numeric SLA published

### Requesty

- Gateway and router to 600+ models from 30+ providers, with failover chains
- Model cost plus a 5% markup, no minimum
- Self-reported 70,000+ developers

### Martian

- Homepage is now an interpretability and benchmark research lab pitch
- Docs describe a gateway to 200+ models. Older material describes a prompt-level
  model router
- TODO(gp): a linked site, thesean.ai, claims 50% lower cost at matched quality
  "backed by an SLA". The link to Martian is not confirmed

### Not Diamond

- Model router: picks the best model for each prompt to cut cost, keeping quality
- Trains custom routers on customer data. Adds about 100-150 ms a request
- Company-reported customers include Hugging Face, Dropbox, and IBM
- Routes inside the customer's own gateway. No auction, no penalty

### Unify

- Name collision: the San Francisco go-to-market platform Unify is a different
  company. This entry is Unify.ai (London, YC W23)
- 2024 product was a learned router that balanced quality, cost, and speed
- Site now pitches continual learning and lists no router, pricing, or customers

### Akash Network

- Marketplace for GPU and CPU leases from independent providers
- Reverse auction: the tenant posts a spec and a max price, providers bid, the
  tenant picks a bid
- Since 2026-03 buyers burn AKT to mint a dollar-pegged credit that pays leases
- Per-lease auction, not a periodic uniform price. Sells raw compute: no answer
  quality, no uptime check found

### Vast.ai

- GPU rental market with hundreds of hosts. Hosts set prices
- Tiers: on-demand, reserved, and interruptible (bid-ordered)
- Reliability score and "verified" data-center label, method undefined
- Higher reliability tends to carry a higher host price, with no formula
- Name collision: not VAST Data (storage)

### RunPod

- GPU pods and serverless inference, with data-center partners and community
  hosts
- Posted prices. No auction
- SLA only on reserved clusters, with no uptime figures

### io.net

- Decentralized GPU cluster network. Suppliers stake the IO token
- Posted per-GPU prices. Pays in IO, USDC, or fiat
- Per a third-party review, uptime and benchmark proofs exist. The verifier is
  closed source

### SF Compute

- Order book for GPU time in 8-GPU node blocks, filled by the minute
- Sellers are cluster owners and buyers reselling idle reserved time
- Continuous order book, first order to arrive sets the fill price. No batch
  auction
- Burn-in and health checks. Automated refunds are partly built

### Compute Exchange

- Brokers reserved GPU capacity (1-36 months), used GPUs, and forward contracts
- Buyer posts specs and gets quotes through a "live auction" among vetted
  providers. Bidding rules are not published
- Provider pays a 4% commission
- Name collision: not Prime Intellect's "Compute Exchange" product

### Andromeda

- Resells GPU clusters: the buyer names GPU, quantity, and region, and Andromeda
  sources, benchmarks, and prices across providers
- Keeps a spread on capacity leased upfront. Posted price, no auction
- Self-reported certification engine benchmarks sellers and gates supply

### Ornn

- Compute marketplace plus the Ornn Compute Price Index, built from settled
  trades and published on Bloomberg Terminal
- ICE plans cash-settled GPU futures on the index, pending regulatory approval
- Also a token price index. Prices tokens, does not check answer quality

### Prime Intellect

- AI stack: compute aggregated from providers, a per-token inference API, RL
  training, and an environments hub
- Posted prices. No auction
- TOPLOC (company blog) checks that inference ran as claimed, not that the
  answer is good

### Bittensor

- Subnets sell a digital good. Miners produce it, validators score miners, and
  the chain pays emissions
- Chutes (subnet 64) sells serverless open-model inference at fixed per-token
  prices
- Validator scores set miner emissions and eligibility. Closest known match to
  the Noesis reputation loop
- No buyer bid, and the score does not set the buyer's price
- Revenue is small next to the emission subsidy: Pine Analytics estimates Chutes
  external revenue at $1.3-2.4M a year against about $52M a year of emissions

### Gensyn

- ML research and production lab. Not an inference exchange
- REE emits a verifiable receipt that proves a reproducible run, not answer
  quality
- Its auction was a token sale (Dec 2025), not an inference market

### Ritual

- Name collision: this entry is ritual.net, the AI-focused L1 chain
- Mainnet and token launched 2026-10-26. Contracts call an LLM precompile and
  pay a per-model rate in RITUAL
- TEE attestation proves which code ran on which input, not answer quality
- Executor registry gates eligibility. No score changes price

### Nosana

- Solana GPU marketplace. Hosts list fixed hourly prices
- Proposal NNP-0001 would weight rewards and scheduling by uptime, latency, and
  job completion, and add collateral. Adoption not confirmed

### Fetch.ai / ASI Alliance

- Agent directory (Agentverse), a personal AI front end, and a developer
  framework. Not an inference exchange
- Agents are paid in the FET token. No auction or answer check found
- Name collision: not the unrelated "Fetch" firms in Latka

### x402 / AP2 (payment rails)

- x402: HTTP 402 payment flow, mainly stablecoins on Base and Solana. Now under
  a Linux Foundation x402 Foundation (2026-07)
- AP2: Google's agent payment authorization protocol, with signed mandates
- Neither sets prices or checks quality. Complements: possible settlement layer
  under Noesis

### Noesis

- Exchange for AI inference: buyers bid on a (tier, latency, reliability) bundle
- A periodic auction clears one price per capability tier
- A gateway meters each request and reports shortfalls to a reputation score
- Fee of `<FEE>`% on cleared volume

## Not Competitors: Prospective Sellers

- Together AI, Fireworks AI, Groq, Baseten, and Fal.ai sell inference at posted
  per-token or per-second prices
- They would be Noesis sellers. None exposes metering as a contract that buyers
  can shop across

## Company Data

| Competitor              | Website                        | Price                                                           | Funding                                                    | Revenue                                                        | Employees                    |
| :---------------------- | :----------------------------- | :-------------------------------------------------------------- | :--------------------------------------------------------- | :------------------------------------------------------------- | :--------------------------- |
| OpenAI API              | https://openai.com/api         | Per-token list; Batch and Flex 50% off, Priority about 2x       | $122B round at $852B post-money (2026-03, eWeek)           | ?                                                              | ?                            |
| AWS Bedrock             | https://aws.amazon.com/bedrock | Per-token; Batch 50% off, Priority +75%; Provisioned hourly     | Amazon subsidiary                                          | Not disclosed per product                                      | n/a                          |
| OpenRouter              | https://openrouter.ai          | 5.5% on credits ($0.80 min); Business 8%; BYOK free to $25K/mo  | About $153M; $113M Series B (2026-05) at about $1.3B       | About $160M annualized (Sacra est., 2026-08; conflicts)        | About 69-100 (est.)          |
| Portkey                 | https://portkey.ai             | Free 10K logs; $49/mo Production; Enterprise custom             | About $18M; $15M Series A (2026-02); acquired 2026-05      | $5-7M (est., Latka and ZoomInfo; conflict)                     | About 11-50 (est.)           |
| LiteLLM                 | https://litellm.ai             | Open source $0; Enterprise on request                           | $1.6M seed (YC, Gravity Fund, Pioneer Fund)                | $2.5-7M ARR (unverified)                                       | About 8-20 (est.)            |
| Requesty                | https://requesty.ai            | Free 200 req/day; 5% markup on model cost                       | $3M seed (2025-09, 20VC)                                   | $1.5M ARR at raise (Respan est.)                               | ?                            |
| Martian                 | https://withmartian.com        | ?                                                               | $9M seed (2023-11, NEA, Prosus, General Catalyst)          | ?                                                              | About 11-50 (est.)           |
| Not Diamond             | https://notdiamond.ai          | $0.05 per 1M tokens routed; Enterprise custom                   | $2.3M pre-seed (2024-07, defy.vc); later round undisclosed | Under $5M (ZoomInfo est.)                                      | About 9-30 (est.; conflict)  |
| Unify                   | https://unify.ai               | ?                                                               | $8M (2024-05, SignalFire, M12); one tracker says $6.65M    | ?                                                              | 3-7 (2024; conflict)         |
| Akash Network           | https://akash.network          | Avg bid per GPU-hour: H100 $2.73, A100 $1.67 (2026-11)          | About $2M (CB Insights est.)                               | $3.15M tenant spend in 2025 (company-reported)                 | ? (est. about 123)           |
| Vast.ai                 | https://vast.ai                | Host-set; interruptible "50%+ cheaper"; fee undisclosed         | About $4M seed (2024-07, DRW); other sites conflict        | Over $1M a month (company-reported, 2026-03)                   | 40+ (press kit)              |
| RunPod                  | https://runpod.io              | H100 SXM $4.39 Secure, from $2.69 Community, per GPU-hour       | $122M; $100M Series A (2026-06) at $1B (Summit Partners)   | ARR over $325M (company-reported, 2026-10)                     | About 80-102 (est.)          |
| io.net                  | https://io.net                 | H100 $1.49-2.20 per GPU-hour (company posts; conflict)          | $30M Series A (2024-03, Hack VC)                           | $20M annualized on-chain (company-reported, 2025-10)           | 50-100 (The Block, 2024)     |
| SF Compute              | https://sfcompute.com          | Order book; fee about 10% per trade (one outlet)                | $40M Series A (2025-11, DCVC, Wing); about $52M total      | ?                                                              | ?                            |
| Compute Exchange        | https://compute.exchange       | 4% commission paid by the provider                              | ? (Variant Fund seed; DRW backed)                          | ?                                                              | ?                            |
| Andromeda               | https://andromeda.ai           | Spread, undisclosed                                             | $1.5B valuation (2026-03, Paradigm); total ?               | About $100M run rate in 2025 (reported, unconfirmed)           | About 20 (2026)              |
| Ornn                    | https://ornn.com               | ?                                                               | $33M seed (2026-06, a16z crypto); $5.7M seed (2025-11)     | ?                                                              | ?                            |
| Prime Intellect         | https://primeintellect.ai      | On-demand H100 $3.50, H200 $5.48 per GPU-hour                   | $130M Series A (2026-07, Radical Ventures); about $150M    | $100M annualized (TechCrunch, 2026-07)                         | About 40-57 (est.)           |
| Bittensor               | https://bittensor.com          | Chutes $0.03-2.45 per 1M input tokens; miners paid by emissions | No equity round found                                      | Chutes $1.3-2.4M a year (Pine est.); network $3-70M (conflict) | 11-50 (Opentensor, LinkedIn) |
| Gensyn                  | https://gensyn.ai              | ? (token sale raised $16.14M, 2025-12)                          | $43M Series A (2023-06, a16z crypto); above $50M total     | ?                                                              | About 44-53 (est.)           |
| Ritual                  | https://ritual.net             | Per-model rate in RITUAL token                                  | $25M (2023-11, Archetype)                                  | ?                                                              | 10 (2023)                    |
| Nosana                  | https://nosana.com             | Listed GPU rates, e.g., H100 $1.50/h, 4090 $0.40/h              | Token sales only; no VC round found                        | ?                                                              | ?                            |
| Fetch.ai / ASI Alliance | https://fetch.ai               | ? (FET token)                                                   | $40M (2023-03, DWF Labs); about $62M total                 | $1-37M (est.; low trust)                                       | About 78-121 (est.)          |
| x402 / AP2              | https://www.x402.org           | Free protocol; facilitator $0.001 per settlement after 1,000/mo | Open standard (Linux Foundation, 2026-07)                  | n/a                                                            | n/a                          |
| **Noesis**              | ?                              | **`<FEE>`% of cleared volume**                                  | Seeking $`<ASK>` `<ROUND>`                                 | None: design not yet tested on real traffic                    | 4 (slide authors)            |

## Notes for the Pitch

- OpenRouter is the main incumbent: a Stripe deal (2026-08) and a $1.3B Series B.
  Its 5.5% credit fee on the Competition slide still matches its pricing page
- Portkey sold to Palo Alto Networks (2026-05): gateways are folding into larger
  security and payments platforms
- Closest on mechanism: Akash, SF Compute, and Compute Exchange for auctions,
  and Bittensor for quality scoring. None prices a verified LLM answer
- x402 and AP2 are payment rails, not competitors. They fit under Noesis as the
  settlement layer
- TODO(gp): the Competition slide says "None" for the penalty of Direct API, but
  OpenAI Scale Tier and Bedrock pay SLA credits for uptime misses. Change it to
  "Uptime credit"
- TODO(gp): ratings here are per company and can differ from the cluster table in
  `07_state_of_the_art.tex` (e.g., RunPod and io.net are posted-price, not
  bid-based)
- TODO(gp): confirm Martian's current product and its link to thesean.ai
- TODO(gp): add a Noesis website once it exists

## Sources

- Noesis: `vc_pitch.slides.md` (Competition, Business Model, Ask),
  `papers/Noesis/07_state_of_the_art.tex`
- OpenAI: https://developers.openai.com/api/docs/pricing,
  https://eweek.com/news/openai-122b-funding-852b-valuation-ai/
- AWS Bedrock: https://aws.amazon.com/bedrock/pricing/,
  https://aws.amazon.com/bedrock/sla/
- OpenRouter: https://openrouter.ai/pricing, https://openrouter.ai/docs/faq,
  https://dealroom.co/news/129679-openrouter-raises-113m-series-b-hits-1-3b-valuation/,
  https://sacra.com/c/openrouter/,
  https://stripe.com/newsroom/news/stripe-agrees-to-acquire-openrouter
- Portkey: https://portkey.ai/pricing, https://portkey.ai/blog/series-a-funding/,
  https://www.paloaltonetworks.com/company/press/2026/palo-alto-networks-completes-acquisition-of-portkey-to-secure-ai-agents,
  https://getlatka.com/companies/portkey.ai,
  https://www.zoominfo.com/c/portkey/566158721
- LiteLLM: https://litellm.ai, https://www.ycombinator.com/companies/litellm,
  https://www.cbinsights.com/company/berriai,
  https://www.zoominfo.com/c/litellm/1326029797
- Requesty: https://www.requesty.ai/pricing,
  https://www.requesty.ai/blog/requesty-raises-3m,
  https://www.respan.ai/market-map/requesty
- Martian: https://withmartian.com, https://docs.withmartian.com,
  https://thesean.ai,
  https://www.maginative.com/article/martian-emerges-from-stealth-with-9m-seed-funding-to-bring-model-router-tech-to-enterprises,
  https://www.respan.ai/market-map/martian
- Not Diamond: https://www.notdiamond.ai/pricing,
  https://www.inovia.vc/?p=14137, https://cbinsights.com/company/not-diamond,
  https://www.zoominfo.com/c/not-diamond/1335158859
- Unify: https://techcrunch.com/2024/05/22/unify-helps-developers-find-the-best-llm-for-the-job, <!-- gitleaks:allow -->
  https://unify.ai, https://www.ycombinator.com/companies/unify
- Akash: https://akash.network/pricing/gpus,
  https://akash.network/roadmap/aep-76/,
  https://akash.network/blog/akash-2025-year-in-review/,
  https://www.cbinsights.com/company/akash/financials
- Vast.ai: https://vast.ai/pricing, https://vast.ai/press-kit,
  https://docs.vast.ai/pricing, https://getlatka.com/companies/vast.ai
- RunPod: https://runpod.io/pricing,
  https://runpod.io/press/runpod-325m-arr-new-cfo,
  https://siliconangle.com/2026/06/25/runpod-raises-100m-build-leading-cloud-platform-ai-developers/,
  https://www.reveliolabs.com/companies/runpod/employees
- io.net: https://io.net/docs/guides/explorer/tne-on-chain,
  https://www.theblock.co/news/deals/2024-03-05-solana-depin-io-net-1-billion-token-valuation-funding-280665,
  https://io.net/p/fhow-much-does-an-h100-cost-per-hour
- SF Compute: https://sfcompute.com/news/sf-compute-series-a,
  https://docs.sfcompute.com/preview/guides/reselling-introduction,
  https://eweek.com/news/sf-compute-ai-funding/,
  https://latent.space/wiki/sf-compute-0739be6d/
- Compute Exchange: https://compute.exchange/fees,
  https://siliconangle.com/2026/07/17/compute-exchange-opens-secondary-market-used-nvidia-h100-a100-gpus/
- Andromeda: https://andromeda.ai,
  https://upstartsmedia.com/p/andromeda-ai-compute-startup-raises-60m
- Ornn: https://ornn.com, https://siliconangle.com/2026/06/24/ornn-raises-33m/,
  https://a16zfrontier.com/posts/article/investing-in-ornn
- Prime Intellect: https://primeintellect.ai/blog/fundraise,
  https://techcrunch.com/?p=3140126,
  https://getlatka.com/companies/prime-intellect
- Bittensor: https://www.bittensor.com/docs/concepts/emissions,
  https://chutes.ai/docs/miner-resources/scoring,
  https://pineanalytics.substack.com/p/the-bear-case-for-bittensor-tao,
  https://cryptobriefing.com/bittensor-revenue-run-rate-70m/,
  https://www.coingecko.com/en/coins/bittensor
- Gensyn: https://docs.gensyn.ai,
  https://tech.eu/2023/06/12/blockchain-based-ai-compute-protocol-gensyn-raises-43-million-in-a16z-led-round/,
  https://www.kucoin.com/news/flash/gensyn-completes-ai-token-public-sale-with-16-14m-raised
- Ritual: https://docs.ritualfoundation.org/,
  https://www.theblock.co/news/deals/2023-11-08-decentralized-ai-compute-platform-ritual-closes-25-million-fundraise,
  https://www.kucoin.com/blog/ritual-mainnet-ritual-tokenomics-airdrop
- Nosana: https://learn.nosana.com/,
  https://nosana.com/blog/from-yield-to-growth-aligning-nos-rewards-with-real-usage
- Fetch.ai / ASI: https://www.coingecko.com/en/coins/artificial-superintelligence-alliance,
  https://techcrunch.com/2023/03/29/blockchain-startup-fetch-ai-grabs-40m-to-provide-monetization-and-other-tooling-for-ai-generated-information,
  https://getlatka.com/companies/fetch.ai
- x402: https://www.x402.org,
  https://www.linuxfoundation.org/press/linux-foundation-announces-operational-launch-of-x402-foundation-to-standardize-internet-native-payments-for-ai-agents-and-applications,
  https://docs.cdp.coinbase.com/x402/welcome
- AP2: https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol

# VC Pitch References

- This file lists every source in `business/vc_pitch.slides.md`, and the vendor
  pages behind `business/competitors.md`
- Each summary comes from the source itself, read in November 2026
- Sources that could not be read say so and have no summary

## 2025: The State of Generative AI in the Enterprise

- **Year**: 2025
- **Link**: [Menlo Ventures, State of Generative AI in the Enterprise 2025](https://menlovc.com/perspective/2025-the-state-of-generative-ai-in-the-enterprise/)
- **Key points**:
  - **Spend**: Enterprise generative AI spend reached $37B in 2025, up from $11.5B in
    2024 (3.2x)
  - **Layers**: Application layer took $19B. Infrastructure layer took $18B, up 2.0x
    from $9.2B in 2024
  - **Share**: Anthropic holds 40% of enterprise LLM spend, OpenAI 27%, Google 21%
  - **Buy vs build**: 76% of AI solutions were purchased, up from 53% in 2024
  - **Open source**: Open-source LLM share of enterprise fell from 19% to 11%

## OpenRouter pricing

- **Year**: 2026
- **Link**: [OpenRouter pricing page](https://openrouter.ai/pricing)
- **Key points**:
  - **Markup**: Inference is billed at each provider's list price on every plan
  - **Fees**: Platform fee is 5.5% on Standard and 8% on Business. Free has no fee.
    Fee applies when credits are bought
  - **Catalog**: Standard and Business list 600+ models and 80+ providers. No minimum
    spend or lock-in
  - **BYOK**: Standard and Business include $25,000 of list-price inference per month
    with no OpenRouter fee. Usage above carries a 5% fee
  - **Enterprise**: Adds SSO/SAML, contractual SLAs and invoicing. Fees are set by
    arrangement

## Google Agentic Payments Protocol + x402: Agents Can Now Actually Pay Each Other (Coinbase)

- **Year**: 2025
- **Link**: [Coinbase developer platform, Google x402 launch](https://www.coinbase.com/developer-platform/discover/launches/google_x402)
- **Not read**: HTTP 403

## Noesis: A Protocol for Trading and Fulfilling Machine Intelligence as a Commodity

- **Year**: 2026
- **Link**: working paper, `papers/Noesis/Noesis.pdf`
- **Key points**:
  - **Problem**: LLM inference sold by static per-model price lists from few
    providers, with no price discovery and no check of what was delivered
  - **Auction**: NoesisMarket runs a call auction every T minutes (example T = 5). It
    clears one uniform price per capability tier
  - **Metering**: NoesisServer gateway tests measured reliability with a one-sided
    lower confidence bound. Example: 0.994 measured over 5,000 requests against 99.9%
    promised is flagged
  - **Reputation**: Seller score updates by exponential decay on each pass or fail. A
    score below a threshold excludes the seller from that tier
  - **Open questions**: Task unit, call versus continuous auction, bid shading,
    collusion, verifier error. No test yet on real traffic

## OpenAI API platform page (OpenAI)

- **Year**: 2026
- **Link**: [OpenAI API platform page](https://openai.com/api)
- **Not read**: HTTP 403

## Pricing (OpenAI API)

- **Year**: 2026
- **Link**: [OpenAI API pricing page](https://developers.openai.com/api/docs/pricing)
- **Key points**:
  - **Standard**: gpt-6.1-sol costs $2.00 input, $10.00 output, $0.10 cached input
    per 1M tokens
  - **Batch**: Batch rates are half of standard. gpt-6.1-sol is $1.00 input and $5.00
    output per 1M tokens
  - **Fast**: Priority processing renamed Fast mode on July 30, 2026. Rates are about
    double standard
  - **Ultrafast**: gpt-6.1-sol costs $12.00 input and $60.00 output per 1M tokens,
    about six times standard
  - **Uplift**: Regional and FedRAMP endpoints carry a 10% uplift on price

## OpenAI 122B funding and 852B valuation (eWeek)

- **Year**: 2026
- **Link**: [eWeek article on OpenAI funding and valuation](https://eweek.com/news/openai-122b-funding-852b-valuation-ai/)
- **Not read**: HTTP 403

## Amazon Bedrock

- **Year**: 2026
- **Link**: [AWS Amazon Bedrock product page](https://aws.amazon.com/bedrock)
- **Key points**:
  - **Product**: Platform to build generative AI apps and agents, with AgentCore and
    access to hundreds of foundation models
  - **Scale**: Powers generative AI for more than 100,000 organizations worldwide,
    company-reported, no date given
  - **Price**: No prices on page. Intelligent Prompt Routing claims up to 30% lower
    cost, new AWS customers get up to $200 in credits
  - **Quality**: Guardrails block up to 88% of harmful content. Automated Reasoning
    checks claim up to 99% accuracy, company-reported
  - **Customer**: Robinhood grew from 500 million to 5 billion tokens daily in six
    months and cut AI costs by 80%

## Amazon Bedrock pricing

- **Year**: 2026
- **Link**: [AWS Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/)
- **Key points**:
  - **Priority**: Priority tier is at a 75% premium to Standard tier pricing
  - **Flex**: Flex tier is at a 50% discount to Standard tier pricing
  - **Batch**: Select Anthropic, Meta, Mistral AI and Amazon models cost 50% less in
    batch than on-demand
  - **Provisioned**: Cohere Command costs $49.50 per model unit per hour with no
    commitment, $23.77 with a 6-month commitment
  - **Routing**: Intelligent Prompt Routing costs $1 per 1,000 requests. Page says it
    can cut costs by up to 30 percent

## Amazon Bedrock Service Level Agreement

- **Year**: 2023
- **Link**: [AWS Bedrock SLA, last updated October 4, 2023](https://aws.amazon.com/bedrock/sla/)
- **Key points**:
  - **Credits**: Monthly uptime below 99.9% earns a 10% credit. Below 99.0% earns
    25%. Below 95.0% earns 100%
  - **Error**: An Error is a Request that returns HTTP 500. Availability is measured
    per 5-minute interval
  - **Claim**: Open an AWS Support case titled "SLA Credit Request" by the end of the
    second billing cycle after the incident
  - **Remedy**: Credit is the sole remedy. It is not refundable and is issued only
    above $1 USD
  - **Exclusions**: Downtime from underlying software causing repeated model crashes
    is excluded from the uptime calculation

## OpenRouter

- **Year**: 2026
- **Link**: [OpenRouter homepage](https://openrouter.ai)
- **Key points**:
  - **Product**: Hosted LLM gateway with one OpenAI-compatible API and one key for
    hundreds of text, image, video, and audio models
  - **Pricing**: Pay-as-you-go with no markup on provider price, billed per token for
    text, per image, per second of video
  - **Free tier**: `:free` model variants cost nothing, with a daily rate limit that
    rises after buying credits
  - **Scale**: Company-reported State of AI study covers more than 100 trillion
    tokens, no date given
  - **SLA**: Enterprise plan includes an SLA, but page states no uptime percentage

## Frequently Asked Questions (OpenRouter)

- **Year**: 2026
- **Link**: [OpenRouter FAQ](https://openrouter.ai/docs/faq)
- **Key points**:
  - **Fee**: Credit purchase fee is 5.5% with a $0.80 minimum. Crypto payments are
    charged 5%
  - **Fallback**: If a provider returns an error, OpenRouter falls back to the next
    provider automatically
  - **Routing**: `:nitro` sorts providers by throughput. `:floor` sorts by price.
    `:exacto` sorts by quality signals for tool calling
  - **Logging**: Prompts and completions are not logged by default. Opting in gives a
    1% discount on usage costs
  - **Refunds**: Unused credits are refundable within 24 hours of purchase. Credits
    may expire after one year

## OpenRouter raises $113M Series B, hits $1.3B valuation | Dealroom News

- **Year**: 2026
- **Link**: [Dealroom news on OpenRouter Series B, May 26, 2026](https://dealroom.co/news/129679-openrouter-raises-113m-series-b-hits-1-3b-valuation/)
- **Key points**:
  - **Funding**: $113M Series B, led by CapitalG (Alphabet's investment arm)
  - **Valuation**: About $1.3B, more than double the previous valuation
  - **Volume**: 25 trillion tokens per week, up fivefold from six months earlier
  - **Revenue**: Annualised revenue rate reportedly above $50M, a fivefold jump since
    October 2025
  - **Catalog**: More than 400 AI models from providers such as OpenAI, Anthropic,
    Google, DeepSeek and Tencent

## OpenRouter revenue, valuation and funding | Sacra

- **Year**: 2026
- **Link**: [Sacra profile of OpenRouter](https://sacra.com/c/openrouter/)
- **Key points**:
  - **Revenue**: Annualized revenue $160M in August 2026, up from $50M at end of 2025
    (Sacra estimate)
  - **Take rate**: About 5% on top of inference spend (Sacra estimate). Implies
    roughly $400K monthly revenue at $8M monthly spend
  - **Valuation**: $1.30B in 2026. A Stripe deal above $7B is reported but neither
    company has confirmed it
  - **Funding**: $28M Series A led by Menlo Ventures (April 2025). $12.5M seed led by
    Andreessen Horowitz (February 2025)
  - **Users**: Page states 8 million developers in one section and 2.5 million users
    in another

## Stripe agrees to acquire OpenRouter to help businesses optimize token routing and usage

- **Year**: 2026
- **Link**: [Stripe newsroom: Stripe agrees to acquire OpenRouter](https://stripe.com/newsroom/news/stripe-agrees-to-acquire-openrouter)
- **Key points**:
  - **Date**: Stripe announced an agreement to acquire OpenRouter on August 19, 2026
  - **Price**: Page does not disclose purchase price, financing, or other deal terms
  - **Scale**: OpenRouter routes across 400-plus models from 80-plus providers
    (Stripe statement)
  - **Routing**: Per request, it picks a model by task complexity, cost, speed, and
    dependability
  - **Users**: NVIDIA, Zoom, and Lovable are named as OpenRouter users

## Production Stack for Gen AI Builders|Portkey

- **Year**: 2026
- **Link**: [Portkey homepage](https://portkey.ai)
- **Key points**:
  - **Product**: AI gateway with observability, guardrails, governance, prompt
    management, and MCP gateway, covering 1,600+ models
  - **Rebrand**: Banner says Portkey is now Prisma AIRS AI Gateway under Palo Alto
    Networks, linked post dated July 2026
  - **Scale**: Company-reported figure of 3000+ GenAI teams served, no date given
  - **SLA**: Page claims strict SLAs and prints uptime as 0.999%, with no SLA terms
    or period
  - **Reputation**: Claims top rating on G2 with no score or review count, plus named
    customer testimonials

## Portkey | Control Panel for Production AI

- **Year**: 2026
- **Link**: [Portkey pricing page](https://portkey.ai/pricing)
- **Key points**:
  - **Price**: Developer plan free forever with 10K recorded logs/month. Production
    plan $49/month with 100K logs
  - **Overage**: Production costs $9 per additional 100K requests. Developer plan
    allows no overage
  - **Enterprise**: Custom price, 10M+ recorded logs/month, SSO, VPC hosting, SOC2
    Type 2, HIPAA, custom BAAs
  - **Retention**: Production keeps logs 30 days and metrics 90 days. Developer keeps
    logs 3 days and metrics 30 days
  - **SLA**: Page says "backed by robust SLAs" for enterprise. It gives no uptime
    figure

## Portkey Raises $15M Series A to Scale the Unified Control Plane for Production AI.

- **Year**: 2026
- **Link**: [Portkey blog: Series A funding](https://portkey.ai/blog/series-a-funding/)
- **Key points**:
  - **Funding**: $15M Series A, February 19, 2026. Led by Elevation Capital with
    Lightspeed participating
  - **Seed**: $3M seed round announced August 2023, per a link on the page
  - **Volume**: Company reports 500B+ tokens and 120M requests processed daily
  - **Spend**: Company reports $180M+ annualized spend under management and $500K+
    managed daily
  - **Users**: Company reports 24,000+ organizations, 60+ providers, and 1,600+
    models supported

## Palo Alto Networks Completes Acquisition of Portkey to Secure AI Agents

- **Year**: 2026
- **Link**: [Palo Alto Networks press release: Portkey acquisition](https://www.paloaltonetworks.com/company/press/2026/palo-alto-networks-completes-acquisition-of-portkey-to-secure-ai-agents)
- **Key points**:
  - **Date**: Press release dated May 29, 2026 says the acquisition has closed
  - **Price**: Page does not state price, consideration, or deal structure. It gives
    no Portkey requests, customers, or headcount
  - **Role**: Portkey becomes the core AI Gateway of Prisma AIRS. It inspects AI
    traffic at runtime
  - **Functions**: Routes requests to the best model, monitors token usage to limit
    runaway costs, and blocks malicious AI behavior
  - **Integrations**: Idira for agent identity security and Chronosphere for AI
    observability

## Portkey, Inc Revenue 2024: $5M Est.

- **Year**: 2024
- **Link**: [GetLatka: Portkey revenue and profile](https://getlatka.com/companies/portkey.ai)
- **Key points**:
  - **Revenue**: About $5M per year. This is a GetLatka estimate dated June 2024. The
    company has not published a figure
  - **Headcount**: 13 employees, September 2024, per Product Hunt
  - **Founder**: Rohit Agarwal is founder and CEO. Based in San Francisco
  - **Competitors**: Page lists OpenRouter, Kong Inc, Unify AI Group, and Eden AI

## ZoomInfo company profile for Portkey

- **Year**: 2026
- **Link**: [ZoomInfo: Portkey company profile](https://www.zoominfo.com/c/portkey/566158721)
- **Not read**: HTTP 403

## LiteLLM - Open-Source AI Gateway and LLM Proxy

- **Year**: 2026
- **Link**: [LiteLLM home page](https://litellm.ai)
- **Key points**:
  - **Price**: Open source self-hosted tier is $0. Enterprise tier has no listed
    price
  - **Adoption**: Company reports 58.8k GitHub stars (header badge), 92M+ PyPI
    downloads per month, and 1,700+ contributors
  - **Coverage**: Company reports 140+ LLM providers and 1,892 unique models
  - **SLA**: Enterprise tier mentions "custom SLAs" with no figures
  - **Benchmark**: Vendor-reported added p99 latency is 0.66 ms for LiteLLM (Rust),
    2.29 ms for Portkey, 4.54 ms for Bifrost

## LiteLLM: Call every LLM API like it's OpenAI [100+ LLMs] | Y Combinator

- **Year**: 2026
- **Link**: [Y Combinator: LiteLLM company page](https://www.ycombinator.com/companies/litellm)
- **Key points**:
  - **Batch**: Y Combinator Winter 2023. Founders are Krrish Dholakia and Ishaan
    Jaffer. Based in San Francisco
  - **Funding**: $1.6M seed round from Y Combinator, Gravity Fund, and Pioneer Fund
  - **Headcount**: Team size 20. Page shows 19 open jobs
  - **Traction**: 50K+ GitHub stars and 100+ supported LLM APIs
  - **Customers**: Page lists Rocket Money, Samsara, Lemonade, and Adobe

## BerriAI - Products, Competitors, Financials, Employees, Headquarters Locations

- **Year**: 2026
- **Link**: [CB Insights: BerriAI (LiteLLM) profile](https://www.cbinsights.com/company/berriai)
- **Key points**:
  - **Funding**: Total raised $500K. Stage shown as "Seed VC - II". Founded 2023, San
    Francisco
  - **Investors**: Venture Bridge, Pioneer Fund, Unpopular Ventures, Y Combinator
  - **News**: Headline dated August 6, 2026: "LiteLLM Raises $1.6M Seed to Scale AI
    Gateway Platform". It differs from the $500K total
  - **Competitors**: Page names OpenRouter and 2 more in its FAQ
  - **Revenue**: No revenue, valuation, or headcount figure on the page

## ZoomInfo company profile for LiteLLM

- **Year**: 2026
- **Link**: [ZoomInfo: LiteLLM company profile](https://www.zoominfo.com/c/litellm/1326029797)
- **Not read**: HTTP 403, then HTTP 429 on retry

## Requesty - The AI Gateway for Production

- **Year**: 2026
- **Link**: [Requesty homepage](https://requesty.ai)
- **Key points**:
  - **Product**: Unified AI gateway and OpenAI-compatible router for 600+ models from
    30+ providers
  - **Price**: 5% markup on model costs with all features included, free start,
    volume discounts for enterprise
  - **Scale**: Company-reported 70,000+ developers and 90+ billion tokens processed
    daily, no date given
  - **SLA**: No uptime percentage stated, page cites provider failover and a status
    page

## Pricing: Free Tier and Pay-As-You-Go LLM Gateway | Requesty

- **Year**: 2026
- **Link**: [Requesty pricing page](https://www.requesty.ai/pricing)
- **Key points**:
  - **Markup**: Pay as you go plan charges 5% markup on model cost. Example: $10 per
    1M tokens at OpenAI becomes $10.50
  - **Free tier**: $0, free models only, 200 requests per day, no credit card
  - **Models**: 600+ models from 30+ providers behind one OpenAI-compatible endpoint
  - **Enterprise**: Custom pricing with SSO, RBAC, audit logs and custom SLAs. No SLA
    figures given
  - **Routing**: Routing, caching, fallbacks and EU data residency listed on every
    plan

## Requesty Raises $3M to Become the Developer's Gateway to Safe AI: The OpenRouter Alternative Europe Has Been Waiting For

- **Year**: 2025
- **Link**: [Requesty seed post](https://www.requesty.ai/blog/requesty-raises-3m)
- **Key points**:
  - **Funding**: $3M seed, September 26, 2025, led by 20VC. Also Tapestry VC,
    Insiders Ventures, Tiny Supercomputer
  - **Savings**: Company claims most customers see 30-50% lower AI cost. One customer
    reportedly saved $400,000 per year
  - **Latency**: Company claims 50ms average added latency, against 200ms+ for
    Portkey
  - **Compliance**: GDPR claimed with EU data residency. SOC 2 Type II listed as "in
    progress"
  - **Support**: 24/7 support with SLAs claimed. No users, revenue or headcount
    stated

## Requesty: LLM Gateways

- **Year**: 2026
- **Link**: [Respan market map entry for Requesty (competitor-authored)](https://www.respan.ai/market-map/requesty)
- **Key points**:
  - **Funding**: Seed of GBP 2M (~$3M) led by 20VC. Page states $1.5M ARR and 25,000+
    developers at that time
  - **Users**: 50,000+ developers by early 2026, per the page. Clients named:
    Shopify, Amadeus, Chargebee, Pfizer
  - **Headcount**: Listed as 2-10 people, with "~5 employees" in Cons. Founded 2023,
    London
  - **SLA**: 99.99% uptime SLA claimed. Failover claimed "under 20ms". Custom SLAs on
    Enterprise
  - **Price**: Free gives $6 credits. Pro is usage-based with 5% markup. Enterprise
    is custom quote

## Martian: Understanding Intelligence

- **Year**: 2026
- **Link**: [Martian home page](https://withmartian.com)
- **Key points**:
  - **Focus**: Research lab on interpretability. Says it commercializes through
    products that scale with global LLM usage
  - **Team**: Page says members worked on early DeepMind science initiatives and on
    Anthropic interpretability
  - **Projects**: ARES (online RL for coding agents), Code Review Bench, K-Steering,
    interpretability challenge
  - **Related**: Links Thesean AI, marked "NEW", described as "A Lab Building Best
    Execution for LLMs"
  - **Gaps**: No pricing, funding, headcount, customers or SLAs on the page

## Martian Documentation

- **Year**: 2026
- **Link**: [Martian docs](https://docs.withmartian.com)
- **Key points**:
  - **Gateway**: "200+ models" through one API, compatible with OpenAI and Anthropic
    APIs
  - **Endpoint**: api.withmartian.com/v1/chat/completions with Bearer token. Model
    set in the `model` field, for example "openai/gpt-4.1-nano"
  - **Dashboard**: Tracks real-time usage, model performance and request history
  - **Open source**: ARES (RL framework for training and evaluating agents) and
    K-Steering (inference-time steering of activations)
  - **Gaps**: Page states no pricing, SLAs, rate limits or routing mechanism

## Thesean AI: A Lab Building Best Execution for LLMs

- **Year**: 2026
- **Link**: [Thesean AI home page](https://thesean.ai)
- **Key points**:
  - **Claim**: "Ship reduces your LLM bill 50% without changing model behavior". Page
    says an SLA guarantees it
  - **Grades**: Models named `ship-like/<model>` aim to match a reference model at
    lower cost. Each has a quality SLA
  - **Coverage**: Reference models include Claude Opus 5, GPT 5.5, Grok 4.6, Kimi-K3
    and DeepSeek V4 Flash
  - **Exceptions**: Some grades say "a fraction of the cost" with no 50% figure, such
    as GPT-6 Luna
  - **Positioning**: Calls its approach best execution. Team described as ex-quant,
    ex-Anthropic, ex-DeepMind. No funding or prices stated

## Martian Emerges from Stealth with $9M Seed Funding to Bring Model Router Tech to Enterprises

- **Year**: 2023
- **Link**: [Maginative article on Martian seed round](https://www.maginative.com/article/martian-emerges-from-stealth-with-9m-seed-funding-to-bring-model-router-tech-to-enterprises)
- **Key points**:
  - **Funding**: $9M seed, November 15, 2023. Led by NEA, with Prosus Ventures, Carya
    Venture Partners, General Catalyst
  - **Product**: Model Router, patent-pending. Routes each query in real time to the
    model judged most effective
  - **Claim**: Company says it can outperform GPT-4 on openai/evals while cutting
    compute cost. No scores or percentages given
  - **Technique**: Model Mapping, which turns opaque neural networks into more
    interpretable architectures
  - **Team**: Founders Shriyash Upadhyay and Etan Ginsberg. San Francisco. No
    customers, pricing or headcount stated

## Martian: LLM Gateways

- **Year**: 2026
- **Link**: [Respan market map entry for Martian (competitor-authored)](https://www.respan.ai/market-map/martian)
- **Key points**:
  - **Headcount**: "11-50 people". Founded 2023, San Francisco
  - **Price**: Usage-based. Free is $0 per month. Pro is custom. Enterprise is custom
    on an annual contract
  - **SLA**: Enterprise lists SLA guarantees. No uptime figure given
  - **Routing**: Routing by prompt type, automatic quality optimization,
    cost-performance tradeoff, OpenAI-compatible API

## Not Diamond - Model Routing for Coding Agents

- **Year**: 2026
- **Link**: [Not Diamond homepage](https://notdiamond.ai)
- **Key points**:
  - **Product**: Model router for coding agents that predicts which model to use for
    each input
  - **Claims**: +5% accuracy, +50% cost savings, and 2x faster dev cycles,
    company-reported, no baseline or method
  - **Calculator**: For 1,000 engineers at $300 monthly spend, modeled cost falls
    from $4.8M/yr to $3.6M/yr
  - **Reputation**: OpenRouter CEO says it powers OpenRouter routing. Rootly reports
    39% average accuracy gain on SRE benchmarks
  - **SLA**: Custom SLAs and 24/7 support offered, no uptime figure. Page states
    SOC-2 and ISO 27001 compliance

## Pricing (Not Diamond)

- **Year**: 2026
- **Link**: [Not Diamond pricing page](https://www.notdiamond.ai/pricing)
- **Key points**:
  - **Price**: Pay-as-you-go at $0.05 per million tokens routed. Enterprise has no
    listed price
  - **Latency**: Average added latency is 100-150ms per routing request. Page says it
    is not a gateway
  - **Savings**: Company claims at least 20-40% cost savings. Example: $4.8M/yr falls
    to $3.6M/yr for 1,000 engineers
  - **Privacy**: Page says it never sees payloads, inputs or outputs. Routing uses
    derived metadata and local features
  - **Security**: SOC-2 and ISO 27001. Enterprise has custom SLAs and 24/7 on-call
    support. No SLA figures given

## Inovia VC post 14137 (cited for Not Diamond)

- **Year**: 2026
- **Link**: [Inovia VC post 14137](https://www.inovia.vc/?p=14137)
- **Not read**: HTTP 429 Too Many Requests, after one retry

## Not Diamond - Products, Competitors, Financials, Employees, Headquarters Locations

- **Year**: 2026
- **Link**: [CB Insights: Not Diamond profile](https://cbinsights.com/company/not-diamond)
- **Key points**:
  - **Funding**: $2.3M seed total, listed as raised "2 yrs ago". Valuation not stated
  - **Investors**: Defy Partners, Julien Chaumond, Zack Kass, Jeff Weiner, Scott
    Belsky, plus 12 others not named
  - **Product**: AI model routing for developers. Picks which model to call per task,
    optimizing speed and cost. Also offers prompt optimization
  - **Launch**: Prompt Adaptation, an agentic system for multi-model enterprise AI,
    launched May 20, 2025
  - **Position**: Named a Challenger in the CB Insights LLM benchmarking and model
    routing map, with OpenRouter among 13 others. Profile is unclaimed

## ZoomInfo company profile for Not Diamond

- **Year**: 2026
- **Link**: [ZoomInfo: Not Diamond profile](https://www.zoominfo.com/c/not-diamond/1335158859)
- **Not read**: HTTP 403 Forbidden, after one retry

## Unify helps developers find the best LLM for the job

- **Year**: 2024
- **Link**: [TechCrunch article on Unify](https://techcrunch.com/2024/05/22/unify-helps-developers-find-the-best-llm-for-the-job) <!-- gitleaks:allow -->
- **Key points**:
  - **Funding**: $8M announced May 22, 2024. Investors: SignalFire, M12, J12, Essence
    VC, A. Capital, Lunar VC, Y Combinator, and angels
  - **Routing**: Router is "a learned neural network" that learns which models do
    best per task. Each new model is benchmarked with a judge model
  - **Pricing**: New users get a $50 credit. Companies pay for custom benchmarks
  - **Users**: About 3,000 signups, a few hundred regular users (company-reported).
    Seven employees, with three more planned in 2024
  - **Competitors**: Names Martian Router, OpenRouter, and Portkey. Unify claims it
    is the only one optimizing quality, cost, and speed together

## unify. Models that keep learning.

- **Year**: 2026
- **Link**: [Unify home page](https://unify.ai)
- **Key points**:
  - **Positioning**: Headline "Models that keep learning." Subhead "Continual
    learning for AI teams." No pricing, credits, or routing details on the page
  - **Backers**: Y Combinator, A.Capital, SignalFire, Microsoft M12, Samsung Next,
    Essence VC. No amounts given
  - **People**: Eight people listed, including Clem Delangue (CEO, Hugging Face) and
    Philip Rathle (CTO, Neo4j). Their role at Unify is not explained
  - **Claims**: Says quality rises and latency and cost fall with use. No figures,
    benchmarks, SLAs, or user counts

## Unify: Models that keep learning | Y Combinator

- **Year**: 2026
- **Link**: [Y Combinator: Unify company page](https://www.ycombinator.com/companies/unify)
- **Key points**:
  - **Batch**: Winter 2023, founded 2023, London. Team size 3. Status active
  - **Routing**: Router picks the best LLM and provider per prompt. Three sliders
    balance quality, speed, and cost
  - **Access**: One API key and one endpoint for many models and providers. A
    dashboard compares quality, cost, and speed
  - **Credits**: $50 in free credits at sign-up
  - **Focus**: Current description is models that continually learn from a stream of
    experience. No funding stated

## Akash Network - Decentralized Compute Marketplace

- **Year**: 2026
- **Link**: [Akash Network homepage](https://akash.network)
- **Key points**:
  - **Product**: Decentralized marketplace to rent GPU and CPU compute from
    independent providers, plus AkashML hosted inference
  - **Auction**: Global reverse auction sets prices. User sets resources, region, and
    max price, providers bid, user accepts a bid to open a lease
  - **Price**: H100 at $1.33/hr on Akash versus $3.93/hr on AWS, company-reported, no
    date or AWS instance type given
  - **Case study**: Razer at $0.01 per generated image and 15x lower inference cost
    than centralized APIs, company-reported, undated
  - **Reputation**: No provider audit or reputation check described. Only filter is
    whether a provider meets the user requirements

## GPU Rental Pricing: Rent H100, H200, B200 and B300 On Demand | Akash

- **Year**: 2026
- **Link**: [Akash GPU rental pricing](https://akash.network/pricing/gpus)
- **Key points**:
  - **Price**: Average per GPU-hour: H100 80 GB SXM $2.73, A100 80 GB SXM $1.67, RTX
    4090 24 GB $0.36
  - **Method**: Prices updated November 13, 2026. Table shows recent bids, weighted
    by GPUs each provider offers. B200 and Vera Rubin show "Request access"
  - **Bidding**: Providers bid on each deployment. Tenant sees all bids and picks
    one. Accepted bid sets the rate. Page does not use the word "auction"
  - **Reputation**: Bids show provider region and uptime stats. Tenant can limit a
    deployment to audited providers
  - **Terms**: Hourly billing, no contract, no egress or bandwidth fees. Reserved
    clusters are custom-quoted, and the FAQ mentions SLA-backed uptime

## Burn Mint Equilibrium On Akash (AEP-76)

- **Year**: 2026
- **Link**: [Akash AEP-76 proposal](https://akash.network/roadmap/aep-76/)
- **Key points**:
  - **Status**: Final, Economics category. Created 9/21/2025, last updated 3/23/2026.
    Author Greg Osuri
  - **Mechanism**: Tenants burn AKT to mint ACT, a non-transferable credit pegged to
    about $1. Providers receive AKT at settlement. USDC payouts are deprecated
  - **Fee**: 25 bps mint/settle spread at launch, adjustable from 0 to 25 bps
  - **Safeguards**: Circuit breakers at collateral ratio 0.95 (warn) and 0.90 (halt).
    Page also cites about 0.93 in one section
  - **Oracle**: Median of a 30-minute Osmosis AKT/USDC TWAP and an external feed.
    Outlier threshold 1.5% from the median

## Akash: 2025 Year in Review

- **Year**: 2026
- **Link**: [Akash 2025 recap](https://akash.network/blog/akash-2025-year-in-review/)
- **Key points**:
  - **Spend**: $3,154,814 total USD spent on network in 2025, up 128% from $1,384,554
    in 2024 (company-reported)
  - **Leases**: 126,025 new leases in 2025 versus 123,342 in 2024. Average fee per
    lease $25.03 versus $11.23
  - **GPUs**: 1,000+ GPUs deployed. About 60% average GPU utilization
    (company-reported)
  - **AkashML**: Launched November 2025. Priced at $2-4 per million tokens. Claims
    70-85% savings versus AWS SageMaker (company claim)
  - **Roadmap**: AEP-29 hardware verification estimated January 2026. AEP-65
    confidential computing estimated March 2026

## Akash Network Stock Price, Funding, Valuation, Revenue and Financial Statements

- **Year**: 2026
- **Link**: [CB Insights Akash](https://www.cbinsights.com/company/akash/financials)
- **Key points**:
  - **Funding**: $2M total across 3 rounds, with 6 investors
  - **Latest round**: Unattributed VC - II, February 1, 2024, Stateless Ventures
    listed. Amount hidden. Earlier round March 30, 2020 is paywalled
  - **Investors**: Visible names are Stateless Ventures, DACM, Zokyo, Genblock
    Capital, and Cypher Capital
  - **Gaps**: Valuation, revenue, headcount, and founding date are not disclosed

## Vast.ai

- **Year**: 2026
- **Link**: [Vast.ai homepage](https://vast.ai)
- **Key points**:
  - **Product**: GPU cloud marketplace for renting and hosting GPU compute
  - **Tools**: Offers API reference, CLI, and Python SDK, links live GPU pricing
    without stating prices

## GPU Pricing - Live Platform Rates | Vast.ai

- **Year**: 2026
- **Link**: [Vast.ai pricing page](https://vast.ai/pricing)
- **Key points**:
  - **Catalog**: 68+ GPU types from RTX 3060 to B200 across 40+ data centers. Prices
    set by supply and demand. Hourly rates are not in the page text
  - **On-demand**: Guaranteed uptime, no interruptions, per-second billing, start or
    stop anytime
  - **Interruptible**: Can be preempted. Priced "50%+ cheaper" with no baseline
    stated. Bidding mechanism is not explained
  - **Reserved**: Terms of 1, 3, or 6 months. "Up to 50% Off" with no baseline
    stated. Pricing requires contacting sales

## Press Kit - Vast.ai

- **Year**: 2026
- **Link**: [Vast.ai press kit](https://vast.ai/press-kit)
- **Key points**:
  - **Scale**: Company-reported 17,000+ GPUs, 350+ independent hosts and 40+ data
    centers on the platform
  - **Headcount**: Company-reported 40+ employees. Offices in Los Angeles and San
    Francisco. Founded June 28, 2016
  - **Offerings**: On-demand, interruptible and reserved GPU instances, serverless
    inference, dedicated clusters and Secure Cloud for enterprise
  - **Pricing**: Two-sided marketplace with prices set by supply and demand. The page
    gives no price, funding or revenue figures
  - **Certification**: SOC 2 Type I, announced April 10, 2025. Virtual machine
    support added December 10, 2024

## Pricing (Vast.ai documentation)

- **Year**: 2026
- **Link**: [Vast.ai pricing docs](https://docs.vast.ai/pricing)
- **Key points**:
  - **Billing**: Billed by the second for actual usage. Storage is billed even on
    stopped instances. Bandwidth is charged per byte
  - **Instance types**: On-demand (fixed price, high priority), reserved (pre-paid,
    discounted) and interruptible (lowest cost, low priority)
  - **Discounts**: Interruptible is often 50%+ cheaper than on-demand. Reserved
    offers up to 50% off with a commitment
  - **Price setting**: Hosts set their own prices. Rates move with supply and demand.
    Higher host reliability typically costs more
  - **Zero balance**: Instances stop and the GPU is released. Without a saved card,
    instances and data are scheduled for deletion

## Vast.ai Revenue 2025: $2.2M Est.

- **Year**: 2026
- **Link**: [GetLatka Vast.ai profile](https://getlatka.com/companies/vast.ai)
- **Key points**:
  - **Revenue**: GetLatka estimate of $2.2M annual revenue as of September 2025. The
    page says Vast.ai has not published its revenue
  - **Headcount**: 20 employees as of September 2025, per LinkedIn
  - **Company**: Page states founded 2018, based in Los Angeles, California. Industry
    listed as IaaS providers
  - **Peers**: GetLatka lists Crusoe at $290.7M revenue with 912 employees and
    FluidStack at $1.5M with 13 employees
  - **Gaps**: No funding, valuation, customer or pricing data on the page

## The AI Developer Cloud | Runpod

- **Year**: 2026
- **Link**: [Runpod homepage](https://runpod.io)
- **Key points**:
  - **Product**: AI cloud with Serverless GPU endpoints, Pods (Reserved or Spot), and
    multi-GPU Clusters
  - **Billing**: Serverless charges only for usage, no contracts or minimums, page
    lists no prices
  - **Scale**: Company-reported 1M+ developers and 30+ GPU SKUs across 31 regions,
    page updated November 2, 2026
  - **SLA**: Page shows 99.9% uptime, FAQ says enterprise agreements add a 99.99%
    uptime SLA
  - **Reputation**: G2 rating of 4.6, FAQ lists SOC 2 Type II, HIPAA, and GDPR
    compliance

## GPU Cloud Pricing | Per-Second H100, A100, RTX | Runpod

- **Year**: 2026
- **Link**: [Runpod pricing page](https://runpod.io/pricing)
- **Key points**:
  - **Billing**: Pods are billed per second with no commitment. The page claims
    pricing up to 90% below traditional clouds. Updated November 12, 2026
  - **H100 SXM**: Pod at $4.39/hr Secure Cloud and $2.69/hr Community Cloud.
    Serverless $5.89/hr. Cluster $4.79/GPU/hr
  - **Other GPUs**: A100 SXM pod $2.19/hr Secure and $1.39/hr Community. RTX 4090 pod
    $0.89/hr Secure and $0.34/hr Community
  - **Storage**: Starts at $0.05/GB/mo. Container disk $0.10/GB/mo. Idle volume disk
    $0.20/GB/mo
  - **Reserved**: Reserved clusters for 1 to 12+ months by contact sales. SLA-backed
    uptime is mentioned with no percentage

## Runpod Surpasses $325 Million in ARR, Appoints Brian Erickson as Chief Financial Officer

- **Year**: 2026
- **Link**: [Runpod press release on ARR and new CFO](https://runpod.io/press/runpod-325m-arr-new-cfo)
- **Key points**:
  - **ARR**: Company-reported ARR above $325M, up from $120M in January 2026. The
    press release dated October 13, 2026 calls this nearly 3x growth
  - **Users**: Company-reported 1.2M+ developers, up from about 495,000 a year
    earlier. Record 95,000+ signups in September 2026
  - **Funding**: Page lists a $100M raise led by Summit Partners on June 24, 2026 at
    a $1.0B valuation
  - **Compliance**: ISO 27001 certification and new governance controls added
    September 24, 2026
  - **Customers**: Named customers include Sony, Toyota, Intel, Columbia, Cornell and
    UCLA

## Runpod raises $100M to build the leading cloud platform for AI developers

- **Year**: 2026
- **Link**: [SiliconANGLE article on Runpod $100M raise](https://siliconangle.com/2026/06/25/runpod-raises-100m-build-leading-cloud-platform-ai-developers/)
- **Key points**:
  - **Funding**: $100M Series A led by Summit Partners, announced by Runpod in June
    2026
  - **Valuation**: $1 billion. The page does not attribute this figure to a source
  - **Total raised**: $122M across rounds. Seed in May 2024 co-led by Intel Capital
    and Dell Technologies Capital
  - **Usage**: Company-reported 1M+ developers. Serverless platform has served 20B+
    inference requests since launch
  - **Onboarding**: Company-reported median time from sign-up to running workload
    under an hour. 90%+ of deployments succeed on the first try

## Runpod Number of Employees 2026 | Employee Count and Headcount Data | Revelio Labs

- **Year**: 2026
- **Link**: [Revelio Labs Runpod headcount page](https://www.reveliolabs.com/companies/runpod/employees)
- **Key points**:
  - **Headcount**: 120 employees worldwide as of June 2026, up 34.8% year over year.
    Headquarters in San Francisco
  - **History**: Annual headcount of 34 (2023), 73 (2024), 90 (2025) and 120 (2026)
  - **Departments**: Engineering 52, Finance and Operations 48, Sales and Marketing
    21
  - **Salary**: Average salary $133,888 in 2026, down 2.0% year over year
  - **Caveat**: Page figures contain internal arithmetic inconsistencies. For
    example, department counts sum to 121, not 120

## io.net Open Source AI Infrastructure Platform homepage

- **Year**: 2026
- **Link**: [io.net homepage](https://io.net)
- **Not read**: Page body empty on two fetches, only page title returned

## TNE On Chain Overview (io.net docs)

- **Year**: 2026
- **Link**: [io.net TNE On Chain docs](https://io.net/docs/guides/explorer/tne-on-chain)
- **Key points**:
  - **Ledger**: TNE On Chain runs on Solana. Bookings, payments and $IO repurchases
    are recorded on-chain in an immutable ledger
  - **Escrow**: Funds are held in escrow by smart contracts. Payments and settlements
    are automated
  - **Audit**: Company claims all financial flows are publicly auditable. Each
    payment, refund or repurchase can be checked independently
  - **Repurchase**: A portion of every transaction buys $IO on the open market. All
    workloads are denominated in IO Coin terms
  - **Epochs**: Epoch Rewards Dashboard shows supplier rewards, active devices, burn
    amount and open debt. Earnings views are estimates only

## Solana-based DePIN io.net hits $1 billion token valuation in latest funding round batch

- **Year**: 2024
- **Link**: [The Block article on io.net funding](https://www.theblock.co/news/deals/2024-03-05-solana-depin-io-net-1-billion-token-valuation-funding-280665)
- **Key points**:
  - **Funding**: $30M Series A led by Hack VC, announced by IO Research
    (company-reported). Article dated March 5, 2024
  - **Valuation**: $1B fully diluted token valuation in the latest batch, per two
    anonymous sources. Most investors joined at $500M
  - **Structure**: SAFE with token warrants at a 1:1 ratio. CEO Ahmad Shadid said all
    investors face a minimum one-year token lock-in
  - **Scale**: Company-reported 25,000+ GPUs and 40,000+ compute hours processed.
    Launched in November 2023
  - **Verification**: Uses Solana for proof-of-compute. A Hack VC partner claims
    customers save up to 90% on cloud AI costs

## FAQ: How Much Does an H100 Cost Per Hour?

- **Year**: 2026
- **Link**: [io.net H100 hourly cost FAQ](https://io.net/p/fhow-much-does-an-h100-cost-per-hour)
- **Key points**:
  - **Price**: Company-reported on-demand H100 SXM5 at $2.20/hour and H100 PCIe at
    $1.49/hour. 8x SXM cluster $17.60/hour. Updated April 2026
  - **Comparison**: Unsourced competitor H100 SXM5 rates: AWS $6.98/hour, Azure
    $7.20/hour, CoreWeave $4.76/hour. Company claims 70% savings
  - **Fees**: Company claims platform fees of 10-20% versus a 300-500% markup on
    traditional clouds. Claims 75-85% utilization
  - **Billing**: Company claims per-second billing, 1-GPU minimum and scaling from 1
    to 100+ GPUs. Off-peak rates sometimes 10-15% lower
  - **Caveat**: Own CoreWeave figures show 54-56% savings, not 70%. Per-token costs
    are about 100x lower than the stated rates imply

## SF Compute

- **Year**: 2026
- **Link**: [SF Compute homepage](https://sfcompute.com)
- **Key points**:
  - **Product**: Data centers, GPU clusters, and cloud with VMs, bare metal, and
    managed Slurm
  - **Resale**: Customers reserve capacity and sublease idle compute to other
    tenants, also via `sf` CLI
  - **Price**: Example chart shows $3.00/GPU/hr reserved versus $4.29/GPU/hr
    realized, not stated if real data
  - **Market**: Careers section mentions building the order book, no auction or
    posted price named
  - **SLA**: Links an SLA page and SOC 2 Trust Center but states no uptime percentage

## SF Compute raised a $40M Series A | SF Compute

- **Year**: 2025
- **Link**: [SF Compute Series A announcement](https://sfcompute.com/news/sf-compute-series-a)
- **Key points**:
  - **Funding**: $40M Series A, announced November 26, 2025
  - **Valuation**: $300M
  - **Lead investors**: DCVC and Wing Venture Capital, jointly
  - **Returning investors**: Electric Capital and Alt Capital

## Sell unused compute time (SF Compute docs)

- **Year**: 2026
- **Link**: [SF Compute docs, reselling introduction](https://docs.sfcompute.com/preview/guides/reselling-introduction)
- **Key points**:
  - **Mechanism**: Sellers offer unused time on the SF Compute orderbook. They get
    credits only when a buyer purchases
  - **Price**: Seller sets `--min-rate` in dollars per node-hour. Example: 4 hours of
    1 node at $8 per node-hour returns $32 before fees
  - **Fills**: Partial fills allowed by default. Minimum fill is 1 node for 1 minute.
    The order that reaches the orderbook first sets the fill price
  - **Fees**: Platform fee has a fixed part per node-hour and a percentage part. The
    page gives no amounts. Fee cannot exceed the sale value
  - **Discount**: Example of $100,000 bought at $4.50 per GPU-hour with $25,000 in
    resale credits gives a 25% realized discount

## SF Compute eWeek funding article

- **Year**: 2026
- **Link**: [eWeek article on SF Compute funding](https://eweek.com/news/sf-compute-ai-funding/)
- **Not read**: HTTP 403 Forbidden

## SF Compute - Latent Space

- **Year**: 2026
- **Link**: [Latent Space wiki profile of SF Compute](https://latent.space/wiki/sf-compute-0739be6d/)
- **Key points**:
  - **Funding**: $12M led by Alt Capital, July 16, 2024. $40M Series A, reported
    December 2, 2025
  - **Contracts**: $245M in compute agreements with an unnamed AI company, September
    10, 2026. Two 36-month deals of $122.5M each for dedicated B300 clusters
  - **Resale**: 25% average realized discount through resale, September 9, 2026.
    Company-reported. Period and sample size not given
  - **Verification**: April 2025 checks included LINPACK burn-in, passive checks
    during customer workloads, and active checks during idle periods
  - **Cash**: In April 2025 the CEO said the company owed about $500,000 every 30
    days and had about $500,000 in the bank

## Compute Exchange - One Platform. Every GPU Market.

- **Year**: 2026
- **Link**: [Compute Exchange homepage](https://compute.exchange)
- **Key points**:
  - **Product**: GPU compute marketplace for used GPU hardware, reserved rental of 1
    to 36 months, and forward contracts
  - **Auction**: Live auction across verified providers returns quotes in under 2
    hours. Buyer signs agreement directly with chosen provider
  - **Vetting**: Each counterparty checked before quoting, covering registration,
    ownership, track record, capacity, SLA history, and contact authority
  - **Price**: No dollar prices or fees on page. Platform does not operate compute or
    take ownership of hardware
  - **Customers**: Logos include Character.ai, DRW, Higgsfield, and Cognichip. Page
    shows a SOC 2 Type II badge

## Payment and Fee Terms | Compute Exchange

- **Year**: 2025
- **Link**: [Compute Exchange payment and fee terms](https://compute.exchange/fees)
- **Key points**:
  - **Commission**: Providers pay 4% on each payment received. Volume discounts on
    request. Resellers also owe commission
  - **Late fee**: Payment due within 2 business days after a Bidding Period. Late
    payment triggers a reservation fee of 10% of total contract value
  - **Payment**: Contracts of 31 days or fewer are paid in full before delivery.
    Longer contracts pay 31 days plus a 10% Cancellation Fee Deposit
  - **Verification**: Approved Customer status is granted at the exchange's sole
    discretion. Non-approved customers must prefund their account
  - **Funds**: Held for the Provider through payment partners "FBO". Direct payments
    to Providers are a material breach

## Compute Exchange opens secondary market for used Nvidia H100 and A100 GPUs - SiliconANGLE

- **Year**: 2026
- **Link**: [SiliconANGLE on Compute Exchange used GPU market](https://siliconangle.com/2026/07/17/compute-exchange-opens-secondary-market-used-nvidia-h100-a100-gpus/)
- **Key points**:
  - **Launch**: July 17, 2026. Marketplace for used and refurbished Nvidia H100 and
    A100 GPUs
  - **Matching**: Buyers submit requests. The platform matches them with suppliers it
    calls qualified. It also handles sourcing and logistics
  - **Scale**: Requests range from "hundreds to tens of thousands of GPUs."
  - **Parent**: Reserved-capacity procurement marketplace with "more than 100
    providers."
  - **Pricing data**: Silicon Data supplies real-time GPU pricing benchmarks. CEO
    Carmen Li also founded Silicon Data

## Access the world's compute (Andromeda)

- **Year**: 2026
- **Link**: [Andromeda home page](https://andromeda.ai)
- **Key points**:
  - **Providers**: 100+ compute providers (company-reported)
  - **Transactions**: 1000+ completed (company-reported)
  - **Liquidity**: Claims "billions" of GPU/hour liquidity. No exact figure
  - **Certification**: Automated engine validates provider performance against
    enterprise-grade benchmarks
  - **Contracts**: SLAs and contracts are standardized. One invoice and one support
    channel. SOC 2 and ISO 27001 shown. No uptime figure given

## AI Compute Startup Andromeda Raises $60M At $1.5B Valuation

- **Year**: 2026
- **Link**: [Upstarts Media on Andromeda $60M raise](https://upstartsmedia.com/p/andromeda-ai-compute-startup-raises-60m)
- **Key points**:
  - **Funding**: Paradigm led a Series A-equivalent round at a $1.5B valuation.
    Paradigm invested $60M. Published March 18, 2026
  - **Revenue**: Run rate passed $100M in 2025, up from $50M-plus the year before.
    Operating profitably since its start
  - **Scale**: Tens of thousands of GPUs under management. Target of up to 100,000
    GPUs in 2026
  - **Headcount**: About 20 people
  - **Model**: Leases capacity from providers such as CoreWeave. Targets customers
    with $250M to $500M in annual compute spend

## Ornn | Financial Products for Compute

- **Year**: 2026
- **Link**: [Ornn home page](https://ornn.com)
- **Key points**:
  - **Contracts**: From one hour to multiple years. From a single GPU to a large
    cluster
  - **Index**: Ornn Compute Price Index updates hourly from settled trades. It covers
    realized rental prices for H100, H200, B200, and B300
  - **Token index**: Ornn Token Price Index tracks inference prices across leading
    model providers
  - **SLA**: One support team and SLA covers every cluster. No numeric SLA given
  - **Resale**: Customer Standard Intelligence says reselling unused reserved hours
    buys "an order of magnitude more compute."

## Ornn raises $33M to help companies buy and sell AI compute as a commodity like oil - SiliconANGLE

- **Year**: 2026
- **Link**: [SiliconANGLE on Ornn $33M raise](https://siliconangle.com/2026/06/24/ornn-raises-33m/)
- **Key points**:
  - **Funding**: $33M seed round, June 24, 2026. Galaxy Ventures co-led. Nordstar and
    SV Angel joined as new investors
  - **Index**: OCPI is a transaction-based compute index called a settlement-grade
    benchmark
  - **Futures**: Intercontinental Exchange will use the index for compute futures,
    subject to regulatory approval. Announced about May 2026
  - **Product**: Ornn Compute aggregates GPU capacity from public clouds and
    neoclouds. It has a secondary market for transfers and on-demand sublets
  - **Caveat**: A linked PR Newswire URL slug says "$5.7 million seed round." This
    conflicts with the $33M in the article body

## Investing in Ornn: A Market for Compute

- **Year**: 2026
- **Link**: [a16z Frontier post on Ornn](https://a16zfrontier.com/posts/article/investing-in-ornn)
- **Key points**:
  - **Funding**: a16z leads the $33 million Ornn seed round. Post dated June 24, 2026
  - **Index**: Ornn Compute Price Index (OCPI) is built from cleared GPU trades, not
    list prices. Live on the Bloomberg Terminal
  - **Tokens**: Ornn Token Price Indices (OTPI) benchmark the realized cost of
    inference tokens from leading model providers
  - **Hedging**: Futures let operators and enterprises hedge forward compute pricing
  - **Residual**: Residual value protection gives market-based GPU value estimates

## Prime Intellect - The Open Superintelligence Stack

- **Year**: 2026
- **Link**: [Prime Intellect homepage](https://primeintellect.ai)
- **Key points**:
  - **Product**: Platform to train, deploy, and improve custom models via Model
    Factory, Prime Inference, and Prime Compute
  - **Price**: On-demand GPUs billed hourly, no contract, H100 $3.50/hr, H200
    $5.48/hr, B300 $6.64/hr
  - **Resale**: Unused capacity can be resold as on-demand or spot compute on Prime
    network and reclaimed later
  - **Funding**: Backers listed as Founders Fund, Radical, NVIDIA, Intel, and Andrej
    Karpathy, no amounts or dates
  - **SLA**: No uptime promise, page describes self-healing clusters, 24/7 on-call
    support, and inference failover

## $15M to Build The Open Superintelligence Stack (Prime Intellect)

- **Year**: 2025
- **Link**: [Prime Intellect fundraise post](https://primeintellect.ai/blog/fundraise)
- **Key points**:
  - **Funding**: $15M led by Founders Fund, February 28, 2025. Total funding now
    exceeds $20M. Menlo Ventures and angels also joined
  - **Marketplace**: Compute Exchange is a distributed GPU marketplace. It links
    developers, startups, cloud providers, and datacenters
  - **Verification**: TOPLOC is a locality-sensitive hashing scheme for efficient
    verifiable inference
  - **Training**: PRIME framework trained a 10B-parameter model across three
    continents (INTELLECT-1)
  - **Headcount**: Careers page lists 38 open roles

## Prime Intellect raises $130M Series A to help enterprises build their own AI agents

- **Year**: 2026
- **Link**: [TechCrunch on Prime Intellect Series A](https://techcrunch.com/?p=3140126)
- **Key points**:
  - **Funding**: $130 million Series A at a $1 billion valuation, July 8, 2026. Led
    by Radical Ventures
  - **Investors**: Nvidia Ventures, Intel Capital, Dell Technologies Capital, and
    Iconiq also joined
  - **Revenue**: Annualized revenue run rate of $100 million, as reported by
    TechCrunch
  - **Customers**: Ramp, Zapier, and Flapping Airplanes. Customers pay for a hosted
    version of the tools
  - **Model**: Marketplace style. Customers pick individual tools. Founded 2024

## Prime Intellect Revenue 2026: $100M ARR, $1B Valuation

- **Year**: 2026
- **Link**: [Latka page on Prime Intellect](https://getlatka.com/companies/prime-intellect)
- **Key points**:
  - **Revenue**: ARR passed $100M in July 2026, citing LinkedIn
  - **Valuation**: $1B in July 2026, led by Radical Ventures. FAQ credits TechCrunch
  - **Headcount**: 40 employees as of July 2026, citing LinkedIn
  - **Gaps**: Page gives no round size, total raised, or customer count. Page dated
    November 10, 2026

## Bittensor - decentralized machine intelligence network

- **Year**: 2026
- **Link**: [Bittensor homepage](https://bittensor.com)
- **Key points**:
  - **Product**: Open network of independent subnets that produce digital
    commodities, with a Python SDK and the btcli tool
  - **Rewards**: Chain pays contributors in TAO. Page does not state reward rules,
    prices, or network counts

## Emissions (Bittensor docs)

- **Year**: 2026
- **Link**: [Bittensor docs on emissions](https://www.bittensor.com/docs/concepts/emissions)
- **Key points**:
  - **Supply**: Base emission 1 TAO per block, 12-second blocks, 21M TAO hard cap.
    First halving December 2025 cut it to 0.5 TAO per block (about 3,600 TAO per day)
  - **Split**: Default per-tempo split is 18% subnet owner, 41% miners, and 41%
    validators and stakers
  - **Subnet share**: Based on subnet EMA price over the sum of all EMA prices. Then
    burn adjustment and an emission gate apply
  - **Consensus**: Yuma Consensus uses a stake-weighted median. Default 128 validator
    permits (MaxAllowedValidators)
  - **Epoch**: Default tempo is 360 blocks, about 72 minutes

## Scoring Metrics and Weights (Chutes docs)

- **Year**: 2026
- **Link**: [Chutes docs on miner scoring](https://chutes.ai/docs/miner-resources/scoring)
- **Key points**:
  - **Weights**: Compute units 55%, invocation count 25%, unique chute score 15%,
    bounty count 5%
  - **Window**: Scoring interval is 7 days by default. Median step and token times
    use the last 2 days
  - **Anti-gaming**: Multi-UID punishment. Only the top hotkey per coldkey is
    rewarded. The rest get zero
  - **Filtering**: Only successful, completed invocations count. Invocations reported
    for issues are excluded
  - **Diversity**: Unique chute score uses exponent 1.3 above the median and 2.2
    below it

## The Bear Case for Bittensor (TAO)

- **Year**: 2026
- **Link**: [Pine Analytics bear case for Bittensor](https://pineanalytics.substack.com/p/the-bear-case-for-bittensor-tao)
- **Key points**:
  - **Valuation**: TAO about $275, $2.6B market cap, $5.8B fully diluted valuation.
    Post dated March 23, 2026
  - **Revenue**: Total identifiable external revenue about $3-15M per year. Author
    estimate, unaudited
  - **Chutes**: SN64 holds 14.4% of emissions. Reported 400,000+ users and 5M+ daily
    requests. Reported price 85% below AWS
  - **Subsidy**: Author calculates 22-40:1 emissions per dollar of Chutes customer
    revenue. Unsubsidized price about $1.41 per million tokens
  - **Taoflow**: Subnet emission share depends on net TAO staking inflows. Negative
    flows get zero emissions

## Bittensor revenue run rate climbs 40% in a month to an estimated $70M

- **Year**: 2026
- **Link**: [Crypto Briefing on Bittensor revenue run rate](https://cryptobriefing.com/bittensor-revenue-run-rate-70m/)
- **Key points**:
  - **Revenue**: About $70M annualized, up from about $50M in October. SubConnect
    report, November 4, 2026
  - **Projection**: SubConnect projects above $100M by year-end 2026. Baseline was
    $32-35M in late August
  - **Mix**: Compute and inference subnets are 78-82% of verified revenue. Leaders
    are Lium (SN51), Targon (SN4), Engy (SN53)
  - **Emissions**: Network emissions above $300M per year. External revenue covers
    9-23% of that
  - **Caveat**: Figures are estimates from off-chain payment records, not audited
    on-chain totals

## Bittensor Price: TAO/USD Live Price Chart, Market Cap and News Today | CoinGecko

- **Year**: 2026
- **Link**: [CoinGecko Bittensor (TAO) page](https://www.coingecko.com/en/coins/bittensor)
- **Key points**:
  - **Price**: $192.87, down 5.80% in 24 hours. 24-hour trading volume $211.571M.
    Page shows no calendar date
  - **Market cap**: $2.264B, rank #40. Circulating supply 11.73M of 21M max
  - **Range**: All-time high $757.60 and all-time low $30.83
  - **Treasuries**: 125,178 TAO held by listed companies. Largest is TAO Synergies
    with 54,058 TAO
  - **Caveat**: Some figures differ inside the page. The FAQ gives market cap
    $2,272,307,811

## Gensyn AI | machines that predict the future

- **Year**: 2026
- **Link**: [Gensyn homepage](https://gensyn.ai)
- **Key points**:
  - **Product**: Machine learning research and production lab with a mission to build
    machines that predict the future
  - **Releases**: Delphi information markets launched 22 Apr 2026. open-8b auditable
    model in training as of 9 Nov 2026

## Welcome (Gensyn developer docs)

- **Year**: 2026
- **Link**: [Gensyn developer documentation](https://docs.gensyn.ai)
- **Key points**:
  - **Verification**: Reproducible Execution Environment (REE) gives the same outputs
    for the same model and inputs. Each run yields a receipt that anyone can check
  - **AXL**: Agent eXchange Layer is encrypted peer-to-peer communication for AI
    agents. No central server. Supports MCP and A2A
  - **Delphi**: Information markets platform with a TypeScript SDK and an Agentic
    Trading Toolkit (ATT)
  - **Archive**: Testnet products RL Swarm, BlockAssist, and CodeAssist are legacy.
    No official swarms are running
  - **Gaps**: Page states no pricing, token details, or dates

## Blockchain-based, AI compute protocol Gensyn raises $43 million in a16z-led round

- **Year**: 2023
- **Link**: [Tech.eu on Gensyn $43M Series A](https://tech.eu/2023/06/12/blockchain-based-ai-compute-protocol-gensyn-raises-43-million-in-a16z-led-round/)
- **Key points**:
  - **Funding**: $43M Series A led by a16z Crypto, reported 12 June 2023
  - **Investors**: CoinFund, Canonical Crypto, Protocol Labs, Eden Block, and Maven
    11 also took part
  - **Verification**: A cryptographic verification system proves that submitted
    machine learning work was completed correctly
  - **Market**: Blockchain protocol matches buyers and sellers of compute. Developers
    pay on demand
  - **Claim**: Co-founder Harry Grieve says a decentralised device network cuts
    intermediary margins and lowers compute prices

## Gensyn Completes AI Token Public Sale with $16.14M Raised | KuCoin

- **Year**: 2025
- **Link**: [KuCoin flash news on Gensyn token sale](https://www.kucoin.com/news/flash/gensyn-completes-ai-token-public-sale-with-16-14m-raised)
- **Key points**:
  - **Raised**: $16.14M in the public token sale, ended December 21, 2025. KuCoin
    relays a Chainthink report
  - **Participants**: 7,412 participants in the sale
  - **Valuation**: $473M fully diluted valuation at the time of the sale
  - **Price**: $0.0473 per token
  - **Date**: Gensyn Foundation to publish token allocation details on December 25

## Ritual | Ritual lets an AI be a company of its own

- **Year**: 2026
- **Link**: [Ritual homepage](https://ritual.net)
- **Key points**:
  - **Product**: Infrastructure for autonomous AI agents that run in TEEs and sign
    their own payments
  - **Compute**: Agents rent hardware from independent providers and pay from own
    wallets while funds last
  - **Price**: Execution paid in RITUAL token, page gives no rates or pricing
    mechanism

## Ritual Chain Documentation

- **Year**: 2026
- **Link**: [Ritual Chain developer documentation](https://docs.ritualfoundation.org/)
- **Key points**:
  - **Design**: EVM chain with off-chain verifiable machine tasks run in TEEs.
    Results are cryptographically bound to their requests (page claim)
  - **Pricing**: Text model input costs 1.0792 RITUAL per 1M tokens. Output costs
    2.1584 per 1M. Cached input costs 0.1079 per 1M
  - **Precompiles**: Page lists 15 precompiles, 4 synchronous and 11 async. They
    include LLM, Image, Audio, Video, and agent calls
  - **Model**: Text inference uses DeepSeek-V4-Flash-0731 with a 1M-token context
  - **Failures**: Failed async jobs settle onchain with an error. There is no
    automatic retry. A failed LLM call carries a fixed 0.01 fee

## Ritual closes $25 million fundraise (The Block, 2023-11-08)

- **Year**: 2023
- **Link**: [The Block article on Ritual $25M fundraise](https://www.theblock.co/news/deals/2023-11-08-decentralized-ai-compute-platform-ritual-closes-25-million-fundraise)
- **Not read**: HTTP 404

## Ritual Mainnet Launches: RITUAL Tokenomics, Supply, and Airdrop Details

- **Year**: 2026
- **Link**: [KuCoin blog on Ritual mainnet and tokenomics](https://www.kucoin.com/blog/ritual-mainnet-ritual-tokenomics-airdrop)
- **Key points**:
  - **Mainnet**: Public mainnet launched October 26, 2026. Testnet had over 2,000
    deployed agents and over 90 million transactions
  - **Supply**: 10 billion RITUAL initial supply, no fixed maximum. 1.8 billion (18%)
    unlocked at launch
  - **Allocation**: Core Contributors 29%, Ecosystem Growth and R&D 25%, Labs
    Investors 11.30%. Insider tokens locked at least one year
  - **Airdrop**: 50 million RITUAL (0.5% of supply) for Infernet node operators,
    Frenrug users, and community members
  - **Issuance**: About 5% annualized initial issuance to validators, AI executors,
    and delegators. EIP-1559-style base fee is burned

## Affordable GPU Rental at Scale - On-Demand GPU Compute Cloud | Nosana

- **Year**: 2026
- **Link**: [Nosana homepage](https://nosana.com)
- **Key points**:
  - **Product**: Open-source GPU cloud that rents GPUs on demand and lets hardware
    owners earn by sharing GPUs
  - **Price**: Hourly rates listed per GPU model, e.g., NVIDIA 4090 at $0.400/h and
    H100 at $1.500/h, undated snapshot
  - **Supply**: Host counts shown per GPU model, e.g., 4090 with 0 of 98 hosts
    available, undated. Pricing method not stated
  - **Token**: NOS market cap $42M+, 14,132+ active stakers, NOS price $0.42,
    company-reported, undated
  - **Savings**: Claims savings of up to 6x depending on workload, with no baseline
    or method given

## Nosana Documentation

- **Year**: 2026
- **Link**: [Nosana Docs home page](https://learn.nosana.com/)
- **Key points**:
  - **Deployments**: Users create deployments through an API or dashboard. They pay
    with credits
  - **Inference**: OpenAI-compatible API for open models on Nosana GPUs. Billed per
    token against credits
  - **Hosting**: GPU owners earn $NOS by joining the Nosana Grid and offering GPUs on
    the marketplace
  - **Programs**: Solana smart contracts cover staking, jobs, nodes, pools, and
    rewards. Page gives no GPU prices or network size

## From Yield to Growth: Aligning NOS Rewards with Real Usage (Nosana blog)

- **Year**: 2025
- **Link**: [Nosana blog on NOS reward changes](https://nosana.com/blog/from-yield-to-growth-aligning-nos-rewards-with-real-usage)
- **Key points**:
  - **Rewards**: Proposal cuts passive rewards. Rewards go to hosts that meet
    reliability thresholds and to customers who keep using the network
  - **Collateral**: A more powerful GPU requires more NOS collateral from the host.
    No amounts are given
  - **Scheduling**: Uptime, latency, and job completion set scheduling and reward
    weight. Poor performers lose weight or become ineligible
  - **Rebates**: Customers would get deployment rebates for repeat usage. Size not
    stated
  - **Governance**: Spec NNP-0001 posted October 2025. On-chain vote planned for
    November. Changes need approval

## Fetch.ai - Search and Discover

- **Year**: 2026
- **Link**: [Fetch.ai homepage](https://fetch.ai)
- **Key points**:
  - **Product**: Personal AI assistant that assigns task parts to brand agents, plus
    Agentverse agent marketplace and ASI:One
  - **Scale**: Agentverse open directory holds 2.7 million agents, company-reported,
    no date given
  - **Verification**: Brand agents built from a business profile get verified. Page
    does not state criteria or who verifies
  - **Price**: Page states no prices, fees, or subscription tiers

## Artificial Superintelligence Alliance (FET) Price and Market Cap | CoinGecko

- **Year**: 2026
- **Link**: [CoinGecko page for Artificial Superintelligence Alliance](https://www.coingecko.com/en/coins/artificial-superintelligence-alliance)
- **Key points**:
  - **Price**: $0.1335 per FET. 24h range $0.1220 to $0.1383. Page shows no price
    timestamp
  - **Market cap**: $308.695M, rank #124. FDV $362.497M. 24h volume $86.972M. Some
    page sections show slightly different figures
  - **Supply**: Circulating supply 2.312B FET. Total and max supply 2.714B
  - **Alliance**: Fetch.ai and SingularityNET merged tokens into FET. Page names no
    other members
  - **Unlock**: Next unlock November 28 releases 2.4M FET for AGIX migration

## Blockchain startup Fetch.ai grabs $40M to provide monetization and other tooling

- **Year**: 2023
- **Link**: [TechCrunch on Fetch.ai $40M from DWF Labs](https://techcrunch.com/2023/03/29/blockchain-startup-fetch-ai-grabs-40m-to-provide-monetization-and-other-tooling-for-ai-generated-information)
- **Key points**:
  - **Funding**: $40M from a single investor, DWF Labs, reported March 29, 2023
  - **Use**: Build the platform ahead of planned commercial launches later in 2023
  - **Company**: Based in Cambridge, England. CEO Humayun Sheikh. Focus on autonomous
    agents, network infrastructure, and decentralised machine learning
  - **Token**: FET token used on the platform. Plan turns generative AI outputs into
    purchases paid in FET
  - **Training**: Planned multi-party model training uses ledgers to trace data
    contributions and share ownership

## Fetch.AI Revenue 2025: $37.4M Est.

- **Year**: 2026
- **Link**: [Latka profile of Fetch.AI](https://getlatka.com/companies/fetch.ai)
- **Key points**:
  - **Revenue**: $37.4M as of Sep 2025. Third-party estimate from Growjo, not
    company-reported
  - **Disclosure**: Page says Fetch.AI has not published its revenue
  - **Headcount**: 92 employees as of Sep 2025, per LinkedIn
  - **Founded**: 2017, headquartered in Cambridge, United Kingdom

## x402

- **Year**: 2026
- **Link**: [x402 home page](https://www.x402.org)
- **Key points**:
  - **Fees**: Page claims no protocol fees for customers or merchants. Only "nominal"
    payment network fees apply
  - **Mechanism**: Server answers an unpaid request with HTTP 402. Client pays in
    stablecoins and retries. No accounts or API keys
  - **Settlement**: Page claims instant settlement with no waiting period
  - **Networks**: All EVM-compatible chains and Solana. Stablecoins are the primary
    use case
  - **Usage**: Page claims "millions of transactions" processed. Logos include AWS,
    Cloudflare, Stripe, Vercel

## Linux Foundation Announces Operational Launch of x402 Foundation to Standardize Internet-Native Payments for AI Agents and Applications

- **Year**: 2026
- **Link**: [Linux Foundation press release on the x402 Foundation](https://www.linuxfoundation.org/press/linux-foundation-announces-operational-launch-of-x402-foundation-to-standardize-internet-native-payments-for-ai-agents-and-applications)
- **Key points**:
  - **Date**: Operational launch announced July 14, 2026. Intent was first announced
    in April
  - **Contribution**: Coinbase completed its contribution of the x402 protocol to the
    Foundation
  - **Members**: 40 organizations joined, including 17 premier members such as
    Google, Mastercard, Visa, Stripe, AWS
  - **Governance**: Vendor-neutral open governance under the Linux Foundation, to
    avoid vendor lock-in
  - **Settlement**: Circle states (company quote) that x402 with USDC settles in
    seconds at a fraction of a cent

## Overview (x402, Coinbase Developer Platform docs)

- **Year**: 2026
- **Link**: [Coinbase CDP x402 overview](https://docs.cdp.coinbase.com/x402/welcome)
- **Key points**:
  - **Volume**: CDP reports more than 100 million x402 payments across Base and
    Solana
  - **Facilitator**: The CDP Facilitator verifies and settles payments
  - **Sellers**: Sellers receive funds at any address they control, including CDP
    wallets, Coinbase Business, Prime, or self-custody
  - **Buyers**: Human or machine buyers pay from a CDP wallet or another signer. No
    checkout page or signup flow
  - **Gaps**: Page states no pricing, fees, settlement time, or SLAs. It links to
    other pages for these

## Powering AI commerce with the new Agent Payments Protocol (AP2)

- **Year**: 2025
- **Link**: [Google Cloud blog, AP2 announcement](https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol)
- **Key points**:
  - **Partners**: Google reports work with more than 60 organizations, including
    Mastercard, PayPal, American Express, Coinbase
  - **Mandates**: Intent Mandate and Cart Mandate are cryptographically signed
    contracts, signed by verifiable credentials
  - **Accountability**: Intent, cart, and payment chain forms a "non-repudiable audit
    trail." Page describes no dispute or chargeback process
  - **Payments**: Supports cards, stablecoins, and real-time bank transfers. Extends
    the A2A and MCP protocols
  - **Date**: Announced September 17, 2025. An A2A x402 extension launched with
    Coinbase for crypto payments
