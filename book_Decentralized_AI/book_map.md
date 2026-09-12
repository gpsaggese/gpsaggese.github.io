# Summary

## Title
- Decentralized AI, DeFi, and Web3: Foundations, Protocols, and Practice
- Trustless by Design: A Technical Guide to Blockchain, Decentralized Finance,
  and Decentralized AI

## Target Audience
- Grad CS/ML students and engineers with a distributed-systems and ML
  background, no prior blockchain experience assumed
- Blockchain and DeFi practitioners who want a rigorous mental model for how
  AI and decentralized systems intersect
- Assumes comfort with Python and basic probability, no prior Solidity or
  tokenomics knowledge

## Approach of the Book
- Focus on:
  - Minimal math, emphasis on intuition and toy examples
  - Making the theory operational via packages (`web3.py`, federated-learning
    frameworks, zk libraries)
  - Jupyter notebooks backing up intuition with toy and real protocol examples
- Provide resources to go one level deeper:
  - DATA605 (`Lesson_93_Sorrentum_Project.txt`), MSML610 (federated and
    multi-agent ML paradigms), Berkeley's Responsible GenAI course
  - Reference books: Mastering Bitcoin, Mastering Ethereum (Antonopoulos),
    DeFi and the Future of Finance (Harvey), Token Economy (Voshmgir), see
    `resources.md`

## Short TOC
- The sequence of the parts in the book is:
  - Foundations and Infrastructure
    - 01, Blockchain Consensus Foundations
    - 02, Distributed Ledger Fundamentals
    - 03, Smart Contract Platforms and Security
    - 04, Layer 2 and Scaling Architectures
    - 05, Cross-Chain Interoperability and Bridges
    - 06, Cryptographic Primitives for Web3
    - 07, Web3 Infrastructure and Node Economics
  - Decentralized AI
    - 08, Federated Learning
    - 09, Decentralized Training
    - 10, Decentralized Compute and Inference Markets
    - 11, Blockchain-AI Integration
    - 12, Zero-Knowledge ML (ZKML)
    - 13, Differential Privacy
    - 14, Secure Multiparty Computation and Homomorphic Encryption
    - 15, Model and Data Provenance
    - 16, Decentralized Data Marketplaces
    - 17, Multi-Agent Systems and Mechanism Design
    - 18, Open-Source vs Closed-Source Ecosystems
  - Decentralized Finance
    - 19, Automated Market Makers (AMMs)
    - 20, Decentralized Exchanges and Order Flow
    - 21, Lending and Borrowing Protocols
    - 22, Stablecoins and Monetary Design
    - 23, Derivatives and Synthetic Assets
    - 24, MEV and Transaction Ordering
    - 25, Oracle Design and Data Integrity
  - Web3 and Digital Ownership
    - 26, Decentralized Identity and Credentials
    - 27, NFTs and Digital Ownership
    - 28, Decentralized Storage and Data Availability
    - 29, Privacy-Preserving Web3
  - Cross-Cutting: Governance, Incentives, Security, Regulation
    - 30, Governance and DAOs
    - 31, Tokenomics and Incentive Design
    - 32, Robustness and Security in AI Networks
    - 33, Security and Auditing
    - 34, Risk Management and Systemic Risk
    - 35, Regulation

## All Lesson Materials
- `data605/all_tocs.md`
- `data605/lectures_source/*.txt`

- `msml610/all_tocs.md`
- `msml610/lectures_source/*.txt`

## Chapter Templates and Invariants
- Follow `.claude/skills/book.rules.md` for the Chapter Template (Goals,
  Topics, TODO, Slides, Lesson Materials, Notes) and Roadmap section
  conventions used throughout this file

# Roadmap

| Chap                                                  | Slides % | Criticize | Tutorial | Book |
| ------------------------------------------------------| ---------| ----------| ---------| -----|
|                                                       |          |           |          |      |
| **Foundations and Infrastructure**                    |          |           |          |      |
| 01. Blockchain Consensus Foundations                  | 100% |           |          |      |
| 02. Distributed Ledger Fundamentals                   |            |          |           |          |
| 03. Smart Contract Platforms and Security             |            |          |           |          |
| 04. Layer 2 and Scaling Architectures                 |            |          |           |          |
| 05. Cross-Chain Interoperability and Bridges          |            |          |           |          |
| 06. Cryptographic Primitives for Web3                 |            |          |           |          |
| 07. Web3 Infrastructure and Node Economics            |            |          |           |          |
| **Decentralized AI**                                  |            |          |           |          |
| 08. Federated Learning                                |            |          |           |          |
| 09. Decentralized Training                            |            |          |           |          |
| 10. Decentralized Compute and Inference Markets       |            |          |           |          |
| 11. Blockchain-AI Integration                         |            |          |           |          |
| 12. Zero-Knowledge ML (ZKML)                          |            |          |           |          |
| 13. Differential Privacy                              |            |          |           |          |
| 14. Secure Multiparty Computation and HE              |            |          |           |          |
| 15. Model and Data Provenance                         |            |          |           |          |
| 16. Decentralized Data Marketplaces                   |            |          |           |          |
| 17. Multi-Agent Systems and Mechanism Design          |            |          |           |          |
| 18. Open-Source vs Closed-Source Ecosystems           |            |          |           |          |
| **Decentralized Finance**                             |            |          |           |          |
| 19. Automated Market Makers (AMMs)                    |            |          |           |          |
| 20. Decentralized Exchanges and Order Flow            |            |          |           |          |
| 21. Lending and Borrowing Protocols                   |            |          |           |          |
| 22. Stablecoins and Monetary Design                   |            |          |           |          |
| 23. Derivatives and Synthetic Assets                  |            |          |           |          |
| 24. MEV and Transaction Ordering                      |            |          |           |          |
| 25. Oracle Design and Data Integrity                  |            |          |           |          |
| **Web3 and Digital Ownership**                        |            |          |           |          |
| 26. Decentralized Identity and Credentials            |            |          |           |          |
| 27. NFTs and Digital Ownership                        |            |          |           |          |
| 28. Decentralized Storage and Data Availability       |            |          |           |          |
| 29. Privacy-Preserving Web3                           |            |          |           |          |
| **Cross-Cutting: Governance, Incentives, Security, Regulation** | |          |           |          |
| 30. Governance and DAOs                               |            |          |           |          |
| 31. Tokenomics and Incentive Design                   |            |          |           |          |
| 32. Robustness and Security in AI Networks            |            |          |           |          |
| 33. Security and Auditing                             |            |          |           |          |
| 34. Risk Management and Systemic Risk                 |            |          |           |          |
| 35. Regulation                                        |            |          |           |          |

## TODOs
- Build `book_Decentralized_AI/lectures_source/` slide decks per chapter
- Create `book_Decentralized_AI/tutorials/` notebooks once slide decks exist
- Incorporate the high-priority reading list from `resources.md` (Harvey's
  DeFi course and book, Voshmgir's Token Economy) into the relevant chapters
- Re-run the coverage check in `### Lesson Materials` whenever new lecture
  files land in `data605/lectures_source/` or `msml610/lectures_source/`

# Detailed TOC

# Part I: Foundations and Infrastructure

## 01: Blockchain Consensus Foundations

### Goals
- Explain how independent nodes agree on one ledger without a trusted party
- Compare UTXO and account-based ledger models and their tradeoffs
- Contrast proof-of-work, proof-of-stake, and BFT consensus on safety/liveness

### Topics
- Ledger and State Models
  - UTXO vs account models
  - Finality, chain reorganizations
- Consensus Algorithms
  - Proof-of-work vs proof-of-stake
  - BFT-style consensus (Tendermint, HotStuff)
  - Liveness vs safety tradeoffs

### Slides
- `book_Decentralized_AI/lectures_source/Lesson01.01_Blockchain_Consensus_Foundations.txt`

### Lesson Materials
- `data605/lectures_source/Lesson_93_Sorrentum_Project.txt`
  - [35%]: Consensus intuition (replicated, shared, synchronized data),
    proof-of-work vs proof-of-stake, permissionless vs permissioned, 51%
    attack bound, Bitcoin/Ethereum/Solana/Cardano as examples
- `IN_PROGRESS.course.cs.Decentralized_finance.Harvey.Coursera.txt` (see
  `resources.md`)
  - [20%]: UTXO model with worked example, general real-world consensus
    definition, PoW mechanics (mempool, Merkle tree, nonce, 51% attack, ASIC
    centralization), PoS mechanics and PoS-vs-PoW tradeoffs, named list of
    other consensus mechanisms (delegated PoS, BFT, proof of
    authority/capacity/identity/activity)
- `IN_PROGRESS.crypto.DeFi_and_the_future_of_finance.Harvey.2021.txt` (see
  `resources.md`)
  - [5%]: Brief PoW/PoS definitions, longest-chain rule
- _Not covered_
  - [40%]: Account-based ledger model (vs UTXO), finality and chain
    reorganizations, BFT-style consensus mechanics (Tendermint/HotStuff),
    formal liveness vs safety tradeoffs

## 02: Distributed Ledger Fundamentals

### Goals
- Show how Merkle trees and hash chains make a ledger tamper-evident
- Model a blockchain as a replicated state machine
- Define Byzantine fault tolerance and why it bounds decentralized trust

### Topics
- Data Structures for Tamper-Evidence
  - Merkle trees
  - Hash chains
- Ledger as a Replicated State Machine
  - State machines
  - Byzantine fault tolerance

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Merkle trees, hash chains, state machines, Byzantine fault
    tolerance

## 03: Smart Contract Platforms and Security

### Goals
- Explain how smart contracts execute deterministically on EVM/WASM chains
- Identify reentrancy and oracle manipulation as dominant exploit classes
- Introduce formal verification as a defense against contract bugs

### Topics
- Execution Models
  - EVM execution model, WASM-based chains
  - Gas models, account abstraction
- Common Vulnerability Classes
  - Reentrancy
  - Oracle manipulation
- Verifying Correctness
  - Formal verification

### Slides
- N/A

### Lesson Materials
- `data605/lectures_source/Lesson_93_Sorrentum_Project.txt`
  - [25%]: Smart contract definition, trust models (law, social custom, math
    and code), Bitcoin as special-purpose vs Ethereum as general-purpose
    Turing-complete (EVM)
- _Not covered_
  - [75%]: WASM-based chains, gas models, account abstraction, reentrancy,
    oracle manipulation, formal verification

## 04: Layer 2 and Scaling Architectures

### Goals
- Compare rollups, sidechains, and state channels as scaling strategies
- Explain sharding as horizontal scaling of a blockchain's state
- Discuss how fee markets and composability change across layers

### Topics
- Rollups and Off-Chain Execution
  - Optimistic vs zk-rollups
  - Sidechains, state channels
- Scaling and Cross-Layer Economics
  - Sharding
  - Fee markets, composability across layers

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Optimistic and zk-rollups, sidechains, state channels, sharding,
    fee markets, composability across layers

## 05: Cross-Chain Interoperability and Bridges

### Goals
- Classify bridge designs by their trust model and failure modes
- Explain light clients and IBC-style protocols as trust-minimized messaging
- Show atomic swaps and rollup-native messaging as alternatives to bridges

### Topics
- Bridging Assets Across Chains
  - Bridge trust models, wrapped assets
  - Bridge exploits
- Trust-Minimized Messaging
  - Light clients, IBC-style protocols
  - Atomic swaps
  - Rollup-native messaging

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Bridge trust models, wrapped assets, bridge exploits, light
    clients, IBC-style protocols, atomic swaps, rollup-native messaging

## 06: Cryptographic Primitives for Web3

### Goals
- Explain digital signatures and hash functions as trust building blocks
- Introduce zero-knowledge proofs as proving a statement without revealing it
- Show threshold signatures as a way to distribute signing authority

### Topics
- Core Primitives
  - Digital signatures, hash functions
- Advanced Primitives
  - Zero-knowledge proofs
  - Threshold signatures

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Digital signatures, hash functions, zero-knowledge proofs,
    threshold signatures

## 07: Web3 Infrastructure and Node Economics

### Goals
- Explain how RPC providers and indexers make on-chain data usable
- Describe validator economics and how staking secures a network
- Connect node infrastructure to the cost of running decentralized apps

### Topics
- Accessing Chain Data
  - RPC providers
  - Indexers (The Graph)
- Running the Network
  - Validator and staking economics

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: RPC providers, indexers (The Graph), validator and staking
    economics

# Part II: Decentralized AI

## 08: Federated Learning

### Goals
- Explain how FedAvg trains one model across devices without centralizing data
- Show why non-IID data forces personalization strategies
- Quantify communication overhead as federated learning's core bottleneck

### Topics
- Aggregating Across Devices
  - FedAvg aggregation
  - Communication overhead
- Handling Heterogeneous Data
  - Personalization, non-IID data

### Slides
- N/A

### Lesson Materials
- `msml610/lectures_source/Lesson02.2-ML_Paradigms.txt`
  - [10%]: Federated learning definition (train across decentralized devices
    without sharing raw data), fraud/credit-scoring example
- _Not covered_
  - [90%]: FedAvg aggregation, personalization and non-IID data,
    communication overhead

## 09: Decentralized Training

### Goals
- Contrast data and model parallelism when nodes do not trust each other
- Explain communication-efficient SGD variants for decentralized training
- Show how bandwidth limits shape decentralized training architecture

### Topics
- Parallelism Across Untrusted Nodes
  - Data parallelism
  - Model parallelism
- Making Training Bandwidth-Efficient
  - Communication-efficient SGD
  - Bandwidth limits

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Data/model parallelism across untrusted nodes,
    communication-efficient SGD, bandwidth limits

## 10: Decentralized Compute and Inference Markets

### Goals
- Survey Bittensor, Gensyn, and Together as decentralized compute markets
- Explain how these networks incentivize honest compute contribution
- Compare training markets to inference markets

### Topics
- Compute-Sharing Networks
  - Bittensor, Gensyn, Together
- Incentive Design
  - Rewarding useful compute contribution

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Bittensor, Gensyn, Together, incentive design for
    compute-sharing networks

### Notes
- `data605/lectures_source/Lesson_93_Sorrentum_Project.txt` describes a
  decentralized financial-ML framework, not a compute marketplace like
  Bittensor/Gensyn; excluded from Lesson Materials as insufficiently on-topic

## 11: Blockchain-AI Integration

### Goals
- Explain how on-chain agents trigger and consume smart contract logic
- Show how smart contracts can trigger off-chain ML inference
- Describe oracles as the bridge for feeding ML outputs on-chain

### Topics
- Agents Acting On-Chain
  - On-chain agents
  - Smart-contract-triggered inference
- Bringing ML Outputs On-Chain
  - Oracles feeding ML outputs on-chain

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: On-chain agents, smart-contract-triggered inference, oracles
    feeding ML outputs on-chain

## 12: Zero-Knowledge ML (ZKML)

### Goals
- Explain how zk-SNARKs let a verifier trust inference without rerunning it
- Introduce proof-of-training as a verifiable claim about a model's origin
- Frame ZKML as the path to trustless model correctness

### Topics
- Proving Inference Ran Correctly
  - zk-SNARKs for verifiable inference
- Proving Training Happened Correctly
  - Proof-of-training
  - Trustless model correctness

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: zk-SNARKs for verifiable inference, proof-of-training, trustless
    model correctness

## 13: Differential Privacy

### Goals
- Explain DP-SGD as gradient training with a formal privacy guarantee
- Define privacy budget and how it accumulates over training
- Show the accuracy cost of privacy in decentralized settings

### Topics
- Training with Privacy Guarantees
  - DP-SGD
- Managing the Privacy Budget
  - Privacy budgets
  - Accuracy tradeoffs in decentralized settings

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: DP-SGD, privacy budgets, accuracy tradeoffs in decentralized
    settings

## 14: Secure Multiparty Computation and Homomorphic Encryption

### Goals
- Explain secure multiparty computation as joint computation without sharing
- Explain homomorphic encryption as computation directly on ciphertext
- Show how both enable privacy-preserving collaborative training/inference

### Topics
- Multiparty Computation
  - Computing jointly without revealing private inputs
- Homomorphic Encryption
  - Computing directly on encrypted data
- Applications
  - Privacy-preserving collaborative training and inference

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Secure multiparty computation, homomorphic encryption,
    privacy-preserving collaborative training and inference

## 15: Model and Data Provenance

### Goals
- Explain watermarking as a way to mark and later identify a model's origin
- Show dataset lineage tracking as a defense against unverifiable data
- Describe tamper-evident training records as an audit trail for ML pipelines

### Topics
- Marking and Tracing Models
  - Watermarking
- Tracing Data and Training
  - Dataset lineage
  - Tamper-evident training records

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Watermarking, dataset lineage, tamper-evident training records

## 16: Decentralized Data Marketplaces

### Goals
- Explain Data DAOs as a governance structure for pooled datasets
- Show how tokenizing datasets creates a market for data contribution
- Design incentive-compatible mechanisms for sharing data fairly

### Topics
- Organizing Around Data
  - Data DAOs
- Pricing and Sharing Data
  - Tokenized datasets
  - Incentive-compatible data sharing

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Data DAOs, tokenized datasets, incentive-compatible data sharing

## 17: Multi-Agent Systems and Mechanism Design

### Goals
- Contrast cooperative and competitive multi-agent learning settings
- Explain incentive alignment as the core problem in mechanism design
- Show self-play economics as a driver of emergent agent strategies

### Topics
- Learning in Shared Environments
  - Cooperative and competitive learning
- Aligning Incentives
  - Incentive alignment
  - Self-play economics

### Slides
- N/A

### Lesson Materials
- `msml610/lectures_source/Lesson02.2-ML_Paradigms.txt`
  - [20%]: Multi-agent learning definition (agents learning and interacting
    in shared, game-theoretic environments), AlphaStar self-play example
- _Not covered_
  - [80%]: Formal incentive alignment and mechanism design, self-play
    economics beyond the AlphaStar example

## 18: Open-Source vs Closed-Source Ecosystems

### Goals
- Compare open-source and closed-source model licensing and trust tradeoffs
- Introduce DecodingTrust-style evaluation of model trustworthiness
- Explain why reproducibility differs across open and closed ecosystems

### Topics
- Licensing and Trust
  - Licensing models
  - Trust implications of open vs closed weights
- Evaluating and Reproducing Results
  - Evaluation frameworks (e.g., DecodingTrust)
  - Reproducibility

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Licensing, trust, evaluation (e.g., DecodingTrust),
    reproducibility

# Part III: Decentralized Finance

## 19: Automated Market Makers (AMMs)

### Goals
- Explain how constant-product and constant-sum curves price trades
- Show how concentrated liquidity improves capital efficiency
- Quantify impermanent loss as the cost of providing liquidity

### Topics
- Pricing Curves
  - Constant-product and constant-sum curves
- Liquidity Provision
  - Concentrated liquidity
  - Impermanent loss

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Constant-product/constant-sum curves, concentrated liquidity,
    impermanent loss

## 20: Decentralized Exchanges and Order Flow

### Goals
- Compare order-book DEXs and aggregators as trade-matching mechanisms
- Explain MEV-aware routing as a defense for traders
- Define slippage and how routing choices affect it

### Topics
- Matching Trades On-Chain
  - Order-book DEXs, aggregators
- Routing and Execution Quality
  - MEV-aware routing
  - Slippage

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Order-book DEXs, aggregators, MEV-aware routing, slippage

## 21: Lending and Borrowing Protocols

### Goals
- Explain overcollateralization as DeFi lending's base safety mechanism
- Show how interest-rate models balance borrower and lender incentives
- Walk through liquidation mechanics and flash loans as edge-case risks

### Topics
- Collateral and Rates
  - Overcollateralization
  - Interest-rate models
- Risk Mechanics
  - Liquidation mechanics
  - Flash loans

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Overcollateralization, interest-rate models, liquidation
    mechanics, flash loans

## 22: Stablecoins and Monetary Design

### Goals
- Compare fiat-collateralized, crypto-backed, and algorithmic stablecoins
- Explain what keeps a peg stable and what makes it fragile
- Use the Terra/UST collapse as a case study in depeg risk

### Topics
- Peg Mechanisms
  - Fiat-collateralized, crypto-backed, algorithmic pegs
- When Pegs Break
  - Depeg risk (Terra/UST case study)

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Fiat-collateralized/crypto-backed/algorithmic pegs, depeg risk
    (Terra/UST case study)

## 23: Derivatives and Synthetic Assets

### Goals
- Explain perpetual futures and options protocols as on-chain derivatives
- Show how synthetic assets replicate exposure without holding the underlying
- Describe funding rates as the mechanism anchoring perpetuals to spot

### Topics
- On-Chain Derivatives
  - Perpetual futures
  - Options protocols
- Synthetic Exposure and Funding
  - Synthetic exposure
  - Funding rates

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Perpetual futures, options protocols, synthetic exposure,
    funding rates

## 24: MEV and Transaction Ordering

### Goals
- Explain sandwich attacks and front-running as MEV extraction strategies
- Show how proposer-builder separation changes who can extract MEV
- Discuss MEV redistribution as a fairness mechanism

### Topics
- Extracting Value from Ordering
  - Sandwich attacks
  - Front-running
- Mitigating MEV
  - Proposer-builder separation (PBS)
  - MEV redistribution

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Sandwich attacks, front-running, PBS (proposer-builder
    separation), MEV redistribution

## 25: Oracle Design and Data Integrity

### Goals
- Explain how Chainlink-style feeds and TWAPs bring off-chain data on-chain
- Show how oracle manipulation attacks exploit price feed weaknesses
- Discuss the latency vs trust tradeoff in oracle design

### Topics
- Feeding Off-Chain Data On-Chain
  - Chainlink-style feeds
  - TWAPs
- Attacks and Tradeoffs
  - Oracle manipulation attacks
  - Latency/trust tradeoffs

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Chainlink-style feeds, TWAPs, oracle manipulation attacks,
    latency/trust tradeoffs

# Part IV: Web3 and Digital Ownership

## 26: Decentralized Identity and Credentials

### Goals
- Explain DIDs and verifiable credentials as portable, verifiable identity
- Show soulbound tokens as non-transferable on-chain reputation
- Frame self-sovereign identity as user-controlled, not platform-controlled

### Topics
- Verifiable Identity
  - DIDs, verifiable credentials
- Non-Transferable Reputation
  - Soulbound tokens
  - Self-sovereign identity

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: DIDs, verifiable credentials, soulbound tokens, self-sovereign
    identity

## 27: NFTs and Digital Ownership

### Goals
- Explain ERC-721 and ERC-1155 as the token standards behind NFTs
- Show how provenance and royalties are enforced on-chain
- Compare on-chain vs off-chain metadata storage tradeoffs

### Topics
- Standards for Unique Assets
  - Token standards (ERC-721/1155)
- Ownership Guarantees
  - Provenance, royalties
  - On-chain vs off-chain metadata

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Token standards (ERC-721/1155), provenance, royalties, on-chain
    vs off-chain metadata

## 28: Decentralized Storage and Data Availability

### Goals
- Compare IPFS, Arweave, and Filecoin as decentralized storage designs
- Explain data availability sampling as a scaling and trust tool
- Connect storage guarantees to what a chain can safely reference off-chain

### Topics
- Storing Data Off-Chain
  - IPFS, Arweave, Filecoin
- Proving Data Is Available
  - Data availability sampling

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: IPFS, Arweave, Filecoin, data availability sampling

## 29: Privacy-Preserving Web3

### Goals
- Explain zk-rollups, mixers, and confidential transactions as privacy tools
- Show what each tool hides and what it still reveals
- Discuss the regulatory tension privacy tooling creates

### Topics
- Hiding Transaction Details
  - zk-rollups for privacy
  - Mixers
  - Confidential transactions
- The Regulatory Tension
  - Regulatory tension around privacy tooling

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: zk-rollups for privacy, mixers, confidential transactions,
    regulatory tension

# Part V: Cross-Cutting: Governance, Incentives, Security, Regulation

## 30: Governance and DAOs

### Goals
- Compare token-weighted voting and delegation as DAO decision mechanisms
- Explain treasury management, including compute treasuries for AI DAOs
- Identify common governance attack vectors and legal wrapper models

### Topics
- Voting and Delegation
  - Voting mechanisms, token-weighted voting
  - Delegation
- Treasury and Legal Structure
  - Treasury management, compute treasury management
  - Legal wrapper models
- Attack Surface
  - Governance attack vectors

### Slides
- N/A

### Lesson Materials
- `data605/lectures_source/Lesson_93_Sorrentum_Project.txt`
  - [20%]: DAO definition (rules encoded in a contract, controlled through a
    governance token, no centralized leadership), Uniswap example, tokenomics
    term introduced
- _Not covered_
  - [80%]: Delegation, treasury and compute-treasury management, governance
    attack vectors, legal wrapper models

## 31: Tokenomics and Incentive Design

### Goals
- Explain token issuance and emission schedules as supply-side design
- Compare ve-token models, liquidity mining, and protocol-owned liquidity
- Analyze incentive alignment through a game-theoretic attack lens

### Topics
- Issuance and Rewards
  - Token issuance, emission schedules
  - Reward mechanisms for compute/data contributors
- Locking and Liquidity
  - ve-token-style models
  - Liquidity mining, protocol-owned liquidity
- Getting Incentives Right
  - Incentive alignment, game-theoretic attack analysis

### Slides
- N/A

### Lesson Materials
- `data605/lectures_source/Lesson_93_Sorrentum_Project.txt`
  - [15%]: Staking/escrow/vesting smart-contract primitives, Sorrentum's own
    utility (SORRE) and governance (NTUM) tokens as a worked tokenomics
    example
- _Not covered_
  - [85%]: Emission schedules, reward mechanisms for compute/data
    contributors, ve-token models, liquidity mining, protocol-owned
    liquidity, game-theoretic attack analysis

## 32: Robustness and Security in AI Networks

### Goals
- Explain poisoning attacks as corruption of training data or updates
- Show Sybil resistance as a defense against fake-identity attacks
- Discuss how adversarial nodes threaten federated/decentralized systems

### Topics
- Attacking Training Data and Nodes
  - Poisoning attacks
  - Adversarial nodes in federated/decentralized systems
- Defending Network Identity
  - Sybil resistance

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Poisoning attacks, Sybil resistance, adversarial nodes in
    federated/decentralized systems

## 33: Security and Auditing

### Goals
- Catalog common smart contract vulnerability classes
- Show formal verification and bug bounties as complementary defenses
- Learn from post-mortem case studies of past exploits

### Topics
- Finding Vulnerabilities
  - Vulnerability classes
  - Formal verification
- Incentivizing and Learning from Failures
  - Bug bounties
  - Post-mortem case studies

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Vulnerability classes, formal verification, bug bounties,
    post-mortem case studies

## 34: Risk Management and Systemic Risk

### Goals
- Explain protocol composability ("money legos") as a source of systemic risk
- Show how contagion and collateral cascades propagate failures
- Introduce stress testing as a tool for systemic risk management

### Topics
- Composability and Contagion
  - Protocol composability ("money legos")
  - Contagion
- Modeling and Testing Risk
  - Collateral cascades
  - Stress testing

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Protocol composability ("money legos"), contagion, collateral
    cascades, stress testing

## 35: Regulation

### Goals
- Explain the tension between regulatory accountability and decentralization
- Compare securities classification and AML/KYC in permissionless systems
- Contrast custody-based CBDCs with permissionless chains for institutions

### Topics
- The Accountability Tension
  - Accountability vs decentralization tension
  - AI Act/MiCA-adjacent frameworks
- Compliance in Permissionless Systems
  - Securities classification/law
  - AML/KYC in permissionless systems
- Institutional Adoption
  - Custody, CBDCs vs permissionless chains
  - Enterprise blockchain use cases

### Slides
- N/A

### Lesson Materials
- _Not covered_
  - [100%]: Accountability vs decentralization tension, AI Act/MiCA-adjacent
    frameworks, securities classification/law, AML/KYC in permissionless
    systems, custody, CBDCs vs permissionless chains, enterprise blockchain
    use cases

### Notes
- Align regulatory framing with `responsible_genai_f23.md` (Berkeley
  CS294/194-196 course covering accountability vs decentralization)

# Appendix

- `README.md`: overlap analysis showing how this map unions
  `deai.map.md`, `defi.map.md`, and `web3.map.md` into the 35 chapters above
- `resources.md`: prioritized external reading list backing the book
- `decentralization_ai_summit_2023.md`: 2023 UC Berkeley summit program,
  source material for the Decentralized AI part
- `responsible_genai_f23.md`: UC Berkeley CS294/194-196 syllabus, source
  material for the Regulation chapter and the decentralized-training/
  federated-learning chapters
