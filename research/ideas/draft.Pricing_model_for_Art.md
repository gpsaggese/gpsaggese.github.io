# Pricing Model for Art

## Status
**Status:**: draft
**Complete Specs:**: 40%
**Assignee:**: 

## Core Idea

- Apply machine learning and deep learning to model and predict artwork
  prices, treating art as an asset class and reusing alt-data expertise
  built for financial markets
- Model painting prices as a factor regression, similar to equity factor
  models: artist reputation, painting attributes, provenance, and sales
  history are the predictors (`F`), with deep learning and NLP layers added
  on top to quantify harder signals (subject matter, critic sentiment,
  artist biography)
- Business rationale for doing this now:
  - Art is an increasingly recognized asset class: large and growing,
    becoming more transparent (many public auction price databases),
    attracting a wider pool of high-net-worth investors, and historically
    low-correlated with equities
  - It is a natural extension of alt-data work already done for financial
    markets
    - E.g., modeling which social-media accounts move a market is
      structurally similar to modeling the "social graph" of art marketing
      (dealers, auction houses, emerging artists)
  - It could open a new revenue stream (fund, consulting, or analytics
    service) and increase the firm's visibility with high-net-worth
    individuals and family offices, who are often both art collectors and
    fund clients
- Why it is non-obvious: the art market is illiquid, largely unregulated,
  and historically driven by insiders and taste rather than public data, so
  it is unclear a priori whether a quantitative factor model can find real
  out-of-sample predictive power (alpha) rather than just fitting noise
- Personal motivation and connections: the author has direct access to the
  art world through his wife, an associate professor of Contemporary Art who
  has consulted on authenticating major works and has relationships with
  senior figures at SF MOMA, Gagosian, and the Venice Biennale, which could
  help with introductions and domain validation

## Formalization

- Baseline model: factor regression on painting prices or returns, given
  target `R` and characteristics `F`, estimated via OLS with regularization
  $$
  R_i = a_i + b_{i1} F_1 + b_{i2} F_2 + \ldots + b_{iK} F_K + \epsilon_i
  $$
- More complex models (deep learning, decision trees) if there is enough
  data and evidence of non-linearities
- Candidate predictors `F_i`:
  - **Artist**: fame, life span, years active, pupils, peer performance
  - **Painting**: area, dimensions, technique (silk screen, paper, canvas),
    materials (oil, ...), signed and dated by the artist
  - **Subject matter**: painting themes, design themes, color scheme
  - **Importance**: literature coverage, number of reproductions
  - **Condition** of the work
  - **Provenance**: previous owners and dealers, sold at auction (which
    house, evening sale, place of sale) or on the primary market, prior
    museum exhibition
  - **Sales history**: date of sale, pre-sale estimates
- Deep learning / NLP extensions for signals that are hard to quantify
  directly:
  - What characteristics of a painting people like (learned from images)
  - Parse artist biography and other background text
  - Sentiment analysis of what art critics say about the artist and the work
  - Graph methods to model the relationship network between artists,
    dealers, and auction houses
- Methodology: gather as many art price databases as possible, estimate the
  betas on a rolling basis, evaluate in-sample $R^2$ and out-of-sample price
  accuracy
- Wild idea: beyond price forecasting, decompose returns into a factor and
  an idiosyncratic component and run mean-variance portfolio optimization
  across paintings, if some are inversely correlated in price

## Key Examples

- **Factor model in action**: predict a painting's price from artist fame,
  size, technique, and auction house, the same way an equity factor model
  regresses returns on value, momentum, and size factors
- **NLP-derived alpha signal**: sentiment analysis of art critic reviews or
  artist biography text as an additional predictor, analogous to using
  social-media sentiment as an alt-data signal in equities
- **Established players validate the approach**: Sotheby's acquired the
  machine learning startup ThreadGenius and the MeiMoses price indices,
  showing that major auction houses already see value in data-driven art
  valuation
- **Failure mode**: if the art market is so illiquid and insider-driven that
  prices do not respond to observable factors, the regression could show a
  high in-sample $R^2$ from overfitting on correlated predictors while
  having no genuine out-of-sample predictive power

## Questions

1. Can a factor model, with or without deep learning extensions, predict
   out-of-sample art prices well enough to generate alpha, or is the market
   too illiquid and insider-driven for a statistical edge to exist?
2. If no alpha is achievable, can a fund still be a viable business as a
   marked-to-model, buy-and-hold vehicle for diversification, given the
   market is largely unregulated and opaque?
3. If this works, does it change how other illiquid, insider-driven markets
   (e.g., wine, collectibles, private equity) should be approached with the
   same alt-data toolkit?

## Research Topics

- **Feasibility pilot**: get a low-cost art price database, build simple
  predictors (artist, size, year, dealer), and test whether out-of-sample
  price prediction is possible at all
- **Factor regression modeling**: predictors from artist, painting, subject
  matter, provenance, and sales history, estimated via OLS with
  regularization, extended to deep learning or decision trees if warranted
- **NLP for qualitative signals**: sentiment analysis of art critic
  commentary, artist biography parsing, and graph-based modeling of the
  dealer / artist / auction-house relationship network
- **Portfolio construction**: factor decomposition into common and
  idiosyncratic risk, and mean-variance optimization across paintings
- **Business model selection**: buy-side fund vs. sell-side consulting to
  family offices vs. an analytics service for auction houses and galleries
  vs. an art ETF vs. acquiring existing IP, and the operational feasibility
  of participating in auctions (execution, insurance, storage)

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: quick feasibility study
  - Focus on paintings, the most liquid segment of the art market
  - Acquire a low-cost art price database (under $500)
  - Assemble a small team to build simple predictors (artist, size, year,
    dealer)
  - This is the result: an early read on whether out-of-sample price
    prediction is possible, and the rough return / Sharpe ratio of a paper
    buy/sell strategy

- Milestone 2: factor model buildout
  - Gather multiple art price databases and estimate rolling betas
  - Evaluate in-sample $R^2$ and out-of-sample prediction accuracy
  - This is the result: a validated factor model with quantified predictive
    power

- Milestone 3: deep learning / NLP signal layer
  - Add NLP-derived signals: critic sentiment, artist biography, and the
    dealer / artist relationship graph
  - This is the result: measured incremental signal value over the baseline
    factor model

- Milestone 4: business model decision
  - Evaluate buy-side fund vs. consulting vs. analytics-as-a-service vs. ETF
    vs. IP acquisition against feasibility and revenue potential
  - This is the result: a recommended go / no-go decision and business model

## References

- Art price databases:
  - [artmarketresearch.com](http://www.artmarketresearch.com/)
  - [artnet.com price database](https://www.artnet.com/price-database/)
  - [artprice.com](https://www.artprice.com/)
  - [askart.com](http://www.askart.com/)
  - [mutualart.com auction results](https://www.mutualart.com/auction-results)
  - [findartinfo.com](http://www.findartinfo.com/english.html)
  - [artbusiness.com free price guide](http://www.artbusiness.com/freeprice.html)
  - [liveauctioneers.com auction results](https://www.liveauctioneers.com/auction-results)
  - [artvalue.com](http://www.artvalue.com/)
  - [Smithsonian Institution price database guide](https://sia.libguides.com/az.php?t=14137)
  - [Smithsonian Institution price database list](https://sia.libguides.com/pricedatabases)
  - [Sotheby's lots archive](http://www.sothebys.com/en/auctions/lots-archive.html)
- Art indices:
  - [How useful are art indices? (Forbes)](https://www.forbes.com/sites/kathryntully/2014/09/05/how-useful-are-art-indices/)
- Art startups:
  - [seditionart.com](https://www.seditionart.com/)
  - [arthena.com](https://arthena.com/)
  - [verisart.com](https://www.verisart.com/)
    - See also [a tech startup cataloguing the art market
      (Bloomberg)](https://www.bloomberg.com/news/articles/2015-07-21/a-tech-startup-is-trying-to-catalogue-every-piece-of-art-on-the-market)
  - [saatchiart.com](https://www.saatchiart.com/)
  - [upriseart.com](https://www.upriseart.com/)
  - [absolutart.com](https://www.absolutart.com/us/)
  - [artsy.net](https://www.artsy.net/)
  - [Amazon Fine Art marketplace](https://www.amazon.com/b?ie=UTF8&node=6685269011)
- Art funds:
  - [Art investment fund overview](http://alternativeinvestmentcoach.com/art-investment-fund/)
  - [Barron's coverage of art funds](https://www.barrons.com/articles/BL-PENTAB-522)
- Sotheby's acquisitions:
  - [Sotheby's acquires ThreadGenius
    (Bloomberg)](https://www.bloomberg.com/news/articles/2018-01-25/sotheby-s-buys-machine-learning-firm-to-discern-the-art-you-love)
  - [Sotheby's acquires the MeiMoses
    indices](http://www.sothebys.com/en/news-video/blogs/all-blogs/sotheby-s-at-large/2016/10/sothebys-acquires-mei-moses-indices.html)
- Background:
  - [Bill Gross sells his stamp collection
    (Bloomberg)](https://www.bloomberg.com/news/articles/2018-03-07/bond-guru-bill-gross-to-sell-42-2-million-stamp-collection)
  - [Basquiat painting sells for $110m
    (LA Times)](http://www.latimes.com/entertainment/la-et-entertainment-news-updates-may-basquiat-painting-auction-1495159714-htmlstory.html)
  - [PEN literary award for best first book on art history
    (UC Press)](https://www.ucpress.edu/blog/18903/reading-basquiat-wins-the-pen-literary-awards-ucp-first-book-award)
