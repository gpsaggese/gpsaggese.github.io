# Train Financial Sentiment Analysis Using Price Response as Labels

## Status

- **Status:**: draft
- **Complete Specs:**: 90%

## Core Idea

- Standard financial sentiment models (e.g., FinBERT) are trained on human-labeled
  positive/negative/neutral news, which is expensive to produce and reflects human
  judgment of tone, not market impact
- Instead, fine-tune an LLM to predict sentiment using the _subsequent market price
  response_ to a news item as a weak, automatically-generated label (e.g., price
  moved up X% in the following N minutes/hours => "positive")
- Key question: does a price-derived label produce a sentiment signal that is more
  predictive of _future_ price moves than a human-labeled sentiment signal, or does
  it just learn to echo already-priced-in information (look-ahead/leakage risk)?

## Formalization

### Price-Response Labeling

- Label: `y_i = sign(return(t_i, t_i + Delta))` for news item `i` published at `t_i`,
  for some horizon `Delta`
- Model: `y_hat_i = f_theta(text_i)` trained to predict `y_i`
- Evaluate on a held-out, strictly-later time period
  - Compare predictive power of `y_hat` for _out-of-sample_ returns against:
    - A human-labeled-sentiment baseline (e.g., FinBERT scores)
    - A naive momentum baseline

### Attribution: Which News Explains a Large Price Move

- The mirror image of the labeling problem above: instead of starting from a news
  item and looking forward at the price, start from a large price move and look
  backward for the news that explains it
- Procedure: detect moves exceeding `k` standard deviations, pull the candidate news
  in a window before the move, and rank candidates by how well they explain it
  (LLM-scored plausibility, or a learned model over headline features)
- This yields a labeled set of "market-moving" vs. "ignored" news, which is a much
  more informative training signal than a raw sentiment label: it directly separates
  news that matters from news that does not
- The hard part is that a large fraction of big moves have _no_ identifiable news
  cause (flows, positioning, unwinds), and a system that always returns an
  explanation will confabulate one: so "no news explains this" must be a first-class
  output, and the false-attribution rate must be reported
- Related: [[draft.Measure_ability_to_predict_events]] for the forecasting side, and
  [[draft.Causal_Analysis_of_Financial_Tradability]] for whether an identified driver
  is tradable at all

## Key Examples

- **Move attribution**: for each daily move over 3 sigma, rank the preceding
  headlines and check against a hand-labeled subset of known event days
- **Unexplained moves**: the fraction of large moves with no plausible news cause is
  itself the result of interest: a baseline that any attribution claim has to beat
- **Earnings announcements**: label by post-announcement price drift; compare against
  analyst-labeled sentiment of the same announcement text
- **Macro news**: label by index-level move following a macro release; check whether
  the model learns genuine content signal or just keyword-level momentum artifacts
  (e.g., "rate cut" => up, regardless of context)

## Questions

1. Does price-derived labeling produce a sentiment signal with genuine out-of-sample
   predictive power, or does it overfit to look-ahead/leakage in the labeling window?
2. How sensitive are results to the choice of horizon `Delta` and to controlling for
   overall market movement (excess return vs. raw return as the label)?
3. Does this approach transfer across asset classes (equities vs. FX vs. crypto), or
   is it tied to one market's specific news-to-price dynamics?
4. What fraction of large moves are attributable to identifiable news at all, and how
   often does an LLM attributor invent a cause for a move that had none?

## Research Topics

- Weak/self-supervised labeling from market data
- Look-ahead bias and leakage control in event-study-style labeling
- Event-study methodology for move detection (abnormal return vs. raw return,
  volatility-adjusted thresholds)
- News-to-move attribution with an explicit "unexplained" class, and measuring the
  confabulation rate
- Comparison against existing financial sentiment benchmarks (FinBERT,
  Loughran-McDonald dictionary)

## Next steps

- [ ] Assemble a news+price dataset with precise timestamps
- [ ] Define the label (return horizon, excess-return control)
- [ ] Fine-tune a baseline model and compare against FinBERT-style sentiment
- [ ] Build the move-detection + attribution pipeline and measure the
      unexplained-move fraction
- [ ] Backtest predictive power out-of-sample, checking for leakage

## Implementation plan

- Milestone 1: dataset and labels
  - Assemble a news and price dataset with precise timestamps
  - Define labels $y_i$ for several horizons $\Delta$, using both raw and
    excess returns
  - Split by time and check for leakage (timestamp alignment, no news after
    the cutoff)
  - This is the result: a versioned dataset, the labeling code, and a leakage
    checklist

- Milestone 2: fine-tune and compare
  - Fine-tune a baseline model on the price-derived labels
  - Compare out-of-sample predictive power against FinBERT, the
    Loughran-McDonald dictionary, and a momentum baseline
  - This is the result: a predictive-power table across horizons and evidence
    for or against price-derived labels

- Milestone 3: move attribution
  - Detect moves above $k$ standard deviations and pull the candidate news
    before each move
  - Rank candidates by LLM plausibility with an explicit "unexplained" class
  - Check the result against a hand-labeled set of known event days
  - This is the result: the unexplained-move fraction and the false-attribution
    rate

## References

- Araci, _FinBERT: Financial Sentiment Analysis with Pre-trained Language Models_.
  (2019)
