# Abstract. AI inference is commonly purchased through direct provider selection or posted-price gateways, which do not necessarily allocate inference to the buyer–seller pairs generating the greatest economic surplus. This paper evaluates the allocative efficiency of Noesis, a two-sided batch call auction, relative to random provider selection and sequential posted-price routing. Three computational experiments compare these mechanisms using matched synthetic markets and an independent welfare-maximizing benchmark. In 200 baseline markets, Noesis achieved 100% allocative efficiency versus approximately 96% for posted pricing. Adding random provider selection showed that Noesis generated 10.7% more total surplus than random allocation, despite trading fewer tasks. A third experiment calibrated simulated markets using observed OpenRouter reference prices and varied assumed willingness to pay and seller costs. Across nine price-calibrated scenarios, Noesis exceeded posted pricing by 6.1–12.2 percentage points in allocative efficiency. The findings establish an implementation-level comparison under simplified static, price-only assumptions; they do not measure real-world profits or guarantee general optimality.

# 1. Introduction

AI inference may be purchased directly from model providers or through
services that aggregate providers behind a common interface. Provider
prices are generally posted in advance, yet buyers differ in willingness
to pay and suppliers differ in costs and available capacity. A fixed
price or arbitrary provider choice can therefore leave economically
beneficial transactions unrealized.

Noesis proposes a two-sided marketplace in which buyers submit bids and
sellers submit asks for inference tasks. Its batch call auction matches
these orders at a uniform clearing price. The relevant economic question
is not whether the auction delivers a lower price for every participant,
but whether it allocates a given set of potential trades to generate
greater total economic surplus.

We restrict the analysis to a single comparable inference tier and to
price and quantity. We examine static market snapshots rather than
changing market conditions over time, and exclude capability, latency,
and reliability because their monetary valuation would require a
distinct model.

## 1.1. Research question

How much economic surplus does the implemented Noesis batch call auction
generate relative to posted-price routing and direct random provider
selection when identical buyers and sellers participate in each
mechanism?

## 1.2. Contributions

This paper (i) defines a consistent framework for surplus, allocative
efficiency, deadweight loss, and matched volume; (ii) compares the
actual Noesis matching implementation with random provider selection,
posted-price routing, and an independent optimum; and (iii) examines
whether its allocation advantage persists in static scenarios anchored
to observed inference prices, with explicitly assumed buyer valuations
and seller costs.

# 2. Economic Framework

Let buyer i value a unit of inference at vᵢ, seller j incur unit cost
cⱼ, and qᵢⱼ tasks trade at execution price pᵢⱼ. A trade is individually
rational when cⱼ ≤ pᵢⱼ ≤ vᵢ. Consumer surplus, producer surplus, and
total surplus for a completed trade are:

CSᵢⱼ = qᵢⱼ (vᵢ − pᵢⱼ)

PSᵢⱼ = qᵢⱼ (pᵢⱼ − cⱼ)

TSᵢⱼ = CSᵢⱼ + PSᵢⱼ = qᵢⱼ (vᵢ − cⱼ)

Total economic surplus sums TSᵢⱼ across completed fills. For a given
trade, its price determines how the gains are divided between buyer and
seller, not the amount of total surplus. Allocative efficiency is
realized surplus divided by the maximum feasible surplus for that same
market; deadweight loss is the difference between maximum and realized
surplus. Matched volume is the number of tasks executed and is reported
separately because higher volume need not imply greater surplus.
Baseline surplus values are normalized simulation units rather than
dollars.

# 3. Mechanisms and Experimental Design

## 3.1. Allocation mechanisms

**Random provider selection**. Buyers are processed sequentially. For
each buyer, sellers with remaining capacity are visited in a locally
seeded random order. The buyer trades with an affordable seller, subject
to capacity, without systematically selecting the cheapest available
offer. Each seller is visited at most once for that buyer.

**Posted-price routing**. Buyers are processed in generated order and
routed to the cheapest available provider charging no more than their
willingness to pay, subject to available capacity. In Experiments 1 and
2, each posted price is 20% above modeled seller cost. This is a
stylized posted-price comparator, not a reproduction of OpenRouter
production routing.

**Noesis auction.** The same generated buyer bids and seller asks are
submitted to the implemented OrderBook.clear_round() batch auction.
Welfare is computed from the completed fills and original buyer
valuations and seller costs.

Independent optimum. A separate welfare-maximizing allocation pairs
high-valuation demand with low-cost supply while feasible beneficial
trades remain. It is a benchmark for economic efficiency, not a stand-in
for Noesis.

## 3.2. Experiment 1: Baseline auction versus posted pricing

Experiment 1 establishes the baseline welfare gap between the
implemented Noesis auction and a stylized posted-price router. It
generates 200 reproducible single-tier markets with 20 buyers and 20
sellers. Buyer willingness to pay and seller costs follow lognormal
distributions with median WTP-to-cost ratio 1.5 and log-price dispersion
0.3. Quantities are independently sampled as integers from 1 to 100
tasks. The same orders are passed unchanged to Noesis, posted-price
routing, and the independent optimum.

## 3.3. Experiment 2: Comparison of purchasing mechanisms

Experiment 2 adds random direct provider selection to the identical 200
seeded markets from Experiment 1. Keeping valuations, costs, quantities,
and provider prices fixed isolates the impact of the purchasing
mechanism itself. The primary outputs are total surplus and efficiency;
deadweight loss and matched task volume clarify whether differences are
attributable to the number or economic value of completed trades. These
results are an expansion of the baseline comparison, not an independent
sample of new markets.

## 3.4. Experiment 3: Price-calibrated static scenarios

Experiment 3 anchors simulated task prices to an archived OpenRouter
model-price snapshot retrieved on November 12, 2026. Each task is
standardized to 1,000 input and 1,000 output tokens. Buyers are assigned
willingness-to-pay premiums of 5%, 10%, or 20% over model-specific
reference prices, while seller costs are assumed to equal 70%, 80%, or
90% of those reference prices. The nine combinations are each evaluated
over 200 seeded markets (1,800 markets total). Posted-price and
random-selection offers use the reference prices; Noesis asks use the
assumed seller costs. All comparisons take place within a simplified
single compatible text-generation tier. Observable posted prices do not
reveal actual buyer valuations or seller marginal costs.

## 3.5. Reproducibility and validation

All mechanisms receive identical input orders within each seeded market.
The experiment runners check participant identities, capacity limits,
individual rationality of execution prices, conservation of surplus
accounting, deterministic regeneration, unchanged input orders, and
surplus no greater than the independently computed optimum. The
price-calibrated experiment archives the OpenRouter snapshot and its
fingerprint. The experiments are static counterfactual allocations, not
simulations of market dynamics.

# 4. Experimental Results

## 4.1. Experiment 1: Baseline surplus comparison

Across 200 matched baseline markets, Noesis generated mean total surplus
of 635.44 normalized units compared with 611.30 under posted-price
routing. The mean paired improvement was 24.14 units (approximate 95%
confidence interval: 20.76–27.53). Noesis matched the independent
maximum in all 200 markets, while posted pricing captured approximately
96% of available surplus. The auction traded fewer tasks on
average—752.06 versus 768.37—while producing greater economic value.

## 4.2. Experiment 2: Alternative purchasing mechanisms

Introducing random provider selection produced mean surplus of 574.15
units and approximately 90% efficiency. Posted-price routing generated
611.30 units (approximately 96% efficiency), while Noesis generated
635.44 units (100% efficiency). Thus Noesis produced 10.7% more surplus
than random selection and 3.9% more than posted pricing; the paired
Noesis-minus-random surplus gain averaged 61.30 units (95% CI:
56.59–66.00).

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 19%" />
<col style="width: 20%" />
<col style="width: 19%" />
<col style="width: 19%" />
</colgroup>
<thead>
<tr>
<th>Mechanism</th>
<th>Mean surplus</th>
<th>Efficiency</th>
<th>Mean DWL</th>
<th>Mean tasks</th>
</tr>
<tr>
<th>Random selection</th>
<th>574.15</th>
<th>90.0%</th>
<th>61.30</th>
<th>851.25</th>
</tr>
<tr>
<th>Posted-price routing</th>
<th>611.30</th>
<th>96.0%</th>
<th>24.14</th>
<th>768.37</th>
</tr>
<tr>
<th>Noesis auction</th>
<th>635.44</th>
<th>100.0%</th>
<th>0.00</th>
<th>752.06</th>
</tr>
<tr>
<th>Independent optimum</th>
<th>635.44</th>
<th>100.0%</th>
<th>0.00</th>
<th>752.06</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

Table 1. Mean outcomes across the same 200 synthetic markets. Surplus
and deadweight loss are normalized units.

Random provider selection executed the highest mean task volume
(851.25), yet generated the lowest mean surplus. Noesis executed the
fewest tasks among the three purchasing mechanisms but achieved the
greatest surplus. This illustrates why transaction count and welfare are
distinct outcomes.

<img
src="Market_Quality_Research.figs/image1.png"
style="width:4.125in;height:2.58333in" />

Figure 1. Mean total economic surplus by allocation mechanism in
Experiment 2. Error bars show uncertainty in each mechanism mean; paired
differences are reported in text.

## 4.3. Experiment 3: OpenRouter price-calibrated scenarios

In the nine OpenRouter price-calibrated scenarios, Noesis exceeded
posted-price routing by 6.1 to 12.2 percentage points in allocative
efficiency and equaled the independent welfare optimum in every one of
the 1,800 tested markets. The largest advantage occurred when buyers
paid at most a 5% premium above reference prices and seller costs were
assumed to be 90% of reference prices. The advantage was smaller, though
still positive, in the higher-WTP and lower-cost scenarios. This is
evidence about modeled allocations under price-calibrated assumptions,
not measured gains in actual OpenRouter transactions.

<table>
<colgroup>
<col style="width: 25%" />
<col style="width: 24%" />
<col style="width: 24%" />
<col style="width: 24%" />
</colgroup>
<thead>
<tr>
<th>Buyer WTP premium</th>
<th>Cost = 70%</th>
<th>Cost = 80%</th>
<th>Cost = 90%</th>
</tr>
<tr>
<th>+5%</th>
<th>+6.8 pp</th>
<th>+8.4 pp</th>
<th>+12.2 pp</th>
</tr>
<tr>
<th>+10%</th>
<th>+7.1 pp</th>
<th>+7.6 pp</th>
<th>+10.6 pp</th>
</tr>
<tr>
<th>+20%</th>
<th>+6.1 pp</th>
<th>+6.1 pp</th>
<th>+8.3 pp</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

Table 2. Noesis allocative-efficiency advantage over posted pricing,
measured in percentage points.

<img
src="Market_Quality_Research.figs/image2.png"
style="width:4.15278in;height:2.86111in" />

Figure 2. Noesis efficiency advantage over posted pricing in Experiment
3, by assumed buyer WTP premium and seller cost relative to the
reference task price.

# 5. Discussion

Across the three experiments, the allocation mechanism affects the
welfare produced by the same buyers and sellers. Random selection does
not systematically steer demand toward low-cost supply, posted-price
routing does so within the limits of posted offers, and the Noesis
auction matches submitted bids and asks. Under the stated simulation
assumptions, the auction achieved the independently computed maximum
surplus.

Experiment 2 highlights that matching more tasks is not equivalent to
producing more welfare: random selection traded more tasks but created
fewer gains from trade. Experiment 3 shows that the allocation gap
remains positive when prices are anchored to observed inference
listings, though its size depends on assumptions about buyer values and
seller costs. The baseline posted-price router was already relatively
efficient, so the additional benefit of Noesis should be characterized
as measurable rather than universal or transformative.

The fact that Noesis matched the independent optimum in these
experiments does not establish a general theorem about batch auctions,
heterogeneous service quality, strategic behavior, or deployed systems.
The welfare benchmark is an independent check within the deliberately
simplified model.

# 6. Limitations

All markets are synthetic and static; the study assumes truthful bids
and asks, homogeneous compatible tasks, and stylized routing rules. In
Experiment 3, observed OpenRouter listings calibrate reference prices
but neither marginal provider costs nor buyer willingness to pay is
directly observed. Price differences among models may reflect meaningful
differences in capability, which the single-tier assumption abstracts
away. The experiments omit market fees, strategic bidding, repeated
participation, latency, reliability, fulfillment risk, and operational
costs. Statistical intervals capture seed-to-seed simulation
variability, not uncertainty about these modeling assumptions.

# 7. Conclusion

The implemented Noesis batch call auction generated greater total
economic surplus than modeled random provider selection and sequential
posted-price routing across the tested static markets, reaching the
independent maximum under the simplified assumptions. The comparisons
show that allocation quality matters independently of transaction
volume. Price-calibrated scenarios further demonstrate that modeled
welfare gaps persist under several explicit buyer valuation and seller
cost assumptions. These results provide a reproducible economic
motivation for studying two-sided inference markets without claiming
guaranteed real-world welfare or profit gains.
