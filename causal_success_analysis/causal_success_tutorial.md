# Additional Documentation

## Model Details

### Talent Dimensions

1. **Intensity**: Represents effort, persistence, and work ethic. Affects the probability of encountering opportunities (surface area of luck).

2. **IQ**: Cognitive ability and problem-solving skills. Determines the probability of successfully exploiting an opportunity when encountered.

3. **Networking**: Social capital and connections. Allows agents to occasionally capture opportunities that were originally assigned to others.

4. **Initial Capital**: Starting wealth. Provides baseline but does not directly affect event probabilities in the basic model.

### Event Mechanics

Events are randomly generated each round with:
- Type: Lucky (positive) or Unlucky (negative)
- Impact: Magnitude sampled from normal distribution
- Assignment: Based on agent talent probabilities

Lucky events multiply capital by (1 + impact)
Unlucky events multiply capital by (1 - impact), with floor at 0.01

### Key Assumptions

1. **Multiplicative Growth**: Capital changes are proportional to current capital, creating compounding effects

2. **Independence**: Talent dimensions are independent (though unrealistic, this simplifies analysis)

3. **Fixed Events**: Number of events per round is constant

4. **No Evolution**: Agent talents do not change over time (could be extended)

## Theoretical Background

### Power Laws in Success

Research shows that many success metrics follow power-law distributions:
- Wealth and income
- Citations and publications
- Company revenues
- Social media followers

Yet individual abilities tend to be normally distributed. This mismatch suggests random processes play a crucial role.

### Multiplicative Dynamics

When growth is multiplicative rather than additive, small advantages compound over time. This mathematical property alone can generate extreme inequality even without differences in underlying ability.

### Meritocracy Paradox

If success were purely merit-based, we would expect:
- Strong correlation between talent and outcomes
- Normal distribution of success mirroring talent distribution
- Top performers to be the most talented

The simulation shows none of these hold when luck is included.

## Causal Inference Approach

### Treatment

Number of lucky events experienced by each agent

### Outcome

Final capital (often log-transformed for better statistical properties)

### Confounders

Talent dimensions that affect both treatment assignment (event probability) and outcome (exploitation probability)

### Estimation Methods

1. **Average Treatment Effect (ATE)**: Mean effect of one additional lucky event
2. **Conditional ATE (CATE)**: Effect heterogeneity based on talent levels
3. **Policy Evaluation**: Comparing different resource allocation strategies

## Extensions and Modifications

### Possible Enhancements

1. **Talent Evolution**: Allow talents to increase with success or decrease with repeated failure

2. **Path Dependence**: Make early events unlock or block later opportunities

3. **Network Structure**: Implement explicit social networks rather than just networking scores

4. **Multiple Event Types**: Different kinds of opportunities with varying impacts

5. **Feedback Loops**: Success generates more opportunities (rich get richer)

6. **Burnout Effects**: Very high intensity could have diminishing or negative returns

### Calibration Approaches

To make the model more realistic:
- Fit impact distributions to real wealth change data
- Match Gini coefficients to observed inequality levels
- Calibrate event frequencies to career opportunity rates
- Use actual talent distributions from standardized tests

## Interpretation Guidelines

### What the Model Shows

- Randomness matters more than typically acknowledged
- Multiplicative processes generate inequality from equality
- Merit is necessary but not sufficient for success
- Policy design affects both efficiency and equity

### What the Model Does Not Show

- Specific optimal policies (too simplified)
- Precise quantitative predictions (stylized model)
- Individual destiny (probabilistic outcomes)
- Complete picture of real-world complexity

## References and Further Reading

Pluchino, A., Biondo, A. E., & Rapisarda, A. (2018). Talent vs Luck: the role of randomness in success and failure. Advances in Complex Systems, 21(03n04), 1850014.

Frank, R. H. (2016). Success and Luck: Good Fortune and the Myth of Meritocracy. Princeton University Press.

Mauboussin, M. J. (2012). The Success Equation: Untangling Skill and Luck in Business, Sports, and Investing. Harvard Business Review Press.

Athey, S., & Wager, S. (2019). Estimating Treatment Effects with Causal Forests. Observational Studies, 5(2), 37-51.
