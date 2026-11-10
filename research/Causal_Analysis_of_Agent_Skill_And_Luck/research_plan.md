# Causal Analysis of Agent Skill and Luck

## Milestone 1 Research Plan

- Current plan: the dated update below contains the confirmed Deal or No Deal
  direction and current implementation status
  - I have retained the earlier proposal for context
  - The update supersedes its pending scope decisions and older progress
    statements

# Goal

- My goal is to bring our work into one paper addressing a shared question:
  - How can we distinguish underlying ability and decision quality from favorable
    circumstances and random outcomes?
  - How can we use that distinction to evaluate success more reliably?
- I propose two separate studies under this question:
  - Niveda will investigate skill and luck in coding-agent benchmark evaluation
  - Riya and Stuti's original proposal extends my societal success simulation to
    study:
    - Learning
    - Networks
    - Resource allocation
- Each study retains its own model and experiments
  - Both studies contribute to a common framework and discussion in the paper
- One scope decision must be resolved before Riya and Stuti begin substantial new
  implementation:
  - GP's
    [latest comment on \#659](https://github.com/gpsaggese/gpsaggese.github.io/issues/659#issuecomment-6290190159)
    recommends considering
    [Deal or No Deal (\#672)](https://github.com/gpsaggese/gpsaggese.github.io/issues/672)
    instead of the original extensions
    - GP describes the original extensions as incremental
  - I include their original plan below and a concrete replacement plan if we take
    GP's recommendation
  - I will confirm that choice with GP and the team
  - We should execute one of those options rather than commit to both

# What Is Done

- My
  [previous project](https://github.com/bkk8403/A-Causal-Analysis-of-Success-in-Modern-Society)
  provides the societal baseline:
  - An agent-based simulation with:
    - Intensity
    - IQ
    - Networking
    - Capital
    - Random events
  - Inequality metrics
  - Causal estimation
  - Allocation-policy comparisons
- The
  [current paper](https://github.com/gpsaggese/gpsaggese.github.io/tree/master/papers/Causal_Analysis_of_Agent_Skill_And_Luck)
  describes this model and a broader mathematical framework
  - I want us to reproduce its reported results before extending or relying on them
- Niveda has implemented a separate synthetic benchmark simulator in
  `skill_luck_sim/`
  - It models:
    - Configuration ability
    - Task difficulty
    - Configuration-task interactions
    - Repeated binary outcomes
  - It includes tests, saved results, and plots for:
    - Winner's curse
    - Ranking errors
    - Fresh-seed versus fresh-task reevaluation
  - [PR \#605](https://github.com/gpsaggese/gpsaggese.github.io/pull/605) was
    merged into the review branch
  - Its parameters are assumed rather than fitted to real agent runs
  - Next steps remain:
    - The real benchmark pilot
    - Integration into the shared research structure
- Riya and Stuti have
  [proposed](https://github.com/gpsaggese/gpsaggese.github.io/issues/659) these
  extensions:
  - Changing skills over time
  - Explicit connection networks
  - Adaptive allocation
  - Possible real-data validation as a follow-up
  - I treat these as proposed extensions rather than completed implementations
  - Their individual division of work still needs agreement
- Our immediate technical gaps are:
  - Reproducibility
  - Consistent definitions
  - Integration
- Niveda has reported:
  - Differences between the societal runs and the paper
  - Skipped tests and old imports
  - A performance-based allocation policy that can match equal allocation when
    starting capital is identical
- The paper's numerical findings should remain provisional until we explain those
  differences

# Next Steps

## Shared Foundation

- By shared foundation, I mean the definitions, working code, and experiment
  records that both studies need before we build on their results
  - Each simulator keeps its own model, baseline, and experiments
  - The foundation connects how we describe and validate the studies
  - It does not combine them into one simulator
- First, agree on the meaning of skill, expected performance, and luck in each
  study
  - Use consistent terminology and symbols while keeping the outcomes distinct:
    - Societal model: capital or another specified success measure
    - Benchmark model: task resolution or mean score
  - For each expected outcome, state what is held fixed and what is randomized
  - Distinguish two notions of luck:
    - Residual luck: observed outcome minus expected outcome
    - The paper's separate event-level score based on utility, surprise, and
      control
- Second, establish a reliable starting implementation for each study:
  - Agree on the reference code versions and integration branch
  - Restore test collection and imports
  - Reproduce the existing reference results
  - Niveda will lead the cleanup assigned by GP
  - I will contribute to baseline reproduction and validation
  - Document so another contributor can rerun the experiments:
    - Configurations
    - Seeds
    - Software versions
    - Commands
- We need this foundation for two reasons:
  - Within each study, it lets us distinguish an effect of a new mechanism or
    evaluation method from:
    - A bug
    - A changed setting
    - An unreproduced baseline
  - Across the studies, it lets us explain precisely which ideas are shared and why
    both belong in the same paper
    - It does not imply that their outcomes or numerical results are
      interchangeable
- Any unexplained discrepancy will be recorded and its impact assessed
  - This happens before we use the affected result to support a conclusion
- This step is complete when we have:
  - A short definition of the quantities and reference conditions for each study
  - Runnable tests and reference experiments
  - Saved results with reproduction instructions
  - Discrepancies either resolved or explicitly tracked, with a clear statement of
    which conclusions remain unsupported

## Plan for Niveda

- Research question: for a coding-agent leaderboard score, how much:
  - Reflects persistent performance
  - Depends on sampled tasks and execution randomness
  - Is optimism introduced when we select the best observed configuration
- **N1 Validate the synthetic baseline**:
  - Reproduce the saved reference experiments
  - Test the distinction between new execution seeds on fixed tasks and genuinely
    new tasks
  - Define expected performance for each target
  - Evaluate:
    - Score inflation for the selected winner
    - Probability of selecting the truly best configuration
    - Regret
  - Validate selected-winner uncertainty explicitly, with shared tasks preserved in
    resampling
  - Output: reproducible baseline figures and a short explanation of the
    assumptions and limits
- **N2 Run the feasibility pilot**:
  - Precondition: GP confirms model access and a capped budget
  - Choose one runnable open-source agent and one fixed configuration
  - Run 20-30 SWE-bench Verified tasks with five independent repeats per task
    - This gives 100-150 attempted task runs before retries
  - Save:
    - Task IDs
    - Binary outcomes
    - Versions
    - Budgets
    - Execution settings
    - Costs
    - Failure categories
  - Specify how infrastructure failures and retries are handled
  - Output: a replayable outcome dataset and a variability/cost report
  - This pilot measures repeatability
    - It cannot establish selection bias across configurations
- **N3 Compare configurations with paired evaluation**:
  - Use the pilot to choose an affordable number of agents or scaffold
    configurations and tasks
    - Do not immediately commit to the idea document's full 2-3 benchmark, 3-5
      agent matrix
  - Evaluate configurations on the same tasks
  - Separate task-sampling from run variability
  - Report paired score differences with confidence intervals
  - Estimate the number of tasks needed to detect a prespecified meaningful gap
  - Output: an uncertainty-aware comparison stating which differences the evidence
    can resolve
- **N4 Test selection and correction**:
  - Prespecify a candidate search, its budget, and a task split
  - Select a configuration using development tasks, freeze it, and evaluate on
    independent held-out tasks
  - Compare:
    - Naive winner reporting
    - Same-task fresh-seed reevaluation
    - Fresh-task evaluation
  - Vary search size first in simulation and then within the available real-run
    budget
  - Treat square-root-log search penalties as approximations to test, not
    universally valid corrections
  - Output: measured optimism and a validated reporting protocol with its cost and
    limitations
- **N5 Extend only after the core result**:
  - If the data and budget support it:
    - Test rank persistence across task splits and a second benchmark
    - Then consider item response theory (IRT) to estimate task difficulty and
      discrimination
  - A meaningful IRT analysis requires substantially more configurations than the
    one-agent pilot
  - Later extensions:
    - Adaptive task selection
    - Contamination analysis
  - Output: evidence about transfer or informative tasks, if feasible, without
    delaying the core paper
- I propose that Niveda develop the state-of-the-art document alongside N1 and N2
  - [Fama-French](https://mba.tuck.dartmouth.edu/bespeneckbo/default/AFA611-Eckbo%20web%20site/AFA611-S8C-FamaFrench-LuckvSkill-JF10.pdf)
    motivates comparing observed winners with chance-based reference distributions
  - [Bailey and colleagues](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
    motivate evaluating the selection process
  - [SIREN](https://arxiv.org/html/2605.05973v1) is a direct comparator for
    selection-aware LLM reporting
  - [SWE-rebench](https://arxiv.org/html/2505.20411v1) motivates repeated,
    standardized evaluations
  - We need to identify what our coding-agent experiments add beyond these existing
    results

## Plan for Riya and Stuti

- Original research question: how does change over time in these factors alter the
  relationship between underlying ability, opportunity, and accumulated success?
  - Ability
  - Social connections
  - Allocation rules
- **R1 Reproduce and clarify the societal baseline**:
  - Run the agreed configuration over independent seeds
  - Regenerate inequality, wealth-share, and skill-outcome results
  - Separate exposure to an opportunity from successfully exploiting it
    - Successful-event counts already depend on ability
  - Clarify allocation timing and outcome scale
  - Check causal estimates against explicitly defined simulator interventions
    - Do not assume that a fitted regression establishes causality
  - Output: a baseline report and a list of resolved or outstanding paper/code
    differences
- **R2 Test learning and skill decay first**:
  - Specify one learning/decay mechanism and how it affects opportunity
    exploitation or exposure
  - Keep it optional so zero learning and zero decay recover the baseline
  - Compare the fixed-skill model with the extension using matched initial agents
    and seed sets
  - Report uncertainty across independent worlds
  - Measure:
    - Inequality
    - Mobility
    - How observed success ranks agents relative to their expected performance
  - Output: one controlled extension with an interpretable result, including a null
    result if that is what we find
- **R3 Add networks as a separate experiment**:
  - Replace or extend the existing scalar networking mechanism with an explicit
    graph
  - Compare a small set of justified graph structures
    - Control overall opportunity supply and other assumptions
  - The existing model already makes exposure and spillovers depend on
    characteristics
    - The new contribution is topology
  - Output: evidence about whether network structure changes the persistence and
    concentration of advantage
- **R4 Evaluate adaptive allocation**:
  - Start after the previous results are stable
  - Compare a specified adaptive policy with fixed-policy baselines using equal
    total budgets and evaluation horizons
  - Any policy based on past performance must observe an initial history before
    allocating
  - Evaluate:
    - Welfare
    - Inequality
    - The risk of repeatedly rewarding fortunate agents
  - Output: a policy comparison with clear timing, information, and resource
    constraints
  - Real-data calibration follows only after a suitable dataset and validation
    target are established
- I propose that Riya and Stuti choose a primary implementer and a reviewer for
  each task and rotate those roles where useful
  - I will support the baseline and review the interpretation
  - Their published issue is a joint proposal
    - I will not assume an individual split before we agree on it
- [Pluchino and colleagues](https://arxiv.org/html/1802.07068v3) provide the
  simulation precedent
  - Our contribution must be a defined mechanism or experiment beyond reproducing
    the existing finding that luck can matter

## If We Follow GP's Deal or No Deal Recommendation

- I propose replacing R2-R4 with a focused decision-quality study
  - My societal model is retained as existing background or a reproduced case
    study
- Riya and Stuti would:
  - Obtain round-level contestant data
  - Reproduce one analysis from the relevant literature
  - Implement an optimal stopping benchmark with an explicit model for future
    banker offers
  - Then incorporate risk preferences
  - Compare observed choices with that benchmark
  - Investigate whether previous gains or losses predict departures after
    accounting for the current game state
- First deliverable: a cleaned, documented dataset and an end-to-end example
  producing a benchmark decision for a small set of rounds
- Next deliverable: a validated analysis of decision quality and sensitivity to the
  risk and offer models
- We must distinguish:
  - A reasonable decision followed by bad luck from a poor decision
  - Descriptive associations from identified causal effects
- [Post and colleagues](https://www.aeaweb.org/articles?id=10.1257/aer.98.1.38) are
  the central empirical reference
- This option is pending the scope decision
  - I am not assigning both it and the original extension programme

## How the Studies Unify Into One Paper

- I propose a common framework based on:
  - Observed outcome
  - Expected performance under a specified reference environment
  - The residual around that expectation
- Caveats:
  - Repeated simulations or evaluations let us estimate uncertainty in expected
    performance
    - A finite-sample estimate is not the true expectation
  - Predictable advantages may include circumstances and resources
    - We should not equate them automatically with earned skill
  - Statistical decomposition alone does not establish a causal attribution
- The two original studies map onto the framework as follows:

| Element          | Niveda study                                                                   | Riya and Stuti original study                                                   |
| :--------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| Outcome          | Task resolution or mean benchmark score                                        | Final capital, inequality, or mobility                                          |
| Reference        | Expected performance of a fixed configuration on a specified task distribution | Expected outcome under a defined agent profile or policy across repeated worlds |
| Random variation | Sampled tasks, execution randomness, and logged infrastructure effects         | Event histories and graph/event realizations under controlled assumptions       |
| Decision tested  | Select a configuration and report its performance                              | Allocate opportunities or resources using defined information                   |
| Shared evidence  | Bias, rank stability, uncertainty, and selection regret                        | Rank stability, uncertainty, welfare, and inequality                            |

- The shared contribution I want to test: do decisions based on one realized
  success measure differ from decisions informed by repeated performance and
  uncertainty?
  - In Niveda's study, the decision is which configuration to select and report
  - In the societal study, the decision is whom to reward or fund
  - Where feasible, I propose comparing realized-outcome ranking with an
    expected-performance reference in both studies
  - If we instead choose Deal or No Deal, the link becomes decision quality versus
    realized payoff:
    - The benchmark study evaluates systems
    - The game-show study evaluates choices under uncertainty
- The paper will contain:
  - A shared introduction and framework
  - Separate methods and results for each study
  - A joint discussion of evaluation and decision-making under uncertainty
  - I propose reporting each study's uncertainty and limitations separately
- We will not:
  - Transfer a percentage of luck from one domain to another
  - Label observational residuals as proven causal luck
  - Promise a new correction before it is validated
- Negative findings can still explain when a method or proposed mechanism does not
  help

## How We Will Proceed

- I have reviewed Niveda's plan and Riya and Stuti's proposal
- My next step is to confirm the research scope with GP and the team
  - Decide whether Riya and Stuti continue the societal simulation extensions or
    take up Deal or No Deal
  - Agree on the questions each study will answer
  - Agree on how its results will contribute to the paper
- Next, agree on the shared definitions and reproduce the existing results for each
  simulator
  - Niveda will lead code cleanup
  - I will contribute to baseline validation
  - The literature review can proceed alongside this work
    - It identifies what each study adds to existing research
- Once the relevant setup is verified, the studies can proceed in parallel
  - Niveda will run the small coding-agent pilot after the evaluation protocol and
    budget are confirmed
  - Riya and Stuti will carry out the first experiment in their agreed direction
  - Neither study needs to wait for unrelated work in the other
- We will review the initial results with GP before expanding the experiments:
  - Niveda's work: decide how to compare multiple configurations and evaluate the
    selected configuration on independent tasks
  - Riya and Stuti's work: decide which further experiment is needed to answer
    their research question
  - Check whether both studies provide evidence for the paper's proposed common
    argument
- Finally, we will build the paper around the validated findings as they become
  available
  - Each study will have its own methods, results, and limitations
  - I will coordinate the shared framework and discussion
    - The paper explains what the studies establish together, where they differ,
      and which questions remain open
- Roles:
  - I will coordinate the paper outline, research decisions, and integration
  - I will contribute to baseline validation
  - Niveda will lead the benchmark study and assigned cleanup and literature work
  - Riya and Stuti will lead their agreed study
  - GP will supervise the research scope and scientific claims
- We will agree on individual task owners and milestones
  - Then track the immediate work in GitHub issues linked to this plan

# Update: 20 October 2026

- Riya and Stuti have confirmed that they will take up Deal or No Deal
  - This updates the earlier plan
  - Their work focuses on decision-making under risk rather than the proposed
    learning, network, and adaptive-allocation extensions to my societal
    simulation
  - The earlier sections are retained as the original proposal
  - This update supersedes those assignments and the statements that their
    direction is still undecided

## Current Implementation Status

- Niveda's benchmark simulator is integrated through PR
  [\#695](https://github.com/gpsaggese/gpsaggese.github.io/pull/695)
  - Merged on 20 October into `UmdTask537_Review_students_research_projects`
  - It now lives in `research/Causal_Analysis_of_Agent_Skill_And_Luck/` as:
    - `agent_benchmark_utils.py`
    - Notebooks
    - Tests
  - The PR reports 21 passing benchmark tests and unchanged outputs for the same
    seeds
  - It does not establish completion of the real-agent pilot or integration into
    the default branch
- PR [\#709](https://github.com/gpsaggese/gpsaggese.github.io/pull/709) is open for
  the societal utilities and test cleanup
  - Niveda reports 74 tests passing with PyMC installed and unchanged notebook
    results
  - Niveda's correction in
    [\#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707)
    identifies the PyMC import skip and outdated imports in the test file
  - The proposed tests still skip when PyMC is unavailable
  - The PR reports unresolved dependency-installation conflicts
  - These are author-reported checks, not results I have independently rerun
- Still needed before calling the cleanup complete:
  - A reproducible environment
  - Full test discovery
  - Review of the notebook/README notation work
  - Reproduction or correction of the paper's numerical results
- The real-agent pilot and the state-of-the-art document remain proposed work
  - I will track the literature work in
    [\#706](https://github.com/gpsaggese/gpsaggese.github.io/issues/706)
  - I will track the notation/reproducibility work in
    [\#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707)
  - I will track the remaining cleanup tasks under
    [\#541](https://github.com/gpsaggese/gpsaggese.github.io/issues/541)
  - Merging a refactor does not resolve the reported Gini, event-count, or
    allocation-policy discrepancies by itself

## Updated Plan for Riya and Stuti

- Their research question: how to distinguish the quality of a contestant's
  decision from the luck of the resulting payoff
  - They will compare actual deal/no-deal choices with a decision benchmark
  - The benchmark is conditional on:
    - The information available at that round
    - Explicitly stated risk-preference assumptions
- First, obtain and document round-level contestant data and reproduce a relevant
  analysis from the existing literature
  - The initial output is a cleaned dataset containing:
    - Remaining case values
    - Banker offers
    - Observed choices
  - The initial output also includes an end-to-end example
  - The data will identify:
    - Contestants
    - Country editions
    - Round schedules
    - Actual decisions
    - Any hypothetical play after acceptance
  - Only actual decision rounds enter the stopping-choice analysis
  - Account for the fact that later rounds are observed only for contestants who
    continued
  - Fit and assess future-offer models on separate games or contestants
  - The evaluated decision uses only information available at that round, without
    later realized offers or case openings
- Next, implement and validate an optimal stopping benchmark under explicit
  assumptions about future banker offers
  - Begin with risk neutrality
  - Then examine how the recommended choices change under different risk
    preferences
  - Output: benchmark decisions and expected values or utilities for the observed
    game states
- Risk preferences:
  - Treat them first as prespecified sensitivity scenarios
  - If Riya and Stuti estimate preference parameters from choices, they will:
    - State the identification assumptions
    - Report uncertainty or compatible parameter ranges
    - Assess predictions on decisions excluded from estimation
  - A single choice may not identify a contestant's preferences
  - Model-based utility regret compares the observed action with the best action
    under the same stated preferences and offer model
    - It is not presented as an observed counterfactual loss
- Then, compare observed choices with the benchmark
  - Investigate whether earlier gains and losses are associated with later
    decisions after accounting for the current state
  - Test sensitivity to the offer and risk-preference assumptions
  - A low realized payoff does not by itself count as evidence of a poor decision
  - Any causal claim requires a separate identification argument

## How the Updated Studies Connect

- The common paper question becomes: how can we evaluate performance and decision
  quality without confusing favorable outcomes with skill or sound choices?
  - Niveda's study examines whether a coding-agent configuration looks better
    because of favorable task samples or execution randomness
  - Riya and Stuti's study examines whether contestants made reasonable choices
    given the risks, even when the eventual payoff was unfavorable
- I propose a shared framework with three quantities:
  - The realized outcome
  - The expected outcome conditional on the relevant configuration or choice and
    information
  - The difference between those two
- Decision quality is a separate comparison
  - It compares the expected value or utility of the chosen action with the best
    available alternative under that study's assumptions
- Caveats:
  - Finite-data estimates carry uncertainty
  - Expected performance can include favorable resources and circumstances as well
    as ability
- Both studies will ask when conclusions drawn from realized success change after
  accounting for the relevant conditional reference and uncertainty
  - Niveda will measure selected-configuration optimism, ranking stability, and
    selection regret using repeat runs and independent tasks
  - Riya and Stuti will measure model-based decision disagreement and conditional
    utility regret
    - They will also explain why realized payoffs can tell a different story
  - Their assessment depends on the preference and offer models
  - Niveda can empirically reevaluate a frozen configuration
  - We will keep those evidential differences explicit
  - We will not pool the outcomes into a universal percentage of luck or infer
    causality from a residual
- The paper will contain:
  - A shared problem statement and framework
  - A benchmark methods/results section
  - A Deal or No Deal methods/results section
  - A joint discussion of the evidence, assumptions, and limits
- Each team will reproduce a relevant published result and identify a specific
  contribution beyond it before expanding the study
- The common contribution is a tested account of how outcome-based evaluation
  differs from evaluation using conditional expectations and uncertainty
  - It is not a claim that one estimator applies everywhere
- My societal success work:
  - Remains prior methodological background
  - Serves as an illustrative case study, where reproduced and useful
  - Must be reproduced or corrected before supporting the paper's claims
  - Is no longer Riya and Stuti's extension roadmap
- GP's Law remains another possible research direction, outside the two current
  studies

## Immediate Sequence and Coordination

- First, I will review the two confirmed directions and proposed common argument
  with GP
  - Niveda will:
    - Verify the integrated benchmark baseline
    - Pursue the remaining cleanup and notation work
  - Riya and Stuti will:
    - Identify a usable Deal or No Deal dataset
    - Reproduce a published example
  - I will contribute to the societal baseline reproduction and paper/code
    discrepancies
  - The two studies can proceed independently where their dependencies allow
- Next:
  - Niveda will run the one-configuration pilot after the execution protocol and
    capped run budget are agreed
  - Riya and Stuti will validate their stopping benchmark and preference
    assumptions before analyzing deviations
  - We will review these initial outputs with GP
  - Then we will choose:
    - For Niveda: the configuration-comparison and selection experiments
    - For Riya and Stuti: the validated decision-quality analysis
  - We will write the methods and results as the evidence becomes available
- Coordination:
  - I will coordinate task ownership, answer questions, review dependencies, and
    maintain the common paper outline
  - Niveda will lead the benchmark work
  - Riya and Stuti will agree on an implementer and reviewer for their next task
    - I do not assume an individual split here
  - GP will supervise the research decisions and scientific claims
- GitHub issues:
  - [\#705](https://github.com/gpsaggese/gpsaggese.github.io/issues/705): this plan
  - [\#706](https://github.com/gpsaggese/gpsaggese.github.io/issues/706): state of
    the art
  - [\#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707): notation
    and reproducibility
  - [\#672](https://github.com/gpsaggese/gpsaggese.github.io/issues/672): Deal or No
    Deal
- Remaining actions:
  - Agree the remaining task owners and dates with the team
  - Collect availability for a recurring stand-up
  - Use the research project to track agreed tasks without duplicating existing
    issues
- Coordination steps still open:
  - Shared project-folder placement
  - Stand-up time
  - Detailed issue breakdown
  - Preparing this document alone does not complete all of Milestone 1
- After the paper is reviewed and its findings are validated, we can proceed to
  GP's publication and blog goals in
  [\#541](https://github.com/gpsaggese/gpsaggese.github.io/issues/541)
