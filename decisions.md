# Decision log

A record of the analysis decisions behind this project, newest first, covering what was decided and the reasoning behind it.
---

## ADR-0006 — Benchmark and comparison checks (Brent, diesel)

**Brent, same spec as `05` (K = 4).** Brent's cross-correlation decays the same way WTI's does, with lag 4 still above the noise band, so K = 4 applies without re-deriving it. The same-week gap is significant under Brent (0.37, p = 0.001), smaller than WTI's 0.43, but the short-run pattern remains the same. Week 1 is marginal (p = 0.053), consistent with week 1 being the least stable horizon across the other robustness checks in `05`.


**Diesel shows the same pattern as gasoline, not a weaker one.** 
Diesel's cross-correlation crosses into the noise band a lag earlier than gasoline's, lag 3 still above it, lag 4 inside it, so K = 3 was used instead of reusing K = 4 from `05`.
At K = 3, diesel's same-week gap is 0.69 (p = 0.002) and week-1 gap is 0.50 (p = 0.017), bboth larger than gasoline's 0.43 and 0.23, with stronger statistical evidence at h = 0 and a similar level of evidence at h = 1. 
Diesel is refined from the same crude but sold through a separate retail market from gasoline, so this points to the asymmetric pass-through pattern being broader than gasoline specifically, not confined to it. 
It does not change `05`'s gasoline result, which stands on its own evidence. 
Where in diesel's own supply chain this asymmetry originates was not tested here.

---

## ADR-0005 — Regional PADD comparison

`05` established the asymmetry using the national average across all five regions. 
That average could reflect a pattern that is broadly present across regions, or one driven mainly by a small number of markets. 
`08` refits the same model separately for each region to distinguish between these possibilities.

**PADD 5 against PADD 3, at the same week, with a \$0.10 threshold.** 
The pair, the horizon, and the threshold were all set before any model was fit. 
PADD 3 represents the more supply-connected market in this comparison, while PADD 5 is the more isolated market. 
If the proposed mechanism is real, the difference should be largest between these two.
Same-week is primary because `05` found same-week estimates survived every robustness check, while week-1 estimates did not.
\$0.10 is about 38% of the national same-week down-response (0.2629).

**Test the down-response, not the up-minus-down gap.** 
The original analysis focused on the difference between the up- and down-responses. 
For the competition hypothesis the down-response is more directly relevant, because the proposed mechanism predicts that weaker competition should
slow price decreases specifically, and says nothing about price rises. 
A gap test only detects a difference confined to one direction and is blind to both directions moving together, which is
what PADD 5 turned out to show. 
The down-response is tested on its own instead, with the up-response and their difference reported alongside it, which makes it possible to tell a true increase in asymmetry apart from a general slowdown in price adjustment.

**Difference the two regions instead of stacking all five into a panel.** 
Using the difference between the two regional retail series directly estimates how their responses differ. 
It also accounts for the fact that the two regions share common movements, which separate confidence intervals do not. 
Differencing removes movements that are common to both regions, such as nationwide shocks that affect them similarly.

---

## ADR-0004 — 2026 event-study design

The 2026 crude spike covers only 18 weeks of the analysis window, so a model that treats every week the same would give those weeks no special role.
This design tests whether the asymmetry itself behaved differently during that episode.

**Window: retail Mondays 2026-03-09 to 2026-07-06, 9 up weeks and 9 down.** 
Drawn from the cluster of weeks beyond 3 standard deviations of weekly Δcrude. 
2026 has 7 such weeks, the largest cluster in the sample, and this window captures a complete rise and reversal.

**Index the event dummies by when the crude shock landed, not when retail reacted.** 
Otherwise an ordinary pre-window shock gets tagged as an event shock whenever its delayed response falls inside the window.

**Keep a plain event dummy alongside the interactions**, so a level shift in Δretail unrelated to crude is not attributed to the interaction coefficients and misread as changed pass-through.

**Two specifications.** 
Spec A interacts the event indicator with the up/down terms at lags 0–4, giving 10 interaction coefficients estimated from 18 event weeks.
Spec B interacts lags 0 and 1 only, where 05 found the strongest evidence of asymmetry.

**Comparison windows were added as a robustness check.** 
If the pass-through relationship isn't perfectly linear in the size of the crude move to begin with, an unusually large move like 2026's could look different from an ordinary week for that reason alone, with nothing to do with 2026 specifically.
Rerunning the identical test on other large moves helps distinguish between these explanations. 2022 is the closest structural comparison. 
2020 is included as a weaker comparison because the COVID demand collapse coincided with the price crash, so any difference could reflect both effects.

---

## ADR-0003 — Cointegration and the asymmetric error-correction model

`05` works entirely in week-to-week price changes. 
That avoids the problem of getting a misleading relationship between two trending price series, but it also leaves out information about the price levels. 
In particular, the model cannot tell whether retail prices are currently high or low relative to crude prices. 
If crude and retail prices share a stable long-run relationship, that information should be included in the model. 
The short-run results also make this worth checking: the down-response peaks one week later and remains positive for several weeks rather than fading quickly. 
This raises the question of whether part of the observed response is related to longer-run adjustment that is not captured by the differences-only model.

This stage therefore checks whether a long-run relationship exists between crude and retail prices. 
First, the level series are tested for a unit root using ADF. If both series are I(1), the next step is a cointegration test using the Engle-Granger procedure. 
`coint()` was used for this rather than running a standard ADF test on the residuals from a levels regression. 
The residuals are estimated rather than observed, so the usual ADF critical values do not apply. 
`coint()` uses the appropriate Engle-Granger critical values.

If the series are cointegrated, the ECM adds an equilibrium-error term that measures how far retail prices are from their long-run relationship with crude. 
This error is then split into positive and negative parts so that the speed of adjustment can differ depending on whether retail prices
are above or below their long-run level.

The main question is whether retail prices move back toward equilibrium faster after a crude price increase or after a decrease. 
This provides a long-run counterpart to the short-run asymmetry tested in `05`.

Both adjustment coefficients are negative, indicating movement back toward the long-run relationship.
Their relative magnitudes point in the opposite direction from the rockets-and-feathers prediction: adjustment is nominally faster after a crude price decrease than after an increase. 
The difference between the two adjustment speeds is not statistically significant, so the ECM does not provide strong evidence of asymmetric long-run adjustment in either direction.

---

## ADR-0002 — Weekly alignment of retail against upstream

Retail is a Monday 8am survey snapshot. 
Weekly crude and spot are Friday-dated, and each value is the mean of that week's Monday through Friday, verified against the daily series. 
Pairing a retail Monday with the Friday of its own week pairs it with a window that includes four trading days that hadn't happened yet. 
This project is specifically about the timing of pass-through, so that leak doesn't just add noise. 
It shifts the whole estimated lag structure by a week.

Measured on the actual data: the same-week pairing gives corr(Δretail, Δcrude) = 0.209, the prior-week pairing gives 0.591. 
The contemporaneous relationship nearly triples under the correct alignment, because the leaky version pushes the real signal into the previous lag.

**Primary: pair each retail Monday with the last weekly Friday strictly before it.** 
That Monday-to-Friday window closes before the survey, so nothing in it postdates the survey. 
It also draws on FRED's own weekly series, already on disk.

**Robustness: the daily crude close of the last trading day before the survey.** Sharper timing, noisier value.

**Guardrails.** `src/point_in_time.py`'s `align_retail_to_upstream` is the only place a retail row is ever joined to an upstream row anywhere in this project. It checks that the upstream window ends before the survey date and raises if it doesn't. 
A test feeds it a deliberately leaky alignment and confirms the check catches it.

---

## ADR-0001 — FRED as the single data source

All series are pulled from FRED. 
Crude and wholesale-spot series are FRED's own, while the retail gasoline and diesel series are EIA data redistributed through FRED, so no direct EIA API call is made. Both publishers release under "Public Domain: Citation Requested."

Working with a single API simplifies the project. 
Maintaining a second one would add work without answering anything the FRED data cannot already answer here.

**Limit.** 
EIA's weekly retail survey gets revised after first publication, and FRED serves the latest revised vintage, so historical values reflect later revisions rather than the values available at the time of publication. 
This is appropriate for a historical price-transmission analysis, but it would not be appropriate for a study that aims to reproduce the information available to researchers or market participants at a specific historical date.