# Building the German BESS Forecast Value Monitor in public

This document is the public product log for the research app. It records what is
working, what is still uncertain, and what will be built next. Claims move into
the app only when the supporting artifact and test are committed with them.

## Product question

How much economic value does a forecast add to a German day-ahead battery
schedule when every model is evaluated through the same physical dispatch
problem and the same information cutoff?

## Current public milestone

- Interactive forecast-value ladder on a common 1,542-day sample.
- One-day dispatch explorer with configurable power, duration, efficiency and
  throughput cost.
- Versioned public price data, KPI artifacts and model-comparison table.
- Explicit separation of executable schedules from the perfect-foresight ceiling.
- Visible limitations, negative findings and data-quality post-mortem.
- Unit tests for dashboard transformations and dispatch settlement.

## Roadmap

### Now — trustworthy historical monitor

- [x] Decision-oriented landing page
- [x] Model information-set table
- [x] Artifact provenance and reproduction contract
- [x] Public methodology article and repository links
- [x] Honest caveats next to headline metrics

### Next — genuine forward evidence

- [ ] Accumulate a meaningful set of pre-auction frozen schedules
- [ ] Publish settled forward capture by model tier
- [ ] Show missed runs and data-health failures instead of backfilling them
- [ ] Add a build timestamp and machine-readable monitor status artifact

### Later — operational realism

- [ ] Rebuild forecasts from point-in-time ENTSO-E publications
- [ ] Move the dispatch horizon to 15-minute market time units
- [ ] Add a licensed or registered German intraday price series
- [ ] Add degradation and cycle-budget scenarios beyond a scalar throughput cost

## Public update template

Each update should answer four questions:

1. What decision or failure motivated the change?
2. What changed in the model, data or product?
3. What evidence verifies the result?
4. What limitation remains?

## Changelog

### 2026-10-08 — Product credibility layer

- Renamed the public experience to **German BESS Forecast Value Monitor**.
- Added an executive decision brief and model information-set table.
- Added artifact provenance, reproduction and interpretation contracts.
- Added repository and research-note entry points.
- Added tested artifact-status logic and this public roadmap.

### 2026-07 — Reproducible research baseline

- Published the Streamlit explorer and compact public price dataset.
- Added forecast-tier comparison, data-integrity checks and automated tests.
- Documented the mislabeled-series investigation and removed invalid intraday claims.
