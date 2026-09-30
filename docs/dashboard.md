# Dashboard and dispatch output

The Dashboard is an operational view, not a second calculation engine. It reads saved predictions,
metrics, station metadata, and the latest status snapshot.

## Map

Each station shows forecast time, 30-minute demand, observed holdout value when in evaluation mode,
current bikes/docks when available, and demand band. When the classification model exists, replace the
demand band with separate empty/full risk colors and show calibrated probabilities.

## Dispatch queue

The initial priority score is expected lost trips:

```text
P(empty) × predicted departures + P(full) × predicted arrivals
```

The queue should expose every component rather than only a composite score. A later routing layer can
combine priority, recommended bike movement, truck capacity, travel time, and service deadlines.

## Drift panel

Show the current error window, reference error, ratio, important PSI values, last training time, last
trigger reason, and whether enough labels have arrived to make a performance decision.

