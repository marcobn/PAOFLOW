# 05: Sweep comparison and branch assignments

```
Call `sweep_comparison(...)` (min_copies from the inventory, and again with min_copies=100). Using the overlay, the
overlay on the experimental map and per_angle_listing.txt:
1. Report the pointwise match rate with BOTH chance baselines and say whether it is above chance.
2. Assign experimental branches to calculated groups from branch SHAPE over the rotation (frequency and
   slope over several angles), carrier type, curvature and mass. For each: deviation range, angular range,
   max/min, 1 + lambda where masses exist, confidence (secure / medium / tentative / unexplained).
3. Flag isolated calculated points that only match at one or two angles.
4. Write <project folder>/data/branch_assignments.csv (include unassigned experimental branches) and re-run
   `sweep_comparison(assignments=...)` so the overlay shows the labels.
Summarise as a table and list the unexplained branches in priority order.
```

**Check:** no assignment rests on a single angle; the unexplained list is honest.
