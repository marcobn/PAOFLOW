# PAOFLOW contribution model (proposed)

This project follows a **Gitflow-inspired** workflow with two long-lived branches:

- `master` is the **stable, released** branch intended for public consumption.
- `develop` is the **integration** branch where new features and bug fixes land.

The goal is simple: keep development work and released work separated so users can
rely on `master`, while developers can move quickly on `develop`.

## Branch roles (invariants)

**`master`**
- Contains only **release-quality** code.
- Moves forward only when `develop` is promoted and tagged.
- Branch protections will ensure merges to `master` come **only** from `develop`.

**`develop`**
- The shared branch for ongoing work.
- All feature branches merge back into `develop` (via PR).
- May be messy at times, but should not remain broken indefinitely.

## Develop branch and naming conventions

Contributors are expected to branch from `develop` and merge back into `develop`
using the naming convention:

- `develop/<feature_name>`

Branch protections will be enabled on `develop` so that PRs require 
at least one approval.

## Release cadence and promotion to `master`

On a periodic basis (according to our release cadence), the `develop` branch will
be run through a standardized test suite. Upon successful completion:

1. `develop` is merged into `master`
2. the merge commit on `master` receives a **release tag** (e.g., `v1.0.0`)
3. GitHub Release is created from that tag

This makes every public release on `master` reproducible by tag.

## Simplifications relative to Gitflow

This is based on the
[Gitflow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).
However, PAOFLOW uses a simplified model:
- We do not maintain a separate long-lived `release` branch.
- Bug fixes follow the same path as features: `develop/<...>` → `develop` → `master`.

## A simple example

```mermaid
---
config:
  gitGraph:
    mainBranchName: 'master'
---
gitGraph TB:
    commit id: ' '
    checkout master
    branch develop
    checkout master
    branch develop/feature
    commit id: 'feature-0'
    checkout develop
    merge develop/feature
    checkout master
    merge develop tag: 'v1.0.0'
    checkout develop/feature
    commit id: 'feature-1'
    checkout develop
    merge develop/feature
    checkout master
    merge develop tag: 'v1.0.1'
```

## The reality (merge conflicts, drift, and other forms of entropy)

Development is messy. Merge conflicts will still happen. The point of this model
is not to eliminate mess, but to contain it: develop absorbs change, while
master stays stable and release-tagged.

A common source of pain is long-lived feature branches drifting away from develop.
When that happens, the fix is to sync the feature branch with develop first
(resolve conflicts on the feature branch), then merge into develop.

## A more realistic example
```mermaid
---
config:
  gitGraph:
    mainBranchName: 'master'
---
gitGraph TB:
    commit id: ' '
    branch develop
    branch develop/jon
    commit id: 'jon-0'
    commit id: 'jon-1'
    merge develop id: 'oops'
    commit id: 'jon-1'
    checkout develop
    branch develop/marcio
    commit id: 'marcio-0'
    checkout develop/jon
    commit id: 'jon-2'
    checkout develop
    merge develop/marcio id: "develop-1"
    checkout develop
    commit id: "develop-1 "
    branch develop/marco
    commit id: 'marco-0'
    checkout develop
    merge develop/jon id: "REJECTED: merge conflicts" type: REVERSE
    checkout develop/jon
    merge develop id: 'git merge origin/develop'
    commit id: 'fix-1'
    checkout develop/marco
    commit id: 'marco-1'
    checkout develop
    merge develop/jon id: "develop-2"
    checkout develop/marco
    merge develop id: 'no issues'
    checkout develop/marcio
    merge develop
    checkout master
    merge develop tag:'v.1.0.0'
    checkout develop
    merge develop/marco id: "develop-3"
    checkout develop/marcio
    commit id: 'marcio-1'
    checkout master
    merge develop tag:'v.1.0.1'
    checkout develop/marco
    commit id: 'marco-3'
```
