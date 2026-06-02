# Control Plane Link — qrl-architecture-comparison

**Date:** 2026-04-17
**Purpose:** Name the control surfaces that govern this repo, so
anyone reading this file can follow the authority trail.

This repo is governed by the three-surface control plane model. It is
not an independent artifact — decisions about its role, its exposure,
and its classification are recorded in the control surfaces listed
below. Edits here must be reconciled with S2; updates to S2 are
mirrored to S3 for ChatGPT continuity.

## Authority surfaces

### S2 — strategic-control-plane (durable repo truth)

- **Path:** `~/Documents/GitHub/strategic-control-plane/`
- **Canonical state:** `CONTROL_PLANE_STATE.md`
- **Delta log:** `CONTROL_PLANE_DELTA_LOG.md`
- **This repo's row in S2:** under MyControl lane and its qrl-
  architecture-comparison subsection (see the 2026-04-17 entry
  covering the integration pass).

### S3 — chatgpt-comprehensive-scp (ChatGPT continuity)

- **Path:** `~/Documents/GitHub/chatgpt-comprehensive-scp/`
- **Canonical state:** `SCP/CURRENT_STATE.md`
- **Delta log:** `SCP/DELTA_LOG.md`
- **This repo's narrative:** integrated into the MyControl /
  architecture-evidence narrative on 2026-04-17.

### S1 — Organized_Control_Plane (artifact reality)

- **Path:** `~/Documents/Organized_Control_Plane/`
- **This repo's evidence placement:** the canonical lab runtime is
  this git repo; S1 carries a pointer record under
  `MyControl/Architecture_Comparison/` once the 2026-04-17 integration
  evidence is promoted from the S2 staging area (see the 2026-04-17
  delta entry in S2 for the staging path — the convention is that S1
  is updated via user-mediated `cp -r` from the S2 staging tree).

## Governance rules applied to this repo

1. **Classification is authoritative.** The four buckets defined in
   `INTEGRATION_CLASSIFICATION.md` (KEEP_AS_LAB, PORT_TO_MYCONTROL,
   SUPPORTS_PHOENIX, DISCARD) govern how any downstream decision
   treats this code. Disagreements get settled by updating the
   classification doc via an S2 delta entry, not by ad-hoc edits.

2. **No silent promotion.** Nothing in `SUPPORTS_PHOENIX` moves into
   Phoenix without an S2 delta entry and a corresponding Phoenix
   doctrine / contract update.

3. **No silent downgrade.** Renaming a file from lab → production
   requires reclassification through this document's system.

4. **Exposure is governed.** `EXPOSURE_SPEC.md` is the contract for
   what the frontend may and may not claim. Changes to the
   presentation narrative — in this repo OR in any MyControl page
   that links here — must respect that contract.

5. **Phoenix is cite-linked, not dependency-linked.** Phoenix does
   not import from this repo and this repo does not ship as part of
   Phoenix. Citations flow one way, Phoenix → this repo, as
   methodology references.

## Changes that require control-plane updates

| Change | S1 | S2 | S3 |
|---|---|---|---|
| Reclassifying a file in `INTEGRATION_CLASSIFICATION.md` | — | delta + state if material | delta if material |
| Promoting a file to `PORT_TO_MYCONTROL` actual port | — | lane file + delta | mirror delta |
| Changing the exposure posture (how MyControl page describes this) | — | delta + lane | delta + lane |
| Retiring this repo | S1 record | state + delta | state + delta |
| New Phoenix citation of this repo (new docs pointer) | — | delta (minor) | delta (minor) |
| Pure lab change (comparison UI, signal tuning) | — | — | — |

Pure lab-internal changes do NOT require control-plane updates. Scope
changes, classification changes, exposure changes, and cross-repo
linkage changes do.

## How a future agent / reader should use this file

If you are reading this as part of booting a new session, agent, or
reviewer context:

1. **Do not treat this repo as independent.** It is governed.
2. **Check S2 first** for the current strategic posture of this repo.
3. **Check S3** for ChatGPT-layer continuity.
4. **Check this repo's three integration docs** (`INTEGRATION_CLASSIFICATION.md`,
   `SYSTEM_PLACEMENT.md`, `EXPOSURE_SPEC.md`) for the local contract.
5. **Only then** make decisions about code, exposure, or linkage.

That order is the control-plane authority order. Do not reverse it.
