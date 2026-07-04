# MACRO — Project Page

Project page for **MACRO: Training-free Multi-plane Attention for Closeup Render Optimization**.

Live site: https://nitzanhod.github.io/MACRO/

Built on the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)
(Bulma-based, plain HTML + `.nojekyll`).

## Structure

- `index.html` — the whole page (hero, TL;DR, teaser, abstract, scale-gap analysis,
  method, qualitative comparisons, trajectory videos, K×M sweep, BibTeX). The trajectory
  video tiles are generated client-side from the `SCENES` list at the bottom of the file.
- `static/images/` — teaser, method, scale-analysis, qualitative tiles (DS1/DS3), sweep.
- `static/videos/` — `<ds>_<scene>_<traj>_<method>.mp4`, where
  `ds ∈ {ds1, ds3}`, `traj ∈ {wcw, ztf}`, `method ∈ {3dgs, difix, macro}`.
- `static/css/` — Bulma + fontawesome (from the template).

## Local preview

```bash
python3 -m http.server 8891   # then open http://localhost:8891/
```

## Editing

- **Add/replace a trajectory scene:** add the six mp4s to `static/videos/` following the
  naming convention, then add an entry to the `SCENES` array in `index.html`.
- **arXiv link:** currently a placeholder (`href="#"`, label "arXiv (soon)"). Replace with
  the real `https://arxiv.org/abs/<id>` once assigned.
- **Asset provenance:** figures come from the paper source; trajectory videos and
  qualitative tiles were pulled from the project's EFS results (`ssh nitz`). See
  `Difix3D/docs/project_page_handoff.md` for the authoritative asset map.
