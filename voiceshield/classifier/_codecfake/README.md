# Vendored Codecfake countermeasure

`model.py` is copied VERBATIM from the Codecfake repo
(https://github.com/xieyuankun/Codecfake, arXiv:2405.04880) so that the
pickled `W2VAASIST` checkpoint can be unpickled. Do not edit it (the upstream
license is CC BY-NC-ND 4.0 — NonCommercial, NoDerivatives).

The `pytorch_model_summary` import in model.py is satisfied by a runtime stub
(see codec_scorer.py); we do not add it as a real dependency.

Weights (`models/codecfake/.../anti-spoofing_feat_model.pt`) are NOT committed
(gitignored) — obtain them from the upstream repo. NonCommercial: hackathon /
research use only, not for a shipped product.
