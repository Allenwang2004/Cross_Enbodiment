"""Bilevel (upper=RetargetNet / lower=RL) cross-embodiment training.

See proposal.md at the repo root for the full design. The short version:

    Upper (phi = RetargetNet):  produces the reference motion at RUNTIME from a
        36-dim global correction conditioned only on the morphology vector beta.
        phi = 0 reproduces scripts/qpos_retarget.py:91 retarget_qpos exactly.
    Lower (psi = LatentAdapter + ActionHead + RootWrenchHead): PPO over 24-step
        windows maximizing tracking + regularization + survival reward.

The two are coupled by TTSA: the lower level updates every iteration, the upper
level every K=10, with eta_phi/eta_psi = 1/30. The hypergradient is truncated
(see upper.py) -- we keep the exact reference-side term and estimate the
simulator-side term with antithetic ES; the learning-response term is dropped.
"""
