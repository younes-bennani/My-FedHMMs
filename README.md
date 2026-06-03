"""
fedhmm.py
═════════════════════════════════════════════════════════════════════
Federated Hidden Markov Models for Privacy-Preserving Analysis of
Governance Action Sequences.

Implements the four algorithms described in:
  Y. Hemmati & Y. Bennani, "Federated Hidden Markov Models", 2026.

Algorithms
----------
  [1] Centralised             : Baum-Welch on all pooled data (oracle upper bound)
  [2] Local-HMM               : independent Baum-Welch per user (privacy baseline)
  [3] FedAvg-HMM              : weighted parameter averaging
  [4] FedEM-HMM               : raw sufficient-statistics aggregation
  [5] Personalised (DITTO)    : FedEM global + local fine-tuning + α-interpolation
  [6] FedProx-HMM             : FedAvg with proximal regularisation (Li et al., 2020).
  [7] FedMA-HMM               : Federated Matched Averaging for HMMs
  [8] CoordMedian-HMM         : Coordinate-wise median aggregation (Yin et al., 2018).


Mathematics (matching the paper exactly)
-----------------------------------------
  Observation alphabet : X = {ADD=0, UPDATE=1, DELETE=2}
  HMM parameters       : θ = (π, A, B)
                         π ∈ ΔK-1, A ∈ [0,1]^{K×K}, B ∈ [0,1]^{K×|X|}
  Sufficient statistics:
    Γ_i   = Σ_s γ^(s)_1(i)
    Ξ_{ij} = Σ_s Σ_{t=1}^{T_s-1} ξ^(s)_t(i,j)     ← t up to T_s-1
    Φ_{io} = Σ_s Σ_{t=1}^{T_s}   γ^(s)_t(i) 1[o^(s)_t = o]
  M-step   : π̂ = Γ/‖Γ‖₁,  Â_{ij} = Ξ_{ij}/Σ_{j'}Ξ_{ij'},  B̂_{io} = Φ_{io}/Σ_{o'}Φ_{io'}
  FedEM    : Γ = Σ_i Γ^(i)  (simple sum, no normalisation)
  BIC      : -2 log L̂ + p(K)·ln(N_tot),  p(K) = (K-1) + K(K-1) + K(|X|-1)

Usage
-----
  python fedhmm.py sessions.jsonl [output.xlsx]

  Default output: fedhmm_results.xlsx
"""


 ================================================================
