# FedHMMs
 ================================================================
Federated Hidden Markov Models for Privacy-Preserving Analysis of
Governance Action Sequences.

Implements the four algorithms described in:
  Hemmati & Bennani, "Federated Hidden Markov Models for
  Privacy-Preserving Analysis of Governance Action Sequences", 2025.

Algorithms
----------
  [0] Centralised     : Baum-Welch on all pooled data (oracle upper bound)
  [1] Local-HMM       : independent Baum-Welch per user (privacy baseline)
  [2] FedAvg-HMM      : Algorithm 2 — weighted parameter averaging
  [3] FedEM-HMM       : Algorithm 1 — raw sufficient-statistics aggregation
  [4] Personalised    : Algorithm 3 — FedEM global + local fine-tuning + α-interpolation

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
