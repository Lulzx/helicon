# Physics Documentation

## Magnetic Nozzle Physics

MagNozzleX simulates plasma expansion through a diverging magnetic field.
The nozzle converts upstream thermal energy into directed exhaust momentum via two mechanisms:

1. **Ion detachment** — ions have large Larmor radii (r_L >> L_B at the throat exit) and decouple from the field ballistically
2. **Electron detachment** — electrons are light and magnetized; their demagnetization drives the hard physics

### The Detachment Problem

The frozen-in flux theorem (ideal MHD) predicts that magnetized plasma follows field lines indefinitely.
Real detachment occurs when the approximations break down:

| Mechanism | Condition | Physics |
|---|---|---|
| **Inertial detachment** | r_L,i > L_B | Ion Larmor radius exceeds field scale → ballistic |
| **Resistive detachment** | Ω_e τ_e ~ 1 | Electron collision rate comparable to gyrofrequency |
| **Pressure anisotropy** | A = P_⊥/P_∥ - 1 ≠ 0 | Fire-hose instability → anomalous transport |
| **Ambipolar field** | E_r coupling | Electron pressure drives radial electric field |

### Detachment Efficiency Definitions

MagNozzleX computes and reports three definitions (all different in the literature):

$$\eta_d^{\text{momentum}} = \frac{\dot{p}_{\text{exit, axial}}}{\dot{p}_{\text{injected}}}$$

$$\eta_d^{\text{particle}} = \frac{N_{\text{exit, axial}}}{N_{\text{injected}}}$$

$$\eta_d^{\text{energy}} = \frac{E_{k,\text{axial, exit}}}{E_{k,\text{injected}}}$$

Always specify which definition when comparing with literature.

## Analytical Models

### Paraxial Thrust Coefficient (Little & Choueiri 2013)

For a polytropic electron gas (index γ) expanding through mirror ratio R_B:

$$C_T = \sqrt{\frac{2(\gamma+1)}{\gamma-1}} \cdot \sqrt{\eta_T}$$

$$\eta_T = 1 - \left(\frac{1}{R_B}\right)^{(\gamma-1)/\gamma}$$

$$\theta_{\text{div}} = \arcsin\left(\frac{1}{\sqrt{R_B}}\right)$$

### Hall Parameter (Resistive Detachment)

The electron Hall parameter from Spitzer collision theory:

$$\Omega_e \tau_e = \frac{eB/m_e}{\nu_{ei}} \quad \text{where} \quad \nu_{ei} = \frac{n e^4 \ln\Lambda}{3 \varepsilon_0^2 \sqrt{2\pi} m_e^{1/2} (k_B T_e)^{3/2}}$$

Resistive detachment onset: Ω_e τ_e ~ 1.

## Biot-Savart Field Computation

For a circular coil at (z_c, a) with current I, the on-axis field:

$$B_z(0, z) = \frac{\mu_0 I a^2}{2(a^2 + (z-z_c)^2)^{3/2}}$$

Off-axis (exact, using elliptic integrals K(k²) and E(k²)):

$$B_z = \frac{\mu_0 I}{2\pi \alpha} \left[ K(k^2) + \frac{a^2 - r^2 - \delta z^2}{\beta^2} E(k^2) \right]$$

where α² = (r+a)² + δz², β² = (r-a)² + δz², k² = 4ar/α².

## References

1. Breizman & Arefiev (2008), PoP 15, 057103 — paraxial nozzle model
2. Little & Choueiri (2013), PoP 20, 103501 — thrust coefficient
3. Merino & Ahedo (2016), PoP 23, 023506 — 2D collisionless nozzle
4. Moses, Gerwin & Schoenberg (1991), AIP CP 246 — resistive detachment
5. Olsen et al. (2015), IEEE TPS 43, 252 — VASIMR experimental data
