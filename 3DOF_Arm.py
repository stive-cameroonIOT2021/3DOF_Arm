#!/usr/bin/env python3
"""
Inverse kinematics for a 2-link planar arm with a vertical base offset.

Angles (degrees):
  - alpha: base yaw in the XY plane (your choice: atan(Y/X), 0 when X=Y=0)
  - Theta: shoulder elevation (elbow-down configuration)
  - beta : elbow internal angle

Geometry:
  L1, L2   : link lengths (> 0)
  Z_offset : shoulder height relative to world Z=0

Formulas (conventional vertical):
  L      = sqrt(X^2 + Y^2)
  z_eff  = Z - Z_offset
  r      = sqrt(L^2 + z_eff^2)
  theta1 = acos( (L1^2 + r^2 - L2^2) / (2 L1 r) )
  theta2 = asin( z_eff / r )
  Theta  = degrees(theta1 + theta2)          # elbow-down
  beta   = degrees( PI - acos( (L1^2 + L2^2 - r^2) / (2 L1 L2) ) )

Now supports the vertical target case X=0, Y=0 (alpha is set to 0 by convention).
"""

import math
from typing import Tuple, Optional

# ===== Default robot geometry (mm) =====
DEFAULT_L1 = 88.3
DEFAULT_L2 = 51.2
DEFAULT_Z_OFFSET = 10.0

# (Optional) joint limits in degrees; set to None to skip a limit
ALPHA_MIN: Optional[float] = 0.0
ALPHA_MAX: Optional[float] = 180.0
THETA_MIN: Optional[float] = -90.0
THETA_MAX: Optional[float] = 90.0
BETA_MIN:  Optional[float] = 0.0
BETA_MAX:  Optional[float] = 180.0


# ---------- helpers ----------
def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi] to prevent tiny float drift breaking acos/asin."""
    return lo if x < lo else hi if x > hi else x

def _check_limit(name: str, val: float, vmin: Optional[float], vmax: Optional[float]) -> None:
    """Optionally enforce a joint limit."""
    if vmin is not None and val < vmin:
        raise ValueError(f"{name} below limit: {val:.2f} < {vmin:.2f} deg")
    if vmax is not None and val > vmax:
        raise ValueError(f"{name} above limit: {val:.2f} > {vmax:.2f} deg")


# ---------- core IK ----------
def inverse_kinematics(
    X: float,
    Y: float,
    Z: float,
    *,
    L1: float,
    L2: float,
    Z_offset: float = 0.0,
    enforce_limits: bool = False,
) -> Tuple[float, float, float]:
    """
    Compute (alpha, Theta, beta) in degrees.

    Accepts X=Y=0 (vertical target). In that case, alpha is defined as 0 by convention.
    """
    # ---- input validation ----
    if X is None or Y is None or Z is None:
        raise ValueError("X, Y, Z must be provided")
    if X < 0 or Y < 0:
        raise ValueError("X and Y must be non-negative")
    if L1 <= 0 or L2 <= 0:
        raise ValueError("L1 and L2 must be > 0")
    if not all(map(math.isfinite, (X, Y, Z, L1, L2, Z_offset))):
        raise ValueError("All inputs must be finite numbers")

    # ---- base yaw alpha ----
    # If X and Y are both zero (vertical), alpha is undefined; choose 0.0 by convention.
    if math.isclose(X, 0.0, abs_tol=1e-12) and math.isclose(Y, 0.0, abs_tol=1e-12):
        alpha = 0.0
        L = 0.0
    else:
        # atan(Y/X) with a tiny guard when X≈0
        X_safe = X if not math.isclose(X, 0.0, abs_tol=1e-12) else 1e-6
        alpha = math.degrees(math.atan(Y / X_safe))
        L = math.hypot(X, Y)  # sqrt(X^2 + Y^2)

        # Early XY-plane reach check: if horizontal distance already exceeds L1+L2, unreachable.
        if L > (L1 + L2):
            raise ValueError("Target is out of reach in the XY plane")

    # ---- vertical relative to shoulder plane & 3D radius ----
    z_eff = Z - Z_offset
    r = math.hypot(L, z_eff)  # sqrt(L^2 + z_eff^2)

    # Full 3D reachability (triangle inequality)
    if r > (L1 + L2):
        raise ValueError("Target is out of reach (too far).")
    if r < abs(L1 - L2):
        raise ValueError("Target is out of reach (too close).")

    # ---- shoulder geometry (theta1) ----
    # Guard against r=0: this means the target is exactly at the shoulder.
    # A pose exists only if L1 == L2 (elbow fully folded). We handle numerically anyway.
    if math.isclose(r, 0.0, abs_tol=1e-12):
        cos_theta1 = 1.0  # theta1 = 0
    else:
        cos_theta1 = _clamp((L1*L1 + r*r - L2*L2) / (2.0 * L1 * r), -1.0, 1.0)
    theta1 = math.acos(cos_theta1)  # radians

    # ---- elevation of target line (theta2) ----
    sin_theta2 = _clamp(0.0 if math.isclose(r, 0.0, abs_tol=1e-12) else (z_eff / r), -1.0, 1.0)
    theta2 = math.asin(sin_theta2)  # radians (sign follows z_eff)

    # ---- shoulder elevation (elbow-down) ----
    Theta = math.degrees(theta1 + theta2)

    # ---- elbow internal angle (beta) ----
    cos_beta = _clamp((L1*L1 + L2*L2 - r*r) / (2.0 * L1 * L2), -1.0, 1.0)
    beta = math.degrees(math.pi - math.acos(cos_beta))  # 0..180°

    # ---- optional joint limit enforcement ----
    if enforce_limits:
        _check_limit("alpha", alpha, ALPHA_MIN, ALPHA_MAX)
        _check_limit("Theta", Theta, THETA_MIN, THETA_MAX)
        _check_limit("beta",  beta,  BETA_MIN,  BETA_MAX)

    return (alpha, Theta, beta)


# ---------- tiny CLI to test quickly ----------
def _demo_cli() -> None:
    print("IK demo — enter X Y Z (mm). Type 'q' to quit.")
    print(f"Using L1={DEFAULT_L1} mm, L2={DEFAULT_L2} mm, Z_offset={DEFAULT_Z_OFFSET} mm")
    while True:
        try:
            raw = input("> ").strip()
            if raw.lower() in {"q", "quit", "exit"}:
                break
            x, y, z = map(float, raw.split())
            a, T, b = inverse_kinematics(
                x, y, z,
                L1=DEFAULT_L1, L2=DEFAULT_L2, Z_offset=DEFAULT_Z_OFFSET,
                enforce_limits=False,  # set True if you want joint limit checks
            )
            print(f"alpha = {a:.2f} deg,  Theta = {T:.2f} deg,  beta = {b:.2f} deg")
        except ValueError as e:
            print("Input Error:", e)
        except Exception as e:
            print("Unexpected error:", e)


if __name__ == "__main__":
    _demo_cli()
