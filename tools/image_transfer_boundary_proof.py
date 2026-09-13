"""Exact-rational propagation of the documented conservative DD error budget.

This checks the recurrence and margin arithmetic, not the compiler, hardware or
the implementation's adherence to the primitive error hypotheses. Those remain
explicit review and qualification obligations.
"""
from fractions import Fraction


def error_budget():
    u=Fraction(1,1<<53)
    # 16384*u^2 exceeds the composed primitive bounds listed in the design.
    eta=16384*u*u
    grid=1<<120
    def upward(x):return Fraction((x.numerator*grid+x.denominator-1)//x.denominator,grid)
    error=Fraction(1)
    for _ in range(24):
        if error>Fraction(1,32):
            t=1+error
            exact=(4*t+1/t**4)/5-1
        else:
            # F''(t)=4/t^6; for |t-1|<=1/4 its Taylor bound is <12*error^2.
            exact=12*error*error
        # Exact Newton maps positive values to >= the root; a rounded step can
        # undershoot only by eta. This term bounds that small negative side.
        error=upward(max(exact,12*eta*eta)+eta)
    propagated=upward((1+error)**12*(1+4096*u*u)-1)
    return dict(unit_roundoff=u,per_step_perturbation=eta,root_relative_bound=error,
                normalized_value_relative_bound=propagated,claimed_value_budget=Fraction(1,1<<84),
                cutoff_margin=Fraction(1,1<<80),iterations=24,
                scope='Conditional analytic bound; requires reviewed primitive hypotheses, strict FP and independent qualification')
