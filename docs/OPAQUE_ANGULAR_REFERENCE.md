# Independent angular-boundary reference

This is development reference machinery,not a changed production renderer.
It certifies fixed packed geometry and constant-albedo derivatives only.
A complete original pixel still has to fit90seconds,256cells,depth10,
32visible pieces and1e-12 outward radius. Native tolerance stays1e-10+1e-9relative.

For E=Q-I on a unit interval,define
K_m(t)=E[(x-t)_+^(m-1)/(m-1)!]. Applying the piecewise Taylor identity gives
E(f)=integral K_3(t)f^(3)(t)dt + sum_jumps sum_{j=0}^2 [f^(j)] K_(j+1).
Thus discontinuities are accounted for explicitly rather than differentiated
as if the whole lookup were smooth. On an interval of length h,the ordinary
term scales as h^4 and jump j as h^(j+1). Positive tensor-product weights give
the two-dimensional bound by applying each one-dimensional error in turn.

The existing Gauss5 nodes/weights are enclosed by outward Decimal60 arithmetic.
Each kernel is a degree<=3 polynomial between adjacent nodes. Four levels of
exact de Casteljau subdivision give Bernstein convex-hull suprema and an upper
bound on its absolute integral; no sampled-convergence estimate is used.
The caller must enclose all ordinary third derivatives and every jump on every
line through the box. Monotonicity proves at most one crossing. Otherwise a
noncurved radial triangle has affine homogeneous vectors along each coordinate;
a squared view-cosine knot is quartic and a light knot quadratic. Identically
constant-knot lines cause no branch switching. Unresolved curved crossings refuse.

Normal/view dot products are expanded exactly as rational polynomials before
interval evaluation. The half-vector identity v.h=sqrt((1+v.l)/2) is used only
for normalized nonopposite vectors. These reductions alter neither the material
nor the error threshold. Moving angular domains need their own transport proof.

Opaque64 preserves two90-second full-reference failures. The second completed
face5739,visible piece0,using183cells;it stopped on piece1. The saved first-piece
certificate is reusable evidence,not permission to restart the failed pixel's
budget. CPU/CUDA comparisons name only that certified piece. Complete original
pixels,frames and the final showcase are still unqualified.

Background: [Peano kernel theorem,section13.2](https://www.math.utah.edu/~feldman/teaching_files/F18/natwo.pdf)
and [NIST DLMF Gauss quadrature](https://dlmf.nist.gov/3.5#v).
The piecewise jump formula above follows by adding truncated Taylor jump terms
before applying the linear functional;the implementation tests exact step,ramp,
curvature-jump,scaled-box and log-polynomial controls independently.
