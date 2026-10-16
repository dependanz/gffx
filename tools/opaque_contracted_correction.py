"""Experimental complete correction with contracted crossing geometry.

Historical proofs and campaign defaults remain unchanged. Recompute physical
jump jets on each contracted support; retain all jump orders and product terms.
"""
import math
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_horizon_correction import HorizonMaterialCorrection
from tools.opaque_peano_material import difference,segmented
from tools.opaque_root_contraction import contract_root

class ContractedMaterialCorrection(HorizonMaterialCorrection):
    contractions=0
    def root(self,box,axis,role,knot,*,initial=None,clipped=False,inputs=None):
        inputs=inputs or self.inputs
        def evaluate(domain):
            args=[Jet(v,order=1) for v in box];args[axis]=Jet.variable(domain,order=1)
            q=inputs(*args)[role];return q[0,0],q[1,0]
        self.contractions+=1
        return contract_root(box[axis],evaluate,knot,initial=initial,clipped=clipped,checkpoint=self.checkpoint)
    def _axis_bounds(self,box,axis,*,input_values=None):
        rows=super()._axis_bounds(box,axis,input_values=input_values);inputs=input_values or self.inputs;e=self.extension;kept=[]
        for old in rows[1]['localized_jumps']:
            role=old['role'];k=old['knot'];position=self.root(box,axis,role,e.cn[k],initial=old['position'],inputs=inputs)
            if position is None:continue
            args=[Jet(v,order=3) for v in box];args[axis]=Jet.variable(position,order=3);narrow=inputs(*args)
            crossing=narrow[role];other=narrow[1-role];other_actual=[a+b for a,b in zip(self.smooth.at(other,e.indices[1-role]),segmented(e,other,e.indices[1-role]))]
            coeff=dict(crossing.c);coeff[0,0]=e.cn[k];at=Jet(order=3,coeff=coeff);delta=difference(e,at,k,k-1)
            light=narrow[1] if role==0 else at;jump=light*self.mix(delta)*self.mix(other_actual)*narrow[3]/(e.pi*self.mean)
            w=dict(old,position=position,jumps=[I(jump[j,0].bound)*math.factorial(j)*old['max_crossings'] for j in range(3)])
            if 'signed_jumps' in old:w['signed_jumps']=[jump[j,0]*math.factorial(j)*old['orientation'] for j in range(3)]
            kept.append(w)
        rows[1]['localized_jumps']=kept
        return rows
    def integrated_bounds(self,box,axis,*,diagnostic=None):
        """Integrate monotone knot witnesses along the transverse coordinate.

        The change of variable is admitted only when both coordinate derivatives
        strictly exclude zero over the entire box. Other witnesses keep the
        existing conservative whole-transverse bound and multiplicity.
        """
        from tools.opaque_transverse_peano import clamped_crossing,weighted_kernel_parts
        rows=self._axis_bounds(box,axis);transverse=1-axis
        derivatives=[]
        for active in (axis,transverse):
            args=[Jet(box[j],order=1) for j in (0,1)];args[active]=Jet.variable(box[active],order=1)
            derivatives.append([v[1,0] for v in self.inputs(*args)[:2]])
        h=I(box[axis].hi)-I(box[axis].lo);low=I(box[axis].lo)
        for row in rows:
            for witness in row.get('localized_jumps',[]):
                detail=dict(role=witness['role'],knot=witness['knot'],position=witness['position'],admission='fallback')
                if 'signed_jumps' not in witness:
                    if diagnostic:diagnostic({**detail,'reason':'unsigned_crossing'})
                    continue
                role=witness['role'];inner,outer=(v[role] for v in derivatives)
                if any(v.lo<=0<=v.hi for v in (inner,outer)):
                    if diagnostic:diagnostic({**detail,'reason':'derivative_contains_zero','inner':inner,'outer':outer})
                    continue
                self.checkpoint();roots=[];knot=self.extension.cn[witness['knot']]
                for edge in (box[transverse].lo,box[transverse].hi):
                    edge_box=list(box);edge_box[transverse]=I(edge)
                    root=self.root(tuple(edge_box),axis,role,knot,clipped=True)
                    t=(root-low)/h;roots.append(I(max(I(0).lo,t.lo),min(I(1).hi,t.hi)))
                # ds/dt=-q_transverse/q_inner. Orient the integration range;
                # the spatial right-minus-left jump already owns its sign.
                if (-outer/inner).hi<0:roots.reverse()
                ratio=inner/outer
                if ratio.hi<0:ratio=-ratio
                terms=[weighted_kernel_parts(jump*ratio*h**(j+2),*roots,j+1) for j,jump in enumerate(witness['signed_jumps'])]
                witness['transverse_integral']=sum((part['value'] for part in terms),I(0))
                if diagnostic:diagnostic({**detail,'admission':'integrated','inner':inner,'outer':outer,'roots':roots,'terms':terms,'value':witness['transverse_integral']})
        if self.smooth_partition:
            from tools.opaque_smooth_peano import partitioned_ordinary
            def third(sub,active):
                args=[Jet(sub[j],order=3) for j in (0,1)];args[active]=Jet.variable(sub[active],order=3)
                return self.ordinary(*args)[1][3,0]*6
            partition=partitioned_ordinary(third,box,axis,checkpoint=self.checkpoint)
            rows[1]['ordinary_integral']=partition['effective']
            rows[1]['ordinary_integral_signed']=partition['signed']
        return rows
