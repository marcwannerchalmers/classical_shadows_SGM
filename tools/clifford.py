from __future__ import annotations

from flax import struct
import jax
from jax import Array, vmap
from jax import numpy as jnp
from jax.lax import cond, fori_loop

from tools.binary_solver import solve_binary, stabilizer_phase


# DISCLAIMER: Can be tuned by using bool arrays
# NOTE: one always needs to call tableau = tableau.OP(...)
class Tableau(struct.PyTreeNode):
    """Integer symplectic tableau for Clifford-state simulation.

    Tableau operations are immutable: every gate returns a replacement object
    and callers must retain that return value.

    Attributes:
        tableau: Binary symplectic rows stored as an `int32` JAX array.
        r: Binary row phases.
    """

    tableau: Array
    r: Array

    # Initialize as empty array for lazy init
    @classmethod
    def create(cls, n: int = 0, N: int = 0, N_rho: int = 0) -> "Tableau":
        """Create identity tableaus with optional sample and state batches.

        Args:
            n: Number of qubits.
            N: Optional number of sample tableaus.
            N_rho: Optional number of state-repetition batches.

        Returns:
            Identity tableau data with zero phases.
        """

        tableau = jnp.eye(2 * n, dtype=jnp.int32)
        r = jnp.zeros(2 * n, dtype=jnp.int32)
        if N > 0:
            tableau = jnp.stack([tableau for _ in range(N)])
            r = jnp.stack([r for _ in range(N)])
        if N_rho > 0:
            tableau = jnp.stack([tableau for _ in range(N_rho)])
            r = jnp.stack([r for _ in range(N_rho)])

        return cls(tableau=tableau, r=r)

    @property
    def n(self) -> int:
        """Number of qubits represented by the tableau."""

        return self.r.shape[-1] // 2

    def H(self, i, condition: Array = jnp.array(True)):
        return cond(
            condition,
            lambda s: s._Hadamard(i),
            lambda s: s,
            self,
        )

    def S(self, i, condition: Array = jnp.array(True)):
        return cond(
            condition,
            lambda s: s._S(i),
            lambda s: s,
            self,
        )

    def CNOT(
        self,
        c: int | Array,
        t: int | Array,
        condition: Array = jnp.array(True),
    ):
        return cond(
            condition,
            lambda s: s._CNOT(c, t),
            lambda s: s,
            self,
        )

    def CZ(
        self,
        c: int | Array,
        t: int | Array,
        condition: Array = jnp.array(True),
    ):
        return cond(
            condition,
            lambda s: s._CZ(c, t),
            lambda s: s,
            self,
        )

    def MultiS(self, bit_encoding: Array):
        x, z, r_vec = vmap(self._cond_S_chunk, 
                           in_axes=(1,1,0), 
                           out_axes=(1,1,1))(self.tableau[:,:self.n],
                                             self.tableau[:,self.n:],
                                             bit_encoding)

        tableau = jnp.concatenate([x,z], axis=1)
        r = (self.r + jnp.sum(r_vec, axis=1, dtype=jnp.int32)) % 2
        return self.replace(tableau=tableau, r=r)

    def MultiSdag(self, bit_encoding: Array):
        x, z, r_vec = vmap(self._cond_Sdag_chunk, 
                            in_axes=(1,1,0), 
                            out_axes=(1,1,1))(self.tableau[:,:self.n],
                                                self.tableau[:,self.n:],
                                                bit_encoding)

        tableau = jnp.concatenate([x,z], axis=1)
        r = (self.r + jnp.sum(r_vec, axis=1, dtype=jnp.int32)) % 2
        return self.replace(tableau=tableau, r=r)

    def MultiHadamard(self, bit_encoding: Array):
        x, z, r_vec = vmap(self._cond_H_chunk, 
                            in_axes=(1,1,0), 
                            out_axes=(1,1,1))(self.tableau[:,:self.n],
                                                self.tableau[:,self.n:],
                                                bit_encoding)
        
        tableau = jnp.concatenate([x,z], axis=1)
        r = (self.r + jnp.sum(r_vec, axis=1, dtype=jnp.int32)) % 2
        return self.replace(tableau=tableau, r=r)

    def Permute(self, permutation: Array):
        tableau = jnp.concatenate([self.tableau[:,permutation],
                                   self.tableau[:,permutation + self.n]],
                                   axis=1)
        
        return self.replace(tableau=tableau)

    def PauliRot(self, i: Array, theta: Array, pauli: Array):
        x, z, r = self._single_PauliRot(theta,
                                        pauli,
                                        self.tableau[:,i],
                                        self.tableau[:,i+self.n])

        tableau = self.tableau.at[:,i].set(x)
        tableau = tableau.at[:,i+self.n].set(z)
        r = (self.r + r) % 2
        return self.replace(tableau=tableau, r=r)

    def MultiPauli(self, paulivector: Array):
        _, _, r_vec = vmap(self._single_Pauli, 
                     in_axes=(0,1,1),
                     out_axes=1)(paulivector, 
                                self.tableau[:,:self.n],
                                self.tableau[:,self.n:])

        r = (self.r + jnp.sum(r_vec, axis=1, dtype=jnp.int32)) % 2
        return self.replace(r=r)

    def _Hadamard(self, i: int):
        xi = self.tableau[:,i]
        zi = self.tableau[:,i+self.n]
        r = (self.r + xi*zi) % 2
        tmp = xi
        tableau = self.tableau.at[:,i].set(zi)
        tableau = tableau.at[:,i+self.n].set(tmp)

        return self.replace(tableau=tableau, r=r)

    def _CNOT(self, control: int | Array, target: int | Array):
        xi = self.tableau[:,control]
        zi = self.tableau[:,control+self.n]
        xj = self.tableau[:,target]
        zj = self.tableau[:,target+self.n]

        r = (self.r + xi*zj*((xj+zi+1)%2)) % 2
        xj = (xj + xi) % 2
        zi = (zi + zj) % 2

        tableau = self.tableau.at[:,target].set(xj)
        tableau = tableau.at[:,control+self.n].set(zi)
        return cond(control != target, 
                    lambda: self.replace(tableau=tableau, r=r),
                    lambda: self)
        
    def _S(self, i: int):
        xi = self.tableau[:,i]
        zi = self.tableau[:,i+self.n]

        r = (self.r + xi*zi) % 2
        tableau = self.tableau.at[:,i+self.n].set((zi+xi)%2)

        return self.replace(tableau=tableau, r=r)

    @staticmethod
    def _S_chunk(xi, zi):
        return xi, (zi+xi)%2, (xi*zi) % 2

    @staticmethod
    def _cond_S_chunk(xi, zi, condition):
        return cond(condition, 
                    Tableau._S_chunk,
                    lambda xi, zi: (xi, zi, jnp.zeros_like(xi)),
                    xi, zi)

    @staticmethod
    def _Sdag_chunk(xi, zi):
        return xi, (zi+xi)%2, xi * ((zi + 1) % 2)

    @staticmethod
    def _cond_Sdag_chunk(xi, zi, condition):
        return cond(condition, 
                    Tableau._Sdag_chunk,
                    lambda xi, zi: (xi, zi, jnp.zeros_like(xi)),
                    xi, zi)

    @staticmethod
    def _H_chunk(xi, zi):
        return zi, xi, (xi*zi)

    @staticmethod
    def _cond_H_chunk(xi, zi, condition):
        return cond(condition, 
                    Tableau._H_chunk,
                    lambda xi, zi: (xi, zi, jnp.zeros_like(xi)),
                    xi, zi)

    @staticmethod
    def _single_PauliRot(theta: Array, pauli: Array, xi: Array, zi: Array):
        r90 = lambda: Tableau._single_Pauli90(pauli, xi, zi)
        rm90 = lambda: Tableau._single_Paulim90(pauli, xi, zi)
        rpi = lambda: Tableau._single_Pauli(pauli, xi, zi)
        rpihalf = lambda: cond(theta == jnp.pi/2, 
                               r90, 
                               rm90)
        admissible = lambda: cond(jnp.abs(theta) == jnp.pi,
                                  rpi,
                                  rpihalf)
        
        # TODO: Add 0 to the admissible ones
        return admissible()

    @staticmethod
    def _single_Pauli(pauli: Array, xi: Array, zi: Array):
        # add x-phase for Y, Z
        x_phase = cond(pauli >= 2, 
                       lambda x: x,
                       lambda x: jnp.zeros_like(x),
                       xi)

        # add z-phase for X,Y
        z_phase = cond((pauli % 3) >= 1, 
                       lambda x: x,
                       lambda x: jnp.zeros_like(x),
                       zi)

        return xi, zi, x_phase + z_phase


    @staticmethod
    def _single_Pauli90(pauli: Array, xi: Array, zi: Array):
        # The symplectic part is the same for +/- pi/2; only the sign differs.
        xi_res = cond(pauli % 3 == 0,
                      lambda: xi, # Id, RZ
                      lambda: cond(pauli == 2,
                                   lambda: zi, # RY
                                   lambda: (xi + zi) % 2) # RX
                      )

        zi_res = cond(pauli <= 1,
                      lambda: zi, # Id, RX
                      lambda: cond(pauli == 2,
                                   lambda: xi, # RY
                                   lambda: (xi + zi) % 2) # RZ
                      )
        
        r_res = cond(pauli < 2,
                     lambda: cond(pauli == 0,
                                  lambda: jnp.zeros_like(xi), # Id
                                  lambda: zi*(1-xi) # RX
                                  ),
                     lambda: cond(pauli == 2,
                                  lambda: xi*(1-zi), # RY
                                  lambda: xi*zi # RZ
                                  )
                                )
        
        return xi_res, zi_res, r_res

    @staticmethod
    def _single_Paulim90(pauli: Array, xi: Array, zi: Array):
        xi_res = cond(pauli % 3 == 0,
                        lambda: xi, # Id,RY,RZ
                        lambda: cond(
                            pauli == 2,
                            lambda: zi,
                            lambda: (xi + zi) % 2,
                        ),
                        )
 
        zi_res = cond(pauli <= 1, 
                        lambda: zi, # Id, RX
                        lambda: cond(
                            pauli == 2,
                            lambda: xi,
                            lambda: (xi + zi) % 2,
                        ),
                        )
        
        r_res = cond(pauli < 2,
                        lambda: cond(pauli == 0,
                                    lambda: jnp.zeros_like(xi), # Id
                                    lambda: xi*zi # RX
                                    ),
                        lambda: cond(pauli == 2,
                                    lambda: zi*(1-xi), # RY
                                    lambda: xi*(1-zi) # RZ
                                    )
                                )
        
        return xi_res, zi_res, r_res


    def _CZ(self, control: int | Array, target: int | Array):
        xi = self.tableau[:,control]
        zi = self.tableau[:,control+self.n]
        xj = self.tableau[:,target]
        zj = self.tableau[:,target+self.n]

        r = (self.r + xi*xj*((zi + zj)%2)) % 2
        tableau = self.tableau.at[:,control+self.n].set((zi+xj)%2)
        tableau = tableau.at[:,target+self.n].set((zj+xi)%2)

        return cond(control != target, 
                    lambda: self.replace(tableau=tableau, r=r),
                    lambda: self)

    # Takes int array
    def expval(self, paulivector: Array):
        p = pauli_to_symp(paulivector)
        M = self.tableau[self.n:].T
        a, contained = solve_binary(M, p)
        return cond(contained,
                    lambda: (-1)**stabilizer_phase(M,self.r[self.n:],a, p),
                    lambda: jnp.array(0, dtype=self.r.dtype))

    def sample(self, key: Array):
        n = self.n
        keys = jax.random.split(key, n)
        x = self.tableau[:, :n]
        z = self.tableau[:, n:]
        outcome = jnp.zeros((n,), dtype=jnp.int32)

        def body_fun(a, val):
            x, z, r, outcome = val
            oc, x, z, r = Tableau._single_post_measurement_state(
                keys[a], x, z, r, a, n
            )
            return x, z, r, outcome.at[a].set(oc)

        _, _, _, outcome = fori_loop(0, n, body_fun, (x, z, self.r, outcome))
        return outcome

    @staticmethod
    def _single_post_measurement_state(
        key: Array,
        x: Array,
        z: Array,
        r: Array,
        a: Array,
        n: int,
    ):
        condition = (x[n:,a] == 1).any()
        def case1(x, z, r):
            # `argmax` returns the first stabilizer row with x[p, a] == 1.
            p = jnp.argmax(x[n:,a]).astype(jnp.int32) + n
            xp, zp, rp = x[p], z[p], r[p]

            def cond_rowsum(i, xh, zh, rh):
                return cond(
                    (i != p) & (xh[a] == 1),
                    lambda: Tableau._rowsum(xh, zh, rh, xp, zp, rp),
                    lambda: (xh, zh, rh),
                )

            x, z, r = vmap(cond_rowsum)(jnp.arange(2*n, dtype=jnp.int32), x, z, r)

            x, z, r = x.at[p-n].set(x[p]), z.at[p-n].set(z[p]), r.at[p-n].set(r[p])
            x = x.at[p].set(jnp.array(0, dtype=x.dtype))
            z = z.at[p].set(jnp.array(0, dtype=z.dtype))
            z = z.at[p,a].set(jnp.array(1, dtype=z.dtype))
            r = r.at[p].set(jax.random.randint(key, (), 0, 2, dtype=jnp.int32))
            return r[p], x, z, r

        def case2(x, z, r):
            init_val = (
                jnp.zeros((n,), dtype=x.dtype),
                jnp.zeros((n,), dtype=z.dtype),
                jnp.array(0, dtype=r.dtype),
            )

            def body_fun(i, scratch):
                return cond(
                    x[i,a] == 1,
                    lambda: Tableau._rowsum(
                        *scratch, x[i+n], z[i+n], r[i+n]
                    ),
                    lambda: scratch,
                )

            _, _, rh = fori_loop(0, n, body_fun, init_val)

            return rh, x,z,r

        oc, x,z,r = cond(condition,
                     case1,
                     case2,
                     x,z,r)

        return oc, x,z,r



    @staticmethod
    def _rowsum(xh: Array, zh: Array, rh: Array,
                xi: Array, zi: Array, ri: Array):

        def g(x1,z1,x2,z2):
            return cond((x1 == 0),
                 lambda: cond(z1==0,
                              lambda: jnp.array(0, dtype=x1.dtype),
                              lambda: x2*(1-2*z2)), # x1 == 0 and z1 == 1
                 lambda: cond(z1==0,
                              lambda: z2*(2*x2-1), # x1 == 1 and z1 == 0
                              lambda: z2-x2) # x1 == 1 and z1 == 1
                 )

        g_vals = vmap(g)(xi,zi,xh,zh)
        phase = (
            2*rh + 2*ri
            + jnp.sum(g_vals, axis=-1, dtype=jnp.int32)
        ) % 4
        rh = phase // 2
        return (xh + xi) % 2, (zh + zi) % 2, rh


def pauli_to_symp(pauli_vector):
    def _single_pauli_to_symp(p: Array):
        return (((p+3)%4)//2+1)%2, p//2

    return jnp.concat(vmap(_single_pauli_to_symp)(pauli_vector))

def F(tableau: Tableau, pauli_indices: Array, Gamma: Array, Delta: Array)->Tableau:
    n = Gamma.shape[0]

    # Match utils.F: traverse the lower triangles in descending order.
    def body_iCX(k: int, tableau: Tableau):
        i = n - 1 - k

        def body_j(l: int, tableau: Tableau):
            j = i - 1 - l
            return tableau.CNOT(i, j, Delta[i,j] == 1)

        return fori_loop(0, i, body_j, tableau)

    tableau = fori_loop(0, n, body_iCX, tableau)

    def body_iCZ(k: int, tableau: Tableau):
        i = n - 1 - k

        def body_j(l: int, tableau: Tableau):
            j = i - 1 - l
            return tableau.CZ(i, j, Gamma[i,j] == 1)

        return fori_loop(0, i, body_j, tableau)

    tableau = fori_loop(0, n, body_iCZ, tableau)
    tableau = tableau.MultiPauli(pauli_indices)
    tableau = tableau.MultiS(Gamma.diagonal())
    return tableau

def canonical_form(tableau: Tableau, Gamma: Array, Delta: Array, 
                   Gammad: Array, Deltad: Array, 
                   h: Array, pauli_indices: Array,
                   S: Array) -> Tableau:
    n = Gamma.shape[0]
    tableau = F(tableau, pauli_indices, Gammad, Deltad)
    tableau = tableau.Permute(S)
    tableau = tableau.MultiHadamard(h)
    tableau = F(tableau, jnp.zeros((n,), dtype=jnp.int32), Gamma, Delta)

    return tableau

def F_rev(tableau: Tableau, pauli_indices: Array, Gamma: Array, Delta: Array)->Tableau:
    n = Gamma.shape[0]
    tableau = tableau.MultiSdag(Gamma.diagonal())
    tableau = tableau.MultiPauli(pauli_indices)
    # Delta is lower triangular
    def body_iCZ(i: int, tableau: Tableau):
        def body_j(j: int, tableau: Tableau):
            return tableau.CZ(i,j, Gamma[i,j] == 1) # reversed loop

        return fori_loop(0, i, body_j, tableau)

    tableau = fori_loop(0, n, body_iCZ, tableau)

    # Delta is lower triangular
    def body_iCX(i: int, tableau: Tableau):
        def body_j(j: int, tableau: Tableau):
            return tableau.CNOT(i,j, Delta[i,j] == 1) # reversed loop

        return fori_loop(0, i, body_j, tableau)

    tableau = fori_loop(0, n, body_iCX, tableau)
    return tableau

def canonical_form_rev(tableau: Tableau, Gamma: Array, Delta: Array, 
                   Gammad: Array, Deltad: Array, 
                   h: Array, pauli_indices: Array,
                   S: Array) -> Tableau:
    n = Gamma.shape[0]
    tableau = F_rev(tableau, jnp.zeros((n,), dtype=jnp.int32), Gamma, Delta)
    tableau = tableau.MultiHadamard(h)
    tableau = tableau.Permute(jnp.argsort(S).astype(jnp.int32))

    tableau = F_rev(tableau, pauli_indices, Gammad, Deltad)

    return tableau

def computational_basis_measurement_paulis(ind: Array):
    n = ind.shape[0]
    gammadelta = [ind[:,i*n:(i+1)*n] for i in range(4)]
    h = ind[:,4*n]
    pauli_indices = ind[:,4*n+1]
    S = ind[:,4*n+4]

    tableau = Tableau.create(n)
    tableau = canonical_form_rev(tableau, *gammadelta, h, pauli_indices, S)

    return (tableau.tableau[n:,:n],
            tableau.tableau[n:,n:],
            tableau.r[n:])

def GHZ_type_state_clifford_rev(selective_block: Array,
                            xy: Array,
                            tableau: Tableau):
    
    n = tableau.n
    # Read utils.parallel_entangler_blocks for more explanation
    sorted_indices = jnp.argsort(
        selective_block, descending=True
    ).astype(jnp.int32)
    sorted_vals = selective_block[sorted_indices] 

    # theta = cond(reversed, lambda: -jnp.pi/2, lambda: jnp.pi/2)
    theta = jnp.asarray(-jnp.pi / 2, dtype=jnp.float32)
    # applies nothing if indices are all 0
    tableau = tableau.PauliRot(jnp.array(sorted_indices[0], dtype=jnp.int32),
                                theta, 
                                sorted_vals[0]*(xy+1)) 

    def body_fun(i, tableau: Tableau):
        return tableau.CNOT(sorted_indices[i],
                            sorted_indices[i+1],
                            (sorted_vals[i] == 1) & (sorted_vals[i+1] == 1))

    tableau = fori_loop(0, n-1, body_fun, tableau)

    return tableau

def GHZ_type_state_clifford(selective_block: Array,
                            xy: Array,
                            tableau: Tableau):
    
    n = tableau.n
    # Read utils.parallel_entangler_blocks for more explanation
    sorted_indices = jnp.argsort(
        selective_block, descending=True
    ).astype(jnp.int32)
    sorted_vals = selective_block[sorted_indices] 
    
    def body_fun(i, tableau: Tableau):
        ind = n-2-i # reversed order 
        return tableau.CNOT(sorted_indices[ind],
                            sorted_indices[ind+1],
                            (sorted_vals[ind] == 1) & (sorted_vals[ind+1] == 1))
    
    tableau = fori_loop(0, n-1, body_fun, tableau)
    
    theta = jnp.asarray(jnp.pi / 2, dtype=jnp.float32)
    tableau = tableau.PauliRot(
        jnp.array(sorted_indices[0], dtype=jnp.int32),
        theta,
        sorted_vals[0] * (xy + 1),
    )
    
    return tableau
