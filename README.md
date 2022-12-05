# Dissipative Symplectic Optimization

We propose dissipative extensions of symplectic integrators, with applications to optimization.
The most useful method is a dissipative extension of leapfrog which is a second order integrator but
has the lowest computational load, i.e., only one gradient per iteration.

As an example, we apply such methods to learn the Ising model with a Restricted Boltzman machine.
We reproduce the heat capacity as illustrated in this figure:

![](https://github.com/guisf/dissipative_symplectic/blob/main/ising_heat.png)

The convergence rates of our method is also faster than standard ones used in machine learning:

![](https://github.com/guisf/dissipative_symplectic/blob/main/ising_convergence.png)

* For more details, see [G. G. França, M. I. Jordan, R. Vidal, "On dissipative symplectic integration with applications to gradient-based optimization," J. Stat. Mech. (2021) 043402](https://iopscience.iop.org/article/10.1088/1742-5468/abf5d4).
* See also this [Presentation](https://github.com/guisf/dissipative_symplectic/blob/main/presymp_talk.pdf) for a quick introduction.

