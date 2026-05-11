! Hybrid TC potential for Tavis-Cummings-like models.
!
! This backend supports a mixed phonon layout:
!   - shared (global) modes that can couple across the QD electronic space
!   - QD-local modes that act only within one QD state block
!
! Per-mode ownership is provided by mode_owner(i):
!   mode_owner(i)=0   -> shared/mixed mode i # shouldn't have one
!   mode_owner(i)=k>0 -> mode i is local to QD k (1-based QD index)
!
! QD state ordering is assumed blockwise in the electronic basis:
!   [QD1 states][QD2 states]...[QDN states][cavity states]

module tchybrid

   use types

   real(dp), allocatable :: Vconst(:,:), Vlin(:,:,:)
   complex(dpc), allocatable :: Vconst_c(:,:)
   integer, allocatable :: mode_owner(:)
   ! Cached active electronic-state range for each phonon mode.
   ! These bounds are computed once during init and reused in the hot loops.
   integer, allocatable :: mode_s0(:), mode_s1(:)
   integer :: n_qd = 0, nstate_per_qd = 0, n_cavity = 0
   logical :: use_complex_vconst = .false.

contains

   ! Initialize the tchybrid backend and register PES callbacks.
   ! Purpose:
   ! - allocate and copy backend data (omega, Vconst, Vlin, mode_owner)
   ! - validate invariants (cavity slices of Vlin must be zero)
   ! - precompute per-mode state-window cache (mode_s0/mode_s1)
   ! - register procedure pointers used by the `pes` driver (pot, grad, grad_a)
   subroutine init(nf_,ns_,mass_,omega_,Vconst_,Vlin_,mode_owner_,n_qd_,nstate_per_qd_,n_cavity_)
      complex(dpc), allocatable :: Vconst_c_(:,:)
      integer, intent(in) :: nf_, ns_
      integer, intent(in) :: mode_owner_(nf_), n_qd_, nstate_per_qd_, n_cavity_
      real(dp), intent(in) :: mass_(nf_), omega_(nf_)
      real(dp), intent(in) :: Vconst_(ns_,ns_), Vlin_(nf_,ns_,ns_)

      ! n_qd_: number of QDs in the block structure
      ! nstate_per_qd_: number of electronic states per QD block
      ! n_cavity_: number of cavity states appended after QD blocks
      ! mode_owner_: owner map for each phonon mode (0 shared, k>0 local to QD k)

      allocate(Vconst_c_(ns_,ns_))
      Vconst_c_ = cmplx(Vconst_, 0.d0, kind=dpc)
      call init_complex(nf_,ns_,mass_,omega_,Vconst_c_,Vlin_,mode_owner_,n_qd_,nstate_per_qd_,n_cavity_)
      deallocate(Vconst_c_)
   end subroutine

   subroutine init_complex(nf_,ns_,mass_,omega_,Vconst_c_,Vlin_,mode_owner_,n_qd_,nstate_per_qd_,n_cavity_)
      use pes, only : nf, ns, omega, cayley, tc_complex_mode
      integer, intent(in) :: nf_, ns_
      integer, intent(in) :: mode_owner_(nf_), n_qd_, nstate_per_qd_, n_cavity_
      integer :: i
      real(dp) :: imag_max
      real(dp), intent(in) :: mass_(nf_), omega_(nf_)
      complex(dpc), intent(in) :: Vconst_c_(ns_,ns_)
      real(dp), intent(in) :: Vlin_(nf_,ns_,ns_)

      if (size(mass_) /= nf_) then
         error stop 'mass array has inconsistent length in tchybrid init_complex'
      end if
      if (nf /= nf_ .or. ns /= ns_) then
         error stop 'pes module not initialized consistently before tchybrid init_complex'
      end if
      tc_complex_mode = .true.
      use_complex_vconst = .true.

      if (allocated(omega)) deallocate(omega)
      if (allocated(Vconst)) deallocate(Vconst)
      if (allocated(Vlin)) deallocate(Vlin)
      if (allocated(Vconst_c)) deallocate(Vconst_c)
      if (allocated(mode_owner)) deallocate(mode_owner)
      if (allocated(mode_s0)) deallocate(mode_s0)
      if (allocated(mode_s1)) deallocate(mode_s1)

      allocate(omega(nf), Vconst(ns,ns), Vconst_c(ns,ns), Vlin(nf,ns,ns), mode_owner(nf), mode_s0(nf), mode_s1(nf))
      omega = omega_
      Vconst_c = Vconst_c_
      Vconst = real(Vconst_c, dp)
      Vlin = Vlin_
      mode_owner = mode_owner_

      n_qd = n_qd_
      nstate_per_qd = nstate_per_qd_
      n_cavity = n_cavity_

      ! Enforce Hermiticity for complex TC constants.
      if (maxval(abs(Vconst_c - transpose(conjg(Vconst_c)))) > 1.d-10) then
         error stop 'Complex Vconst must be Hermitian in tchybrid.f90'
      end if
      do i = 1, ns
         if (abs(aimag(Vconst_c(i,i))) > 1.d-10) then
            error stop 'Complex Vconst diagonal must be real in tchybrid.f90'
         end if
      end do
      imag_max = maxval(abs(aimag(Vconst_c)))
      if (imag_max <= 1.d-12) then
         write(*,'(A)') 'INFO[tchybrid]: Vconst is numerically real; continuing in complex TC mode.'
      else
         write(*,'(A,1X,ES12.4)') 'INFO[tchybrid]: Vconst has nonzero imaginary couplings; max|Im(Vconst)| =', imag_max
      end if

      if (n_cavity > 0) then
         if (ns - n_cavity < 0) then
            error stop 'Invalid cavity block: n_cavity exceeds ns in tchybrid.f90'
         end if
         if (any(Vlin(:, ns - n_cavity + 1:ns, :) /= 0.d0) .or. &
             any(Vlin(:, :, ns - n_cavity + 1:ns) /= 0.d0)) then
            error stop 'Cavity phonon-coupling block in Vlin must be zero in tchybrid.f90'
         end if
      end if

      call cache_mode_bounds(ns)
      cayley = .true.
   end subroutine

   subroutine cache_mode_bounds(ns)
      integer, intent(in) :: ns
      integer :: i

      ! Precompute the per-mode state window so pot/grad/grad_a do not
      ! recompute block boundaries for every q-point evaluation.
      ! Purpose: fill `mode_s0(i)`/`mode_s1(i)` for each phonon mode i
      ! so hot loops can index contiguous sub-blocks of the electronic
      ! Hamiltonian without recomputing owner->range logic.
      do i = 1, size(mode_owner)
         call mode_state_bounds(mode_owner(i), ns, mode_s0(i), mode_s1(i))
      end do
   end subroutine

   subroutine mode_state_bounds(owner, ns, s0, s1)
      integer, intent(in) :: owner, ns
      integer, intent(out) :: s0, s1
      integer :: ns_ex

      ! Translate owner label into the active electronic state range.
      ! - owner<=0 would indicate a shared/global mode (not supported here)
      ! - owner>0 means this mode acts only on the owner's QD block
      ! The returned s0/s1 define a contiguous index window on the
      ! electronic basis: U(s0:s1, : ) picks the active subspace.

      ! Cavity electronic states are appended after the QD blocks and must
      ! never receive phonon coupling. Therefore the active QD subspace ends
      ! at ns - n_cavity.
      ns_ex = ns - n_cavity
      if (ns_ex < 0) ns_ex = 0

      if (n_qd <= 0 .or. nstate_per_qd <= 0) then
         error stop 'Invalid block-structure (n_qd<0 or nstate_per_qd<0) parameters in tchybrid.f90'
      end if

      if (owner <= 0) then
         ! Shared/global mode: act on the full QD electronic subspace.
         ! raise an error
         error stop 'Invalid block-structure (owner<=0) parameters in tchybrid.f90'

      else
         ! QD-local mode: act on the owner's QD block only.
         s0 = (owner-1) * nstate_per_qd + 1
         s1 = min(owner * nstate_per_qd, ns_ex)
      end if
   end subroutine

   subroutine pot_real(q, V)
      use pes, only : mass, ns, nf, omega
      real(dp), intent(in) :: q(nf)
      real(dp), intent(out) :: V(ns,ns)

      integer :: i, n, s0, s1
      real(dp) :: V0

      ! Build diabatic potential: start from the constant part and add
      ! linear contributions from each phonon mode restricted to the
      ! mode's active electronic block.
      V = Vconst
      V0 = 0.5d0 * sum(mass*(omega*q)**2)

      do i = 1, nf
         ! Keep the explicit loop: each mode can target a different block,
         ! so SUM would not avoid the per-mode slice selection here.
         s0 = mode_s0(i)
         s1 = mode_s1(i)
         if (s1 >= s0) then
            V(s0:s1,s0:s1) = V(s0:s1,s0:s1) + q(i) * Vlin(i,s0:s1,s0:s1)
         end if
      end do

      ! The harmonic nuclear energy is a scalar V0(q) * I. It must be added
      ! to every diabatic state; otherwise photon states are artificially
      ! detuned from QD states by the bath energy.
      do n = 1, ns
         V(n,n) = V(n,n) + V0
      end do
   end subroutine

   subroutine pot_complex(q, V)
      use pes, only : mass, ns, nf, omega
      real(dp), intent(in) :: q(nf)
      complex(dpc), intent(out) :: V(ns,ns)
      integer :: i, n, s0, s1
      real(dp) :: V0

      if (.not. use_complex_vconst) then
         V = cmplx(Vconst, 0.d0, kind=dpc)
      else
         V = Vconst_c
      end if
      V0 = 0.5d0 * sum(mass*(omega*q)**2)

      do i = 1, nf
         s0 = mode_s0(i)
         s1 = mode_s1(i)
         if (s1 >= s0) then
            V(s0:s1,s0:s1) = V(s0:s1,s0:s1) + q(i) * cmplx(Vlin(i,s0:s1,s0:s1), 0.d0, kind=dpc)
         end if
      end do

      do n = 1, ns
         V(n,n) = V(n,n) + cmplx(V0, 0.d0, kind=dpc)
      end do
   end subroutine

   subroutine grad_real(q, G)
      use pes, only : mass, nf, ns, omega, cayley
      real(dp), intent(in) :: q(:)
      real(dp), intent(out) :: G(nf,ns,ns)

      integer :: n
      real(dp), allocatable :: G0(:)

      ! Build gradient tensor G(i,:,:) = dV/dq_i.
      ! For efficiency we copy the precomputed linear tensors `Vlin` into
      ! `G` and then add the harmonic diagonal term if required.
      G = Vlin

      if (cayley) return

      allocate(G0(nf))
      G0 = mass*omega**2 * q
      do n = 1, ns
         G(:,n,n) = G(:,n,n) + G0
      end do
      deallocate(G0)
   end subroutine

   ! Compute adiabatic force derivatives for adiabatic state `a`.
   !
   ! Inputs:
   ! - `q` : phonon coordinates (size `nf`)
   ! - `U` : matrix of adiabatic eigenvectors (size ns x ns); column `a`
   !         is the eigenvector for which we compute derivatives.
   ! - `a` : adiabatic-state index (integer). The eigenvector may contain
   !         both QD and cavity components, so forces are evaluated from
   !         its QD components through the owner-restricted Vlin blocks.
   !
   ! Outputs:
   ! - `dvdq(i)` = <U_a | dV/dq_i | U_a> for each phonon mode i.
   !
   ! Local variables of note:
   ! - `s0,s1`  : active electronic-state window for mode i (from cache)
   ! - `Ublk`   : the subvector U(s0:s1,a) restricted to the mode's active electronic subspace.
   !             It is used to compute the quadratic form
   !             Ublk^T * Vlin(i,block) * Ublk.
   ! - `tmp`    : temporary vector = Vlin(i,block) * Ublk.
   !
   ! Implementation details:
   ! For each mode i we restrict the operation to the precomputed
   ! window s0:s1, compute tmp = Vlin(i,block)*Ublk and then
   ! dvdq(i) = dot_product(Ublk,tmp) which equals <U_a|dV/dq_i|U_a>.
   subroutine grad_a_tchybrid(q, U, a, dvdq)
      use pes, only : nf, cayley, mass, omega
      real(dp), intent(in) :: q(:), U(:,:)
      integer :: a
      real(dp), intent(out) :: dvdq(nf)

      integer :: i, s0, s1
      real(dp), allocatable :: Ublk(:), tmp(:)

      dvdq = 0.d0
      do i = 1, nf
         ! Use the same owner-restricted subspace in adiabatic force evaluation.
         s0 = mode_s0(i)
         s1 = mode_s1(i)
         if (s1 < s0) cycle
         ! Ublk is the adiabatic eigenvector restricted to the active block
         ! for this phonon mode; we form the quadratic form
         ! Ublk^T * Vlin(i,block) * Ublk via a matmul + dot_product.
         Ublk = U(s0:s1, a)
         tmp = matmul(Vlin(i, s0:s1, s0:s1), Ublk)
         dvdq(i) = dot_product(Ublk, tmp)
      end do

      if (.not. cayley) then
         dvdq = dvdq + mass * omega**2 * q
      end if
   end subroutine

   subroutine grad_a_tchybrid_complex(q, U, a, dvdq)
      use pes, only : nf, cayley, mass, omega
      real(dp), intent(in) :: q(:)
      complex(dpc), intent(in) :: U(:,:)
      integer :: a
      real(dp), intent(out) :: dvdq(nf)
      integer :: i, s0, s1
      complex(dpc), allocatable :: Ublk(:), tmp(:)
      complex(dpc) :: z

      dvdq = 0.d0

      do i = 1, nf
         s0 = mode_s0(i)
         s1 = mode_s1(i)
         if (s1 < s0) cycle
         Ublk = U(s0:s1, a)
         tmp = matmul(Vlin(i, s0:s1, s0:s1), Ublk)
         z = dot_product(Ublk, tmp)
         dvdq(i) = real(z, dp)
      end do

      if (.not. cayley) then
         dvdq = dvdq + mass * omega**2 * q
      end if
   end subroutine

   ! Return a copy of the constant (q-independent) diabatic matrix.
   subroutine get_vconst(Vconst_)
      use pes, only : ns
      real(dp), intent(out) :: Vconst_(ns,ns)
      if (use_complex_vconst) then
         ! Keep TC propagation fully complex; this getter returns only real
         ! part for compatibility with legacy real interfaces.
         Vconst_ = real(Vconst_c, dp)
      else
         Vconst_ = Vconst
      end if
   end subroutine

   ! Return a copy of the per-mode linear coupling tensors Vlin(i,:,:)
   ! such that V = Vconst + sum_i q(i) * Vlin(i,:,:).
   subroutine get_vlin(vlin_)
      use pes, only : ns, nf
      real(dp), intent(out) :: vlin_(nf,ns,ns)
      vlin_ = vlin
   end subroutine

end module
