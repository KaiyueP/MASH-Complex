!f2py

! =========== Potential-specific initializations =============
subroutine init_linvib(mass,omega,Vconst,Vlin,nf,ns)
   use types
   use linvib, only : init
   integer  :: nf,ns
   real(dp) :: mass(nf), omega(nf)
   real(dp) :: Vconst(ns,ns), Vlin(nf,ns,ns)
!
!  Initialize linear vibronic potential
!
   call init(nf,ns,mass,omega,Vconst,Vlin)
end subroutine

subroutine init_qudvib(mass,omega,Vconst,Vlin,Wqud,nf,ns)
   use types
   use qudvib, only : init
   integer  :: nf,ns
   real(dp) :: mass(nf), omega(nf)
   real(dp) :: Vconst(ns,ns), Vlin(nf,ns,ns), Wqud(nf,ns,ns)
!
!  Initialize quadratic vibronic potential
!
   call init(nf,ns,mass,omega,Vconst,Vlin,Wqud)
end subroutine

subroutine init_frexc(mass,omega,Vconst,kappa,nf,nf_bath,ns)
   use types
   use frexc, only : init
   integer  :: nf,nf_bath,ns
   real(dp) :: mass(nf), omega(nf)
   real(dp) :: Vconst(ns,ns), kappa(nf_bath,ns)
!
!  Initialize Frenkel-exciton potential (faster than linvib, used where all sites are coupled to identical baths)
!
   call init(nf,ns,mass,omega,Vconst,kappa)
end subroutine

subroutine init_tchybrid(mass,omega,Vconst,Vlin,mode_owner,nf,ns,n_qd,nstate_per_qd,n_cavity)
   use types
   use tchybrid, only : init
   integer, intent(in) :: nf, ns, n_qd, nstate_per_qd, n_cavity
   integer, intent(in) :: mode_owner(nf)
   real(dp), intent(in) :: mass(nf), omega(nf)
   real(dp), intent(in) :: Vconst(ns,ns), Vlin(nf,ns,ns)
!
!  Initialize TC hybrid potential (shared + QD-local modes)
!
   call init(nf,ns,mass,omega,Vconst,Vlin,mode_owner,n_qd,nstate_per_qd,n_cavity)
end subroutine

subroutine init_tully(model,mass)
   use types
   use tully, only : init
   integer :: model
   real(dp) :: mass(1)
!
!  Initialize Tully model
!
   call init(model,mass)
end subroutine

! ================ General initializations =====================

subroutine init_mash(beta)
   use types
   use mash, only : init
   real(dp) :: beta
!
!  Initialize MASH module.
!
   call init(beta)
end subroutine

! ================ Useful wrappers for potentials etc. =================

subroutine get_vad(q,nf,ns,Vad,U)
   use types
   use pes, only : potad
   integer, intent(in) :: nf,ns
   real(dp), intent(in) :: q(nf)
   real(dp), intent(out) :: Vad(ns), U(ns,ns)
!
!  Wrapper for adiabatic potential
!
   call potad(q,Vad,U)
end subroutine

subroutine pot_mat(q,nf_,ns_,V,dVdq)
   use types
   use pes, only : pot, grad
   integer, intent(in)  :: nf_,ns_
   real(dp), intent(in) :: q(nf_)
   real(dp), intent(out):: v(ns_,ns_), dVdq(nf_,ns_,ns_)
!
!  Wrapper for diabatic potential matrix
!
   call pot(q,v)
   call grad(q,dVdq)
end subroutine

subroutine mashpot(q,qe,pe,nf,ns,v)
   use types
   use mash, only : mash_pot
   integer, intent(in)  :: nf
   real(dp), intent(in) :: q(nf),qe(ns),pe(ns)
   real(dp), intent(out):: v
!
!  Wrapper for mash potential
!
   call mash_pot(q,qe,pe,v)
end subroutine

subroutine mashgrad(q,qe,pe,nf_,ns,dvdq)
   use types
   use pes, only : potad, grad_a
   use mash, only : cstate
   integer, intent(in) :: nf_, ns
   real(dp), intent(in) :: q(nf_), qe(ns), pe(ns)
   real(dp), intent(out) :: dvdq(nf_)
!
!  Wrapper for gradient
!
   real(dp), allocatable :: Vad(:), U(:,:)
   integer :: a
   allocate(Vad(ns),U(ns,ns))
   call potad(q, Vad, U)
   call cstate(qe,pe,U,a)
   call grad_a(q, U, a, dvdq)
   deallocate(Vad, U)
end subroutine

subroutine nac(q,d,nf,ns)
   use types
   use pes, only : potad, mynac => nac
   integer, intent(in) :: nf, ns
   real(dp), intent(in) :: q(nf)
   real(dp), intent(out) :: d(nf,ns,ns)
!
!   Wrapper for nonadiabatic coupling vector
!
   real(dp), allocatable :: Vad(:), U(:,:)
   allocate(Vad(ns),U(ns,ns))
   call potad(q,Vad,U)
   call mynac(q,Vad,U,d)
   deallocate(Vad,U)
end subroutine

subroutine mashnacdir(q,cad,Vad,U,n,m,d,nf,ns)
   use types
   use pes, only : nacdir
   integer, intent(in) :: nf, ns
   real(dp), intent(in) :: q(nf), Vad(ns), U(ns,ns)
   complex(dpc), intent(in) :: cad(ns)
   integer, intent(in) :: n,m
   real(dp), intent(out) :: d(nf)
!
!   Wrapper for direction of momentum rescaling/reversal
!
   call nacdir(q,cad,Vad,U,n,m,d)
end subroutine

subroutine dia2ad(q,qe,pe,qa,pa,nf,ns)
   use types
   use pes, only : potad
   real(dp), intent(in) :: q(nf), qe(ns), pe(ns)
   real(dp), intent(out) :: qa(ns), pa(ns)
   integer, intent(in) :: nf, ns
!
!  Convert diabatic amplitudes to adiabatic
!
   real(dp), allocatable :: Vad(:), U(:,:), eye(:,:)
   allocate(Vad(ns),U(ns,ns),eye(ns,ns))
   call potad(q,Vad,U)
   eye = 0.d0
   do n=1,ns
      eye(n,n) = 1.d0
   end do
   do n=1,ns
      if (dot_product(U(:,n),eye(:,n)).lt.0.d0) then
         U(:,n) = -U(:,n)
      end if
   end do
   qa(:) = matmul(qe, U)
   pa(:) = matmul(pe, U)
   deallocate(Vad,U,eye)
end subroutine

subroutine mash_pops(q, qe, pe, pop, rep, nf, ns)
   use types
   use mash, only : pops
   real(dp), intent(in) :: q(nf), qe(ns), pe(ns)
   integer, intent(in) :: nf, ns
   character, intent(in) :: rep
   real(dp), intent(out) :: pop(ns)
!
!  Wrapper for mash.pops
!
   call pops(q, qe, pe, pop, rep)
end subroutine


! =============== Main functions for running a trajectory ===============
subroutine runtrj(q, p, qe, pe, qt, pt, qet, pet, at, Et, &
   dt, ierr, nt, nf_, ns)
   use types
   use pes, only : potad, grad_a, mass
   use mash, only : store, evolve, cstate
   integer :: nt, nf_, ns
   real(dp), intent(in) :: dt
   real(dp), intent(inout) :: q(nf_), p(nf_), qe(ns), pe(ns)
   real(dp), intent(out) :: qt(nt+1,nf_), pt(nt+1,nf_), &
      qet(nt+1,ns), pet(nt+1,ns), Et(nt+1)
   integer, intent(out) :: ierr, at(nt+1)
!
!  Run a trajectory
!
   real(dp), allocatable :: Vad(:), U(:,:), dvdq(:)
   integer :: a , atmp

   allocate(Vad(ns),U(ns,ns),dvdq(nf_))

   call potad(q,Vad,U)
   call cstate(qe,pe,U,a)

   call grad_a(q,U,a,dvdq)
   ierr = 0
   do it = 1, nt
      call store(q, p, qe, pe, a, it, qt, pt, qet, pet, at)
      Et(it) = Vad(a) + 0.5d0*sum(p**2/mass) ! ham_a(q,p,a)
      
      ! Debug print
      ! write(*,'(I5,2ES23.15)') it, q(1), Et(it)

      atmp = a

      call evolve(q, p, qe, pe, Vad, U, dvdq, a, dt)

      if (ierr.gt.0) exit ! Don't waste time if trajectory will be discarded

      if (q(1).ne.q(1)) then
         ! NaN: mark trajectory for discard
         ierr = 1
         exit
      end if
   end do

   if (ierr.eq.0) then
      call store(q, p, qe, pe, a, nt+1, qt, pt, qet, pet, at)
      Et(nt+1) = Vad(a) + 0.5d0*sum(p**2/mass) ! ham_a(q,p,a)
   end if

   deallocate(Vad,U,dvdq)
end subroutine


subroutine runtrj_obs(q, p, qe, pe, bt, Et, rep, dt, ierr, nt, nf_, ns)
   use types
   use pes,  only : potad, grad_a, mass
   use mash, only : evolve, cstate, pops_phi
   integer, intent(in) :: nt, nf_, ns
   real(dp), intent(in) :: dt
   character, intent(in) :: rep
   real(dp), intent(inout) :: q(nf_), p(nf_), qe(ns), pe(ns)
   real(dp), intent(out) :: bt(nt+1,ns), Et(nt+1)
   integer, intent(out) :: ierr

   real(dp), allocatable :: Vad(:), U(:,:), dvdq(:)
   real(dp), allocatable :: U0(:,:), Vad0(:), q0(:)
   complex(dpc), allocatable :: c(:)
   integer :: it, a

   ierr = 0

   allocate(Vad(ns), U(ns,ns), dvdq(nf_), c(ns))

   ! If "exciton basis" output is requested (rep='e'),
   ! original code uses potad(q*0) every time; we cache U0 once.
   if (rep.eq.'e') then
      allocate(U0(ns,ns), Vad0(ns), q0(nf_))
      q0 = 0.d0
      call potad(q0, Vad0, U0)
      deallocate(Vad0, q0)
   end if

   call potad(q, Vad, U)
   call cstate(qe, pe, U, a)
   call grad_a(q, U, a, dvdq)

   do it = 1, nt
      ! ---- populations at current time (matches original timing: before evolve) ----
      if (rep.eq.'d') then
         c = dcmplx(qe, pe)
         call pops_phi(c, bt(it,:))
      else if (rep.eq.'a') then
         c = dcmplx(matmul(qe, U), matmul(pe, U))
         call pops_phi(c, bt(it,:))
      else if (rep.eq.'e') then
         c = dcmplx(matmul(qe, U0), matmul(pe, U0))
         call pops_phi(c, bt(it,:))
      end if

      ! ---- energy on active surface (no extra potad) ----
      Et(it) = Vad(a) + 0.5d0*sum(p**2/mass)

      call evolve(q, p, qe, pe, Vad, U, dvdq, a, dt)

      if (q(1).ne.q(1)) then
         ierr = 1
         exit
      end if
   end do

   if (ierr.eq.0) then
      ! final point (t = nt*dt)
      if (rep.eq.'d') then
         c = dcmplx(qe, pe)
         call pops_phi(c, bt(nt+1,:))
      else if (rep.eq.'a') then
         c = dcmplx(matmul(qe, U), matmul(pe, U))
         call pops_phi(c, bt(nt+1,:))
      else if (rep.eq.'e') then
         c = dcmplx(matmul(qe, U0), matmul(pe, U0))
         call pops_phi(c, bt(nt+1,:))
      end if
      Et(nt+1) = Vad(a) + 0.5d0*sum(p**2/mass)
   else
      ! define remaining values deterministically (old code could leave junk)
      bt(it:nt+1,:) = 0.d0
      Et(it:nt+1)   = 0.d0
   end if

   if (rep.eq.'e') deallocate(U0)
   deallocate(Vad, U, dvdq, c)
end subroutine


subroutine runtrj_obs_ead(q, p, qe, pe, bt, Et, Ead, V0t, rep, dt, ierr, nt, nf_, ns)
   use types
   use pes,  only : potad, grad_a, mass, omega
   use mash, only : evolve, cstate, pops_phi
   integer, intent(in) :: nt, nf_, ns
   real(dp), intent(in) :: dt
   character, intent(in) :: rep
   real(dp), intent(inout) :: q(nf_), p(nf_), qe(ns), pe(ns)
   real(dp), intent(out) :: bt(nt+1,ns), Et(nt+1), Ead(nt+1), V0t(nt+1)
   integer, intent(out) :: ierr

   real(dp), allocatable :: Vad(:), U(:,:), dvdq(:)
   real(dp), allocatable :: U0(:,:), Vad0(:), q0(:)
   complex(dpc), allocatable :: c(:)
   real(dp), allocatable :: pop_ad(:)
   integer :: it, a

   ierr = 0

   allocate(Vad(ns), U(ns,ns), dvdq(nf_), c(ns), pop_ad(ns))

   if (rep.eq.'e') then
      allocate(U0(ns,ns), Vad0(ns), q0(nf_))
      q0 = 0.d0
      call potad(q0, Vad0, U0)
      deallocate(Vad0, q0)
   end if

   call potad(q, Vad, U)
   call cstate(qe, pe, U, a)
   call grad_a(q, U, a, dvdq)

   do it = 1, nt
      ! ---- bt in requested representation ----
      if (rep.eq.'d') then
         c = dcmplx(qe, pe)
         call pops_phi(c, bt(it,:))
      else if (rep.eq.'a') then
         c = dcmplx(matmul(qe, U), matmul(pe, U))
         call pops_phi(c, bt(it,:))
      else if (rep.eq.'e') then
         c = dcmplx(matmul(qe, U0), matmul(pe, U0))
         call pops_phi(c, bt(it,:))
      end if

      ! ---- Et on active surface ----
      Et(it) = Vad(a) + 0.5d0*sum(p**2/mass)

      ! ---- Ead = sum_a Phi_a * Vad(a) in adiabatic basis ----
      c = dcmplx(matmul(qe, U), matmul(pe, U))
      call pops_phi(c, pop_ad)
      Ead(it) = sum(pop_ad * Vad)

      ! ---- V0(t) = 1/2 sum_i m_i * (omega_i * q_i)^2  (matches linvib%pot) ----
      if (allocated(omega)) then
         V0t(it) = 0.5d0 * sum(mass * (omega*q)**2)
      else
         V0t(it) = 0.d0
      end if

      call evolve(q, p, qe, pe, Vad, U, dvdq, a, dt)

      if (q(1).ne.q(1)) then
         ierr = 1
         exit
      end if
   end do

   if (ierr.eq.0) then
      ! final point
      if (rep.eq.'d') then
         c = dcmplx(qe, pe)
         call pops_phi(c, bt(nt+1,:))
      else if (rep.eq.'a') then
         c = dcmplx(matmul(qe, U), matmul(pe, U))
         call pops_phi(c, bt(nt+1,:))
      else if (rep.eq.'e') then
         c = dcmplx(matmul(qe, U0), matmul(pe, U0))
         call pops_phi(c, bt(nt+1,:))
      end if

      Et(nt+1) = Vad(a) + 0.5d0*sum(p**2/mass)

      c = dcmplx(matmul(qe, U), matmul(pe, U))
      call pops_phi(c, pop_ad)
      Ead(nt+1) = sum(pop_ad * Vad)

      if (allocated(omega)) then
         V0t(nt+1) = 0.5d0 * sum(mass * (omega*q)**2)
      else
         V0t(nt+1) = 0.d0
      end if
   else
      bt(it:nt+1,:)  = 0.d0
      Et(it:nt+1)    = 0.d0
      Ead(it:nt+1)   = 0.d0
      V0t(it:nt+1)   = 0.d0
   end if

   if (rep.eq.'e') deallocate(U0)
   deallocate(Vad, U, dvdq, c, pop_ad)
end subroutine


! ============= Parallelized functions ==============
subroutine runpar(q, p, qe, pe, bt, Et, ierr, rep, dt, nt, nf, ns, np)
   use types
   use omp_lib
   integer :: nt, nf, ns, np
   real(dp), intent(in) :: dt
   real(dp), intent(inout) :: q(nf,np), p(nf,np), qe(ns,np), pe(ns,np)
   real(dp), intent(out) :: bt(nt+1,ns)
   real(dp), intent(out) :: Et(nt+1)
   integer, intent(out) :: ierr(np)
   character, intent(in) :: rep

   real(dp), allocatable :: dbt(:,:,:)
   real(dp), allocatable :: dEt(:,:)
   integer :: j

   allocate(dbt(nt+1,ns,np), dEt(nt+1,np))

!$omp parallel do default(shared) private(j)
   do j=1,np
      call runtrj_obs(q(:,j), p(:,j), qe(:,j), pe(:,j), dbt(:,:,j), dEt(:,j), rep, dt, ierr(j), nt, nf, ns)

      if (ierr(j).ne.0) then
         ! match old runpar behavior: discard pops, keep Et (but now Et is defined/zeroed after failure)
         dbt(:,:,j) = 0.d0
      end if
   end do
!$omp end parallel do

   bt = sum(dbt, dim=3)
   Et = sum(dEt, dim=2)

   deallocate(dbt, dEt)
end subroutine


subroutine runpar_ead(q, p, qe, pe, bt, Et, Ead, V0t, ierr, rep, dt, nt, nf, ns, np)
   use types
   use omp_lib
   integer :: nt, nf, ns, np
   real(dp), intent(in) :: dt
   real(dp), intent(inout) :: q(nf,np), p(nf,np), qe(ns,np), pe(ns,np)
   real(dp), intent(out) :: bt(nt+1,ns)
   real(dp), intent(out) :: Et(nt+1)
   real(dp), intent(out) :: Ead(nt+1)
   real(dp), intent(out) :: V0t(nt+1)
   integer, intent(out) :: ierr(np)
   character, intent(in) :: rep

   real(dp), allocatable :: dbt(:,:,:)
   real(dp), allocatable :: dEt(:,:), dEad(:,:), dV0(:,:)
   integer :: j

   allocate(dbt(nt+1,ns,np), dEt(nt+1,np), dEad(nt+1,np), dV0(nt+1,np))

!$omp parallel do default(shared) private(j)
   do j=1,np
      call runtrj_obs_ead(q(:,j), p(:,j), qe(:,j), pe(:,j), dbt(:,:,j), dEt(:,j), dEad(:,j), dV0(:,j), rep, dt, ierr(j), nt, nf, ns)

      if (ierr(j).ne.0) then
         dbt(:,:,j) = 0.d0
         dEt(:,j)   = 0.d0
         dEad(:,j)  = 0.d0
         dV0(:,j)   = 0.d0
      end if
   end do
!$omp end parallel do

   bt  = sum(dbt,  dim=3)
   Et  = sum(dEt,  dim=2)
   Ead = sum(dEad, dim=2)
   V0t = sum(dV0,  dim=2)

   deallocate(dbt, dEt, dEad, dV0)
end subroutine
