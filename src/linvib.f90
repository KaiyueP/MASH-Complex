! Linear vibronic coupling potential

module linvib

   use types

   real(dp), allocatable :: Vconst(:,:), Vlin(:,:,:)

contains

   subroutine init(nf_,ns_,mass_,omega_,Vconst_,Vlin_)
      use pes, only : pesinit=>init, potptr=>pot, gradptr=>grad, nf, ns, omega, cayley
      use pes, only : pes_get_vconst=>get_vconst,pes_get_vlin=>get_vlin
      use pes, only : grad_a
      integer :: nf_, ns_
      real(dp) :: mass_(nf_), omega_(nf_)
      real(dp) :: Vconst_(ns_,ns_), Vlin_(nf_,ns_,ns_)
!
!     Initialize module
!
      call pesinit(nf_,ns_,mass_)
      potptr => pot
      gradptr => grad
      pes_get_vconst => get_vconst
      pes_get_vlin => get_vlin
      if (allocated(omega))  deallocate(omega)
      if (allocated(Vconst)) deallocate(Vconst)
      if (allocated(Vlin))   deallocate(Vlin)
      allocate(omega(nf), Vconst(ns,ns), Vlin(nf,ns,ns))
      omega = omega_
      Vconst = Vconst_
      Vlin = Vlin_
      cayley = .true. !.true.
      grad_a => grad_a_linvib

   end subroutine

   ! ============================================================
   ! >>> Allocation-free adiabatic force for LVC
   !     Avoids allocating/copying Gdia(nf,ns,ns) in pes%gradad_diag
   ! ============================================================
   subroutine grad_a_linvib(q, U, a, dvdq)
      use pes, only : nf, ns, cayley, mass, omega
      real(dp), intent(in)  :: q(:), U(:,:) 
      integer :: a
      real(dp), intent(out) :: dvdq(nf)

      integer :: i
      real(dp) :: tmp(ns)

      ! --- Linear vibronic part: Gdia(i,:,:) = Vlin(i,:,:)
      do i = 1, nf
         tmp      = matmul(Vlin(i,:,:), U(:,a))
         dvdq(i)  = dot_product(U(:,a), tmp)
      end do

      ! --- If not using Cayley, harmonic force contributes equally to all states
      !     because V0(q) * I is added to the diabatic matrix.
      if (.not. cayley) then
         dvdq = dvdq + mass * omega**2 * q
      end if
   end subroutine grad_a_linvib



   subroutine pot(q, V)
      use pes, only : mass,ns,nf,omega
      real(dp), intent(in) :: q(nf)
      real(dp), intent(out) :: V(ns,ns)
!
!     Diabatic potential matrix
!
      real(dp) :: V0
      V0 = 0.5d0 * sum(mass*(omega*q)**2)
      do n=1,ns
         do m=1,ns
            V(n,m) = Vconst(n,m) + sum(Vlin(:,n,m)*q)
         end do
         V(n,n) = V(n,n) + V0
      end do
   end subroutine


   subroutine grad(q, G)
      use pes, only : mass,nf,ns,omega,cayley
      real(dp), intent(in) :: q(:)
      real(dp), intent(out) :: G(nf,ns,ns)
!
!     Analytic gradient of diabatic potential 
!
      real(dp), allocatable :: G0(:)
      integer :: i
      G = Vlin
      if (cayley) return

      allocate(G0(nf))
      G0 = mass*omega**2 * q
      do i=1,ns
         G(:,i,i) = G(:,i,i) + G0
      end do
      deallocate(G0)
   end subroutine

   subroutine get_vconst(Vconst_)
      use pes, only : ns
      real(dp), intent(out) :: Vconst_(ns,ns)
      Vconst_ = Vconst
   end subroutine

   subroutine get_vlin(vlin_)
      use pes, only : ns, nf
      real(dp), intent(out) :: vlin_(nf,ns,ns)
      vlin_ = vlin
   end subroutine


end module
