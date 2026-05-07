! quadratic vibronic coupling potential

module qudvib

   use types
   implicit none

   ! Vconst  : constant diabatic matrix elements  V_const(n,m)
   ! Vlin    : linear vibronic couplings          Vlin(alpha,n,m)
   ! Wqud    : quadratic vibronic couplings       Wqud(alpha,n,m)
   real(dp), allocatable :: Vconst(:,:), Vlin(:,:,:), Wqud(:,:,:)

contains

   subroutine init(nf_,ns_,mass_,omega_,Vconst_,Vlin_,Wqud_)
      use pes, only : pesinit=>init, potptr=>pot, gradptr=>grad, nf, ns, omega, cayley
      use pes, only : pes_get_vconst=>get_vconst, pes_get_vlin=>get_vlin, pes_get_wqud=>get_wqud
      implicit none
      integer :: nf_, ns_
      real(dp) :: mass_(nf_), omega_(nf_)
      real(dp) :: Vconst_(ns_,ns_), Vlin_(nf_,ns_,ns_), Wqud_(nf_,ns_,ns_)

!
!     Initialize module
!
      call pesinit(nf_,ns_,mass_)
      potptr       => pot
      gradptr      => grad
      pes_get_vconst => get_vconst
      pes_get_vlin   => get_vlin
      pes_get_wqud   => get_wqud
      call pesinit(nf_,ns_,mass_)

      allocate(omega(nf), Vconst(ns,ns), Vlin(nf,ns,ns), Wqud(nf,ns,ns))
      omega  = omega_
      Vconst = Vconst_
      Vlin   = Vlin_
      Wqud   = Wqud_

      cayley = .true.  ! same behavior as in LVC: skip harmonic gradient
   end subroutine init


   subroutine pot(q, V)
      use pes, only : mass, ns, nf, omega
      implicit none
      real(dp), intent(in)  :: q(nf)
      real(dp), intent(out) :: V(ns,ns)

      ! local
      real(dp) :: V0
      integer  :: n, m

!
!     Diabatic potential matrix for quadratic vibronic coupling
!
!     V_nm(q) = Vconst_nm
!             + sum_alpha Vlin(alpha,n,m) * q_alpha
!             + sum_alpha Wqud(alpha,n,m) * q_alpha^2
!             + delta_nm * (1/2 sum_alpha m_alpha omega_alpha^2 q_alpha^2)
!
      V0 = 0.5_dp * sum( mass * (omega*q)**2 )

      do n = 1, ns
         do m = 1, ns
            V(n,m) = Vconst(n,m)                                  &
                  + sum( Vlin(:,n,m) * q )                        &
                  + sum( Wqud(:,n,m) * q**2 )
         end do
         V(n,n) = V(n,n) + V0
      end do
   end subroutine pot


   subroutine grad(q, G)
      use pes, only : mass, nf, ns, omega, cayley
      implicit none
      real(dp), intent(in)  :: q(:)
      real(dp), intent(out) :: G(nf,ns,ns)

      ! local
      real(dp), allocatable :: G0(:)
      integer :: i

!
!     Analytic gradient of diabatic potential:
!
!     dV_nm/dq_alpha = Vlin(alpha,n,m)
!                    + 2 * Wqud(alpha,n,m) * q_alpha
!                    + delta_nm * m_alpha * omega_alpha^2 * q_alpha
!
!     As in linvib, the harmonic contribution (last term) is skipped
!     when cayley = .true., but the quadratic couplings are always included.
!
      ! Start with linear contribution
      G = Vlin

      ! Add quadratic contribution: 2 Wqud(alpha,n,m) * q_alpha
      do i = 1, nf
         G(i,:,:) = G(i,:,:) + 2.0_dp * q(i) * Wqud(i,:,:)
      end do

      ! If Cayley integrator used for the harmonic reference, skip adding
      ! the state-independent harmonic gradient (same logic as linvib).
      if (cayley) return

      ! Add harmonic part: d/dq_alpha [ 1/2 m omega^2 q^2 ] = m omega^2 q
      allocate(G0(nf))
      G0 = mass * omega**2 * q

      do i = 1, ns
         G(:,i,i) = G(:,i,i) + G0
      end do

      deallocate(G0)
   end subroutine grad


   subroutine get_vconst(Vconst_)
      use pes, only : ns
      implicit none
      real(dp), intent(out) :: Vconst_(ns,ns)

      Vconst_ = Vconst
   end subroutine get_vconst


   subroutine get_vlin(Vlin_)
      use pes, only : ns, nf
      implicit none
      real(dp), intent(out) :: Vlin_(nf,ns,ns)

      Vlin_ = Vlin
   end subroutine get_vlin


   subroutine get_wqud(Wqud_)
      use pes, only : ns, nf
      implicit none
      real(dp), intent(out) :: Wqud_(nf,ns,ns)

      Wqud_ = Wqud
   end subroutine get_wqud

end module qudvib
