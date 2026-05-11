! Generic pes module

module pes
   use types
   integer :: nf ! number of nuclear degrees of freedom
   integer :: ns ! number of electronic states
   real(dp), allocatable :: mass(:), omega(:)
   procedure (pot_interface), pointer :: pot
   procedure (potc_interface), pointer :: potc
   procedure (grad_interface), pointer :: grad
   procedure (grad_a_interface), pointer :: grad_a
    procedure (grad_a_c_interface), pointer :: grad_a_c
   procedure (grad_ab_interface), pointer :: grad_ab
   procedure (grad_av_interface), pointer :: grad_av
   procedure (get_vlin_interface), pointer :: get_vlin
   procedure (get_wqud_interface), pointer :: get_wqud
   procedure (get_vconst_interface), pointer :: get_vconst
   logical :: cayley = .false.
   logical :: tc_complex_mode = .false.

   interface
      subroutine pot_interface(q,V)
         ! Diabatic potential matrix
         import ns,nf,dp
         real(dp), intent(in) :: q(nf)
         real(dp), intent(out) :: V(ns,ns)
      end subroutine pot_interface

      subroutine potc_interface(q,V)
         ! Diabatic potential matrix (complex Hermitian)
         import ns,nf,dp,dpc
         real(dp), intent(in) :: q(nf)
         complex(dpc), intent(out) :: V(ns,ns)
      end subroutine potc_interface

      subroutine grad_interface(q,dVdq)
         ! Diabatic gradient matrix
         import nf,ns,dp
         real(dp), intent(in) :: q(:)
         real(dp), intent(out) :: dVdq(nf,ns,ns)
      end subroutine grad_interface

      subroutine grad_ab_interface(q,U,dVdq)
         ! Adiabatic gradient matrix
         import nf,ns,dp
         real(dp), intent(in) :: q(:), U(:,:)
         real(dp), intent(out) :: dVdq(nf,ns,ns)
      end subroutine grad_ab_interface

      subroutine grad_av_interface(q,U,Gad,a)
         ! Adiabatic gradient vector
         import nf,ns,dp
         real(dp), intent(in) :: q(:), U(:,:)
         integer :: a
         real(dp), intent(out) :: Gad(nf,ns)
      end subroutine grad_av_interface

      subroutine grad_a_interface(q,U,a,dVdq)
         ! Adiabatic gradient of single state
         import nf,dp
         real(dp), intent(in) :: q(:), U(:,:)
         integer :: a
         real(dp), intent(out) :: dVdq(nf)
      end subroutine grad_a_interface

      subroutine grad_a_c_interface(q,U,a,dVdq)
         ! Adiabatic gradient of single state (complex eigenvectors)
         import nf,dp,dpc
         real(dp), intent(in) :: q(:)
         complex(dpc), intent(in) :: U(:,:)
         integer :: a
         real(dp), intent(out) :: dVdq(nf)
      end subroutine grad_a_c_interface

      subroutine get_vconst_interface(Vconst)
         ! Constant part of potential
         import ns,dp
         real(dp), intent(out) :: Vconst(ns,ns)
      end subroutine get_vconst_interface

      subroutine get_vlin_interface(Vlin)
         ! Linear part of potential
         import nf,ns,dp
         real(dp), intent(out) :: Vlin(nf,ns,ns)
      end subroutine get_vlin_interface

      subroutine get_wqud_interface(Wqud)
         ! Linear part of potential
         import nf,ns,dp
         real(dp), intent(out) :: Wqud(nf,ns,ns)
      end subroutine get_wqud_interface
   end interface

   contains
   subroutine init(nf_,ns_,mass_)
      real(dp) :: mass_(:)
      nf = nf_
      ns = ns_
      if (.not. allocated(mass)) allocate(mass(nf))
      mass = mass_
      ! Default adiabatic routines: compute from diabatic 
      grad_a => gradad_diag
      grad_ab => gradad
      grad_av => grad_vector
      tc_complex_mode = .false.
      nullify(potc)
      nullify(grad_a_c)
   end subroutine

   subroutine potad(q,Vad,U)
      use maths, only : symevp
      real(dp), intent(in) :: q(:)
      real(dp), intent(out) :: Vad(ns), U(ns,ns)
!
!     Compute full set of adiabatic potentials and eigenstates
!
      call pot(q,U)
      call symevp(U,ns,ns,Vad,ierr)
   end subroutine

   subroutine potad_complex(q,Vad,U)
      real(dp), intent(in) :: q(:)
      real(dp), intent(out) :: Vad(ns)
      complex(dpc), intent(out) :: U(ns,ns)
      complex(dpc), allocatable :: work(:)
      real(dp), allocatable :: rwork(:)
      integer, allocatable :: iwork(:)
      complex(dpc) :: workq(1)
      real(dp) :: rworkq(1)
      integer :: iworkq(1)
      integer :: lwork, lrwork, liwork
      integer :: info
!
!     Compute full set of adiabatic potentials/eigenvectors for
!     complex-Hermitian diabatic Hamiltonians.
!
      if (.not. associated(potc)) then
         error stop 'potad_complex called but potc pointer is not associated'
      end if
      call potc(q,U)

      lwork = -1
      lrwork = -1
      liwork = -1
      call zheevd('V','U',ns,U,ns,Vad,workq,lwork,rworkq,lrwork,iworkq,liwork,info)
      if (info /= 0) then
         error stop 'zheevd workspace query failed in potad_complex'
      end if

      lwork = max(1, int(real(workq(1), dp)))
      lrwork = max(1, int(rworkq(1)))
      liwork = max(1, iworkq(1))
      allocate(work(lwork), rwork(lrwork), iwork(liwork))
      call zheevd('V','U',ns,U,ns,Vad,work,lwork,rwork,lrwork,iwork,liwork,info)
      deallocate(work, rwork, iwork)
      if (info /= 0) then
         error stop 'zheevd diagonalization failed in potad_complex'
      end if
   end subroutine

   subroutine gradad_diag(q, U, a, dvdq)
      real(dp), intent(in) :: q(:), U(:,:)
      integer :: a
      real(dp), intent(out) :: dvdq(nf)
!
!     Compute gradient of potential for a given adiabatic state.
!     (This is faster than "gradad", which computes all elements of the adiabatic gradient matrix)
!
      real(dp), allocatable :: Gdia(:,:,:)

      allocate(Gdia(nf,ns,ns))
      
      ! get H' in diabatic basis
      call grad(q,Gdia)

      ! convert H' to adia. Only compute needed bits of Gad
      do i=1,nf
         dvdq(i) = dot_product(U(:,a),matmul(Gdia(i,:,:),U(:,a)))
      end do

      deallocate(Gdia)
   end subroutine

   subroutine gradad(q,U,Gad)
      real(dp), intent(in) :: q(:), U(:,:)
      real(dp), intent(out) :: Gad(nf,ns,ns)
!
!     Gradient in adiabatic representation
!     U -- rotation from adiabatic to diabatic representation (output of potad)
!
      real(dp), allocatable :: Gdia(:,:,:)

      allocate(Gdia(nf,ns,ns))
      
      ! get H' in diabatic basis
      call grad(q,Gdia)

      ! convert H' to adia
      do i=1,nf
         Gad(i,:,:) = matmul(transpose(U),matmul(Gdia(i,:,:),U))
      end do
      deallocate(Gdia)
   end subroutine

   subroutine grad_vector(q,U,Gad,b)
      real(dp), intent(in) :: q(:), U(:,:)
      integer :: b
      real(dp), intent(out) :: Gad(nf,ns)
!
!     Returns Gad(i,a,b) for a given adiabatic state b
!
      real(dp), allocatable :: Gdia(:,:,:)

      allocate(Gdia(nf,ns,ns))
      
      ! get H' in diabatic basis
      call grad(q,Gdia)

      ! convert H' to adia.
      do i=1,nf
         Gad(i,:) = matmul(Gdia(i,:,:),U(:,b))
         Gad(i,:) = matmul(transpose(U),matmul(Gdia(i,:,:),U(:,b)))
      end do

      deallocate(Gdia)
   end subroutine

   subroutine nac(q,Vad,U,d)
      real(dp), intent(in) :: q(:), Vad(:), U(:,:)
      real(dp), intent(out) :: d(nf,ns,ns)
!
!     Nonadiabatic coupling vector using Hellman-Feynman theorem
!
      real(dp), allocatable :: Gad(:,:,:)
      allocate(Gad(nf,ns,ns))
      call grad_ab(q,U,Gad)
      do k=1,ns
         d(:,k,k) = 0.d0
         do l=k+1,ns
            d(:,k,l) = Gad(:,k,l)/(Vad(l)-Vad(k))
            d(:,l,k) = -d(:,k,l)
         end do
      end do
      deallocate(Gad)
   end subroutine

   subroutine nac_complex(q,Vad,U,d)
      real(dp), intent(in) :: q(:), Vad(:)
      complex(dpc), intent(in) :: U(:,:)
      real(dp), intent(out) :: d(nf,ns,ns)
!
!     Nonadiabatic coupling vector using Hellmann-Feynman theorem
!     for complex adiabatic eigenvectors.
!
      real(dp), allocatable :: Gdia(:,:,:)
      complex(dpc), allocatable :: tmp(:)
      complex(dpc) :: gkl
      real(dp) :: denom

      allocate(Gdia(nf,ns,ns), tmp(ns))
      call grad(q,Gdia)
      do i=1,nf
         do k=1,ns
            d(i,k,k) = 0.d0
            tmp = matmul(Gdia(i,:,:), U(:,k))
            do l=k+1,ns
               denom = Vad(l)-Vad(k)
               if (abs(denom).gt.1.d-14) then
                  gkl = dot_product(U(:,l), tmp)
                  d(i,k,l) = real(gkl/denom)
               else
                  d(i,k,l) = 0.d0
               end if
               d(i,l,k) = -d(i,k,l)
            end do
         end do
      end do
      deallocate(Gdia, tmp)
   end subroutine

   subroutine nac_a(q,b,Vad,U,d)
      real(dp), intent(in) :: q(:), Vad(:), U(:,:)
      integer :: a,b
      real(dp), intent(out) :: d(nf,ns)
!
!     Returns components d(i,a,b) for a given adiabatic state b
!
      real(dp), allocatable :: Gad(:,:)
      allocate(Gad(nf,ns))
      call grad_av(q,U,Gad,b)
      do a=1,ns
         if (a.eq.b) then
            d(:,a) = 0.d0
         else
            d(:,a) = Gad(:,a)/(Vad(b)-Vad(a))
         end if
      end do
      deallocate(Gad)
   end subroutine


   subroutine nacdir(q,cad,Vad,U,a,b,dj)
      real(dp), intent(in) :: q(:), Vad(:), U(:,:)
      complex(dpc), intent(in) :: cad(:)
      real(dp), intent(out) :: dj(nf)
      integer :: a,b
!
!     Direction of momentum rescaling/reversal.
!     Note: does not include masses
!
      real(dp), allocatable :: d(:,:)
      allocate(d(nf,ns))
      call nac_a(q,a,Vad,U,d)
      dj = 0.d0
      do k=1,ns
         dj = dj + d(:,k)*real(conjg(cad(k))*cad(a))
      end do
      call nac_a(q,b,Vad,U,d)
      do k=1,ns
         dj = dj - d(:,k)*real(conjg(cad(k))*cad(b))
      end do
      deallocate(d)
   end subroutine

   subroutine nacdir_complex(q,cad,Vad,U,a,b,dj)
      real(dp), intent(in) :: q(:), Vad(:)
      complex(dpc), intent(in) :: cad(:), U(:,:)
      real(dp), intent(out) :: dj(nf)
      integer :: a,b
      real(dp), allocatable :: d(:,:), Gdia(:,:,:)
      complex(dpc), allocatable :: tmp(:)
      complex(dpc) :: gka
      real(dp) :: denom
!
!     Direction of momentum rescaling/reversal for complex adiabatic basis.
!     Uses d(i,k,a) = Re[ <k|dV/dq_i|a> / (Va - Vk) ].
!
      allocate(d(nf,ns), Gdia(nf,ns,ns), tmp(ns))
      call grad(q,Gdia)
      dj = 0.d0

      do i=1,nf
         tmp = matmul(Gdia(i,:,:), U(:,a))
         do k=1,ns
            if (k.eq.a) then
               d(i,k) = 0.d0
            else
               denom = Vad(a)-Vad(k)
               if (abs(denom).gt.1.d-14) then
                  gka = dot_product(U(:,k), tmp)
                  d(i,k) = real(gka/denom)
               else
                  d(i,k) = 0.d0
               end if
            end if
         end do
      end do
      do k=1,ns
         dj = dj + d(:,k)*real(conjg(cad(k))*cad(a))
      end do

      do i=1,nf
         tmp = matmul(Gdia(i,:,:), U(:,b))
         do k=1,ns
            if (k.eq.b) then
               d(i,k) = 0.d0
            else
               denom = Vad(b)-Vad(k)
               if (abs(denom).gt.1.d-14) then
                  gka = dot_product(U(:,k), tmp)
                  d(i,k) = real(gka/denom)
               else
                  d(i,k) = 0.d0
               end if
            end if
         end do
      end do
      do k=1,ns
         dj = dj - d(:,k)*real(conjg(cad(k))*cad(b))
      end do
      deallocate(d, Gdia, tmp)
   end subroutine

end module