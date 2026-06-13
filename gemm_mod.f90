! Original Author: Arjen Tamerus at DiRAC Hackathon Feb 2024
! Released to AMD as sample code
module gemm_mod
        use hipfort
        use hipblas
        use iso_c_binding
        implicit none

        public :: do_gemm

        contains

        subroutine do_gemm()
                use hipblas_enums

                complex(kind=c_double_complex), dimension(:,:), allocatable, target :: A, B, C

                integer(kind=c_int) :: m, n, k, lda, ldb, ldc
                complex(kind=c_double_complex), target :: alpha, beta
                type(c_ptr) :: handle

                integer(kind(HIPBLAS_OP_N)) :: ta, tb
                integer :: status

                ta = HIPBLAS_OP_N
                tb = HIPBLAS_OP_N

                m = 1024
                n = 1024
                k = 1024
                lda = 1024
                ldb = 1024
                ldc = 1024

                allocate(A(m,k))
                allocate(B(k,n))
                allocate(C(m,n))

                alpha = 1.0
                beta = 1.0

                A = cmplx(1.0,0.0,c_double_complex)
                B = cmplx(2.0,0.0,c_double_complex)
                C = cmplx(0.0,0.0,c_double_complex)

                status = hipblasCreate(handle)
                write(0,*) "CREATE", status, status .eq. HIPBLAS_STATUS_SUCCESS

                !$omp target data map(to:A,B) map(tofrom:C)
                !$omp target data use_device_addr(A,B,C)
                status = hipblasZgemm(handle, ta, tb, &
                        m, n, k, c_loc(alpha), c_loc(A), lda, & 
                        c_loc(B), ldb, c_loc(beta), c_loc(C), ldc)

                status = hipdevicesynchronize()
                !$omp end target data
                !$omp end target data

                status = hipblasDestroy(handle)

                write(0,*) "ZGEMM", status, status .eq. HIPBLAS_STATUS_SUCCESS

                write(0,*) "C(1,1):", C(1,1)

                deallocate(A,B,C)

        end subroutine

end module
