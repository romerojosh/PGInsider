!
! Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
!
!
! Permission is hereby granted, free of charge, to any person obtaining a
! copy of this software and associated documentation files (the "Software"),
! to deal in the Software without restriction, including without limitation
! the rights to use, copy, modify, merge, publish, distribute, sublicense,
! and/or sell copies of the Software, and to permit persons to whom the
! Software is furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
! THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
! FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
! DEALINGS IN THE SOFTWARE.
!
module zhegst_gpu
  use cudafor
  use cublas
  use nvtx

  contains

  ! zhegst completed with 2 global triangular solves on gpu
  subroutine zhegst_gpu_v1(itype, uplo, N, A, lda, B, ldb)
    implicit none
    integer, intent(in) :: itype, N, lda, ldb
    character, intent(in) :: uplo
    complex(8), device, dimension(1:ldb,1:N), intent(in) :: B
    complex(8), device, dimension(1:lda,1:N) :: A
    complex(8), parameter :: cone = cmplx(1.,0,8)
    integer :: i, j, istat
    type(cublasHandle) :: H1

    if (itype .ne. 1 .or. uplo == 'U') then
      write(*,*), "Provided problem configuration not supported!"
      return
    endif

    istat = cublasCreate(H1)

    ! Populate A matrix with complete Hermitian entries
    !$cuf kernel do(2) <<<*,*>>>
    do j = 1, N
      do i = 1, N
        if (j > i) then
          A(i,j) = conjg(A(j,i))
        endif
      end do
    end do

    istat =  cublasztrsm_v2(H1, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, &
                            CUBLAS_OP_C, CUBLAS_OP_N, N, N, &
                            cone, B, ldb, A, lda)  
    istat =  cublasztrsm_v2(H1, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, &
                            CUBLAS_OP_N, CUBLAS_OP_N, N, N, &
                            cone, B, ldb, A, lda)  

    istat = cublasDestroy(H1)
  end subroutine zhegst_gpu_V1

  ! zhegst completed in blocks, using 2 ztrsms to solve subblock problem
  subroutine zhegst_gpu_v2(itype, uplo, N, A, lda, B, ldb, nb)
    implicit none
    integer, intent(in) :: itype, N, lda, ldb, nb
    character, intent(in) :: uplo
    complex(8), device, dimension(1:ldb, 1:N), intent(in) :: B
    complex(8), device, dimension(1:lda, 1:N), intent(inout) :: A
    complex(8), parameter :: cone = cmplx(1.0,0.0,8)
    complex(8), parameter :: chalf = cmplx(0.5,0.0,8)
    real(8), parameter :: one = 1.0_8 
    type(dim3) :: threads, blocks

    integer :: i, j, k, kb, istat
    integer(kind = cuda_stream_kind) :: stream1, stream2
    type(cublasHandle) :: H1

    if (itype .ne. 1 .or. uplo == 'U') then
      write(*,*), "Provided problem configuration not supported!"
      return
    endif

    ! Setup cublas handle and streams
    istat = cublasCreate(H1)
    istat = cudaStreamCreate(stream1) ! "CPU" stream
    istat = cudaStreamCreate(stream2) ! "GPU" stream

    do k = 1, N, nb
      kb = min(N-k+1, nb)

      istat = cublasSetStream(H1, stream1)

      call nvtxStartRangeAsync("Subblock computation", 3)
      ! Populate subblock with complete Hermitian entries
      !$cuf kernel do(2) <<<*,*,0,stream1>>>
      do j = k,k+kb-1
        do i = k,k+kb-1
          if (j > i) then
            A(i,j) = conjg(A(j,i))
          endif
        end do
      end do

      ! Solve subblock problem (results in fully populated A subblock)
      istat = cublasztrsm_v2(H1, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, &
                              CUBLAS_OP_C, CUBLAS_OP_N, kb, kb, &
                              cone, B(k,k), ldb, A(k,k), lda)  
      istat = cublasztrsm_v2(H1, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, &
                              CUBLAS_OP_N, CUBLAS_OP_N, kb, kb, &
                              cone, B(k,k), ldb, A(k,k), lda)  

      if (k + kb .le. N) then
        ! Perform bulk matrix update
        istat = cublasSetStream(H1, stream2)

        istat = cublasztrsm_v2(H1, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, &
                                CUBLAS_OP_C, CUBLAS_OP_N, N-k-kb+1, kb, cone, &
                                B(k, k), ldb, A(k+kb, k), lda) 

        istat = cudaStreamSynchronize(stream1)
        call nvtxEndRangeAsync

        ! Since the A subblock is fully populated, use zgemm instead of zhemm here
        istat = cublaszgemm_v2(H1, CUBLAS_OP_N, CUBLAS_OP_N, N-k-kb+1, kb, kb, &
                                -chalf, B(k+kb, k), ldb, A(k, k), lda, cone, &
                                A(k+kb, k), lda)

        istat = cublaszher2k_v2(H1, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N-k-kb+1, &
                                kb, -cone, A(k+kb, k), lda, B(k+kb, k), ldb, one, &
                                A(k+kb, k+kb), lda)

        istat = cudaStreamSynchronize(stream2)

        ! Again, we use zgemm instead of zhemm here
        istat = cublaszgemm_v2(H1, CUBLAS_OP_N, CUBLAS_OP_N, N-k-kb+1, kb, kb, &
                                -chalf, B(k+kb, k), ldb, A(k, k), lda, cone, &
                                A(k+kb, k), lda)

        istat = cublasztrsm_v2(H1, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, &
                               CUBLAS_OP_N, CUBLAS_OP_N, N-k-kb+1, kb, cone, &
                               B(k+kb, k+kb), ldb, A(k+kb, k), lda) 
      else
        istat = cudaStreamSynchronize(stream1)
        call nvtxEndRangeAsync
      end if
    end do

    ! Cleanup
    istat = cublasDestroy(H1)
    istat = cudaStreamDestroy(stream1)
    istat = cudaStreamDestroy(stream2)

  end subroutine zhegst_gpu_v2

end module zhegst_gpu
