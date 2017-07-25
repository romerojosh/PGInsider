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
program main
  use compare_utils
  use cublas
  use cudafor
  use cusolverDn
  use funcs
  use nvtx
  use zhegst_gpu
  implicit none
  
  integer :: N, nb, i, lda, istat, Lwork
  character(len=20) :: arg
  real(8) :: wallclock, ts, te
  complex(8), dimension(:,:), allocatable :: Aref, Bref, A1, B1
  complex(8), dimension(:,:), allocatable, device :: A1_d, A2_d, A3_d, A4_d, B1_d, B2_d, B3_d
  complex(8), dimension(:), allocatable, device :: workspace_d
  integer, device :: devInfo_d
  type(cusolverDnHandle) :: h


  ! Parse command line arguments
  i = command_argument_count()

  if (i == 1) then
    ! If N is provided, generate random hermetian matrices for A and B
    print*, "Using randomly-generated matrices..."
    call get_command_argument(1, arg)
    read(arg, *)  N
    print*, "Running with N = ", N
    print*
    lda = N

    ! Create random positive-definite hermetian matrices on host
    call create_random_hermetian_pd(Aref, N)
    call create_random_hermetian_pd(Bref, N)

  else
    print*, "Usage:\n\t ./main [N]"
    call exit
  endif

  ! Copy matrices to device
  allocate(A1, source = Aref)
  allocate(A1_d, source = Aref)
  allocate(A2_d, source = Aref)
  allocate(A3_d, source = Aref)
  allocate(A4_d, source = Aref)

  allocate(B1, source = Bref)
  allocate(B1_d, source = Bref)
  allocate(B2_d, source = Bref)
  allocate(B3_d, source = Bref)

  ! Initialize magma, cublas, etc
  istat = cublasInit
  if (istat /= CUBLAS_STATUS_SUCCESS) write(*,*) 'cublas intialization failed'

#ifdef HAVE_MAGMA
  call magmaf_init
#endif

  ! TASK 1:  Perform cholesky factorization of B, B := L ---------------------
  print*, "Performing cholesky factorizations...."
  ! CASE 1: Factor using Intel MKL 
  call zpotrf('L', N, B1, lda, istat)
  B1 = Bref
  ts = wallclock()
  call nvtxStartRange("MKL zpotrf")
  call zpotrf('L', N, B1, lda, istat)
  call nvtxEndRange
  te = wallclock()
  if (istat /= 0) write(*,*) 'MKL zpotrf failed'
  print*, "\tTime for MKL zpotrf = ", (te - ts) * 1000.0d0
  print*

  B1_d = B1

  ! CASE 2: Factor using CUSOLVER
  istat = cusolverDnCreate(h)
  if (istat /= CUSOLVER_STATUS_SUCCESS) write(*,*) 'handle creation failed'

  istat = cusolverDnZpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, N, B2_d, lda, Lwork)
  if (istat /= CUSOLVER_STATUS_SUCCESS) write(*,*) 'cusolverDnZpotrf_buffersize failed'

  allocate(workspace_d(Lwork))

  istat = cusolverDnZpotrf(h, CUBLAS_FILL_MODE_LOWER, N, B2_d, lda, workspace_d, Lwork, devInfo_d)
  B2_d = Bref
  ts = wallclock()
  call nvtxStartRange("cusolverDnZpotrf")
  istat = cusolverDnZpotrf(h, CUBLAS_FILL_MODE_LOWER, N, B2_d, lda, workspace_d, Lwork, devInfo_d)
  call nvtxEndRange
  te = wallclock()
  print*, "\tTime for cusolverDnZpotrf = ", (te - ts) * 1000.0d0
  call compare_ltri(B1_d, B2_d, N, N)
  print*

  if (istat /= CUSOLVER_STATUS_SUCCESS) write(*,*) 'cusolverDnZpotrf failed'

  istat = devInfo_d
  if (istat /= 0) write(*,*) 'Cholesky factorization failed'

  istat = cusolverDnDestroy(h)
  if (istat /= CUSOLVER_STATUS_SUCCESS) write(*,*) 'handle destruction failed'

#ifdef HAVE_MAGMA
  ! CASE 3: Factor using MAGMA
  call magmaf_zpotrf_gpu('L', N, B3_d, lda, istat)
  B3_d = Bref
  ts = wallclock()
  call nvtxStartRange("magmaf_zpotrf_gpu")
  call magmaf_zpotrf_gpu('L', N, B3_d, lda, istat)
  call nvtxEndRange
  te = wallclock()
  if (istat /= 0) write(*,*) 'magmaf_zpotrf failed'
  print*, "\tTime for magmaf_zpotrf_gpu = ", (te - ts) * 1000.0d0
  call compare_ltri(B1_d, B3_d, N, N)
  print*
#endif


  ! TASK 2:  Reduce generalized Eigenproblem to standard form ----------------
  print*, "Performing reduction to generalized Eigenproblem...."
  ! CASE 1: Perform computation on CPU using MKL
  call zhegst(1, 'L', N, A1, lda, B1, lda, istat)
  A1 = Aref
  ts = wallclock()
  call nvtxStartRange("MKL")
  call zhegst(1, 'L', N, A1, lda, B1, lda, istat)
  call nvtxEndRange
  te = wallclock()
  if (istat /= 0) write(*,*) 'MKL zhegst failed'
  print*, "\tTime for MKL zhegst = ", (te - ts) * 1000.0d0
  print*

  A1_d = A1

  ! CASE 2: Perform computation using 2 cublas ZTRSM calls
  call zhegst_gpu_v1(1, 'L', N, A2_d, lda, B2_d, lda)
  A2_d = Aref
  ts = wallclock()
  call nvtxStartRange("GPU V1", 1)
  call zhegst_gpu_v1(1, 'L', N, A2_d, lda, B2_d, lda)
  call nvtxEndRange
  te = wallclock()

  print*, "\tTime for zhegst_gpu_v1 = ", (te - ts) * 1000.0d0
  call compare_ltri(A1_d, A2_d, N, N)
  print*

  ! CASE 3: Perform computation using 2 cublas ZTRSM calls for subblock
  nb = 448
  call zhegst_gpu_v2(1, 'L', N, A3_d, lda, B2_d, lda, nb)
  A3_d = Aref
  ts = wallclock()
  call nvtxStartRange("GPU V2", 0)
  call zhegst_gpu_v2(1, 'L', N, A3_d, lda, B2_d, lda, nb)
  call nvtxEndRange
  te = wallclock()
  print*, "\tTime for zhegst_gpu_v2 = ", (te - ts) * 1000.0d0
  call compare_ltri(A1_d, A3_d, N, N)
  print*

#ifdef HAVE_MAGMA
   ! CASE 4: Perform computation using MAGMA routine
  call magmaf_zhegst_gpu(1, 'L', N, A4_d, lda, B3_d, lda, istat)
  A4_d = Aref
  ts = wallclock()
  call nvtxStartRange("MAGMA", 0)
  call magmaf_zhegst_gpu(1, 'L', N, A4_d, lda, B3_d, lda, istat)
  call nvtxEndRange
  te = wallclock()
  if (istat /= 0) write(*,*) 'magma zhegst failed'
  print*, "\tTime for magmaf_zhegst_gpu = ", (te - ts) * 1000.0d0
  call compare_ltri(A1_d, A4_d, N, N)
  print*
#endif


end program main
