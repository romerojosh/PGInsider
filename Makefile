# Modify these entries as needed
MKLDIR = ${MKLROOT}
CUDADIR = ${CUDA_HOME}
MAGMADIR =

FLAGS = -O3 -mp -pgf90libs -Mcuda=cc60,cuda8.0 -Mlarge_arrays

LIBS = -L${MKLDIR}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_pgi_thread -pgf90libs -lpthread -lm -ldl
LIBS += -Mcudalib=cublas,cusolver -L${CUDADIR}/lib64 -lnvToolsExt
INCS = -I${MKLDIR}/include

# If MAGMADIR is provided, compile with MAGMA
ifneq ($(MAGMADIR),)
FLAGS += -DHAVE_MAGMA
LIBS += -L${MAGMADIR}/lib -lmagma
INCS += -I${MAGMADIR}/include
endif

main:  main.F90 zhegst_gpu.F90 cusolverDn_m.cuf toolbox.F90 wallclock.c 
	pgcc -c ${FLAGS} wallclock.c
	pgf90 -c ${FLAGS} toolbox.F90
	pgf90 -c ${FLAGS} cusolverDn_m.cuf
	pgf90 -c ${FLAGS} zhegst_gpu.F90
	pgf90 -o main main.F90 zhegst_gpu.o cusolverDn_m.o toolbox.o wallclock.o ${LIBS} ${FLAGS} -pgf90libs ${INCS}
	pgf90 -o testDn testDn.cuf cusolverDn_m.o ${LIBS} ${FLAGS}
clean:
	rm main testDn *.mod *.o
