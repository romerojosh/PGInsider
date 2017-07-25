# PGInsider
Source files for PGInsider blog post on Cholesky factorization and reduction of generalized eigenproblem

### Building
* Compilation of these files requires the PGI compiler version 17.4 or higher.
* The provided `Makefile` requires definition of `MKL_ROOT`, and `CUDA_HOME` environment varibles, the root directory of 
your Intel MKL installation and CUDA Toolkit installation respectively. 
* The `Makefile` is configured for modern Pascal P100 GPUs. If you are using an older model, modify the compute capability flag (`cc60`) as needed. 
* Optionally, populate `MAGMADIR` variable in the `Makefile` with the root directory of your MAGMA installation to enable
compilation with MAGMA.
* With these settings, compile with `make` command.

### Running
Successful compilation will result in two executables, `testDn` and `main`. 
* `testDn` runs a quick test of the cuSOLVER dense Cholesky factorization:

	$ ./testDn

* `main` runs and compares times of the Cholesky factorizations and eigenproblem reduction for system of size `N`:

	$ ./main N


### License
This code is released under an MIT license which can be found in `LICENSE`. 
