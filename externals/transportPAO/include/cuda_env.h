/*
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 * Copyright (C) 2001-2011 Quantum ESPRESSO group
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 * author(s):	Ivan Girotto (ivan.girotto@ichec.ie),
 * 				Filippo Spiga (filippo.spiga@ichec.ie)
 */

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#if defined __CUDA_3
#include <cublas.h>
#else
#include "cublas_api.h"
#include "cublas_v2.h"
#endif


#if defined(__PARA)
#include <mpi.h>
#endif

#ifndef __QE_CUDA_ENVIRONMENT_H
#define __QE_CUDA_ENVIRONMENT_H

#if defined __GPU_NVIDIA_13

#define __CUDA_THREADPERBLOCK__ 256
#define __NUM_FFT_MULTIPLAN__ 4
#define __CUDA_TxB_ADDUSDENS_COMPUTE_AUX__ __CUDA_THREADPERBLOCK__
#define __CUDA_TxB_VLOCPSI_PSIC__ __CUDA_THREADPERBLOCK__
#define __CUDA_TxB_VLOCPSI_PROD__ __CUDA_THREADPERBLOCK__
#define __CUDA_TxB_VLOCPSI_HPSI__ __CUDA_THREADPERBLOCK__

#elif defined __GPU_NVIDIA_20

#define __CUDA_THREADPERBLOCK__ 512
#define __NUM_FFT_MULTIPLAN__ 4
#define __CUDA_TxB_ADDUSDENS_COMPUTE_AUX__ 1024
#define __CUDA_TxB_VLOCPSI_PSIC__ 64
#define __CUDA_TxB_VLOCPSI_PROD__ 128
#define __CUDA_TxB_VLOCPSI_HPSI__ 448

#else

#define __CUDA_THREADPERBLOCK__ 256
#define __NUM_FFT_MULTIPLAN__ 1

#endif

/* Sometimes it is not possible to use 'cuMemGetInfo()' to know the amount
 * of memory on the GPU card. For this reason this macro define a "fixed"
 * amount of memory to use in case this behavior happens. Use carefully
 * and edit the amount (in byte) accordingly to the real amount of memory
 * on the card minus ~500MB. [NdFilippo]
 */
#if defined __CUDA_GET_MEM_HACK
#define __GPU_MEM_AMOUNT_HACK__ 2400000000
#endif

#if defined __MAGMA
#define __SCALING_MEM_FACTOR__ 0.6
#else
#define __SCALING_MEM_FACTOR__ 0.80
#endif

#define MAX_QE_GPUS 8

typedef void* qeCudaMemDevPtr[MAX_QE_GPUS];
typedef size_t qeCudaMemSizes[MAX_QE_GPUS];
typedef int qeCudaDevicesBond[MAX_QE_GPUS];

extern qeCudaMemDevPtr dev_scratch_QE;
extern qeCudaMemSizes cuda_memory_allocated;
extern qeCudaDevicesBond qe_gpu_bonded;

extern int ngpus_detected;
extern int ngpus_used;
extern int ngpus_per_process;

size_t initCudaEnv();
void closeCudaEnv();
void preallocateDeviceMemory();
void initPhigemm();
#if defined __PARA
void gpuparallelbinding();
#endif
void gpuSerialBinding();

/* These routines are exactly the same in "cutil_inline_runtime.h" but,
 * replicating them here, we remove the annoying dependency to CUTIL & SDK (Filippo)
 *
 * We define these calls here, so the user doesn't need to include __FILE__ and __LINE__
 * The advantage is the developers gets to use the inline function so they can debug
 */

#define qecudaSafeCall(err)  __qecudaSafeCall(err, __FILE__, __LINE__)
#define qecudaGetLastError(msg)  __qecudaGetLastError(msg, __FILE__, __LINE__)
#define qecheck_cufft_call(err) __qecheck_cufft_call(err, __FILE__, __LINE__)

inline void __qecudaSafeCall( cudaError_t err, const char *file, const int line )
{
    if( cudaSuccess != err) {
    	printf("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
                file, line, cudaGetErrorString( err) ); fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
    }
}


inline void __qecudaGetLastError(const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
    	printf("%s(%i) : qecudaGetLastError() error : %s : %s.\n",
                file, line, errorMessage, cudaGetErrorString( err) ); fflush(stdout);
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
    }
}


inline void __qecheck_cufft_call(  cufftResult cufft_err, const char *file, const int line )
{

	switch ( cufft_err ) {

	case CUFFT_INVALID_PLAN :
		fprintf( stderr, "\n[%s:%d] The plan parameter is not a valid handle! Program exits... \n", file, line );
		break;

	case CUFFT_INVALID_VALUE :
		fprintf( stderr, "\n[%s:%d] The idata, odata, and/or direction parameter is not valid! Program exits... \n", file, line );
		break;

	case CUFFT_EXEC_FAILED :
		fprintf( stderr, "\n[%s:%d] CUFFT failed to execute the transform on GPU! Program exits... \n", file, line );
		break;

	case CUFFT_SETUP_FAILED :
		fprintf( stderr, "\n[%s:%d] CUFFT library failed to initialize! Program exits... \n", file, line );
		break;

	case CUFFT_INVALID_SIZE :
		fprintf( stderr, "\n[%s:%d] The nx parameter is not a supported size! Program exits... \n", file, line );
		break;

	case CUFFT_INVALID_TYPE :
		fprintf( stderr, "\n[%s:%d] The type parameter is not supported! Program exits... \n", file, line );
		break;

	case CUFFT_ALLOC_FAILED :
		fprintf( stderr, "\n[%s:%d] Allocation of GPU resources for the plan failed! Program exits... \n", file, line );
		break;

	case CUFFT_SUCCESS:
		break;

	default:
		fprintf( stderr, "\n[%s:%d] CUFFT returned not recognized value! %d\n", file, line, cufft_err );
		break;
	}

	if (cufft_err != CUFFT_SUCCESS) {
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
		exit(EXIT_FAILURE);
#endif
	}
}

#endif // __QE_CUDA_ENVIRONMENT_H
