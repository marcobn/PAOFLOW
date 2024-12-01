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
 * reviewer(s): Filippo Spiga (filippo.spiga@ichec.ie)
 *
 */

#ifdef __CUDA

#include <stdlib.h>
#include <stdio.h>

#include <driver_types.h>

#if defined(__TIMELOG)
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#endif

#if defined(__PARA)
#include <mpi.h>
#include <string.h>
#endif

#include "cuda_env.h"

#if defined(__PHIGEMM)
#include "phigemm.h"
#endif

qeCudaMemDevPtr dev_scratch_QE;
qeCudaMemSizes cuda_memory_allocated;
qeCudaMemSizes device_memory_left;
qeCudaDevicesBond qe_gpu_bonded;

// global useful information
int ngpus_detected;
int ngpus_used;
int ngpus_per_process;
int procs_per_gpu;

#if defined(__PARA)
char lNodeName[MPI_MAX_PROCESSOR_NAME];
int lRank;
#else
const char lNodeName[] = "localhost";
#endif

#if defined(__TIMELOG)
double cuda_cclock(void)
{
	struct timeval tv;
	struct timezone tz;
	double t;

	gettimeofday(&tv, &tz);

	t = (double)tv.tv_sec;
	t += ((double)tv.tv_usec)/1000000.0;

	return t;
}
#endif



void gpuserialbinding_(){

	int ierr = 0;
	int lNumDevicesThisNode = 0;
	int i;

	size_t free, total;

	cudaGetDeviceCount(&lNumDevicesThisNode);

	if (lNumDevicesThisNode == 0)
	{
		fprintf( stderr,"***ERROR*** no CUDA-capable devices were found on the machine.\n");
		exit(EXIT_FAILURE);
	}

	ngpus_detected = lNumDevicesThisNode;

	/* multi-GPU in serial calculations is allowed ONLY if CUDA >= 4.0 */
#if defined(__MULTI_GPU) && !defined(__CUDA_3)
	ngpus_used = ngpus_per_process = lNumDevicesThisNode;
#else
	ngpus_used = ngpus_per_process = 1;
#endif

	for (i = 0; i < ngpus_per_process; i++) {
		/* Bond devices
		 * NOTE: qe_gpu_bonded[0] is ALWAYS the main device for non multi-GPU
		 *       kernels.
		 */
		qe_gpu_bonded[i] = i;
	}
}

#if defined(__PARA)
void gpuparallelbinding_() {

	int ierr = 0;
	int lNumDevicesThisNode = 0;
	int i;

	size_t free, total;

	int lSize, sDeviceBoundTo, tmp;
	int lNodeNameLength, sIsCudaCapable, lNumRanksThisNode;
	int lRankThisNode = 0, lSizeThisNode = 0;
	char *lNodeNameRbuf;
	int *lRanksThisNode;

	MPI_Group lWorldGroup;
	MPI_Group lThisNodeGroup;
	MPI_Comm  lThisNodeComm;

	MPI_Comm_rank(MPI_COMM_WORLD, &lRank);
	MPI_Comm_size(MPI_COMM_WORLD, &lSize);

	MPI_Get_processor_name(lNodeName, &lNodeNameLength);

	lNodeNameRbuf = (char*) malloc(lSize * MPI_MAX_PROCESSOR_NAME * sizeof(char));

	MPI_Allgather(lNodeName, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, lNodeNameRbuf, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	// lRanksThisNode is a list of the global ranks running on this node
	lRanksThisNode = (int*) malloc(lSize * sizeof(int));

	for(i=0; i<lSize; i++)
	{
		if(strncmp(lNodeName, (lNodeNameRbuf + i * MPI_MAX_PROCESSOR_NAME), MPI_MAX_PROCESSOR_NAME) == 0)
		{
			lRanksThisNode[lNumRanksThisNode] = i;
			lNumRanksThisNode++;
		}
	}

	/* Create a communicator consisting of the ranks running on this node. */
	MPI_Comm_group(MPI_COMM_WORLD, &lWorldGroup);
	MPI_Group_incl(lWorldGroup, lNumRanksThisNode, lRanksThisNode, &lThisNodeGroup);
	MPI_Comm_create(MPI_COMM_WORLD, lThisNodeGroup, &lThisNodeComm);
	MPI_Comm_rank(lThisNodeComm, &lRankThisNode);
	MPI_Comm_size(lThisNodeComm, &lSizeThisNode);

	/* Attach all MPI processes on this node to the available GPUs
	 * in round-robin fashion
	 */
	cudaGetDeviceCount(&lNumDevicesThisNode);

	if (lNumDevicesThisNode == 0 && lRankThisNode == 0)
	{
		printf("***ERROR: no CUDA-capable devices were found on node %s.\n", lNodeName);
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	ngpus_detected = lNumDevicesThisNode;

	if ( (lSizeThisNode % lNumDevicesThisNode ) != 0  )
	{
		printf("***WARNING: unbalanced configuration (%d MPI per node, %d GPUs per node)\n", lSizeThisNode, lNumDevicesThisNode);
		fflush(stdout);
	}

	if (ngpus_detected <= lSizeThisNode ){
		/* if GPUs are less then (or equal of) the number of  MPI processes on a single node,
		 * then PWscf uses all the GPU and one single GPU is assigned to one or multiple MPI processes with overlapping. */
		ngpus_used = ngpus_detected;
		ngpus_per_process = 1;
	} else {
		/* multi-GPU in parallel calculations is allowed ONLY if CUDA >= 4.0 */
#if defined(__CUDA_3)
		ngpus_used = ngpus_detected;
		ngpus_per_process = 1;
#else
		/* if GPUs are more than the MPI processes on a single node,
		 * then PWscf uses all the GPU and one or more GPUs are assigned
		 * to every single MPI processes without overlapping.
		 * *** NOT IMPLEMENTED YET ***
		 */
		ngpus_used = ngpus_detected;
		ngpus_per_process = 1;
#endif
	}

	procs_per_gpu = (lSizeThisNode < lNumDevicesThisNode) ? lSizeThisNode : lSizeThisNode / lNumDevicesThisNode;

	for (i = 0; i < ngpus_per_process; i++) {

		qe_gpu_bonded[i] = lRankThisNode % lNumDevicesThisNode;

#if defined(__CUDA_DEBUG)
		printf("Binding GPU %d on node %s to rank: %d (internal rank:%d)\n", qe_gpu_bonded[i], lNodeName, lRank, lRankThisNode); fflush(stdout);
#endif

	}

}
#endif

void initphigemm_(){

#if defined(__PHIGEMM)

#if defined(__CUDA_3)

	/* Compatibility with CUDA 3.x (phiGEMM v0.7): no multi-GPU, no smart bonding GPU-processes */
	phiGemmInit((void**)&dev_scratch_QE[0], (size_t*)&cuda_memory_allocated[0]);

#else
	/* Compatibility with CUDA 4.x (latest phiGEMM): */
	phiGemmInit(ngpus_per_process , (qeCudaMemDevPtr*)&dev_scratch_QE, (qeCudaMemSizes*)&cuda_memory_allocated, (int *)qe_gpu_bonded);
#endif

#endif
}


void preallocatedevicememory_(){

	int ierr = 0;
	int i;

	size_t free, total;

	for (i = 0; i < ngpus_per_process; i++) {

		/* query the real free memory, taking into account the "stack" */
		if ( cudaSetDevice(qe_gpu_bonded[i]) != cudaSuccess) {
			printf("*** ERROR *** cudaSetDevice(%d) failed!", qe_gpu_bonded[i] );fflush(stdout);
#if defined(__PARA)
			MPI_Abort( MPI_COMM_WORLD, -1 );
#else
			exit(EXIT_FAILURE);
#endif
		}

		cuda_memory_allocated[i] = (size_t) 0;

		ierr = cudaMalloc ( (void**) &(dev_scratch_QE[i]), cuda_memory_allocated[i] );
		if ( ierr != cudaSuccess) {
			fprintf( stderr, "\nError in (first zero) memory allocation [%s:%d] , program will be terminated!!! Bye...\n\n", lNodeName, qe_gpu_bonded[i]);
#if defined(__PARA)
			MPI_Abort( MPI_COMM_WORLD, -1 );
#else
			exit(EXIT_FAILURE);
#endif
		}

#if defined(__PARA)
	}

	/* is this barrier smart? I guess yes... */
//	MPI_Barrier(lThisNodeComm);

	for (i = 0; i < ngpus_per_process; i++) {
#endif
		cuMemGetInfo(&free, &total);

		// see cuda_env.h for a description of the hack
		// this does *NOT* work if everything is not performed at the bginning...
#if defined(__CUDA_GET_MEM_HACK)
		free = (size_t)  __GPU_MEM_AMOUNT_HACK__;
#else
		cuMemGetInfo(&free, &total);
#endif

#if defined(__CUDA_DEBUG)
#if defined(__PARA)
		printf("[GPU %d - rank: %d] before: %lu (total: %lu)\n", qe_gpu_bonded[i], lRank, (unsigned long)free, (unsigned long)total); fflush(stdout);
#else
		printf("[GPU %d] before: %lu (total: %lu)\n", qe_gpu_bonded[i], (unsigned long)free, (unsigned long)total); fflush(stdout);
#endif
#endif


#if defined(__PARA)
		cuda_memory_allocated[i] = (size_t) (free * __SCALING_MEM_FACTOR__ / procs_per_gpu);
#else
		cuda_memory_allocated[i] = (size_t) (free * __SCALING_MEM_FACTOR__);
#endif

		ierr = cudaMalloc ( (void**) &(dev_scratch_QE[i]), (size_t) cuda_memory_allocated[i] );
		if ( ierr != cudaSuccess) {
			fprintf( stderr, "\nError in memory allocation [%s:%d] , program will be terminated (%d)!!! Bye...\n\n", lNodeName, qe_gpu_bonded[i], ierr );
#if defined(__PARA)
			MPI_Abort( MPI_COMM_WORLD, -1 );
#else
			exit(EXIT_FAILURE);
#endif
		}

#if defined(__CUDA_DEBUG)
		cuMemGetInfo(&free, &total);
#endif

		/* It can be useful to track this information... */
#if defined(__CUDA_GET_MEM_HACK)
		device_memory_left[i] = __GPU_MEM_AMOUNT_HACK__ - cuda_memory_allocated[i];
#else
		device_memory_left[i] = free;
#endif

#if defined(__CUDA_DEBUG)
#if defined(__PARA)
		printf("[GPU %d - rank: %d] after: %lu (total: %lu)\n", qe_gpu_bonded[i], lRank, (unsigned long)free, (unsigned long)total); fflush(stdout);
#else
		printf("[GPU %d] after: %lu (total: %lu)\n", qe_gpu_bonded[i], (unsigned long)free, (unsigned long)total); fflush(stdout);
#endif
#endif
	}

}

void initcudaenv_()
{

#if defined(__PARA)
	gpuparallelbinding_();
#else
	gpuserialbinding_();
#endif

	preallocatedevicememory_();

	initphigemm_();
}

void closecudaenv_()
{
	int ierr = 0;

	ierr = cudaFree ( dev_scratch_QE[0] );

	if(ierr != cudaSuccess) {
		fprintf( stderr, "\nError in memory release, program will be terminated!!! Bye...\n\n" );
#if defined(__PARA)
		MPI_Abort( MPI_COMM_WORLD, -1 );
#else
		exit(EXIT_FAILURE);
#endif
	}

#if defined(__PHIGEMM)
	phiGemmShutdown();
#endif

}

#endif
