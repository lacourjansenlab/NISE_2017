#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cblas.h>
#include "omp.h"
#include "types.h"
#include "NISE_subs.h"
#include "mcfret.h"
#include "project.h"
#include "propagate.h"
#include "read_trajectory.h"

/* Main MCFRET routine only calling and combining the other subroutines */ 
void mcfret(t_non *non){
    int nn2;
    int segments;
    /* Response functions for emission and absorption: real and imaginary part*/
    float *re_Abs,*im_Abs;
    float *re_Emi,*im_Emi;
    float *re_e,*im_e; /* Eigenvalues */
    float *vl,*vr; /* Left and right eigenvectors */
    float *energy_cor; /* Effective correction energy for QC */
    float *J;
    float *E;
    float *rate_matrix;
    float *coherence_matrix;
    float *ave_vecr;

    /* Allocate memory for the response functions */
    nn2=non->singles*non->singles;
    re_Abs=(float *)calloc(nn2*non->tmax1,sizeof(float));
    im_Abs=(float *)calloc(nn2*non->tmax1,sizeof(float));
    re_Emi=(float *)calloc(nn2*non->tmax1,sizeof(float));
    im_Emi=(float *)calloc(nn2*non->tmax1,sizeof(float));
    J=(float *)calloc(nn2,sizeof(float));
    E=(float *)calloc(non->singles,sizeof(float));
    ave_vecr=(float *)calloc(non->singles*non->singles,sizeof(float));

    /* The rate matrix is determined by the integral over t1 for */
    /* Tr [ J * Abs(t1) * J * Emi(t1) ] */

    segments=project_dim(non);
    if (segments<2){
        printf(RED "Too few segments defined for MCFRET calculation!" RESET);
        exit(0);
    }
    rate_matrix=(float *)calloc(segments*segments,sizeof(float));
    coherence_matrix=(float *)calloc(segments*segments,sizeof(float));

    /* Tell the user that we are in the MCFRET Routine */
    if (string_in_array(non->technique,(char*[]){"MCFRET",
        "MCFRET-Autodetect","MCFRET-Absorption","ECFRET-Emission",
        "MCFRET-Coupling","MCFRET-Rate","MCFRET-Analyse",
        "MCFRET-Density","MCFRET-4th-approx", "MCFRET-4th-full" },10)){
        printf("Performing MCFRET calculation.\n");
    }

    if (!strcmp(non->technique, "MCFRET") || (!strcmp(non->technique, "MCFRET-Density"))){
        /* Calculate the average density matrix */
	printf("Starting calculation of the average density matrix.\n");
        average_density_matrix(ave_vecr,non);
        write_matrix_to_file("Average_Density.dat",ave_vecr,non->singles);
    }

    /* Call the absorption routine */
    if (!strcmp(non->technique, "MCFRET") || (!strcmp(non->technique, "MCFRET-Absorption"))){
	printf("Starting calculation of the MCFRET absorption matrix.\n");
        mcfret_propagation_segmented(re_Abs,im_Abs,non);
    }
   
    // /* Call the emission routine */
    // if (!strcmp(non->technique, "MCFRET") || (!strcmp(non->technique, "MCFRET-Emission"))){
    //     printf("Starting calculation of the MCFRET emission matrix.\n");
    // 	/* Read precalculated average density matrix */ 
    // 	if (!strcmp(non->technique, "MCFRET-Emission")){
    //        printf("Using precalculated average density matrix from file Average_Density.dat.\n");
    //	    read_matrix_from_file("Average_Density.dat",ave_vecr,non->singles);
    // 	}
    //    mcfret_response_function(re_Emi,im_Emi,non,1,ave_vecr);
    // }
    
    /* Call the coupling routine */
    if (!strcmp(non->technique, "MCFRET") || (!strcmp(non->technique, "MCFRET-Coupling"))){
        printf("Starting calculation of the average inter segment coupling.\n");
        mcfret_coupling(J,non);
    }

    /* Call the rate routine routine */
    if (!strcmp(non->technique, "MCFRET") || (!strcmp(non->technique, "MCFRET-Rate"))){
        printf("Starting calculation of the rate response function.\n");
        if ((!strcmp(non->technique, "MCFRET-Rate"))){
            /* Read in absorption, emission and coupling from file if needed */
	    printf("Calculating rate from precalculated absorption and coupling!\n");
	    read_matrix_from_file("CouplingMCFRET.dat",J,non->singles);
	    read_matrix_from_file("Average_Density.dat",ave_vecr,non->singles); 
	    read_response_from_file("TD_absorption_matrix.dat",re_Abs,im_Abs,non->singles,non->tmax1);
	    // read_response_from_file("TD_emission_matrix.dat",re_Emi,im_Emi,non->singles,non->tmax1);
            printf("Completed reading pre-calculated data.\n");
        }
        // mcfret_rate(rate_matrix,coherence_matrix,segments,re_Abs,im_Abs,re_Emi,im_Emi,J,non);
        mcfret_rate_from_abs(rate_matrix,coherence_matrix,segments,re_Abs,im_Abs,ave_vecr,J,non);

        /* Write the calculated ratematrix to file */
        write_matrix_to_file("RateMatrix.dat",rate_matrix,segments);
        /* Write the calculated coherence matrix to file */
        write_matrix_to_file("CoherenceMatrix.dat",coherence_matrix,segments);
    }

    /* Call the 4th order approximation routine */
    if (!strcmp(non->technique, "MCFRET-4th-approx")){
        printf("Starting calculation of the 4th order correction: traces.\n");
	printf("Calculating 4th order from precalculated coupling and density\n");
	
	read_matrix_from_file("CouplingMCFRET.dat",J,non->singles);
	read_matrix_from_file("Average_Density.dat",ave_vecr,non->singles);
        printf("Completed reading pre-calculated data.\n");
        
	// ave_vecr is average density matrix, J is full average intersegment coupling
   	compute_all_traces_4th_order(ave_vecr, J, non); 
        
	printf("Done with computing the 4th order traces.\n");
    }

    /* Call the 4th order approximation routine */
    if (!strcmp(non->technique, "MCFRET-4th-full")){
        printf("Starting calculation of the full 4th order rates.\n");
        printf("Calculating 4th order from precalculated coupling and density\n");

        read_matrix_from_file("CouplingMCFRET.dat",J,non->singles);
        read_matrix_from_file("Average_Density.dat",ave_vecr,non->singles);
        printf("Completed reading pre-calculated data.\n");

        // ave_vecr is average density matrix, J is full average intersegment coupling
        full_4th_order_main(ave_vecr,J,non);

        printf("Done with computing the full 4th order rates.\n");
    }

    /* Call the MCFRET Analyse routine */
    if (!strcmp(non->technique, "MCFRET") || (!strcmp(non->technique, "MCFRET-Analyse"))){    
        printf("Starting analysis of the MCFRET rate.\n");
	/* If analysis is done as post processing first read the rate matrix */
        if ((!strcmp(non->technique, "MCFRET-Analyse"))){
            read_matrix_from_file("RateMatrix.dat",rate_matrix,segments);
        }

	/* Define various arrays */
	re_e=(float *)calloc(segments,sizeof(float));
	im_e=(float *)calloc(segments,sizeof(float));
	vl=(float *)calloc(segments*segments,sizeof(float));
	vr=(float *)calloc(segments*segments,sizeof(float));
	energy_cor=(float *)calloc(segments,sizeof(float));
	/* Find Eigenvalues and vectors */
	mcfret_eigen(non,rate_matrix,re_e,im_e,vl,vr,segments,energy_cor);
        /* Calculate the expectation value of the segment energies */
        mcfret_energy(E,non,segments, ave_vecr,energy_cor);
        /* Analyse the rate matrix */
        mcfret_analyse(E,rate_matrix,non,segments);	
        free(re_e),free(im_e),free(vl),free(vr);	
	free(energy_cor);
    }


    free(re_Abs);
    free(im_Abs);
    free(re_Emi);
    free(im_Emi);
    free(J);
    free(E);
    free(rate_matrix);
    free(coherence_matrix);
    return;
}

/* The routine to compute all traces of chosen matrix products for the 4th order correction to TD-MCFRET */
void compute_all_traces_4th_order(float *rho_0,float *J_full,t_non *non){
    
    /* Preset standard NISE variables*/
    int x; //used in display of propagator unction

    /* Hamiltonian of the whole system - all donors and acceptors included */
    float *Hamil_i_e;
    /* Transition dipoles for coupling on the fly */
    float *mu_xyz;
    float shift1;

    /* Time parameters */
    time_t time_now,time_old,time_0;
    /* Initialize time */
    time(&time_now);
    time(&time_0);
    shift1=(non->max1+non->min1)/2;
    printf("Frequency shift %f.\n",shift1);
    non->shifte=shift1;

    /* File handles */
    FILE *H_traj;
    // FILE *C_traj;
    FILE *mu_traj;
    FILE *log;
    FILE *Cfile;
    FILE *all_traces_file;

    mu_xyz=(float *)calloc(non->singles*3,sizeof(float));
    Hamil_i_e=(float *)calloc((non->singles+1)*non->singles/2,sizeof(float));
    /* Open Trajectory files */
    open_files(non,&H_traj,&mu_traj,&Cfile);

    /* Here we want to call the routine for checking the trajectory files */
    control(non);

    /* Initialize sample numbers */
    int samples, N_samples, N_segments;
    N_samples=determine_samples(non);
    N_segments=project_dim(non);
    log=fopen("NISE.log","a");
    fprintf(log,"Begin sample: %d, End sample: %d.\n",non->begin,non->end);
    fclose(log);


    /* End of the preset*/
    int N_tw;
    N_tw = non->tmax1;

    int N, N_i, N_j;
    int *H_indices_si, *H_indices_sj;

    N = non->singles;
    H_indices_si = (int *)calloc(N,sizeof(int));
    H_indices_sj = (int *)calloc(N,sizeof(int));

    // prepare traces I and II for every pair direction, so 2 * ns * (ns - 1) time dependent traces will be the result of this routine
    // for simplicity 2 * ns * ns columns will be given (2 * ns are zero)
    // a single array is easiest
    float *all_traces;
    all_traces = (float *)calloc(N_tw*2*N_segments*N_segments,sizeof(float));
    
    int N_rows;
    N_rows = 2*N_segments*N_segments;

    float trace_I, trace_II;
    int si, sj;
    int t_ref, tw, tj;
    int idx, row;


    for (si=0;si<N_segments;si++){        
        N_i = find_H_indices_segment(non->psites, H_indices_si, si, non);
	
	/* Vectors representing time dependent states: real and imaginary part */
        float *U_re, *U_im;
	U_re=(float *)calloc(N_i*N_i,sizeof(float));
	U_im=(float *)calloc(N_i*N_i,sizeof(float));
        float *U_re_snap, *U_im_snap;
	U_re_snap=(float *)calloc(N_i*N_i,sizeof(float));
	U_im_snap=(float *)calloc(N_i*N_i,sizeof(float));
        float *work_re_si, *work_im_si;
        work_re_si =(float *)calloc(N_i*N_i,sizeof(float));
        work_im_si =(float *)calloc(N_i*N_i,sizeof(float));
	
        
	/* The segment si hamiltonian in upper triangle format */
        float *Hamiltonian_segment_triu;
	
	Hamiltonian_segment_triu = (float *)calloc((N_i+1)*N_i/2,sizeof(float));

	// preset rho for the case where si is donor
        float *rho_0_si;
        rho_0_si = (float *)calloc(N_i*N_i,sizeof(float));
        isolate_segment_Hamiltonian(rho_0, rho_0_si, H_indices_si,N_i,non);
        
        // for trace I, prep rho J J.T (to be multiplied with UD J J.T UD)
        // allocate memory for the (ns-1) matrices of size n_i * n_i each
        // total space needed is nr of segments * n_i*ni matrix space, so usually quite small
        // starting index for segment j will skip all previous segments (sum_0^(j-1) n_i*n_i)
        // prepare array_I of starting indices for each segment j
        float *rho_ii_JijJji;
        rho_ii_JijJji = (float *)calloc(N_i*N_i*(N_segments),sizeof(float));

        // for trace II, prep J.T rho J (to be multiplied with UA J.T J UA)
        // allocate memory for the (ns-1) matrices of size n_j * n_j each		
        // total size is sum_j n_i*n_i (all j)
        // starting index for segment j will skip all previous segments (sum_0^(j-1) n_j*n_j)
        // prepare array_II of starting indices for each segment j
        float *Jij_rho_jj_Jji;
        Jij_rho_jj_Jji = (float *)calloc(N_i*N_i*(N_segments),sizeof(float));

        // preset JijJji
        float *JijJji;
        JijJji = (float *)calloc(N_i*N_i*(N_segments),sizeof(float));
	
       	// preset UJJU
        float *UJJU_re;
        UJJU_re = (float *)calloc(N_i*N_i,sizeof(float));

        // can put this in its own function
        for (sj=0;sj<N_segments;sj++){
            if (sj != si){
                // this loop serves to prepare coupling and density matrices products and hold them in memory
                // otherwise these have to be calculated at every time step
                // this set of matrices (one for each j) is updated with every i

                N_j = find_H_indices_segment(non->psites, H_indices_sj, sj, non);
                // isolate the density matrix for segment j
                float *rho_0_sj;
                rho_0_sj = (float *)calloc(N_j*N_j,sizeof(float));
                isolate_segment_Hamiltonian(rho_0, rho_0_sj, H_indices_sj,N_j,non);

                // define matrix J_ij left hand side index i, right hand side index j
                // retrieve the appropriate inter-segment J block for segments i and j
                float *Jij;
                Jij = (float *)calloc(N_i*N_j,sizeof(float));
                isolate_coupling_block(J_full, Jij, N_i, N_j, H_indices_si, H_indices_sj, non);

                // Here, segment i has a donor relation to j
                // segment i has initial population in density matrix
                // prepare matrix product I for rate i -> j
                // compute rho_i Jij Jji
                compute_rhoJJ(rho_ii_JijJji, JijJji, Jij, rho_0_si, N_i, N_j, sj);

                // segment i has an acceptor relation to j
                // segment j has initial population in density matrix
                // prepare matrix product II for rate j-> i
                // compute Jij rho_j Jji
                compute_JrhoJ(Jij_rho_jj_Jji, Jij, rho_0_sj, N_i, N_j, sj); 
                free(rho_0_sj);
                free(Jij);
            }
        }
        
	/* Update NISE log file */
	log=fopen("NISE.log","a");
	fprintf(log,"Finished preparing all constant matrices for segment %d\n",si);
	time_now=log_time(time_now,log);
	fclose(log);

        for (samples=non->begin;samples<non->end;samples++){
            t_ref = samples*non->sample;
            // initialize the n_i * n_i propagator as a unit matrix
	    unitmat(U_re,N_i);
            clearvec(U_im,N_i*N_i);
 
	    /* Loop over waiting time */
            for (tw=0;tw<N_tw;tw++){
		tj = t_ref+tw;
                /* Read Hamiltonian */
                read_Hamiltonian(non,Hamil_i_e,H_traj,tj);
		
            	// isolate the segment i with projection routine to obtain smaller matrix
		// printf("N_i= %d\n",N_i);
		isolate_segment_Hamiltonian_triu(Hamil_i_e, Hamiltonian_segment_triu, H_indices_si, N_i, non);

		// printf("Matrix sum U_re %f\n",matrix_sum(U_re,N_i));
	    	// printf("Matrix sum U_im %f\n",matrix_sum(U_im,N_i));
		
		for (sj=0;sj<N_segments;sj++){
		    // printf("Entering si sj loop over the waiting times si sj tw sample = %d %d %d %d\n",si,sj,tw,samples);
		    if (si != sj){
			// most expensive steps:
			// calculate U_i_JijJji_Ui (twice ~ N_i^3)
			// special subroutine that makes use of the fact that final product is hermitian
			
			// only the real part of UJJU is needed for the 4th order correction    
			compute_UJJU(UJJU_re, JijJji + N_i*N_i * sj, U_re, U_im, N_i, sj);

		        // OPTION 1: immediately calculate the trace, not too expensive because of the special function and requires very little memory
		        // Alternative is to keep the time dependent UJJU in memory, but that requires much more memory

			// calculate trace I (make use of special matrix product trace function, ~N_i^2 )
			// here i is the donor segment
			// (use subroutine, input 'rho J J.T' and 'UDJJUD')
	                // printf("Matrix sum UJJU_re %f\n",matrix_sum(UJJU_re,N_i));
			trace_I = matrix_mul_traced(rho_ii_JijJji + N_i*N_i*sj, UJJU_re, N_i);

			//calculate trace II (make use of special matrix product trace function, ~N_i^2)
			// here i is the acceptor segment
			// (use subroutine, input 'J.T rho J' and 'UAJJUA')
			trace_II = matrix_mul_traced(Jij_rho_jj_Jji + N_i*N_i*sj, UJJU_re, N_i);
			
			// update trace I (rate i to j) at tw
			all_traces[(si*2*N_segments + 2*sj+0)*N_tw+tw] += trace_I;
			// update trace II (rate j to i) at tw
			all_traces[(sj*2*N_segments + 2*si+1)*N_tw+tw] += trace_II;
		    }
		// printf("Closing the loop over segments sj\n");
		}/* Closing loop over segment sj*/

             	// Propagate segment i (~N_i^3 process)
		// only here is propagation of the hamiltonian performed
		// propagate_matrix_segments(non,Hamiltonian_segment_triu,U_re,U_im,-1,samples,tw*x, N_i);
	    
	        // try with new, special propagation routine
            	time_evolution_mat_non_sparse(non, Hamiltonian_segment_triu, U_re_snap, U_im_snap, N_i);
            	propagate_snapshot(U_re_snap, U_im_snap, &U_re, &U_im, &work_re_si, &work_im_si, N_i);
	
		
	    }/* Closing loop over waiting time */
	    // printf("Closing the loop over waiting time\n");
            /* Update NISE log file */
	    log=fopen("NISE.log","a");
            fprintf(log,"Finished sample %d for segment %d\n",samples,si);
            time_now=log_time(time_now,log);
            fclose(log);
	}/* Closing loop over samples */
    
	printf("Closing the loop over samples\n");
        free(rho_0_si);
        free(rho_ii_JijJji);
	free(JijJji);
	free(UJJU_re);
	free(U_re);
	free(U_im);
        free(U_re_snap);
	free(U_im_snap);
	free(work_re_si);
	free(work_im_si);
	free(Jij_rho_jj_Jji);	
	free(Hamiltonian_segment_triu);
    }/* Closing loop over segment si */

    // normalise the traces
    for (idx=0;idx<N_rows*N_tw;idx++){
        all_traces[idx] /= N_samples;
    }
    
    // write all traces to file
    all_traces_file = fopen("all_traces_file.dat","w");
    for (tw=0;tw<N_tw;tw++){
	fprintf(all_traces_file,"%f ",tw*non->deltat);
    	for (row=0;row<N_rows;row++){
            fprintf(all_traces_file,"%f ",all_traces[row*N_tw+tw]);
	}
	fprintf(all_traces_file,"\n");
    }
    fclose(all_traces_file);
    free(H_indices_si);
    free(H_indices_sj);
    free(all_traces);
    free(mu_xyz);
    free(Hamil_i_e);
    return;
}

/* For testing purposes */
/* Find the sum of all matrix elements */
float matrix_sum(float *matrix,int N){
    int i,j;
    float sum;
    sum=0;
    for (i=0;i<N;i++){
        for (j=0;j<N;j++){
            sum=sum+matrix[N*i+j];
            // printf("element %f\n",matrix[N*i+j]);
        }
    }
    return sum;
}
/* For testing purposes */


void compute_UJJU(float *UJJU_re, float *JJ, float *U_re, float *U_im, int N_i,int sj){
    int i1, i2, i3;
    float *intermediate_re, *intermediate_im;
    float elem;
    intermediate_re = (float *)calloc(N_i*N_i,sizeof(float));
    intermediate_im = (float *)calloc(N_i*N_i,sizeof(float));

    clearvec(UJJU_re, N_i*N_i);

    // first a full matrix product
#pragma omp parallel for
    for (i1=0;i1<N_i;i1++){
        for (i2=0;i2<N_i;i2++){
            for (i3=0;i3<N_i;i3++){
	        intermediate_re[i1*N_i+i2] += JJ[i1*N_i+i3] * U_re[i3*N_i+i2];
	        intermediate_im[i1*N_i+i2] += JJ[i1*N_i+i3] * U_im[i3*N_i+i2];
	    }
	}	    
    }
    // only compute half the matrix and make a copy for the other half
    // as the resulting matrix is hermitian
#pragma omp parallel for
    for (i1=0;i1<N_i;i1++){
        for (i2=i1;i2<N_i;i2++){
  	    // could include an if-statement, so as not to write the diagonal twice
	    // no if-statement, as not including it will likely be faster (Check N^2 statements or write N floats)
            for (i3=0;i3<N_i;i3++){
		// elem = U_dag_re[i1*N_i+i3] * intermediate_re[i3*N_i+i2] - U_dag_im[i1*N_i+i3] * intermediate_im[i3*N_i+i2];
		// use that U_dag_re[i1*N_i+i3] = U_re[i3*N_i+i1]
		// use that U_dag_im[i1*N_i+i3] = - U_im[i3*N_i+i1]
		
		elem = U_re[i3*N_i+i1] * intermediate_re[i3*N_i+i2] + U_im[i3*N_i+i1] * intermediate_im[i3*N_i+i2];
                UJJU_re[i1*N_i+i2] += elem;
            }
    	    // write hermitian conjugate: real part identical
	    UJJU_re[i2*N_i+i1] = UJJU_re[i1*N_i+i2];
    	}
    }

    free(intermediate_re);
    free(intermediate_im);
    return;
}

void compute_JrhoJ(float *Jij_rho_jj_Jji, float* Jij, float *rho_0_sj, int N_i, int N_j, int sj){
    int j1,j2,j3,i,i1;
    int start_idx;
    start_idx = N_i * N_i * sj;

    float *rhoJ;
    rhoJ = (float *)calloc(N_j*N_i,sizeof(float));

#pragma omp parallel for
    for (j1=0;j1<N_j;j1++){
        for (i=0;i<N_i;i++){
            for (j3=0;j3<N_j;j3++){
            	rhoJ[j1*N_i + i] += rho_0_sj[j1*N_j+j3] * Jij[i*N_j+j3]; // * Jji[j3*N_i+i]
            }
        }
    }
    
#pragma omp parallel for
    for (i=0;i<N_i;i++){
        for (i1=0;i1<N_i;i1++){
            for (j1=0;j1<N_j;j1++){
            	Jij_rho_jj_Jji[start_idx + i*N_i + i1] += Jij[i*N_j+j1] * rhoJ[j1*N_i+i1];
            }
        }
    }

    free(rhoJ);
    return;
}

void compute_rhoJJ(float *rho_ii_JijJji, float *JijJji, float* Jij, float *rho_0_si, int N_i, int N_j, int sj){
    int i1,i2,i3,j;
    int start_idx;
    start_idx = N_i * N_i * sj;

    float *JJii;
    //immediately update the JijJji matrix for the segment pair si <-> sj
    JJii = JijJji + N_i * N_i * sj;
#pragma omp parallel for
    for (i1=0;i1<N_i;i1++){
        for (i2=0;i2<N_i;i2++){
            for (j=0;j<N_j;j++){
            JJii[i1*N_i + i2] += Jij[i1*N_j+j] * Jij[i2*N_j+j]; // * Jji[j*N_i+i2]
	    }
        }
    }
    
#pragma omp parallel for
    for (i1=0;i1<N_i;i1++){
        for (i2=0;i2<N_i;i2++){
            for (i3=0;i3<N_i;i3++){
            rho_ii_JijJji[start_idx + i1*N_i + i2] += rho_0_si[i1*N_i+i3] * JJii[i3*N_i+i2];
            }
        }
    }
    // free(JJii) 
    return;
}

int find_H_indices_segment(int *psites, int *H_indices_si,int si, t_non *non){
    // find the indices in the full system hamiltonian
    // for the specific segment such that this only needs to be done once
    int i, N, n_i;
    n_i = 0;
    N=non->singles;
    clearvec_int(H_indices_si,N);

    // overwrite the input H_index array, for the segment concerned
    for (i=0;i<N;i++){
       if (psites[i]==si){
           H_indices_si[n_i]=i;
	    n_i++;
        }
    }

    // return the number of molecules in this segment for convenient looping
    return n_i;
}

int find_max_segment_size(int *psites, t_non *non){
    // analyse the projection to find the size of the largest segment
    int i, N, n_max;
    int *segment_sizes;
    N=non->singles;
    n_max = 0;
    // this array is longer than necessary in certain cases
    segment_sizes = (int *)calloc(N,sizeof(int));
    for (i=0;i<N;i++){
        segment_sizes[psites[i]] += 1;
    }
    for (i=0;i<N;i++){
        if (segment_sizes[i] > n_max){
	    n_max = segment_sizes[i];
	}
    }

    free(segment_sizes);
    // return the max segment size
    return n_max;
}


void isolate_segment_Hamiltonian_triu(float *Hamiltonian_full_triu, float *Hamiltonian_segment_triu, int *H_indices_si, int N_i, t_non *non){
    int N, H_a, H_b, i, j;
    int  H_triu_full_idx, H_triu_full_idx_inter, H_triu_si_idx;
    H_triu_si_idx = 0;
    H_triu_full_idx = 0;
    N = non->singles;
    clearvec(Hamiltonian_segment_triu,(N_i+1)*N_i/2);
 
    /* make use of the formula k = N(N-1)/2 - (N-i)(N-1-i)/2+j to convert to upper triangle index*/
    for (i=0;i<N_i;i++){
        H_a = H_indices_si[i];
	H_triu_full_idx_inter = N*(N-1)/2 - (N-H_a)*(N-1-H_a)/2; 
        for (j=i;j<N_i;j++){
            H_b = H_indices_si[j];
	    H_triu_full_idx = H_triu_full_idx_inter + H_b;
	    Hamiltonian_segment_triu[H_triu_si_idx] = Hamiltonian_full_triu[H_triu_full_idx];
	    // printf("si idx %d\n",H_triu_si_idx);
	    H_triu_si_idx++;
	    //Hamiltonian_segment[i*N_i+j] = Hamiltonian_full[H_a*N+H_b];
        }
    }
    return;
}

void isolate_segment_Hamiltonian(float *Hamiltonian_full, float *Hamiltonian_segment, int *H_indices_si, int N_i, t_non *non){
    int N, H_a, H_b, i, j;
    N = non->singles;
    clearvec(Hamiltonian_segment,N_i*N_i);
 
    for (i=0;i<N_i;i++){
        H_a = H_indices_si[i];
        for (j=0;j<N_i;j++){
            H_b = H_indices_si[j];
            Hamiltonian_segment[i*N_i+j] = Hamiltonian_full[H_a*N+H_b];
        }
    }
    return;
}

void isolate_coupling_block(float *J_full, float *J_ij, int N_i, int N_j, int *H_indices_si, int *H_indices_sj, t_non *non){
    int N, H_a, H_b, i, j;
    N = non->singles;
    clearvec(J_ij, N_i*N_j);

    for (i=0;i<N_i;i++){
        H_a = H_indices_si[i];
        for (j=0;j<N_j;j++){
            H_b = H_indices_sj[j];
            J_ij[i*N_j+j] = J_full[H_a*N+H_b];
	    // printf("Jij %f\n",J_ij[i*N_j+j]);
        }
    }
    return;
}

float matrix_mul_traced(float *A, float *B, int N_i){
    /* Calculate the trace of the product of two square matricesof size N_i */
    /* Directly computing the trace greatly reduces the number of operations needed */
    /* from N^3 to N^2 */

    int i,i3;
    float the_trace_re;

    the_trace_re = 0;

// #pragma omp parallel for
    for (i=0;i<N_i;i++){
   	for (i3=0;i3<N_i;i3++){
            the_trace_re += A[i*N_i+i3] * B[i3*N_i+i];
	}
    }
    return the_trace_re;
}


float matrix_mul_traced_DA(float *A, float *B, int N_i, int N_i3){
    /* Calculate the trace of the product of two differently sized matrices of dimension N_i* Ni_3 */
    /* Directly computing the trace greatly reduces the number of operations needed */
    /* from N^3 to N^2 */
    // N_i3 is the shared dimension of the matrices A & B
    // N_i is the dimension of the resulting square matrix
    int i,i3;
    float the_trace;

    the_trace = 0;

// #pragma omp parallel for
    for (i=0;i<N_i;i++){
   	for (i3=0;i3<N_i3;i3++){
            the_trace += A[i*N_i3+i3] * B[i3*N_i+i];
	}
    }
    return the_trace;
}


// Set all elements of a vector to zero, here for integers
void clearvec_int(int *a, int N) {
    int i;
    for (i = 0; i < N; i++) a[i] = 0;
}


/* Standard propagation of a single vector */
/* display is t1*x for displaying info at first step, that is when t1 and x are both zero */
/* and we have the first sample */
void propagate_vector_segments(t_non *non,float * Hamil_i_e,float *vecr,float *veci,int sign,int samples,int display, int N_i){
   int elements;
   if (non->propagation==1) propagate_vec_coupling_S_segments(non,Hamil_i_e,vecr,veci,non->ts,sign, N_i);
   if (non->propagation==3) propagate_vec_RK4(non,Hamil_i_e,vecr,veci,non->ts,sign);
   if (non->propagation==0){
      if (non->thres==0 || non->thres>1){
         propagate_vec_DIA(non,Hamil_i_e,vecr,veci,sign);
      } else {
         elements=propagate_vec_DIA_S(non,Hamil_i_e,vecr,veci,sign);
         if (samples==non->begin){
             if (display==0){
                 printf("Sparce matrix efficiency: %f pct.\n",(1-(1.0*elements/(non->singles*non->singles)))*100);
                 printf("Pressent tuncation %f.\n",non->thres/((non->deltat*icm2ifs*twoPi/non->ts)*(non->deltat*icm2ifs*twoPi/non->ts)));
                 printf("Suggested truncation %f.\n",0.001);
             }
         }
      }
   }
   return;
}


/* Standard propagation of a collection of N vectors */
void propagate_matrix_segments(t_non *non,float * Hamil_i_e,float *vecr,float *veci,int sign,int samples,int display, int N_i){
   int elements;
   int N,j;
   N = N_i;
   // N=non->singles;
#pragma omp parallel for
   for (j=0;j<N_i;j++){
        propagate_vector_segments(non,Hamil_i_e,vecr+j*N,veci+j*N,sign,samples,display*j, N_i);
   }
   return;
}



/* Propagate using diagonal vs. coupling sparce algorithm */
void propagate_vec_coupling_S_segments(t_non* non, float* Hamiltonian_i, float* cr, float* ci, int m, int sign, int N_i) {
    float f;
    int index, N;
    float *H1, *H0, *re_U, *im_U;
    int *col, *row;
    float *ocr, *oci;
    int a, b, c;
    float J;
    float cr1, cr2, ci1, ci2;
    float co, si;
    int i, k, kmax;
    int N2;

    // N = non->singles;
    N = N_i;
    N2=(N*(N-1))/2;
    f = non->deltat * icm2ifs * twoPi * sign / m;
    H0 = (float *)malloc(N*sizeof(float));
    H1 = (float *)malloc(N2*sizeof(float));
    col = (int *)malloc(N2*sizeof(int));
    row = (int *)malloc(N2*sizeof(int));
    re_U = (float *)malloc(N*sizeof(float));
    im_U = (float *)malloc(N*sizeof(float));
    ocr = (float *)malloc(N*sizeof(float));
    oci = (float *)malloc(N*sizeof(float));


    /* Build Hamiltonians H0 (diagonal) and H1 (coupling) */
    k = 0;
    for (a = 0; a < N; a++) {
        H0[a] = Hamiltonian_i[Sindex(a, a, N)]; /* Diagonal */
        for (b = a + 1; b < N; b++) {
            index = Sindex(a, b, N);
            if (fabs(Hamiltonian_i[index]) > non->couplingcut) {
                H1[k] = Hamiltonian_i[index];
                col[k] = a, row[k] = b;
                k++;
            }
        }
    }
    kmax = k;

    /* Exponentiate diagonal [U=exp(-i/2h H0 dt)] */
    for (a = 0; a < N; a++) {
        re_U[a] = cos(0.5 * H0[a] * f);
        im_U[a] = -sin(0.5 * H0[a] * f);
    }

    for (i = 0; i < m; i++) {
        /* Multiply on vector first time */
        for (a = 0; a < N; a++) {
            ocr[a] = cr[a] * re_U[a] - ci[a] * im_U[a];
            oci[a] = ci[a] * re_U[a] + cr[a] * im_U[a];
        }

        /* Account for couplings */
        for (k = 0; k < kmax; k++) {
            a = col[k];
            b = row[k];
            J = H1[k];
            J = J * f;
            si = -sin(J);
            co = sqrt(1 - si * si);
            cr1 = co * ocr[a] - si * oci[b];
            ci1 = co * oci[a] + si * ocr[b];
            cr2 = co * ocr[b] - si * oci[a];
            ci2 = co * oci[b] + si * ocr[a];
            ocr[a] = cr1, oci[a] = ci1, ocr[b] = cr2, oci[b] = ci2;
        }

        /* Multiply on vector second time */
        for (a = 0; a < N; a++) {
            cr[a] = ocr[a] * re_U[a] - oci[a] * im_U[a];
            ci[a] = oci[a] * re_U[a] + ocr[a] * im_U[a];
        }
    }

    free(ocr), free(oci), free(re_U), free(im_U), free(H1), free(H0);
    free(col), free(row);
}

// Segmented Hamiltonian propagation
/* Calculate Absorption matrix */
void mcfret_propagation_segmented(float *re_S_1,float *im_S_1,t_non *non){
    /* Define variables and arrays */
    /* Integers */
    int nn2;
    int itime,N_samples;
    int samples;
    int x,ti,tj,i,j;
    int t1;
    int elements;
    int cl,Ncl;
    int N_segments;

    /* Hamiltonian of the whole system - all donors and acceptors included */
    float *Hamil_i_e;
    /* Vectors representing time dependent states: real and imaginary part */
    float *vecr, *veci;
    /* Transition dipoles for coupling on the fly */
    float *mu_xyz;
    float shift1; 

    /* Time parameters */
    time_t time_now,time_old,time_0;
    /* Initialize time */
    time(&time_now);
    time(&time_0);
    shift1=(non->max1+non->min1)/2;
    printf("Frequency shift %f.\n",shift1);
    non->shifte=shift1;
    
    /* File handles */
    FILE *H_traj;
    FILE *C_traj;
    FILE *mu_traj;
    FILE *log;
    FILE *Cfile;
    FILE *absorption_matrix; 

    /* Allocate memory for all the variables */
    mu_xyz=(float *)calloc(non->singles*3,sizeof(float));
    Hamil_i_e=(float *)calloc((non->singles+1)*non->singles/2,sizeof(float));
    /* Open Trajectory files */
    open_files(non,&H_traj,&mu_traj,&Cfile);

    /* Here we want to call the routine for checking the trajectory files */ 
    control(non);

    itime=0;

    /* Initialize sample numbers */
    N_samples=determine_samples(non);
    N_segments=project_dim(non);
    log=fopen("NISE.log","a");
    fprintf(log,"Begin sample: %d, End sample: %d.\n",non->begin,non->end);
    fclose(log);

    /* Read coupling, this is done if the coupling and transition-dipoles are */
    /* time-independent and only one snapshot is stored */
    read_coupling(non,C_traj,mu_traj,Hamil_i_e,mu_xyz);

    clearvec(re_S_1,non->singles*non->singles*non->tmax1);
    
    int N_i, N, si;
    int *H_indices_si;
    N = non->singles;
    H_indices_si = (int *)calloc(N,sizeof(int));
    
    for (si=0;si<N_segments;si++){
            // find the full matrix indices for this segment
	    N_i = find_H_indices_segment(non->psites, H_indices_si, si, non);
	    printf("Segment size: %d\n",N_i); 
	    /* Allocating memory for the real and imaginary part of the wave function that we need to propagate */
        float *U_re, *U_im;
        float *U_re_snap, *U_im_snap;
	    U_re=(float *)calloc(N_i*N_i,sizeof(float));	
	    U_im=(float *)calloc(N_i*N_i,sizeof(float));
	    U_re_snap=(float *)calloc(N_i*N_i,sizeof(float));	
	    U_im_snap=(float *)calloc(N_i*N_i,sizeof(float));
        float *work_re_si, *work_im_si;
        work_re_si =(float *)calloc(N_i*N_i,sizeof(float));
        work_im_si =(float *)calloc(N_i*N_i,sizeof(float));
 
	    /* The segment si hamiltonian in upper triangle format */
            float *Hamiltonian_segment_triu;
	    Hamiltonian_segment_triu = (float *)calloc((N_i+1)*N_i/2,sizeof(float));

            /* Looping over samples: Each sample represents a different starting point on the Hamiltonian trajectory */
	    for (samples=non->begin;samples<non->end;samples++){
		ti=samples*non->sample;
		if (non->cluster!=-1){
		    if (read_cluster(non,ti,&cl,Cfile)!=1){
			    printf("Cluster trajectory file to short, could not fill buffer!!!\n");
			    printf("ITIME %d\n",ti);
			    exit(1);
		    }
		    /* Configuration belong to cluster */ 
		    if (non->cluster==cl){
			    Ncl++;
		    }
		}
		unitmat(U_re,N_i);
		clearvec(U_im,N_i*N_i);
		
		/* Loop over delay */ 
	       for (t1=0;t1<non->tmax1;t1++){
      		   tj=ti+t1;
		   /* Read Hamiltonian */
		   read_Hamiltonian(non,Hamil_i_e,H_traj,tj);
	
	           ///isolate segment Hamiltonian
		   isolate_segment_Hamiltonian_triu(Hamil_i_e, Hamiltonian_segment_triu, H_indices_si, N_i, non);
			
		   /* Update the MCFRET 'absorpion matrix' or propagator */

           // original, working with coupling propagation scheme
		   mcfret_response_function_sub_segments(re_S_1, im_S_1,t1,non,U_re,U_im,H_indices_si, N_i);        
		   // propagate_matrix_segments(non,Hamiltonian_segment_triu,U_re,U_im,-1,samples,tj*x, N_i);

            // try with new, special propagation routine
            time_evolution_mat_non_sparse(non, Hamiltonian_segment_triu, U_re_snap, U_im_snap, N_i);
            propagate_snapshot(U_re_snap, U_im_snap, &U_re, &U_im, &work_re_si, &work_im_si, N_i);
		   /* We are closing the loop over time delays - t1 times */
	       }

	    /* Update NISE log file */ 
	    log=fopen("NISE.log","a");
	    fprintf(log,"Finished sample %d\n",samples);
		  
	    time_now=log_time(time_now,log);
	    fclose(log);
	    }/* Closing the loop over samples */
    	free(U_re);
	    free(U_im);
    	free(U_re_snap);
	    free(U_im_snap);
    	free(work_re_si);
	    free(work_im_si);
        free(Hamiltonian_segment_triu);
    }

    /* The calculation is finished, lets write output */
    log=fopen("NISE.log","a");
    fprintf(log,"Finished Calculating MCFRET segmented propagators!\n");
    fprintf(log,"Writing to file!\n");  
    fclose(log);

    if (non->cluster!=-1){
        printf("Of %d samples %d belonged to cluster %d.\n",samples,Ncl,non->cluster);
        if (samples==0){ /* Avoid dividing by zero */ 
            samples=1;
        }
    }

    /* Normalize response */
    for (t1=0;t1<non->tmax1*non->singles*non->singles;t1++){
        re_S_1[t1]=re_S_1[t1]/samples;
        im_S_1[t1]=im_S_1[t1]/samples;
    }

    fclose(H_traj);
    if (non->cluster!=-1){
        fclose(Cfile);
    }

    /* Save time domain response */
    absorption_matrix=fopen("TD_absorption_matrix.dat","w");
    fprintf(absorption_matrix,"Samples %d\n",samples);
    fprintf(absorption_matrix,"Dimension %d\n",non->singles*non->singles*non->tmax1);
    for (t1=0;t1<non->tmax1;t1++){
        fprintf(absorption_matrix,"%f ",t1*non->deltat);
	    for (i=0;i<non->singles;i++){
	        for (j=0;j<non->singles;j++){
	            fprintf(absorption_matrix,"%e %e ",re_S_1[t1*non->singles*non->singles+i*non->singles+j],im_S_1[t1*non->singles*non->singles+i*non->singles+j]);
	        }
	    }
	    fprintf(absorption_matrix,"\n");
    }
    fclose(absorption_matrix);
    
    /*Free the memory*/
    free(Hamil_i_e);
    free(mu_xyz);
    free(H_indices_si);
}


// full Hamiltonian propagation
/* Calculate Absorption/Emission matrix (depending on emission variable 0/1) */
void mcfret_response_function(float *re_S_1,float *im_S_1,t_non *non,int emission,float *ave_vecr){
    /* Define variables and arrays */
    /* Integers */
    int nn2;
    int itime,N_samples;
    int samples;
    int x,ti,tj,i,j;
    int t1;
    int elements;
    int cl,Ncl;
    int segments;

    /* Hamiltonian of the whole system - all donors and acceptors included */
    float *Hamil_i_e;
    /* Vectors representing time dependent states: real and imaginary part */
    float *vecr, *veci;
    /* Transition dipoles for coupling on the fly */
    float *mu_xyz;
    float shift1; 

    /* Time parameters */
    time_t time_now,time_old,time_0;
    /* Initialize time */
    time(&time_now);
    time(&time_0);
    shift1=(non->max1+non->min1)/2;
    printf("Frequency shift %f.\n",shift1);
    non->shifte=shift1;

    /* File handles */
    FILE *H_traj;
    FILE *C_traj;
    FILE *mu_traj;
    FILE *log;
    FILE *Cfile;
    FILE *absorption_matrix; 

    /* Allocate memory for all the variables */
    /* Allocating memory for the real and imaginary part of the wave function that we need to propagate */
    vecr=(float *)calloc(non->singles*non->singles,sizeof(float));	
    veci=(float *)calloc(non->singles*non->singles,sizeof(float));
    mu_xyz=(float *)calloc(non->singles*3,sizeof(float));
    Hamil_i_e=(float *)calloc((non->singles+1)*non->singles/2,sizeof(float));
    /* Open Trajectory files */
    open_files(non,&H_traj,&mu_traj,&Cfile);

    /* Here we want to call the routine for checking the trajectory files */ 
    control(non);

    itime=0;

    /* Initialize sample numbers */
    N_samples=determine_samples(non);
    segments=project_dim(non);
    log=fopen("NISE.log","a");
    fprintf(log,"Begin sample: %d, End sample: %d.\n",non->begin,non->end);
    fclose(log);

    /* Read coupling, this is done if the coupling and transition-dipoles are */
    /* time-independent and only one snapshot is stored */
    read_coupling(non,C_traj,mu_traj,Hamil_i_e,mu_xyz);

    clearvec(re_S_1,non->singles*non->singles*non->tmax1);
    /* Looping over samples: Each sample represents a different starting point on the Hamiltonian trajectory */
    for (samples=non->begin;samples<non->end;samples++){
        ti=samples*non->sample;
        if (non->cluster!=-1){
            if (read_cluster(non,ti,&cl,Cfile)!=1){
	            printf("Cluster trajectory file to short, could not fill buffer!!!\n");
	            printf("ITIME %d\n",ti);
	            exit(1);
            }
            /* Configuration belong to cluster */ 
            if (non->cluster==cl){
	            Ncl++;
            }
        }
        if (non->cluster==-1 || non->cluster==cl){
            /* Initialize time-evolution operator */
            if (emission==1){
                /* Read Hamiltonian */
                read_Hamiltonian(non,Hamil_i_e,H_traj,ti);
	          
                /* Remove couplings between segments */
                multi_projection_Hamiltonian(Hamil_i_e,non);

                /* Use the provided density matrix as initial state */
                copyvec(ave_vecr,vecr,non->singles*non->singles);
            } else { 
                unitmat(vecr,non->singles);
                /* write_matrix_to_file("Unit.dat",vecr,non->singles); */
            }
            clearvec(veci,non->singles*non->singles);
        
            /* Loop over delay */ 
            for (t1=0;t1<non->tmax1;t1++){
	        tj=ti+t1;
	        /* Read Hamiltonian */
                read_Hamiltonian(non,Hamil_i_e,H_traj,tj);
	          
                /* Remove couplings between segments */
                multi_projection_Hamiltonian(Hamil_i_e,non);
                
                /* Update the MCFRET Response */
                mcfret_response_function_sub(re_S_1, im_S_1,t1,non,vecr,veci);        
                if (emission==0){         
                    propagate_matrix(non,Hamil_i_e,vecr,veci,-1,samples,t1*x);
                } else {
		            propagate_matrix(non,Hamil_i_e,vecr,veci,1,samples,t1*x);
		}	   
            }/* We are closing the loop over time delays - t1 times */
        } /* We are closing the cluster loop */

        /* Update NISE log file */ 
        log=fopen("NISE.log","a");
        fprintf(log,"Finished sample %d\n",samples);
          
        time_now=log_time(time_now,log);
        fclose(log);
    }/* Closing the loop over samples */
    
    /* The calculation is finished, lets write output */
    log=fopen("NISE.log","a");
    if (emission==1){
        fprintf(log,"Finished Calculating Emission Response Matrix!\n");
    } else {
	fprintf(log,"Finished Calculating Absorption Response Matrix!\n");
    }
    fprintf(log,"Writing to file!\n");  
    fclose(log);

    if (non->cluster!=-1){
        printf("Of %d samples %d belonged to cluster %d.\n",samples,Ncl,non->cluster);
        if (samples==0){ /* Avoid dividing by zero */ 
            samples=1;
        }
    }

    /* Normalize response */
    for (t1=0;t1<non->tmax1*non->singles*non->singles;t1++){
        re_S_1[t1]=re_S_1[t1]/samples;
        im_S_1[t1]=im_S_1[t1]/samples;
    }

    fclose(H_traj);
    if (non->cluster!=-1){
        fclose(Cfile);
    }

    /* Save time domain response */
    if (emission==0){ 
        absorption_matrix=fopen("TD_absorption_matrix.dat","w");
    } else {
        absorption_matrix=fopen("TD_emission_matrix.dat","w");
    }
    fprintf(absorption_matrix,"Samples %d\n",samples);
    fprintf(absorption_matrix,"Dimension %d\n",non->singles*non->singles*non->tmax1);
    for (t1=0;t1<non->tmax1;t1++){
        fprintf(absorption_matrix,"%f ",t1*non->deltat);
	    for (i=0;i<non->singles;i++){
	        for (j=0;j<non->singles;j++){
	            fprintf(absorption_matrix,"%e %e ",re_S_1[t1*non->singles*non->singles+i*non->singles+j],im_S_1[t1*non->singles*non->singles+i*non->singles+j]);
	        }
	    }
	    fprintf(absorption_matrix,"\n");
    }
    fclose(absorption_matrix);
    
    /*Free the memory*/
    free(vecr);	
    free(veci);  
    free(Hamil_i_e);
    free(mu_xyz);
}


/* Sub routine for adding up the calculated response in the response function */
void mcfret_response_function_sub_segments(float *re_S_1,float *im_S_1,int t1,t_non *non,float *cr,float *ci, int *H_indices_si,int N_i){
    int i1,i2;
    int N,nn2;
    int Ni,tnn;
    int H_a, H_b, N_ref;
    N=non->singles;
    nn2=N*N;
    tnn=t1*nn2;
    /* Update response matrix */
    for (i1=0;i1<N_i;i1++){
        H_a = H_indices_si[i1];
	N_ref = N * H_a;
        for (i2=0; i2<N_i; i2++){
            H_b = H_indices_si[i2];

            /* We store response function so we can do matrix multiplication */ 
	    re_S_1[tnn+(N_ref+H_b)]+=cr[i1 * N_i + i2];
            im_S_1[tnn+(N_ref+H_b)]+=ci[i1 * N_i + i2];
        }
    }
    return;
}

/* Sub routine for adding up the calculated response in the response function */
void mcfret_response_function_sub(float *re_S_1,float *im_S_1,int t1,t_non *non,float *cr,float *ci){
    int i,k;
    int N,nn2;
    int Ni,tnn;
    N=non->singles;
    nn2=N*N;
    tnn=t1*nn2;

    /* Update response matrix */
    for (i=0;i<N;i++){
        Ni=N*i;
        for (k=0; k<N; k++){
            /* We store response function so we can do matrix multiplication */
            re_S_1[tnn+(Ni+k)]+=cr[Ni+k]; 
            im_S_1[tnn+(Ni+k)]+=ci[Ni+k];
        }
    }
    return;
}

/* Find the average couplings but only between different segments */
void mcfret_coupling(float *J,t_non *non){
    /* Define variables and arrays */
    /* Integers */
    int N;
    int nn2;
    int N_samples;
    int samples;
    int ti,i,j;
    int cl,Ncl;

    /* Hamiltonian of the whole system - all donors and acceptors included */
    float *Hamil_i_e;
    float *mu_xyz;
    float shift1;
    float invsamp;

    /* Time parameters */
    time_t time_now,time_old,time_0;
    /* Initialize time */
    time(&time_now);
    time(&time_0);
    shift1=(non->max1+non->min1)/2;
    printf("Frequency shift %f.\n",shift1);
    non->shifte=shift1;

    /* File handles */
    FILE *H_traj;
    FILE *C_traj;
    FILE *mu_traj;
    /* FILE *absorption_matrix, *emission_matrix,*/
    FILE *log;
    FILE *Cfile;

    mu_xyz=(float *)calloc(non->singles*3,sizeof(float));
    Hamil_i_e=(float *)calloc((non->singles+1)*non->singles/2,sizeof(float));

    /* Open Trajectory files */
    open_files(non,&H_traj,&mu_traj,&Cfile);

    /* Here we want to call the routine for checking the trajectory files */ 
    control(non);

    /* Initialize sample numbers */
    N_samples=determine_samples(non);
    Ncl=0;

    /* Read coupling, this is done if the coupling and transition-dipoles are */
    /* time-independent and only one snapshot is stored */
    read_coupling(non,C_traj,mu_traj,Hamil_i_e,mu_xyz);
    samples=1;


    log=fopen("NISE.log","a");
    fprintf(log,"Begin sample: %d, End sample: %d.\n",non->begin,non->end);
    fclose(log);
    N=non->singles;

    /* Looping over samples: Each sample represents a different starting point on the Hamiltonian trajectory */
    for (samples=non->begin;samples<non->end;samples++){ 
        ti=samples*non->sample;
        if (non->cluster!=-1){
            if (read_cluster(non,ti,&cl,Cfile)!=1){
	            printf("Cluster trajectory file to short, could not fill buffer!!!\n");
	            printf("ITIME %d\n",ti);
	            exit(1);
            }
            /* Configuration belong to cluster */ 
            if (non->cluster==cl){
	            Ncl++;
            }
        }
        if (non->cluster==-1 || non->cluster==cl){
            /* Read Hamiltonian */
            read_Hamiltonian(non,Hamil_i_e,H_traj,ti);
          	    
            /* Extract couplings between segments */
            multi_projection_Coupling(Hamil_i_e,non);
            for (i=0;i<N;i++){
                for (j=i+1;j<N;j++){
                    // J[non->singles*i+j]+=Hamil_i_e[Sindex(i,j,non->singles)];
                    // J[non->singles*j+i]+=Hamil_i_e[Sindex(i,j,non->singles)];
                    J[N*i+j]+=Hamil_i_e[j+i*((N*2)-i-1)/2];
                    J[N*j+i]+=Hamil_i_e[j+i*((N*2)-i-1)/2];
                }
            }
        } /* We are closing the cluster loop */

        /* Update NISE log file */ 
        log=fopen("NISE.log","a");
        fprintf(log,"Finished sample %d\n",samples);
          
        time_now=log_time(time_now,log);
        fclose(log);
    }/* Closing the loop over samples */
    
    /* Divide with total number of samples */
    invsamp=1.0/samples;
    for (i=0;i<N;i++){
        for (j=0;j<N;j++){
            J[N*i+j]=J[N*i+j]*invsamp;
        }
    }
    write_matrix_to_file("CouplingMCFRET.dat",J,non->singles);

    /* The calculation is finished, lets write output */
    log=fopen("NISE.log","a");
    fprintf(log,"Finished Averaging Intersegment Coupling Response!\n");
    fprintf(log,"Writing to file!\n");  
    fclose(log);

    if (non->cluster!=-1){
        printf("Of %d samples %d belonged to cluster %d.\n",samples,Ncl,non->cluster);
        if (samples==0){ /* Avoid dividing by zero */ 
            samples=1;
        }
    }

    fclose(mu_traj),fclose(H_traj);
    if (non->cluster!=-1){
        fclose(Cfile);
    }
    free(Hamil_i_e);
    free(mu_xyz);  
    return;

}


/* Find MCFRET segments using an automatic scheme */
void mcfret_autodetect(t_non *non, float treshold){
    printf("Use the analyse technique for auto detection.\n");
    return;
}


/* Calculate actual rate matrix */
void mcfret_rate_from_abs(float *rate_matrix,float *coherence_matrix,int segments,float *re_Abs,float *im_Abs, float *rho_0,float *J,t_non *non){
    
    int nn2,N;
    int si,sj;
    int i,j,k;
    int *ns; /* Segment dimensions */
    int t1;
    float *rate_response, *rate_response_imag, *abs_rate_response;
    float rate;
    float isimple,is13; /* Variables for integrals */
    float *re_Abs_hermi,*im_Abs_hermi;
    float *re_aux_mat,*im_aux_mat;
    float *re_aux_mat2,*im_aux_mat2;
    float *Zeros;
    float twoPi2;
    float trace_reaux, trace_imaux;
    FILE *ratefile, *ratefile_imag;
    N=non->singles;
    nn2=non->singles*non->singles;
    twoPi2=twoPi*twoPi;

    rate_response=(float *)calloc(non->tmax,sizeof(float));
    rate_response_imag=(float *)calloc(non->tmax,sizeof(float));
    abs_rate_response=(float *)calloc(non->tmax,sizeof(float));
    re_Abs_hermi=(float *)calloc(nn2,sizeof(float));
    im_Abs_hermi=(float *)calloc(nn2,sizeof(float)); 
    re_aux_mat=(float *)calloc(nn2,sizeof(float));
    im_aux_mat=(float *)calloc(nn2,sizeof(float));
    re_aux_mat2=(float *)calloc(nn2,sizeof(float));
    im_aux_mat2=(float *)calloc(nn2,sizeof(float));
    Zeros=(float *)calloc(nn2,sizeof(float));
  
    ratefile=fopen("RateFile.dat","w");
    ratefile_imag=fopen("RateFile_imag.dat","w");
    /* Do one rate at a time - so first we loop over segments */
    /* Tr [ J * Abs(t1) * J * Emi(t1) ] */
    for (si=0;si<segments;si++){
        for (sj=0;sj<segments;sj++){
            /* Exclude rate between same segments */
            if (sj!=si){
                /* Loop over time delay */
                for (t1=0;t1<non->tmax;t1++){    
                    /* compute the hermitian conjugate of the absorption matrix */
                    hermitian_conjugate(re_Abs+nn2*t1,im_Abs+nn2*t1,re_Abs_hermi,im_Abs_hermi,N,N);

                    /* Matrix multiplication - J Abs_hermi */
                    segment_matrix_mul(J,Zeros,re_Abs_hermi,im_Abs_hermi,
                    re_aux_mat,im_aux_mat,non->psites,segments,si,sj,sj,N);

                    /* Matrix multiplication - (J Abs_hermi) rho_0 */
                    segment_matrix_mul(re_aux_mat,im_aux_mat,rho_0,Zeros,
                    re_aux_mat2,im_aux_mat2,non->psites,segments,si,sj,sj,N);


                    // printf("matrix check  %f\n",matrix_sum(J,N));
                    // /* Matrix multiplication - J Emi */
                    // segment_matrix_mul(J,Zeros,re_Emi+nn2*t1,im_Emi+nn2*t1,
                    // re_aux_mat,im_aux_mat,non->psites,segments,si,sj,sj,N);

                    /* Matrix multiplication - Abs (J Abs_hermi rho_0) */
                    segment_matrix_mul(re_Abs+nn2*t1,im_Abs+nn2*t1,re_aux_mat2,im_aux_mat2,
                    re_aux_mat,im_aux_mat,non->psites,segments,si,si,sj,N);
                    /* Matrix multiplication - J  (Abs J Abs_hermi rho_0) */
                    segment_matrix_mul(J,Zeros,re_aux_mat,im_aux_mat,
                    re_aux_mat2,im_aux_mat2,non->psites,segments,sj,si,sj,N);
                    /* Take the trace */
                    trace_reaux=trace_rate(re_aux_mat2,N);
                    trace_imaux=trace_rate(im_aux_mat2,N);


	            rate_response[t1]=trace_reaux*twoPi2;
	            rate_response_imag[t1] = trace_imaux * twoPi2;
		    abs_rate_response[t1]=sqrt(trace_reaux*trace_reaux
                                    +trace_imaux*trace_imaux)*twoPi2;
                    fprintf(ratefile,"%f %f\n",t1*non->deltat,rate_response[t1]);
                    fprintf(ratefile_imag,"%f %f\n",t1*non->deltat,rate_response_imag[t1]);
                }
                /* Update rate matrix */
	        integrate_rate_response(rate_response,non->tmax,&is13,&isimple);
		/* We use the Trapezium, which is most accurate in most cases */
                rate=2*isimple*non->deltat*icm2ifs*icm2ifs*1000;
                rate_matrix[si*segments+sj]=rate;
                rate_matrix[sj*segments+sj]-=rate;
	        /* Calculate the rate of coherence decay in ps-1 */
	        integrate_rate_response(abs_rate_response,non->tmax,&is13,&isimple);
	        coherence_matrix[si*segments+sj]=1000*abs_rate_response[0]/isimple/non->deltat;
            }
        }
    }
    fclose(ratefile);
    fclose(ratefile_imag);

    free(rate_response);
    free(rate_response_imag);
    free(abs_rate_response);
    free(re_Abs_hermi);
    free(im_Abs_hermi);
    free(re_aux_mat);
    free(im_aux_mat);
    free(re_aux_mat2);
    free(im_aux_mat2);
    free(Zeros);
    free(ns);
    return;
}

/* Find Hermitian conjugate of square NxN matrix */
void hermitian_conjugate(float *A_re, float *A_im, float *hermi_re, float *hermi_im, int N1, int N2){
    int a,b;
    clearvec(hermi_re,N1*N2);
    clearvec(hermi_im,N1*N2);

    for (a=0;a<N1;a++){
        for (b=0;b<N2;b++){
            hermi_re[a+b*N1] = A_re[b+a*N2];
            hermi_im[a+b*N1] = -A_im[b+a*N2];
        }
    }
    return;
}

/* Calculate actual rate matrix */
void mcfret_rate(float *rate_matrix,float *coherence_matrix,int segments,float *re_Abs,float *im_Abs,
    float *re_Emi,float *im_Emi,float *J,t_non *non){
    
    int nn2,N;
    int si,sj;
    int i,j,k;
    int *ns; /* Segment dimensions */
    int t1;
    float *rate_response, *rate_response_imag, *abs_rate_response;
    float rate;
    float isimple,is13; /* Variables for integrals */
    float *re_aux_mat,*im_aux_mat;
    float *re_aux_mat2,*im_aux_mat2;
    float *Zeros;
    float twoPi2;
    float trace_reaux, trace_imaux;
    FILE *ratefile, *ratefile_imag;
    N=non->singles;
    nn2=non->singles*non->singles;
    twoPi2=twoPi*twoPi;

    rate_response=(float *)calloc(non->tmax,sizeof(float));
    rate_response_imag=(float *)calloc(non->tmax,sizeof(float));
    abs_rate_response=(float *)calloc(non->tmax,sizeof(float));
    re_aux_mat=(float *)calloc(nn2,sizeof(float));
    im_aux_mat=(float *)calloc(nn2,sizeof(float));
    re_aux_mat2=(float *)calloc(nn2,sizeof(float));
    im_aux_mat2=(float *)calloc(nn2,sizeof(float));
    Zeros=(float *)calloc(nn2,sizeof(float));
  
    ratefile=fopen("RateFile.dat","w");
    ratefile_imag=fopen("RateFile_imag.dat","w");
    /* Do one rate at a time - so first we loop over segments */
    /* Tr [ J * Abs(t1) * J * Emi(t1) ] */
    for (si=0;si<segments;si++){
        for (sj=0;sj<segments;sj++){
            /* Exclude rate between same segments */
            if (sj!=si){
                /* Loop over time delay */
                for (t1=0;t1<non->tmax;t1++){
                    /* Matrix multiplication - J Emi */
                    segment_matrix_mul(J,Zeros,re_Emi+nn2*t1,im_Emi+nn2*t1,
                    re_aux_mat,im_aux_mat,non->psites,segments,si,sj,sj,N);
                    /* Matrix multiplication - Abs (J Emi) */
                    segment_matrix_mul(re_Abs+nn2*t1,im_Abs+nn2*t1,re_aux_mat,im_aux_mat,
                    re_aux_mat2,im_aux_mat2,non->psites,segments,si,si,sj,N);
                    /* Matrix multiplication - J (Abs J Emi) */
                    segment_matrix_mul(J,Zeros,re_aux_mat2,im_aux_mat2,
                    re_aux_mat,im_aux_mat,non->psites,segments,sj,si,sj,N);
                    /* Take the trace */
                    trace_reaux=trace_rate(re_aux_mat,N);
                    trace_imaux=trace_rate(im_aux_mat,N);
                    rate_response[t1]=trace_reaux*twoPi2;
	            rate_response_imag[t1] = trace_imaux * twoPi2;
		    abs_rate_response[t1]=sqrt(trace_reaux*trace_reaux
                                    +trace_imaux*trace_imaux)*twoPi2;
                    fprintf(ratefile,"%f %f\n",t1*non->deltat,rate_response[t1]);
                    fprintf(ratefile_imag,"%f %f\n",t1*non->deltat,rate_response_imag[t1]);
                }
                /* Update rate matrix */
	        integrate_rate_response(rate_response,non->tmax,&is13,&isimple);
		/* We use the Trapezium, which is most accurate in most cases */
                rate=2*isimple*non->deltat*icm2ifs*icm2ifs*1000;
                rate_matrix[si*segments+sj]=rate;
                rate_matrix[sj*segments+sj]-=rate;
	        /* Calculate the rate of coherence decay in ps-1 */
	        integrate_rate_response(abs_rate_response,non->tmax,&is13,&isimple);
	        coherence_matrix[si*segments+sj]=1000*abs_rate_response[0]/isimple/non->deltat;
            }
        }
    }
    fclose(ratefile);
    fclose(ratefile_imag);

    free(rate_response);
    free(rate_response_imag);
    free(abs_rate_response);
    free(re_aux_mat);
    free(im_aux_mat);
    free(re_aux_mat2);
    free(im_aux_mat2);
    free(Zeros);
    free(ns);
    return;
}

/* Check if mcfret rates are in the incoherent limit */
void mcfret_validate(t_non *non);

/* Find Eigenvalues and eigenvectors of rate matrix */
void mcfret_eigen(t_non *non,float *rate_matrix,float *re_e,float *im_e,float *vl,float *vr,int segments,float *energy_cor){
    //char jobvl = 'V';  // Compute left eigenvectors
    //char jobvr = 'V';  // Compute right eigenvectors
    //int lwork = segments * segments;  // Work array size
    //float work[lwork];
    float *rate; /* Rate Matrix to be destroyed */
    int *degen; /* Degeneracies of segments */
    int i;
    int imax;
    float fmax;
    float popnorm;
    FILE *Efile;
    float *ivr,*ivl;

    rate=(float *)calloc(segments*segments,sizeof(float));
    degen=(int *)calloc(segments,sizeof(int));
    ivr=(float *)calloc(segments*segments,sizeof(float));
    ivl=(float *)calloc(segments*segments,sizeof(float));

    copyvec(rate_matrix,rate,segments*segments);

    diagonalize_real_nonsym(rate_matrix,re_e,im_e,vl,ivl,vr,ivr,segments);

    /* Call LAPACK function sgeev to compute eigenvalues and eigenvectors */
    //sgeev_(&jobvl, &jobvr, &segments, rate, &segments, re_e, im_e, vl, &segments, vr, &segments, work, &lwork, &info);
    //free(rate);

    /* Check for imaginary eigenvalues and find the one closest to zero */
    imax=0;
    fmax=re_e[0];
    
    for (i=0;i<segments;i++){
	    /* Weak check */
	    if (fabs(im_e[i])>0.1*fabs(re_e[i])){
            printf(RED "An imaginary rate matrix eigenvalue larger than 10%%\n");
	        printf("of the real value found! Averaging over more relaizations\n");
	        printf("is adviseable. Use rate matrix with caution!\n" RESET);
	        exit(0);

	    /* Hard Check */
	    } else if (fabs(im_e[i])>1e-8) {
            printf(YELLOW "Warning! An imaginary eigenvalue of the rate matrix was found.");
            printf("Check validity. Averaging over more disorder realizations\n");
	        printf("may remove imaginary eigenvalues." RESET);
	    }

	    /* Check if it is larger than the previous ones */
	    if (re_e[i]>fmax){
            fmax=re_e[i];
	        imax=i;
	    }
    }
    
    /* Write eigenvalues to file and find normalization for the */
    /* equilibrium population. */
    popnorm=0;
    Efile=fopen("RateMatrixEigenvalues.dat","w");
    fprintf(Efile,"# - Eigenvalue (real and imaginary part in ps-1) \n");
    for (i=0;i<segments;i++){
        fprintf(Efile,"%d %f %f\n",i,re_e[i],im_e[i]);
	popnorm+=vl[i+segments*imax];
    }
    fclose(Efile);

    /* Find number of degeneracies */
    project_degeneracies(non,degen,segments);

    /* Write Segment Equilibrium Populations to file */
    /* and find effective energy correction to give equal populations */
    Efile=fopen("SegmentPopulation.dat","w");
    fprintf(Efile,"# - Equilibrium Population\n");
    for (i=0;i<segments;i++){
        fprintf(Efile,"%d %f\n",i,vl[i+segments*imax]/popnorm);
	/* Skip adjusting quantum correction in the high-temperature limit */
	if (non->temperature<100000){
	    energy_cor[i]=-non->temperature*k_B*logf(vl[i+segments*imax]/popnorm/degen[i]);
    }
    }
    fclose(Efile);

    write_matrix_to_file("LeftVectorRateMatrix.dat",vl,segments);
    write_matrix_to_file("RightVectorRateMatrix.dat",vr,segments);
    free(degen);
    free(ivr);
    free(ivl);
    return;
}

/* Analyse rate matrix and find thermal correction */
void mcfret_analyse(float *E,float *rate_matrix,t_non *non,int segments){
      float *qc_rate_matrix,*qc;
      /* Thermal correction */
      float *tc_rate_matrix;
      float *partition_functions;
      float partition_function_i, partition_function_j, equilibration_rate;
      float C;
      float column;
      int i,j;
      float kBT=non->temperature*k_B; /* Kelvin to cm-1 */                     
  
      /* Allocate memory for the partition functions and rate matrices */
      qc_rate_matrix=(float *)calloc(segments*segments,sizeof(float));
      qc=(float *)calloc(segments*segments,sizeof(float)); 
      tc_rate_matrix=(float *)calloc(segments*segments,sizeof(float));
      partition_functions = (float *)calloc(segments,sizeof(float));        
      
      /* load the segment ensemble avg partition functions */
      read_vector_from_file("Segment_Partition_Functions.dat",partition_functions,segments); 

      /* Find quantum correction factors */                                    
      for (i=0;i<segments;i++){
          partition_function_i = partition_functions[i];
	  column = 0;
          for (j=0;j<segments;j++){                                            
              if (i!=j){
                  /* Quantum correction factor from D.W. Oxtoby. */
                  /* Annu. Rev. Phys. Chem., 32(1):77–101, (1981).*/       
                  C=2/(1+exp((E[i]-E[j])/kBT));                            
                  qc_rate_matrix[i*segments+j]=rate_matrix[i*segments+j]*C;
                  qc_rate_matrix[j*segments+j]-=rate_matrix[i*segments+j]*C;
                  qc[i*segments+j]=C;
		  /* Partition Function Based Thermal Correction */
		  equilibration_rate = rate_matrix[i*segments+j] + rate_matrix[j*segments+i];
		  partition_function_j = partition_functions[j];
		  /* rate from segment i to segment j */
		  tc_rate_matrix[i+segments*j] = equilibration_rate * partition_function_j / (partition_function_i + partition_function_j);
                  column += tc_rate_matrix[i+segments*j];
	      }       
              else{ 
                  qc[i*segments+j]=0;
              }       
          }
	  /* Ensure population conservation in the TC_rate_matrix */
	  tc_rate_matrix[i+segments*i] = -column;
      }
  
      /* Write the quantum corrected rate matrix. */
      write_matrix_to_file("QC_RateMatrix.dat",qc_rate_matrix,segments);
      /* Write the thermal corrected rate matrix. */
      write_matrix_to_file("TC_RateMatrix.dat",tc_rate_matrix,segments);
      /* Write the applied quantum correction factors. */
      write_matrix_to_file("QC.dat",qc,segments);                              
      free(qc_rate_matrix);
      free(tc_rate_matrix);
      free(qc);
      return;                                                                  
  }

/* Find the energy of each segment */
void mcfret_energy(float *E,t_non *non,int segments, float *ave_vecr,float *energy_cor){
    /* Define variables and arrays */
    /* Integers */
    int nn2;
    int N_samples;
    int samples;
    int ti,i,j;
    int cl,Ncl;

    /* Hamiltonian of the whole system - all donors and acceptors included */
    float *Hamil_i_e;
    float *vecr;
    float *mu_xyz;
    float shift1;

    /* Time parameters */
    time_t time_now,time_old,time_0;
    /* Initialize time */
    time(&time_now);
    time(&time_0);
    shift1=(non->max1+non->min1)/2;
    printf("Frequency shift %f.\n",shift1);
    non->shifte=shift1;
    /* File handles */
    FILE *H_traj;
    FILE *C_traj;
    FILE *mu_traj;
    /* FILE *absorption_matrix, *emission_matrix,*/
    FILE *log;
    FILE *Cfile;
    FILE *Efile;

    mu_xyz=(float *)calloc(non->singles*3,sizeof(float));
    Hamil_i_e=(float *)calloc((non->singles+1)*non->singles/2,sizeof(float));
    vecr=(float *)calloc(non->singles*non->singles,sizeof(float));

    /* Open Trajectory files */
    open_files(non,&H_traj,&mu_traj,&Cfile);

    /* Here we want to call the routine for checking the trajectory files */
    control(non);

    /* Initialize sample numbers */
    N_samples=determine_samples(non);
    Ncl=0;

    /* Read coupling, this is done if the coupling and transition-dipoles are */
    /* time-independent and only one snapshot is stored */
    read_coupling(non,C_traj,mu_traj,Hamil_i_e,mu_xyz);
    samples=1;

    log=fopen("NISE.log","a");
    fprintf(log,"Begin sample: %d, End sample: %d.\n",non->begin,non->end);
    fclose(log);


    /* Looping over samples: Each sample represents a different starting point on the Hamiltonian trajectory */
    for (samples=non->begin;samples<non->end;samples++){
	ti=samples*non->sample;
        if (non->cluster!=-1){
            if (read_cluster(non,ti,&cl,Cfile)!=1){
                printf("Cluster trajectory file to short, could not fill buffer!!!\n");
                printf("ITIME %d\n",ti);
                exit(1);
            }
            /* Configuration belong to cluster */ 
            if (non->cluster==cl){
                Ncl++;
            }
        }
        if (non->cluster==-1 || non->cluster==cl){
            /* Read Hamiltonian */
            read_Hamiltonian(non,Hamil_i_e,H_traj,ti);
    
            /* Remove couplings between segments */
            multi_projection_Hamiltonian(Hamil_i_e,non);	    
            /* Find density matrix */
            copyvec(ave_vecr,vecr,non->singles*non->singles);
	    /* H * rho */
            triangular_on_square(Hamil_i_e,vecr,non->singles); 
	    /* Add energy contribution for each segment */
	    /* that is take the trace for each segment */
	    for (i=0;i<non->singles;i++){
	        E[non->psites[i]]+=vecr[i*non->singles+i];    
            }
	    
	    clearvec(vecr,non->singles*non->singles);
        } /* We are closing the cluster loop */

        /* Update NISE log file */
        log=fopen("NISE.log","a");
        fprintf(log,"Segement Energy Finished sample %d\n",samples);

        time_now=log_time(time_now,log);
        fclose(log);
    }/* Closing the loop over samples */

    /* Divide with total number of samples */
    for (i=0;i<segments;i++){
        E[i]=E[i]/N_samples;
    }
    Efile=fopen("SegmentEnergies.dat","w");
    fprintf(Efile,"# Segment number - Average segment energy - Energy correction %d\n",N_samples);
    for (i=0;i<segments;i++){
        fprintf(Efile,"%d %f %f\n",i,E[i]+shift1,energy_cor[i]);
	E[i]=E[i]-energy_cor[i];
    }
    fclose(Efile);

    /* The calculation is finished, lets write output */
    log=fopen("NISE.log","a");
    fprintf(log,"Finished Calculating Segment Energies!\n");
    fprintf(log,"Writing to file!\n");
    fclose(log);

    if (non->cluster!=-1){
        printf("Of %d samples %d belonged to cluster %d.\n",samples,Ncl,non->cluster);
        if (samples==0){ /* Avoid dividing by zero */
            samples=1;
        }
    }
    fclose(mu_traj),fclose(H_traj);
    if (non->cluster!=-1){
        fclose(Cfile);
    }
    free(Hamil_i_e);
    free(mu_xyz);
    free(vecr);
    return;
}

/* This function will create a density matrix where every term is weighted with a Boltzmann weight */
void density_matrix(float *density_matrix, float *Hamiltonian_i,t_non *non,int segments, float *partition_functions){
    int index,N;
    float *H,*e;
    double *c2;
    double *cnr;
    double *matrix;

    N=non->singles;
    H=(float *)calloc(N*N,sizeof(float));
    e=(float *)calloc(N,sizeof(float));
    c2=(double *)calloc(N,sizeof(double));
    cnr=(double *)calloc(N*N,sizeof(double));
    matrix=(double *)calloc(N*N,sizeof(double));

    int a,b,c,s;
    double kBT=(double) non->temperature*k_B; /* Kelvin to cm-1 */
    double *Q,iQ;
 
    clearvec(density_matrix,N*N);

    Q=(double *)calloc(segments,sizeof(double));  

    /* Build Hamiltonian */
    for (a=0;a<N;a++){
        H[a+N*a]=Hamiltonian_i[a+N*a-(a*(a+1))/2]; /* Diagonal */
        for (b=a+1;b<N;b++){
            H[a+N*b]=Hamiltonian_i[b+N*a-(a*(a+1))/2];
            H[b+N*a]=Hamiltonian_i[b+N*a-(a*(a+1))/2];
        }
    }
    /* Find eigenvalues and eigenvectors */
    diagonalizeLPD(H,e,N); 
 
    /* Exponentiate [U=exp(-H/kBT)] */
    for (a=0;a<N;a++){
        if (non->temperature==0){
            printf("Temperature is 0, the equilirbium density matrix will be nan,we suggestion to use a low non-zero temperature instead");
            exit(0);
        }

        c2[a]=exp(-((double) (e[a]-e[N-1]))/kBT);
        /* Apply strict high temperature limit when T>100000 */
        if (non->temperature>100000){
	        c2[a]=1.0;
        }
    }

    /* Transform back to site basis */ 
    for (a=0;a<N;a++){
        for (b=0;b<N;b++){
            cnr[b+a*N]+=((double) H[b+a*N])*c2[b];
        }
    }  

// #pragma omp parallel for
    for (a=0;a<N;a++){
        for (b=0;b<N;b++){
            for (c=0;c<N;c++){
                matrix[a+c*N]+=H[b+a*N]*cnr[b+c*N];
            }
        }
    }
  
    /* Find the partition function for each segment */
    for (a=0;a<N;a++){
        Q[non->psites[a]]+=matrix[a+a*N];
    }
    /* Re-normalize */
    for (a=0;a<N;a++){
        for (b=0;b<N;b++){
    	    density_matrix[a+b*N]=(float) (matrix[a+b*N]/Q[non->psites[a]]);
        }
    }      

    /* Update the ensemble average partition function for each segment*/
    for(s=0;s<segments;s++){
       partition_functions[s] += Q[s];
    }

    free(H);
    free(c2);
    free(e);
    free(cnr);
    free(Q);
    return;
}

void average_density_matrix(float *ave_den_mat,t_non *non){
/* Define variables and arrays */
   /* Integers */
    int ti;
    int segments,s;
    int samples;
    int ele;
    int my_samples;
    int N,a,b;
    float i_samples;
    /* Vectors representing time dependent states: real and imaginary part */
    float *vecr;
    float *Hamiltonian_i;
    /* Vector representing the ensemble average partition function for each segment */
    float *partition_functions;
    /* File handles */
    FILE *H_traj;
    FILE *mu_traj;
    FILE *Cfile;
    FILE *avg_partition_functions;
    /* Open Trajectory files */
    open_files(non,&H_traj,&mu_traj,&Cfile);

    /* Allocating memory for the real and imaginary part of the wave function that we need to propagate */
    vecr=(float *)calloc(non->singles*non->singles,sizeof(float));
    Hamiltonian_i=(float *)calloc(non->singles*(non->singles+1)/2,sizeof(float));
    //ave_den_mat=(float *)calloc(non->singles*non->singles,sizeof(float));
    /* Initialize sample numbers */
    segments=project_dim(non);
    N=non->singles;
  
    /* Allocate memory for the partition function vector */
    partition_functions = (float *)calloc(segments,sizeof(float));
    
    clearvec(ave_den_mat,N*N);
    /* Initialize sample numbers */
    my_samples=determine_samples(non);

    if (non->end-non->begin<my_samples){
      my_samples=non->end-non->begin;
    }
// #pragma omp parallel for
    for (samples=non->begin;samples<non->end;samples++){
      ti=samples*non->sample; 
      read_Hamiltonian(non,Hamiltonian_i,H_traj,ti);
      /* Use the thermal equilibrium as initial state */
      density_matrix(vecr,Hamiltonian_i,non,segments,partition_functions);

      for (ele=0; ele<non->singles*non->singles; ele++){
          ave_den_mat[ele] +=vecr[ele]; 
      }
    }
    /* Zero the coupling between different segments for the averaged density *
     * matrix and normalize */
    i_samples=1.0/my_samples;
    for (a=0;a<non->singles;a++){
        for (b=0;b<non->singles;b++){
	    ave_den_mat[non->singles*b+a]*=i_samples;
            if (non->psites[a] != non->psites[b]){
               ave_den_mat[non->singles*a+b]=0.0;
               /* ave_den_mat[non->singles*b+a]=0.0; */
            } 
        }
    }

    /* Normalise the segments' partition funcions */
    for (s=0;s<segments;s++){
        partition_functions[s] *= i_samples;
    }

    /* Write partition function vector to a file */
    avg_partition_functions = fopen("Segment_Partition_Functions.dat","w");
    for (s=0;s<segments;s++){
        fprintf(avg_partition_functions,"%f\n",partition_functions[s]);
    }
    fclose(avg_partition_functions);

    free(partition_functions);
    free(vecr); 
    free(Hamiltonian_i); 
    return;
}

/* Matrix multiplication for different segments */
void segment_matrix_mul(float *rA,float *iA,float *rB,float *iB,
    float *rC,float *iC,int *psites,int segments,int si,int sk,int sj,int N){
    int i,j,k;
    /* Set initial values of results matrix to zero to be sure */
    clearvec(rC,N*N);
    clearvec(iC,N*N);

    int Npsj, Npsk;
    int cj, ck;
    Npsj=0;
    Npsk=0;

    float *psj, *psk;
    psj=(float *)calloc(N,sizeof(float));
    psk=(float *)calloc(N,sizeof(float));

    for (i=0;i<N;i++){
        if (psites[i]==sj){
            psj[Npsj]=i;
            Npsj++;
        }
        if (psites[i]==sk){
            psk[Npsk]=i;
            Npsk++;
        }
    }

#pragma omp parallel for
    for (i=0;i<N;i++){
        if (psites[i]==si){
            for (cj=0;cj<Npsj;cj++){
                j=psj[cj];
                for (ck=0;ck<Npsk;ck++){
                    k=psk[ck];
                    rC[i*N+j]+=rA[i*N+k]*rB[k*N+j]-iA[i*N+k]*iB[k*N+j];
                    iC[i*N+j]+=rA[i*N+k]*iB[k*N+j]+iA[i*N+k]*rB[k*N+j];
                }
            }
        }
    }

    /*
    for (i=0;i<N;i++){
        if (psites[i]==si){
            for (j=0;j<N;j++){
                if (psites[j]==sj){
                    for (k=0;k<N;k++){
                        if (psites[k]==sk){
                            rC[i*N+j]+=rA[i*N+k]*rB[k*N+j]-iA[i*N+k]*iB[k*N+j];
                            iC[i*N+j]+=rA[i*N+k]*iB[k*N+j]+iA[i*N+k]*rB[k*N+j];
                        } 
                    } 
                }
            }
        }
    }
    */
    free(psj);
    free(psk);
    return;
} 

/* Find the trace of the matrix */
float trace_rate(float *matrix,int N){
    int i;
    float trace;
    trace=0;
    for (i=0;i<N;i++){
        trace=trace+matrix[N*i+i];
    }
    return trace;
}

/* Read the absorption/emission function from file */
void read_response_from_file(char fname[],float *re_R,float *im_R,int N,int tmax){
    FILE *file_handle;
    int i,j;
    int t1;
    int dummy;
    float dummyf;
    file_handle=fopen(fname,"r");
    if (file_handle == NULL) {
        printf("Error opening the file %s.\n",fname);
        exit(0);
    }
  
    /* Read initial info */
    fscanf(file_handle, "Samples %d\n", &dummy);
    fscanf(file_handle, "Dimension %d\n", &dummy);
  
    for (t1=0;t1<tmax;t1++){
        /* Read time */
        fscanf(file_handle,"%f",&dummyf);
        for (i=0;i<N;i++){
            for (j=0;j<N;j++){
	            fscanf(file_handle,"%f %f",&re_R[t1*N*N+i*N+j],&im_R[t1*N*N+i*N+j]);
            }
        }
    }
    fclose(file_handle);
}

/* Multiply a triangular matrix on a square one and return */
/* The result in the square matrix */
/* "S=T*S" */
void triangular_on_square(float *T,float *S,int N){
    float *inter;
    int a,c,b;
    int index;
    inter=(float *)calloc(N*N,sizeof(float));
    /* Do matrix multiplication */
    for (a=0;a<N;a++){
        for (b=0;b<N;b++){
            for (c=0;c<N;c++){
                index=Sindex(a,b,N);
                inter[a+c*N]+=T[index]*S[b+c*N]; // TLC b -> c
            }       
        }
    }
    /* Copy result back */
    for (a=0;a<N;a++){
        for (b=0;b<N;b++){
            S[a+b*N]=inter[a+b*N];
        }
    }
    free(inter);
    return;
}

void write_propagator_to_big_array(float *big_array, float *propagator, int sample_length, int si, int N_site_si, int largest_segment_size, int ti){
    // This storing of the propagator is used for both the real and imaginator parts of the propagators
    // dim1 = N_segments;
    size_t dim2 = sample_length;
    size_t dim3 = largest_segment_size;
    size_t dim4 = largest_segment_size;
    int k,l;
    size_t i = si;
    size_t j = ti;
    for (k=0;k<N_site_si;k++){
        for (l=0;l<N_site_si;l++){
            big_array[i * (dim2 * dim3 * dim4) + j * (dim3 * dim4) + k * (dim4) + l] = propagator[k * N_site_si + l];
	}
    }
    return;
}

void read_propagator_from_big_array(float *big_array, float *propagator, int sample_length, int si, int N_site_si, int largest_segment_size, int ti){
    // This storing of the propagator is used for both the real and imaginator parts of the propagators
    // dim1 = N_segments;
    size_t dim2 = sample_length;
    size_t dim3 = largest_segment_size;
    size_t dim4 = largest_segment_size;
    int k,l;
    size_t i = si;
    size_t j = ti;
    for (k=0;k<N_site_si;k++){
        for (l=0;l<N_site_si;l++){
            propagator[k * N_site_si + l] = big_array[i * (dim2 * dim3 * dim4) + j * (dim3 * dim4) + k * (dim4) + l];
        }
    }

    return;
}

void propagate_snapshot(float *U_snap_re, float *U_snap_im, float **U_comp_re, float **U_comp_im, float **temp_re, float **temp_im, int N){
    // multiply the propagator with the next snapshot
    // requires precalculated snapshots
    // overwrites the original propagator
    int i,j;
    complex_matrix_product(U_snap_re, U_snap_im, *U_comp_re, *U_comp_im, *temp_re, *temp_im, N,N,N);

    // swap pointers of the temp and comp matrices
    // this ensures proper pointing to the updated propagator after each iteration
    float *tmp;
    tmp = *U_comp_re; *U_comp_re = *temp_re; *temp_re = tmp;
    tmp = *U_comp_im; *U_comp_im = *temp_im; *temp_im = tmp;
}

// void complex_matrix_product(float *A_re, float *A_im, float *B_re, float *B_im, float *C_re, float *C_im,int N_1,int N_2,int N_3){

//     int i1, i2, i3;
//     float Aim_i1i3, Are_i1i3;
//     clearvec(C_re,N_1*N_2);
//     clearvec(C_im,N_1*N_2);

// // can make paralllel but check for loop order: warnings recieved
// #pragma omp parallel for private(i2,i3,Aim_i1i3,Are_i1i3)
// for (i1=0;i1<N_1;i1++){
//     for (i3=0;i3<N_3;i3++){
// 	    Aim_i1i3 = A_im[i1*N_3+i3];
// 	    Are_i1i3 = A_re[i1*N_3+i3];
//             for (i2=0;i2<N_2;i2++){
//                     C_re[i1*N_2+i2] += Are_i1i3 * B_re[i3*N_2+i2] - Aim_i1i3 * B_im[i3*N_2+i2];
//                     C_im[i1*N_2+i2] += Are_i1i3 * B_im[i3*N_2+i2] + Aim_i1i3 * B_re[i3*N_2+i2];
//                 }
//             }
//         }
// }


void complex_matrix_product(float *A_re, float *A_im,
                            float *B_re, float *B_im,
                            float *C_re, float *C_im,
                            int N_1, int N_2, int N_3)
{
    // Row-major matrices:
    // A: N_1 x N_3
    // B: N_3 x N_2
    // C: N_1 x N_2

    const int sizeA = N_1 * N_3;
    const int sizeB = N_3 * N_2;
    const int sizeC = N_1 * N_2;

    // Allocate interleaved complex buffers
    float *A = (float*) malloc(sizeof(float) * 2 * sizeA);
    float *B = (float*) malloc(sizeof(float) * 2 * sizeB);
    float *C = (float*) malloc(sizeof(float) * 2 * sizeC);

    if (!A || !B || !C) {
        free(A);
        free(B);
        free(C);
        return; // allocation failed
    }

    // Convert A to interleaved complex
    for (int i = 0; i < sizeA; i++) {
        A[2*i]     = A_re[i];
        A[2*i + 1] = A_im[i];
    }

    // Convert B
    for (int i = 0; i < sizeB; i++) {
        B[2*i]     = B_re[i];
        B[2*i + 1] = B_im[i];
    }

    // Zero C
    memset(C, 0, sizeof(float) * 2 * sizeC);

    const float alpha[2] = {1.0f, 0.0f};
    const float beta[2]  = {0.0f, 0.0f};

    cblas_cgemm(CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                N_1, N_2, N_3,
                alpha,
                A, N_3,
                B, N_2,
                beta,
                C, N_2);

    // Convert result back to split format
    for (int i = 0; i < sizeC; i++) {
        C_re[i] = C[2*i];
        C_im[i] = C[2*i + 1];
    }

    free(A);
    free(B);
    free(C);
}

void compute_UDh_rho_J_UA_t1(float *UDh_rho_J_UA_re_t1, 
                             float *UDh_rho_J_UA_im_t1, 
                             float *UAh_Jh_rho_UD_re_t1, 
                             float *UAh_Jh_rho_UD_im_t1, 
                             fourth_order_params *p) 
{

   // compute this matrix product outside the large 3d time loop and reuse it there for efficiency
    float *rho0_D = p->rho0_D;
    float *J      = p->J;
    float *U_re_t1_array = p->U_re_t1_array;
    float *U_im_t1_array = p->U_im_t1_array;
    
    int N_A      = p->N_A;
    int N_D      = p->N_D;
    int s_A      = p->s_A;
    int s_D      = p->s_D;
    int N_t1     = p->N_t1;
    int largest_segment_size = p->largest_segment_size;

    int i1, i2, i3, t1;
    float *rho_J, *intermediate_re, *intermediate_im;
    float elem;
    float temp;
    rho_J = (float *)calloc(N_D*N_A,sizeof(float));
    intermediate_re = (float *)calloc(N_D*N_A,sizeof(float));
    intermediate_im = (float *)calloc(N_D*N_A,sizeof(float));
    float *UD_re, *UD_im;
    float *UD_h_re, *UD_h_im;
    float *UA_re, *UA_im;
    UD_re=(float *)calloc(N_D*N_D,sizeof(float));
    UD_im=(float *)calloc(N_D*N_D,sizeof(float));
    UD_h_re=(float *)calloc(N_D*N_D,sizeof(float));
    UD_h_im=(float *)calloc(N_D*N_D,sizeof(float));  
    UA_re=(float *)calloc(N_A*N_A,sizeof(float));
    UA_im=(float *)calloc(N_A*N_A,sizeof(float));    

    // first compute the constant rho_J
#pragma omp parallel for
    for (i1=0;i1<N_D;i1++){
        for (i3=0;i3<N_D;i3++){
            temp = rho0_D[i1*N_D+i3];
	    for (i2=0;i2<N_A;i2++){
                rho_J[i1*N_A+i2] += temp * J[i3*N_A+i2];
            }
        }
    }

    // loop over the first coherence interval
    for (t1=0;t1<N_t1;t1++){
        // read the precalculated t1 unitary propagators of the Donor and Acceptor 
	    read_propagator_from_big_array(U_re_t1_array, UD_re, N_t1, s_D, N_D, largest_segment_size, t1);
	    read_propagator_from_big_array(U_im_t1_array, UD_im, N_t1, s_D, N_D, largest_segment_size, t1);
        read_propagator_from_big_array(U_re_t1_array, UA_re, N_t1, s_A, N_A, largest_segment_size, t1); 
        read_propagator_from_big_array(U_im_t1_array, UA_im, N_t1, s_A, N_A, largest_segment_size, t1); 
        hermitian_conjugate(UD_re, UD_im, UD_h_re, UD_h_im, N_D, N_D);
        
        clearvec(intermediate_re, N_A * N_D);
        clearvec(intermediate_im, N_A * N_D);
            // compute UDh_rho_J at this timestep
        #pragma omp parallel for
                for (i2=0;i2<N_A;i2++){	 
                    for (i3=0;i3<N_D;i3++){
                        temp = rho_J[i3*N_A+i2];
                        for (i1=0;i1<N_D;i1++){
                            intermediate_re[i1*N_A+i2] += UD_h_re[i1*N_D+i3] * temp;
                            intermediate_im[i1*N_A+i2] += UD_h_im[i1*N_D+i3] * temp;
                }
                    }
            }

        #pragma omp parallel for
                for (i1=0;i1<N_D;i1++){
                    for (i2=0;i2<N_A;i2++){
                        for (i3=0;i3<N_A;i3++){
                            UDh_rho_J_UA_re_t1[t1*N_A*N_D + i1*N_A+i2] += intermediate_re[i1*N_A+i3] * UA_re[i3*N_A+i2] - intermediate_im[i1*N_A+i3] * UA_im[i3*N_A+i2];
                            UDh_rho_J_UA_im_t1[t1*N_A*N_D + i1*N_A+i2] += intermediate_re[i1*N_A+i3] * UA_im[i3*N_A+i2] + intermediate_im[i1*N_A+i3] * UA_re[i3*N_A+i2];
                }
                // the hermitian conjugate of this matrix is also used later and is stored separately to keep the triple time loop clean
                UAh_Jh_rho_UD_re_t1[t1*N_A*N_D + i1+i2*N_D] = UDh_rho_J_UA_re_t1[t1*N_A*N_D + i1*N_A+i2];
                UAh_Jh_rho_UD_im_t1[t1*N_A*N_D + i1+i2*N_D] = -UDh_rho_J_UA_im_t1[t1*N_A*N_D + i1*N_A+i2];	
                    }
                }
    }

    free(UA_re);
    free(UA_im);
    free(UD_re);
    free(UD_im);
    free(rho_J);
    free(intermediate_re);
    free(intermediate_im);
}

void compute_4_intermediate_products(fourth_order_workspace *ws, fourth_order_params *p) {
    // calculate the intermediate products for this specific tw for all matrices for each t1
    // this saves matrix multiplications in the most nested 3d time loop.
    // intermediate_product_1 = [UD_tw @ Ui @ Jh_UDh_tw @ J for Ui in UDh_rho_J_UA_t1]
    // intermediate_product_2 = [Jh_UD_tw @ Ui @ Jh_UDh_tw for Ui in UDh_rho_J_UA_t1]
    // intermediate_product_3 = [UA_Jh_tw @ Ui @ UAh_Jh_tw for Ui in UDh_rho_J_UA_t1]
    // intermediate_product_4 = [UA_tw @ Ui @ J_UAh_Jh_tw for Ui in UAh_Jh_rho_UD_t1]

    // Outputs
    float *intermediate_product_1_re = ws->intermediate_product_1_re;
    float *intermediate_product_1_im = ws->intermediate_product_1_im;
    float *intermediate_product_2_re = ws->intermediate_product_2_re;
    float *intermediate_product_2_im = ws->intermediate_product_2_im;
    float *intermediate_product_3_re = ws->intermediate_product_3_re;
    float *intermediate_product_3_im = ws->intermediate_product_3_im;
    float *intermediate_product_4_re = ws->intermediate_product_4_re;
    float *intermediate_product_4_im = ws->intermediate_product_4_im;

    // Inputs (t1 fase)
    float *UDh_rho_J_UA_re_t1  = ws->UDh_rho_J_UA_re_t1;
    float *UDh_rho_J_UA_im_t1  = ws->UDh_rho_J_UA_im_t1;
    float *UAh_Jh_rho_UD_re_t1 = ws->UAh_Jh_rho_UD_re_t1;
    float *UAh_Jh_rho_UD_im_t1 = ws->UAh_Jh_rho_UD_im_t1;

    // Inputs (tw phase)
    float *Jh_UDh_tw_re  = ws->Jh_UDh_tw_re;
    float *Jh_UDh_tw_im  = ws->Jh_UDh_tw_im;
    float *Jh_UD_tw_re   = ws->Jh_UD_tw_re;
    float *Jh_UD_tw_im   = ws->Jh_UD_tw_im;
    float *UD_tw_re      = ws->UD_tw_re;
    float *UD_tw_im      = ws->UD_tw_im;
    float *UA_Jh_tw_re   = ws->UA_Jh_tw_re;
    float *UA_Jh_tw_im   = ws->UA_Jh_tw_im;
    float *UAh_Jh_tw_re  = ws->UAh_Jh_tw_re;
    float *UAh_Jh_tw_im  = ws->UAh_Jh_tw_im;
    float *J_UAh_Jh_tw_re = ws->J_UAh_Jh_tw_re;
    float *J_UAh_Jh_tw_im = ws->J_UAh_Jh_tw_im;
    float *UA_tw_re      = ws->UA_tw_re;
    float *UA_tw_im      = ws->UA_tw_im;
    float *J_zeros       = ws->J_zeros;

    // Basic parameters
    int N_D = p->N_D;
    int N_A = p->N_A;
    int N_t1 = p->N_t1;
    float *J = p->J;

    int t1;
    float *factor_1_re, *factor_1_im;
    float *factor_2_re, *factor_2_im;
    //suffix indicates matrix dimension
    float *intermediate_DA_re, *intermediate_DA_im;
    float *intermediate_AD_re, *intermediate_AD_im;
    float *intermediate_DD_re, *intermediate_DD_im;
    intermediate_DD_re = (float *)calloc(N_D*N_D,sizeof(float));
    intermediate_DD_im = (float *)calloc(N_D*N_D,sizeof(float));
    intermediate_AD_re = (float *)calloc(N_A*N_D,sizeof(float));
    intermediate_AD_im = (float *)calloc(N_A*N_D,sizeof(float));
    intermediate_DA_re = (float *)calloc(N_A*N_D,sizeof(float));
    intermediate_DA_im = (float *)calloc(N_A*N_D,sizeof(float));

    for (t1=0;t1<N_t1;t1++){
	    factor_1_re = UDh_rho_J_UA_re_t1 + N_A*N_D * t1;
	    factor_1_im = UDh_rho_J_UA_im_t1 + N_A*N_D * t1;
        // product 1
	    // Ui @ Jh_UDh_tw = intermediate_product_DD
        complex_matrix_product(factor_1_re, factor_1_im, Jh_UDh_tw_re, Jh_UDh_tw_im, intermediate_DD_re, intermediate_DD_im, N_D,N_D,N_A);
        complex_matrix_product(intermediate_DD_re, intermediate_DD_im, J, J_zeros, intermediate_DA_re, intermediate_DA_im, N_D,N_A,N_D);
        complex_matrix_product(UD_tw_re, UD_tw_im, intermediate_DA_re, intermediate_DA_im, intermediate_product_1_re+t1*N_D*N_A, intermediate_product_1_im+t1*N_D*N_A,N_D,N_A,N_D);
        // product 2 
	    complex_matrix_product(Jh_UD_tw_re, Jh_UD_tw_im, intermediate_DD_re, intermediate_DD_im, intermediate_product_2_re+t1*N_A*N_D, intermediate_product_2_im+t1*N_A*N_D,N_A,N_D,N_D);
        // product 3
	    // reuse the intermediate_DD slot here
        complex_matrix_product(factor_1_re, factor_1_im, UAh_Jh_tw_re, UAh_Jh_tw_im, intermediate_DD_re, intermediate_DD_im, N_D,N_D,N_A);
        complex_matrix_product(UA_Jh_tw_re, UA_Jh_tw_im, intermediate_DD_re, intermediate_DD_im, intermediate_product_3_re+t1*N_A*N_D, intermediate_product_3_im+t1*N_A*N_D, N_A,N_D,N_D);	
        // product 4
        factor_2_re = UAh_Jh_rho_UD_re_t1 + N_A * N_D * t1;
        factor_2_im = UAh_Jh_rho_UD_im_t1 + N_A * N_D * t1;
        complex_matrix_product(factor_2_re, factor_2_im, J_UAh_Jh_tw_re, J_UAh_Jh_tw_im, intermediate_AD_re, intermediate_AD_im,N_A,N_D,N_D);
        complex_matrix_product(UA_tw_re, UA_tw_im, intermediate_AD_re, intermediate_AD_im,intermediate_product_4_re + t1*N_A*N_D, intermediate_product_4_im + t1*N_A*N_D, N_A,N_D,N_A);
    }
    // free(factor_1_re);
    // free(factor_1_im);
    // free(factor_2_re);
    // free(factor_2_im);
    free(intermediate_AD_re);
    free(intermediate_AD_im);
    free(intermediate_DD_re);
    free(intermediate_DD_im);
    free(intermediate_DA_re);
    free(intermediate_DA_im);
}

void compute_rate_from_4th_response(float *responses_4th_tw, float *rate_matrix_4th_I, float *rate_matrix_4th_II, float *rate_matrix_2nd, int N_segments, int N_tw, t_non *non, float prefactor, int samples){
    // integrate the waiting time response for each segment combination 
    int si, sj, tw;
    float response, plateau_I, plateau_II;
    float dt = (float)non->deltat;
    float weight;
    clearvec(rate_matrix_4th_I, N_segments*N_segments);
    clearvec(rate_matrix_4th_II, N_segments*N_segments);

    for (si=0;si<N_segments;si++){
        for (sj=0;sj<N_segments;sj++){
            if (si != sj){	
                plateau_I = -1e-6 * rate_matrix_2nd[si+N_segments*sj] * (rate_matrix_2nd[si+N_segments*sj] + rate_matrix_2nd[sj+N_segments*si]);
                plateau_II = responses_4th_tw[N_segments * N_segments * (N_tw - 1) + si * N_segments + sj] * prefactor / (samples + 1);
                for (tw=0;tw<N_tw;tw++){
		            weight = 1;
                    response = responses_4th_tw[N_segments * N_segments * tw + si * N_segments + sj] * prefactor / (samples + 1);
                    if (tw == 0){
                        weight =  0.5;
                    }
                    rate_matrix_4th_I[si + N_segments * sj] += weight * dt * (response - plateau_I) * 1e3; // convert to inverse picoseconds
                    rate_matrix_4th_II[si + N_segments * sj] += weight * dt * (response - plateau_II) * 1e3;
                }
            }
	}
    }
}

void fourth_order_response_1_sample(fourth_order_params *p) {
    // compute the 4th order response for a single sample of the propagators
    // The inter-segment coupling matrix and donor density matrices can be based on many samples
    // this rate is between segment si and sj

    float *rho0_D                = p->rho0_D;
    float *J                     = p->J;
    float *integrated_response_tw = p->integrated_response_tw;
    float *U_re_t1_array         = p->U_re_t1_array;
    float *U_im_t1_array         = p->U_im_t1_array;
    float *U_re_tw_array         = p->U_re_tw_array;
    float *U_im_tw_array         = p->U_im_tw_array;
    float *U_re_t2_array         = p->U_re_t2_array;
    float *U_im_t2_array         = p->U_im_t2_array;

    float *diagram_1             = p->diagram_1;
    float *diagram_2             = p->diagram_2;
    float *diagram_3             = p->diagram_3;
    float *diagram_4             = p->diagram_4;

    // Integers uitpakken
    int N_A                  = p->N_A;
    int N_D                  = p->N_D;
    int N_t1                 = p->N_t1;
    int N_tw                 = p->N_tw;
    int N_t2                 = p->N_t2;
    int times_N2             = p->times_N2;
    int s_D                  = p->s_D;
    int s_A                  = p->s_A;
    int N_segments           = p->N_segments;
    int largest_segment_size = p->largest_segment_size;
    t_non *non               = p->non;

    int t1, tw, t2, i1, i2;

    // used as a dummy matrix in a few matrix products below
    float *J_zeros;
    J_zeros = (float * )calloc(N_A*N_D,sizeof(float));
    // create a transposed copy of J
    float *JT;
    JT = (float * )calloc(N_A*N_D,sizeof(float));
    for (i1=0;i1<N_D;i1++){
        for (i2=0;i2<N_A;i2++){
	        JT[i2 * N_D + i1] = J[i1 * N_A + i2];
	    }
    }

    float *UDh_rho_J_UA_re_t1, *UAh_Jh_rho_UD_re_t1;
    float *UDh_rho_J_UA_im_t1, *UAh_Jh_rho_UD_im_t1;
    UDh_rho_J_UA_re_t1 = (float *)calloc(N_A*N_D*N_t1,sizeof(float));
    UDh_rho_J_UA_im_t1 = (float *)calloc(N_A*N_D*N_t1,sizeof(float));
    UAh_Jh_rho_UD_re_t1 = (float *)calloc(N_A*N_D*N_t1,sizeof(float));
    UAh_Jh_rho_UD_im_t1 = (float *)calloc(N_A*N_D*N_t1,sizeof(float));
    compute_UDh_rho_J_UA_t1(UDh_rho_J_UA_re_t1, 
                        UDh_rho_J_UA_im_t1, 
                        UAh_Jh_rho_UD_re_t1, 
                        UAh_Jh_rho_UD_im_t1, 
                        p);

    float *UA_tw_re, *UA_tw_im;
    float *UD_tw_re, *UD_tw_im;
    float *UA_tw_h_re, *UA_tw_h_im;
    float *UD_tw_h_re, *UD_tw_h_im;
    float *Jh_UDh_tw_re, *Jh_UDh_tw_im;
    float *J_UAh_Jh_tw_re, *J_UAh_Jh_tw_im;
    float *Jh_UD_tw_re, *Jh_UD_tw_im;
    float *UAh_Jh_tw_re, *UAh_Jh_tw_im;
    float *UA_Jh_tw_re, *UA_Jh_tw_im;

    float *intermediate_product_1_re, *intermediate_product_1_im;
    float *intermediate_product_2_re, *intermediate_product_2_im;
    float *intermediate_product_3_re, *intermediate_product_3_im;
    float *intermediate_product_4_re, *intermediate_product_4_im;

    UA_tw_re = (float *)calloc(N_A*N_A,sizeof(float));
    UA_tw_im = (float *)calloc(N_A*N_A,sizeof(float));
    UD_tw_re = (float *)calloc(N_D*N_D,sizeof(float));
    UD_tw_im = (float *)calloc(N_D*N_D,sizeof(float));
    UA_tw_h_re = (float *)calloc(N_A*N_A,sizeof(float));
    UA_tw_h_im = (float *)calloc(N_A*N_A,sizeof(float));
    UD_tw_h_re = (float *)calloc(N_D*N_D,sizeof(float));
    UD_tw_h_im = (float *)calloc(N_D*N_D,sizeof(float));
    Jh_UDh_tw_re = (float *)calloc(N_D*N_A,sizeof(float));
    Jh_UDh_tw_im = (float *)calloc(N_D*N_A,sizeof(float));
    Jh_UD_tw_re = (float *)calloc(N_D*N_A,sizeof(float));
    Jh_UD_tw_im = (float *)calloc(N_D*N_A,sizeof(float));
    J_UAh_Jh_tw_re = (float *)calloc(N_D*N_D,sizeof(float));
    J_UAh_Jh_tw_im = (float *)calloc(N_D*N_D,sizeof(float));
    UAh_Jh_tw_re = (float *)calloc(N_D*N_A,sizeof(float));
    UAh_Jh_tw_im = (float *)calloc(N_D*N_A,sizeof(float));
    UA_Jh_tw_re = (float *)calloc(N_D*N_A,sizeof(float));
    UA_Jh_tw_im = (float *)calloc(N_D*N_A,sizeof(float));

    intermediate_product_1_re = (float *)calloc(N_D*N_A*N_t1,sizeof(float));
    intermediate_product_1_im = (float *)calloc(N_D*N_A*N_t1,sizeof(float));
    intermediate_product_2_re = (float *)calloc(N_D*N_A*N_t1,sizeof(float));
    intermediate_product_2_im = (float *)calloc(N_D*N_A*N_t1,sizeof(float));
    intermediate_product_3_re = (float *)calloc(N_D*N_A*N_t1,sizeof(float));
    intermediate_product_3_im = (float *)calloc(N_D*N_A*N_t1,sizeof(float));
    intermediate_product_4_re = (float *)calloc(N_D*N_A*N_t1,sizeof(float));
    intermediate_product_4_im = (float *)calloc(N_D*N_A*N_t1,sizeof(float));

    float *UAh_Jh_UD_t2_im, *UAh_Jh_UD_t2_re;
    UAh_Jh_UD_t2_im = (float *)calloc(N_D*N_A,sizeof(float));
    UAh_Jh_UD_t2_re = (float *)calloc(N_D*N_A,sizeof(float));
    float *UDh_J_UA_t2_re, *UDh_J_UA_t2_im;
    UDh_J_UA_t2_re = (float *)calloc(N_D*N_A,sizeof(float));
    UDh_J_UA_t2_im = (float *)calloc(N_D*N_A,sizeof(float));

    float *UA_snap_t2_re, *UA_snap_t2_im;
    float *UD_snap_t2_re, *UD_snap_t2_im;
    UD_snap_t2_re = (float *)calloc(N_D*N_D,sizeof(float));
    UD_snap_t2_im = (float *)calloc(N_D*N_D,sizeof(float));
    UA_snap_t2_re = (float *)calloc(N_A*N_A,sizeof(float));
    UA_snap_t2_im = (float *)calloc(N_A*N_A,sizeof(float));

    float *UA_comp_t2_re, *UA_comp_t2_im;
    float *UA_comp_h_t2_re, *UA_comp_h_t2_im;
    float *UD_comp_t2_re, *UD_comp_t2_im;
    UD_comp_t2_re = (float *)calloc(N_D*N_D,sizeof(float));
    UD_comp_t2_im = (float *)calloc(N_D*N_D,sizeof(float));
    UA_comp_t2_re = (float *)calloc(N_A*N_A,sizeof(float));
    UA_comp_t2_im = (float *)calloc(N_A*N_A,sizeof(float));
    UA_comp_h_t2_re = (float *)calloc(N_A*N_A,sizeof(float));
    UA_comp_h_t2_im = (float *)calloc(N_A*N_A,sizeof(float));

    float *temp_re, *temp_im;
    temp_re = (float *)calloc(N_D*N_A,sizeof(float));
    temp_im = (float *)calloc(N_D*N_A,sizeof(float));

    float *work_re_A, *work_im_A;
    float *work_re_D, *work_im_D;
    work_re_A = (float *)calloc(N_A*N_A,sizeof(float));
    work_im_A = (float *)calloc(N_A*N_A,sizeof(float));
    work_re_D = (float *)calloc(N_D*N_D,sizeof(float));
    work_im_D = (float *)calloc(N_D*N_D,sizeof(float));

    // the actual three fold integral
    for (tw=0;tw<N_tw;tw++){
        // read the propagator at this waiting time
        read_propagator_from_big_array(U_re_tw_array, UA_tw_re, N_tw, s_A, N_A, largest_segment_size, tw);
        read_propagator_from_big_array(U_im_tw_array, UA_tw_im, N_tw, s_A, N_A, largest_segment_size, tw);
        read_propagator_from_big_array(U_re_tw_array, UD_tw_re, N_tw, s_D, N_D, largest_segment_size, tw);
        read_propagator_from_big_array(U_im_tw_array, UD_tw_im, N_tw, s_D, N_D, largest_segment_size, tw);

        hermitian_conjugate(UA_tw_re, UA_tw_im, UA_tw_h_re, UA_tw_h_im, N_A, N_A);
        hermitian_conjugate(UD_tw_re, UD_tw_im, UD_tw_h_re, UD_tw_h_im, N_D, N_D);

        // Jh_UDh_tw = J.T @ UD_tw_h
        complex_matrix_product(JT, J_zeros, UD_tw_h_re,UD_tw_h_im,Jh_UDh_tw_re,Jh_UDh_tw_im,N_A,N_D,N_D);	
        // Jh_UD_tw =  J.T @ UD_tw
        complex_matrix_product(JT, J_zeros, UD_tw_re,UD_tw_im,Jh_UD_tw_re,Jh_UD_tw_im,N_A,N_D,N_D);	
        // UAh_Jh_tw = UA_tw_h @ J.T
        complex_matrix_product(UA_tw_h_re, UA_tw_h_im, JT, J_zeros, UAh_Jh_tw_re, UAh_Jh_tw_im,N_A,N_D,N_A);
        // J_UAh_Jh_tw = J @ UAh_Jh_tw
        complex_matrix_product(J, J_zeros, UAh_Jh_tw_re, UAh_Jh_tw_im, J_UAh_Jh_tw_re, J_UAh_Jh_tw_im,N_D,N_D,N_A);
        // UA_Jh_tw = UA_tw @ J.T
        complex_matrix_product(UA_tw_re, UA_tw_im, JT, J_zeros, UA_Jh_tw_re, UA_Jh_tw_im, N_A, N_D, N_A);

        // clear the four intermediate product arrays
        clearvec(intermediate_product_1_re, N_D*N_A*N_t1);
        clearvec(intermediate_product_1_im, N_D*N_A*N_t1);
        clearvec(intermediate_product_2_re, N_D*N_A*N_t1);
        clearvec(intermediate_product_2_im, N_D*N_A*N_t1);
        clearvec(intermediate_product_3_re, N_D*N_A*N_t1);
        clearvec(intermediate_product_3_im, N_D*N_A*N_t1);
        clearvec(intermediate_product_4_re, N_D*N_A*N_t1);
        clearvec(intermediate_product_4_im, N_D*N_A*N_t1);

        // calculate the intermediate products for this specific tw for all matrices at each t1
        // this saves matrix multiplications in the most nested loop.

        // store the many input matrices/trajectories in a structure
        fourth_order_workspace ws;

        // outputs
        ws.intermediate_product_1_re = intermediate_product_1_re;
        ws.intermediate_product_1_im = intermediate_product_1_im;
        ws.intermediate_product_2_re = intermediate_product_2_re;
        ws.intermediate_product_2_im = intermediate_product_2_im;
        ws.intermediate_product_3_re = intermediate_product_3_re;
        ws.intermediate_product_3_im = intermediate_product_3_im;
        ws.intermediate_product_4_re = intermediate_product_4_re;
        ws.intermediate_product_4_im = intermediate_product_4_im;

        // inputs (t1 fase)
        ws.UDh_rho_J_UA_re_t1  = UDh_rho_J_UA_re_t1;
        ws.UDh_rho_J_UA_im_t1  = UDh_rho_J_UA_im_t1;
        ws.UAh_Jh_rho_UD_re_t1 = UAh_Jh_rho_UD_re_t1;
        ws.UAh_Jh_rho_UD_im_t1 = UAh_Jh_rho_UD_im_t1;

        // inputs (tw fase)
        ws.Jh_UDh_tw_re   = Jh_UDh_tw_re;
        ws.Jh_UDh_tw_im   = Jh_UDh_tw_im;
        ws.Jh_UD_tw_re    = Jh_UD_tw_re;
        ws.Jh_UD_tw_im    = Jh_UD_tw_im;
        ws.UD_tw_re       = UD_tw_re;
        ws.UD_tw_im       = UD_tw_im;
        ws.UA_Jh_tw_re    = UA_Jh_tw_re;
        ws.UA_Jh_tw_im    = UA_Jh_tw_im;
        ws.UAh_Jh_tw_re   = UAh_Jh_tw_re;
        ws.UAh_Jh_tw_im   = UAh_Jh_tw_im;
        ws.J_UAh_Jh_tw_re = J_UAh_Jh_tw_re;
        ws.J_UAh_Jh_tw_im = J_UAh_Jh_tw_im;
        ws.UA_tw_re       = UA_tw_re;
        ws.UA_tw_im       = UA_tw_im;
            
        // constant
        ws.J_zeros = J_zeros;

        compute_4_intermediate_products(&ws, p);

        // initialize the acceptor propagator over t2 as a unit matrix
        unitmat(UA_comp_t2_re,N_A);
        clearvec(UA_comp_t2_im,N_A*N_A);
        unitmat(UD_comp_t2_re,N_D);
        clearvec(UD_comp_t2_im,N_D*N_D);

        // clear the diagram for each waiting time
        clearvec(diagram_1, N_t1*N_t2);
        clearvec(diagram_2, N_t1*N_t2);
        clearvec(diagram_3, N_t1*N_t2);
        clearvec(diagram_4, N_t1*N_t2);

	    for(t2=0;t2<N_t2;t2++){
            //compute the hermitian conjugate of the acceptor t2 propagator
            hermitian_conjugate(UA_comp_t2_re, UA_comp_t2_im, UA_comp_h_t2_re, UA_comp_h_t2_im, N_A, N_A);
	        // printf("N_A UA_comp_t2_re, UA_comp_t2_im: %d %f, %f\n",N_A, matrix_sum(UA_comp_t2_re, N_A), matrix_sum(UA_comp_t2_im, 9));
            // printf("UD_comp_t2_re, UD_comp_t2_im: %f, %f\n", matrix_sum(UD_comp_t2_re,9), matrix_sum(UD_comp_t2_im, 9) );
            // printf("N_D, UA_snap_t2_re, UA_snap_t2_im: %d %f, %f\n",N_D, matrix_sum(UA_snap_t2_re, N_A), matrix_sum(UA_snap_t2_im, 9));
            // printf("N_D, UD_snap_t2_re, UD_snap_t2_im: %d %f, %f\n",N_D, matrix_sum(UD_snap_t2_re, 9), matrix_sum(UD_snap_t2_im, 9));

            // the hermitian product of the donor t2 propagator is not needed
            // U_D_comp_t2_h = U_D_comp_t2.conj().T

            // precalculate the full t2 factors
            // UAh_Jh_UD_t2 = UA_comp_t2_h @ J.T @ UD_comp_t2
	        // the J_zeros could be made faster, as this step is unnecessary 
            // printf("UA_comp_h_t2_re, UA_comp_h_t2_im: %f, %f\n", matrix_sum(UA_comp_h_t2_re,9), matrix_sum(UA_comp_h_t2_im, 9) );
            complex_matrix_product(UA_comp_h_t2_re, UA_comp_h_t2_im, JT, J_zeros, temp_re, temp_im,N_A,N_D,N_A);
            // printf("temp_re, temp_im: %f, %f\n", matrix_sum(temp_re,9), matrix_sum(temp_im, 9) );
            // printf("JT , J_zeros: %f, %f\n", matrix_sum(J,9), matrix_sum(J_zeros, 9) );
            // printf("UD_comp_t2_re, UD_comp_t2_im: %f, %f\n", matrix_sum(UD_comp_t2_re,9), matrix_sum(UD_comp_t2_im, 9) );
	        complex_matrix_product(temp_re, temp_im, UD_comp_t2_re, UD_comp_t2_im,UAh_Jh_UD_t2_re, UAh_Jh_UD_t2_im, N_A, N_D, N_D);
	        //UDh_J_UA_t2 = UAh_Jh_UD_t2.conj().T
            hermitian_conjugate(UAh_Jh_UD_t2_re,UAh_Jh_UD_t2_im, UDh_J_UA_t2_re, UDh_J_UA_t2_im, N_A, N_D);
             
	        // precalculated all t1 information outside the loops, so it is more efficient
            // now the most nested loop only involves a single matrix multiplication per Feynman Diagram
             
            for (t1=0;t1<N_t1;t1++){
                // define the 4 unique matrix products (one for each double-sided Feynman diagram)
		        // only the real part of the final products will be needed
		        // matrix_product_1 = intermediate_product_1[t1] @ UAh_Jh_UD_t2
                // matrix_product_2 = intermediate_product_2[t1] @ UDh_J_UA_t2
                // matrix_product_3 = intermediate_product_3[t1] @ UDh_J_UA_t2
                // matrix_product_4 = intermediate_product_4[t1] @ UDh_J_UA_t2
		        
                // there are 8 ~N^2 computations in this most nested loop
                // printf("intermediate_product_1_re, UAh_Jh_UD_t2_re: %f, %f\n",matrix_sum(intermediate_product_1_re+t1*N_A*N_D, N_A), matrix_sum(UAh_Jh_UD_t2_re, 9));
                // printf("intermediate_product_1_im, UAh_Jh_UD_t2_im: %f, %f\n",matrix_sum(intermediate_product_1_im+t1*N_A*N_D, N_A), matrix_sum(UAh_Jh_UD_t2_im, 9));
                diagram_1[t1 + N_t1*t2] += matrix_mul_traced_DA(intermediate_product_1_re+t1*N_A*N_D, UAh_Jh_UD_t2_re, N_D, N_A);
                diagram_1[t1 + N_t1*t2] -= matrix_mul_traced_DA(intermediate_product_1_im+t1*N_A*N_D, UAh_Jh_UD_t2_im, N_D, N_A);
                diagram_2[t1 + N_t1*t2] += matrix_mul_traced_DA(intermediate_product_2_re+t1*N_A*N_D, UDh_J_UA_t2_re, N_A, N_D);
                diagram_2[t1 + N_t1*t2] -= matrix_mul_traced_DA(intermediate_product_2_im+t1*N_A*N_D, UDh_J_UA_t2_im, N_A, N_D);
                diagram_3[t1 + N_t1*t2] += matrix_mul_traced_DA(intermediate_product_3_re+t1*N_A*N_D, UDh_J_UA_t2_re, N_A, N_D);
                diagram_3[t1 + N_t1*t2] -= matrix_mul_traced_DA(intermediate_product_3_im+t1*N_A*N_D, UDh_J_UA_t2_im, N_A, N_D);
                diagram_4[t1 + N_t1*t2] += matrix_mul_traced_DA(intermediate_product_4_re+t1*N_A*N_D, UDh_J_UA_t2_re, N_A, N_D);
                diagram_4[t1 + N_t1*t2] -= matrix_mul_traced_DA(intermediate_product_4_im+t1*N_A*N_D, UDh_J_UA_t2_im, N_A, N_D);

                // printf("diagram1: %f\n",matrix_sum(diagram_1, N_t1));
                // printf("diagram2: %f\n",matrix_sum(diagram_2, N_t1));
                // printf("diagram3: %f\n",matrix_sum(diagram_3, N_t1));
                // printf("diagram4: %f\n",matrix_sum(diagram_4, N_t1));

            }//close the loop over t1

	        // U_A_comp_t2 represents U_A(tw+t2,tw)
            // multiply from the left, because t2 is forward in time
            // place this after the t1 loop such that the first t2 diagram has the unity propagator.
  
	        // calculate the compounded propagators for each t2
            // these propagators run from
            //     'coherence interval length' + 'waiting time'
            //                            to
            //     'coherence interval length' + 'waiting time' + 't2'
            // in the passed sample trajectory

            // read the propagators for this snapshot from precalculated array
            read_propagator_from_big_array(U_re_t2_array, UA_snap_t2_re, times_N2, s_A, N_A, largest_segment_size, tw + t2);
            read_propagator_from_big_array(U_im_t2_array, UA_snap_t2_im, times_N2, s_A, N_A, largest_segment_size, tw + t2);
	        read_propagator_from_big_array(U_re_t2_array, UD_snap_t2_re, times_N2, s_D, N_D, largest_segment_size, tw + t2);
            read_propagator_from_big_array(U_im_t2_array, UD_snap_t2_im, times_N2, s_D, N_D, largest_segment_size, tw + t2);
            // propagate t2
            propagate_snapshot(UA_snap_t2_re, UA_snap_t2_im, &UA_comp_t2_re, &UA_comp_t2_im, &work_re_A, &work_im_A, N_A);
            propagate_snapshot(UD_snap_t2_re, UD_snap_t2_im, &UD_comp_t2_re, &UD_comp_t2_im, &work_re_D, &work_im_D, N_D);
        }// close the loop over t2
    
    // integrate over the two coherence intervals
    // this divisoin actually assumes that N_t1 and N_t2 are equal
    // this should be turned on for better integration (turned off for intermediate testing)
    for (t1 = 0; t1< N_t1;t1++){
        diagram_1[t1+0] /= 2; // first row/column
	    diagram_1[t1*N_t1+0] /=2; //first row/column
        diagram_2[t1+0] /= 2; // first row/column
	    diagram_2[t1*N_t1+0] /=2; //first row/column
        diagram_3[t1+0] /= 2; // first row/column
	    diagram_3[t1*N_t1+0] /=2; //first row/column
        diagram_4[t1+0] /= 2; // first row/column
	    diagram_4[t1*N_t1+0] /=2; //first row/column
    }
    for (t1 = 0; t1< N_t1;t1++){
        for (t2 = 0; t2< N_t2;t2++){
            integrated_response_tw[s_D * N_segments + s_A + N_segments * N_segments * tw] += diagram_1[t1 + N_t1*t2] + diagram_2[t1 + N_t1*t2] + diagram_3[t1 + N_t1*t2] + diagram_4[t1 + N_t1*t2];
	    }
    }

    }//close the loop over the waiting time
    // cleanup
    free(JT);
    free(J_zeros);
    free(temp_re);
    free(temp_im);

    free(UDh_rho_J_UA_re_t1);
    free(UAh_Jh_rho_UD_re_t1);
    free(UDh_rho_J_UA_im_t1);
    free(UAh_Jh_rho_UD_im_t1);

    free(UAh_Jh_UD_t2_im);
    free(UAh_Jh_UD_t2_re);
    free(UDh_J_UA_t2_re);
    free(UDh_J_UA_t2_im);

    free(UA_tw_re);
    free(UA_tw_im);
    free(UD_tw_re);
    free(UD_tw_im);
    free(UA_tw_h_re);
    free(UA_tw_h_im);
    free(UD_tw_h_re);
    free(UD_tw_h_im);
 
    free(UD_snap_t2_re);
    free(UD_snap_t2_im);
    free(UA_snap_t2_re);
    free(UA_snap_t2_im);

    free(UD_comp_t2_re);
    free(UD_comp_t2_im);
    free(UA_comp_t2_re);
    free(UA_comp_t2_im);
    free(UA_comp_h_t2_re);
    free(UA_comp_h_t2_im);
 
    free(intermediate_product_1_re);
    free(intermediate_product_1_im);
    free(intermediate_product_2_re);
    free(intermediate_product_2_im);
    free(intermediate_product_3_re);
    free(intermediate_product_3_im);
    free(intermediate_product_4_re);
    free(intermediate_product_4_im);

    free(UA_Jh_tw_re);
    free(UA_Jh_tw_im);
    free(J_UAh_Jh_tw_re);
    free(J_UAh_Jh_tw_im);

    free(UAh_Jh_tw_re);
    free(UAh_Jh_tw_im);

    free(Jh_UD_tw_re);
    free(Jh_UD_tw_im);
    free(Jh_UDh_tw_re);
    free(Jh_UDh_tw_im);

    free(work_re_A);
    free(work_im_A);
    free(work_re_D);
    free(work_im_D);
}

/* The routine to compute the full 4th order TD-MCFRET rate matrix */
void full_4th_order_main(float *rho_0,float *J_full,t_non *non){
    
    /* Preset standard NISE variables*/
    int x; //used in display of propagator unction

    /* Hamiltonian of the whole system - all donors and acceptors included */
    float *Hamil_i_e;
    /* Transition dipoles for coupling on the fly */
    // float *mu_xyz;
    float shift1;

    // int prefactor = -2 * (np.pi * 2 * c)**4;
    float c = 2.9979245800e-5; // cm/fs
    float prefactor = -2 * (twoPi*c)*(twoPi*c)*(twoPi*c)*(twoPi*c)*(float)(non->deltat)*(non->deltat);//could be done more cleanly


    /* Time parameters */
    time_t time_now,time_old,time_0;
    /* Initialize time */
    time(&time_now);
    time(&time_0);
    shift1=(non->max1+non->min1)/2;
    printf("Frequency shift %f.\n",shift1);
    non->shifte=shift1;

    int row, column;
    /* File handles */
    FILE *H_traj;
    FILE *mu_traj;
    // FILE *C_traj;
    FILE *log;
    FILE *Cfile;
    FILE *rate_matrix_4th_I_file;
    FILE *rate_matrix_4th_II_file;
    FILE *responses_4th_file;

    // mu_xyz=(float *)calloc(non->singles*3,sizeof(float));
    Hamil_i_e=(float *)calloc((non->singles+1)*non->singles/2,sizeof(float));
    /* Open Trajectory files */
    open_files(non,&H_traj,&mu_traj,&Cfile);
    /* Here we want to call the routine for checking the trajectory files */
    control(non);

    /* Initialize sample numbers */
    int samples, N_samples, N_segments;
    N_samples=determine_samples(non);
    N_segments=project_dim(non);
    log=fopen("NISE.log","a");
    fprintf(log,"Begin sample: %d, End sample: %d.\n",non->begin,non->end);
    fclose(log);

    /* End of the preset*/
    int N_t1, N_tw, N_t2;
    N_tw = non->tmax2;
    N_t1 = non->tmax1;
    N_t2 = non->tmax3;

    int N;
    int N_site_si, N_site_sj, N_site_max;
    int *H_indices_si, *H_indices_sj;

    N = non->singles;
    H_indices_si = (int *)calloc(N,sizeof(int));
    H_indices_sj = (int *)calloc(N,sizeof(int));
 
    int si, sj;
    int t_ref, tw, ti, tj;
    int idx;

    float *rate_matrix_2nd;
    rate_matrix_2nd = (float *)calloc(N_segments*N_segments,sizeof(float));
    read_matrix_from_file("RateMatrix.dat",rate_matrix_2nd,N_segments);

    // prepate the 4th order rate matrices (two variants)
    float *rate_matrix_4th_I, *rate_matrix_4th_II;
    rate_matrix_4th_I = (float *)calloc(N_segments*N_segments,sizeof(float));
    rate_matrix_4th_II = (float *)calloc(N_segments*N_segments,sizeof(float));

    float *responses_4th_tw;
    responses_4th_tw = (float *)calloc(N_segments * N_segments*N_tw,sizeof(float));

    // prepate the propagator arrays
    float *big_propagator_array_t1_re, *big_propagator_array_t1_im;
    float *big_propagator_array_tw_re, *big_propagator_array_tw_im;
    float *big_propagator_array_t2_re, *big_propagator_array_t2_im;
    int sample_length = N_t1+N_tw+N_t2;
    // maximum size of segment (Nsite) as two dimension lengths: this implies storing many zeros
    // could be stored in a smaller 3d array as well, for speedup and smaller memory needs
    N_site_max = find_max_segment_size(non->psites, non);
    size_t N_dim_big_array_t1 = (size_t)N_segments*(size_t)N_t1*(size_t)N_site_max*(size_t)N_site_max;
    size_t N_dim_big_array_tw = (size_t)N_segments*(size_t)N_tw*(size_t)N_site_max*(size_t)N_site_max;
    // the t2 interval needs trajectory snapshots from 0 to N_tw+N_t2
    int times_N2 = N_tw+N_t2+1;
    size_t N_dim_big_array_t2 = (size_t)N_segments*(size_t)times_N2*(size_t)N_site_max*(size_t)N_site_max;
    big_propagator_array_t1_re = (float *)calloc(N_dim_big_array_t1,sizeof(float));
    big_propagator_array_t1_im = (float *)calloc(N_dim_big_array_t1,sizeof(float));
    big_propagator_array_tw_re = (float *)calloc(N_dim_big_array_tw,sizeof(float));
    big_propagator_array_tw_im = (float *)calloc(N_dim_big_array_tw,sizeof(float));
    big_propagator_array_t2_re = (float *)calloc(N_dim_big_array_t2,sizeof(float));
    big_propagator_array_t2_im = (float *)calloc(N_dim_big_array_t2,sizeof(float));

    // initialise the diagram arrays, to be used for each segment combination at each sample
    float *diagram_1, *diagram_2, *diagram_3, *diagram_4;
    diagram_1 = (float *)calloc(N_t1*N_t2,sizeof(float));
    diagram_2 = (float *)calloc(N_t1*N_t2,sizeof(float));
    diagram_3 = (float *)calloc(N_t1*N_t2,sizeof(float));
    diagram_4 = (float *)calloc(N_t1*N_t2,sizeof(float));

    // write the diagrams for segments 0 and 1 (s01)to a file for checking by the user
    float *diagram_1_s01, *diagram_2_s01, *diagram_3_s01, *diagram_4_s01;
    diagram_1_s01 = (float *)calloc(N_t1*N_t2,sizeof(float));
    diagram_2_s01 = (float *)calloc(N_t1*N_t2,sizeof(float));
    diagram_3_s01 = (float *)calloc(N_t1*N_t2,sizeof(float));
    diagram_4_s01 = (float *)calloc(N_t1*N_t2,sizeof(float));

    float *diagram_list[] = {diagram_1, diagram_2, diagram_3, diagram_4};
    float *diagram_s01_list[] = {diagram_1_s01, diagram_2_s01, diagram_3_s01, diagram_4_s01};
    char *diagram_filenames[] = {"diagram_1_s01.dat", "diagram_2_s01.dat", "diagram_3_s01.dat", "diagram_4_s01.dat"};
    int diag;
    FILE *diagram_file;


    // suggestion: printlevel 2: write response functions 

    // loop over samples and store the 4th order rate after each sample
    printf("Starting the loop over the samples.\n");
    for (samples=non->begin;samples<non->end;samples++){
        //starting index of this sample
	    t_ref = samples*non->sample;
        clearvec(big_propagator_array_t2_re,N_dim_big_array_t2);
        clearvec(big_propagator_array_t2_im,N_dim_big_array_t2);
        clearvec(big_propagator_array_t1_re,N_dim_big_array_t1);
        clearvec(big_propagator_array_t1_im,N_dim_big_array_t1);
        clearvec(big_propagator_array_tw_re,N_dim_big_array_tw);
        clearvec(big_propagator_array_tw_im,N_dim_big_array_tw);

        // single loop over segments to precalculate the propagator 
        log=fopen("NISE.log","a");
        fprintf(log,"Starting the computation of the propagators in sample %d\n", samples);
        time_now=log_time(time_now,log);
        fclose(log);
        for (si=0;si<N_segments;si++){
            N_site_si = find_H_indices_segment(non->psites, H_indices_si, si, non);
            /* The segment si hamiltonian in upper triangle format */
            float *Hamiltonian_segment_triu;
            Hamiltonian_segment_triu = (float *)calloc((N_site_si+1)*N_site_si/2,sizeof(float));
            float *U_re, *U_im, *U_h_re, *U_h_im, *U_re_snap, *U_im_snap;
            U_im=(float *)calloc(N_site_si*N_site_si,sizeof(float));
            U_h_im=(float *)calloc(N_site_si*N_site_si,sizeof(float));
            U_re=(float *)calloc(N_site_si*N_site_si,sizeof(float));
            U_h_re=(float *)calloc(N_site_si*N_site_si,sizeof(float));
            U_re_snap =(float *)calloc(N_site_si*N_site_si,sizeof(float));
            U_im_snap =(float *)calloc(N_site_si*N_site_si,sizeof(float));

            float *work_re_si, *work_im_si;
            work_re_si =(float *)calloc(N_site_si*N_site_si,sizeof(float));
            work_im_si =(float *)calloc(N_site_si*N_site_si,sizeof(float));

            // initialize the n_i * n_i propagator as a unit matrix
            unitmat(U_re,N_site_si);
            clearvec(U_im,N_site_si*N_site_si);

            log=fopen("NISE.log","a");
            fprintf(log,"Starting the computation of the t1 propagators of seg %d, in sample %d\n", si, samples);
            time_now=log_time(time_now,log);
            fclose(log);
            /* Loop over the needed t1 interval. propagate this one backward in time */
            for (ti=0;ti<N_t1;ti++){
                        tj = t_ref + N_t1 - ti;
                // store the hermitian conjugate of the propagators that are calculated backward in time
                hermitian_conjugate(U_re, U_im, U_h_re, U_h_im, N_site_si,N_site_si); 
                write_propagator_to_big_array(big_propagator_array_t1_re,U_h_re,N_t1,si,N_site_si, N_site_max, ti);
                write_propagator_to_big_array(big_propagator_array_t1_im,U_h_im,N_t1,si,N_site_si, N_site_max, ti);
                /* Read Hamiltonian */
                read_Hamiltonian(non,Hamil_i_e,H_traj,tj);
                // isolate the segment i with projection routine to obtain smaller matrix
                isolate_segment_Hamiltonian_triu(Hamil_i_e, Hamiltonian_segment_triu, H_indices_si, N_site_si, non);
                        // Propagate segment i (~N_i^3 process)
                        // propagate after writing to big array, such that first propagator is identity
                //        propagate_matrix_segments(non,Hamiltonian_segment_triu,U_re,U_im,-1,samples,tj*x, N_site_si);
                time_evolution_mat_non_sparse(non, Hamiltonian_segment_triu, U_re_snap, U_im_snap, N_site_si);
                propagate_snapshot(U_re_snap, U_im_snap, &U_re, &U_im, &work_re_si, &work_im_si, N_site_si);
            }//closing the prerun over t1
                
            // initialize the n_i * n_i propagator as a unit matrix
            unitmat(U_re,N_site_si);
            clearvec(U_im,N_site_si*N_site_si);

            log=fopen("NISE.log","a");
            fprintf(log,"Starting the computation of the tw propagators of seg %d, in sample %d\n", si, samples);
            time_now=log_time(time_now,log);
            fclose(log);
            /* Loop over the needed tw interval. propagate this one forward in time */
            for (ti=0;ti<N_tw;ti++){
                tj = t_ref+N_t1 + ti + 1; //+1 to ensure the +0 snapshot belongs to the t1 interval
                write_propagator_to_big_array(big_propagator_array_tw_re,U_re,N_tw,si,N_site_si, N_site_max, ti);
                write_propagator_to_big_array(big_propagator_array_tw_im,U_im,N_tw,si,N_site_si, N_site_max, ti);
                /* Read Hamiltonian */
                read_Hamiltonian(non,Hamil_i_e,H_traj,tj);
                // isolate the segment i with projection routine to obtain smaller matrix
                isolate_segment_Hamiltonian_triu(Hamil_i_e, Hamiltonian_segment_triu, H_indices_si, N_site_si, non);
                // Propagate segment i (~N_i^3 process)
                // propagate after writing to big array, such that first propagator is identity
                time_evolution_mat_non_sparse(non, Hamiltonian_segment_triu, U_re_snap, U_im_snap, N_site_si);
                propagate_snapshot(U_re_snap, U_im_snap, &U_re, &U_im, &work_re_si, &work_im_si, N_site_si);

            }//closing the prerun over tw
                 
            
            log=fopen("NISE.log","a");
            fprintf(log,"Starting the computation of the t2 propagators of seg %d, in sample %d\n", si, samples);
            time_now=log_time(time_now,log);
            fclose(log);
            /* Loop over the needed t2 interval, this naturally has overlap with the snapshots of tw. propagate this one forward in time */
            /* For the t2 interval, store the individual snapshots, rather than the compounded propagators */
            /* The actual propagation (combination of specific snapshot propagators) will be done in the 3d time loop */
            /* First snapshot properly stored (no unit matrix here as starting point here, but in triple time loop instead)*/
            for (ti=0;ti<times_N2;ti++){
                tj = t_ref+N_t1 + ti +1; //+1 to ensure the +0 snapshot belongs to the t1 interval
                /* Read Hamiltonian */
                read_Hamiltonian(non,Hamil_i_e,H_traj,tj);
                // isolate the segment i with projection routine to obtain smaller matrix
                isolate_segment_Hamiltonian_triu(Hamil_i_e, Hamiltonian_segment_triu, H_indices_si, N_site_si, non);
                // calculate the propagators of the individual snapshots, rather than calculating their product
                time_evolution_mat_non_sparse(non, Hamiltonian_segment_triu, U_re_snap, U_im_snap, N_site_si);
                write_propagator_to_big_array(big_propagator_array_t2_re,U_re_snap,times_N2,si,N_site_si, N_site_max, ti);
                write_propagator_to_big_array(big_propagator_array_t2_im,U_im_snap,times_N2,si,N_site_si, N_site_max, ti);
            }//closing the prerun over t2

            free(U_re);
            free(U_im);
            free(U_re_snap);
            free(U_im_snap);
            free(U_h_re);
            free(U_h_im);
            free(Hamiltonian_segment_triu);
            free(work_re_si);
            free(work_im_si);

        }//closing the single loop over segments

        log=fopen("NISE.log","a");
        fprintf(log,"Starting the double loop over the segments, in sample %d\n", samples);
        time_now=log_time(time_now,log);
        fclose(log);
        // double loop over segments to compute fourth order in each direction
        for (si=0;si<N_segments;si++){
            // si is the donor segment in this loop
            N_site_si = find_H_indices_segment(non->psites, H_indices_si, si, non);
            // find rho_si
            float *rho_0_si;
            rho_0_si = (float *)calloc(N_site_si*N_site_si,sizeof(float));
            isolate_segment_Hamiltonian(rho_0, rho_0_si, H_indices_si,N_site_si,non);
	    
            // nested segment loop
            for (sj=0;sj<N_segments;sj++){
                if (si != sj){
                    // sj is the acceptor segment in this loop
                    N_site_sj = find_H_indices_segment(non->psites, H_indices_sj, sj, non);
                    // define matrix J_ij left hand side index i, right hand side index j
                    // retrieve the appropriate inter-segment J block for segments i and j
                    float *Jij;
                    Jij = (float *)calloc(N_site_si*N_site_sj,sizeof(float));
                    isolate_coupling_block(J_full, Jij, N_site_si, N_site_sj, H_indices_si, H_indices_sj, non);	    
                    // compute R4 for 1 sample of the propagators

                    fourth_order_params p;
                    // fill the new struct
                    p.rho0_D = rho_0_si;
                    p.J = Jij;
                    p.integrated_response_tw = responses_4th_tw;
                    p.U_re_t1_array = big_propagator_array_t1_re;
                    p.U_im_t1_array = big_propagator_array_t1_im;
                    p.U_re_tw_array = big_propagator_array_tw_re;
                    p.U_im_tw_array = big_propagator_array_tw_im;
                    p.U_re_t2_array = big_propagator_array_t2_re;
                    p.U_im_t2_array = big_propagator_array_t2_im;

                    p.diagram_1 = diagram_1;
                    p.diagram_2 = diagram_2;
                    p.diagram_3 = diagram_3;
                    p.diagram_4 = diagram_4;

                    p.N_A = N_site_sj;
                    p.N_D = N_site_si;
                    p.N_t1 = N_t1;
                    p.N_tw = N_tw;
                    p.N_t2 = N_t2;
                    p.times_N2 = times_N2;
                    p.s_D = si;
                    p.s_A = sj;
                    p.N_segments = N_segments;
                    p.largest_segment_size = N_site_max;
                    p.non = non;

                    log=fopen("NISE.log","a");
                    fprintf(log,"Computing rate between segments %d and %d, in sample %d\n", si, sj, samples);
                    time_now=log_time(time_now,log);
                    fclose(log);

                    fourth_order_response_1_sample(&p);

                    // copy the diagrams at the after the 3d time loop (such that the final waiting time 2 D response is shown)
                    if (sj == 1 || si ==0){
                        add_diagrams(diagram_1, diagram_2, diagram_3, diagram_4, diagram_1_s01, diagram_2_s01, diagram_3_s01, diagram_4_s01, N_t1, N_t2);
                    }

                    free(Jij);
                }
            } // closing the nested segment loop
            free(rho_0_si);
        }//closing the double loop over the segments

    /* Update NISE log file */
    log=fopen("NISE.log","a");
    fprintf(log,"Finished the double segment loop (and computing the k4 matrix) in sample %d\n", samples);
    time_now=log_time(time_now,log);
    fclose(log);

    // compute the two rate matrices here, using either the TD-MCFRET information or the limiting response value as plateau
    compute_rate_from_4th_response(responses_4th_tw, rate_matrix_4th_I, rate_matrix_4th_II, rate_matrix_2nd, N_segments, N_tw, non, prefactor, samples);
       
    // write Rate Matrix 4th order to file
    int row, column;
    rate_matrix_4th_I_file = fopen("RateMatrix_4th_I.dat","w");
    rate_matrix_4th_II_file = fopen("RateMatrix_4th_II.dat","w");
    for (row=0;row<N_segments;row++){
        for (column=0;column<N_segments;column++){
            fprintf(rate_matrix_4th_I_file,"%e ", rate_matrix_4th_I[column + row *N_segments]);
            fprintf(rate_matrix_4th_II_file,"%e ", rate_matrix_4th_II[column + row *N_segments]);
        }
        fprintf(rate_matrix_4th_I_file,"\n");
        fprintf(rate_matrix_4th_II_file,"\n");
    }
    fclose(rate_matrix_4th_I_file); 
    fclose(rate_matrix_4th_II_file);
    
    int N_column = N_segments * N_segments;
    responses_4th_file = fopen("Responses_4th.dat","w");
    fprintf(responses_4th_file,"The response function between segments si * N_segments + sj, where si is the donor segment. First column gives waiting time in fs.\n");
    for (tw=0;tw<N_tw;tw++){
        fprintf(responses_4th_file,"%f ",tw*non->deltat);
            for (column=0;column<N_column;column++){
                fprintf(responses_4th_file,"%e ", prefactor * responses_4th_tw[column + tw * N_column] / (samples + 1));
            }
        fprintf(responses_4th_file,"\n");
    }
    fclose(responses_4th_file);

    }// closing the loop over the samples
    
    // print the diagrams for a single segment combination
    for (diag=0; diag<4; diag++){
        diagram_file = fopen(diagram_filenames[diag],"w");
        for (row=0;row<N_t2;row++){
            for (column=0;column<N_t1;column++){
                fprintf(diagram_file,"%e ", diagram_s01_list[diag][column + row *N_t1] / N_samples);
            }
            fprintf(diagram_file,"\n");
        }
        fclose(diagram_file);
    }

    //cleanup
    free(H_indices_si);
    free(H_indices_sj);
    free(rate_matrix_2nd);
    free(rate_matrix_4th_I);
    free(rate_matrix_4th_II);
    free(responses_4th_tw);
    free(big_propagator_array_t1_re);
    free(big_propagator_array_t1_im);
    free(big_propagator_array_t2_re);
    free(big_propagator_array_t2_im);
    free(big_propagator_array_tw_re);
    free(big_propagator_array_tw_im);
    free(diagram_1);
    free(diagram_2);
    free(diagram_3);
    free(diagram_4);
    free(diagram_1_s01);
    free(diagram_2_s01);
    free(diagram_3_s01);
    free(diagram_4_s01);
    // free(mu_xyz);
    free(Hamil_i_e);
    return;
}

void add_diagrams(float *diagram_1, float *diagram_2, float *diagram_3, float *diagram_4, float *diagram_1_s01, float *diagram_2_s01, float *diagram_3_s01, float *diagram_4_s01, int N_t1, int N_t2){
    // simply add the current diagrams to the specific ones to be written to a file for segments 0 and 1
    int i;
    for(i=0;i<N_t1*N_t2;i++){
        diagram_1_s01[i] += diagram_1[i];
        diagram_2_s01[i] += diagram_2[i];
        diagram_3_s01[i] += diagram_3[i];
        diagram_4_s01[i] += diagram_4[i];
    }
    return;
}
