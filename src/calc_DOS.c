#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <fftw3.h>
#include "omp.h"
#include "types.h"
#include "NISE_subs.h"
#include "propagate.h"
#include "calc_DOS.h"
#include "1DFFT.h"
#include "project.h"

void calc_DOS(t_non *non){
  // Initialize variables
  float *re_S_1,*im_S_1; // The first-order response function
  float *Hamil_i_e;
  // Aid arrays
  float *vecr, *veci;
  /* Floats */
  float shift1;
  /* 1D Fourier transform */
  fftw_complex *fftIn,*fftOut;
  fftw_plan fftPlan;

  /* File handles */
  FILE *H_traj,*mu_traj;
  FILE *C_traj;
  FILE *outone,*log;
  FILE *Cfile;

  /* Integers */
  int nn2;
  int itime,N_samples;
  int samples;
  int x,ti,tj,i,j;
  int t1,fft;
  int elements;
  int cl,Ncl;
  int pro_dim, ip;

  /* Time parameters */
  time_t time_now,time_old,time_0;
  /* Initialize time */
  time(&time_now);
  time(&time_0);
  shift1=(non->max1+non->min1)/2;
  printf("Frequency shift %f.\n",shift1);
  non->shifte=shift1;

  /* Check if projection keyword is defined and select number of subsets in */
  /* projection */
  pro_dim=project_dim(non);

  // Allocate memory
  re_S_1=(float *)calloc(non->tmax*pro_dim,sizeof(float));
  im_S_1=(float *)calloc(non->tmax*pro_dim,sizeof(float));
  nn2=non->singles*(non->singles+1)/2;
  Hamil_i_e=(float *)calloc(nn2,sizeof(float));

  /* Open Trajectory files */
  H_traj=fopen(non->energyFName,"rb");
  if (H_traj==NULL){
    printf("Hamiltonian file not found!\n");
    exit(1);
  }

  //printf("A\n");
  /* Open file with cluster information if appicable */
  if (non->cluster!=-1){
    Cfile=fopen("Cluster.bin","rb");
    if (Cfile==NULL){
      printf("Cluster option was activated but no Cluster.bin file provided.\n");
      printf("Please, provide cluster file or remove Cluster keyword from\n");
      printf("input file.\n");
      exit(0);
    }
    Ncl=0; // Counter for snapshots calculated
  }

  // Here we want to call the routine for checking the trajectory files
  control(non);

  itime=0;
  // Do calculation
  N_samples=(non->length-non->tmax1-1)/non->sample+1;
  if (N_samples>0) {
    printf("Making %d samples!\n",N_samples);
  } else {
    printf("Insufficient data to calculate spectrum.\n");
    printf("Please, lower max times or provide longer\n");
    printf("trajectory.\n");
    exit(1);
  }

  if (non->end==0) non->end=N_samples;
  if (non->end>N_samples){
    printf(RED "Endpoint larger than number of samples was specified.\n" RESET);
    printf(RED "Endpoint was %d but cannot be larger than %d.\n" RESET,non->end,N_samples);
    exit(0);
  }

  log=fopen("NISE.log","a");
  fprintf(log,"Begin sample: %d, End sample: %d.\n",non->begin,non->end);
  fclose(log);

  vecr=(float *)calloc(non->singles*non->singles,sizeof(float));	
  veci=(float *)calloc(non->singles*non->singles,sizeof(float));

  /* Read coupling */
  if (!strcmp(non->hamiltonian,"Coupling")){
    C_traj=fopen(non->couplingFName,"rb");
    if (C_traj==NULL){
      printf("Coupling file not found!\n");
      exit(1);
    }
    if (read_He(non,Hamil_i_e,C_traj,-1)!=1){
      printf("Coupling trajectory file to short, could not fill buffer!!!\n");
      exit(1);
    }
    fclose(C_traj);
  }

  /* Loop over samples */
  for (samples=non->begin;samples<non->end;samples++){

    /* Calculate linear response */   
    ti=samples*non->sample;
    if (non->cluster!=-1){
      if (read_cluster(non,ti,&cl,Cfile)!=1){
	printf("Cluster trajectory file to short, could not fill buffer!!!\n");
	printf("ITIME %d\n",ti);
	exit(1);
      }
      //      printf("%d\n",cl);
      // Configuration belong to cluster
      if (non->cluster==cl){
	Ncl++;
      }
    }
    if (non->cluster==-1 || non->cluster==cl){
      // Initialize time-evolution operator
      unitmat(vecr,non->singles);
      // Do projection on selected sites if asked
      if (non->Npsites==1){
        for (i=0;i<non->singles;i++){
          j=i*non->singles;
          projection(vecr+j,non);
        }
      }
      clearvec(veci,non->singles*non->singles);

      // Loop over delay
      for (t1=0;t1<non->tmax;t1++){
	tj=ti+t1;
	/* Read Hamiltonian */
	if (!strcmp(non->hamiltonian,"Coupling")){
          if (read_Dia(non,Hamil_i_e,H_traj,tj)!=1){
            printf("Hamiltonian trajectory file to short, could not fill buffer!!!\n");
            exit(1);
          }
        } else {
	  if (read_He(non,Hamil_i_e,H_traj,tj)!=1){
	    printf("Hamiltonian trajectory file to short, could not fill buffer!!!\n");
	    exit(1);
  	  }
        }
	
	// Find response

        /* Do projection on selected sites if asked and calculate response function */
        if (non->Npsites<2){
          /* Find response without projection option */
            c_calc_DOS(re_S_1,im_S_1,t1,non,vecr,veci);
          } 
	// else if (non->Npsites<non->singles){ }
        else {
            /* Find response with projection on multiple segments */
            for (ip=0;ip<pro_dim;ip++){
	      c_calc_DOS_multi_project(re_S_1+non->tmax*ip,im_S_1+non->tmax*ip,t1,non,vecr,veci,ip);
            }
         }
	// Loop over vectors to propagate
	       propagate_matrix(non,Hamil_i_e,vecr,veci,1,samples,t1);
      }
    } // Cluster loop
  
    // Log time
    log=fopen("NISE.log","a");
    fprintf(log,"Finished sample %d\n",samples);
          
    time_now=log_time(time_now,log);
    fclose(log);
  }

  free(vecr);
  free(veci);
  free(Hamil_i_e);

  // The calculation is finished, lets write output
  log=fopen("NISE.log","a");
  fprintf(log,"Finished Calculating Response!\n");
  fprintf(log,"Writing to file!\n");  
  fclose(log);

  if (non->cluster!=-1){
    printf("Of %d samples %d belonged to cluster %d.\n",samples,Ncl,non->cluster);
    //    printf("Changing normalization to number of samples used.");
    //    samples=Ncl;
    if (samples==0){ // Avoid dividing by zero
      samples=1;
    }
  }

  fclose(H_traj);
  if (non->cluster!=-1){
    fclose(Cfile);
  }

  /* Save time domain response */
  outone=fopen("TD_DOS.dat","w");
  for (t1=0;t1<non->tmax1;t1+=non->dt1){
    fprintf(outone,"%f ",t1*non->deltat);
    for (ip=0;ip<pro_dim;ip++){
      fprintf(outone,"%e %e ",re_S_1[t1+ip*non->tmax]/samples,im_S_1[t1+ip*non->tmax]/samples);
    }
    fprintf(outone,"\n");
  }
  fclose(outone);

  /* Do Forier transform and save */
  do_1DFFT(non,"DOS.dat",re_S_1,im_S_1,samples);

  free(re_S_1),free(im_S_1);

  printf("----------------------------------------------\n");
  printf(" DOS calculation succesfully completed\n");
  printf("----------------------------------------------\n\n");

  return;
}	

void c_calc_DOS(float *re_S_1,float *im_S_1,int t1,t_non *non,float *cr,float *ci){
  int i,j;
  // Take trace of time-evolution operator
  for (i=0;i<non->singles;i++){
    j=i+i*non->singles;
    re_S_1[t1]+=cr[j];
    im_S_1[t1]+=ci[j];
  }
  return;
}

void c_calc_DOS_multi_project(float *re_S_1,float *im_S_1,int t1,t_non *non,float *cr,float *ci,int ip){
  int i,j;
  // Take trace of time-evolution operator
  for (i=0;i<non->singles;i++){
    if (non->psites[i]==ip){
      j=i+i*non->singles;
      re_S_1[t1]+=cr[j];
      im_S_1[t1]+=ci[j];
    }
  }
  return;
}

