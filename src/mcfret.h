#ifndef _MCFRET_ /* ignore */
#define _MCFRET_

typedef struct {
    float *rho0_D;
    float *J;
    float *integrated_response_tw;
    float *U_re_t1_array;
    float *U_im_t1_array;
    float *U_re_tw_array;
    float *U_im_tw_array;
    float *U_re_t2_array;
    float *U_im_t2_array;
    float *diagram_1;
    float *diagram_2;
    float *diagram_3;
    float *diagram_4;

    int N_A;
    int N_D;
    int N_t1;
    int N_tw;
    int N_t2;
    int times_N2;
    int s_D;
    int s_A;
    int N_segments;
    int largest_segment_size;
    
    t_non *non;
} fourth_order_params;

typedef struct {
    // the four complex intermediate products that can be reused in the triple time loop
    float *intermediate_product_1_re; float *intermediate_product_1_im;
    float *intermediate_product_2_re; float *intermediate_product_2_im;
    float *intermediate_product_3_re; float *intermediate_product_3_im;
    float *intermediate_product_4_re; float *intermediate_product_4_im;

    // t1 dependent products with the density matrix
    float *UDh_rho_J_UA_re_t1; float *UDh_rho_J_UA_im_t1;
    float *UAh_Jh_rho_UD_re_t1; float *UAh_Jh_rho_UD_im_t1;

    // tw dependent matrix products
    float *Jh_UDh_tw_re;  float *Jh_UDh_tw_im;
    float *Jh_UD_tw_re;   float *Jh_UD_tw_im;
    float *UD_tw_re;      float *UD_tw_im;
    float *UA_Jh_tw_re;   float *UA_Jh_tw_im;
    float *UAh_Jh_tw_re;  float *UAh_Jh_tw_im;
    float *J_UAh_Jh_tw_re; float *J_UAh_Jh_tw_im;
    float *UA_tw_re;      float *UA_tw_im;

    float *J_zeros;
} fourth_order_workspace;

void mcfret(t_non *non);
void density_matrix(float *density_matrix, float *Hamiltonian_i,t_non *non,int segments, float *partition_functions);
void mcfret_autodetect(t_non *non, float treshold);
void mcfret_response_function(float *re_S_1,float *im_S_1,t_non *non,int emission, float *ave_vecr);
void mcfret_coupling(float *J,t_non *non);
void mcfret_energy(float *E,t_non *non,int segments, float *ave_vecr,float *energy_cor);
void mcfret_rate(float *rate_matrix,float *coherence_matrix,int segments,float *re_Abs,float *im_Abs,float *re_Emi,float *im_Emi,float *J,t_non *non);
void mcfret_validate(t_non *non);
void mcfret_eigen(t_non *non,float *rate_matrix,float *re_e,float *im_e,float *vl,float *vr,int segments,float *energy_cor);
void mcfret_analyse(float *E,float *rate_matrix,t_non *non,int segments);
void mcfret_response_function_sub(float *re_S_1,float *im_S_1,int t1,t_non *non,float *cr,float *ci);
void segment_matrix_mul(float *rA,float *iA,float *rB,float *iB,float *rC,float *iC,int *psites,int segments,int si,int sj,int sk,int N);
float trace_rate(float *matrix,int N);
void write_matrix_to_file_float(char fname[],float *matrix,int N);
void read_response_from_file(char fname[],float *re_R,float *im_R,int N,int tmax);
void triangular_on_square(float *T,float *S,int N);
void average_density_matrix(float *ave_den_mat,t_non *non);
void compute_all_traces_4th_order(float *rho_0,float *J_full,t_non *non);
void compute_UJJU(float *UJJU_re, float *JJ, float *U_re, float *U_im, int N_i,int sj);
void compute_JrhoJ(float *Jij_rho_jj_Jji, float* Jij, float *rho_0_sj, int N_i, int N_j, int sj);
void compute_rhoJJ(float *rho_ii_JijJji, float *JijJji, float* Jij, float *rho_0_si, int N_i, int N_j, int sj);
int find_H_indices_segment(int *psites, int *H_indices_si,int si, t_non *non);
void isolate_segment_Hamiltonian_triu(float *Hamiltonian_full_triu, float *Hamiltonian_segment_triu, int *H_indices_si, int N_i, t_non *non);
void isolate_segment_Hamiltonian(float *Hamiltonian_full, float *Hamiltonian_segment, int *H_indices_si, int N_i, t_non *non);
void isolate_coupling_block(float *J_full, float *J_ij, int N_i, int N_j, int *H_indices_si, int *H_indices_sj, t_non *non);
float matrix_mul_traced(float *A, float *Bi, int N_i);
void clearvec_int(int *a, int N);
float matrix_sum(float *matrix,int N);


void mcfret_propagation_segmented(float *re_S_1,float *im_S_1,t_non *non);
void mcfret_response_function_sub_segments(float *re_S_1,float *im_S_1,int t1,t_non *non,float *cr,float *ci, int *H_indices_si,int N_i);


void propagate_vector_segments(t_non *non,float * Hamil_i_e,float *vecr,float *veci,int sign,int samples,int display, int N_i);
void propagate_matrix_segments(t_non *non,float * Hamil_i_e,float *vecr,float *veci,int sign,int samples,int display, int N_i);
void propagate_vec_coupling_S_segments(t_non* non, float* Hamiltonian_i, float* cr, float* ci, int m, int sign, int N_i);

void hermitian_conjugate(float *A_re, float *A_im, float *hermi_re, float *hermi_im, int N1, int N2);
void mcfret_rate_from_abs(float *rate_matrix,float *coherence_matrix,int segments,float *re_Abs,float *im_Abs, float *rho_0,float *J,t_non *non);

void write_propagator_to_big_array(float *big_array, float *propagator, int sample_length, int si, int N_site_si, int largest_segment_size, int ti);
void read_propagator_from_big_array(float *big_array, float *propagator, int sample_length, int si, int N_site_si, int largest_segment_size, int ti);
void propagate_snapshot(float *U_snap_re, float *U_snap_im, float **U_comp_re, float **U_comp_im, float **temp_re, float **temp_im, int N);
void complex_matrix_product(float *A_re, float *A_im, float *B_re, float *B_im, float *C_re, float *C_im,int N1,int N2,int N3);
void compute_UDh_rho_J_UA_t1(float *UDh_rho_J_UA_re_t1,
                             float *UDh_rho_J_UA_im_t1,
                             float *UAh_Jh_rho_UD_re_t1,
                             float *UAh_Jh_rho_UD_im_t1,
                             fourth_order_params *p);
void compute_rate_from_4th_response(float *responses_4th_tw, float *rate_matrix_4th_I, float *rate_matrix_4th_II, float *rate_matrix_2nd, int N_segments, int N_tw, t_non *non, float prefactor, int samples);
void full_4th_order_main(float *rho_0,float *J_full,t_non *non);
void fourth_order_response_1_sample(fourth_order_params *p);
void add_diagrams(float *diagram_1, float *diagram_2, float *diagram_3, float *diagram_4, float *diagram_1_s01, float *diagram_2_s01, float *diagram_3_s01, float *diagram_4_s01, int N_t1, int N_t2);
void compute_4_intermediate_products(fourth_order_workspace *ws, fourth_order_params *p);

#endif /* _MCFRET_ */
