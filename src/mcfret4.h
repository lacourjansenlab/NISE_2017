#ifndef _MCFRET4_ /* ignore */
#define _MCFRET4_

void compute_all_traces_4th_order(float *rho_0,float *J_full,t_non *non);
void compute_UJJU(float *UJJU_re, float *JJ, float *U_re, float *U_im, int N_i,int sj);
void compute_JrhoJ(float *Jij_rho_jj_Jji, float* Jij, float *rho_0_sj, int N_i, int N_j, int sj);
void compute_rhoJJ(float *rho_ii_JijJji, float *JijJji, float* Jij, float *rho_0_si, int N_i, int N_j, int sj);

void write_propagator_to_big_array(float *big_array, float *propagator, int sample_length, int si, int N_site_si, int largest_segment_size, int ti);
void read_propagator_from_big_array(float *big_array, float *propagator, int sample_length, int si, int N_site_si, int largest_segment_size, int ti);

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



#endif /* _MCFRET4_ */