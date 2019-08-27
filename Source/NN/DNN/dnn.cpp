

#include "dnn.h"
//float gruou1[10];

int relu( float* dst, int length)

{ for (int i = 0; i < length; ++i) {
  if(dst[i]>0)
  continue;
  else
   dst[i]=0;
  
		//dst[i] = max((float)0., dst[i]);
}
  return 0;
}

const float gate_weights[26*32]=GATE_WEIGHTS;
const float gate_bias[32]=GATE_BIAS;
const float candidate_weights[26*16]=CANDIDATE_WEIGHTS;
const float candidate_bias[16]=CANDIDATE_BIAS;
const float last_weights[16*5]=LAST_WEIGHTS;
const float last_bias[5]=LAST_BIAS;


// const float ip1_wt[250*16]=IP1_WT;
// const float ip1_bias[16]=IP1_BIAS;
// const float ip2_wt[16*16]=IP2_WT;
// const float ip2_bias[16]=IP2_BIAS;
// const float ip3_wt[16*16]=IP3_WT;
// const float ip3_bias[16]=IP3_BIAS;
// const float ip4_wt[16*5]=IP4_WT;
// const float ip4_bias[5]=IP4_BIAS;

void gru(const float *gate_weights, const float *gate_bias,

         const float *candidate_weights, const float *candidate_bias,

         float *data, float *output_data) {

  float state[16];

  for (int i = 0; i < 16; i++) {

    state[i] = 0;

  }

  float r_t[16];

  float z_t[16];

  float h_t[16];

  for (int l = 0; l < 25; l++) {

    for (int o = 0; o < 16; o++) {

      r_t[o] = 0;

      z_t[o] = 0;

      for (int i = 0; i < 16; i++) {

        r_t[o] += state[i] * gate_weights[(i + 10) * 2 * 16 + o];

        z_t[o] += state[i] * gate_weights[(i + 10) * 2 * 16 + 16 + o];

      }
    
    //   if (l==3) {
    //    for (int i =0;i<5;i++){
    //    gruou[i] = z_t[i];
    //   }
    //  }
      

      for (int i = 0; i < 10; i++) {

        r_t[o] += data[i + l * 10] * gate_weights[i * 2 * 16 + o];

        z_t[o] += data[i + l * 10] * gate_weights[i * 2 * 16 + 16 + o];

      }//lianghua2
      // if (l==0) {
      // for (int i =0;i<5;i++){
       //gruou[i] = z_t[i];
      //}
      //}

      r_t[o] = 1. / (1. + exp(-(r_t[o] + gate_bias[o])));

      z_t[o] = 1. / (1. + exp(-(z_t[o] + gate_bias[16 + o])));
      // if (l==1) {
      //   for (int i =0;i<5;i++){
      //   gruou[i] = z_t[i];
      //  }
      // }
    }
    // if (l==13) {
    //    for (int i =0;i<5;i++){
    //    gruou[i] =z_t[i]; //r_t[i];
    //   }
    //  }

    for (int o = 0; o < 16; o++) {

      h_t[o] = 0;

      for (int i = 0; i < 16; i++) {

        h_t[o] += state[i] * r_t[i] * candidate_weights[(i + 10) * 16 + o];//lianghua3

      }

      for (int i = 0; i < 10; i++) {

        h_t[o] += data[i + l * 10] * candidate_weights[i * 16 + o];//lianghua4

      }

      h_t[o] += candidate_bias[o];//lianghua5

      h_t[o] = (1. - exp(-2 * h_t[o])) / (1. + exp(-2 * h_t[o]));

    }
    // if (l==2) {
    //    for (int i =0;i<5;i++){
    //    gruou[i] =state[i]; //r_t[i];
    //   }
    //  }

    for (int o = 0; o < 16; o++) {

      state[o] = z_t[o] * state[o] + (1 - z_t[o]) * h_t[o];//lianghua6

    //   if (l==24) {
    //   for (int i =0;i<5;i++){
    //    gruou[i] =state[i]; //r_t[i];
    //   }
    //  }

      //output_data[o + l * 16] = state[o];

    }
      
  }
  for (int o = 0; o < 16; o++) {
       output_data[o] = state[o];}
}

DNN::DNN()
{
  
  frame_len = FRAME_LEN;
  frame_shift = FRAME_SHIFT;
  num_mfcc_features = NUM_MFCC_COEFFS;
  num_frames = NUM_FRAMES;
  num_out_classes = OUT_DIM;
 
}



void DNN::run_nn (float* in_data, float* out_data)
{  //ip1

  //  float ip1out[16];
  // for (int i =0;i<16;i++){
  //     ip1out[i] =0;
  //     for(int o=0;o<250;o++){
  //       ip1out[i]+=in_data[o]*ip1_wt[o*16+i];
  //     }
  //     ip1out[i]+=ip1_bias[i];
  //   }
  //   relu(ip1out,16);
  // //ip2
  //  float ip2out[16];
  // for (int i=0;i<16;i++){
  //     ip2out[i]=0;
  //     for(int o=0;o<16;o++){
  //       ip2out[i]+=ip1out[o]*ip2_wt[o*16+i];
  //     }
  //     ip2out[i]+=ip2_bias[i];
  //  }
  //   relu(ip2out,16);
  //  //ip3
  //   float ip3out[16];
  // for (int i=0;i<16;i++){
  //     ip3out[i]=0;
  //     for(int o=0;o<16;o++){
  //       ip3out[i]+=ip2out[o]*ip3_wt[o*16+i];
  //     }
  //     ip3out[i]+=ip3_bias[i];
  // }
  //   relu(ip3out,16);
  // //ip4
  // for (int i=0;i<5;i++){
  //     out_data[i]=0;
  //     for(int o=0;o<16;o++){
  //       out_data[i]+=ip3out[o]*ip4_wt[o*5+i];
  //     }
  //     out_data[i]+=ip4_bias[i];
  // }
 
  
  
  float output_data[16];
  gru(gate_weights, gate_bias, candidate_weights, candidate_bias, in_data, output_data);

   for (int o=0; o<5; o++){
     out_data[o]=0;
     for(int i=0; i<16; i++){
       out_data[o] +=output_data[i]*last_weights[i*5+o];
       
     }
     out_data[o] +=last_bias[o];
   }
 //for (int i =240;i<250;i++){
      // gruou1[i-240] = in_data[i];}
 
 }


