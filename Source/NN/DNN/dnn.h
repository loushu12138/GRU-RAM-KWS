/*
 * Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __DNN_H__
#define __DNN_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn.h"
#include "dnn_weights.h"
#include "arm_nnfunctions.h"
#include "arm_math.h"
//#include "typedef.h"

#define SAMP_FREQ 16000
#define FRAME_SHIFT_MS 40
#define FRAME_SHIFT ((int16_t)(SAMP_FREQ * 0.001 * FRAME_SHIFT_MS))
#define NUM_FRAMES 25 
#define NUM_MFCC_COEFFS 10
#define MFCC_BUFFER_SIZE (NUM_FRAMES*NUM_MFCC_COEFFS)
#define FRAME_LEN_MS 40
#define FRAME_LEN ((int16_t)(SAMP_FREQ * 0.001 * FRAME_LEN_MS))

#define IN_DIM (NUM_FRAMES*NUM_MFCC_COEFFS)
#define DIM_HISTORY 16
#define DIM_INPUT 10
#define DIM_VEC 26
#define OUT_DIM 5



class DNN : public NN {

  public:
    DNN();
    //~DNN();
    void run_nn(float* in_data, float* out_data);
};
void gru(const float *gate_weights, const float *gate_bias,

         const float *candidate_weights, const float *candidate_bias,

         float *data, float *output_data);


#endif
