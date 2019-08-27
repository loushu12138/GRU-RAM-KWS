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

/*
 * Description: Keyword spotting example code using MFCC feature extraction
 * and neural network. 
 */

#include "kws.h"
#include "wav_data.h"
#include "up.h"
#include "no.h"
#include "yes.h"
//#include "LCD_DISCO_F746NG.h"

float gruou2[10];



//softmax function
void softmax(float *x, int row, int column)
{
    for (int j = 0; j < row; ++j)
    {
        float max = 0.0;
        float sum = 0.0;
        for (int k = 0; k < column; ++k)
            if (max < x[k + j*column])
                max = x[k + j*column];
        for (int k = 0; k < column; ++k)
        {
            x[k + j*column] = exp(x[k + j*column] - max);    // prevent data overflow
            sum += x[k + j*column];
        }
        for (int k = 0; k < column; ++k)
         x[k + j*column] /= sum;
    }
}   //row*column
//copy(float) function
void copy_float(
  const float * pSrc,
        float * pDst,
        unsigned int blockSize)
{
  unsigned int blkCnt;                               /* Loop counter */
#if defined (ARM_MATH_LOOPUNROLL)

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = A */

    /* read 4 samples at a time */
    write_q7x4_ia (&pDst, read_q7x4_ia ((float **) &pSrc));

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Loop unrolling: Compute remaining outputs */
  blkCnt = blockSize % 0x4U;

#else

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

#endif /* #if defined (ARM_MATH_LOOPUNROLL) */

  while (blkCnt > 0U)
  {
    /* C = A */

    /* Copy and store result in destination buffer */
    *pDst++ = *pSrc++;

    /* Decrement loop counter */
    blkCnt--;
  }
}

KWS::KWS()
{
}

KWS::~KWS()
{
  delete mfcc;
  delete mfcc_buffer;
  delete mfcc8;
  delete mfcc9;
  delete output;
  delete predictions;
  delete averaged_output;
}

void KWS::init_kws()
{
  num_mfcc_features = nn->get_num_mfcc_features();
  num_frames = nn->get_num_frames();
  frame_len = nn->get_frame_len();
  frame_shift = nn->get_frame_shift();
  int mfcc_dec_bits = nn->get_in_dec_bits();
  num_out_classes = nn->get_num_out_classes();
  mfcc = new MFCC(num_mfcc_features, frame_len, mfcc_dec_bits);
  mfcc_buffer = new float[num_frames*num_mfcc_features];
  mfcc8 = new q7_t[num_frames*num_mfcc_features];
  mfcc9 = new float[num_frames*num_mfcc_features];
  output = new float[num_out_classes];
  averaged_output = new float[5];
  predictions = new float[sliding_window_len*num_out_classes];
  audio_block_size = recording_win*frame_shift;
  audio_buffer_size = audio_block_size + frame_len - frame_shift;
}
void KWS::extract_features() 
{
  if(num_frames>recording_win) {
    //move old features left 
  memmove(mfcc_buffer,mfcc_buffer+(recording_win*num_mfcc_features),4*(num_frames-recording_win)*num_mfcc_features);}
  //compute features only for the newly recorded audio
  //const int16_t audio_buffer[16000]=NO;
  int32_t mfcc_buffer_head =(num_frames-recording_win)*num_mfcc_features; 
  for (uint16_t f = 0; f <recording_win; f++) 
  {
    mfcc->mfcc_compute(audio_buffer+(f*frame_shift),&mfcc_buffer[mfcc_buffer_head]);

    mfcc_buffer_head += num_mfcc_features;
  }
  for(int i=0;i<250;i++)
  { mfcc9[i]=mfcc_buffer[i];
    mfcc9[i] *= (0x1<<2);
    mfcc9[i] = round(mfcc9[i]); 
    if(mfcc9[i] >= 127)
      mfcc8[i] = 127;
    else if(mfcc9[i] <= -128)
      mfcc8[i] = -128;
    else
      mfcc8[i] = mfcc9[i];
  }
  
  
  for (int i = 240; i < 250; i++) {
    gruou2[i-240] = mfcc_buffer[i];
  }
  // gruou2[0]=num_frames;
  // gruou2[1]=recording_win;
  // gruou2[2]=num_mfcc_features;
  // gruou2[3]=frame_shift;
  // gruou2[4]=frame_len;
}

void KWS::classify()
{  // for (int i =0;i<5;i++){
//     gruou[i] = mfcc_buffer[i];
//  }
  nn->run_nn(mfcc_buffer, output);
  // Softmax new
  //arm_softmax_q7(output,num_out_classes,output);
  softmax(output,1,5);
  //for (int i =0;i<5;i++){
    //gruou[i] = output[i];}

}
int KWS::get_top_class(float* prediction)
{
  int max_ind=0;
  float max_val=0;
  for(int i=0;i<num_out_classes;i++) {
    if(max_val<prediction[i]) {
      max_val = prediction[i];
      max_ind = i;
    }    
  }
  return max_ind;
}

void KWS::average_predictions()
{
  // shift the old predictions left
  copy_float((float *)(predictions+num_out_classes), (float *)predictions, (sliding_window_len-1)*num_out_classes);//sliding_window_len=1
  // add new predictions at the end
  copy_float((float *)output, (float *)(predictions+(sliding_window_len-1)*num_out_classes), num_out_classes);
   //copy_float((float *)output, (float *)(predictions), 5);
  //compute averages
  float sum;
  for(int j=0;j<num_out_classes;j++) {
    sum=0;
    for(int i=0;i<sliding_window_len;i++) 
      sum += predictions[i*num_out_classes+j];
    averaged_output[j] = (float)(sum/sliding_window_len);
  }   

}
  
