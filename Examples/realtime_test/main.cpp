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
 * Description: End-to-end example code for running keyword spotting on 
 * STM32F746NG development kit (DISCO_F746NG in mbed-cli). The example is 
 * derived from https://os.mbed.com/teams/ST/code/DISCO-F746NG_AUDIO_demo
 */

#include "kws_f746ng.h"
#include "plot_utils.h"
#include "LCD_DISCO_F746NG.h"
#include <ctime>
//float averaged_output[5]={0.1,0,0,0,0.9};
//int start;
void delay_msec(int msec)

{ 

	clock_t now = clock();

	while(clock()-now < msec);

}

LCD_DISCO_F746NG lcd;
Serial pc(USBTX, USBRX);
KWS_F746NG *kws;
Timer T;
//extern float gruou[5];
char lcd_output_string[64];
char output_class[5][8] = {"Silence", "Unknown","up","no","yes"};
// Tune the following three parameters to improve the detection accuracy
//  and reduce false positives
// Longer averaging window and higher threshold reduce false positives
//  but increase detection latency and reduce true positive detections.

// (recording_win*frame_shift) is the actual recording window size
int recording_win = 3; 
// Averaging window for smoothing out the output predictions
int averaging_window_len = 3;  
float detection_threshold = 90.0;  //in percent

void run_kws();

int main()
{
  pc.baud(9600);
  kws = new KWS_F746NG(recording_win,averaging_window_len);
  init_plot();
  kws->start_kws();

  T.start();
  //start = T.read_us();
  while (1) {
  /* A dummy loop to wait for the interrupts. Feature extraction and
     neural network inference are done in the interrupt service routine. */
    __WFI();
  }
}


/*
 * The audio recording works with two ping-pong buffers.
 * The data for each window will be tranfered by the DMA, which sends
 * sends an interrupt after the transfer is completed.
 */

// Manages the DMA Transfer complete interrupt.
void BSP_AUDIO_IN_TransferComplete_CallBack(void)
{
  arm_copy_q7((q7_t *)kws->audio_buffer_in + kws->audio_block_size*4, (q7_t *)kws->audio_buffer_out + kws->audio_block_size*4, kws->audio_block_size*4);
  if(kws->frame_len != kws->frame_shift) {
    //copy the last (frame_len - frame_shift) audio data to the start
    arm_copy_q7((q7_t *)(kws->audio_buffer)+2*(kws->audio_buffer_size-(kws->frame_len-kws->frame_shift)), (q7_t *)kws->audio_buffer, 2*(kws->frame_len-kws->frame_shift));
  }
  // copy the new recording data 
  for (int i=0;i<kws->audio_block_size;i++) {
    kws->audio_buffer[kws->frame_len-kws->frame_shift+i] = kws->audio_buffer_in[2*kws->audio_block_size+i*2];
  }
  run_kws();
  return;
}

// Manages the DMA Half Transfer complete interrupt.
void BSP_AUDIO_IN_HalfTransfer_CallBack(void)
{
  arm_copy_q7((q7_t *)kws->audio_buffer_in, (q7_t *)kws->audio_buffer_out, kws->audio_block_size*4);
  if(kws->frame_len!=kws->frame_shift) {
    //copy the last (frame_len - frame_shift) audio data to the start
    arm_copy_q7((q7_t *)(kws->audio_buffer)+2*(kws->audio_buffer_size-(kws->frame_len-kws->frame_shift)), (q7_t *)kws->audio_buffer, 2*(kws->frame_len-kws->frame_shift));
  }
  // copy the new recording data 
  for (int i=0;i<kws->audio_block_size;i++) {
    kws->audio_buffer[kws->frame_len-kws->frame_shift+i] = kws->audio_buffer_in[i*2];
  }
  run_kws();
  return;
}

void run_kws()
{
  kws->extract_features();    //extract mfcc features
  kws->classify();	      //classify using dnn
  kws->average_predictions();//
  plot_mfcc();
  plot_waveform();
  int max_ind = kws->get_top_class(kws->averaged_output);
  // print to pc
  //int end = T.read_us();
  // extern float gruou[10];
   //extern int screen_size_y;
   //extern int mfcc_update_counter;
  // extern float gruou1[10];
  // extern float gruou2[10];
  //pc.printf("\033[2J\033[1H");
  //printf("Total time : %d us\r\n",end-start);
 // printf("Detected %s (%d%%)\r\n",output_class[max_ind],(int)(kws->averaged_output[max_ind]*100));
  //for(int i=50;i<60;i++) {
//  printf("%d\t",screen_size_y);
//  printf("\r\n");
//  printf("%d\t",mfcc_update_counter);
 //printf("%d\t",kws->mfcc8[i]);
  //}
  //printf("\r\n");
//   for(int i=0;i<10;i++) {
//  printf("%.3f\t",gruou1[i]);
//   }
//   printf("\r\n");
  //for(int i=50;i<60;i++) {
 //printf("%.3f\t",gruou2[i]);
// printf("%d\t",kws->mfcc9[i]);
 // }
  //printf("\r\n");
  //printf("\r\n");
   


  if(kws->averaged_output[max_ind]>detection_threshold/100)
  //int max_ind = kws->get_top_class(kws->output);
  //sprintf(lcd_output_string,"%d%% %s",(int)(kws->averaged_output[max_ind]*100),output_class[max_ind]);
  //if(kws->averaged_output[max_ind]>detection_threshold*128/100)
  //sprintf(lcd_output_string,"%d%% %s",((int)kws->averaged_output[max_ind]*100/128),output_class[max_ind]);
  
  //  extern float gruou[250];
  //   for(int i=240;i<250;i++){
  //  sprintf(lcd_output_string,"%f",kws->averaged_output[i]);
  //sprintf(lcd_output_string,"%f",kws->averaged_output[max_ind]);
   //sprintf(lcd_output_string,"%d",kws->recording_win);
  //  sprintf(lcd_output_string,"%f",gruou[i]);

  // lcd.ClearStringLine(8);
  // lcd.DisplayStringAt(0, LINE(8), (uint8_t *) lcd_output_string, CENTER_MODE);//uint8_t
  //  delay_msec(200);
  //  }
  // sprintf(lcd_output_string,"%d",max_ind);
  // lcd.ClearStringLine(8);
  // lcd.DisplayStringAt(0, LINE(8), (uint8_t *) lcd_output_string, CENTER_MODE); 
  //  delay_msec(200);
  // sprintf(lcd_output_string,"%f",kws->averaged_output[max_ind]);
  // lcd.ClearStringLine(8);
  // lcd.DisplayStringAt(0, LINE(8), (uint8_t *) lcd_output_string, CENTER_MODE); 
  //  delay_msec(200);
  sprintf(lcd_output_string,"%d%% %s",(int)(kws->averaged_output[max_ind]*100),output_class[max_ind]);
  lcd.ClearStringLine(8);
  lcd.DisplayStringAt(0, LINE(8), (uint8_t *) lcd_output_string, CENTER_MODE); 
   //delay_msec(200);
 }

