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
 * Description: Example code for running keyword spotting on Cortex-M boards
 */

//#include "kws_ds_cnn.h"
#include "kws_dnn.h"
#include "wav_data.h"
#include "plot_utils.h"
#include "LCD_DISCO_F746NG.h" 
LCD_DISCO_F746NG lcd;

int16_t audio_buffer[16000]=WAVE_DATA;
char lcd_output_string[64];

Timer T;
Serial pc(USBTX, USBRX);

int main()
{
  char output_class[12][8] = {"Silence", "Unknown","yes","no","up","down","left","right","on","off","stop","go"};
  KWS_DS_CNN kws(audio_buffer);
  //KWS_DNN kws(audio_buffer);

  T.start();
  int start=T.read_us();
  kws.extract_features(); //extract mfcc features
  kws.classify();	  //classify using dnn
  int end=T.read_us();
  T.stop();
  int max_ind = kws.get_top_class(kws.output);
  pc.printf("Total time : %d us\r\n",end-start);
  printf("Detected %s (%d%%)\r\n",output_class[max_ind],((int)kws.output[max_ind]*100/128));
  //extern q7_t gruout[100];
   //for(int i=0;i<100;i++){
  //printf("%d",gruout[i]);}
   plot_mfcc();
   plot_waveform();
//   extern q7_t gruout[100];
//   for(int i=0;i<100;i++){
//   sprintf(lcd_output_string,"%d",gruout[i]);
//  
//  }
  lcd.ClearStringLine(8);
  lcd.DisplayStringAt(0, LINE(8), (uint8_t *) lcd_output_string, CENTER_MODE);

  return 0; 
}
