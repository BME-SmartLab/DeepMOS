# DeepMOS
Deep learning approach for automatic MOS prediction.

Our solution is a modified version of the following GitHub repository: https://github.com/nii-yamagishilab/mos-finetune-ssl

The instructions of dataset access can be found in the original repository. 

# Environment

You can run our solution in a Docker container. Please set the host and container directories in the `run_docker.sh` file, and run it. In the container, please run `install_in_docker.sh` before running the scripts. 

# Training (transfer learning from pretrained Wav2Vec2 models)

The training can be performed for the MAIN track of VoiceMOS Challange 2022 in the following way:

`python deepmos.py --datadir <FOLDER FOR DATA> --fairseq_base_model <PRETRAINED WAV2VEC2 MODEL, E.G. libri960_big.pt> --nunits <LSTM SIZE> --nlstm <LSTM LAYERS> --dropout <DROPOUT RATE> --batch_size <BATCH SIZE>`

There are additional settings in deepmos.py, which you can see by running `python deepmos.py -h`.

For the OOD track you can run the same training scripcs. In case of OOD we recommend to train a model with the MAIN track model first, and then train that model further with the OOD data. 

If you run out of GPU memory, you may need to decrease the `batch_size`. E.g. `batch_size=4` works with NVIDIA V100 32GB GPUs. 

# Predictions

The training scripts results in JSON files in the checkpoints folder. In order to run the predictions, the following command should be run:

`python predict.py --datadir <FOLDER FOR DATA> --json <JSON FILE OF THE TRAINED MODEL> --outfile <FILE FOR THE PREDICTIONS>` 

## Acknowledgements

If you are using our solution, thank you for citeing the corresponding paper (submitted to Interspeech 2022):

B. Gyires-Tóth, Cs. Zainkó (2022). Improving Self-Supervised Learning-based MOS Prediction Networks. arXiv preprint arXiv:2204.11030.

URL: https://arxiv.org/abs/2204.11030

## License

BSD 3-Clause License

Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
