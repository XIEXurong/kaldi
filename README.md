This is the modified version of Kaldi toolkit for Bayesian adaptation approaches proposed in the published paper:
	Xurong Xie, Xunying Liu, Tan Lee, & Lan Wang (2021). Bayesian Learning for Deep Neural Network Adaptation. IEEE/ACM Transactions on Audio, Speech, and Language Processing. (An updated version with appendix is available in arXiv https://arxiv.org/abs/2012.07460)
This is originally proposed in the paper:
	Xurong Xie, Xunying Liu, Tan Lee, Shoukang Hu, & Lan Wang (2019, May). BLHUC: Bayesian learning of hidden unit contributions for deep neural network speaker adaptation. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 5711-5715).

The codes and scripts of the LHUC based adaptation techniques on end-to-end LF-MMI models are released (more clear scripts willed be released later), which are proposed in the paper:
	Xurong Xie, Xunying Liu, Hui Chen, Hongan Wang (2023). "Unsupervised model-based speaker adaptation of end-to-end lattice-free MMI model for speech recognition." Accepted by ICASSP 2023.

The codes are mainly based on src/nnet3, please find the running examples in egs/swbd/s5c.

                                                                           --By Xurong Xie
                                                                                             
#################################################################################################
# The original readme


[![Build Status](https://travis-ci.com/kaldi-asr/kaldi.svg?branch=master)](https://travis-ci.com/kaldi-asr/kaldi)
[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-Ready--to--Code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/kaldi-asr/kaldi) 
Kaldi Speech Recognition Toolkit
================================

To build the toolkit: see `./INSTALL`.  These instructions are valid for UNIX
systems including various flavors of Linux; Darwin; and Cygwin (has not been
tested on more "exotic" varieties of UNIX).  For Windows installation
instructions (excluding Cygwin), see `windows/INSTALL`.

To run the example system builds, see `egs/README.txt`

If you encounter problems (and you probably will), please do not hesitate to
contact the developers (see below). In addition to specific questions, please
let us know if there are specific aspects of the project that you feel could be
improved, that you find confusing, etc., and which missing features you most
wish it had.

Kaldi information channels
--------------------------

For HOT news about Kaldi see [the project site](http://kaldi-asr.org/).

[Documentation of Kaldi](http://kaldi-asr.org/doc/):
- Info about the project, description of techniques, tutorial for C++ coding.
- Doxygen reference of the C++ code.

[Kaldi forums and mailing lists](http://kaldi-asr.org/forums.html):

We have two different lists
- User list kaldi-help
- Developer list kaldi-developers:

To sign up to any of those mailing lists, go to
[http://kaldi-asr.org/forums.html](http://kaldi-asr.org/forums.html):


Development pattern for contributors
------------------------------------

1. [Create a personal fork](https://help.github.com/articles/fork-a-repo/)
   of the [main Kaldi repository](https://github.com/kaldi-asr/kaldi) in GitHub.
2. Make your changes in a named branch different from `master`, e.g. you create
   a branch `my-awesome-feature`.
3. [Generate a pull request](https://help.github.com/articles/creating-a-pull-request/)
   through the Web interface of GitHub.
4. As a general rule, please follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
   There are a [few exceptions in Kaldi](http://kaldi-asr.org/doc/style.html).
   You can use the [Google's cpplint.py](https://raw.githubusercontent.com/google/styleguide/gh-pages/cpplint/cpplint.py)
   to verify that your code is free of basic mistakes.

Platform specific notes
-----------------------

### PowerPC 64bits little-endian (ppc64le)

- Kaldi is expected to work out of the box in RHEL >= 7 and Ubuntu >= 16.04 with
  OpenBLAS, ATLAS, or CUDA.
- CUDA drivers for ppc64le can be found at [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
- An [IBM Redbook](https://www.redbooks.ibm.com/abstracts/redp5169.html) is
  available as a guide to install and configure CUDA.

### Android

- Kaldi supports cross compiling for Android using Android NDK, clang++ and
  OpenBLAS.
- See [this blog post](http://jcsilva.github.io/2017/03/18/compile-kaldi-android/)
  for details.

### Web Assembly

- Kaldi supports cross compiling for Web Assembly for in-browser execution
  using [emscripten](https://emscripten.org/) and CLAPACK.
- See [this post](https://gitlab.inria.fr/kaldi.web/kaldi-wasm/-/wikis/build_details.md)
  for a step-by-step description of the build process.
