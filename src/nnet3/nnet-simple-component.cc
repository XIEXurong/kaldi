// nnet3/nnet-simple-component.cc

// Copyright 2015-2017  Johns Hopkins University (author: Daniel Povey)
//                2015  Xiaohui Zhang
//                2015  Guoguo Chen
//                2015  Daniel Galvez
//                2016  Yiming Wang

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <iterator>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-parse.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet3 {

void PnormComponent::Init(int32 input_dim, int32 output_dim)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0 &&
               input_dim_ % output_dim_ == 0);
}

void PnormComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || cfl->HasUnusedValues() || output_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim);
}


void* PnormComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                               const CuMatrixBase<BaseFloat> &in,
                               CuMatrixBase<BaseFloat> *out) const {
  BaseFloat p = 2.0;
  out->GroupPnorm(in, p);
  return NULL;
}

void PnormComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("PnormComponent::Backprop");
  if (!in_deriv)
    return;
  BaseFloat p = 2.0;
  in_deriv->DiffGroupPnorm(in_value, out_value, out_deriv, p);
}

void PnormComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PnormComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "</PnormComponent>");
}

void PnormComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PnormComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "</PnormComponent>");
}

DropoutComponent::DropoutComponent(const DropoutComponent &other):
    RandomComponent(other),
    dim_(other.dim_),
    dropout_proportion_(other.dropout_proportion_),
    dropout_per_frame_(other.dropout_per_frame_) { }

Component* DropoutComponent::Copy() const {
  DropoutComponent *ans = new DropoutComponent(*this);
  return ans;
}

void DropoutComponent::Init(int32 dim, BaseFloat dropout_proportion,
                            bool dropout_per_frame) {
  dropout_proportion_ = dropout_proportion;
  dropout_per_frame_ = dropout_per_frame;
  dim_ = dim;
}

void DropoutComponent::InitFromConfig(ConfigLine *cfl) {
  int32 dim = 0;
  BaseFloat dropout_proportion = 0.0;
  bool dropout_per_frame = false;
  test_mode_ = false;
  bool ok = cfl->GetValue("dim", &dim) &&
    cfl->GetValue("dropout-proportion", &dropout_proportion);
  cfl->GetValue("dropout-per-frame", &dropout_per_frame);
  // It only makes sense to set test-mode in the config for testing purposes.
  cfl->GetValue("test-mode", &test_mode_);
    // for this stage, dropout is hard coded in
    // normal mode if not declared in config
  if (!ok || cfl->HasUnusedValues() || dim <= 0 ||
      dropout_proportion < 0.0 || dropout_proportion > 1.0)
       KALDI_ERR << "Invalid initializer for layer of type "
                 << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(dim, dropout_proportion, dropout_per_frame);
}

std::string DropoutComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_
         << ", dropout-proportion=" << dropout_proportion_
         << ", dropout-per-frame=" << (dropout_per_frame_ ? "true" : "false");
  return stream.str();
}

void* DropoutComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(out->NumRows() == in.NumRows() && out->NumCols() == in.NumCols()
               && in.NumCols() == dim_);

  BaseFloat dropout = dropout_proportion_;
  KALDI_ASSERT(dropout >= 0.0 && dropout <= 1.0);
  if (test_mode_) {
    out->CopyFromMat(in);
    out->Scale(1.0 - dropout);
    return NULL;
  }
  if (!dropout_per_frame_) {
    // This const_cast is only safe assuming you don't attempt
    // to use multi-threaded code with the GPU.
    const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(out);

    out->Add(-dropout);  // now, a proportion "dropout" will be <0.0
    // apply the function (x>0?1:0).  Now, a proportion
    // "dropout" will be zero and (1 - dropout) will be 1.0.
    out->ApplyHeaviside();

    out->MulElements(in);
  } else {
    // randomize the dropout matrix by row,
    // i.e. [[1,1,1,1],[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]]
    CuMatrix<BaseFloat> tmp(1, out->NumRows(), kUndefined);
    // This const_cast is only safe assuming you don't attempt
    // to use multi-threaded code with the GPU.
    const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(&tmp);
    tmp.Add(-dropout);
    tmp.ApplyHeaviside();
    out->CopyColsFromVec(tmp.Row(0));
    out->MulElements(in);
  }
  return NULL;
}


void DropoutComponent::Backprop(const std::string &debug_info,
                                const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &in_value,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                void *memo,
                                Component *to_update,
                                CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("DropoutComponent::Backprop");
  KALDI_ASSERT(in_value.NumRows() == out_value.NumRows() &&
               in_value.NumCols() == out_value.NumCols());

  KALDI_ASSERT(in_value.NumRows() == out_deriv.NumRows() &&
               in_value.NumCols() == out_deriv.NumCols());
  in_deriv->SetMatMatDivMat(out_deriv, out_value, in_value);
}



void DropoutComponent::Read(std::istream &is, bool binary) {
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<DropoutComponent>") {
    ReadToken(is, binary, &token);
  }
  KALDI_ASSERT(token == "<Dim>");
  ReadBasicType(is, binary, &dim_);  // read dimension.
  ReadToken(is, binary, &token);
  KALDI_ASSERT(token == "<DropoutProportion>");
  ReadBasicType(is, binary, &dropout_proportion_);  // read dropout rate
  ReadToken(is, binary, &token);
  if (token == "<DropoutPerFrame>") {
    ReadBasicType(is, binary, &dropout_per_frame_);  // read dropout mode
    ReadToken(is, binary, &token);
  } else {
    dropout_per_frame_ = false;
  }
  if (token == "<TestMode>") {
    ReadBasicType(is, binary, &test_mode_);  // read test mode
    ExpectToken(is, binary, "</DropoutComponent>");
  } else {
    test_mode_ = false;
    KALDI_ASSERT(token == "</DropoutComponent>");
  }
}

void DropoutComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<DropoutComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<DropoutProportion>");
  WriteBasicType(os, binary, dropout_proportion_);
  WriteToken(os, binary, "<DropoutPerFrame>");
  WriteBasicType(os, binary, dropout_per_frame_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "</DropoutComponent>");
}

void ElementwiseProductComponent::Init(int32 input_dim, int32 output_dim)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ >= 0);
  KALDI_ASSERT(input_dim_ > output_dim_);
  KALDI_ASSERT(input_dim_ % output_dim_ == 0);
}

void ElementwiseProductComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || cfl->HasUnusedValues() || output_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim);
}

void* ElementwiseProductComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == input_dim_);
  int32 num_inputs = input_dim_ / output_dim_;
  for (int32 i = 0; i < num_inputs; i++)  {
    CuSubMatrix<BaseFloat> current_in(in, 0, in.NumRows(),
                                      i * output_dim_, output_dim_);
    if (i == 0) {
      out->CopyFromMat(current_in);
    } else  {
      out->MulElements(current_in);
    }
  }
  return NULL;
}

void ElementwiseProductComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("ElementwiseProductComponent::Backprop");
  if (!in_deriv)  return;
  int32 num_inputs = input_dim_ / output_dim_;
  for (int32 i = 0; i < num_inputs; i++)  {
    CuSubMatrix<BaseFloat> current_in_deriv(*in_deriv, 0, in_deriv->NumRows(),
                                            i * output_dim_,
                                            output_dim_);
    current_in_deriv.CopyFromMat(out_deriv);
    for (int32 j = 0; j < num_inputs; j++)  {
      if (i == j)
        continue;
      CuSubMatrix<BaseFloat> in_value_partition(in_value, 0,
                                                in_value.NumRows(),
                                                j * output_dim_,
                                                output_dim_);
      current_in_deriv.MulElements(in_value_partition);
    }
  }
}

void ElementwiseProductComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<ElementwiseProductComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "</ElementwiseProductComponent>");
}

void ElementwiseProductComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<ElementwiseProductComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "</ElementwiseProductComponent>");
}

void* SigmoidComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  out->Sigmoid(in);
  return NULL;
}

void SigmoidComponent::Backprop(const std::string &debug_info,
                                const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                void *memo,
                                Component *to_update_in,
                                CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("SigmoidComponent::Backprop");
  if (in_deriv != NULL) {
    in_deriv->DiffSigmoid(out_value, out_deriv);
    SigmoidComponent *to_update = dynamic_cast<SigmoidComponent*>(to_update_in);
    if (to_update != NULL) {
      RepairGradients(out_value, in_deriv, to_update);
      to_update->StoreBackpropStats(out_deriv);
    }
  }
}

void SigmoidComponent::RepairGradients(
    const CuMatrixBase<BaseFloat> &out_value,
    CuMatrixBase<BaseFloat> *in_deriv,
    SigmoidComponent *to_update) const {
  KALDI_ASSERT(to_update != NULL);
  // maximum possible derivative of SigmoidComponent is 0.25.
  // the default lower-threshold on the derivative, below which we
  // add a term to the derivative to encourage the inputs to the sigmoid
  // to be closer to zero, is 0.05, which means the derivative is on average
  // 5 times smaller than its maximum possible value.
  BaseFloat default_lower_threshold = 0.05;

  // we use this 'repair_probability' (hardcoded for now) to limit
  // this code to running on about half of the minibatches.
  BaseFloat repair_probability = 0.5;

  to_update->num_dims_processed_ += dim_;

  if (self_repair_scale_ == 0.0 || count_ == 0.0 || deriv_sum_.Dim() != dim_ ||
      RandUniform() > repair_probability)
    return;

  // check that the self-repair scale is in a reasonable range.
  KALDI_ASSERT(self_repair_scale_ > 0.0 && self_repair_scale_ < 0.1);
  BaseFloat unset = kUnsetThreshold; // -1000.0
  BaseFloat lower_threshold = (self_repair_lower_threshold_ == unset ?
                               default_lower_threshold :
                               self_repair_lower_threshold_) *
      count_;
  if (self_repair_upper_threshold_ != unset) {
    KALDI_ERR << "Do not set the self-repair-upper-threshold for sigmoid "
              << "components, it does nothing.";
  }

  // thresholds_vec is actually a 1-row matrix.  (the ApplyHeaviside
  // function isn't defined for vectors).
  CuMatrix<BaseFloat> thresholds(1, dim_);
  CuSubVector<BaseFloat> thresholds_vec(thresholds, 0);
  thresholds_vec.AddVec(-1.0, deriv_sum_);
  thresholds_vec.Add(lower_threshold);
  thresholds.ApplyHeaviside();
  to_update->num_dims_self_repaired_ += thresholds_vec.Sum();

  // At this point, 'thresholds_vec' contains a 1 for each dimension of
  // the output that is 'problematic', i.e. for which the avg-deriv
  // is less than the self-repair lower threshold, and a 0 for
  // each dimension that is not problematic.

  // what we want to do is to add
  // -self_repair_scale_ / repair_probability times (2 * output-valiue - 1.0)
  // to the input derivative for each problematic dimension.

  // Here, 2 * output - 1.0 is a version of the sigmoid that goes from -1.0 to
  // 1.0, like a tanh.  the negative sign is so that for inputs <0, we push them
  // up towards 0, and for inputs >0, we push them down towards 0.
  // Our use of this sigmoid-type function here is just a convenience since
  // we have it available.  We could use just about any function that is positive
  // for inputs < 0 and negative for inputs > 0.

  // We can rearrange the above as: for only the problematic columns,
  //   input-deriv -= 2 * self-repair-scale / repair-probabilty * output
  //   input-deriv +=  self-repair-scale / repair-probabilty
  // which we can write as:
  //   input-deriv -= 2 * self-repair-scale / repair-probabilty * output * thresholds-vec
  //   input-deriv +=  self-repair-scale / repair-probabilty * thresholds-vec

  in_deriv->AddMatDiagVec(-2.0 * self_repair_scale_ / repair_probability,
                          out_value, kNoTrans, thresholds_vec);
  in_deriv->AddVecToRows(self_repair_scale_ / repair_probability,
                         thresholds_vec);
}



void SigmoidComponent::StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_value,
                                  void *memo) {
  // Only store stats about every other minibatch (but on the first minibatch,
  // always store it, which is necessary for the ConsolidateMemory() operation
  // to work correctly.
  if (RandInt(0, 1) == 0 && count_ != 0)
    return;
  // derivative of the nonlinearity is out_value * (1.0 - out_value);
  CuMatrix<BaseFloat> temp_deriv(out_value.NumRows(), out_value.NumCols(),
                                 kUndefined);
  temp_deriv.Set(1.0);
  temp_deriv.AddMat(-1.0, out_value);
  temp_deriv.MulElements(out_value);
  StoreStatsInternal(out_value, &temp_deriv);
}



void* NoOpComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in,
                              CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
  return NULL;
}

void NoOpComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             void *memo,
                             Component *to_update, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("NoOpComponent::Backprop");
  in_deriv->CopyFromMat(out_deriv);
  if (backprop_scale_ != 1.0)
    in_deriv->Scale(backprop_scale_);
}

void NoOpComponent::InitFromConfig(ConfigLine *cfl) {
  backprop_scale_ = 1.0;
  cfl->GetValue("backprop-scale", &backprop_scale_);
  if (!cfl->GetValue("dim", &dim_) ||
      dim_ <= 0 || cfl->HasUnusedValues()) {
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  }
}

std::string NoOpComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_;
  if (backprop_scale_ != 1.0)
    stream << ", backprop-scale=" << backprop_scale_;
  return stream.str();
}

void NoOpComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NoOpComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<BackpropScale>");
  WriteBasicType(os, binary, backprop_scale_);
  WriteToken(os, binary, "</NoOpComponent>");
}

void NoOpComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<NoOpComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);

  if (PeekToken(is, binary) == 'V') {
    // This is the old format, from when NoOpComponent inherited from
    // NonlinearComponent.
    backprop_scale_ = 1.0;
    ExpectToken(is, binary, "<ValueAvg>");
    CuVector<BaseFloat> temp_vec;
    temp_vec.Read(is, binary);
    ExpectToken(is, binary, "<DerivAvg>");
    temp_vec.Read(is, binary);
    ExpectToken(is, binary, "<Count>");
    BaseFloat temp_float;
    ReadBasicType(is, binary, &temp_float);
    if (PeekToken(is, binary) == 'O') {
      ExpectToken(is, binary, "<OderivRms>");
      temp_vec.Read(is, binary);
      ExpectToken(is, binary, "<OderivCount>");
      ReadBasicType(is, binary, &temp_float);
    }
    std::string token;
    ReadToken(is, binary, &token);
    if (token[0] != '<') {
      // this should happen only rarely, in case we couldn't push back the
      // '<' to the stream in PeekToken().
      token = '<' + token;
    }
    if (token == "<NumDimsSelfRepaired>") {
      ReadBasicType(is, binary, &temp_float);
      ReadToken(is, binary, &token);
    }
    if (token == "<NumDimsProcessed>") {
      ReadBasicType(is, binary, &temp_float);
      ReadToken(is, binary, &token);
    }
    KALDI_ASSERT(token == "</NoOpComponent>");
    return;
  } else {
    ExpectToken(is, binary, "<BackpropScale>");
    ReadBasicType(is, binary, &backprop_scale_);
    ExpectToken(is, binary, "</NoOpComponent>");
  }
}


void ClipGradientComponent::Read(std::istream &is, bool binary) {
  // might not see the "<NaturalGradientAffineComponent>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, "<ClipGradientComponent>",
                       "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<ClippingThreshold>");
  ReadBasicType(is, binary, &clipping_threshold_);
  ExpectToken(is, binary, "<NormBasedClipping>");
  ReadBasicType(is, binary, &norm_based_clipping_);
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<SelfRepairClippedProportionThreshold>") {
    ReadBasicType(is, binary, &self_repair_clipped_proportion_threshold_);
    ExpectToken(is, binary, "<SelfRepairTarget>");
    ReadBasicType(is, binary, &self_repair_target_);
    ExpectToken(is, binary, "<SelfRepairScale>");
    ReadBasicType(is, binary, &self_repair_scale_);
    ExpectToken(is, binary, "<NumElementsClipped>");
  } else {
    self_repair_clipped_proportion_threshold_ = 1.0;
    self_repair_target_ = 0.0;
    self_repair_scale_ = 0.0;
    KALDI_ASSERT(token == "<NumElementsClipped>");
  }
  ReadBasicType(is, binary, &num_clipped_);
  ExpectToken(is, binary, "<NumElementsProcessed>");
  ReadBasicType(is, binary, &count_);
  ReadToken(is, binary, &token);
  if (token == "<NumSelfRepaired>") {
    ReadBasicType(is, binary, &num_self_repaired_);
    ExpectToken(is, binary, "<NumBackpropped>");
    ReadBasicType(is, binary, &num_backpropped_);
    ExpectToken(is, binary, "</ClipGradientComponent>");
  } else {
    num_self_repaired_ = 0;
    num_backpropped_ = 0;
    KALDI_ASSERT(token == "</ClipGradientComponent>");
  }
}

void ClipGradientComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<ClipGradientComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<ClippingThreshold>");
  WriteBasicType(os, binary, clipping_threshold_);
  WriteToken(os, binary, "<NormBasedClipping>");
  WriteBasicType(os, binary, norm_based_clipping_);
  WriteToken(os, binary, "<SelfRepairClippedProportionThreshold>");
  WriteBasicType(os, binary, self_repair_clipped_proportion_threshold_);
  WriteToken(os, binary, "<SelfRepairTarget>");
  WriteBasicType(os, binary, self_repair_target_);
  WriteToken(os, binary, "<SelfRepairScale>");
  WriteBasicType(os, binary, self_repair_scale_);
  WriteToken(os, binary, "<NumElementsClipped>");
  WriteBasicType(os, binary, num_clipped_);
  WriteToken(os, binary, "<NumElementsProcessed>");
  WriteBasicType(os, binary, count_);
  WriteToken(os, binary, "<NumSelfRepaired>");
  WriteBasicType(os, binary, num_self_repaired_);
  WriteToken(os, binary, "<NumBackpropped>");
  WriteBasicType(os, binary, num_backpropped_);
  WriteToken(os, binary, "</ClipGradientComponent>");
}

std::string ClipGradientComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_
         << ", norm-based-clipping="
         << (norm_based_clipping_ ? "true" : "false")
         << ", clipping-threshold=" << clipping_threshold_
         << ", clipped-proportion="
         << (count_ > 0 ? static_cast<BaseFloat>(num_clipped_)/count_ : 0);
  if (self_repair_scale_ != 0.0)
    stream << ", self-repair-clipped-proportion-threshold="
           << self_repair_clipped_proportion_threshold_
           << ", self-repair-target=" << self_repair_target_
           << ", self-repair-scale=" << self_repair_scale_;
  return stream.str();
}

void ClipGradientComponent::Init(int32 dim,
                                 BaseFloat clipping_threshold,
                                 bool norm_based_clipping,
                                 BaseFloat self_repair_clipped_proportion_threshold,
                                 BaseFloat self_repair_target,
                                 BaseFloat self_repair_scale,
                                 int32 num_clipped,
                                 int32 count,
                                 int32 num_self_repaired,
                                 int32 num_backpropped)  {
  KALDI_ASSERT(clipping_threshold >= 0 && dim > 0 &&
      self_repair_clipped_proportion_threshold >= 0.0 &&
      self_repair_target >= 0.0 && self_repair_scale >= 0.0);
  dim_ = dim;
  norm_based_clipping_ = norm_based_clipping;
  clipping_threshold_ = clipping_threshold;
  self_repair_clipped_proportion_threshold_ =
      self_repair_clipped_proportion_threshold;
  self_repair_target_ = self_repair_target;
  self_repair_scale_ = self_repair_scale;
  num_clipped_ = num_clipped;
  count_ = count;
  num_self_repaired_ = num_self_repaired;
  num_backpropped_ = num_backpropped;
}

void ClipGradientComponent::InitFromConfig(ConfigLine *cfl) {
  int32 dim = 0;
  bool ok = cfl->GetValue("dim", &dim);
  bool norm_based_clipping = false;
  BaseFloat clipping_threshold = 15.0;
  BaseFloat self_repair_clipped_proportion_threshold = 0.01;
  BaseFloat self_repair_target = 0.0;
  BaseFloat self_repair_scale = 1.0;
  cfl->GetValue("clipping-threshold", &clipping_threshold);
  cfl->GetValue("norm-based-clipping", &norm_based_clipping);
  cfl->GetValue("self-repair-clipped-proportion-threshold",
                &self_repair_clipped_proportion_threshold);
  cfl->GetValue("self-repair-target",
                &self_repair_target);
  cfl->GetValue("self-repair-scale", &self_repair_scale);
  if (!ok || cfl->HasUnusedValues() ||
      clipping_threshold < 0 || dim <= 0 ||
      self_repair_clipped_proportion_threshold < 0.0 ||
      self_repair_target < 0.0 || self_repair_scale < 0.0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(dim, clipping_threshold, norm_based_clipping,
       self_repair_clipped_proportion_threshold,
       self_repair_target,
       self_repair_scale, 0, 0, 0, 0);
}

void* ClipGradientComponent::Propagate(
                                 const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
  return NULL;
}


void ClipGradientComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &in_value,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             void *memo,
                             Component *to_update_in, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("ClipGradientComponent::Backprop");
  // the following statement will do nothing if in_deriv and out_deriv have same
  // memory.
  in_deriv->CopyFromMat(out_deriv);

  ClipGradientComponent *to_update =
      dynamic_cast<ClipGradientComponent*>(to_update_in);

  if (clipping_threshold_ > 0) {
    if (norm_based_clipping_) {
      // each row in the derivative matrix, which corresponds to one sample in
      // the mini-batch, is scaled to have a max-norm of clipping_threshold_
      CuVector<BaseFloat> clipping_scales(in_deriv->NumRows());
      clipping_scales.AddDiagMat2(pow(clipping_threshold_, -2), *in_deriv,
                                  kNoTrans, 0.0);
     // now clipping_scales contains the squared (norm of each row divided by
     //  clipping_threshold)
      int32 num_not_scaled;
      clipping_scales.ApplyFloor(1.0, &num_not_scaled);
     // now clipping_scales contains min(1,
     //    squared-(norm/clipping_threshold))
      if (num_not_scaled != clipping_scales.Dim()) {
        clipping_scales.ApplyPow(-0.5);
        // now clipping_scales contains max(1,
        //       clipping_threshold/vector_norm)
        in_deriv->MulRowsVec(clipping_scales);
        if (to_update != NULL)
          to_update->num_clipped_ += (clipping_scales.Dim() - num_not_scaled);
       }
      if (to_update != NULL)
        to_update->count_ += clipping_scales.Dim();
    } else {
      // each element of the derivative matrix, is clipped to be below the
      // clipping_threshold_
      in_deriv->ApplyCeiling(clipping_threshold_);
      in_deriv->ApplyFloor(-1 * clipping_threshold_);
    }

    if (to_update != NULL) {
      to_update->num_backpropped_ += 1;
      RepairGradients(debug_info, in_value, in_deriv, to_update);
    }
  } else if (clipping_threshold_ == 0.0) {
    in_deriv->SetZero();
  }
}

// This function will add a self-repair term to in-deriv, attempting to shrink
// the magnitude of the input towards self_repair_target_.
// This term is proportional to [-(input vector - self_repair_target_)].
// The avarage magnitude of this term is equal to
// [self_repair_scale_ * clipped_proportion * average norm of input derivative].
// We use norm of input derivative when computing the magnitude so that it is
// comparable to the magnitude of input derivative, especially when the gradient
// explosion is actually happening.
void ClipGradientComponent::RepairGradients(
    const std::string &debug_info,
    const CuMatrixBase<BaseFloat> &in_value,
    CuMatrixBase<BaseFloat> *in_deriv, ClipGradientComponent *to_update) const {
  KALDI_ASSERT(to_update != NULL);

  // we use this 'repair_probability' (hardcoded for now) to limit
  // this code to running on about half of the minibatches.
  BaseFloat repair_probability = 0.5;
  if (self_repair_clipped_proportion_threshold_ >= 1.0 ||
      self_repair_scale_ == 0.0 || count_ == 0 ||
      RandUniform() > repair_probability)
    return;

  KALDI_ASSERT(self_repair_target_ >= 0.0 && self_repair_scale_ > 0.0);

  BaseFloat clipped_proportion =
    (count_ > 0 ? static_cast<BaseFloat>(num_clipped_) / count_ : 0);
  // in-deriv would be modified only when clipped_proportion exceeds the
  // threshold
  if (clipped_proportion <= self_repair_clipped_proportion_threshold_)
    return;

  to_update->num_self_repaired_ += 1;
  if (to_update->debug_info_ == "") // get the component-node name
    to_update->debug_info_ = debug_info;
  if (to_update->num_self_repaired_ == 1)
    KALDI_LOG << "ClipGradientComponent(node_name=" << debug_info
              << ")'s self-repair was activated as the first time at the "
              << to_update->num_backpropped_
              << "-th call of Backprop() in this training job.";

  // sign_mat = sign(in_value), i.e.,
  // An element in sign_mat is 1 if its corresponding element in in_value > 0,
  // or -1 otherwise
  CuMatrix<BaseFloat> sign_mat(in_value);
  sign_mat.ApplyHeaviside();
  sign_mat.Scale(2.0);
  sign_mat.Add(-1.0);

  // repair_mat =
  // floor(abs(in_value) - self_repair_target_, 0) .* sign(in_value)
  CuMatrix<BaseFloat> repair_mat(in_value);
  repair_mat.ApplyPowAbs(1.0);
  repair_mat.Add(-self_repair_target_);
  repair_mat.ApplyFloor(0.0);
  repair_mat.MulElements(sign_mat);

  // magnitude =
  // self_repair_scale_ * clipped_proportion * average norm of in-deriv
  CuVector<BaseFloat> in_deriv_norm_vec(in_deriv->NumRows());
  in_deriv_norm_vec.AddDiagMat2(1.0, *in_deriv, kNoTrans, 0.0);
  in_deriv_norm_vec.ApplyPow(0.5);
  double in_deriv_norm_sum = in_deriv_norm_vec.Sum();
  BaseFloat magnitude = self_repair_scale_ * clipped_proportion *
                        (in_deriv_norm_sum / in_deriv_norm_vec.Dim());

  CuVector<BaseFloat> repair_mat_norm_vec(repair_mat.NumRows());
  repair_mat_norm_vec.AddDiagMat2(1.0, repair_mat, kNoTrans, 0.0);
  repair_mat_norm_vec.ApplyPow(0.5);
  double repair_mat_norm_sum = repair_mat_norm_vec.Sum();
  double scale = 0.0;
  if (repair_mat_norm_sum != 0.0)
    scale = magnitude / (repair_mat_norm_sum / repair_mat_norm_vec.Dim());
  // repair_mat is scaled so that on average the rows have the norm
  // (magnitude / repair_probability). This will give higher magnitude of
  // self-repair to input vectors that have larger absolute value, which tend to
  // be those that are diverging.
  in_deriv->AddMat(-scale / repair_probability, repair_mat);
  CuVector<BaseFloat> in_deriv_repaired_norm_vec(in_deriv->NumRows());
  in_deriv_repaired_norm_vec.AddDiagMat2(1.0, *in_deriv, kNoTrans, 0.0);
  in_deriv_repaired_norm_vec.ApplyPow(0.5);
  // scale in_deriv to have the same norm as that before adding the self-repair
  // term, in order to avoid increase of the norm caused by self-repair,
  // which may incur more clip of gradient and thus more self-repair
  double in_deriv_repaired_norm_sum = in_deriv_repaired_norm_vec.Sum();
  if (in_deriv_repaired_norm_sum != 0.0)
    in_deriv->Scale(in_deriv_norm_sum / in_deriv_repaired_norm_sum);
}

void ClipGradientComponent::ZeroStats()  {
  count_ = 0.0;
  num_clipped_ = 0.0;
  num_self_repaired_ = 0;
  num_backpropped_ = 0;
}

void ClipGradientComponent::Scale(BaseFloat scale) {
  count_ *= scale;
  num_clipped_ *= scale;
}

void ClipGradientComponent::Add(BaseFloat alpha, const Component &other_in) {
  const ClipGradientComponent *other =
      dynamic_cast<const ClipGradientComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  count_ += alpha * other->count_;
  num_clipped_ += alpha * other->num_clipped_;
}

void* TanhComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in,
                              CuMatrixBase<BaseFloat> *out) const {
  // Apply tanh function to each element of the output...
  // the tanh function may be written as -1 + ( 2 / (1 + e^{-2 x})),
  // which is a scaled and shifted sigmoid.
  out->Tanh(in);
  return NULL;
}


void TanhComponent::RepairGradients(
    const CuMatrixBase<BaseFloat> &out_value,
    CuMatrixBase<BaseFloat> *in_deriv,
    TanhComponent *to_update) const {
  KALDI_ASSERT(to_update != NULL);
  // maximum possible derivative of SigmoidComponent is 1.0
  // the default lower-threshold on the derivative, below which we
  // add a term to the derivative to encourage the inputs to the sigmoid
  // to be closer to zero, is 0.2, which means the derivative is on average
  // 5 times smaller than its maximum possible value.
  BaseFloat default_lower_threshold = 0.2;

  // we use this 'repair_probability' (hardcoded for now) to limit
  // this code to running on about half of the minibatches.
  BaseFloat repair_probability = 0.5;

  to_update->num_dims_processed_ += dim_;

  if (self_repair_scale_ == 0.0 || count_ == 0.0 || deriv_sum_.Dim() != dim_ ||
      RandUniform() > repair_probability)
    return;

  // check that the self-repair scale is in a reasonable range.
  KALDI_ASSERT(self_repair_scale_ > 0.0 && self_repair_scale_ < 0.1);
  BaseFloat unset = kUnsetThreshold; // -1000.0
  BaseFloat lower_threshold = (self_repair_lower_threshold_ == unset ?
                               default_lower_threshold :
                               self_repair_lower_threshold_) *
      count_;
  if (self_repair_upper_threshold_ != unset) {
    KALDI_ERR << "Do not set the self-repair-upper-threshold for sigmoid "
              << "components, it does nothing.";
  }

  // thresholds_vec is actually a 1-row matrix.  (the ApplyHeaviside
  // function isn't defined for vectors).
  CuMatrix<BaseFloat> thresholds(1, dim_);
  CuSubVector<BaseFloat> thresholds_vec(thresholds, 0);
  thresholds_vec.AddVec(-1.0, deriv_sum_);
  thresholds_vec.Add(lower_threshold);
  thresholds.ApplyHeaviside();
  to_update->num_dims_self_repaired_ += thresholds_vec.Sum();

  // At this point, 'thresholds_vec' contains a 1 for each dimension of
  // the output that is 'problematic', i.e. for which the avg-deriv
  // is less than the self-repair lower threshold, and a 0 for
  // each dimension that is not problematic.

  // what we want to do is to add -self_repair_scale_ / repair_probability times
  // output-valiue) to the input derivative for each problematic dimension.
  // note that for the tanh, the output-value goes from -1.0 when the input is
  // -inf to +1.0 when the input is +inf.  The negative sign is so that for
  // inputs <0, we push them up towards 0, and for inputs >0, we push them down
  // towards 0.  Our use of the tanh here is just a convenience since we have it
  // available.  We could use just about any function that is positive for
  // inputs < 0 and negative for inputs > 0.

  // We can rearrange the above as: for only the problematic columns,
  //   input-deriv -= self-repair-scale / repair-probabilty * output
  // which we can write as:
  //   input-deriv -=  self-repair-scale / repair-probabilty * output * thresholds-vec

  in_deriv->AddMatDiagVec(-self_repair_scale_ / repair_probability,
                          out_value, kNoTrans, thresholds_vec);
}

void TanhComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_value,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             void *memo,
                             Component *to_update_in, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("TanhComponent::Backprop");
  if (in_deriv != NULL) {
    in_deriv->DiffTanh(out_value, out_deriv);
    TanhComponent *to_update = dynamic_cast<TanhComponent*>(to_update_in);
    if (to_update != NULL) {
      RepairGradients(out_value, in_deriv, to_update);
      to_update->StoreBackpropStats(out_deriv);
    }
  }
}

/*
  Note on the derivative of the tanh function:
  tanh'(x) = sech^2(x) = -(tanh(x)+1) (tanh(x)-1) = 1 - tanh^2(x)

  The element by element equation of what we're doing would be:
  in_deriv = out_deriv * (1.0 - out_value^2).
  We can accomplish this via calls to the matrix library. */
void TanhComponent::StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &out_value,
                               void *memo) {
  // Only store stats about every other minibatch (but on the first minibatch,
  // always store it, which is necessary for the ConsolidateMemory() operation
  // to work correctly.
  if (RandInt(0, 1) == 0 && count_ != 0)
    return;
  // derivative of the onlinearity is out_value * (1.0 - out_value);
  CuMatrix<BaseFloat> temp_deriv(out_value);
  temp_deriv.ApplyPow(2.0);
  temp_deriv.Scale(-1.0);
  temp_deriv.Add(1.0);
  StoreStatsInternal(out_value, &temp_deriv);
}

void* RectifiedLinearComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  // Apply rectified linear function (x >= 0 ? 1.0 : 0.0)
  out->CopyFromMat(in);
  out->ApplyFloor(0.0);
  return NULL;
}

void RectifiedLinearComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &, //in_value
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("RectifiedLinearComponent::Backprop");
  if (in_deriv != NULL) {
    in_deriv->Heaviside(out_value);
    in_deriv->MulElements(out_deriv);
    RectifiedLinearComponent *to_update =
        dynamic_cast<RectifiedLinearComponent*>(to_update_in);
    if (to_update != NULL) {
      RepairGradients(in_deriv, to_update);
      to_update->StoreBackpropStats(out_deriv);
    }
  }
}


void RectifiedLinearComponent::RepairGradients(
    CuMatrixBase<BaseFloat> *in_deriv,
    RectifiedLinearComponent *to_update) const {
  KALDI_ASSERT(to_update != NULL);
  int32 dim = dim_, block_dim = block_dim_;
  BaseFloat default_lower_threshold = 0.05,
      default_upper_threshold = 0.95;
  // we use this 'repair_probability' (hardcoded for now) to limit
  // this code to running on about half of the minibatches.
  BaseFloat repair_probability = 0.5;
  KALDI_ASSERT(in_deriv->NumCols() == dim || in_deriv->NumCols() == block_dim);
  if (self_repair_scale_ == 0.0 || count_ == 0.0 ||
      deriv_sum_.Dim() != dim)
    return;

  if (in_deriv->NumCols() != block_dim) {
    KALDI_ASSERT(in_deriv->NumCols() == in_deriv->Stride());
    int32 dim_multiple = dim / block_dim;
    CuSubMatrix<BaseFloat> in_deriv_reshaped(in_deriv->Data(),
                                             in_deriv->NumRows() * dim_multiple,
                                             block_dim, block_dim);
    RepairGradients(&in_deriv_reshaped, to_update);
    return;
  }

  // By now we know that in_deriv->NumCols() == block_dim.

  if (RandUniform() > repair_probability)
    return;

  to_update->num_dims_processed_ += block_dim;

  // check that the self-repair scale is in a reasonable range.
  KALDI_ASSERT(self_repair_scale_ > 0.0 && self_repair_scale_ < 0.1);
  BaseFloat unset = kUnsetThreshold; // -1000.0
  BaseFloat count = count_,
      lower_threshold = (self_repair_lower_threshold_ == unset ?
                         default_lower_threshold :
                         self_repair_lower_threshold_) * count,
      upper_threshold = (self_repair_upper_threshold_ == unset ?
                         default_upper_threshold :
                         self_repair_upper_threshold_) * count;

  CuMatrix<BaseFloat> storage(2, block_dim + 2, kUndefined);
  CuSubVector<BaseFloat> thresholds_vec(storage.RowData(0) + block_dim, 2);
  CuSubMatrix<BaseFloat> stats_mat(storage, 0, 2, 0, block_dim);
  thresholds_vec(0) = -lower_threshold;
  thresholds_vec(1) = -upper_threshold;
  CuSubVector<BaseFloat> row0(stats_mat, 0);
  CuSubVector<BaseFloat> row1(stats_mat, 1);

  if (block_dim == dim) {
    row0.CopyFromVec(deriv_sum_);
  } else {
    CuSubMatrix<double> deriv_sum_mat(deriv_sum_.Data(),
                                      dim / block_dim,
                                      block_dim, block_dim);
    CuVector<double> deriv_sum_dbl(block_dim);
    // get the average of the deriv-sums over the blocks.
    deriv_sum_dbl.AddRowSumMat(block_dim * 1.0 / dim, deriv_sum_mat);
    row0.CopyFromVec(deriv_sum_dbl);
  }
  row1.CopyFromVec(row0);
  stats_mat.AddVecToCols(1.0, thresholds_vec, 1.0);
  // now row0 equals stats - lower_threshold, and
  //     row1 equals stats - upper_threshold.
  stats_mat.ApplyHeaviside();
  // now row0 equals (stats > lower_threshold ? 1 : 0), and
  //     row1 equals (stats > upper_threshold ? 1 : 0).
  // what we want is:
  // self_repair_scale * ((stats <= lower_threshold ? 1 : 0) +
  //                         (stats > upper_threshold ? -1 : 0)).
  //
  // we can get these in stats_mat.Row(0) by computing:
  // -self_repair_scale * (stats_mat.Row(1)  + stats_mat.Row(0) - 1).
  row0.AddVec(1.0, row1, 1.0);
  row0.Add(-1.0);
  CuVector<BaseFloat> temp(row0);
  temp.ApplyPow(2.0);
  to_update->num_dims_self_repaired_ += temp.Sum();
  // [actually we need to divide by repair_probability also, to
  //  correct for the fact that we only do this on some frames.]
  row0.Scale(-self_repair_scale_ / repair_probability);
  in_deriv->AddVecToRows(1.0, row0, 1.0);
}


void RectifiedLinearComponent::StoreStats(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    void *memo) {
  // Only store stats about every other minibatch (but on the first minibatch,
  // always store it, which is necessary for the ConsolidateMemory() operation
  // to work correctly.
  if (RandInt(0, 1) == 0 && count_ != 0)
    return;
  CuMatrix<BaseFloat> temp_deriv(out_value.NumRows(),
                                 out_value.NumCols(),
                                 kUndefined);
  temp_deriv.Heaviside(out_value);
  StoreStatsInternal(out_value, &temp_deriv);
}

void AffineComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    // If scale == 0.0 we call SetZero() which will get rid of NaN's and inf's.
    linear_params_.SetZero();
    bias_params_.SetZero();
  } else {
    linear_params_.Scale(scale);
    bias_params_.Scale(scale);
  }
}

void AffineComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  bias_params_.Resize(output_dim);
  linear_params_.Resize(output_dim, input_dim);
}

void AffineComponent::Add(BaseFloat alpha, const Component &other_in) {
  const AffineComponent *other =
      dynamic_cast<const AffineComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

AffineComponent::AffineComponent(const AffineComponent &component):
    UpdatableComponent(component),
    linear_params_(component.linear_params_),
    bias_params_(component.bias_params_),
    orthonormal_constraint_(component.orthonormal_constraint_) { }

AffineComponent::AffineComponent(const CuMatrixBase<BaseFloat> &linear_params,
                                 const CuVectorBase<BaseFloat> &bias_params,
                                 BaseFloat learning_rate):
    linear_params_(linear_params),
    bias_params_(bias_params),
    orthonormal_constraint_(0.0) {
  SetUnderlyingLearningRate(learning_rate);
  KALDI_ASSERT(linear_params.NumRows() == bias_params.Dim()&&
               bias_params.Dim() != 0);
}

void AffineComponent::SetParams(const CuVectorBase<BaseFloat> &bias,
                                const CuMatrixBase<BaseFloat> &linear) {
  bias_params_ = bias;
  linear_params_ = linear;
  KALDI_ASSERT(bias_params_.Dim() == linear_params_.NumRows());
}

void AffineComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);

  CuVector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

std::string AffineComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info();
  if (orthonormal_constraint_ != 0.0)
    stream << ", orthonormal-constraint=" << orthonormal_constraint_;
  PrintParameterStats(stream, "linear-params", linear_params_,
                      false, // include_mean
                      true, // include_row_norms
                      true, // include_column_norms
                      GetVerboseLevel() >= 2); // include_singular_values
  PrintParameterStats(stream, "bias", bias_params_, true);
  return stream.str();
}

Component* AffineComponent::Copy() const {
  AffineComponent *ans = new AffineComponent(*this);
  return ans;
}

BaseFloat AffineComponent::DotProduct(const UpdatableComponent &other_in) const {
  const AffineComponent *other =
      dynamic_cast<const AffineComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans)
      + VecVec(bias_params_, other->bias_params_);
}

void AffineComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev, BaseFloat bias_stddev) {
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
}

void AffineComponent::Init(std::string matrix_filename) {
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  bias_params_.CopyColFromMat(mat, input_dim);
}

void AffineComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  InitLearningRatesFromConfig(cfl);
  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 1.0;
    cfl->GetValue("param-stddev", &param_stddev);
    cfl->GetValue("bias-stddev", &bias_stddev);
    Init(input_dim, output_dim,
         param_stddev, bias_stddev);
  }
  cfl->GetValue("orthonormal-constraint", &orthonormal_constraint_);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}




void* AffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {

  // No need for asserts as they'll happen within the matrix operations.
  out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);
  return NULL;
}

void AffineComponent::UpdateSimple(const CuMatrixBase<BaseFloat> &in_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv) {
  bias_params_.AddRowSumMat(learning_rate_, out_deriv, 1.0);
  linear_params_.AddMatMat(learning_rate_, out_deriv, kTrans,
                           in_value, kNoTrans, 1.0);
}

void AffineComponent::Backprop(const std::string &debug_info,
                               const ComponentPrecomputedIndexes *indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               void *memo,
                               Component *to_update_in,
                               CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("AffineComponent::Backprop");
  AffineComponent *to_update = dynamic_cast<AffineComponent*>(to_update_in);

  // Propagate the derivative back to the input.
  // add with coefficient 1.0 since property kBackpropAdds is true.
  // If we wanted to add with coefficient 0.0 we'd need to zero the
  // in_deriv, in case of infinities.
  if (in_deriv)
    in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, linear_params_, kNoTrans,
                        1.0);

  if (to_update != NULL) {
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    if (to_update->is_gradient_)
      to_update->UpdateSimple(in_value, out_deriv);
    else  // the call below is to a virtual function that may be re-implemented
      to_update->Update(debug_info, in_value, out_deriv);  // by child classes.
  }
}

void AffineComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  if (PeekToken(is, binary) == 'I') {
    // for back compatibility; we don't write this here any
    // more as it's written and read in Write/ReadUpdatableCommon
    ExpectToken(is, binary, "<IsGradient>");
    ReadBasicType(is, binary, &is_gradient_);
  }
  if (PeekToken(is, binary) == 'O') {
    ExpectToken(is, binary, "<OrthonormalConstraint>");
    ReadBasicType(is, binary, &orthonormal_constraint_);
  } else {
    orthonormal_constraint_ = 0.0;
  }
  ExpectToken(is, binary, "</AffineComponent>");
}

void AffineComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  if (orthonormal_constraint_ != 0.0) {
    WriteToken(os, binary, "<OrthonormalConstraint>");
    WriteBasicType(os, binary, orthonormal_constraint_);
  }
  WriteToken(os, binary, "</AffineComponent>");
}

int32 AffineComponent::NumParameters() const {
  return (InputDim() + 1) * OutputDim();
}
void AffineComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->Range(0, InputDim() * OutputDim()).CopyRowsFromMat(linear_params_);
  params->Range(InputDim() * OutputDim(),
                OutputDim()).CopyFromVec(bias_params_);
}
void AffineComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
  bias_params_.CopyFromVec(params.Range(InputDim() * OutputDim(),
                                        OutputDim()));
}

RepeatedAffineComponent::RepeatedAffineComponent(const RepeatedAffineComponent & component) :
    UpdatableComponent(component),
    linear_params_(component.linear_params_),
    bias_params_(component.bias_params_),
    num_repeats_(component.num_repeats_) {}


void RepeatedAffineComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    linear_params_.SetZero();
    bias_params_.SetZero();
  } else {
    linear_params_.Scale(scale);
    bias_params_.Scale(scale);
  }
}

void RepeatedAffineComponent::Add(BaseFloat alpha, const Component &other_in) {
  const RepeatedAffineComponent *other =
      dynamic_cast<const RepeatedAffineComponent *>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

void RepeatedAffineComponent::PerturbParams(BaseFloat stddev){
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);
  CuVector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

std::string RepeatedAffineComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", num-repeats=" << num_repeats_;
  PrintParameterStats(stream, "linear-params", linear_params_);
  PrintParameterStats(stream, "bias", bias_params_, true);
  return stream.str();
}

Component* RepeatedAffineComponent::Copy() const {
  RepeatedAffineComponent *ans = new RepeatedAffineComponent(*this);
  return ans;
}

BaseFloat RepeatedAffineComponent::DotProduct(const UpdatableComponent &other_in) const {
  const RepeatedAffineComponent *other =
      dynamic_cast<const RepeatedAffineComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans)
                     + VecVec(bias_params_, other->bias_params_);
}

void RepeatedAffineComponent::Init(int32 input_dim, int32 output_dim, int32 num_repeats,
                                   BaseFloat param_stddev, BaseFloat bias_mean,
                                   BaseFloat bias_stddev) {
  KALDI_ASSERT(input_dim % num_repeats == 0 && output_dim % num_repeats == 0);
  linear_params_.Resize(output_dim / num_repeats, input_dim / num_repeats);
  bias_params_.Resize(output_dim / num_repeats);
  num_repeats_ = num_repeats;
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  bias_params_.Add(bias_mean);
  SetNaturalGradientConfigs();
}


void RepeatedAffineComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  int32 num_repeats = num_repeats_;
  int32 input_dim = -1, output_dim = -1;
  InitLearningRatesFromConfig(cfl);
  ok = cfl->GetValue("num-repeats", &num_repeats) && ok;
  ok = cfl->GetValue("input-dim", &input_dim) && ok;
  ok = cfl->GetValue("output-dim", &output_dim) && ok;
  KALDI_ASSERT(input_dim % num_repeats == 0 &&
               "num-repeats must divide input-dim");
  KALDI_ASSERT(output_dim % num_repeats == 0 &&
               "num-repeats must divide output-dim");
  BaseFloat param_stddev = 1.0 / std::sqrt(input_dim / num_repeats),
      bias_mean = 0.0, bias_stddev = 0.0;
  cfl->GetValue("param-stddev", &param_stddev);
  cfl->GetValue("bias-mean", &bias_mean);
  cfl->GetValue("bias-stddev", &bias_stddev);
  Init(input_dim, output_dim,
       num_repeats, param_stddev, bias_mean, bias_stddev);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void* RepeatedAffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                        const CuMatrixBase<BaseFloat> &in,
                                        CuMatrixBase<BaseFloat> *out) const {
  // we gave the kInputContiguous and kOutputContiguous flags-- check that they
  // are honored.
  KALDI_ASSERT(in.NumCols() == in.Stride() &&
               out->NumCols() == out->Stride() &&
               out->NumRows() == in.NumRows());

  int32 num_repeats = num_repeats_,
      num_rows = in.NumRows(),
      block_dim_out = linear_params_.NumRows(),
      block_dim_in = linear_params_.NumCols();

  CuSubMatrix<BaseFloat> in_reshaped(in.Data(), num_rows * num_repeats,
                                     block_dim_in, block_dim_in),
      out_reshaped(out->Data(), num_rows * num_repeats,
                   block_dim_out, block_dim_out);

  out_reshaped.CopyRowsFromVec(bias_params_);

  out_reshaped.AddMatMat(1.0, in_reshaped, kNoTrans,
                         linear_params_, kTrans, 1.0);
  return NULL;
}

void RepeatedAffineComponent::Backprop(const std::string &debug_info,
                                       const ComponentPrecomputedIndexes *indexes,
                                       const CuMatrixBase<BaseFloat> &in_value,
                                       const CuMatrixBase<BaseFloat> &, // out_value
                                       const CuMatrixBase<BaseFloat> &out_deriv,
                                       void *memo,
                                       Component *to_update_in,
                                       CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("RepeatedAffineComponent::Backprop");
  KALDI_ASSERT(out_deriv.NumCols() == out_deriv.Stride() &&
       (in_value.NumCols() == 0 || in_value.NumCols() == in_value.Stride()) &&
               (!in_deriv || in_deriv->NumCols() == in_deriv->Stride()));

  RepeatedAffineComponent *to_update = dynamic_cast<RepeatedAffineComponent*>(
      to_update_in);

  // Propagate the derivative back to the input.
  // add with coefficient 1.0 since property kBackpropAdds is true.
  // If we wanted to add with coefficient 0.0 we'd need to zero the
  // in_deriv, in case of infinities.
  if (in_deriv) {
    int32 num_repeats = num_repeats_,
        num_rows = out_deriv.NumRows(),
        block_dim_out = linear_params_.NumRows(),
        block_dim_in = linear_params_.NumCols();

    CuSubMatrix<BaseFloat> in_deriv_reshaped(in_deriv->Data(),
                                             num_rows * num_repeats,
                                             block_dim_in, block_dim_in),
        out_deriv_reshaped(out_deriv.Data(),
                           num_rows * num_repeats,
                           block_dim_out, block_dim_out);
    in_deriv_reshaped.AddMatMat(1.0, out_deriv_reshaped, kNoTrans,
                                linear_params_, kNoTrans, 1.0);
  }

  // Next update the model (must do this 2nd so the derivatives we propagate are
  // accurate, in case this == to_update_in.)
  if (to_update != NULL)
    to_update->Update(in_value, out_deriv);
}

void RepeatedAffineComponent::Update(const CuMatrixBase<BaseFloat> &in_value,
                                     const CuMatrixBase<BaseFloat> &out_deriv) {
  KALDI_ASSERT(out_deriv.NumCols() == out_deriv.Stride() &&
               in_value.NumCols() == in_value.Stride() &&
               in_value.NumRows() == out_deriv.NumRows());


    int32 num_repeats = num_repeats_,
        num_rows = in_value.NumRows(),
        block_dim_out = linear_params_.NumRows(),
        block_dim_in = linear_params_.NumCols();

    CuSubMatrix<BaseFloat> in_value_reshaped(in_value.Data(),
                                             num_rows * num_repeats,
                                             block_dim_in, block_dim_in),
        out_deriv_reshaped(out_deriv.Data(),
                           num_rows * num_repeats,
                           block_dim_out, block_dim_out);


  linear_params_.AddMatMat(learning_rate_, out_deriv_reshaped, kTrans,
                           in_value_reshaped, kNoTrans, 1.0);
  bias_params_.AddRowSumMat(learning_rate_,
                            out_deriv_reshaped);
}

void RepeatedAffineComponent::Read(std::istream &is, bool binary) {
  // This Read function also works for NaturalGradientRepeatedAffineComponent.
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<NumRepeats>");
  ReadBasicType(is, binary, &num_repeats_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  if (PeekToken(is, binary) == 'I') {
    // for back compatibility; we don't write this here any
    // more as it's written and read in Write/ReadUpdatableCommon
    ExpectToken(is, binary, "<IsGradient>");
    ReadBasicType(is, binary, &is_gradient_);
  }
  ExpectToken(is, binary, std::string("</") + Type() + std::string(">"));
  SetNaturalGradientConfigs();
}

void RepeatedAffineComponent::Write(std::ostream &os, bool binary) const {
  // This Write function also works for NaturalGradientRepeatedAffineComponent.
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<NumRepeats>");
  WriteBasicType(os, binary, num_repeats_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  // write closing token.
  WriteToken(os, binary, std::string("</") + Type() + std::string(">"));
}

int32 RepeatedAffineComponent::NumParameters() const {
  // Note: unlike AffineComponent, InputDim() & OutputDim() are not used here and below,
  // for they are multipled by num_repeats_.
  return linear_params_.NumCols() * linear_params_.NumRows() + bias_params_.Dim();
}

void RepeatedAffineComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->Range(0, linear_params_.NumCols() * linear_params_.NumRows()).CopyRowsFromMat(linear_params_);
  params->Range(linear_params_.NumCols() * linear_params_.NumRows(),
                bias_params_.Dim()).CopyFromVec(bias_params_);
}

void RepeatedAffineComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, linear_params_.NumCols() * linear_params_.NumRows()));
  bias_params_.CopyFromVec(params.Range(linear_params_.NumCols() * linear_params_.NumRows(),
                                        bias_params_.Dim()));
}

void NaturalGradientRepeatedAffineComponent::SetNaturalGradientConfigs() {
  int32 rank_in = 40;
  int32 input_dim = linear_params_.NumCols();
  if (rank_in > input_dim / 2)
    rank_in = input_dim / 2;
  if (rank_in < 1)
    rank_in = 1;
  preconditioner_in_.SetRank(rank_in);
  preconditioner_in_.SetUpdatePeriod(4);
}

NaturalGradientRepeatedAffineComponent::NaturalGradientRepeatedAffineComponent(
    const NaturalGradientRepeatedAffineComponent &other):
    RepeatedAffineComponent(other),
    preconditioner_in_(other.preconditioner_in_) { }

// virtual
Component* NaturalGradientRepeatedAffineComponent::Copy() const {
  return new NaturalGradientRepeatedAffineComponent(*this);
}

void NaturalGradientRepeatedAffineComponent::Update(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  KALDI_ASSERT(out_deriv.NumCols() == out_deriv.Stride() &&
               in_value.NumCols() == in_value.Stride() &&
               in_value.NumRows() == out_deriv.NumRows());

  int32 num_repeats = num_repeats_,
      num_rows = in_value.NumRows(),
      block_dim_out = linear_params_.NumRows(),
      block_dim_in = linear_params_.NumCols();

  CuSubMatrix<BaseFloat> in_value_reshaped(in_value.Data(),
                                           num_rows * num_repeats,
                                           block_dim_in, block_dim_in),
        out_deriv_reshaped(out_deriv.Data(),
                           num_rows * num_repeats,
                           block_dim_out, block_dim_out);

  CuVector<BaseFloat> bias_deriv(block_dim_out);
  bias_deriv.AddRowSumMat(1.0, out_deriv_reshaped);

  CuMatrix<BaseFloat> deriv(block_dim_out,
                            block_dim_in + 1);
  deriv.ColRange(0, block_dim_in).AddMatMat(
      1.0, out_deriv_reshaped, kTrans,
      in_value_reshaped, kNoTrans, 1.0);
  deriv.CopyColFromVec(bias_deriv, block_dim_in);

  BaseFloat scale = 1.0;
  if (!is_gradient_) {
    try {
      // Only apply the preconditioning/natural-gradient if we're not computing
      // the exact gradient.
      preconditioner_in_.PreconditionDirections(&deriv, &scale);
    } catch (...) {
      int32 num_bad_rows = 0;
      for (int32 i = 0; i < out_deriv.NumRows(); i++) {
        BaseFloat f = out_deriv.Row(i).Sum();
        if (!(f - f == 0)) num_bad_rows++;
      }
      KALDI_ERR << "Preonditioning failed, in_value sum is "
                << in_value.Sum() << ", out_deriv sum is " << out_deriv.Sum()
                << ", out_deriv has " << num_bad_rows << " bad rows.";
    }
  }
  linear_params_.AddMat(learning_rate_ * scale,
                        deriv.ColRange(0, block_dim_in));
  bias_deriv.CopyColFromMat(deriv, block_dim_in);
  bias_params_.AddVec(learning_rate_ * scale, bias_deriv);
}

void NaturalGradientRepeatedAffineComponent::ConsolidateMemory() {
  OnlineNaturalGradient temp(preconditioner_in_);
  preconditioner_in_.Swap(&temp);
}


BlockAffineComponent::BlockAffineComponent(const BlockAffineComponent &other) :
  UpdatableComponent(other),
  linear_params_(other.linear_params_),
  bias_params_(other.bias_params_),
  num_blocks_(other.num_blocks_) {}

BlockAffineComponent::BlockAffineComponent(const RepeatedAffineComponent &rac) :
  UpdatableComponent(rac),
  linear_params_(rac.num_repeats_ * rac.linear_params_.NumRows(),
                 rac.linear_params_.NumCols(), kUndefined),
  bias_params_(rac.num_repeats_ * rac.linear_params_.NumRows(), kUndefined),
  num_blocks_(rac.num_repeats_) {
  // copy rac's linear_params_ and bias_params_ to this.
  int32 num_rows_in_block = rac.linear_params_.NumRows();
  for(int32 block_counter = 0; block_counter < num_blocks_; block_counter++) {
    int32 row_offset = block_counter * num_rows_in_block;
    CuSubMatrix<BaseFloat> block = this->linear_params_.RowRange(row_offset,
                                                                 num_rows_in_block);
    block.CopyFromMat(rac.linear_params_);
    CuSubVector<BaseFloat> block_bias = this->bias_params_.Range(row_offset,
                                                                 num_rows_in_block);
    block_bias.CopyFromVec(rac.bias_params_);
  }
}

Component* BlockAffineComponent::Copy() const {
  BlockAffineComponent *ans = new BlockAffineComponent(*this);
  return ans;
}

std::string BlockAffineComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", num-blocks=" << num_blocks_;
  PrintParameterStats(stream, "linear-params", linear_params_);
  PrintParameterStats(stream, "bias", bias_params_, true);
  return stream.str();
}

void BlockAffineComponent::Init(int32 input_dim,
                                int32 output_dim, int32 num_blocks,
                                BaseFloat param_stddev, BaseFloat bias_mean,
                                BaseFloat bias_stddev) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0 && num_blocks >= 1);
  KALDI_ASSERT(output_dim % num_blocks == 0 && input_dim % num_blocks == 0);
  const int32 num_columns_per_block = input_dim / num_blocks;
  linear_params_.Resize(output_dim, num_columns_per_block);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(param_stddev >= 0.0 && bias_stddev >= 0.0);
  linear_params_.SetRandn();
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  bias_params_.Add(bias_mean);
  num_blocks_ = num_blocks;
}

void BlockAffineComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = -1, output_dim = -1, num_blocks = -1;
  if(!cfl->GetValue("input-dim", &input_dim) ||
     !cfl->GetValue("output-dim", &output_dim) ||
     !cfl->GetValue("num-blocks", &num_blocks))
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  InitLearningRatesFromConfig(cfl);
  BaseFloat param_stddev = 1.0 / std::sqrt(input_dim / num_blocks),
      bias_mean = 0.0, bias_stddev = 1.0;
  cfl->GetValue("param-stddev", &param_stddev);
  cfl->GetValue("bias-stddev", &bias_stddev);
  cfl->GetValue("bias-mean", &bias_mean);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";

  Init(input_dim, output_dim, num_blocks,
       param_stddev, bias_mean, bias_stddev);
}

void* BlockAffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                     const CuMatrixBase<BaseFloat> &in,
                                     CuMatrixBase<BaseFloat> *out) const {
  out->CopyRowsFromVec(bias_params_);
  // block_dimension is both the number of columns, and the number of rows,
  // of a block.
  int32 num_rows_in_block = linear_params_.NumRows() / num_blocks_;
  int32 num_cols_in_block = linear_params_.NumCols();
  std::vector<CuSubMatrix<BaseFloat> *> in_batch, out_batch,
    linear_params_batch;
  for(int block_counter = 0; block_counter < num_blocks_; block_counter++) {
    CuSubMatrix<BaseFloat> *in_block =
      new CuSubMatrix<BaseFloat>(in.ColRange(block_counter * num_cols_in_block,
                                   num_cols_in_block));
    in_batch.push_back(in_block);

    CuSubMatrix<BaseFloat> *out_block =
      new CuSubMatrix<BaseFloat>(out->ColRange(block_counter * num_rows_in_block,
                                    num_rows_in_block));
    out_batch.push_back(out_block);

    CuSubMatrix<BaseFloat> *linear_params_block =
      new CuSubMatrix<BaseFloat>(linear_params_.RowRange(block_counter * num_rows_in_block,
                                              num_rows_in_block));
    linear_params_batch.push_back(linear_params_block);
  }
  AddMatMatBatched<BaseFloat>(1.0, out_batch, in_batch, kNoTrans,
                              linear_params_batch, kTrans, 1.0);

  DeletePointers(&in_batch);
  DeletePointers(&out_batch);
  DeletePointers(&linear_params_batch);
  return NULL;
}

void BlockAffineComponent::Backprop(const std::string &debug_info,
                                    const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &in_value,
                                    const CuMatrixBase<BaseFloat> &, // out_value
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    void *memo,
                                    Component *to_update_in,
                                    CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("BlockAffineComponent::Backprop");
  BlockAffineComponent *to_update = dynamic_cast<BlockAffineComponent*>(to_update_in);

  const int32 num_rows_in_block = linear_params_.NumRows() / num_blocks_;
  const int32 num_cols_in_block = linear_params_.NumCols();

  // Propagate the derivative back to the input.
  // add with coefficient 1.0 since property kBackpropAdds is true.
  // If we wanted to add with coefficient 0.0 we'd need to zero the
  // in_deriv, in case of infinities.
  if (in_deriv) {
    std::vector<CuSubMatrix<BaseFloat> *> in_deriv_batch, out_deriv_batch, linear_params_batch;

    for(int block_counter = 0; block_counter < num_blocks_; block_counter++) {
      CuSubMatrix<BaseFloat> *in_deriv_block =
        new CuSubMatrix<BaseFloat>(in_deriv->ColRange(block_counter * num_cols_in_block,
                                                      num_cols_in_block));
      in_deriv_batch.push_back(in_deriv_block);

      CuSubMatrix<BaseFloat> *out_deriv_block =
        new CuSubMatrix<BaseFloat>(out_deriv.ColRange(block_counter * num_rows_in_block,
                                                       num_rows_in_block));
      out_deriv_batch.push_back(out_deriv_block);

      CuSubMatrix<BaseFloat> *linear_params_block =
        new CuSubMatrix<BaseFloat>(linear_params_.RowRange(block_counter * num_rows_in_block,
                                                          num_rows_in_block));
      linear_params_batch.push_back(linear_params_block);
    }

    AddMatMatBatched<BaseFloat>(1.0, in_deriv_batch, out_deriv_batch, kNoTrans,
                                linear_params_batch, kNoTrans, 1.0);

    DeletePointers(&in_deriv_batch);
    DeletePointers(&out_deriv_batch);
    DeletePointers(&linear_params_batch);
  }

  if (to_update != NULL) {

    { // linear params update

      std::vector<CuSubMatrix<BaseFloat> *> in_value_batch,
        out_deriv_batch, linear_params_batch;

      for (int block_counter = 0; block_counter < num_blocks_; block_counter++) {
        CuSubMatrix<BaseFloat> *in_value_block =
          new CuSubMatrix<BaseFloat>(in_value.ColRange(block_counter * num_cols_in_block,
                                                       num_cols_in_block));
        in_value_batch.push_back(in_value_block);

        CuSubMatrix<BaseFloat> *out_deriv_block =
          new CuSubMatrix<BaseFloat>(out_deriv.ColRange(block_counter * num_rows_in_block,
                                                        num_rows_in_block));
        out_deriv_batch.push_back(out_deriv_block);

        CuSubMatrix<BaseFloat> *linear_params_block =
          new CuSubMatrix<BaseFloat>(to_update->linear_params_.RowRange(block_counter * num_rows_in_block,
                                                                        num_rows_in_block));
        linear_params_batch.push_back(linear_params_block);
      }

      AddMatMatBatched<BaseFloat>(to_update->learning_rate_,
                                  linear_params_batch,
                                  out_deriv_batch, kTrans,
                                  in_value_batch, kNoTrans, 1.0);

      DeletePointers(&in_value_batch);
      DeletePointers(&out_deriv_batch);
      DeletePointers(&linear_params_batch);
    } // end linear params update

    { // bias update
      to_update->bias_params_.AddRowSumMat(to_update->learning_rate_,
                                           out_deriv, 1.0);
    } // end bias update
  }
}

void BlockAffineComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    linear_params_.SetZero();
    bias_params_.SetZero();
  } else {
    linear_params_.Scale(scale);
    bias_params_.Scale(scale);
  }
}

void BlockAffineComponent::Add(BaseFloat alpha, const Component &other_in) {
  const BlockAffineComponent *other =
    dynamic_cast<const BlockAffineComponent *>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

void BlockAffineComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);

  CuVector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

BaseFloat BlockAffineComponent::DotProduct(const UpdatableComponent &other_in) const {
  const BlockAffineComponent *other =
    dynamic_cast<const BlockAffineComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans) +
    VecVec(bias_params_, other->bias_params_);
}

void BlockAffineComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<NumBlocks>");
  ReadBasicType(is, binary, &num_blocks_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  if (PeekToken(is, binary) == 'I') {
    // for back compatibility; we don't write this here any
    // more as it's written and read in Write/ReadUpdatableCommon
    ExpectToken(is, binary, "<IsGradient>");
    ReadBasicType(is, binary, &is_gradient_);
  }
  ExpectToken(is, binary, "</BlockAffineComponent>");
}

void BlockAffineComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<NumBlocks>");
  WriteBasicType(os, binary, num_blocks_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "</BlockAffineComponent>");
}

int32 BlockAffineComponent::NumParameters() const {
  return linear_params_.NumCols() * linear_params_.NumRows() + bias_params_.Dim();
}

void BlockAffineComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  int32 num_linear_params = linear_params_.NumCols() * linear_params_.NumRows();
  int32 num_bias_params = bias_params_.Dim();
  params->Range(0, num_linear_params).CopyRowsFromMat(linear_params_);
  params->Range(num_linear_params, num_bias_params).CopyFromVec(bias_params_);
}

void BlockAffineComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  int32 num_linear_params = linear_params_.NumCols() * linear_params_.NumRows();
  int32 num_bias_params = bias_params_.Dim();
  linear_params_.CopyRowsFromVec(params.Range(0, num_linear_params));
  bias_params_.CopyFromVec(params.Range(num_linear_params, num_bias_params));
}

void PerElementScaleComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    scales_.SetZero();
  } else {
    scales_.Scale(scale);
  }
}

void PerElementScaleComponent::Add(BaseFloat alpha,
                                   const Component &other_in) {
  const PerElementScaleComponent *other =
      dynamic_cast<const PerElementScaleComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  scales_.AddVec(alpha, other->scales_);
}

PerElementScaleComponent::PerElementScaleComponent(
    const PerElementScaleComponent &component):
    UpdatableComponent(component),
    scales_(component.scales_) { }

void PerElementScaleComponent::PerturbParams(BaseFloat stddev) {
  CuVector<BaseFloat> temp_scales(scales_.Dim(), kUndefined);
  temp_scales.SetRandn();
  scales_.AddVec(stddev, temp_scales);
}

std::string PerElementScaleComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", scales-min=" << scales_.Min()
         << ", scales-max=" << scales_.Max();
  PrintParameterStats(stream, "scales", scales_, true);
  return stream.str();
}

Component* PerElementScaleComponent::Copy() const {
  return new PerElementScaleComponent(*this);
}

BaseFloat PerElementScaleComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const PerElementScaleComponent *other =
      dynamic_cast<const PerElementScaleComponent*>(&other_in);
  return VecVec(scales_, other->scales_);
}

void PerElementScaleComponent::Init(int32 dim,
                                    BaseFloat param_mean,
                                    BaseFloat param_stddev) {
  KALDI_ASSERT(dim > 0 && param_stddev >= 0.0);
  scales_.Resize(dim);
  scales_.SetRandn();
  scales_.Scale(param_stddev);
  scales_.Add(param_mean);
}

void PerElementScaleComponent::Init(std::string vector_filename) {
  CuVector<BaseFloat> vec;
  ReadKaldiObject(vector_filename, &vec); // will abort on failure.
  scales_.Resize(vec.Dim());
  scales_.CopyFromVec(vec);
}

void PerElementScaleComponent::InitFromConfig(ConfigLine *cfl) {
  std::string vector_filename;
  int32 dim = -1;
  InitLearningRatesFromConfig(cfl);
  if (cfl->GetValue("vector", &vector_filename)) {
    Init(vector_filename);
    if (cfl->GetValue("dim", &dim))
      KALDI_ASSERT(dim == InputDim() &&
                   "input-dim mismatch vs. vector.");
  } else {
    if(!cfl->GetValue("dim", &dim))
      KALDI_ERR << "'dim' not provided in the config line.";
    BaseFloat param_mean = 1.0, param_stddev = 0.0;
    cfl->GetValue("param-mean", &param_mean);
    cfl->GetValue("param-stddev", &param_stddev);
    Init(dim, param_mean, param_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
}

void* PerElementScaleComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
  out->MulColsVec(scales_);
  return NULL;
}

void PerElementScaleComponent::UpdateSimple(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  scales_.AddDiagMatMat(learning_rate_, out_deriv, kTrans,
                        in_value, kNoTrans, 1.0);
}

void PerElementScaleComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("PerElementScaleComponent::Backprop");
  PerElementScaleComponent *to_update =
      dynamic_cast<PerElementScaleComponent*>(to_update_in);

  if (to_update != NULL) {
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    if (to_update->is_gradient_)
      to_update->UpdateSimple(in_value, out_deriv);
    else  // the call below is to a virtual function that may be re-implemented
      to_update->Update(debug_info, in_value, out_deriv);  // by child classes.
  }

  if (in_deriv) {
    // Propagate the derivative back to the input.
    if (in_deriv->Data() != out_deriv.Data())
      in_deriv->CopyFromMat(out_deriv);
    in_deriv->MulColsVec(scales_);
  }
}

void PerElementScaleComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // Read opening tag and learning rate.
  ExpectToken(is, binary, "<Params>");
  scales_.Read(is, binary);
  if (PeekToken(is, binary) == 'I') {
    // for back compatibility; we don't write this here any
    // more as it's written and read in Write/ReadUpdatableCommon
    ExpectToken(is, binary, "<IsGradient>");
    ReadBasicType(is, binary, &is_gradient_);
  }
  ExpectToken(is, binary, "</PerElementScaleComponent>");
}

void PerElementScaleComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate.
  WriteToken(os, binary, "<Params>");
  scales_.Write(os, binary);
  WriteToken(os, binary, "</PerElementScaleComponent>");
}

int32 PerElementScaleComponent::NumParameters() const {
  return InputDim();
}

void PerElementScaleComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  params->CopyFromVec(scales_);
}

void PerElementScaleComponent::UnVectorize(
    const VectorBase<BaseFloat> &params) {
  scales_.CopyFromVec(params);
}

void PerElementOffsetComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    offsets_.SetZero();
  } else {
    offsets_.Scale(scale);
  }
}


void PerElementOffsetComponent::Add(BaseFloat alpha,
                                   const Component &other_in) {
  const PerElementOffsetComponent *other =
      dynamic_cast<const PerElementOffsetComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  offsets_.AddVec(alpha, other->offsets_);
}

PerElementOffsetComponent::PerElementOffsetComponent(
    const PerElementOffsetComponent &component):
    UpdatableComponent(component),
    offsets_(component.offsets_),
    dim_(component.dim_),
    use_natural_gradient_(component.use_natural_gradient_),
    preconditioner_(component.preconditioner_) { }

void PerElementOffsetComponent::PerturbParams(BaseFloat stddev) {
  CuVector<BaseFloat> temp_offsets(offsets_.Dim(), kUndefined);
  temp_offsets.SetRandn();
  offsets_.AddVec(stddev, temp_offsets);
}

std::string PerElementOffsetComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", offsets-min=" << offsets_.Min()
         << ", offsets-max=" << offsets_.Max()
         << ", block-dim=" << offsets_.Dim()
         << ", use-natural-gradient="
         << (use_natural_gradient_ ? "true" : "false");
  PrintParameterStats(stream, "offsets", offsets_, true);
  return stream.str();
}

Component* PerElementOffsetComponent::Copy() const {
  return new PerElementOffsetComponent(*this);
}

BaseFloat PerElementOffsetComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const PerElementOffsetComponent *other =
      dynamic_cast<const PerElementOffsetComponent*>(&other_in);
  return VecVec(offsets_, other->offsets_);
}


void PerElementOffsetComponent::InitFromConfig(ConfigLine *cfl) {
  std::string vector_filename;
  InitLearningRatesFromConfig(cfl);
  if (cfl->GetValue("vector", &vector_filename)) {
    ReadKaldiObject(vector_filename, &offsets_);
    dim_ = offsets_.Dim();  // if dim is not supplied, it defaults to this.
    cfl->GetValue("dim", &dim_);
    if (dim_ <= 0 || offsets_.Dim() % dim_ != 0)
      KALDI_ERR << "Invalid dimension dim=" << dim_;
  } else {
    if(!cfl->GetValue("dim", &dim_))
      KALDI_ERR << "'dim' not provided in the config line.";
    if (dim_ <= 0)
      KALDI_ERR << "Invalid dimension dim=" << dim_;
    BaseFloat param_mean = 0.0, param_stddev = 0.0;
    cfl->GetValue("param-mean", &param_mean);
    cfl->GetValue("param-stddev", &param_stddev);
    int32 block_dim = dim_;
    cfl->GetValue("block-dim", &block_dim);
    if (block_dim <= 0 || dim_ % block_dim !=  0)
      KALDI_ERR << "Invalid value block-dim=" << block_dim;
    offsets_.Resize(block_dim);
    offsets_.SetRandn();
    offsets_.Scale(param_stddev);
    offsets_.Add(param_mean);
  }
  use_natural_gradient_ = true;
  cfl->GetValue("use-natural-gradient", &use_natural_gradient_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  // For now you can't modify these defaults of the natural gradient.
  // This code must be kept in sync with the code in Read().
  preconditioner_.SetRank(20);
  preconditioner_.SetUpdatePeriod(4);
}

void* PerElementOffsetComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  if (in.Data() != out->Data())
    out->CopyFromMat(in);
  if (dim_ == offsets_.Dim()) {
    out->AddVecToRows(1.0, offsets_);
  } else {
    KALDI_ASSERT(out->Stride() == out->NumCols());
    int32 block_dim = offsets_.Dim(), multiple = dim_ / block_dim,
        num_rows = out->NumRows() * multiple;
    CuSubMatrix<BaseFloat> out_rearranged(out->Data(), num_rows,
                                          block_dim, block_dim);
    out_rearranged.AddVecToRows(1.0, offsets_);
  }
  return NULL;
}

void PerElementOffsetComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &, // in_value
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("PerElementOffsetComponent::Backprop");
  PerElementOffsetComponent *to_update =
      dynamic_cast<PerElementOffsetComponent*>(to_update_in);

  if (in_deriv && in_deriv->Data() != out_deriv.Data()) {
    // Propagate the derivative back to the input.
    in_deriv->CopyFromMat(out_deriv);
  }

  if (to_update != NULL) {
    // we may have to reshape out_deriv, if "block-dim" was set
    // in the config file when initializing the object, leading
    // to dim_ being a multiple >1 of offset_.Dim().
    // To avoid having separate code paths we create a sub-matrix
    // in any case, but this may just be a copy of out_deriv.
    int32 block_dim = offsets_.Dim(), multiple = dim_ / block_dim,
        block_stride = (multiple == 1 ? out_deriv.Stride() : block_dim),
        num_rows = out_deriv.NumRows() * multiple;
    KALDI_ASSERT(multiple == 1 || out_deriv.Stride() == out_deriv.NumCols());
    CuSubMatrix<BaseFloat> out_deriv_reshaped(out_deriv.Data(), num_rows,
                                              block_dim, block_stride);
    if (!to_update->use_natural_gradient_ || to_update->is_gradient_) {
      KALDI_LOG << "Using non-NG update, lr = " << to_update->learning_rate_;
      to_update->offsets_.AddRowSumMat(to_update->learning_rate_,
                                       out_deriv_reshaped);
    } else {
      KALDI_LOG << "Using NG update, lr = " << to_update->learning_rate_;
      // make a copy as we don't want to modify the data of 'out_deriv', which
      // was const (even though CuSubMatrix does not respect const-ness in
      // this scenario)
      CuMatrix<BaseFloat> out_deriv_copy(out_deriv_reshaped);
      BaseFloat scale = 1.0;
      to_update->preconditioner_.PreconditionDirections(&out_deriv_copy,
                                                        &scale);
      to_update->offsets_.AddRowSumMat(scale * to_update->learning_rate_,
                                       out_deriv_copy);
    }
  }
}

void PerElementOffsetComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // Read opening tag and learning rate
  ExpectToken(is, binary, "<Offsets>");
  offsets_.Read(is, binary);
  if (PeekToken(is, binary) == 'I') {
    // for back compatibility; we don't write this here any
    // more as it's written and read in Write/ReadUpdatableCommon
    ExpectToken(is, binary, "<IsGradient>");
    ReadBasicType(is, binary, &is_gradient_);
  }
  if (PeekToken(is, binary) != '/') {
    ExpectToken(is, binary, "<Dim>");
    ReadBasicType(is, binary, &dim_);
    ExpectToken(is, binary, "<UseNaturalGradient>");
    ReadBasicType(is, binary, &use_natural_gradient_);
  } else {
    dim_ = offsets_.Dim();
    use_natural_gradient_ = true;
  }
  // For now you can't modify these defaults of the natural gradient.
  // This code must be kept in sync with the code in InitFromConfig().
  preconditioner_.SetRank(20);
  preconditioner_.SetUpdatePeriod(4);
  ExpectToken(is, binary, "</PerElementOffsetComponent>");
}

void PerElementOffsetComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<Offsets>");
  offsets_.Write(os, binary);
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<UseNaturalGradient>");
  WriteBasicType(os, binary, use_natural_gradient_);
  WriteToken(os, binary, "</PerElementOffsetComponent>");
}

int32 PerElementOffsetComponent::NumParameters() const {
  return offsets_.Dim();
}

void PerElementOffsetComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  params->CopyFromVec(offsets_);
}

void PerElementOffsetComponent::UnVectorize(
    const VectorBase<BaseFloat> &params) {
  offsets_.CopyFromVec(params);
}

std::string ScaleAndOffsetComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", rank=" << scale_preconditioner_.GetRank();
  if (dim_ != scales_.Dim())
    stream << ", block-size=" << scales_.Dim();
  PrintParameterStats(stream, "scales", scales_, true);
  PrintParameterStats(stream, "offsets", offsets_, true);
  return stream.str();
}

void ScaleAndOffsetComponent::InitFromConfig(ConfigLine *cfl) {

  InitLearningRatesFromConfig(cfl);
  if (!cfl->GetValue("dim", &dim_) || dim_ <= 0) {
    KALDI_ERR << "Dimension 'dim' must be specified and >0: "
              << cfl->WholeLine();
  }
  use_natural_gradient_ = true;
  cfl->GetValue("use-natural-gradient", &use_natural_gradient_);
  int32 block_dim = dim_,
      rank = 20;
  cfl->GetValue("block-dim", &block_dim);
  if (block_dim <= 0 || dim_ % block_dim != 0) {
    KALDI_ERR << "Invalid block-dim: " << cfl->WholeLine();
  }
  cfl->GetValue("rank", &rank);
  scales_.Resize(block_dim);
  scales_.Set(1.0);
  offsets_.Resize(block_dim);
  // offsets are all zero when initialized.
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  offset_preconditioner_.SetRank(rank);
  scale_preconditioner_.SetRank(rank);
  // the update period can't be configured for now; we'll add an option if we
  // want to.
  offset_preconditioner_.SetUpdatePeriod(4);
  scale_preconditioner_.SetUpdatePeriod(4);
}

void ScaleAndOffsetComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // Read opening tag and learning rate
  ExpectToken(is, binary, "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<Scales>");
  scales_.Read(is, binary);
  ExpectToken(is, binary, "<Offsets>");
  offsets_.Read(is, binary);
  ExpectToken(is, binary, "<UseNaturalGradient>");
  ReadBasicType(is, binary, &use_natural_gradient_);
  int32 rank;
  ExpectToken(is, binary, "<Rank>");
  ReadBasicType(is, binary, &rank);
  scale_preconditioner_.SetRank(rank);
  offset_preconditioner_.SetRank(rank);
  ExpectToken(is, binary, "</ScaleAndOffsetComponent>");
}

void ScaleAndOffsetComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<Scales>");
  scales_.Write(os, binary);
  WriteToken(os, binary, "<Offsets>");
  offsets_.Write(os, binary);
  WriteToken(os, binary, "<UseNaturalGradient>");
  WriteBasicType(os, binary, use_natural_gradient_);
  WriteToken(os, binary, "<Rank>");
  WriteBasicType(os, binary, scale_preconditioner_.GetRank());
  WriteToken(os, binary, "</ScaleAndOffsetComponent>");
}

void ScaleAndOffsetComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    scales_.SetZero();
    offsets_.SetZero();
  } else {
    scales_.Scale(scale);
    offsets_.Scale(scale);
  }
}

void ScaleAndOffsetComponent::Add(BaseFloat alpha,
                                  const Component &other_in) {
  const ScaleAndOffsetComponent *other =
      dynamic_cast<const ScaleAndOffsetComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  scales_.AddVec(alpha, other->scales_);
  offsets_.AddVec(alpha, other->offsets_);
}

ScaleAndOffsetComponent::ScaleAndOffsetComponent(
    const ScaleAndOffsetComponent &component):
    UpdatableComponent(component),
    dim_(component.dim_),
    scales_(component.scales_),
    offsets_(component.offsets_),
    use_natural_gradient_(component.use_natural_gradient_),
    scale_preconditioner_(component.scale_preconditioner_),
    offset_preconditioner_(component.offset_preconditioner_) { }

void ScaleAndOffsetComponent::PerturbParams(BaseFloat stddev) {
  CuVector<BaseFloat> temp(scales_.Dim(), kUndefined);
  temp.SetRandn();
  scales_.AddVec(stddev, temp);
  temp.SetRandn();
  offsets_.AddVec(stddev, temp);
}

BaseFloat ScaleAndOffsetComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const ScaleAndOffsetComponent *other =
      dynamic_cast<const ScaleAndOffsetComponent*>(&other_in);
  return VecVec(other->scales_, scales_) + VecVec(other->offsets_, offsets_);
}

void ScaleAndOffsetComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  int32 dim = scales_.Dim();
  params->Range(0, dim).CopyFromVec(scales_);
  params->Range(dim, dim).CopyFromVec(offsets_);
}

void ScaleAndOffsetComponent::UnVectorize(
    const VectorBase<BaseFloat> &params) {
  int32 dim = scales_.Dim();
  scales_.CopyFromVec(params.Range(0, dim));
  offsets_.CopyFromVec(params.Range(dim, dim));
}

void* ScaleAndOffsetComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  if (dim_ == scales_.Dim()) {
    PropagateInternal(in, out);
  } else {
    int32 multiple = dim_ / scales_.Dim(),
        num_rows = in.NumRows(), block_dim = scales_.Dim();
    KALDI_ASSERT(in.NumCols() == in.Stride() &&
                 SameDimAndStride(in, *out));
    // Reinterpret the data as matrices with more rows but fewer columns.
    CuSubMatrix<BaseFloat> in_rearranged(in.Data(), num_rows * multiple,
                                         block_dim, block_dim),
        out_rearranged(out->Data(), num_rows * multiple,
                       block_dim, block_dim);
    PropagateInternal(in_rearranged, &out_rearranged);
  }
  return NULL;
}

void ScaleAndOffsetComponent::PropagateInternal(
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  if (out->Data() != in.Data())
    out->CopyFromMat(in);
  BaseFloat epsilon = Epsilon();
  int32 dim = scales_.Dim();
  CuVector<BaseFloat> scales_nonzero(dim, kUndefined);
  cu::EnsureNonzero(scales_, epsilon, &scales_nonzero);
  out->MulColsVec(scales_nonzero);
  out->AddVecToRows(1.0, offsets_);
}

void ScaleAndOffsetComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &, // in_value
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("ScaleAndOffsetComponent::Backprop");
  ScaleAndOffsetComponent *to_update =
      dynamic_cast<ScaleAndOffsetComponent*>(to_update_in);

  KALDI_ASSERT(SameDim(out_value, out_deriv));

  if (dim_ == scales_.Dim()) {
    BackpropInternal(debug_info, out_value, out_deriv,
                     to_update, in_deriv);
  } else {
    KALDI_ASSERT(out_value.NumCols() == out_value.Stride() &&
                 SameDimAndStride(out_value, out_deriv) &&
                 (!in_deriv || SameDimAndStride(out_value, *in_deriv)));
    int32 multiple = dim_ / scales_.Dim(),
        num_rows = out_value.NumRows(),
        block_dim = scales_.Dim();
    CuSubMatrix<BaseFloat> out_value_rearranged(out_value.Data(),
                                                num_rows * multiple,
                                                block_dim, block_dim),
        out_deriv_rearranged(out_deriv.Data(), num_rows * multiple,
                             block_dim, block_dim);
    if (in_deriv) {
      CuSubMatrix<BaseFloat> in_deriv_rearranged(in_deriv->Data(),
                                                 num_rows * multiple,
                                                 block_dim, block_dim);
      BackpropInternal(debug_info, out_value_rearranged,
                       out_deriv_rearranged, to_update,
                       &in_deriv_rearranged);
    } else {
      BackpropInternal(debug_info, out_value_rearranged,
                       out_deriv_rearranged, to_update,
                       NULL);
    }
  }
}


  // Internal version of backprop, where the num-cols of the
  // argument matrices are equal to scales_.Dim().
void ScaleAndOffsetComponent::BackpropInternal(
    const std::string &debug_info,
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    ScaleAndOffsetComponent *to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  if (to_update) {
    if (!to_update->use_natural_gradient_ || to_update->is_gradient_) {
      to_update->offsets_.AddRowSumMat(to_update->learning_rate_,
                                       out_deriv);
    } else {
      BaseFloat scale = 1.0;
      CuMatrix<BaseFloat> out_deriv_copy(out_deriv);
      to_update->offset_preconditioner_.PreconditionDirections(
          &out_deriv_copy, &scale);
      to_update->offsets_.AddRowSumMat(scale * to_update->learning_rate_,
                                       out_deriv_copy);
    }
    // The backprop actually needs the input to the component, not the output;
    // but we make the output available because in the common topologies that
    // will already be required for backprop-- it's for memory efficiency.
    CuMatrix<BaseFloat> in_value_reconstructed(out_value);
    int32 dim = scales_.Dim();
    CuVector<BaseFloat> scales_nonzero(dim, kUndefined);
    BaseFloat epsilon = Epsilon();
    cu::EnsureNonzero(scales_, epsilon, &scales_nonzero);
    scales_nonzero.InvertElements();
    in_value_reconstructed.AddVecToRows(-1.0, offsets_);
    // Actually scales_nonzero are now the inverses of the scales.
    in_value_reconstructed.MulColsVec(scales_nonzero);
    // OK, at this point in_value_reconstructed is the input to the component.
    // Multiply its elements by 'out_deriv' to get the derivatives
    // (for each frame) w.r.t. the scales.
    in_value_reconstructed.MulElements(out_deriv);
    BaseFloat scale = 1.0;
    if (to_update->use_natural_gradient_ && !to_update->is_gradient_) {
      to_update->scale_preconditioner_.PreconditionDirections(
          &in_value_reconstructed, &scale);
    }
    to_update->scales_.AddRowSumMat(scale * to_update->learning_rate_,
                                    in_value_reconstructed);
  }
  if (in_deriv) {
    if (in_deriv->Data() != out_deriv.Data())
      in_deriv->CopyFromMat(out_deriv);
    in_deriv->MulColsVec(scales_);
  }
}

void ScaleAndOffsetComponent::ConsolidateMemory() {
  OnlineNaturalGradient temp_scale(scale_preconditioner_);
  scale_preconditioner_.Swap(&temp_scale);
  OnlineNaturalGradient temp_offset(offset_preconditioner_);
  offset_preconditioner_.Swap(&temp_offset);
}


std::string ConstantFunctionComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", " << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim()
         << ", is-updatable=" << std::boolalpha << is_updatable_
         << ", use-natural-gradient=" << std::boolalpha
         << use_natural_gradient_;
  PrintParameterStats(stream, "output", output_, true);
  return stream.str();
}

ConstantFunctionComponent::ConstantFunctionComponent():
    UpdatableComponent(), input_dim_(-1), is_updatable_(true),
    use_natural_gradient_(true) { }

ConstantFunctionComponent::ConstantFunctionComponent(
    const ConstantFunctionComponent &other):
    UpdatableComponent(other), input_dim_(other.input_dim_),
    output_(other.output_), is_updatable_(other.is_updatable_),
    use_natural_gradient_(other.use_natural_gradient_),
    preconditioner_(other.preconditioner_) { }

void* ConstantFunctionComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  out->CopyRowsFromVec(output_);
  return NULL;
}

void ConstantFunctionComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &, // in_value
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("ConstantFunctionComponent::Backprop");
  // we don't update in_deriv, since we set the flag
  // kBackpropAdds, and the output doesn't depend on the
  // input, so the input-derivative is zero.
  if (to_update_in) {
    ConstantFunctionComponent *to_update =
      dynamic_cast<ConstantFunctionComponent*>(to_update_in);
    if (to_update->is_updatable_) {
      // only do the update if the is_updatable_ flag is set.
      KALDI_ASSERT(to_update && to_update->is_updatable_);
      if (to_update->use_natural_gradient_ && !to_update->is_gradient_) {
        CuMatrix<BaseFloat> out_deriv_copy(out_deriv);
        BaseFloat scale = 1.0;
        to_update->preconditioner_.PreconditionDirections(&out_deriv_copy,
                                                          &scale);
        to_update->output_.AddRowSumMat(scale * to_update->learning_rate_,
                                        out_deriv_copy);
      } else {
        to_update->output_.AddRowSumMat(to_update->learning_rate_,
                                        out_deriv);
      }
    }
  }
}

void ConstantFunctionComponent::Read(std::istream &is, bool binary) {
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<ConstantFunctionComponent>") {
    ReadToken(is, binary, &token);
  }
  if (token == "<LearningRateFactor>") {
    ReadBasicType(is, binary, &learning_rate_factor_);
    ReadToken(is, binary, &token);
  } else {
    learning_rate_factor_ = 1.0;
  }
  if (token == "<IsGradient>") {
    ReadBasicType(is, binary, &is_gradient_);
    ReadToken(is, binary, &token);
  } else {
    is_gradient_ = false;
  }
  if (token == "<LearningRate>") {
    ReadBasicType(is, binary, &learning_rate_);
    ReadToken(is, binary, &token);
  } else {
    learning_rate_ = 0.001;
  }
  if (token == "<InputDim>") {
    ReadBasicType(is, binary, &input_dim_);
  } else {
    KALDI_ERR << "Expected token <InputDim>, got "
              << token;
  }
  ExpectToken(is, binary, "<Output>");
  output_.Read(is, binary);
  ExpectToken(is, binary, "<IsUpdatable>");
  ReadBasicType(is, binary, &is_updatable_);
  ExpectToken(is, binary, "<UseNaturalGradient>");
  ReadBasicType(is, binary, &use_natural_gradient_);
  ExpectToken(is, binary, "</ConstantFunctionComponent>");
}

void ConstantFunctionComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<Output>");
  output_.Write(os, binary);
  WriteToken(os, binary, "<IsUpdatable>");
  WriteBasicType(os, binary, is_updatable_);
  WriteToken(os, binary, "<UseNaturalGradient>");
  WriteBasicType(os, binary, use_natural_gradient_);
  WriteToken(os, binary, "</ConstantFunctionComponent>");
}

Component* ConstantFunctionComponent::Copy() const {
  return new ConstantFunctionComponent(*this);
}

void ConstantFunctionComponent::Scale(BaseFloat scale) {
  if (is_updatable_) {
    if (scale == 0.0) {
      output_.SetZero();
    } else {
      output_.Scale(scale);
    }
  }
}

void ConstantFunctionComponent::Add(BaseFloat alpha, const Component &other_in) {
  if (is_updatable_) {
    const ConstantFunctionComponent *other =
        dynamic_cast<const ConstantFunctionComponent*>(&other_in);
    KALDI_ASSERT(other != NULL);
    output_.AddVec(alpha, other->output_);
  }
}

void ConstantFunctionComponent::PerturbParams(BaseFloat stddev) {
  CuVector<BaseFloat> temp_output(output_.Dim(), kUndefined);
  temp_output.SetRandn();
  output_.AddVec(stddev, temp_output);
}

BaseFloat ConstantFunctionComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  KALDI_ASSERT(is_updatable_);
  const ConstantFunctionComponent *other =
      dynamic_cast<const ConstantFunctionComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  return VecVec(output_, other->output_);
}

void ConstantFunctionComponent::InitFromConfig(ConfigLine *cfl) {
  int32 output_dim = 0;
  InitLearningRatesFromConfig(cfl);
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim_);
  cfl->GetValue("is-updatable", &is_updatable_);
  cfl->GetValue("use-natural-gradient", &use_natural_gradient_);
  BaseFloat output_mean = 0.0, output_stddev = 0.0;
  cfl->GetValue("output-mean", &output_mean);
  cfl->GetValue("output-stddev", &output_stddev);
  if (!ok || cfl->HasUnusedValues() || input_dim_ <= 0 ||
      output_dim <= 0) {
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
  }
  Vector<BaseFloat> output(output_dim);
  output.SetRandn();
  output.Scale(output_stddev);
  output.Add(output_mean);
  output_ = output;
}

int32 ConstantFunctionComponent::NumParameters() const {
  KALDI_ASSERT(is_updatable_);
  return output_.Dim();
}

void ConstantFunctionComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  params->CopyFromVec(output_);
}

void ConstantFunctionComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  output_.CopyFromVec(params);
}

void ConstantFunctionComponent::ConsolidateMemory() {
  OnlineNaturalGradient temp(preconditioner_);
  preconditioner_.Swap(&temp);
}

void NaturalGradientAffineComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // Read the opening tag and learning rate
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);

  BaseFloat num_samples_history, alpha;
  int32 rank_in, rank_out, update_period;

  ExpectToken(is, binary, "<RankIn>");
  ReadBasicType(is, binary, &rank_in);
  ExpectToken(is, binary, "<RankOut>");
  ReadBasicType(is, binary, &rank_out);
  if (PeekToken(is, binary) == 'O') {
    ExpectToken(is, binary, "<OrthonormalConstraint>");
    ReadBasicType(is, binary, &orthonormal_constraint_);
  } else {
    orthonormal_constraint_ = 0.0;
  }
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period);
  ExpectToken(is, binary, "<NumSamplesHistory>");
  ReadBasicType(is, binary, &num_samples_history);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha);

  preconditioner_in_.SetNumSamplesHistory(num_samples_history);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history);
  preconditioner_in_.SetAlpha(alpha);
  preconditioner_out_.SetAlpha(alpha);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetUpdatePeriod(update_period);
  preconditioner_out_.SetUpdatePeriod(update_period);

  if (PeekToken(is, binary) == 'M') {
    // MaxChangePerSample, long ago removed; back compatibility.
    ExpectToken(is, binary, "<MaxChangePerSample>");
    BaseFloat temp;
    ReadBasicType(is, binary, &temp);
  }
  if (PeekToken(is, binary) == 'I') {
    // for back compatibility; we don't write this here any
    // more as it's written and read in Write/ReadUpdatableCommon
    ExpectToken(is, binary, "<IsGradient>");
    ReadBasicType(is, binary, &is_gradient_);
  }
  if (PeekToken(is, binary) == 'U') {
    ExpectToken(is, binary, "<UpdateCount>");
    // back-compatibility branch (these configs were added and then removed).
    double temp;
    ReadBasicType(is, binary, &temp);
    ExpectToken(is, binary, "<ActiveScalingCount>");
    ReadBasicType(is, binary, &temp);
    ExpectToken(is, binary, "<MaxChangeScaleStats>");
    ReadBasicType(is, binary, &temp);
  }
  std::string token;
  ReadToken(is, binary, &token);
  // the following has to handle a couple variants of
  if (token.find("NaturalGradientAffineComponent>") == std::string::npos)
    KALDI_ERR << "Expected <NaturalGradientAffineComponent> or "
              << "</NaturalGradientAffineComponent>, got " << token;
}


NaturalGradientAffineComponent::NaturalGradientAffineComponent(
    const CuMatrixBase<BaseFloat> &linear_params,
    const CuVectorBase<BaseFloat> &bias_params):
    AffineComponent(linear_params, bias_params, 0.001) {
  KALDI_ASSERT(bias_params.Dim() == linear_params.NumRows() &&
               bias_params.Dim() != 0);

  // set some default natural gradient configs.
  preconditioner_in_.SetRank(20);
  preconditioner_out_.SetRank(80);
  preconditioner_in_.SetUpdatePeriod(4);
  preconditioner_out_.SetUpdatePeriod(4);
}

void NaturalGradientAffineComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;

  is_gradient_ = false;  // not configurable; there's no reason you'd want this

  InitLearningRatesFromConfig(cfl);

  if (cfl->GetValue("matrix", &matrix_filename)) {
    CuMatrix<BaseFloat> mat;
    ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
    KALDI_ASSERT(mat.NumCols() >= 2);
    int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
    linear_params_.Resize(output_dim, input_dim);
    bias_params_.Resize(output_dim);
    linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
    bias_params_.CopyColFromMat(mat, input_dim);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    int32 input_dim = -1, output_dim = -1;

    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    if (!ok)
      KALDI_ERR << "Bad initializer " << cfl->WholeLine();
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 1.0, bias_mean = 0.0;
    cfl->GetValue("param-stddev", &param_stddev);
    cfl->GetValue("bias-stddev", &bias_stddev);
    cfl->GetValue("bias-mean", &bias_mean);
    linear_params_.Resize(output_dim, input_dim);
    bias_params_.Resize(output_dim);
    KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0 &&
                 bias_stddev >= 0.0);
    linear_params_.SetRandn(); // sets to random normally distributed noise.
    linear_params_.Scale(param_stddev);
    bias_params_.SetRandn();
    bias_params_.Scale(bias_stddev);
    bias_params_.Add(bias_mean);
  }

  orthonormal_constraint_ = 0.0;
  cfl->GetValue("orthonormal-constraint", &orthonormal_constraint_);

  // Set natural-gradient configs.
  BaseFloat num_samples_history = 2000.0,
      alpha = 4.0;
  int32 rank_in = -1, rank_out = -1,
      update_period = 4;
  cfl->GetValue("num-samples-history", &num_samples_history);
  cfl->GetValue("alpha", &alpha);
  cfl->GetValue("rank-in", &rank_in);
  cfl->GetValue("rank-out", &rank_out);
  cfl->GetValue("update-period", &update_period);

  if (rank_in < 0)
    rank_in = std::min<int32>(20, (InputDim() + 1) / 2);
  if (rank_out < 0)
    rank_out = std::min<int32>(80, (OutputDim() + 1) / 2);

  preconditioner_in_.SetNumSamplesHistory(num_samples_history);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history);
  preconditioner_in_.SetAlpha(alpha);
  preconditioner_out_.SetAlpha(alpha);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetUpdatePeriod(update_period);
  preconditioner_out_.SetUpdatePeriod(update_period);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void NaturalGradientAffineComponent::Write(std::ostream &os,
                                           bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<RankIn>");
  WriteBasicType(os, binary, preconditioner_in_.GetRank());
  WriteToken(os, binary, "<RankOut>");
  WriteBasicType(os, binary, preconditioner_out_.GetRank());
  if (orthonormal_constraint_ != 0.0) {
    WriteToken(os, binary, "<OrthonormalConstraint>");
    WriteBasicType(os, binary, orthonormal_constraint_);
  }
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, preconditioner_in_.GetUpdatePeriod());
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, preconditioner_in_.GetNumSamplesHistory());
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, preconditioner_in_.GetAlpha());
  WriteToken(os, binary, "</NaturalGradientAffineComponent>");
}

std::string NaturalGradientAffineComponent::Info() const {
  std::ostringstream stream;
  stream << AffineComponent::Info();
  stream << ", rank-in=" << preconditioner_in_.GetRank()
         << ", rank-out=" << preconditioner_out_.GetRank()
         << ", num-samples-history=" << preconditioner_in_.GetNumSamplesHistory()
         << ", update-period=" << preconditioner_in_.GetUpdatePeriod()
         << ", alpha=" << preconditioner_in_.GetAlpha();
  return stream.str();
}

Component* NaturalGradientAffineComponent::Copy() const {
  return new NaturalGradientAffineComponent(*this);
}

NaturalGradientAffineComponent::NaturalGradientAffineComponent(
    const NaturalGradientAffineComponent &other):
    AffineComponent(other),
    preconditioner_in_(other.preconditioner_in_),
    preconditioner_out_(other.preconditioner_out_) { }

void NaturalGradientAffineComponent::Update(
    const std::string &debug_info,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  CuMatrix<BaseFloat> in_value_temp;

  in_value_temp.Resize(in_value.NumRows(),
                       in_value.NumCols() + 1, kUndefined);
  in_value_temp.Range(0, in_value.NumRows(),
                      0, in_value.NumCols()).CopyFromMat(in_value);

  // Add the 1.0 at the end of each row "in_value_temp"
  in_value_temp.Range(0, in_value.NumRows(),
                      in_value.NumCols(), 1).Set(1.0);

  CuMatrix<BaseFloat> out_deriv_temp(out_deriv);

  // These "scale" values get will get multiplied into the learning rate (faster
  // than having the matrices scaled inside the preconditioning code).
  BaseFloat in_scale, out_scale;

  preconditioner_in_.PreconditionDirections(&in_value_temp, &in_scale);
  preconditioner_out_.PreconditionDirections(&out_deriv_temp, &out_scale);

  // "scale" is a scaling factor coming from the PreconditionDirections calls
  // (it's faster to have them output a scaling factor than to have them scale
  // their outputs).
  BaseFloat scale = in_scale * out_scale;

  CuSubMatrix<BaseFloat> in_value_precon_part(in_value_temp,
                                              0, in_value_temp.NumRows(),
                                              0, in_value_temp.NumCols() - 1);
  // this "precon_ones" is what happens to the vector of 1's representing
  // offsets, after multiplication by the preconditioner.
  CuVector<BaseFloat> precon_ones(in_value_temp.NumRows());

  precon_ones.CopyColFromMat(in_value_temp, in_value_temp.NumCols() - 1);

  BaseFloat local_lrate = scale * learning_rate_;

  bias_params_.AddMatVec(local_lrate, out_deriv_temp, kTrans,
                         precon_ones, 1.0);
  linear_params_.AddMatMat(local_lrate, out_deriv_temp, kTrans,
                           in_value_precon_part, kNoTrans, 1.0);
}

void NaturalGradientAffineComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    linear_params_.SetZero();
    bias_params_.SetZero();
  } else {
    linear_params_.Scale(scale);
    bias_params_.Scale(scale);
  }
}

void NaturalGradientAffineComponent::Add(BaseFloat alpha, const Component &other_in) {
  const NaturalGradientAffineComponent *other =
      dynamic_cast<const NaturalGradientAffineComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

void NaturalGradientAffineComponent::FreezeNaturalGradient(bool freeze) {
  preconditioner_in_.Freeze(freeze);
  preconditioner_out_.Freeze(freeze);
}

void NaturalGradientAffineComponent::ConsolidateMemory() {
  OnlineNaturalGradient temp_in(preconditioner_in_);
  preconditioner_in_.Swap(&temp_in);
  OnlineNaturalGradient temp_out(preconditioner_out_);
  preconditioner_out_.Swap(&temp_out);
}

void LinearComponent::Read(std::istream &is, bool binary) {
  std::string token = ReadUpdatableCommon(is, binary);
  KALDI_ASSERT(token == "");
  ExpectToken(is, binary, "<Params>");
  params_.Read(is, binary);
  if (PeekToken(is, binary) == 'O') {
    ExpectToken(is, binary, "<OrthonormalConstraint>");
    ReadBasicType(is, binary, &orthonormal_constraint_);
  } else {
    orthonormal_constraint_ = 0.0;
  }
  ExpectToken(is, binary, "<UseNaturalGradient>");
  ReadBasicType(is, binary, &use_natural_gradient_);

  // Read various natural-gradient-related configs.
  int32 rank_in,  rank_out, update_period;
  BaseFloat alpha, num_samples_history;
  ExpectToken(is, binary, "<RankInOut>");
  ReadBasicType(is, binary, &rank_in);
  ReadBasicType(is, binary, &rank_out);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha);
  ExpectToken(is, binary, "<NumSamplesHistory>");
  ReadBasicType(is, binary, &num_samples_history);
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period);

  preconditioner_in_.SetAlpha(alpha);
  preconditioner_out_.SetAlpha(alpha);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history);
  preconditioner_in_.SetUpdatePeriod(update_period);
  preconditioner_out_.SetUpdatePeriod(update_period);

  ExpectToken(is, binary, "</LinearComponent>");
}

void LinearComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  is_gradient_ = false;  // not configurable; there's no reason you'd want this

  InitLearningRatesFromConfig(cfl);

  int32 input_dim = -1, output_dim = -1;
  if (cfl->GetValue("matrix", &matrix_filename)) {
    ReadKaldiObject(matrix_filename, &params_); // will abort on failure.
    KALDI_ASSERT(params_.NumRows() != 0);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    if (!ok)
      KALDI_ERR << "Bad initializer " << cfl->WholeLine();
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim);
    cfl->GetValue("param-stddev", &param_stddev);
    params_.Resize(output_dim, input_dim);
    KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
    params_.SetRandn(); // sets to random normally distributed noise.
    params_.Scale(param_stddev);
  }
  // Read various natural-gradient-related configs.
  int32 rank_in = -1, rank_out = -1, update_period = 4;
  BaseFloat alpha = 4.0,
      num_samples_history = 2000.0;

  use_natural_gradient_ = true;

  cfl->GetValue("num-samples-history", &num_samples_history);
  cfl->GetValue("alpha", &alpha);
  cfl->GetValue("rank-in", &rank_in);
  cfl->GetValue("rank-out", &rank_out);
  cfl->GetValue("update-period", &update_period);
  cfl->GetValue("use-natural-gradient", &use_natural_gradient_);

  if (rank_in < 0)
    rank_in = std::min<int32>(20, (InputDim() + 1) / 2);
  if (rank_out < 0)
    rank_out = std::min<int32>(80, (OutputDim() + 1) / 2);

  preconditioner_in_.SetAlpha(alpha);
  preconditioner_out_.SetAlpha(alpha);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history);
  preconditioner_in_.SetUpdatePeriod(update_period);
  preconditioner_out_.SetUpdatePeriod(update_period);

  orthonormal_constraint_ = 0.0;
  cfl->GetValue("orthonormal-constraint", &orthonormal_constraint_);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
}


void LinearComponent::Write(std::ostream &os,
                            bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
  WriteToken(os, binary, "<Params>");
  params_.Write(os, binary);
  if (orthonormal_constraint_ != 0.0) {
    WriteToken(os, binary, "<OrthonormalConstraint>");
    WriteBasicType(os, binary, orthonormal_constraint_);
  }
  WriteToken(os, binary, "<UseNaturalGradient>");
  WriteBasicType(os, binary, use_natural_gradient_);

  int32 rank_in = preconditioner_in_.GetRank(),
      rank_out = preconditioner_out_.GetRank(),
      update_period = preconditioner_in_.GetUpdatePeriod();
  BaseFloat alpha = preconditioner_in_.GetAlpha(),
      num_samples_history = preconditioner_in_.GetNumSamplesHistory();
  WriteToken(os, binary, "<RankInOut>");
  WriteBasicType(os, binary, rank_in);
  WriteBasicType(os, binary, rank_out);
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha);
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, num_samples_history);
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, update_period);
  WriteToken(os, binary, "</LinearComponent>");
}

std::string LinearComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info();
  PrintParameterStats(stream, "params", params_,
                      false, // include_mean
                      true, // include_row_norms
                      true, // include_column_norms
                      GetVerboseLevel() >= 2); // include_singular_values
  if (orthonormal_constraint_ != 0.0)
    stream << ", orthonormal-constraint=" << orthonormal_constraint_;
  stream << ", use-natural-gradient="
         << (use_natural_gradient_ ? "true" : "false")
         << ", rank-in=" << preconditioner_in_.GetRank()
         << ", rank-out=" << preconditioner_out_.GetRank()
         << ", num-samples-history="
         << preconditioner_in_.GetNumSamplesHistory()
         << ", update-period=" << preconditioner_in_.GetUpdatePeriod()
         << ", alpha=" << preconditioner_in_.GetAlpha();
  return stream.str();
}

void* LinearComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  out->AddMatMat(1.0, in, kNoTrans, params_, kTrans, 1.0);
  return NULL;
}

void LinearComponent::Backprop(const std::string &debug_info,
                               const ComponentPrecomputedIndexes *indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               void *memo,
                               Component *to_update_in,
                               CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("LinearComponent::Backprop");
  LinearComponent *to_update = dynamic_cast<LinearComponent*>(to_update_in);

  // Propagate the derivative back to the input.  add with coefficient 1.0 since
  // property kBackpropAdds is true.  If we wanted to add with coefficient 0.0
  // we'd need to zero the in_deriv, in case of infinities.
  if (in_deriv)
    in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, params_, kNoTrans, 1.0);

  if (to_update != NULL) {
    if (!to_update->is_gradient_) {
      CuMatrix<BaseFloat> in_value_temp(in_value), out_deriv_temp(out_deriv);
      // These "scale" values get will get multiplied into the learning rate (faster
      // than having the matrices scaled inside the preconditioning code).
      BaseFloat in_scale, out_scale;
      to_update->preconditioner_in_.PreconditionDirections(&in_value_temp,
                                                           &in_scale);
      to_update->preconditioner_out_.PreconditionDirections(&out_deriv_temp,
                                                            &out_scale);
      BaseFloat local_lrate = in_scale * out_scale * to_update->learning_rate_;

      to_update->params_.AddMatMat(local_lrate, out_deriv_temp, kTrans,
                                   in_value_temp, kNoTrans, 1.0);
    } else {
      to_update->params_.AddMatMat(to_update->learning_rate_,
                                   out_deriv, kTrans,
                                   in_value, kNoTrans, 1.0);
    }
  }
}


Component* LinearComponent::Copy() const {
  return new LinearComponent(*this);
}

LinearComponent::LinearComponent(
    const LinearComponent &other):
    UpdatableComponent(other),
    params_(other.params_),
    orthonormal_constraint_(other.orthonormal_constraint_),
    use_natural_gradient_(other.use_natural_gradient_),
    preconditioner_in_(other.preconditioner_in_),
    preconditioner_out_(other.preconditioner_out_) { }

LinearComponent::LinearComponent(const CuMatrix<BaseFloat> &params):
    params_(params),
    orthonormal_constraint_(0.0),
    use_natural_gradient_(true) {
  // Set defaults for natural gradient.
  preconditioner_in_.SetRank(40);
  preconditioner_out_.SetRank(80);
  preconditioner_in_.SetUpdatePeriod(4);
  preconditioner_out_.SetUpdatePeriod(4);
  // the component-level defaults of alpha and num_samples_history, at 4.0 and
  // 2000.0, are the same as in the NaturalGradientOnline code, so there is no
  // need to set those here.
}

void LinearComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) params_.SetZero();
  else params_.Scale(scale);
}

void LinearComponent::Add(BaseFloat alpha, const Component &other_in) {
  const LinearComponent *other =
      dynamic_cast<const LinearComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  params_.AddMat(alpha, other->params_);
}

void LinearComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_params(params_);
  temp_params.SetRandn();
  params_.AddMat(stddev, temp_params);
}
int32 LinearComponent::NumParameters() const {
  return params_.NumRows() * params_.NumCols();
}
void LinearComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->CopyRowsFromMat(params_);
}
void LinearComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  params_.CopyRowsFromVec(params);
}
BaseFloat LinearComponent::DotProduct(const UpdatableComponent &other_in) const {
  const LinearComponent *other =
      dynamic_cast<const LinearComponent*>(&other_in);
  return TraceMatMat(params_, other->params_, kTrans);
}

void LinearComponent::FreezeNaturalGradient(bool freeze) {
  preconditioner_in_.Freeze(freeze);
  preconditioner_out_.Freeze(freeze);
}

void LinearComponent::ConsolidateMemory() {
  OnlineNaturalGradient temp_in(preconditioner_in_);
  preconditioner_in_.Swap(&temp_in);
  OnlineNaturalGradient temp_out(preconditioner_out_);
  preconditioner_out_.Swap(&temp_out);
}

std::string FixedAffineComponent::Info() const {
  std::ostringstream stream;
  stream << Component::Info();
  PrintParameterStats(stream, "linear-params", linear_params_);
  PrintParameterStats(stream, "bias", bias_params_, true);
  return stream.str();
}

void FixedAffineComponent::Init(const CuMatrixBase<BaseFloat> &mat) {
  KALDI_ASSERT(mat.NumCols() > 1);
  linear_params_ = mat.Range(0, mat.NumRows(), 0, mat.NumCols() - 1);
  bias_params_.Resize(mat.NumRows());
  bias_params_.CopyColFromMat(mat, mat.NumCols() - 1);
}

void FixedAffineComponent::InitFromConfig(ConfigLine *cfl) {
  std::string filename;
  // Two forms allowed: "matrix=<rxfilename>", or "input-dim=x output-dim=y"
  // (for testing purposes only).
  if (cfl->GetValue("matrix", &filename)) {
    if (cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";

    bool binary;
    Input ki(filename, &binary);
    CuMatrix<BaseFloat> mat;
    mat.Read(ki.Stream(), binary);
    KALDI_ASSERT(mat.NumRows() != 0);
    Init(mat);
  } else {
    int32 input_dim = -1, output_dim = -1;
    if (!cfl->GetValue("input-dim", &input_dim) ||
        !cfl->GetValue("output-dim", &output_dim) || cfl->HasUnusedValues()) {
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    }
    CuMatrix<BaseFloat> mat(output_dim, input_dim + 1);
    mat.SetRandn();
    Init(mat);
  }
}


FixedAffineComponent::FixedAffineComponent(const AffineComponent &c):
    linear_params_(c.LinearParams()),
    bias_params_(c.BiasParams()) { }

void* FixedAffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                     const CuMatrixBase<BaseFloat> &in,
                                     CuMatrixBase<BaseFloat> *out) const  {
  out->CopyRowsFromVec(bias_params_); // Adds the bias term first.
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);
  return NULL;
}

void FixedAffineComponent::Backprop(const std::string &debug_info,
                                    const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &, //in_value
                                    const CuMatrixBase<BaseFloat> &, //out_value
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    void *memo,
                                    Component *, //to_update
                                    CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("FixedAffineComponent::Backprop");
  // kBackpropAdds is true. It's the user's responsibility to zero out
  // <in_deriv> if they need it to be so.
  if (in_deriv)
    in_deriv->AddMatMat(1.0, out_deriv, kNoTrans,
                        linear_params_, kNoTrans, 1.0);
}

Component* FixedAffineComponent::Copy() const {
  FixedAffineComponent *ans = new FixedAffineComponent();
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  return ans;
}

void FixedAffineComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<FixedAffineComponent>");
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "</FixedAffineComponent>");
}

void FixedAffineComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<FixedAffineComponent>", "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "</FixedAffineComponent>");
}

void SumGroupComponent::Init(const std::vector<int32> &sizes) {
  KALDI_ASSERT(!sizes.empty());
  std::vector<Int32Pair> cpu_vec(sizes.size());
  std::vector<int32> reverse_cpu_vec;
  int32 cur_index = 0;
  for (size_t i = 0; i < sizes.size(); i++) {
    KALDI_ASSERT(sizes[i] > 0);
    cpu_vec[i].first = cur_index;
    cpu_vec[i].second = cur_index + sizes[i];
    cur_index += sizes[i];
    for (int32 j = cpu_vec[i].first; j < cpu_vec[i].second; j++)
      reverse_cpu_vec.push_back(i);
  }
  this->indexes_ = cpu_vec;
  this->reverse_indexes_ = reverse_cpu_vec;
  this->input_dim_ = cur_index;
  this->output_dim_ = sizes.size();
}

void SumGroupComponent::Init(int32 input_dim, int32 output_dim) {
  const int32 num_groups = output_dim;
  KALDI_ASSERT(input_dim % num_groups == 0);
  const int32 group_size = input_dim / num_groups;

  std::vector<Int32Pair> cpu_vec(num_groups);
  std::vector<int32> reverse_cpu_vec;
  int32 cur_index = 0;
  for (size_t i = 0; i < num_groups; i++) {
    cpu_vec[i].first = cur_index;
    cpu_vec[i].second = cur_index + group_size;
    cur_index += group_size;
    for (int32 j = cpu_vec[i].first; j < cpu_vec[i].second; j++)
      reverse_cpu_vec.push_back(i);
  }
  this->indexes_ = cpu_vec;
  this->reverse_indexes_ = reverse_cpu_vec;
  this->input_dim_ = input_dim;
  this->output_dim_ = num_groups;
}

void SumGroupComponent::InitFromConfig(ConfigLine *cfl) {
  std::vector<int32> sizes;
  bool has_sizes = cfl->GetValue("sizes", &sizes);
  if (has_sizes) {
    if (cfl->HasUnusedValues() || sizes.empty())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    this->Init(sizes);
  } else { // each group has the same size
    int32 input_dim = -1, output_dim = -1;
    if (!cfl->GetValue("input-dim", &input_dim) ||
        !cfl->GetValue("output-dim", &output_dim) || cfl->HasUnusedValues()) {
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    }
    Init(input_dim, output_dim);
  }
}

Component* SumGroupComponent::Copy() const {
  SumGroupComponent *ans = new SumGroupComponent();
  ans->indexes_ = indexes_;
  ans->reverse_indexes_ = reverse_indexes_;
  ans->input_dim_ = input_dim_;
  ans->output_dim_ = output_dim_;
  return ans;
}

void SumGroupComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SumGroupComponent>", "<Sizes>");
  std::vector<int32> sizes;
  ReadIntegerVector(is, binary, &sizes);

  std::string token;
  ReadToken(is, binary, &token);
  if (!(token == "<SumGroupComponent>" ||
        token == "</SumGroupComponent>")) {
    KALDI_ERR << "Expected </SumGroupComponent>, got " << token;
  }
  this->Init(sizes);
}

void SumGroupComponent::GetSizes(std::vector<int32> *sizes) const {
  std::vector<Int32Pair> indexes;
  indexes_.CopyToVec(&indexes);
  sizes->resize(indexes.size());
  for (size_t i = 0; i < indexes.size(); i++) {
    (*sizes)[i] = indexes[i].second - indexes[i].first;
    if (i == 0) { KALDI_ASSERT(indexes[i].first == 0); }
    else { KALDI_ASSERT(indexes[i].first == indexes[i-1].second); }
    KALDI_ASSERT(indexes[i].second > indexes[i].first);
    (*sizes)[i] = indexes[i].second - indexes[i].first;
  }
}

void SumGroupComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SumGroupComponent>");
  WriteToken(os, binary, "<Sizes>");
  std::vector<int32> sizes;
  this->GetSizes(&sizes);
  WriteIntegerVector(os, binary, sizes);
  WriteToken(os, binary, "</SumGroupComponent>");
}

void* SumGroupComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                  const CuMatrixBase<BaseFloat> &in,
                                  CuMatrixBase<BaseFloat> *out) const {
  out->SumColumnRanges(in, indexes_);
  return NULL;
}

void SumGroupComponent::Backprop(const std::string &debug_info,
                                 const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &, // in_value,
                                 const CuMatrixBase<BaseFloat> &, // out_value
                                 const CuMatrixBase<BaseFloat> &out_deriv,
                                 void *memo,
                                 Component *to_update_in,
                                 CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("SumGroupComponent::Backprop");
  in_deriv->CopyCols(out_deriv, reverse_indexes_);
}

void* SoftmaxComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  // Apply softmax function to each row of the output...
  // for that row, we do
  // x_i = exp(x_i) / sum_j exp(x_j).
  out->SoftMaxPerRow(in);

  // This floor on the output helps us deal with
  // almost-zeros in a way that doesn't lead to overflow.
  out->ApplyFloor(1.0e-20);

  return NULL;
}

void SoftmaxComponent::Backprop(const std::string &debug_info,
                                const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &, // in_value,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                void *memo,
                                Component *to_update_in,
                                CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("SoftmaxComponent::Backprop");

  if (to_update_in) {
    SoftmaxComponent *to_update =
        dynamic_cast<SoftmaxComponent*>(to_update_in);
    to_update->StoreBackpropStats(out_deriv);
  }

  if (in_deriv == NULL)
    return;
  /*
    Note on the derivative of the softmax function: let it be
    p_i = exp(x_i) / sum_i exp_i
    The [matrix-valued] Jacobian of this function is
    diag(p) - p p^T
    Let the derivative vector at the output be e, and at the input be
    d.  We have
    d = diag(p) e - p (p^T e).
    d_i = p_i e_i - p_i (p^T e).
  */
  in_deriv->DiffSoftmaxPerRow(out_value, out_deriv);
}

void SoftmaxComponent::StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_value,
                                  void *memo) {
  // We don't store derivative stats for this component type, just activation
  // stats.
  StoreStatsInternal(out_value, NULL);
}


void* LogSoftmaxComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &in,
                                    CuMatrixBase<BaseFloat> *out) const {
  // Applies log softmax function to each row of the output. For each row, we do
  // x_i = x_i - log(sum_j exp(x_j))
  out->LogSoftMaxPerRow(in);
  return NULL;
}

void LogSoftmaxComponent::Backprop(const std::string &debug_info,
                                   const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &, // in_value
                                   const CuMatrixBase<BaseFloat> &out_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv,
                                   void *memo,
                                   Component *to_update_in,
                                   CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("LogSoftmaxComponent::Backprop");
  if (to_update_in) {
    LogSoftmaxComponent *to_update =
        dynamic_cast<LogSoftmaxComponent*>(to_update_in);
    to_update->StoreBackpropStats(out_deriv);
  }
  if (in_deriv == NULL)
    return;
  in_deriv->DiffLogSoftmaxPerRow(out_value, out_deriv);
}


void FixedScaleComponent::Init(const CuVectorBase<BaseFloat> &scales) {
  KALDI_ASSERT(scales.Dim() != 0);
  scales_ = scales;
}


void FixedScaleComponent::InitFromConfig(ConfigLine *cfl) {
  std::string filename;
  // Accepts "scales" config (for filename) or "dim" -> random init, for testing.
  if (cfl->GetValue("scales", &filename)) {
    if (cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    CuVector<BaseFloat> vec;
    ReadKaldiObject(filename, &vec);
    Init(vec);
  } else {
    int32 dim;
    BaseFloat scale = 1.0;
    bool scale_is_set = cfl->GetValue("scale", &scale);
    if (!cfl->GetValue("dim", &dim) || cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    KALDI_ASSERT(dim > 0);
    CuVector<BaseFloat> vec(dim);
    if (scale_is_set) {
      vec.Set(scale);
    } else {
      vec.SetRandn();
    }
    Init(vec);
  }
}


std::string FixedScaleComponent::Info() const {
  std::ostringstream stream;
  stream << Component::Info();
  PrintParameterStats(stream, "scales", scales_, true);
  return stream.str();
}

void* FixedScaleComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                     const CuMatrixBase<BaseFloat> &in,
                                     CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);  // does nothing if same matrix.
  out->MulColsVec(scales_);
  return NULL;
}

void FixedScaleComponent::Backprop(const std::string &debug_info,
                                   const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &, // in_value
                                   const CuMatrixBase<BaseFloat> &, // out_value
                                   const CuMatrixBase<BaseFloat> &out_deriv,
                                   void *memo,
                                   Component *, // to_update
                                   CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("FixedScaleComponent::Backprop");
  in_deriv->CopyFromMat(out_deriv);  // does nothing if same memory.
  in_deriv->MulColsVec(scales_);
}

Component* FixedScaleComponent::Copy() const {
  FixedScaleComponent *ans = new FixedScaleComponent();
  ans->scales_ = scales_;
  return ans;
}


void FixedScaleComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<FixedScaleComponent>");
  WriteToken(os, binary, "<Scales>");
  scales_.Write(os, binary);
  WriteToken(os, binary, "</FixedScaleComponent>");
}

void FixedScaleComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<FixedScaleComponent>", "<Scales>");
  scales_.Read(is, binary);
  ExpectToken(is, binary, "</FixedScaleComponent>");
}

void FixedBiasComponent::Init(const CuVectorBase<BaseFloat> &bias) {
  KALDI_ASSERT(bias.Dim() != 0);
  bias_ = bias;
}

void FixedBiasComponent::InitFromConfig(ConfigLine *cfl) {
  std::string filename;
  // Accepts "bias" config (for filename) or "dim" -> random init, for testing.
  if (cfl->GetValue("bias", &filename)) {
    if (cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    CuVector<BaseFloat> vec;
    ReadKaldiObject(filename, &vec);
    Init(vec);
  } else {
    int32 dim;
    if (!cfl->GetValue("dim", &dim) || cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    KALDI_ASSERT(dim > 0);
    CuVector<BaseFloat> vec(dim);
    vec.SetRandn();
    Init(vec);
  }
}

std::string FixedBiasComponent::Info() const {
  std::ostringstream stream;
  stream << Component::Info();
  PrintParameterStats(stream, "bias", bias_, true);
  return stream.str();
}

void* FixedBiasComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);  // will do nothing if in and out have same memory.
  out->AddVecToRows(1.0, bias_, 1.0);
  return NULL;
}

void FixedBiasComponent::Backprop(const std::string &debug_info,
                                  const ComponentPrecomputedIndexes *indexes,
                                  const CuMatrixBase<BaseFloat> &, // in_value
                                  const CuMatrixBase<BaseFloat> &, // out_value
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  void *memo,
                                  Component *, // to_update
                                  CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("FixedBiasComponent::Backprop");
  // the following statement will do nothing if in_deriv and out_deriv have same
  // memory.
  in_deriv->CopyFromMat(out_deriv);
}

Component* FixedBiasComponent::Copy() const {
  FixedBiasComponent *ans = new FixedBiasComponent();
  ans->bias_ = bias_;
  return ans;
}


void FixedBiasComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<FixedBiasComponent>");
  WriteToken(os, binary, "<Bias>");
  bias_.Write(os, binary);
  WriteToken(os, binary, "</FixedBiasComponent>");
}

void FixedBiasComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<FixedBiasComponent>", "<Bias>");
  bias_.Read(is, binary);
  ExpectToken(is, binary, "</FixedBiasComponent>");
}


void NaturalGradientPerElementScaleComponent::Read(
    std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // Read the opening tag and learning rate
  ExpectToken(is, binary, "<Params>");
  scales_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  int32 rank, update_period;
  ExpectToken(is, binary, "<Rank>");
  ReadBasicType(is, binary, &rank);
  preconditioner_.SetRank(rank);
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period);
  preconditioner_.SetUpdatePeriod(update_period);
  BaseFloat num_samples_history, alpha;
  ExpectToken(is, binary, "<NumSamplesHistory>");
  ReadBasicType(is, binary, &num_samples_history);
  preconditioner_.SetNumSamplesHistory(num_samples_history);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha);
  preconditioner_.SetAlpha(alpha);
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<MaxChangePerMinibatch>") {
    // back compatibility; this was removed, it's now handled by the
    // 'max-change' config variable.
    BaseFloat temp;
    ReadBasicType(is, binary, &temp);
    ReadToken(is, binary, &token);
  }
  KALDI_ASSERT(token == "</NaturalGradientPerElementScaleComponent>");
}

void NaturalGradientPerElementScaleComponent::Write(std::ostream &os,
                                                    bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
  WriteToken(os, binary, "<Params>");
  scales_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "<Rank>");
  WriteBasicType(os, binary, preconditioner_.GetRank());
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, preconditioner_.GetUpdatePeriod());
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, preconditioner_.GetNumSamplesHistory());
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, preconditioner_.GetAlpha());
  WriteToken(os, binary, "</NaturalGradientPerElementScaleComponent>");
}

std::string NaturalGradientPerElementScaleComponent::Info() const {
  std::ostringstream stream;
  stream << PerElementScaleComponent::Info()
         << ", rank=" << preconditioner_.GetRank()
         << ", update-period=" << preconditioner_.GetUpdatePeriod()
         << ", num-samples-history=" << preconditioner_.GetNumSamplesHistory()
         << ", alpha=" << preconditioner_.GetAlpha();
  return stream.str();
}

void NaturalGradientPerElementScaleComponent::InitFromConfig(ConfigLine *cfl) {
  // First set various configuration values that have defaults.
  int32 rank = 8,  // Use a small rank because in this case the amount of memory
                   // for the preconditioner actually exceeds the memory for the
                   // parameters (by "rank").
      update_period = 10;
  BaseFloat num_samples_history = 2000.0, alpha = 4.0;
  cfl->GetValue("rank", &rank);
  cfl->GetValue("update-period", &update_period);
  cfl->GetValue("num-samples-history", &num_samples_history);
  cfl->GetValue("alpha", &alpha);
  InitLearningRatesFromConfig(cfl);
  std::string filename;
  // Accepts "scales" config (for filename) or "dim" -> random init, for testing.
  if (cfl->GetValue("scales", &filename)) {
    if (cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    Init(filename, rank, update_period, num_samples_history, alpha);

  } else {
    BaseFloat param_mean = 1.0, param_stddev = 0.0;
    cfl->GetValue("param-mean", &param_mean);
    cfl->GetValue("param-stddev", &param_stddev);

    int32 dim;
    if (!cfl->GetValue("dim", &dim) || cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    KALDI_ASSERT(dim > 0);

    Init(dim, param_mean, param_stddev, rank, update_period,
         num_samples_history, alpha);
  }
}

void NaturalGradientPerElementScaleComponent::Init(
    int32 dim, BaseFloat param_mean,
    BaseFloat param_stddev, int32 rank, int32 update_period,
    BaseFloat num_samples_history, BaseFloat alpha) {
  PerElementScaleComponent::Init(dim, param_mean,
                                 param_stddev);
  preconditioner_.SetRank(rank);
  preconditioner_.SetUpdatePeriod(update_period);
  preconditioner_.SetNumSamplesHistory(num_samples_history);
  preconditioner_.SetAlpha(alpha);
}

void NaturalGradientPerElementScaleComponent::Init(
    std::string vector_filename,
    int32 rank, int32 update_period, BaseFloat num_samples_history,
    BaseFloat alpha) {
  PerElementScaleComponent::Init(vector_filename);
  preconditioner_.SetRank(rank);
  preconditioner_.SetUpdatePeriod(update_period);
  preconditioner_.SetNumSamplesHistory(num_samples_history);
  preconditioner_.SetAlpha(alpha);
}


NaturalGradientPerElementScaleComponent::NaturalGradientPerElementScaleComponent(
    const NaturalGradientPerElementScaleComponent &other):
    PerElementScaleComponent(other),
    preconditioner_(other.preconditioner_) { }




Component* NaturalGradientPerElementScaleComponent::Copy() const {
  return new NaturalGradientPerElementScaleComponent(*this);
}

void NaturalGradientPerElementScaleComponent::Update(
    const std::string &debug_info,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {

  CuMatrix<BaseFloat> derivs_per_frame(in_value);
  derivs_per_frame.MulElements(out_deriv);
  // the non-natural-gradient update would just do
  // scales_.AddRowSumMat(learning_rate_, derivs_per_frame).

  BaseFloat scale;
  preconditioner_.PreconditionDirections(&derivs_per_frame, &scale);

  CuVector<BaseFloat> delta_scales(scales_.Dim());
  delta_scales.AddRowSumMat(scale * learning_rate_, derivs_per_frame);
  scales_.AddVec(1.0, delta_scales);
}

void NaturalGradientPerElementScaleComponent::FreezeNaturalGradient(bool freeze) {
  preconditioner_.Freeze(freeze);
}

void NaturalGradientPerElementScaleComponent::ConsolidateMemory() {
  OnlineNaturalGradient temp(preconditioner_);
  preconditioner_.Swap(&temp);
}

void PermuteComponent::ComputeReverseColumnMap() {
  int32 dim = column_map_.Dim();
  KALDI_ASSERT(dim > 0);
  std::vector<int32> reverse_column_map_cpu(dim, -1),
      column_map_cpu(dim);
  column_map_.CopyToVec(&column_map_cpu);
  for (int32 i = 0; i < dim; i++) {
    int32 &dest = reverse_column_map_cpu[column_map_cpu[i]];
    if (dest != -1)
      KALDI_ERR << "Column map does not represent a permutation.";
    dest = i;
  }
  reverse_column_map_.Resize(dim);
  reverse_column_map_.CopyFromVec(reverse_column_map_cpu);
}

Component* PermuteComponent::Copy() const {
  PermuteComponent *ans = new PermuteComponent();
  ans->column_map_ = column_map_;
  ans->reverse_column_map_ = reverse_column_map_;
  return ans;
}

void* PermuteComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const  {
  out->CopyCols(in, column_map_);
  return NULL;
}
void PermuteComponent::Backprop(const std::string &debug_info,
                                const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &, //in_value
                                const CuMatrixBase<BaseFloat> &, // out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                void *memo,
                                Component *to_update,
                                CuMatrixBase<BaseFloat> *in_deriv) const  {
  NVTX_RANGE("PermuteComponent::Backprop");
  in_deriv->CopyCols(out_deriv, reverse_column_map_);
}

void PermuteComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string column_map_str;
  ok = ok && cfl->GetValue("column-map", &column_map_str);
  std::vector<int32> column_map;
  if (!SplitStringToIntegers(column_map_str, ",", true, &column_map))
    KALDI_ERR << "Bad initializer in PermuteComponent: column-map="
              << column_map_str;
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(column_map);
}

void PermuteComponent::Init(const std::vector<int32> &column_map) {
  KALDI_ASSERT(column_map.size() > 0);
  column_map_.CopyFromVec(column_map);
  ComputeReverseColumnMap();
}

void PermuteComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PermuteComponent>", "<ColumnMap>");
  std::vector<int32> column_map;
  if (binary && is.peek() == 'F') {
    // back-compatibility code [temporary]
    Vector<BaseFloat> float_map;
    float_map.Read(is, binary);
    column_map.resize(float_map.Dim());
    for (int32 i = 0; i < float_map.Dim(); i++) {
      // note: casting truncates toward zero: add 0.5 to approximate rounding.
      column_map[i] = static_cast<int32>(float_map(i) + 0.5);
    }
    // the next line is a workaround for a bug in the old
    // writing code, which now causes an assert failure.  it's only
    // valid for the permutations we're currently using.  anyway all this
    // code is only temporary.
    column_map.back() = float_map.Dim() - 1;
  } else {
    ReadIntegerVector(is, binary, &column_map);
  }
  column_map_.CopyFromVec(column_map);
  ExpectToken(is, binary, "</PermuteComponent>");
  ComputeReverseColumnMap();
}

void PermuteComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PermuteComponent>");
  WriteToken(os, binary, "<ColumnMap>");
  std::ostringstream buffer;
  std::vector<int32> column_map;
  column_map_.CopyToVec(&column_map);
  WriteIntegerVector(os, binary, column_map);
  WriteToken(os, binary, "</PermuteComponent>");
}

std::string PermuteComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << column_map_.Dim();
  stream << " , column-map=[ ";
  std::vector<int32> column_map(column_map_.Dim());
  column_map_.CopyToVec(&column_map);
  int32 max_size = 5;
  for (size_t i = 0; i < column_map.size() && i < max_size; i++)
    stream << column_map[i] << ' ';
  if (static_cast<int32>(column_map.size()) > max_size)
    stream << "... ";
  stream << "]";
  return stream.str();
}


bool CompositeComponent::IsUpdatable() const {
  for (std::vector<Component*>::const_iterator iter = components_.begin(),
           end = components_.end(); iter != end; ++iter)
    if (((*iter)->Properties() & kUpdatableComponent) != 0)
      return true;
  return false;
}

// virtual
int32 CompositeComponent::InputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.front()->InputDim();
}

// virtual
int32 CompositeComponent::OutputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.back()->OutputDim();
}

// virtual
int32 CompositeComponent::Properties() const {
  KALDI_ASSERT(!components_.empty());
  int32 last_component_properties = components_.back()->Properties(),
      first_component_properties = components_.front()->Properties();
  // We always assume backprop needs the input, as this would be necessary to
  // get the activations at intermediate layers, if these were not needed in
  // backprop, there would be no reason to use a CompositeComponent.
  int32 ans = kSimpleComponent | kBackpropNeedsInput |
      (last_component_properties &
       (kPropagateAdds|kBackpropNeedsOutput|kOutputContiguous)) |
       (first_component_properties &
        (kBackpropAdds|kInputContiguous)) |
       (IsUpdatable() ? kUpdatableComponent : 0);
  // note, we don't return the kStoresStats property because that function is
  // not implemented; instead, for efficiency, we call StoreStats() on any
  // sub-components as part of the backprop phase.
  if (last_component_properties & kStoresStats)
    ans |= kBackpropNeedsOutput;
  return ans;
}


MatrixStrideType CompositeComponent::GetStrideType(int32 i) const {
  int32 num_components = components_.size();
  if ((components_[i]->Properties() & kOutputContiguous) ||
      (i + 1 < num_components &&
       (components_[i + 1]->Properties() & kInputContiguous)))
    return kStrideEqualNumCols;
  else
    return kDefaultStride;
}


// virtual
void* CompositeComponent::Propagate(
    const ComponentPrecomputedIndexes *, // indexes
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumRows() == out->NumRows() && in.NumCols() == InputDim() &&
               out->NumCols() == OutputDim());
  int32 num_rows = in.NumRows(),
      num_components = components_.size();
  if (max_rows_process_ > 0 && num_rows > max_rows_process_) {
    // recurse and process smaller parts of the data, to save memory.
    for (int32 row_offset = 0; row_offset < num_rows;
         row_offset += max_rows_process_) {
      int32 this_num_rows = std::min<int32>(max_rows_process_,
                                            num_rows - row_offset);
      const CuSubMatrix<BaseFloat> in_part(in, row_offset, this_num_rows,
                                           0, in.NumCols());
      CuSubMatrix<BaseFloat> out_part(*out, row_offset, this_num_rows,
                                      0, out->NumCols());
      this->Propagate(NULL, in_part, &out_part);
    }
    return NULL;
  }
  std::vector<CuMatrix<BaseFloat> > intermediate_outputs(num_components - 1);
  for (int32 i = 0; i < num_components; i++) {
    if (i + 1 < num_components) {
      MatrixResizeType resize_type =
          ((components_[i]->Properties() & kPropagateAdds) ?
           kSetZero : kUndefined);
      intermediate_outputs[i].Resize(num_rows, components_[i]->OutputDim(),
                                     resize_type, GetStrideType(i));
    }
    const CuMatrixBase<BaseFloat> &this_in = (i == 0 ? in :
                                              intermediate_outputs[i-1]);
    CuMatrixBase<BaseFloat> *this_out = (i + 1 == num_components ?
                                         out : &(intermediate_outputs[i]));
    void *memo =  components_[i]->Propagate(NULL, this_in, this_out);
    // we'll re-do the forward propagation in the backprop, and we can
    // regenerate any memos there, so no need to keep them.
    if (memo != NULL)
      components_[i]->DeleteMemo(memo);
    if (i > 0)
      intermediate_outputs[i-1].Resize(0, 0);
  }
  return NULL;
}


void CompositeComponent::Init(const std::vector<Component*> &components,
                              int32 max_rows_process) {
  DeletePointers(&components_);  // clean up.
  components_ = components;
  KALDI_ASSERT(!components.empty());
  max_rows_process_ = max_rows_process;

  for (size_t i = 0; i < components_.size(); i++) {
    // make sure all constituent components are simple.
    KALDI_ASSERT(components_[i]->Properties() & kSimpleComponent);
    if (i > 0) {
      // make sure all the internal dimensions match up.
      KALDI_ASSERT(components_[i]->InputDim() ==
                   components_[i-1]->OutputDim());
    }
  }
}

// virtual
void CompositeComponent::Read(std::istream &is, bool binary) {
  // Because we didn't previously write out the learning rate,
  // we need some temporary code.
  int32 max_rows_process;
  if (false) {
    ReadUpdatableCommon(is, binary);
    ExpectToken(is, binary, "<MaxRowsProcess>");
    ReadBasicType(is, binary, &max_rows_process);
  } else {  // temporary code.
    std::string token;
    ReadToken(is, binary, &token);
    if (token == "<CompositeComponent>") {
      // if the first token is the opening tag, then
      // ignore it and get the next tag.
      ReadToken(is, binary, &token);
    }
    if (token == "<LearningRateFactor>") {
      ReadBasicType(is, binary, &learning_rate_factor_);
      ReadToken(is, binary, &token);
    } else {
      learning_rate_factor_ = 1.0;
    }
    if (token == "<IsGradient>") {
      ReadBasicType(is, binary, &is_gradient_);
      ReadToken(is, binary, &token);
    } else {
      is_gradient_ = false;
    }
    if (token == "<LearningRate>") {
      ReadBasicType(is, binary, &learning_rate_);
      ReadToken(is, binary, &token);
    }
    if (token != "<MaxRowsProcess>") {
      KALDI_ERR << "Expected token <MaxRowsProcess>, got "
                << token;
    }
    ReadBasicType(is, binary, &max_rows_process);
  }
  ExpectToken(is, binary, "<NumComponents>");
  int32 num_components;
  ReadBasicType(is, binary, &num_components); // Read dimension.
  if (num_components < 0 || num_components > 100000)
    KALDI_ERR << "Bad num-components";
  std::vector<Component*> components(num_components);
  for (int32 i = 0; i < num_components; i++)
    components[i] = ReadNew(is, binary);
  Init(components, max_rows_process);
  ExpectToken(is, binary, "</CompositeComponent>");
}

// virtual
void CompositeComponent::ZeroStats() {
  // we call ZeroStats() on all components without checking their flags; this
  // will do nothing if the component doesn't store stats.  (components like
  // ReLU and sigmoid and tanh store stats on activations).
  for (size_t i = 0; i < components_.size(); i++)
   components_[i]->ZeroStats();
}

// virtual
void CompositeComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate.
  WriteToken(os, binary, "<MaxRowsProcess>");
  WriteBasicType(os, binary, max_rows_process_);
  WriteToken(os, binary, "<NumComponents>");
  int32 num_components = components_.size();
  WriteBasicType(os, binary, num_components);
  for (int32 i = 0; i < num_components; i++)
    components_[i]->Write(os, binary);
  WriteToken(os, binary, "</CompositeComponent>");
}


// virtual
void CompositeComponent::Backprop(const std::string &debug_info,
                                  const ComponentPrecomputedIndexes *indexes,
                                  const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_value,
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  void *memo,
                                  Component *to_update,
                                  CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("CompositeComponent::Backprop");
  KALDI_ASSERT(in_value.NumRows() == out_deriv.NumRows() &&
               in_value.NumCols() == InputDim() &&
               out_deriv.NumCols() == OutputDim());
  int32 num_rows = in_value.NumRows(),
      num_components = components_.size();
  if (max_rows_process_ > 0 && num_rows > max_rows_process_) {
    KALDI_ASSERT(max_rows_process_ > 0);
    // recurse and process smaller parts of the data, to save memory.
    for (int32 row_offset = 0; row_offset < num_rows;
         row_offset += max_rows_process_) {
      bool have_output_value = (out_value.NumRows() != 0);
      int32 this_num_rows = std::min<int32>(max_rows_process_,
                                            num_rows - row_offset);
      // out_value_part will only be used if out_value is nonempty; otherwise we
      // make it a submatrix of 'out_deriv' to avoid errors in the constructor.
      const CuSubMatrix<BaseFloat> out_value_part(have_output_value ? out_value : out_deriv,
                                                  row_offset, this_num_rows,
                                                  0, out_deriv.NumCols());
      // in_deriv_value_part will only be used if in_deriv != NULL; otherwise we
      // make it a submatrix of 'in_value' to avoid errors in the constructor.
      CuSubMatrix<BaseFloat> in_deriv_part(in_deriv != NULL ? *in_deriv : in_value,
                                            row_offset, this_num_rows,
                                            0, in_value.NumCols());
      CuSubMatrix<BaseFloat> in_value_part(in_value, row_offset, this_num_rows,
                                           0, in_value.NumCols());
      const CuSubMatrix<BaseFloat> out_deriv_part(out_deriv,
                                                  row_offset, this_num_rows,
                                                  0, out_deriv.NumCols());
      CuMatrix<BaseFloat>  empty_mat;
      this->Backprop(debug_info, NULL, in_value_part,
                     (have_output_value ? static_cast<const CuMatrixBase<BaseFloat>&>(out_value_part) :
                      static_cast<const CuMatrixBase<BaseFloat>&>(empty_mat)),
                     out_deriv_part, NULL, to_update,
                     in_deriv != NULL ? &in_deriv_part : NULL);
    }
    return;
  }
  // For now, assume all intermediate values and derivatives need to be
  // computed.  in_value and out_deriv will always be supplied.

  // intermediate_outputs[i] contains the output of component i.
  std::vector<CuMatrix<BaseFloat> > intermediate_outputs(num_components);
  // intermediate_derivs[i] contains the deriative at the output of component i.
  std::vector<CuMatrix<BaseFloat> > intermediate_derivs(num_components - 1);

  KALDI_ASSERT(memo == NULL);
  // note: only a very few components use memos, but we need to support them.
  std::vector<void*> memos(num_components, NULL);

  int32 num_components_to_propagate = num_components;
  if (!(components_[num_components - 1]->Properties() & kUsesMemo)) {
    // we only need to propagate the very last component if it uses a memo.
    num_components_to_propagate--;
    if (num_components > 1) {
      // skip the last-but-one component's propagate if the last component's
      // backprop doesn't need the input and the last-but-one component's
      // backprop doesn't need the output.  This is the lowest hanging fruit for
      // optimization; other propagates might also be skippable.
      int32 properties = components_[num_components - 2]->Properties(),
          next_properties = components_[num_components - 1]->Properties();
      if (!(properties & (kBackpropNeedsOutput | kUsesMemo)) &&
          !(next_properties & kBackpropNeedsInput)) {
        num_components_to_propagate--;
      }
    }
  }


  // Do the propagation again.
  for (int32 i = 0; i < num_components_to_propagate; i++) {
    MatrixResizeType resize_type =
        ((components_[i]->Properties() & kPropagateAdds) ?
         kSetZero : kUndefined);
    intermediate_outputs[i].Resize(num_rows, components_[i]->OutputDim(),
                                   resize_type, GetStrideType(i));
    memos[i] =
        components_[i]->Propagate(NULL,
                             (i == 0 ? in_value : intermediate_outputs[i-1]),
                              &(intermediate_outputs[i]));
  }

  for (int32 i = num_components - 1; i >= 0; i--) {
    const CuMatrixBase<BaseFloat> &this_in_value =
        (i == 0 ? in_value : intermediate_outputs[i-1]),
        &this_out_value =
        (i == num_components - 1 ? out_value : intermediate_outputs[i]);

    Component *component_to_update =
        (to_update == NULL ? NULL :
         dynamic_cast<CompositeComponent*>(to_update)->components_[i]);

    if (component_to_update != NULL  &&
        components_[i]->Properties() & kStoresStats)
      component_to_update->StoreStats(this_in_value, this_out_value, memos[i]);

    if (i > 0) {
      MatrixResizeType resize_type =
          ((components_[i]->Properties() & kBackpropAdds) ?
           kSetZero : kUndefined);
      intermediate_derivs[i-1].Resize(num_rows, components_[i]->InputDim(),
                                      resize_type, GetStrideType(i - 1));
    }
    // skip the first component's backprop if it's not updatable and in_deriv is
    // not requested.  Again, this is the lowest-hanging fruit to optimize.
    if (!(i == 0 && !(components_[0]->Properties() & kUpdatableComponent) &&
          in_deriv == NULL)) {
      components_[i]->Backprop(debug_info, NULL,
                this_in_value, this_out_value,
                (i + 1 == num_components ? out_deriv : intermediate_derivs[i]),
                memos[i], component_to_update,
                (i == 0 ? in_deriv : &(intermediate_derivs[i-1])));
    }
    if (memos[i] != NULL)
      components_[i]->DeleteMemo(memos[i]);
  }
}


// virtual
std::string CompositeComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << " ";
  for (size_t i = 0; i < components_.size(); i++) {
    if (i > 0) stream << ", ";
    stream << "sub-component" << (i+1) << " = { "
           << components_[i]->Info() << " }";
  }
  return stream.str();
}

// virtual
void CompositeComponent::Scale(BaseFloat scale) {
  for (size_t i = 0; i < components_.size(); i++)
    components_[i]->Scale(scale);
}

// virtual
void CompositeComponent::Add(BaseFloat alpha, const Component &other_in) {
  const CompositeComponent *other = dynamic_cast<const CompositeComponent*>(
      &other_in);
  KALDI_ASSERT(other != NULL && other->components_.size() ==
               components_.size() && "Mismatching nnet topologies");
  for (size_t i = 0; i < components_.size(); i++)
    components_[i]->Add(alpha, *(other->components_[i]));
}

// virtual
void CompositeComponent::PerturbParams(BaseFloat stddev) {
  KALDI_ASSERT(this->IsUpdatable());  // or should not be called.
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      uc->PerturbParams(stddev);
    }
  }
}

void CompositeComponent::SetUnderlyingLearningRate(BaseFloat lrate) {
  KALDI_ASSERT(this->IsUpdatable());  // or should not be called.
  UpdatableComponent::SetUnderlyingLearningRate(lrate);

  // apply any learning-rate-factor that's set at this level (ill-advised, but
  // we'll do it.)
  BaseFloat effective_lrate = LearningRate();
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      uc->SetUnderlyingLearningRate(effective_lrate);
    }
  }
}

void CompositeComponent::SetActualLearningRate(BaseFloat lrate) {
  KALDI_ASSERT(this->IsUpdatable());  // or should not be called.
  UpdatableComponent::SetActualLearningRate(lrate);
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      uc->SetActualLearningRate(lrate);
    }
  }
}

// virtual
void CompositeComponent::SetAsGradient() {
  KALDI_ASSERT(this->IsUpdatable());  // or should not be called.
  UpdatableComponent::SetAsGradient();
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      uc->SetAsGradient();
    }
  }
}

// virtual
int32 CompositeComponent::NumParameters() const {
  KALDI_ASSERT(this->IsUpdatable());  // or should not be called.
  int32 ans = 0;
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      ans += uc->NumParameters();
    }
  }
  return ans;
}

// virtual
void CompositeComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  int32 cur_offset = 0;
  KALDI_ASSERT(this->IsUpdatable());  // or should not be called.
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      int32 this_size = uc->NumParameters();
      SubVector<BaseFloat> params_range(*params, cur_offset, this_size);
      uc->Vectorize(&params_range);
      cur_offset += this_size;
    }
  }
  KALDI_ASSERT(cur_offset == params->Dim());
}

// virtual
void CompositeComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  int32 cur_offset = 0;
  KALDI_ASSERT(this->IsUpdatable());  // or should not be called.
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      int32 this_size = uc->NumParameters();
      SubVector<BaseFloat> params_range(params, cur_offset, this_size);
      uc->UnVectorize(params_range);
      cur_offset += this_size;
    }
  }
  KALDI_ASSERT(cur_offset == params.Dim());
}

// virtual
BaseFloat CompositeComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const CompositeComponent *other = dynamic_cast<const CompositeComponent*>(
      &other_in);
  KALDI_ASSERT(other != NULL && other->components_.size() ==
               components_.size() && "Mismatching nnet topologies");
  BaseFloat ans = 0.0;
  for (size_t i = 0.0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      const UpdatableComponent *uc_other =
          dynamic_cast<UpdatableComponent*>(other->components_[i]);
      KALDI_ASSERT(uc != NULL && uc_other != NULL);
      ans += uc->DotProduct(*uc_other);
    }
  }
  return ans;
}

/// virtual
void CompositeComponent::FreezeNaturalGradient(bool freeze) {
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      KALDI_ASSERT(uc != NULL);
      uc->FreezeNaturalGradient(freeze);
    }
  }
}

// virtual
Component* CompositeComponent::Copy() const {
  std::vector<Component*> components(components_.size());
  for (size_t i = 0; i < components_.size(); i++)
    components[i] = components_[i]->Copy();
  CompositeComponent *ans = new CompositeComponent();
  ans->Init(components, max_rows_process_);
  return ans;
}


// virtual
void CompositeComponent::InitFromConfig(ConfigLine *cfl) {
  int32 max_rows_process = 4096, num_components = -1;
  cfl->GetValue("max-rows-process", &max_rows_process);
  if (!cfl->GetValue("num-components", &num_components) ||
      num_components < 1)
    KALDI_ERR << "Expected num-components to be defined in "
              << "CompositeComponent config line '" << cfl->WholeLine() << "'";
  std::vector<Component*> components;
  for (int32 i = 1; i <= num_components; i++) {
    std::ostringstream name_stream;
    name_stream << "component" << i;
    std::string component_config;
    if (!cfl->GetValue(name_stream.str(), &component_config)) {
      DeletePointers(&components);
      KALDI_ERR << "Expected '" << name_stream.str() << "' to be defined in "
                << "CompositeComponent config line '" << cfl->WholeLine() << "'";
    }
    ConfigLine nested_line;
    // note: the nested line may not contain comments.
    std::string component_type;
    Component *this_component = NULL;
    if (!nested_line.ParseLine(component_config) ||
        !nested_line.GetValue("type", &component_type) ||
        !(this_component = NewComponentOfType(component_type)) ||
        nested_line.FirstToken() != "") {
      DeletePointers(&components);
      KALDI_ERR << "Could not parse config line for '" << name_stream.str()
                << "(or undefined or bad component type [type=xxx]), in "
                << "CompositeComponent config line '" << cfl->WholeLine() << "'";
    }
    if(this_component->Type() == "CompositeComponent") {
      DeletePointers(&components);
      delete this_component;
      // This is not allowed.  If memory is too much with just one
      // CompositeComponent, try decreasing max-rows-process instead.
      KALDI_ERR << "Found CompositeComponent nested within CompositeComponent."
                << "Nested line: '" << nested_line.WholeLine() << "'\n"
                << "Toplevel CompositeComponent line '" << cfl->WholeLine()
                << "'";
    }
    this_component->InitFromConfig(&nested_line);
    int32 props = this_component->Properties();
    if ((props & kRandomComponent) != 0 ||
        (props & kSimpleComponent) == 0) {
      KALDI_ERR << "CompositeComponent contains disallowed component type: "
                << nested_line.WholeLine();
    }
    components.push_back(this_component);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  this->Init(components, max_rows_process);
}

const Component* CompositeComponent::GetComponent(int32 i) const {
  KALDI_ASSERT(static_cast<size_t>(i) < components_.size());
  return components_[i];
}

void CompositeComponent::SetComponent(int32 i, Component *component) {
  KALDI_ASSERT(static_cast<size_t>(i) < components_.size());
  delete components_[i];
  components_[i] = component;
}


SumBlockComponent::SumBlockComponent(const SumBlockComponent &other):
    input_dim_(other.input_dim_), output_dim_(other.output_dim_),
    scale_(other.scale_) { }

void SumBlockComponent::InitFromConfig(ConfigLine *cfl) {
  scale_ = 1.0;
  bool ok = cfl->GetValue("input-dim", &input_dim_) &&
      cfl->GetValue("output-dim", &output_dim_);
  if (!ok)
    KALDI_ERR << "input-dim and output-dim must both be provided.";
  if (input_dim_ <= 0 || input_dim_ % output_dim_ != 0)
    KALDI_ERR << "Invalid values input-dim=" << input_dim_
              << " output-dim=" << output_dim_;
  cfl->GetValue("scale", &scale_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
}

void SumBlockComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SumBlockComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "</SumBlockComponent>");
}

void SumBlockComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SumBlockComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "</SumBlockComponent>");
}

std::string SumBlockComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", scale=" << scale_;
  return stream.str();
}

void* SumBlockComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(out->NumRows() == in.NumRows() &&
               out->NumCols() == output_dim_ &&
               in.NumCols() == input_dim_);
  out->AddMatBlocks(scale_, in, kNoTrans);
  return NULL;
}

void SumBlockComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &, //in_value
    const CuMatrixBase<BaseFloat> &, // out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  NVTX_RANGE("SumBlockComponent::Backprop");
  if (in_deriv) {
    in_deriv->AddMatBlocks(scale_, out_deriv, kNoTrans);
  }
}



// Modified version for adaptation!!!

//////////

void NormalRandComponent::Init(int32 dim, int32 input_dim, bool rand_per_frame) {
  rand_per_frame_ = rand_per_frame;
  dim_ = dim;
  input_dim_ = input_dim;
}

void* NormalRandComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in,
                              CuMatrixBase<BaseFloat> *out) const {
  
  if (!test_mode_) {
	  if (!rand_per_frame_) {
		  CuVector<BaseFloat> rand_vec_(dim_);
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(&rand_vec_);
		  out->CopyRowsFromVec(rand_vec_);
	  } else {
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(out);
	  }
  } else {
	  out->Set(0.0);
  }
  return NULL;
}

void NormalRandComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             void *memo,
                             Component *to_update, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
	in_deriv->SetZero();
  }
}

void NormalRandComponent::InitFromConfig(ConfigLine *cfl) {
  rand_per_frame_ = false;
  test_mode_ = false;
  if (!cfl->GetValue("output-dim", &dim_) || !cfl->GetValue("input-dim", &input_dim_) || dim_ <= 0) {
    KALDI_ERR << "Invalid values dim=" << dim_ << " input-dim=" << input_dim_;
  }
  cfl->GetValue("rand-per-frame", &rand_per_frame_);
  cfl->GetValue("test-mode", &test_mode_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  //Init(dim_, input_dim_, rand_per_frame_);
}

std::string NormalRandComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_
	<< ", rand-per-frame=" << (rand_per_frame_ ? "true" : "false")
	<< ", test-mode=" << (test_mode_ ? "true" : "false");
  return stream.str();
}

void NormalRandComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NormalRandComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<RandPerFrame>");
  WriteBasicType(os, binary, rand_per_frame_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "</NormalRandComponent>");
}

void NormalRandComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<NormalRandComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<RandPerFrame>");
  ReadBasicType(is, binary, &rand_per_frame_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "</NormalRandComponent>");
}

//////////

void NormalRandFComponent::Init(int32 dim, int32 input_dim, bool rand_per_frame) {
  rand_per_frame_ = rand_per_frame;
  dim_ = dim;
  input_dim_ = input_dim;
}

void* NormalRandFComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in,
                              CuMatrixBase<BaseFloat> *out) const {
  
  if (!rand_per_frame_) {
	  CuVector<BaseFloat> rand_vec_(dim_);
	  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(&rand_vec_);
	  out->CopyRowsFromVec(rand_vec_);
  } else {
	  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(out);
  }
  return NULL;
}

void NormalRandFComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             void *memo,
                             Component *to_update, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
	in_deriv->SetZero();
  }
}

void NormalRandFComponent::InitFromConfig(ConfigLine *cfl) {
  rand_per_frame_ = false;
  test_mode_ = false;
  if (!cfl->GetValue("output-dim", &dim_) || !cfl->GetValue("input-dim", &input_dim_) || dim_ <= 0) {
    KALDI_ERR << "Invalid values dim=" << dim_ << " input-dim=" << input_dim_;
  }
  cfl->GetValue("rand-per-frame", &rand_per_frame_);
  cfl->GetValue("test-mode", &test_mode_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  //Init(dim_, input_dim_, rand_per_frame_);
}

std::string NormalRandFComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_
	<< ", rand-per-frame=" << (rand_per_frame_ ? "true" : "false")
	<< ", test-mode=" << (test_mode_ ? "true" : "false");
  return stream.str();
}

void NormalRandFComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NormalRandFComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<RandPerFrame>");
  WriteBasicType(os, binary, rand_per_frame_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "</NormalRandFComponent>");
}

void NormalRandFComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<NormalRandFComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<RandPerFrame>");
  ReadBasicType(is, binary, &rand_per_frame_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "</NormalRandFComponent>");
}


//////////

void DropoutMaskSimpleComponent::Init(int32 dim, int32 input_dim, bool rand_per_frame, BaseFloat rand_num) {
  rand_per_frame_ = rand_per_frame;
  dim_ = dim;
  input_dim_ = input_dim;
  rand_num_ = rand_num;
}

void* DropoutMaskSimpleComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in,
                              CuMatrixBase<BaseFloat> *out) const {
  
  CuMatrix<BaseFloat> One_mix_all(in.NumRows(), dim_);
  One_mix_all.Set(1.0);
  
  if (!test_mode_) {
	  CuMatrix<BaseFloat> rand_vec(1, dim_);
	  if (!rand_per_frame_) {
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(&rand_vec);
		  rand_vec.Add(-rand_num_);
		  rand_vec.ApplyHeaviside();
		  out->CopyColsFromVec(rand_vec.Row(0));
	  } else {
		  CuMatrix<BaseFloat> rand_mat_tp(dim_, in.NumRows());
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(out);
		  out->Add(-rand_num_);
		  out->ApplyHeaviside();
	  }
  } else {
	  out->Set(1.0);
  }
  return NULL;
}

void DropoutMaskSimpleComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             void *memo,
                             Component *to_update, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
	in_deriv->SetZero();
  }
}

void DropoutMaskSimpleComponent::InitFromConfig(ConfigLine *cfl) {
  rand_per_frame_ = false;
  test_mode_ = false;
  rand_num_ = 0.5;
  if (!cfl->GetValue("output-dim", &dim_) || !cfl->GetValue("input-dim", &input_dim_) || dim_ <= 0) {
    KALDI_ERR << "Invalid values dim=" << dim_ << " input-dim=" << input_dim_;
  }
  cfl->GetValue("rand-per-frame", &rand_per_frame_);
  cfl->GetValue("test-mode", &test_mode_);
  cfl->GetValue("rand-num", &rand_num_);
  KALDI_ASSERT(rand_num_ <= 1.0);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  //Init(dim_, input_dim_, rand_per_frame_);
}

std::string DropoutMaskSimpleComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_
	<< ", rand-per-frame=" << (rand_per_frame_ ? "true" : "false")
	<< ", rand-num=" << rand_num_
	<< ", test-mode=" << (test_mode_ ? "true" : "false");
  return stream.str();
}

void DropoutMaskSimpleComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<DropoutMaskSimpleComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<RandPerFrame>");
  WriteBasicType(os, binary, rand_per_frame_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "<RandNum>");
  WriteBasicType(os, binary, rand_num_);
  WriteToken(os, binary, "</DropoutMaskSimpleComponent>");
}

void DropoutMaskSimpleComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<DropoutMaskSimpleComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<RandPerFrame>");
  ReadBasicType(is, binary, &rand_per_frame_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "<RandNum>");
  ReadBasicType(is, binary, &rand_num_);
  ExpectToken(is, binary, "</DropoutMaskSimpleComponent>");
}

//////////

void RandSelectNComponent::Init(int32 dim, int32 input_dim, bool rand_per_frame, int32 rand_num) {
  rand_per_frame_ = rand_per_frame;
  dim_ = dim;
  input_dim_ = input_dim;
  rand_num_ = rand_num;
}

void* RandSelectNComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in,
                              CuMatrixBase<BaseFloat> *out) const {
  
  CuMatrix<BaseFloat> One_mix_all(in.NumRows(), dim_);
  One_mix_all.Set(1.0);
  
  if (!test_mode_) {
	  CuMatrix<BaseFloat> rand_mat(in.NumRows(), dim_);
	  CuVector<BaseFloat> rand_vec(dim_);
	  if (!rand_per_frame_) {
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(&rand_vec);
		  BaseFloat max1;		  
		  for (int32 i=0; i<rand_num_; i++) {
			  max1 = rand_vec.Max(); // the current max
			  rand_vec.ReplaceValue(max1, -1.0); // the max one will be -1
		  }
		  rand_mat.CopyRowsFromVec(rand_vec);
		  rand_mat.ApplyHeaviside(); // the selected will be 0, others 1
		  One_mix_all.AddMat(-1.0, rand_mat); // 1 || 0
	  } else {
		  CuMatrix<BaseFloat> rand_mat_tp(dim_, in.NumRows());
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(&rand_mat_tp);
		  BaseFloat max1;
		  for (int32 j=0; j<in.NumRows(); j++) {
			  rand_vec.CopyColFromMat(rand_mat_tp, j); // for jth row	  
			  for (int32 i=0; i<rand_num_; i++) {
				  max1 = rand_vec.Max(); // the current max
				  rand_vec.ReplaceValue(max1, -1.0); // the max one will be -1
			  }
			  rand_mat_tp.CopyColFromVec(rand_vec, j);
		  }
		  rand_mat.CopyFromMat(rand_mat_tp, kTrans);
		  rand_mat.ApplyHeaviside(); // the selected will be 0, others 1
		  One_mix_all.AddMat(-1.0, rand_mat); // 1 || 0
	  }
  }
  out->CopyFromMat(One_mix_all);
  return NULL;
}

void RandSelectNComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             void *memo,
                             Component *to_update, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
	in_deriv->SetZero();
  }
}

void RandSelectNComponent::InitFromConfig(ConfigLine *cfl) {
  rand_per_frame_ = false;
  test_mode_ = false;
  rand_num_ = 1;
  if (!cfl->GetValue("output-dim", &dim_) || !cfl->GetValue("input-dim", &input_dim_) || dim_ <= 0) {
    KALDI_ERR << "Invalid values dim=" << dim_ << " input-dim=" << input_dim_;
  }
  cfl->GetValue("rand-per-frame", &rand_per_frame_);
  cfl->GetValue("test-mode", &test_mode_);
  cfl->GetValue("rand-num", &rand_num_);
  KALDI_ASSERT(rand_num_ <= dim_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  //Init(dim_, input_dim_, rand_per_frame_);
}

std::string RandSelectNComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_
	<< ", rand-per-frame=" << (rand_per_frame_ ? "true" : "false")
	<< ", rand-num=" << rand_num_
	<< ", test-mode=" << (test_mode_ ? "true" : "false");
  return stream.str();
}

void RandSelectNComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<RandSelectNComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<RandPerFrame>");
  WriteBasicType(os, binary, rand_per_frame_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "<RandNum>");
  WriteBasicType(os, binary, rand_num_);
  WriteToken(os, binary, "</RandSelectNComponent>");
}

void RandSelectNComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<RandSelectNComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<RandPerFrame>");
  ReadBasicType(is, binary, &rand_per_frame_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "<RandNum>");
  ReadBasicType(is, binary, &rand_num_);
  ExpectToken(is, binary, "</RandSelectNComponent>");
}

//////////

void NormalRandMultiSampleComponent::Init(int32 dim, int32 input_dim, bool rand_per_frame, bool test_time_sample) {
  test_time_sample_ = test_time_sample;
  rand_per_frame_ = rand_per_frame;
  dim_ = dim;
  input_dim_ = input_dim;
}

void* NormalRandMultiSampleComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in,
                              CuMatrixBase<BaseFloat> *out) const {
  
  if (!test_mode_ || test_time_sample_) {
	  if (!rand_per_frame_) {
		  CuVector<BaseFloat> rand_vec_(dim_);
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(&rand_vec_);
		  out->CopyRowsFromVec(rand_vec_);
	  } else {
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(out);
	  }
  } else {
	  out->Set(0.0);
  }
  return NULL;
}

void NormalRandMultiSampleComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             void *memo,
                             Component *to_update, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
	in_deriv->SetZero();
  }
}

void NormalRandMultiSampleComponent::InitFromConfig(ConfigLine *cfl) {
  rand_per_frame_ = false;
  test_time_sample_ = false;
  test_mode_ = false;
  if (!cfl->GetValue("output-dim", &dim_) || !cfl->GetValue("input-dim", &input_dim_) || dim_ <= 0) {
    KALDI_ERR << "Invalid values dim=" << dim_ << " input-dim=" << input_dim_;
  }
  cfl->GetValue("test-time-sample", &test_time_sample_);
  cfl->GetValue("rand-per-frame", &rand_per_frame_);
  cfl->GetValue("test-mode", &test_mode_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  //Init(dim_, input_dim_, rand_per_frame_);
}

std::string NormalRandMultiSampleComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_
	<< ", test-time-sample=" << (test_time_sample_ ? "true" : "false")
	<< ", rand-per-frame=" << (rand_per_frame_ ? "true" : "false")
	<< ", test-mode=" << (test_mode_ ? "true" : "false");
  return stream.str();
}

void NormalRandMultiSampleComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NormalRandMultiSampleComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<RandPerFrame>");
  WriteBasicType(os, binary, rand_per_frame_);
  WriteToken(os, binary, "<TestTimeSample>");
  WriteBasicType(os, binary, test_time_sample_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "</NormalRandMultiSampleComponent>");
}

void NormalRandMultiSampleComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<NormalRandMultiSampleComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<RandPerFrame>");
  ReadBasicType(is, binary, &rand_per_frame_);
  ExpectToken(is, binary, "<TestTimeSample>");
  ReadBasicType(is, binary, &test_time_sample_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "</NormalRandMultiSampleComponent>");
}

//////////

void TestZeroComponent::Init(int32 dim, int32 input_dim) {
  dim_ = dim;
  input_dim_ = input_dim;
}

void* TestZeroComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in,
                              CuMatrixBase<BaseFloat> *out) const {
  
  if (!test_mode_) {
	  out->Set(1.0);
  } else {
	  out->Set(0.0);
  }
  return NULL;
}

void TestZeroComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             void *memo,
                             Component *to_update, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
	in_deriv->SetZero();
  }
}

void TestZeroComponent::InitFromConfig(ConfigLine *cfl) {
  test_mode_ = false;
  if (!cfl->GetValue("output-dim", &dim_) || !cfl->GetValue("input-dim", &input_dim_) || dim_ <= 0) {
    KALDI_ERR << "Invalid values dim=" << dim_ << " input-dim=" << input_dim_;
  }
  cfl->GetValue("test-mode", &test_mode_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  //Init(dim_, input_dim_, rand_per_frame_);
}

std::string TestZeroComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_
	<< ", test-mode=" << (test_mode_ ? "true" : "false");
  return stream.str();
}

void TestZeroComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<TestZeroComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "</TestZeroComponent>");
}

void TestZeroComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<TestZeroComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "</TestZeroComponent>");
}

//////////

CopyNComponent::CopyNComponent(const CopyNComponent &other):
    input_dim_(other.input_dim_), output_dim_(other.output_dim_),
    scale_(other.scale_) { }

void CopyNComponent::InitFromConfig(ConfigLine *cfl) {
  scale_ = 1.0;
  bool ok = cfl->GetValue("input-dim", &input_dim_) &&
      cfl->GetValue("output-dim", &output_dim_);
  if (!ok)
    KALDI_ERR << "input-dim and output-dim must both be provided.";
  if (input_dim_ <= 0 || output_dim_ % input_dim_ != 0)
    KALDI_ERR << "Invalid values input-dim=" << input_dim_
              << " output-dim=" << output_dim_;
  cfl->GetValue("scale", &scale_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
}

void CopyNComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<CopyNComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "</CopyNComponent>");
}

void CopyNComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<CopyNComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "</CopyNComponent>");
}

std::string CopyNComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", scale=" << scale_;
  return stream.str();
}

void* CopyNComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(out->NumRows() == in.NumRows() &&
               out->NumCols() == output_dim_ &&
               in.NumCols() == input_dim_);
  out->AddMatBlocks(scale_, in, kNoTrans);
  return NULL;
}

void CopyNComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &, //in_value
    const CuMatrixBase<BaseFloat> &, // out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv) {
    in_deriv->AddMatBlocks(scale_, out_deriv, kNoTrans);
  }
}

//////////

CopyNPerDimComponent::CopyNPerDimComponent(const CopyNPerDimComponent &other):
    input_dim_(other.input_dim_), output_dim_(other.output_dim_),
	copy_num_(other.copy_num_), scale_(other.scale_) { }

void CopyNPerDimComponent::InitFromConfig(ConfigLine *cfl) {
  scale_ = 1.0;
  bool ok = cfl->GetValue("input-dim", &input_dim_) &&
      cfl->GetValue("output-dim", &output_dim_);
  if (!ok)
    KALDI_ERR << "input-dim and output-dim must both be provided.";
  if (input_dim_ <= 0 || output_dim_ % input_dim_ != 0)
    KALDI_ERR << "Invalid values input-dim=" << input_dim_
              << " output-dim=" << output_dim_;
  cfl->GetValue("scale", &scale_);
  copy_num_ = input_dim_;
  cfl->GetValue("copy-num", &copy_num_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
}

void CopyNPerDimComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<CopyNPerDimComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<CopyNum>");
  ReadBasicType(is, binary, &copy_num_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "</CopyNPerDimComponent>");
}

void CopyNPerDimComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<CopyNPerDimComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<CopyNum>");
  WriteBasicType(os, binary, copy_num_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "</CopyNPerDimComponent>");
}

std::string CopyNPerDimComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
		 << ", copy-num=" << copy_num_
         << ", scale=" << scale_;
  return stream.str();
}

void* CopyNPerDimComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(out->NumRows() == in.NumRows() &&
               out->NumCols() == output_dim_ &&
               in.NumCols() == input_dim_ &&
			   input_dim_%copy_num_ == 0 &&
			   output_dim_%copy_num_ == 0);
  int32 in_dim = input_dim_/copy_num_;
  int32 out_dim = output_dim_/copy_num_;
  for (int32 i=0; i<copy_num_; i++) {
	  (out->ColRange(i*out_dim, out_dim)).AddMatBlocks(scale_, in.ColRange(i*in_dim, in_dim), kNoTrans);
  }
  return NULL;
}

void CopyNPerDimComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &, //in_value
    const CuMatrixBase<BaseFloat> &, // out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  int32 in_dim = input_dim_/copy_num_;
  int32 out_dim = output_dim_/copy_num_;
  for (int32 i=0; i<copy_num_; i++) {
	  if (in_deriv) {
	  (in_deriv->ColRange(i*in_dim, in_dim)).AddMatBlocks(scale_, out_deriv.ColRange(i*out_dim, out_dim), kNoTrans);
	  }
  }
}

//////////

DimRangeComponent::DimRangeComponent(const DimRangeComponent &other):
    input_dim_(other.input_dim_), output_dim_(other.output_dim_),
    dim_offset_(other.dim_offset_) { }

void DimRangeComponent::InitFromConfig(ConfigLine *cfl) {
  dim_offset_ = 0;
  bool ok = cfl->GetValue("input-dim", &input_dim_) &&
      cfl->GetValue("output-dim", &output_dim_);
  if (!ok)
    KALDI_ERR << "input-dim and output-dim must both be provided.";
  cfl->GetValue("dim-offset", &dim_offset_);
  KALDI_ASSERT(output_dim_+dim_offset_ <= input_dim_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
}

void DimRangeComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<DimRangeComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<DimOffset>");
  ReadBasicType(is, binary, &dim_offset_);
  ExpectToken(is, binary, "</DimRangeComponent>");
  KALDI_ASSERT(output_dim_+dim_offset_ <= input_dim_);
}

void DimRangeComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<DimRangeComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<DimOffset>");
  WriteBasicType(os, binary, dim_offset_);
  WriteToken(os, binary, "</DimRangeComponent>");
}

std::string DimRangeComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", dim-offset=" << dim_offset_;
  return stream.str();
}

void* DimRangeComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(out->NumRows() == in.NumRows() &&
               out->NumCols() == output_dim_ &&
               in.NumCols() == input_dim_);
  out->CopyFromMat(in.ColRange(dim_offset_, output_dim_));
  return NULL;
}

void DimRangeComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &, //in_value
    const CuMatrixBase<BaseFloat> &, // out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv) {
    in_deriv->SetZero();
	(in_deriv->ColRange(dim_offset_, output_dim_)).CopyFromMat(out_deriv);
  }
}


//////////

void* MinValueComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in,
                              CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
  if (scale_ != 1.0)
    out->Scale(scale_);
  return NULL;
}

void MinValueComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             void *memo,
                             Component *to_update, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  in_deriv->Set(-1.0*scale_);
}

void MinValueComponent::InitFromConfig(ConfigLine *cfl) {
  scale_ = 1.0;
  cfl->GetValue("scale", &scale_);
  if (!cfl->GetValue("dim", &dim_) ||
      dim_ <= 0 || cfl->HasUnusedValues()) {
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  }
}

std::string MinValueComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_;
  if (scale_ != 1.0)
    stream << ", scale=" << scale_;
  return stream.str();
}

void MinValueComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<MinValueComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<BackpropScale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "</MinValueComponent>");
}

void MinValueComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<MinValueComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<BackpropScale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "</MinValueComponent>");
}


//////////

void* ExpComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
  out->ApplyCeiling(46.05); // avoid inf
  out->ApplyExp();
  return NULL;
}

void ExpComponent::Backprop(const std::string &debug_info,
                                const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                void *memo,
                                Component *to_update_in,
                                CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
	in_deriv->CopyFromMat(out_deriv);
	in_deriv->MulElements(out_value); // ey = y ex
  }
}


void ExpComponent::StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_value,
                                  void *memo) {
  // Only store stats about every other minibatch (but on the first minibatch,
  // always store it, which is necessary for the ConsolidateMemory() operation
  // to work correctly.
  if (RandInt(0, 1) == 0 && count_ != 0)
    return;
  CuMatrix<BaseFloat> temp_deriv(out_value.NumRows(),
                                 out_value.NumCols(),
                                 kUndefined);
  temp_deriv.CopyFromMat(out_value);
  StoreStatsInternal(out_value, &temp_deriv);
}


//////////

void* LogComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
  out->ApplyFloor(1e-20); // keep > 0
  out->ApplyLog();
  return NULL;
}

void LogComponent::Backprop(const std::string &debug_info,
                                const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &in_value,
                                const CuMatrixBase<BaseFloat> &,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                void *memo,
                                Component *to_update_in,
                                CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
	in_deriv->CopyFromMat(out_deriv);
	
	CuMatrix<BaseFloat> invx(in_value);
	invx.ApplyFloor(1e-20); // keep > 0
	invx.InvertElements(); // 1/x
	in_deriv->MulElements(invx); // ex = ey/x
  }
}


void LogComponent::StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_value,
                                  void *memo) {
  // Only store stats about every other minibatch (but on the first minibatch,
  // always store it, which is necessary for the ConsolidateMemory() operation
  // to work correctly.
  if (RandInt(0, 1) == 0 && count_ != 0)
    return;
  CuMatrix<BaseFloat> temp_deriv(in_value.NumRows(),
                                 in_value.NumCols(),
                                 kUndefined);
  CuMatrix<BaseFloat> invx(in_value);
  invx.ApplyFloor(1e-20); // keep > 0
  invx.InvertElements(); // 1/x
  temp_deriv.CopyFromMat(invx);
  StoreStatsInternal(out_value, &temp_deriv);
}



/////////////////////////////////////


void* SoftReLUComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  // y = log(1+exp(x))
  out->SoftHinge(in);
  return NULL;
}

void SoftReLUComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &, //out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
    // deriv_in = deriv_y * exp(x)/(1+exp(x))
	in_deriv->Sigmoid(in_value);
	in_deriv->MulElements(out_deriv);
  }
}

void SoftReLUComponent::StoreStats(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    void *memo) {
  // Only store stats about every other minibatch (but on the first minibatch,
  // always store it, which is necessary for the ConsolidateMemory() operation
  // to work correctly.
  if (RandInt(0, 1) == 0 && count_ != 0)
    return;
  CuMatrix<BaseFloat> temp_deriv(in_value.NumRows(),
                                 in_value.NumCols(),
                                 kUndefined);
  temp_deriv.Sigmoid(in_value);
  StoreStatsInternal(out_value, &temp_deriv);
}

//////////

void* FloorReLUComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
  out->ApplyFloor(0.001);
  return NULL;
}

void FloorReLUComponent::Backprop(const std::string &debug_info,
                                const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                void *memo,
                                Component *to_update_in,
                                CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
    in_deriv->CopyFromMat(out_deriv);
  }
}

void FloorReLUComponent::StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_value,
                                  void *memo) {
  // Only store stats about every other minibatch (but on the first minibatch,
  // always store it, which is necessary for the ConsolidateMemory() operation
  // to work correctly.
  if (RandInt(0, 1) == 0 && count_ != 0)
    return;
  CuMatrix<BaseFloat> temp_deriv(out_value.NumRows(),
                                 out_value.NumCols(),
                                 kUndefined);
  temp_deriv.Set(1.0);
  StoreStatsInternal(out_value, &temp_deriv);
}

//////////

void KLGaussianComponent::Init(int32 input_dim, int32 output_dim, BaseFloat scale, bool input_frame_scale, bool inv_frame_scale, bool output_sum, bool has_output)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  scale_ = scale;
  input_frame_scale_ = input_frame_scale;
  inv_frame_scale_ = inv_frame_scale;
  output_sum_ = output_sum;
  has_output_ = has_output;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0);
  KALDI_ASSERT(input_dim_ > output_dim_);
  if (output_sum_) {
	  KALDI_ASSERT(output_dim_ == 1);
  } else {
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == output_dim_*4+1);
	  } else {
		  KALDI_ASSERT(input_dim_ == output_dim_*4);
	  }
  }
}

void KLGaussianComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  BaseFloat scale = 1.0;
  bool input_frame_scale = false;
  bool inv_frame_scale = false;
  bool output_sum = false;
  bool has_output = true;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || output_dim <= 0)
    KALDI_ERR << "Invalid dim for layer of type "
              << Type();
  cfl->GetValue("scale", &scale);
  cfl->GetValue("input-frame-scale", &input_frame_scale);
  cfl->GetValue("inv-frame-scale", &inv_frame_scale);
  cfl->GetValue("output-sum", &output_sum);
  cfl->GetValue("has-output", &has_output);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim, scale, input_frame_scale, inv_frame_scale, output_sum, has_output);
}

/*
void KLGaussianComponent::SetTermps(const CuMatrixBase<BaseFloat> &inv_prior_std,
                 const CuMatrixBase<BaseFloat> &diff,
  			   const CuMatrixBase<BaseFloat> &diff_power_2,
  			   const CuMatrixBase<BaseFloat> &rate,
			   const CuVectorBase<BaseFloat> &frame_scale) const {
	inv_prior_std_->Resize(inv_prior_std.NumRows(), inv_prior_std.NumCols());
	inv_prior_std_->CopyFromMat(inv_prior_std);
	diff_->Resize(diff.NumRows(), diff.NumCols());
	diff_->CopyFromMat(diff);
	diff_power_2_->Resize(diff_power_2.NumRows(), diff_power_2.NumCols());
	diff_power_2_->CopyFromMat(diff_power_2);
	rate_->Resize(rate.NumRows(), rate.NumCols());
	rate_->CopyFromMat(rate);
	frame_scale_->Resize(frame_scale.Dim());
	frame_scale_->CopyFromVec(frame_scale);
}

void KLGaussianComponent::SetTermps(const CuMatrixBase<BaseFloat> &inv_prior_std,
                 const CuMatrixBase<BaseFloat> &diff,
  			   const CuMatrixBase<BaseFloat> &diff_power_2,
  			   const CuMatrixBase<BaseFloat> &rate) const {
	inv_prior_std_->Resize(inv_prior_std.NumRows(), inv_prior_std.NumCols());
	inv_prior_std_->CopyFromMat(inv_prior_std);
	diff_->Resize(diff.NumRows(), diff.NumCols());
	diff_->CopyFromMat(diff);
	diff_power_2_->Resize(diff_power_2.NumRows(), diff_power_2.NumCols());
	diff_power_2_->CopyFromMat(diff_power_2);
	rate_->Resize(rate.NumRows(), rate.NumCols());
	rate_->CopyFromMat(rate);
}
*/

void* KLGaussianComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == input_dim_);
  
  if (has_output_) {
	  
	  int32 num_rows = in.NumRows();
	  int32 num_dims;
	  if (input_frame_scale_) {
		num_dims = (input_dim_-1)/4;
	  } else {
		num_dims = input_dim_/4;
	  }
	  
	  CuSubMatrix<BaseFloat> post_mean(in, 0, num_rows, 0, num_dims);
	  CuSubMatrix<BaseFloat> post_std(in, 0, num_rows, num_dims, num_dims);
	  CuSubMatrix<BaseFloat> prior_mean(in, 0, num_rows, 2*num_dims, num_dims);
	  CuSubMatrix<BaseFloat> prior_std(in, 0, num_rows, 3*num_dims, num_dims);
	  
	  CuMatrix<BaseFloat> inv_prior_std(num_rows, num_dims);
	  CuMatrix<BaseFloat> diff(num_rows, num_dims);
	  CuMatrix<BaseFloat> diff_power_2(num_rows, num_dims);
	  CuMatrix<BaseFloat> rate(num_rows, num_dims);
	  
	  CuMatrix<BaseFloat> out_temp(num_rows, num_dims);
	  
	  inv_prior_std.CopyFromMat(prior_std);
	  inv_prior_std.InvertElements(); // 1/prior_std
	  
	  rate.CopyFromMat(post_std);
	  rate.MulElements(inv_prior_std); // post_std/prior_std
	  CuMatrix<BaseFloat> rate_power_2(rate);
	  rate_power_2.MulElements(rate); // post_std^2/prior_std^2
	  
	  diff.CopyFromMat(post_mean);
	  diff.AddMat(-1.0, prior_mean); // post_mean - prior_mean
	  diff.MulElements(inv_prior_std); // (post_mean - prior_mean)/prior_std
	  
	  diff_power_2.CopyFromMat(diff);
	  diff_power_2.MulElements(diff); // (post_mean - prior_mean)^2/prior_std^2
	  diff_power_2.AddMat(1.0, rate_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2
	  diff_power_2.Add(-1.0); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
	  
	  out_temp.CopyFromMat(diff_power_2);
	  rate_power_2.Add(1e-20); // keep > 0
	  rate_power_2.ApplyLog(); // log (post_std^2/prior_std^2)
	  out_temp.AddMat(-1.0, rate_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1
	  
	  if (output_sum_) {
		  out->AddMatBlocks(1.0, out_temp, kNoTrans); // sum_j KL_{i,j}
	  } else {
		  out->CopyFromMat(out_temp);
	  }
	  
	  if (input_frame_scale_) {
		  CuVector<BaseFloat> frame_scale(num_rows);
		  frame_scale.CopyColFromMat(in, 4*num_dims);
		  if (inv_frame_scale_) {
			  frame_scale.InvertElements();
		  }
		  out->MulRowsVec(frame_scale); // {(post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1} / N
		  // SetTermps(inv_prior_std, diff, diff_power_2, rate, frame_scale);
	  } else {
		  // SetTermps(inv_prior_std, diff, diff_power_2, rate);
	  }
	  
	  if (scale_ != 2.0) {
		  out->Scale(scale_*0.5); // scale * 0.5*{(post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1} (/N)
	  }
	  
  } else {
	  out->SetZero();
  }
  
  return NULL;
}

void KLGaussianComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {
  if (!in_deriv)  return;
  int32 num_rows = in_value.NumRows();
  int32 num_dims;
	  if (input_frame_scale_) {
		num_dims = (input_dim_-1)/4;
	  } else {
		num_dims = input_dim_/4;
	  }
  
  CuSubMatrix<BaseFloat> post_mean_deriv(*in_deriv, 0, num_rows, 0, num_dims);
  CuSubMatrix<BaseFloat> post_std_deriv(*in_deriv, 0, num_rows, num_dims, num_dims);
  CuSubMatrix<BaseFloat> prior_mean_deriv(*in_deriv, 0, num_rows, 2*num_dims, num_dims);
  CuSubMatrix<BaseFloat> prior_std_deriv(*in_deriv, 0, num_rows, 3*num_dims, num_dims);
  
  CuSubMatrix<BaseFloat> post_mean(in_value, 0, num_rows, 0, num_dims);
  CuSubMatrix<BaseFloat> post_std(in_value, 0, num_rows, num_dims, num_dims);
  CuSubMatrix<BaseFloat> prior_mean(in_value, 0, num_rows, 2*num_dims, num_dims);
  CuSubMatrix<BaseFloat> prior_std(in_value, 0, num_rows, 3*num_dims, num_dims);
  
  CuMatrix<BaseFloat> inv_prior_std(num_rows, num_dims);
  CuMatrix<BaseFloat> diff(num_rows, num_dims);
  CuMatrix<BaseFloat> diff_power_2(num_rows, num_dims);
  CuMatrix<BaseFloat> rate(num_rows, num_dims);
  
  
  inv_prior_std.CopyFromMat(prior_std);
  inv_prior_std.InvertElements(); // 1/prior_std
  
  rate.CopyFromMat(post_std);
  rate.MulElements(inv_prior_std); // post_std/prior_std
  CuMatrix<BaseFloat> rate_power_2(rate);
  rate_power_2.MulElements(rate); // post_std^2/prior_std^2
  
  diff.CopyFromMat(post_mean);
  diff.AddMat(-1.0, prior_mean); // post_mean - prior_mean
  diff.MulElements(inv_prior_std); // (post_mean - prior_mean)/prior_std
  
  diff_power_2.CopyFromMat(diff);
  diff_power_2.MulElements(diff); // (post_mean - prior_mean)^2/prior_std^2
  diff_power_2.AddMat(1.0, rate_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2
  diff_power_2.Add(-1.0); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
  
  post_mean_deriv.CopyFromMat(diff); // (post_mean - prior_mean)/prior_std
  post_mean_deriv.MulElements(inv_prior_std); // // (post_mean - prior_mean)/prior_std^2
  
  prior_mean_deriv.CopyFromMat(post_mean_deriv);
  prior_mean_deriv.Scale(-1.0); // - (post_mean - prior_mean)/prior_std^2
  
  post_std_deriv.CopyFromMat(rate); // post_std/prior_std
  post_std_deriv.MulElements(inv_prior_std); // post_std/prior_std^2
  CuMatrix<BaseFloat> inv_post_std(post_std);
  inv_post_std.InvertElements(); // 1/post_std
  post_std_deriv.AddMat(-1.0, inv_post_std); // post_std/prior_std^2 - 1/post_std
  
  prior_std_deriv.CopyFromMat(diff_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
  prior_std_deriv.MulElements(inv_prior_std); // (post_mean - prior_mean)^2/prior_std^3 + post_std^2/prior_std^3 - 1/prior_std
  prior_std_deriv.Scale(-1.0); // -(post_mean - prior_mean)^2/prior_std^3 - post_std^2/prior_std^3 + 1/prior_std
  
  CuMatrix<BaseFloat> out_deriv_temp(num_rows, input_dim_);
  out_deriv_temp.AddMatBlocks(1.0, out_deriv, kNoTrans);
  in_deriv->MulElements(out_deriv_temp); // min(KL) || -1 * KL
  
  if (input_frame_scale_) {
	  (in_deriv->ColRange(4*num_dims, 1)).Set(0.0);
	  CuVector<BaseFloat> frame_scale(num_rows);
	  frame_scale.CopyColFromMat(in_value, 4*num_dims);
	  if (inv_frame_scale_) {
		  frame_scale.InvertElements();
	  }
	  in_deriv->MulRowsVec(frame_scale); // deriv / N
  }
  
  if (scale_ != 1.0) {
	  in_deriv->Scale(scale_);
  }
  
}

void KLGaussianComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<KLGaussianComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "<InputeFrameScale>");
  ReadBasicType(is, binary, &input_frame_scale_);
  ExpectToken(is, binary, "<InvFrameScale>");
  ReadBasicType(is, binary, &inv_frame_scale_);
  ExpectToken(is, binary, "<OutputSum>");
  ReadBasicType(is, binary, &output_sum_);
  ExpectToken(is, binary, "<HasOutput>");
  ReadBasicType(is, binary, &has_output_);
  ExpectToken(is, binary, "</KLGaussianComponent>");
}

void KLGaussianComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<KLGaussianComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "<InputeFrameScale>");
  WriteBasicType(os, binary, input_frame_scale_);
  WriteToken(os, binary, "<InvFrameScale>");
  WriteBasicType(os, binary, inv_frame_scale_);
  WriteToken(os, binary, "<OutputSum>");
  WriteBasicType(os, binary, output_sum_);
  WriteToken(os, binary, "<HasOutput>");
  WriteBasicType(os, binary, has_output_);
  WriteToken(os, binary, "</KLGaussianComponent>");
}

std::string KLGaussianComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", scale=" << scale_
	<< ", input-frame-scale=" << (input_frame_scale_ ? "true" : "false")
	<< ", inv-frame-scale=" << (inv_frame_scale_ ? "true" : "false")
	<< ", output-sum=" << (output_sum_ ? "true" : "false")
	<< ", has-output=" << (has_output_ ? "true" : "false");
  return stream.str();
}

//////////

void KLGaussianLogStdComponent::Init(int32 input_dim, int32 output_dim, BaseFloat scale, bool input_frame_scale, bool inv_frame_scale, bool output_sum, bool has_output)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  scale_ = scale;
  input_frame_scale_ = input_frame_scale;
  inv_frame_scale_ = inv_frame_scale;
  output_sum_ = output_sum;
  has_output_ = has_output;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0);
  KALDI_ASSERT(input_dim_ > output_dim_);
  if (output_sum_) {
	  KALDI_ASSERT(output_dim_ == 1);
  } else {
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == output_dim_*4+1);
	  } else {
		  KALDI_ASSERT(input_dim_ == output_dim_*4);
	  }
  }
}

void KLGaussianLogStdComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  BaseFloat scale = 1.0;
  bool input_frame_scale = false;
  bool inv_frame_scale = false;
  bool output_sum = false;
  bool has_output = true;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || output_dim <= 0)
    KALDI_ERR << "Invalid dim for layer of type "
              << Type();
  cfl->GetValue("scale", &scale);
  cfl->GetValue("input-frame-scale", &input_frame_scale);
  cfl->GetValue("inv-frame-scale", &inv_frame_scale);
  cfl->GetValue("output-sum", &output_sum);
  cfl->GetValue("has-output", &has_output);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim, scale, input_frame_scale, inv_frame_scale, output_sum, has_output);
}

/*
void KLGaussianLogStdComponent::SetTermps(const CuMatrixBase<BaseFloat> &inv_prior_std,
                 const CuMatrixBase<BaseFloat> &diff,
  			   const CuMatrixBase<BaseFloat> &diff_power_2,
  			   const CuMatrixBase<BaseFloat> &rate,
			   const CuVectorBase<BaseFloat> &frame_scale) const {
	inv_prior_std_->Resize(inv_prior_std.NumRows(), inv_prior_std.NumCols());
	inv_prior_std_->CopyFromMat(inv_prior_std);
	diff_->Resize(diff.NumRows(), diff.NumCols());
	diff_->CopyFromMat(diff);
	diff_power_2_->Resize(diff_power_2.NumRows(), diff_power_2.NumCols());
	diff_power_2_->CopyFromMat(diff_power_2);
	rate_->Resize(rate.NumRows(), rate.NumCols());
	rate_->CopyFromMat(rate);
	frame_scale_->Resize(frame_scale.Dim());
	frame_scale_->CopyFromVec(frame_scale);
}

void KLGaussianLogStdComponent::SetTermps(const CuMatrixBase<BaseFloat> &inv_prior_std,
                 const CuMatrixBase<BaseFloat> &diff,
  			   const CuMatrixBase<BaseFloat> &diff_power_2,
  			   const CuMatrixBase<BaseFloat> &rate) const {
	inv_prior_std_->Resize(inv_prior_std.NumRows(), inv_prior_std.NumCols());
	inv_prior_std_->CopyFromMat(inv_prior_std);
	diff_->Resize(diff.NumRows(), diff.NumCols());
	diff_->CopyFromMat(diff);
	diff_power_2_->Resize(diff_power_2.NumRows(), diff_power_2.NumCols());
	diff_power_2_->CopyFromMat(diff_power_2);
	rate_->Resize(rate.NumRows(), rate.NumCols());
	rate_->CopyFromMat(rate);
}
*/

void* KLGaussianLogStdComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == input_dim_);
  
  if (has_output_) {
	  
	  int32 num_rows = in.NumRows();
	  int32 num_dims;
	  if (input_frame_scale_) {
		num_dims = (input_dim_-1)/4;
	  } else {
		num_dims = input_dim_/4;
	  }
	  
	  CuSubMatrix<BaseFloat> post_mean(in, 0, num_rows, 0, num_dims);
	  CuSubMatrix<BaseFloat> log_post_std(in, 0, num_rows, num_dims, num_dims);
	  CuSubMatrix<BaseFloat> prior_mean(in, 0, num_rows, 2*num_dims, num_dims);
	  CuSubMatrix<BaseFloat> log_prior_std(in, 0, num_rows, 3*num_dims, num_dims);
	  
	  CuMatrix<BaseFloat> inv_prior_std(num_rows, num_dims);
	  CuMatrix<BaseFloat> diff(num_rows, num_dims);
	  CuMatrix<BaseFloat> diff_power_2(num_rows, num_dims);
	  CuMatrix<BaseFloat> rate(num_rows, num_dims);
	  
	  CuMatrix<BaseFloat> out_temp(num_rows, num_dims);
	  
	  inv_prior_std.CopyFromMat(log_prior_std);
	  inv_prior_std.Scale(-1.0); // - log prior_std
	  inv_prior_std.ApplyCeiling(46.05); // avoid inf
	  inv_prior_std.ApplyExp(); // 1/prior_std
	  
	  rate.CopyFromMat(log_post_std);
	  rate.AddMat(-1.0, log_prior_std); // log post_std - log prior_std
	  rate.Scale(2.0); // 2 log post_std - 2 log prior_std
	  CuMatrix<BaseFloat> rate_power_2(rate);
	  rate_power_2.ApplyCeiling(46.05); // avoid inf
	  rate_power_2.ApplyExp(); // post_std^2/prior_std^2
	  
	  diff.CopyFromMat(post_mean);
	  diff.AddMat(-1.0, prior_mean); // post_mean - prior_mean
	  diff.MulElements(inv_prior_std); // (post_mean - prior_mean)/prior_std
	  
	  diff_power_2.CopyFromMat(diff);
	  diff_power_2.MulElements(diff); // (post_mean - prior_mean)^2/prior_std^2
	  diff_power_2.AddMat(1.0, rate_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2
	  diff_power_2.Add(-1.0); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
	  
	  out_temp.CopyFromMat(diff_power_2);
	  out_temp.AddMat(-1.0, rate); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1
	  
	  if (output_sum_) {
		  out->AddMatBlocks(1.0, out_temp, kNoTrans); // sum_j KL_{i,j}
	  } else {
		  out->CopyFromMat(out_temp);
	  }
	  
	  if (input_frame_scale_) {
		  CuVector<BaseFloat> frame_scale(num_rows);
		  frame_scale.CopyColFromMat(in, 4*num_dims);
		  if (inv_frame_scale_) {
			  frame_scale.InvertElements();
		  }
		  out->MulRowsVec(frame_scale); // {(post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1} / N
		  // SetTermps(inv_prior_std, diff, diff_power_2, rate, frame_scale);
	  } else {
		  // SetTermps(inv_prior_std, diff, diff_power_2, rate);
	  }
	  
	  if (scale_ != 2.0) {
		  out->Scale(scale_*0.5); // scale * 0.5*{(post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1} (/N)
	  }
	  
  } else {
	  out->SetZero();
  }
  
  return NULL;
}

void KLGaussianLogStdComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {
  if (!in_deriv)  return;
  int32 num_rows = in_value.NumRows();
  int32 num_dims;
	  if (input_frame_scale_) {
		num_dims = (input_dim_-1)/4;
	  } else {
		num_dims = input_dim_/4;
	  }
  
  CuSubMatrix<BaseFloat> post_mean_deriv(*in_deriv, 0, num_rows, 0, num_dims);
  CuSubMatrix<BaseFloat> log_post_std_deriv(*in_deriv, 0, num_rows, num_dims, num_dims);
  CuSubMatrix<BaseFloat> prior_mean_deriv(*in_deriv, 0, num_rows, 2*num_dims, num_dims);
  CuSubMatrix<BaseFloat> log_prior_std_deriv(*in_deriv, 0, num_rows, 3*num_dims, num_dims);
  
  CuSubMatrix<BaseFloat> post_mean(in_value, 0, num_rows, 0, num_dims);
  CuSubMatrix<BaseFloat> log_post_std(in_value, 0, num_rows, num_dims, num_dims);
  CuSubMatrix<BaseFloat> prior_mean(in_value, 0, num_rows, 2*num_dims, num_dims);
  CuSubMatrix<BaseFloat> log_prior_std(in_value, 0, num_rows, 3*num_dims, num_dims);
  
  CuMatrix<BaseFloat> inv_prior_std(num_rows, num_dims);
  CuMatrix<BaseFloat> diff(num_rows, num_dims);
  CuMatrix<BaseFloat> diff_power_2(num_rows, num_dims);
  CuMatrix<BaseFloat> rate(num_rows, num_dims);
  
  
  inv_prior_std.CopyFromMat(log_prior_std);
  inv_prior_std.Scale(-1.0); // - log prior_std
  inv_prior_std.ApplyCeiling(46.05); // avoid inf
  inv_prior_std.ApplyExp(); // 1/prior_std
  
  rate.CopyFromMat(log_post_std);
  rate.AddMat(-1.0, log_prior_std); // log post_std - log prior_std
  rate.Scale(2.0); // 2 log post_std - 2 log prior_std
  CuMatrix<BaseFloat> rate_power_2(rate);
  rate_power_2.ApplyCeiling(46.05); // avoid inf
  rate_power_2.ApplyExp(); // post_std^2/prior_std^2
  
  diff.CopyFromMat(post_mean);
  diff.AddMat(-1.0, prior_mean); // post_mean - prior_mean
  diff.MulElements(inv_prior_std); // (post_mean - prior_mean)/prior_std
  
  diff_power_2.CopyFromMat(diff);
  diff_power_2.MulElements(diff); // (post_mean - prior_mean)^2/prior_std^2
  diff_power_2.AddMat(1.0, rate_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2
  diff_power_2.Add(-1.0); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
  
  post_mean_deriv.CopyFromMat(diff); // (post_mean - prior_mean)/prior_std
  post_mean_deriv.MulElements(inv_prior_std); // // (post_mean - prior_mean)/prior_std^2
  
  prior_mean_deriv.CopyFromMat(post_mean_deriv);
  prior_mean_deriv.Scale(-1.0); // - (post_mean - prior_mean)/prior_std^2
  
  log_post_std_deriv.CopyFromMat(rate_power_2); // post_std^2/prior_std^2
  log_post_std_deriv.Add(-1.0); // post_std^2/prior_std^2 - 1
  
  log_prior_std_deriv.CopyFromMat(diff_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
  log_prior_std_deriv.Scale(-1.0); // -(post_mean - prior_mean)^2/prior_std^2 - post_std^2/prior_std^2 + 1
  
  CuMatrix<BaseFloat> out_deriv_temp(num_rows, input_dim_);
  out_deriv_temp.AddMatBlocks(1.0, out_deriv, kNoTrans);
  in_deriv->MulElements(out_deriv_temp); // min(KL) || -1 * KL
  
  if (input_frame_scale_) {
	  (in_deriv->ColRange(4*num_dims, 1)).Set(0.0);
	  CuVector<BaseFloat> frame_scale(num_rows);
	  frame_scale.CopyColFromMat(in_value, 4*num_dims);
	  if (inv_frame_scale_) {
		  frame_scale.InvertElements();
	  }
	  in_deriv->MulRowsVec(frame_scale); // deriv / N
  }
  
  if (scale_ != 1.0) {
	  in_deriv->Scale(scale_);
  }
  
}

void KLGaussianLogStdComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<KLGaussianLogStdComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "<InputeFrameScale>");
  ReadBasicType(is, binary, &input_frame_scale_);
  ExpectToken(is, binary, "<InvFrameScale>");
  ReadBasicType(is, binary, &inv_frame_scale_);
  ExpectToken(is, binary, "<OutputSum>");
  ReadBasicType(is, binary, &output_sum_);
  ExpectToken(is, binary, "<HasOutput>");
  ReadBasicType(is, binary, &has_output_);
  ExpectToken(is, binary, "</KLGaussianLogStdComponent>");
}

void KLGaussianLogStdComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<KLGaussianLogStdComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "<InputeFrameScale>");
  WriteBasicType(os, binary, input_frame_scale_);
  WriteToken(os, binary, "<InvFrameScale>");
  WriteBasicType(os, binary, inv_frame_scale_);
  WriteToken(os, binary, "<OutputSum>");
  WriteBasicType(os, binary, output_sum_);
  WriteToken(os, binary, "<HasOutput>");
  WriteBasicType(os, binary, has_output_);
  WriteToken(os, binary, "</KLGaussianLogStdComponent>");
}

std::string KLGaussianLogStdComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", scale=" << scale_
	<< ", input-frame-scale=" << (input_frame_scale_ ? "true" : "false")
	<< ", inv-frame-scale=" << (inv_frame_scale_ ? "true" : "false")
	<< ", output-sum=" << (output_sum_ ? "true" : "false")
	<< ", has-output=" << (has_output_ ? "true" : "false");
  return stream.str();
}

//////////

void BayesVecKLGaussianLogStdComponent::Init(int32 input_dim, int32 output_dim, BaseFloat KL_scale, bool input_frame_scale, bool inv_frame_scale, bool rand_per_frame, bool KL_output)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  KL_scale_ = KL_scale;
  input_frame_scale_ = input_frame_scale;
  inv_frame_scale_ = inv_frame_scale;
  rand_per_frame_ = rand_per_frame;
  KL_output_ = KL_output;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0);
  if (KL_output_) {
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == (output_dim_-1)*4+1);
	  } else {
		  KALDI_ASSERT(input_dim_ == (output_dim_-1)*4);
	  }
  } else {
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == output_dim_*4+1);
	  } else {
		  KALDI_ASSERT(input_dim_ == output_dim_*4);
	  }
  }
}

void BayesVecKLGaussianLogStdComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  BaseFloat KL_scale = 1.0;
  test_mode_ = false;
  bool input_frame_scale = false;
  bool inv_frame_scale = false;
  bool rand_per_frame = false;
  bool KL_output = true;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || output_dim <= 0)
    KALDI_ERR << "Invalid dim for layer of type "
              << Type();
  cfl->GetValue("KL-scale", &KL_scale);
  cfl->GetValue("input-frame-scale", &input_frame_scale);
  cfl->GetValue("inv-frame-scale", &inv_frame_scale);
  cfl->GetValue("rand-per-frame", &rand_per_frame);
  cfl->GetValue("test-mode", &test_mode_);
  cfl->GetValue("KL-output", &KL_output);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim, KL_scale, input_frame_scale, inv_frame_scale, rand_per_frame, KL_output);
}

void* BayesVecKLGaussianLogStdComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == input_dim_ && out->NumCols() == output_dim_);
  
  int32 num_rows = in.NumRows();
  int32 num_dims;
  if (input_frame_scale_) {
	  num_dims = (input_dim_-1)/4;
  } else {
	  num_dims = input_dim_/4;
  }
  if (KL_output_) {
	  output_dim_ == num_dims+1;
  } else {
	  output_dim_ == num_dims;
  }
  
  CuSubMatrix<BaseFloat> post_mean(in, 0, num_rows, 0, num_dims);
  CuSubMatrix<BaseFloat> log_post_std(in, 0, num_rows, num_dims, num_dims);
  CuSubMatrix<BaseFloat> out_vec(*out, 0, num_rows, 0, num_dims);
  
  if (!test_mode_) {
	  
	  Memo *memo = new Memo;
	  (memo->post_std).Resize(num_rows, num_dims);
	  (memo->post_std).CopyFromMat(log_post_std);
	  (memo->post_std).ApplyCeiling(46.05); // avoid inf
	  (memo->post_std).ApplyExp(); // post_std
	  
	  out_vec.CopyFromMat(memo->post_std); // std
	  
	  if (!rand_per_frame_) {
		  (memo->rand_vec).Resize(num_dims);
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(&(memo->rand_vec));
		  out_vec.MulColsVec(memo->rand_vec); // std * rand
	  } else {
		  (memo->rand_mat).Resize(num_rows, num_dims);
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(&(memo->rand_mat));
		  out_vec.MulElements(memo->rand_mat); // std * rand
	  }
	  
	  out_vec.AddMat(1.0, post_mean); // std * rand + mean
		  
	  if (KL_scale_ != 0) {
	  
		  CuSubMatrix<BaseFloat> prior_mean(in, 0, num_rows, 2*num_dims, num_dims);
		  CuSubMatrix<BaseFloat> log_prior_std(in, 0, num_rows, 3*num_dims, num_dims);
		  
		  (memo->inv_prior_std).Resize(num_rows, num_dims);
		  (memo->diff).Resize(num_rows, num_dims);
		  
		  CuMatrix<BaseFloat> rate(num_rows, num_dims);
		  
		  (memo->inv_prior_std).CopyFromMat(log_prior_std);
		  (memo->inv_prior_std).Scale(-1.0); // - log prior_std
		  (memo->inv_prior_std).ApplyCeiling(46.05); // avoid inf
		  (memo->inv_prior_std).ApplyExp(); // 1/prior_std
		  
		  rate.CopyFromMat(log_post_std);
		  rate.AddMat(-1.0, log_prior_std); // log post_std - log prior_std
		  rate.Scale(2.0); // 2 log post_std - 2 log prior_std
		  
		  memo->rate_power_2 = rate;
		  (memo->rate_power_2).ApplyCeiling(46.05); // avoid inf
		  (memo->rate_power_2).ApplyExp(); // post_std^2/prior_std^2
		  
		  (memo->diff).CopyFromMat(post_mean);
		  (memo->diff).AddMat(-1.0, prior_mean); // post_mean - prior_mean
		  (memo->diff).MulElements(memo->inv_prior_std); // (post_mean - prior_mean)/prior_std
		  
		  memo->diff_power_2 = memo->diff;
		  (memo->diff_power_2).MulElements(memo->diff); // (post_mean - prior_mean)^2/prior_std^2
		  (memo->diff_power_2).AddMat(1.0, memo->rate_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2
		  (memo->diff_power_2).Add(-1.0); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
		  
		  if (input_frame_scale_) {
			  (memo->frame_scale).Resize(num_rows);
			  (memo->frame_scale).CopyColFromMat(in, 4*num_dims);
			  if (inv_frame_scale_) {
				  (memo->frame_scale).InvertElements();
			  }
		  }
		  
		  if (KL_output_) {
			  CuSubMatrix<BaseFloat> out_KL(*out, 0, num_rows, num_dims, 1);
			  out_KL.SetZero();
			  CuMatrix<BaseFloat> out_temp = memo->diff_power_2;
			  out_temp.AddMat(-1.0, rate); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1
			  out_KL.AddMatBlocks(0.5, out_temp, kNoTrans); // 1/2 sum_j KL_{i,j}
			  
			  if (input_frame_scale_) {
				  out_KL.MulRowsVec(memo->frame_scale); // {(post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1} / N
			  }
			  
			  if (KL_scale_ != 1.0) {
				  out_KL.Scale(KL_scale_); // scale * 0.5*{(post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1} (/N)
			  }
		  }
		  
	  }
	  
	  return static_cast<void*>(memo);
  } else {
	  out_vec.CopyFromMat(post_mean); // std * 0 + mean
	  return NULL;
  }
  
}

void BayesVecKLGaussianLogStdComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {

  int32 num_rows = in_value.NumRows();
  int32 num_dims;
  if (input_frame_scale_) {
	  num_dims = (input_dim_-1)/4;
  } else {
	  num_dims = input_dim_/4;
  }
  
  CuSubMatrix<BaseFloat> post_mean_deriv(*in_deriv, 0, num_rows, 0, num_dims);
  CuSubMatrix<BaseFloat> log_post_std_deriv(*in_deriv, 0, num_rows, num_dims, num_dims);
  CuSubMatrix<BaseFloat> out_deriv_vec(out_deriv, 0, num_rows, 0, num_dims);

  post_mean_deriv.CopyFromMat(out_deriv_vec); // d/dmean = deriv
  if (!test_mode_) {
	  
	  Memo *memo_in = static_cast<Memo*>(memo);
	  KALDI_ASSERT(memo_in != NULL && "memo not passed into backprop");
	  
	  log_post_std_deriv.CopyFromMat(out_deriv_vec); // deriv
	  log_post_std_deriv.MulElements(memo_in->post_std); // deriv*post_std
	  if (!rand_per_frame_) {
		  log_post_std_deriv.MulColsVec(memo_in->rand_vec); // deriv*post_std * rand
	  } else {
		  log_post_std_deriv.MulElements(memo_in->rand_mat); // deriv*post_std * rand
	  }

	  if (KL_scale_ != 0) {
		  CuSubMatrix<BaseFloat> prior_mean_deriv(*in_deriv, 0, num_rows, 2*num_dims, num_dims);
		  CuSubMatrix<BaseFloat> log_prior_std_deriv(*in_deriv, 0, num_rows, 3*num_dims, num_dims);
		  
		  CuMatrix<BaseFloat> post_mean_deriv_KL = memo_in->diff; // (post_mean - prior_mean)/prior_std
		  post_mean_deriv_KL.MulElements(memo_in->inv_prior_std); // (post_mean - prior_mean)/prior_std^2
		  
		  CuMatrix<BaseFloat> log_post_std_deriv_KL = memo_in->rate_power_2; // post_std^2/prior_std^2
		  log_post_std_deriv_KL.Add(-1.0); // post_std^2/prior_std^2 - 1
		  
		  prior_mean_deriv.SetZero();
		  log_prior_std_deriv.SetZero();
		  prior_mean_deriv.AddMat(KL_scale_, post_mean_deriv_KL); // - (post_mean - prior_mean)/prior_std^2
		  log_prior_std_deriv.AddMat(KL_scale_, memo_in->diff_power_2); // -(post_mean - prior_mean)^2/prior_std^2 - post_std^2/prior_std^2 + 1
		  
		  if (input_frame_scale_) {
			  (in_deriv->ColRange(4*num_dims, 1)).SetZero();
			  post_mean_deriv_KL.MulRowsVec(memo_in->frame_scale); // deriv_post_mean_KL / N
			  log_post_std_deriv_KL.MulRowsVec(memo_in->frame_scale); // deriv_post_std_KL / N
			  prior_mean_deriv.MulRowsVec(memo_in->frame_scale); // deriv_prior_mean / N
			  log_prior_std_deriv.MulRowsVec(memo_in->frame_scale); // deriv_prior_std / N
		  }
		  post_mean_deriv.AddMat(-1.0*KL_scale_, post_mean_deriv_KL);
		  log_post_std_deriv.AddMat(-1.0*KL_scale_, log_post_std_deriv_KL);
	  }
  }
}

void BayesVecKLGaussianLogStdComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<BayesVecKLGaussianLogStdComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<KLScale>");
  ReadBasicType(is, binary, &KL_scale_);
  ExpectToken(is, binary, "<InputeFrameScale>");
  ReadBasicType(is, binary, &input_frame_scale_);
  ExpectToken(is, binary, "<InvFrameScale>");
  ReadBasicType(is, binary, &inv_frame_scale_);
  ExpectToken(is, binary, "<RandPerFrame>");
  ReadBasicType(is, binary, &rand_per_frame_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "<KLOutput>");
  ReadBasicType(is, binary, &KL_output_);
  ExpectToken(is, binary, "</BayesVecKLGaussianLogStdComponent>");
}

void BayesVecKLGaussianLogStdComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<BayesVecKLGaussianLogStdComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<KLScale>");
  WriteBasicType(os, binary, KL_scale_);
  WriteToken(os, binary, "<InputeFrameScale>");
  WriteBasicType(os, binary, input_frame_scale_);
  WriteToken(os, binary, "<InvFrameScale>");
  WriteBasicType(os, binary, inv_frame_scale_);
  WriteToken(os, binary, "<RandPerFrame>");
  WriteBasicType(os, binary, rand_per_frame_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "<KLOutput>");
  WriteBasicType(os, binary, KL_output_);
  WriteToken(os, binary, "</BayesVecKLGaussianLogStdComponent>");
}

std::string BayesVecKLGaussianLogStdComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", KL-scale=" << KL_scale_
	<< ", input-frame-scale=" << (input_frame_scale_ ? "true" : "false")
	<< ", inv-frame-scale=" << (inv_frame_scale_ ? "true" : "false")
	<< ", rand-per-frame=" << (rand_per_frame_ ? "true" : "false")
	<< ", test-mode=" << (test_mode_ ? "true" : "false")
	<< ", KL-output=" << (KL_output_ ? "true" : "false");
  return stream.str();
}

//////////

void BayesVecKLGaussianComponent::Init(int32 input_dim, int32 output_dim, BaseFloat KL_scale, bool input_frame_scale, bool inv_frame_scale, bool rand_per_frame, bool KL_output, bool fix_rand_val)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  KL_scale_ = KL_scale;
  input_frame_scale_ = input_frame_scale;
  inv_frame_scale_ = inv_frame_scale;
  rand_per_frame_ = rand_per_frame;
  KL_output_ = KL_output;
  fix_rand_val_ = fix_rand_val;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0);
  if (KL_output_) {
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == (output_dim_-1)*4+1);
	  } else {
		  KALDI_ASSERT(input_dim_ == (output_dim_-1)*4);
	  }
  } else {
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == output_dim_*4+1);
	  } else {
		  KALDI_ASSERT(input_dim_ == output_dim_*4);
	  }
  }
}

void BayesVecKLGaussianComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  BaseFloat KL_scale = 1.0;
  test_mode_ = false;
  bool input_frame_scale = false;
  bool inv_frame_scale = false;
  bool rand_per_frame = false;
  bool KL_output = true;
  bool fix_rand_val = false;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || output_dim <= 0)
    KALDI_ERR << "Invalid dim for layer of type "
              << Type();
  cfl->GetValue("KL-scale", &KL_scale);
  cfl->GetValue("input-frame-scale", &input_frame_scale);
  cfl->GetValue("inv-frame-scale", &inv_frame_scale);
  cfl->GetValue("rand-per-frame", &rand_per_frame);
  cfl->GetValue("test-mode", &test_mode_);
  cfl->GetValue("KL-output", &KL_output);
  cfl->GetValue("fix-rand-val", &fix_rand_val);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim, KL_scale, input_frame_scale, inv_frame_scale, rand_per_frame, KL_output, fix_rand_val);
}

void* BayesVecKLGaussianComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == input_dim_ && out->NumCols() == output_dim_);
  
  int32 num_rows = in.NumRows();
  int32 num_dims;
  if (input_frame_scale_) {
	  num_dims = (input_dim_-1)/4;
  } else {
	  num_dims = input_dim_/4;
  }
  if (KL_output_) {
	  output_dim_ == num_dims+1;
  } else {
	  output_dim_ == num_dims;
  }
  
  CuSubMatrix<BaseFloat> post_mean(in, 0, num_rows, 0, num_dims);
  CuSubMatrix<BaseFloat> post_std(in, 0, num_rows, num_dims, num_dims);
  CuSubMatrix<BaseFloat> out_vec(*out, 0, num_rows, 0, num_dims);
  
  if (!test_mode_) {
	  
	  Memo *memo = new Memo;
	  
	  out_vec.CopyFromMat(post_std); // std
	  out_vec.ApplyFloor(1e-20); // keep > 0
	  
	  if (!rand_per_frame_) {
		  (memo->rand_vec).Resize(num_dims);
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(&(memo->rand_vec));
		  if (fix_rand_val_) {
		    (memo->rand_vec).SetZero();
		  }
		  out_vec.MulColsVec(memo->rand_vec); // std * rand
	  } else {
		  (memo->rand_mat).Resize(num_rows, num_dims);
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(&(memo->rand_mat));
		  if (fix_rand_val_) {
		    (memo->rand_mat).SetZero();
		  }
		  out_vec.MulElements(memo->rand_mat); // std * rand
	  }
	  
	  out_vec.AddMat(1.0, post_mean); // std * rand + mean
	  
	  if (KL_scale_ != 0) {
	  
		  CuSubMatrix<BaseFloat> prior_mean(in, 0, num_rows, 2*num_dims, num_dims);
		  CuSubMatrix<BaseFloat> prior_std(in, 0, num_rows, 3*num_dims, num_dims);
		  
		  (memo->inv_prior_std).Resize(num_rows, num_dims);
		  (memo->diff).Resize(num_rows, num_dims);
		  (memo->rate).Resize(num_rows, num_dims);
		  
		  (memo->inv_prior_std).CopyFromMat(prior_std);
		  (memo->inv_prior_std).ApplyFloor(1e-20); // keep > 0
		  (memo->inv_prior_std).InvertElements(); // 1/prior_std
		  
		  (memo->rate).CopyFromMat(post_std);
		  (memo->rate).ApplyFloor(1e-20); // keep > 0
		  (memo->rate).MulElements(memo->inv_prior_std); // post_std/prior_std
		  
		  memo->rate_power_2 = memo->rate;
		  (memo->rate_power_2).MulElements(memo->rate); // post_std^2/prior_std^2
		  
		  (memo->diff).CopyFromMat(post_mean);
		  (memo->diff).AddMat(-1.0, prior_mean); // post_mean - prior_mean
		  (memo->diff).MulElements(memo->inv_prior_std); // (post_mean - prior_mean)/prior_std
		  
		  memo->diff_power_2 = memo->diff;
		  (memo->diff_power_2).MulElements(memo->diff); // (post_mean - prior_mean)^2/prior_std^2
		  (memo->diff_power_2).AddMat(1.0, memo->rate_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2
		  (memo->diff_power_2).Add(-1.0); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
		  
		  if (input_frame_scale_) {
			  (memo->frame_scale).Resize(num_rows);
			  (memo->frame_scale).CopyColFromMat(in, 4*num_dims);
			  if (inv_frame_scale_) {
				  (memo->frame_scale).InvertElements();
			  }
		  }
		  
		  if (KL_output_) {
			  CuSubMatrix<BaseFloat> out_KL(*out, 0, num_rows, num_dims, 1);
			  out_KL.SetZero();
			  CuMatrix<BaseFloat> out_temp = memo->diff_power_2;
			  CuMatrix<BaseFloat> rate_power_2 = memo->rate_power_2;
			  rate_power_2.Add(1e-20); // keep > 0
			  rate_power_2.ApplyLog(); // log (post_std^2/prior_std^2)
			  out_temp.AddMat(-1.0, rate_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1
			  out_KL.AddMatBlocks(0.5, out_temp, kNoTrans); // 1/2 sum_j KL_{i,j}
			  
			  if (KL_scale_ != 1.0) {
				  out_KL.Scale(KL_scale_); // scale * 0.5*{(post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1} (/N)
			  }
			  
			  (memo->KL_out).Resize(num_rows, 1);
			  (memo->KL_out).CopyFromMat(out_KL);
			  
			  if (input_frame_scale_) {
				  out_KL.MulRowsVec(memo->frame_scale); // {(post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1} / N
			  }
		  }
		  
	  }
	  
	  return static_cast<void*>(memo);
  } else {
	  out_vec.CopyFromMat(post_mean); // std * 0 + mean
	  return NULL;
  }
  
}

void BayesVecKLGaussianComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {

  int32 num_rows = in_value.NumRows();
  int32 num_dims;
  if (input_frame_scale_) {
	  num_dims = (input_dim_-1)/4;
  } else {
	  num_dims = input_dim_/4;
  }
  
  CuSubMatrix<BaseFloat> post_mean_deriv(*in_deriv, 0, num_rows, 0, num_dims);
  CuSubMatrix<BaseFloat> post_std_deriv(*in_deriv, 0, num_rows, num_dims, num_dims);
  CuSubMatrix<BaseFloat> out_deriv_vec(out_deriv, 0, num_rows, 0, num_dims);

  post_mean_deriv.CopyFromMat(out_deriv_vec); // d/dmean = deriv
  if (!test_mode_) {
	  
	  Memo *memo_in = static_cast<Memo*>(memo);
	  KALDI_ASSERT(memo_in != NULL && "memo not passed into backprop");
	  
	  post_std_deriv.CopyFromMat(out_deriv_vec); // deriv
	  if (!rand_per_frame_) {
		  post_std_deriv.MulColsVec(memo_in->rand_vec); // deriv * rand
	  } else {
		  post_std_deriv.MulElements(memo_in->rand_mat); // deriv * rand
	  }

	  if (KL_scale_ != 0) {
		  CuSubMatrix<BaseFloat> prior_mean_deriv(*in_deriv, 0, num_rows, 2*num_dims, num_dims);
		  CuSubMatrix<BaseFloat> prior_std_deriv(*in_deriv, 0, num_rows, 3*num_dims, num_dims);
		  CuSubMatrix<BaseFloat> post_std(in_value, 0, num_rows, num_dims, num_dims);
		  
		  CuMatrix<BaseFloat> post_mean_deriv_KL = memo_in->diff; // (post_mean - prior_mean)/prior_std
		  post_mean_deriv_KL.MulElements(memo_in->inv_prior_std); // (post_mean - prior_mean)/prior_std^2
		  
		  CuMatrix<BaseFloat> post_std_deriv_KL = memo_in->rate; // post_std/prior_std
		  post_std_deriv_KL.MulElements(memo_in->inv_prior_std); // post_std/prior_std^2
		  CuMatrix<BaseFloat> inv_post_std(post_std);
		  inv_post_std.ApplyFloor(1e-20); // keep > 0
		  inv_post_std.InvertElements(); // 1/post_std
		  post_std_deriv_KL.AddMat(-1.0, inv_post_std); // post_std/prior_std^2 - 1/post_std
		  
		  prior_mean_deriv.SetZero();
		  prior_std_deriv.SetZero();
		  prior_mean_deriv.AddMat(KL_scale_, post_mean_deriv_KL); // - (post_mean - prior_mean)/prior_std^2
		  prior_std_deriv.AddMat(KL_scale_, memo_in->diff_power_2); // -(post_mean - prior_mean)^2/prior_std^2 - post_std^2/prior_std^2 + 1
		  prior_std_deriv.MulElements(memo_in->inv_prior_std); // -(post_mean - prior_mean)^2/prior_std^3 - post_std^2/prior_std^3 + 1/prior_std
		  
		  if (input_frame_scale_) {
			  (in_deriv->ColRange(4*num_dims, 1)).SetZero();
			  if (KL_output_) {
			  (in_deriv->ColRange(4*num_dims, 1)).CopyFromMat(memo_in->KL_out);
			  }
			  post_mean_deriv_KL.MulRowsVec(memo_in->frame_scale); // deriv_post_mean_KL / N
			  post_std_deriv_KL.MulRowsVec(memo_in->frame_scale); // deriv_post_std_KL / N
			  prior_mean_deriv.MulRowsVec(memo_in->frame_scale); // deriv_prior_mean / N
			  prior_std_deriv.MulRowsVec(memo_in->frame_scale); // deriv_prior_std / N
		  }
		  post_mean_deriv.AddMat(-1.0*KL_scale_, post_mean_deriv_KL);
		  post_std_deriv.AddMat(-1.0*KL_scale_, post_std_deriv_KL);
	  }
  }
}

void BayesVecKLGaussianComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<BayesVecKLGaussianComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<KLScale>");
  ReadBasicType(is, binary, &KL_scale_);
  ExpectToken(is, binary, "<InputeFrameScale>");
  ReadBasicType(is, binary, &input_frame_scale_);
  ExpectToken(is, binary, "<InvFrameScale>");
  ReadBasicType(is, binary, &inv_frame_scale_);
  ExpectToken(is, binary, "<RandPerFrame>");
  ReadBasicType(is, binary, &rand_per_frame_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "<KLOutput>");
  ReadBasicType(is, binary, &KL_output_);
  ExpectToken(is, binary, "<FixRandVal>");
  ReadBasicType(is, binary, &fix_rand_val_);
  ExpectToken(is, binary, "</BayesVecKLGaussianComponent>");
}

void BayesVecKLGaussianComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<BayesVecKLGaussianComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<KLScale>");
  WriteBasicType(os, binary, KL_scale_);
  WriteToken(os, binary, "<InputeFrameScale>");
  WriteBasicType(os, binary, input_frame_scale_);
  WriteToken(os, binary, "<InvFrameScale>");
  WriteBasicType(os, binary, inv_frame_scale_);
  WriteToken(os, binary, "<RandPerFrame>");
  WriteBasicType(os, binary, rand_per_frame_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "<KLOutput>");
  WriteBasicType(os, binary, KL_output_);
  WriteToken(os, binary, "<FixRandVal>");
  WriteBasicType(os, binary, fix_rand_val_);
  WriteToken(os, binary, "</BayesVecKLGaussianComponent>");
}

std::string BayesVecKLGaussianComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", KL-scale=" << KL_scale_
	<< ", input-frame-scale=" << (input_frame_scale_ ? "true" : "false")
	<< ", inv-frame-scale=" << (inv_frame_scale_ ? "true" : "false")
	<< ", rand-per-frame=" << (rand_per_frame_ ? "true" : "false")
	<< ", test-mode=" << (test_mode_ ? "true" : "false")
	<< ", KL-output=" << (KL_output_ ? "true" : "false")
	<< ", fix-rand-val=" << (fix_rand_val_ ? "true" : "false");
  return stream.str();
}

//////////

void BayesVecKLGaussianFComponent::Init(int32 input_dim, int32 output_dim, BaseFloat KL_scale, bool input_frame_scale, bool inv_frame_scale, bool rand_per_frame, bool KL_output, bool fix_rand_val)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  KL_scale_ = KL_scale;
  input_frame_scale_ = input_frame_scale;
  inv_frame_scale_ = inv_frame_scale;
  rand_per_frame_ = rand_per_frame;
  KL_output_ = KL_output;
  fix_rand_val_ = fix_rand_val;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0);
  if (KL_output_) {
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == (output_dim_-1)*4+1);
	  } else {
		  KALDI_ASSERT(input_dim_ == (output_dim_-1)*4);
	  }
  } else {
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == output_dim_*4+1);
	  } else {
		  KALDI_ASSERT(input_dim_ == output_dim_*4);
	  }
  }
}

void BayesVecKLGaussianFComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  BaseFloat KL_scale = 1.0;
  test_mode_ = false;
  bool input_frame_scale = false;
  bool inv_frame_scale = false;
  bool rand_per_frame = false;
  bool KL_output = true;
  bool fix_rand_val = false;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || output_dim <= 0)
    KALDI_ERR << "Invalid dim for layer of type "
              << Type();
  cfl->GetValue("KL-scale", &KL_scale);
  cfl->GetValue("input-frame-scale", &input_frame_scale);
  cfl->GetValue("inv-frame-scale", &inv_frame_scale);
  cfl->GetValue("rand-per-frame", &rand_per_frame);
  cfl->GetValue("test-mode", &test_mode_);
  cfl->GetValue("KL-output", &KL_output);
  cfl->GetValue("fix-rand-val", &fix_rand_val);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim, KL_scale, input_frame_scale, inv_frame_scale, rand_per_frame, KL_output, fix_rand_val);
}

void* BayesVecKLGaussianFComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == input_dim_ && out->NumCols() == output_dim_);
  
  int32 num_rows = in.NumRows();
  int32 num_dims;
  if (input_frame_scale_) {
	  num_dims = (input_dim_-1)/4;
  } else {
	  num_dims = input_dim_/4;
  }
  if (KL_output_) {
	  output_dim_ == num_dims+1;
  } else {
	  output_dim_ == num_dims;
  }
  
  CuSubMatrix<BaseFloat> post_mean(in, 0, num_rows, 0, num_dims);
  CuSubMatrix<BaseFloat> post_std(in, 0, num_rows, num_dims, num_dims);
  CuSubMatrix<BaseFloat> out_vec(*out, 0, num_rows, 0, num_dims);
  
	  
  Memo *memo = new Memo;
  
  out_vec.CopyFromMat(post_std); // std
  out_vec.ApplyFloor(1e-20); // keep > 0
  
  if (!rand_per_frame_) {
	  (memo->rand_vec).Resize(num_dims);
	  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(&(memo->rand_vec));
	  if (fix_rand_val_) {
		(memo->rand_vec).SetZero();
	  }
	  out_vec.MulColsVec(memo->rand_vec); // std * rand
  } else {
	  (memo->rand_mat).Resize(num_rows, num_dims);
	  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(&(memo->rand_mat));
	  if (fix_rand_val_) {
		(memo->rand_mat).SetZero();
	  }
	  out_vec.MulElements(memo->rand_mat); // std * rand
  }
  
  out_vec.AddMat(1.0, post_mean); // std * rand + mean
  
  if (KL_scale_ != 0) {
  
	  CuSubMatrix<BaseFloat> prior_mean(in, 0, num_rows, 2*num_dims, num_dims);
	  CuSubMatrix<BaseFloat> prior_std(in, 0, num_rows, 3*num_dims, num_dims);
	  
	  (memo->inv_prior_std).Resize(num_rows, num_dims);
	  (memo->diff).Resize(num_rows, num_dims);
	  (memo->rate).Resize(num_rows, num_dims);
	  
	  (memo->inv_prior_std).CopyFromMat(prior_std);
	  (memo->inv_prior_std).ApplyFloor(1e-20); // keep > 0
	  (memo->inv_prior_std).InvertElements(); // 1/prior_std
	  
	  (memo->rate).CopyFromMat(post_std);
	  (memo->rate).ApplyFloor(1e-20); // keep > 0
	  (memo->rate).MulElements(memo->inv_prior_std); // post_std/prior_std
	  
	  memo->rate_power_2 = memo->rate;
	  (memo->rate_power_2).MulElements(memo->rate); // post_std^2/prior_std^2
	  
	  (memo->diff).CopyFromMat(post_mean);
	  (memo->diff).AddMat(-1.0, prior_mean); // post_mean - prior_mean
	  (memo->diff).MulElements(memo->inv_prior_std); // (post_mean - prior_mean)/prior_std
	  
	  memo->diff_power_2 = memo->diff;
	  (memo->diff_power_2).MulElements(memo->diff); // (post_mean - prior_mean)^2/prior_std^2
	  (memo->diff_power_2).AddMat(1.0, memo->rate_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2
	  (memo->diff_power_2).Add(-1.0); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
	  
	  if (input_frame_scale_) {
		  (memo->frame_scale).Resize(num_rows);
		  (memo->frame_scale).CopyColFromMat(in, 4*num_dims);
		  if (inv_frame_scale_) {
			  (memo->frame_scale).InvertElements();
		  }
	  }
	  
	  if (KL_output_) {
		  CuSubMatrix<BaseFloat> out_KL(*out, 0, num_rows, num_dims, 1);
		  out_KL.SetZero();
		  CuMatrix<BaseFloat> out_temp = memo->diff_power_2;
		  CuMatrix<BaseFloat> rate_power_2 = memo->rate_power_2;
		  rate_power_2.Add(1e-20); // keep > 0
		  rate_power_2.ApplyLog(); // log (post_std^2/prior_std^2)
		  out_temp.AddMat(-1.0, rate_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1
		  out_KL.AddMatBlocks(0.5, out_temp, kNoTrans); // 1/2 sum_j KL_{i,j}
		  
		  if (KL_scale_ != 1.0) {
			  out_KL.Scale(KL_scale_); // scale * 0.5*{(post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1} (/N)
		  }
		  
		  (memo->KL_out).Resize(num_rows, 1);
		  (memo->KL_out).CopyFromMat(out_KL);
		  
		  if (input_frame_scale_) {
			  out_KL.MulRowsVec(memo->frame_scale); // {(post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1} / N
		  }
	  }
	  
  }
  
  return static_cast<void*>(memo);
  
}

void BayesVecKLGaussianFComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {

  int32 num_rows = in_value.NumRows();
  int32 num_dims;
  if (input_frame_scale_) {
	  num_dims = (input_dim_-1)/4;
  } else {
	  num_dims = input_dim_/4;
  }
  
  CuSubMatrix<BaseFloat> post_mean_deriv(*in_deriv, 0, num_rows, 0, num_dims);
  CuSubMatrix<BaseFloat> post_std_deriv(*in_deriv, 0, num_rows, num_dims, num_dims);
  CuSubMatrix<BaseFloat> out_deriv_vec(out_deriv, 0, num_rows, 0, num_dims);

  post_mean_deriv.CopyFromMat(out_deriv_vec); // d/dmean = deriv

  Memo *memo_in = static_cast<Memo*>(memo);
  KALDI_ASSERT(memo_in != NULL && "memo not passed into backprop");
  
  post_std_deriv.CopyFromMat(out_deriv_vec); // deriv
  if (!rand_per_frame_) {
	  post_std_deriv.MulColsVec(memo_in->rand_vec); // deriv * rand
  } else {
	  post_std_deriv.MulElements(memo_in->rand_mat); // deriv * rand
  }

  if (KL_scale_ != 0) {
	  CuSubMatrix<BaseFloat> prior_mean_deriv(*in_deriv, 0, num_rows, 2*num_dims, num_dims);
	  CuSubMatrix<BaseFloat> prior_std_deriv(*in_deriv, 0, num_rows, 3*num_dims, num_dims);
	  CuSubMatrix<BaseFloat> post_std(in_value, 0, num_rows, num_dims, num_dims);
	  
	  CuMatrix<BaseFloat> post_mean_deriv_KL = memo_in->diff; // (post_mean - prior_mean)/prior_std
	  post_mean_deriv_KL.MulElements(memo_in->inv_prior_std); // (post_mean - prior_mean)/prior_std^2
	  
	  CuMatrix<BaseFloat> post_std_deriv_KL = memo_in->rate; // post_std/prior_std
	  post_std_deriv_KL.MulElements(memo_in->inv_prior_std); // post_std/prior_std^2
	  CuMatrix<BaseFloat> inv_post_std(post_std);
	  inv_post_std.ApplyFloor(1e-20); // keep > 0
	  inv_post_std.InvertElements(); // 1/post_std
	  post_std_deriv_KL.AddMat(-1.0, inv_post_std); // post_std/prior_std^2 - 1/post_std
	  
	  prior_mean_deriv.SetZero();
	  prior_std_deriv.SetZero();
	  prior_mean_deriv.AddMat(KL_scale_, post_mean_deriv_KL); // - (post_mean - prior_mean)/prior_std^2
	  prior_std_deriv.AddMat(KL_scale_, memo_in->diff_power_2); // -(post_mean - prior_mean)^2/prior_std^2 - post_std^2/prior_std^2 + 1
	  prior_std_deriv.MulElements(memo_in->inv_prior_std); // -(post_mean - prior_mean)^2/prior_std^3 - post_std^2/prior_std^3 + 1/prior_std
	  
	  if (input_frame_scale_) {
		  (in_deriv->ColRange(4*num_dims, 1)).SetZero();
		  if (KL_output_) {
		  (in_deriv->ColRange(4*num_dims, 1)).CopyFromMat(memo_in->KL_out);
		  }
		  post_mean_deriv_KL.MulRowsVec(memo_in->frame_scale); // deriv_post_mean_KL / N
		  post_std_deriv_KL.MulRowsVec(memo_in->frame_scale); // deriv_post_std_KL / N
		  prior_mean_deriv.MulRowsVec(memo_in->frame_scale); // deriv_prior_mean / N
		  prior_std_deriv.MulRowsVec(memo_in->frame_scale); // deriv_prior_std / N
	  }
	  post_mean_deriv.AddMat(-1.0*KL_scale_, post_mean_deriv_KL);
	  post_std_deriv.AddMat(-1.0*KL_scale_, post_std_deriv_KL);
  }
}

void BayesVecKLGaussianFComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<BayesVecKLGaussianFComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<KLScale>");
  ReadBasicType(is, binary, &KL_scale_);
  ExpectToken(is, binary, "<InputeFrameScale>");
  ReadBasicType(is, binary, &input_frame_scale_);
  ExpectToken(is, binary, "<InvFrameScale>");
  ReadBasicType(is, binary, &inv_frame_scale_);
  ExpectToken(is, binary, "<RandPerFrame>");
  ReadBasicType(is, binary, &rand_per_frame_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "<KLOutput>");
  ReadBasicType(is, binary, &KL_output_);
  ExpectToken(is, binary, "<FixRandVal>");
  ReadBasicType(is, binary, &fix_rand_val_);
  ExpectToken(is, binary, "</BayesVecKLGaussianFComponent>");
}

void BayesVecKLGaussianFComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<BayesVecKLGaussianFComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<KLScale>");
  WriteBasicType(os, binary, KL_scale_);
  WriteToken(os, binary, "<InputeFrameScale>");
  WriteBasicType(os, binary, input_frame_scale_);
  WriteToken(os, binary, "<InvFrameScale>");
  WriteBasicType(os, binary, inv_frame_scale_);
  WriteToken(os, binary, "<RandPerFrame>");
  WriteBasicType(os, binary, rand_per_frame_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "<KLOutput>");
  WriteBasicType(os, binary, KL_output_);
  WriteToken(os, binary, "<FixRandVal>");
  WriteBasicType(os, binary, fix_rand_val_);
  WriteToken(os, binary, "</BayesVecKLGaussianFComponent>");
}

std::string BayesVecKLGaussianFComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", KL-scale=" << KL_scale_
	<< ", input-frame-scale=" << (input_frame_scale_ ? "true" : "false")
	<< ", inv-frame-scale=" << (inv_frame_scale_ ? "true" : "false")
	<< ", rand-per-frame=" << (rand_per_frame_ ? "true" : "false")
	<< ", test-mode=" << (test_mode_ ? "true" : "false")
	<< ", KL-output=" << (KL_output_ ? "true" : "false")
	<< ", fix-rand-val=" << (fix_rand_val_ ? "true" : "false");
  return stream.str();
}

//////////

void BayesVecKLGaussianVarComponent::Init(int32 input_dim, int32 output_dim, BaseFloat KL_scale, bool input_frame_scale, bool inv_frame_scale, bool rand_per_frame, bool KL_output)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  KL_scale_ = KL_scale;
  input_frame_scale_ = input_frame_scale;
  inv_frame_scale_ = inv_frame_scale;
  rand_per_frame_ = rand_per_frame;
  KL_output_ = KL_output;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0);
  if (KL_output_) {
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == (output_dim_-1)*4+1);
	  } else {
		  KALDI_ASSERT(input_dim_ == (output_dim_-1)*4);
	  }
  } else {
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == output_dim_*4+1);
	  } else {
		  KALDI_ASSERT(input_dim_ == output_dim_*4);
	  }
  }
}

void BayesVecKLGaussianVarComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  BaseFloat KL_scale = 1.0;
  test_mode_ = false;
  bool input_frame_scale = false;
  bool inv_frame_scale = false;
  bool rand_per_frame = false;
  bool KL_output = true;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || output_dim <= 0)
    KALDI_ERR << "Invalid dim for layer of type "
              << Type();
  cfl->GetValue("KL-scale", &KL_scale);
  cfl->GetValue("input-frame-scale", &input_frame_scale);
  cfl->GetValue("inv-frame-scale", &inv_frame_scale);
  cfl->GetValue("rand-per-frame", &rand_per_frame);
  cfl->GetValue("test-mode", &test_mode_);
  cfl->GetValue("KL-output", &KL_output);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim, KL_scale, input_frame_scale, inv_frame_scale, rand_per_frame, KL_output);
}

void* BayesVecKLGaussianVarComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == input_dim_ && out->NumCols() == output_dim_);
  
  int32 num_rows = in.NumRows();
  int32 num_dims;
  if (input_frame_scale_) {
	  num_dims = (input_dim_-1)/4;
  } else {
	  num_dims = input_dim_/4;
  }
  if (KL_output_) {
	  output_dim_ == num_dims+1;
  } else {
	  output_dim_ == num_dims;
  }
  
  CuSubMatrix<BaseFloat> post_mean(in, 0, num_rows, 0, num_dims);
  CuSubMatrix<BaseFloat> post_var(in, 0, num_rows, num_dims, num_dims);
  CuSubMatrix<BaseFloat> out_vec(*out, 0, num_rows, 0, num_dims);
  
  if (!test_mode_) {
	  
	  Memo *memo = new Memo;
	  
	  out_vec.CopyFromMat(post_var); // var
	  out_vec.ApplyFloor(1e-20); // keep > 0
	  out_vec.ApplyPow(0.5); // std
	  
	  if (!rand_per_frame_) {
		  (memo->rand_vec).Resize(num_dims);
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(&(memo->rand_vec));
		  out_vec.MulColsVec(memo->rand_vec); // std * rand
	  } else {
		  (memo->rand_mat).Resize(num_rows, num_dims);
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(&(memo->rand_mat));
		  out_vec.MulElements(memo->rand_mat); // std * rand
	  }
	  
	  out_vec.AddMat(1.0, post_mean); // std * rand + mean
		  
	  if (KL_scale_ != 0) {
	  
		  CuSubMatrix<BaseFloat> prior_mean(in, 0, num_rows, 2*num_dims, num_dims);
		  CuSubMatrix<BaseFloat> prior_var(in, 0, num_rows, 3*num_dims, num_dims);
		  
		  (memo->inv_prior_var).Resize(num_rows, num_dims);
		  (memo->diff).Resize(num_rows, num_dims);
		  (memo->rate_power_2).Resize(num_rows, num_dims);
		  
		  (memo->inv_prior_var).CopyFromMat(prior_var);
		  (memo->inv_prior_var).ApplyFloor(1e-20); // keep > 0
		  (memo->inv_prior_var).InvertElements(); // 1/prior_var
		  
		  (memo->rate_power_2).CopyFromMat(post_var);
		  (memo->rate_power_2).ApplyFloor(1e-20); // keep > 0
		  (memo->rate_power_2).MulElements(memo->inv_prior_var); // post_var/prior_var
		  
		  (memo->diff).CopyFromMat(post_mean);
		  (memo->diff).AddMat(-1.0, prior_mean); // post_mean - prior_mean
		  
		  memo->diff_power_2 = memo->diff;
		  (memo->diff_power_2).MulElements(memo->diff); // (post_mean - prior_mean)^2
		  (memo->diff_power_2).MulElements(memo->inv_prior_var); // (post_mean - prior_mean)^2/prior_var
		  (memo->diff_power_2).AddMat(1.0, memo->rate_power_2); // (post_mean - prior_mean)^2/prior_var + post_var/prior_var
		  (memo->diff_power_2).Add(-1.0); // (post_mean - prior_mean)^2/prior_var + post_var/prior_var - 1
		  
		  if (input_frame_scale_) {
			  (memo->frame_scale).Resize(num_rows);
			  (memo->frame_scale).CopyColFromMat(in, 4*num_dims);
			  if (inv_frame_scale_) {
				  (memo->frame_scale).InvertElements();
			  }
		  }
		  
		  if (KL_output_) {
			  CuSubMatrix<BaseFloat> out_KL(*out, 0, num_rows, num_dims, 1);
			  out_KL.SetZero();
			  CuMatrix<BaseFloat> out_temp = memo->diff_power_2;
			  CuMatrix<BaseFloat> rate_power_2 = memo->rate_power_2;
			  rate_power_2.Add(1e-20); // keep > 0
			  rate_power_2.ApplyLog(); // log (post_var/prior_var)
			  out_temp.AddMat(-1.0, rate_power_2); // (post_mean - prior_mean)^2/prior_var + post_var/prior_var - log (post_var/prior_var) - 1
			  out_KL.AddMatBlocks(0.5, out_temp, kNoTrans); // 1/2 sum_j KL_{i,j}
			  
			  if (input_frame_scale_) {
				  out_KL.MulRowsVec(memo->frame_scale); // 0.5*{(post_mean - prior_mean)^2/prior_var + post_var/prior_var - log (post_var/prior_var) - 1} / N
			  }
			  
			  if (KL_scale_ != 1.0) {
				  out_KL.Scale(KL_scale_); // scale * 0.5*{(post_mean - prior_mean)^2/prior_var + post_var/prior_var - log (post_var/prior_var) - 1} (/N)
			  }
		  }
		  
	  }
	  
	  return static_cast<void*>(memo);
  } else {
	  out_vec.CopyFromMat(post_mean); // std * 0 + mean
	  return NULL;
  }
  
}

void BayesVecKLGaussianVarComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {

  int32 num_rows = in_value.NumRows();
  int32 num_dims;
  if (input_frame_scale_) {
	  num_dims = (input_dim_-1)/4;
  } else {
	  num_dims = input_dim_/4;
  }
  
  CuSubMatrix<BaseFloat> post_mean_deriv(*in_deriv, 0, num_rows, 0, num_dims);
  CuSubMatrix<BaseFloat> post_var_deriv(*in_deriv, 0, num_rows, num_dims, num_dims);
  CuSubMatrix<BaseFloat> out_deriv_vec(out_deriv, 0, num_rows, 0, num_dims);

  post_mean_deriv.CopyFromMat(out_deriv_vec); // d/dmean = deriv
  if (!test_mode_) {
	  
	  Memo *memo_in = static_cast<Memo*>(memo);
	  KALDI_ASSERT(memo_in != NULL && "memo not passed into backprop");
	  
	  post_var_deriv.CopyFromMat(out_deriv_vec); // deriv
	  post_var_deriv.MulElements(post_var_deriv); // deriv^2
	  post_var_deriv.Scale(0.5); // d/dvar = 0.5*deriv^2

	  if (KL_scale_ != 0) {
		  CuSubMatrix<BaseFloat> prior_mean_deriv(*in_deriv, 0, num_rows, 2*num_dims, num_dims);
		  CuSubMatrix<BaseFloat> prior_var_deriv(*in_deriv, 0, num_rows, 3*num_dims, num_dims);
		  CuSubMatrix<BaseFloat> post_var(in_value, 0, num_rows, num_dims, num_dims);
		  
		  CuMatrix<BaseFloat> post_mean_deriv_KL = memo_in->diff; // (post_mean - prior_mean)
		  post_mean_deriv_KL.MulElements(memo_in->inv_prior_var); // (post_mean - prior_mean)/prior_var
		  
		  CuMatrix<BaseFloat> post_var_deriv_KL = memo_in->inv_prior_var; // 1/prior_var
		  CuMatrix<BaseFloat> inv_post_var(post_var);
		  inv_post_var.ApplyFloor(1e-20); // keep > 0
		  inv_post_var.InvertElements(); // 1/post_var
		  post_var_deriv_KL.AddMat(-1.0, inv_post_var); // 1/prior_var - 1/post_var
		  post_var_deriv_KL.Scale(0.5); // 0.5(1/prior_var - 1/post_var)
		  
		  prior_mean_deriv.SetZero();
		  prior_var_deriv.SetZero();
		  prior_mean_deriv.AddMat(KL_scale_, post_mean_deriv_KL); // - (post_mean - prior_mean)/prior_var
		  prior_var_deriv.AddMat(KL_scale_, memo_in->diff_power_2); // -(post_mean - prior_mean)^2/prior_var - post_var/prior_var + 1
		  prior_var_deriv.MulElements(memo_in->inv_prior_var); // -(post_mean - prior_mean)^2/prior_var^2 - post_var/prior_var^2 + 1/prior_var
		  
		  if (input_frame_scale_) {
			  (in_deriv->ColRange(4*num_dims, 1)).SetZero();
			  post_mean_deriv_KL.MulRowsVec(memo_in->frame_scale); // deriv_post_mean_KL / N
			  post_var_deriv_KL.MulRowsVec(memo_in->frame_scale); // deriv_post_var_KL / N
			  prior_mean_deriv.MulRowsVec(memo_in->frame_scale); // deriv_prior_mean / N
			  prior_var_deriv.MulRowsVec(memo_in->frame_scale); // deriv_prior_var / N
		  }
		  post_mean_deriv.AddMat(-1.0*KL_scale_, post_mean_deriv_KL);
		  post_var_deriv.AddMat(-1.0*KL_scale_, post_var_deriv_KL);
	  }
  }
}

void BayesVecKLGaussianVarComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<BayesVecKLGaussianVarComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<KLScale>");
  ReadBasicType(is, binary, &KL_scale_);
  ExpectToken(is, binary, "<InputeFrameScale>");
  ReadBasicType(is, binary, &input_frame_scale_);
  ExpectToken(is, binary, "<InvFrameScale>");
  ReadBasicType(is, binary, &inv_frame_scale_);
  ExpectToken(is, binary, "<RandPerFrame>");
  ReadBasicType(is, binary, &rand_per_frame_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "<KLOutput>");
  ReadBasicType(is, binary, &KL_output_);
  ExpectToken(is, binary, "</BayesVecKLGaussianVarComponent>");
}

void BayesVecKLGaussianVarComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<BayesVecKLGaussianVarComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<KLScale>");
  WriteBasicType(os, binary, KL_scale_);
  WriteToken(os, binary, "<InputeFrameScale>");
  WriteBasicType(os, binary, input_frame_scale_);
  WriteToken(os, binary, "<InvFrameScale>");
  WriteBasicType(os, binary, inv_frame_scale_);
  WriteToken(os, binary, "<RandPerFrame>");
  WriteBasicType(os, binary, rand_per_frame_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "<KLOutput>");
  WriteBasicType(os, binary, KL_output_);
  WriteToken(os, binary, "</BayesVecKLGaussianVarComponent>");
}

std::string BayesVecKLGaussianVarComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", KL-scale=" << KL_scale_
	<< ", input-frame-scale=" << (input_frame_scale_ ? "true" : "false")
	<< ", inv-frame-scale=" << (inv_frame_scale_ ? "true" : "false")
	<< ", rand-per-frame=" << (rand_per_frame_ ? "true" : "false")
	<< ", test-mode=" << (test_mode_ ? "true" : "false")
	<< ", KL-output=" << (KL_output_ ? "true" : "false");
  return stream.str();
}


/////////////////////////////////////////


void InputVectorLinearComponent::Init(const std::vector<int32> &sizes) {
  KALDI_ASSERT(!sizes.empty());
  std::vector<Int32Pair> cpu_vec(sizes.size());
  std::vector<int32> reverse_cpu_vec;
  int32 cur_index = 0;
  for (size_t i = 0; i < sizes.size(); i++) {
    KALDI_ASSERT(sizes[i] > 0);
    cpu_vec[i].first = cur_index;
    cpu_vec[i].second = cur_index + sizes[i];
    cur_index += sizes[i];
    for (int32 j = cpu_vec[i].first; j < cpu_vec[i].second; j++)
      reverse_cpu_vec.push_back(i);
  }
  this->indexes_ = cpu_vec;
  this->reverse_indexes_ = reverse_cpu_vec;
  //this->input_dim_ = cur_index;
  this->output_dim_ = sizes.size();
}

void InputVectorLinearComponent::Init(int32 input_dim, int32 output_dim) {
  const int32 num_groups = output_dim;
  KALDI_ASSERT(input_dim % (num_groups+1) == 0);
  const int32 group_size = input_dim / (num_groups+1);

  std::vector<Int32Pair> cpu_vec(num_groups);
  std::vector<int32> reverse_cpu_vec;
  int32 cur_index = 0;
  for (size_t i = 0; i < num_groups; i++) {
    cpu_vec[i].first = cur_index;
    cpu_vec[i].second = cur_index + group_size;
    cur_index += group_size;
    for (int32 j = cpu_vec[i].first; j < cpu_vec[i].second; j++)
      reverse_cpu_vec.push_back(i);
  }
  this->indexes_ = cpu_vec;
  this->reverse_indexes_ = reverse_cpu_vec;
  this->input_dim_ = input_dim;
  this->output_dim_ = num_groups;
}

void InputVectorLinearComponent::InitFromConfig(ConfigLine *cfl) {
  std::vector<int32> sizes;
  bool has_sizes = cfl->GetValue("sizes", &sizes);
  if (has_sizes) {
    if (cfl->HasUnusedValues() || sizes.empty())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    this->Init(sizes);
  } else { // each group has the same size
    int32 input_dim = -1, output_dim = -1;
    if (!cfl->GetValue("input-dim", &input_dim) ||
        !cfl->GetValue("output-dim", &output_dim) || cfl->HasUnusedValues()) {
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    }
    Init(input_dim, output_dim);
  }
}

Component* InputVectorLinearComponent::Copy() const {
  InputVectorLinearComponent *ans = new InputVectorLinearComponent();
  ans->indexes_ = indexes_;
  ans->reverse_indexes_ = reverse_indexes_;
  ans->input_dim_ = input_dim_;
  ans->output_dim_ = output_dim_;
  return ans;
}

void InputVectorLinearComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<InputVectorLinearComponent>", "<Sizes>");
  std::vector<int32> sizes;
  ReadIntegerVector(is, binary, &sizes);
  ExpectToken(is, binary, "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);

  std::string token;
  ReadToken(is, binary, &token);
  if (!(token == "<InputVectorLinearComponent>" ||
        token == "</InputVectorLinearComponent>")) {
    KALDI_ERR << "Expected </InputVectorLinearComponent>, got " << token;
  }
  this->Init(sizes);
}

void InputVectorLinearComponent::GetSizes(std::vector<int32> *sizes) const {
  std::vector<Int32Pair> indexes;
  indexes_.CopyToVec(&indexes);
  sizes->resize(indexes.size());
  for (size_t i = 0; i < indexes.size(); i++) {
    (*sizes)[i] = indexes[i].second - indexes[i].first;
    if (i == 0) { KALDI_ASSERT(indexes[i].first == 0); }
    else { KALDI_ASSERT(indexes[i].first == indexes[i-1].second); }
    KALDI_ASSERT(indexes[i].second > indexes[i].first);
    (*sizes)[i] = indexes[i].second - indexes[i].first;
  }
}

void InputVectorLinearComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<InputVectorLinearComponent>");
  WriteToken(os, binary, "<Sizes>");
  std::vector<int32> sizes;
  this->GetSizes(&sizes);
  WriteIntegerVector(os, binary, sizes);
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "</InputVectorLinearComponent>");
}

void* InputVectorLinearComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                  const CuMatrixBase<BaseFloat> &in,
                                  CuMatrixBase<BaseFloat> *out) const {
  int32 variable_size = input_dim_ / (output_dim_+1);
  int32 linear_size = input_dim_ - variable_size;
  int32 num_rows = in.NumRows();
  
  CuSubMatrix<BaseFloat> variable(in, 0, num_rows, linear_size, variable_size);
  CuSubMatrix<BaseFloat> Linear(in, 0, num_rows, 0, linear_size);
  
  CuMatrix<BaseFloat> temp_variable(num_rows, linear_size);
  
  temp_variable.AddMatBlocks(1.0, variable, kNoTrans);
  
  temp_variable.MulElements(Linear);
  out->SumColumnRanges(temp_variable, indexes_);
  
  return NULL;
}

void InputVectorLinearComponent::Backprop(const std::string &debug_info,
                                 const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in_value,
                                 const CuMatrixBase<BaseFloat> &out_value,
                                 const CuMatrixBase<BaseFloat> &out_deriv,
                                 void *memo,
                                 Component *to_update_in,
                                 CuMatrixBase<BaseFloat> *in_deriv) const {
  if (!in_deriv)  return;
  
  int32 variable_size = input_dim_ / (output_dim_+1);
  int32 linear_size = input_dim_ - variable_size;
  int32 num_rows = in_value.NumRows();
  
  CuSubMatrix<BaseFloat> variable(in_value, 0, num_rows, linear_size, variable_size);
  CuSubMatrix<BaseFloat> Linear(in_value, 0, num_rows, 0, linear_size);
  CuSubMatrix<BaseFloat> variable_deriv(*in_deriv, 0, num_rows, linear_size, variable_size);
  CuSubMatrix<BaseFloat> Linear_deriv(*in_deriv, 0, num_rows, 0, linear_size);
  
  CuMatrix<BaseFloat> temp_out(num_rows, linear_size);
  temp_out.CopyCols(out_deriv, reverse_indexes_);
  
  Linear_deriv.CopyFromMat(temp_out);
  
  temp_out.MulElements(Linear);
  variable_deriv.AddMatBlocks(1.0, temp_out, kNoTrans);
  
  CuMatrix<BaseFloat> temp_variable(num_rows, linear_size);
  temp_variable.AddMatBlocks(1.0, variable, kNoTrans);
  Linear_deriv.MulElements(temp_variable);
  
}

////////////////////////////////////////////////////////////////


void BayesAffineComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    // If scale == 0.0 we call SetZero() which will get rid of NaN's and inf's.
    linear_params_mean_.SetZero();
	linear_params_std_.SetZero();
	linear_prior_mean_.SetZero();
	linear_prior_std_.SetZero();
    bias_params_.SetZero();
  } else {
	linear_params_mean_.Scale(scale);
	linear_params_std_.Scale(scale);
	linear_prior_mean_.Scale(scale);
	linear_prior_std_.Scale(scale);
    bias_params_.Scale(scale);
  }
}

void BayesAffineComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  bias_params_.Resize(output_dim);
  linear_params_mean_.Resize(output_dim, input_dim);
  linear_prior_mean_.Resize(output_dim, input_dim);
  //rand_mat_.Resize(output_dim, input_dim);
  //linear_params_.Resize(output_dim, input_dim);
  
  int32 input_dim_re = input_dim;
  int32 output_dim_re = output_dim;
  if (share_std_input_) {
	  input_dim_re = 1;
  }
  if (share_std_output_) {
	  output_dim_re = 1;
  }
  linear_params_std_.Resize(output_dim_re, input_dim_re);
  linear_prior_std_.Resize(output_dim_re, input_dim_re);
}

void BayesAffineComponent::Add(BaseFloat alpha, const Component &other_in) {
  const BayesAffineComponent *other =
      dynamic_cast<const BayesAffineComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_mean_.AddMat(alpha, other->linear_params_mean_);
  linear_prior_mean_.AddMat(alpha, other->linear_prior_mean_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

BayesAffineComponent::BayesAffineComponent(const BayesAffineComponent &component):
    UpdatableComponent(component),
    linear_params_mean_(component.linear_params_mean_),
	linear_params_std_(component.linear_params_std_),
	linear_prior_mean_(component.linear_prior_mean_),
	linear_prior_std_(component.linear_prior_std_),
    bias_params_(component.bias_params_),
    test_mode_(component.test_mode_),
    share_std_input_(component.share_std_input_),
    share_std_output_(component.share_std_output_),
    share_std_input_sampling_(component.share_std_input_sampling_),
    share_std_output_sampling_(component.share_std_output_sampling_),
    use_exp_std_(component.use_exp_std_),
	update_prior_(component.update_prior_),
	KL_scale_(component.KL_scale_),
    orthonormal_constraint_(component.orthonormal_constraint_) { 
		//rand_mat_.Resize(linear_params_mean_.NumRows(), linear_params_mean_.NumCols());
		//linear_params_.Resize(linear_params_mean_.NumRows(), linear_params_mean_.NumCols());
		}

BayesAffineComponent::BayesAffineComponent(const CuMatrixBase<BaseFloat> &linear_params_mean,
					const CuMatrixBase<BaseFloat> &linear_params_std,
					const CuMatrixBase<BaseFloat> &linear_prior_mean,
					const CuMatrixBase<BaseFloat> &linear_prior_std,
                  const CuVectorBase<BaseFloat> &bias_params,
                                 BaseFloat learning_rate):
    linear_params_mean_(linear_params_mean),
	linear_params_std_(linear_params_std),
	linear_prior_mean_(linear_prior_mean),
	linear_prior_std_(linear_prior_std),
    bias_params_(bias_params),
	test_mode_(false),
	share_std_input_(false),
	share_std_output_(false),
	share_std_input_sampling_(false),
	share_std_output_sampling_(false),
	use_exp_std_(false),
	update_prior_(false),
	KL_scale_(0.00001),
    orthonormal_constraint_(0.0) {
  //rand_mat_.Resize(linear_params_mean_.NumRows(), linear_params_mean_.NumCols());
  //linear_params_.Resize(linear_params_mean_.NumRows(), linear_params_mean_.NumCols());
  SetUnderlyingLearningRate(learning_rate);
  KALDI_ASSERT(linear_params_mean_.NumRows() == bias_params.Dim()&&
               bias_params.Dim() != 0);
}

void BayesAffineComponent::SetParams(const CuVectorBase<BaseFloat> &bias,
                         const CuMatrixBase<BaseFloat> &linear_mean,
						 const CuMatrixBase<BaseFloat> &linear_std,
						 const CuMatrixBase<BaseFloat> &linear_prior_mean,
						 const CuMatrixBase<BaseFloat> &linear_prior_std) {
  bias_params_ = bias;
  linear_params_mean_ = linear_mean;
  linear_params_std_ = linear_std;
  linear_prior_mean_ = linear_prior_mean;
  linear_prior_std_ = linear_prior_std;
  KALDI_ASSERT(bias_params_.Dim() == linear_params_mean_.NumRows());
  //rand_mat_.Resize(linear_params_mean_.NumRows(), linear_params_mean_.NumCols());
  //linear_params_.Resize(linear_params_mean_.NumRows(), linear_params_mean_.NumCols());
}

void BayesAffineComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_mean_);
  temp_linear_params.SetRandn();
  linear_params_mean_.AddMat(stddev, temp_linear_params);

  CuVector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

std::string BayesAffineComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info();
  if (orthonormal_constraint_ != 0.0)
    stream << ", orthonormal-constraint=" << orthonormal_constraint_;
  stream << ", test-mode=" << test_mode_;
  stream << ", share-std-input=" << share_std_input_;
  stream << ", share-std-output=" << share_std_output_;
  stream << ", share-std-input-sampling=" << share_std_input_sampling_;
  stream << ", share-std-output-sampling=" << share_std_output_sampling_;
  stream << ", use-exp-std=" << use_exp_std_;
  stream << ", update-prior=" << update_prior_;
  stream << ", KL-scale=" << KL_scale_;
  PrintParameterStats(stream, "linear-params-mean", linear_params_mean_,
                      false, // include_mean
                      true, // include_row_norms
                      true, // include_column_norms
                      GetVerboseLevel() >= 2); // include_singular_values
  PrintParameterStats(stream, "linear-params-std", linear_params_std_,
                      false, // include_mean
                      true, // include_row_norms
                      true, // include_column_norms
                      GetVerboseLevel() >= 2); // include_singular_values
  PrintParameterStats(stream, "linear-prior-mean", linear_prior_mean_,
                      false, // include_mean
                      true, // include_row_norms
                      true, // include_column_norms
                      GetVerboseLevel() >= 2); // include_singular_values
  PrintParameterStats(stream, "linear-prior-std", linear_prior_std_,
                      false, // include_mean
                      true, // include_row_norms
                      true, // include_column_norms
                      GetVerboseLevel() >= 2); // include_singular_values
  PrintParameterStats(stream, "bias", bias_params_, true);
  return stream.str();
}

Component* BayesAffineComponent::Copy() const {
  BayesAffineComponent *ans = new BayesAffineComponent(*this);
  return ans;
}

BaseFloat BayesAffineComponent::DotProduct(const UpdatableComponent &other_in) const {
  const BayesAffineComponent *other =
      dynamic_cast<const BayesAffineComponent*>(&other_in);
  return TraceMatMat(linear_params_mean_, other->linear_params_mean_, kTrans)
      + TraceMatMat(linear_prior_mean_, other->linear_prior_mean_, kTrans)
      + VecVec(bias_params_, other->bias_params_);
}

void BayesAffineComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_mean_stddev, BaseFloat param_std_stddev, BaseFloat bias_stddev, BaseFloat prior_mean, BaseFloat prior_std) {
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_mean_stddev >= 0.0 && param_std_stddev >= 0.0);
  linear_params_mean_.Resize(output_dim, input_dim);
  linear_prior_mean_.Resize(output_dim, input_dim);
  int32 input_dim_re = input_dim;
  int32 output_dim_re = output_dim;
  if (share_std_input_) {
	  input_dim_re = 1;
  }
  if (share_std_output_) {
	  output_dim_re = 1;
  }
  linear_params_std_.Resize(output_dim_re, input_dim_re);
  linear_prior_std_.Resize(output_dim_re, input_dim_re);
  bias_params_.Resize(output_dim);
  
  linear_params_mean_.SetRandn(); // sets to random normally distributed noise.
  linear_params_mean_.Scale(param_mean_stddev);
  
  linear_params_std_.SetRandn(); // sets to random normally distributed noise.
  linear_params_std_.Scale(param_std_stddev);
  if (!use_exp_std_) {
	  linear_params_std_.Add(1.0); // init around 1
  }
  
  linear_prior_mean_.Set(prior_mean);
  linear_prior_std_.Set(prior_std);
  
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  
  //rand_mat_.Resize(linear_params_mean_.NumRows(), linear_params_mean_.NumCols());
  //linear_params_.Resize(linear_params_mean_.NumRows(), linear_params_mean_.NumCols());
}

void BayesAffineComponent::Init(std::string matrix_filename) {
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 output_dim = mat.NumRows();
  int32 input_dim = (mat.NumCols() - 1)/4;
  linear_params_mean_.Resize(output_dim, input_dim);
  linear_prior_mean_.Resize(output_dim, input_dim);
  int32 input_dim_re = input_dim;
  int32 output_dim_re = output_dim;
  if (share_std_input_) {
	  input_dim_re = 1;
  }
  if (share_std_output_) {
	  output_dim_re = 1;
  }
  linear_params_std_.Resize(output_dim_re, input_dim_re);
  linear_prior_std_.Resize(output_dim_re, input_dim_re);
  bias_params_.Resize(output_dim);
  
  linear_params_mean_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  linear_params_std_.CopyFromMat(mat.Range(0, output_dim_re, input_dim, input_dim_re));
  linear_prior_mean_.CopyFromMat(mat.Range(0, output_dim, 2*input_dim, input_dim));
  linear_prior_std_.CopyFromMat(mat.Range(0, output_dim_re, 3*input_dim, input_dim_re));
  bias_params_.CopyColFromMat(mat, 4*input_dim);
  
  //rand_mat_.Resize(linear_params_mean_.NumRows(), linear_params_mean_.NumCols());
  //linear_params_.Resize(linear_params_mean_.NumRows(), linear_params_mean_.NumCols());
}

void BayesAffineComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  test_mode_ = false;
  share_std_input_ = false;
  share_std_output_ = false;
  share_std_input_sampling_ = false;
  share_std_output_sampling_ = false;
  use_exp_std_ = false;
  update_prior_ = false;
  KL_scale_ = 0.00001;
  InitLearningRatesFromConfig(cfl);
  cfl->GetValue("test-mode", &test_mode_);
  cfl->GetValue("share-std-input", &share_std_input_);
  cfl->GetValue("share-std-output", &share_std_output_);
  cfl->GetValue("share-std-input-sampling", &share_std_input_sampling_);
  cfl->GetValue("share-std-output-sampling", &share_std_output_sampling_);
  cfl->GetValue("use-exp-std", &use_exp_std_);
  cfl->GetValue("use-exp-std", &update_prior_);
  cfl->GetValue("KL-scale", &KL_scale_);
  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    BaseFloat param_mean_stddev = 1.0 / std::sqrt(input_dim),
		param_std_stddev = 1.0 / std::sqrt(input_dim),
		prior_mean = 0, prior_std,
        bias_stddev = 1.0;
	if (!use_exp_std_) {
		prior_std = 1.0;
	} else {
		prior_std = 0.0;
	}
    cfl->GetValue("param-stddev", &param_mean_stddev);
	cfl->GetValue("param-std-stddev", &param_std_stddev);
	cfl->GetValue("prior-mean", &prior_mean);
	cfl->GetValue("prior-std", &prior_std);
    cfl->GetValue("bias-stddev", &bias_stddev);
    Init(input_dim, output_dim,
         param_mean_stddev, param_std_stddev, bias_stddev, prior_mean, prior_std);
  }
  cfl->GetValue("orthonormal-constraint", &orthonormal_constraint_);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}




void* BayesAffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {

  // No need for asserts as they'll happen within the matrix operations.
  out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.
  
  int32 dim_input = linear_params_mean_.NumCols();
  int32 dim_output = linear_params_mean_.NumRows();
  //CuMatrix<BaseFloat> rand_mat(dim_output, dim_input);
  
  int32 dim_input_rand = dim_input;
  int32 dim_output_rand = dim_output;
  if (share_std_input_sampling_) {
	  dim_input_rand = 1;
  }
  if (share_std_output_sampling_) {
	  dim_output_rand = 1;
  }
  CuMatrix<BaseFloat> rand_mat_temp(dim_output_rand, dim_input_rand);
  CuMatrix<BaseFloat> rand_mat_(dim_output, dim_input);
  CuRand<BaseFloat> rand;
  if (test_mode_) {
	rand_mat_.SetZero();
  } else {
	rand.RandGaussian(&rand_mat_temp);
	rand_mat_.AddMatBlocks(1.0, rand_mat_temp, kNoTrans);
  }
  
  CuMatrix<BaseFloat> linear_params_(dim_output, dim_input);
  linear_params_.AddMatBlocks(1.0, linear_params_std_, kNoTrans); // std
  if (use_exp_std_) {
	  linear_params_.ApplyCeiling(46.05); // avoid inf
	  linear_params_.ApplyExp(); // std = exp()
  }
  
  linear_params_.MulElements(rand_mat_); // std*rand
  linear_params_.AddMat(1.0, linear_params_mean_); // std*rand + mean
  
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);
  
  CuMatrix<BaseFloat> *rand_memo = new CuMatrix<BaseFloat>(dim_output, 2*dim_input);
  (rand_memo->ColRange(0,dim_input)).CopyFromMat(rand_mat_);
  (rand_memo->ColRange(dim_input,dim_input)).CopyFromMat(linear_params_);
  return rand_memo;
}

void BayesAffineComponent::UpdateSimple(const CuMatrixBase<BaseFloat> &in_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv,
								   const CuMatrixBase<BaseFloat> &rand_mat) {
  bias_params_.AddRowSumMat(learning_rate_, out_deriv, 1.0);
  
  int32 dim_input = linear_params_mean_.NumCols();
  int32 dim_output = linear_params_mean_.NumRows();
  CuMatrix<BaseFloat> linear_deriv(dim_output, dim_input);
  linear_deriv.AddMatMat(1.0, out_deriv, kTrans,
                           in_value, kNoTrans, 0.0);
  linear_params_mean_.AddMat(learning_rate_, linear_deriv);
  linear_deriv.MulElements(rand_mat);
  if (use_exp_std_) {
	  CuMatrix<BaseFloat> linear_params_std_exp(linear_params_std_);
	  linear_params_std_exp.ApplyCeiling(46.05);
	  linear_params_std_exp.ApplyExp();
	  linear_deriv.MulElements(linear_params_std_exp);
  }
  linear_params_std_.AddMatBlocks(learning_rate_, linear_deriv, kNoTrans);
  
  if (KL_scale_ != 0) {
	  int32 num_frame = in_value.NumRows();
	  
	  CuMatrix<BaseFloat> inv_prior_std(dim_output, dim_input);
	  CuMatrix<BaseFloat> diff(dim_output, dim_input);
	  CuMatrix<BaseFloat> diff_power_2(dim_output, dim_input);
	  CuMatrix<BaseFloat> rate(dim_output, dim_input);
	  
	  CuMatrix<BaseFloat> post_mean_deriv(dim_output, dim_input);
	  CuMatrix<BaseFloat> post_std_deriv(dim_output, dim_input);
	  CuMatrix<BaseFloat> prior_mean_deriv(dim_output, dim_input);
	  CuMatrix<BaseFloat> prior_std_deriv(dim_output, dim_input);
	  
	  CuMatrix<BaseFloat> linear_params_std_resize(dim_output, dim_input);
	  CuMatrix<BaseFloat> linear_prior_std_resize(dim_output, dim_input);
	  linear_params_std_resize.AddMatBlocks(1.0, linear_params_std_, kNoTrans);
	  linear_prior_std_resize.AddMatBlocks(1.0, linear_prior_std_, kNoTrans);
	  
	  if (use_exp_std_) {
		  linear_params_std_resize.ApplyCeiling(46.05); // avoid inf
		  linear_params_std_resize.ApplyExp(); // std = exp()
		  linear_prior_std_resize.ApplyCeiling(46.05); // avoid inf
		  linear_prior_std_resize.ApplyExp(); // std = exp()
	  }
  
	  inv_prior_std.CopyFromMat(linear_prior_std_resize);
	  inv_prior_std.InvertElements(); // 1/prior_std
  
	  rate.CopyFromMat(linear_params_std_resize);
	  rate.MulElements(inv_prior_std); // post_std/prior_std
	  CuMatrix<BaseFloat> rate_power_2(rate);
	  rate_power_2.MulElements(rate); // post_std^2/prior_std^2
  
	  diff.CopyFromMat(linear_params_mean_);
	  diff.AddMat(-1.0, linear_prior_mean_); // post_mean - prior_mean
	  diff.MulElements(inv_prior_std); // (post_mean - prior_mean)/prior_std
  
	  diff_power_2.CopyFromMat(diff);
	  diff_power_2.MulElements(diff); // (post_mean - prior_mean)^2/prior_std^2
	  diff_power_2.AddMat(1.0, rate_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2
	  diff_power_2.Add(-1.0); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
	  
	  post_mean_deriv.CopyFromMat(diff); // (post_mean - prior_mean)/prior_std
	  post_mean_deriv.MulElements(inv_prior_std); // // (post_mean - prior_mean)/prior_std^2
	  
	  post_std_deriv.CopyFromMat(rate); // post_std/prior_std
	  post_std_deriv.MulElements(inv_prior_std); // post_std/prior_std^2
	  CuMatrix<BaseFloat> inv_post_std(dim_output, dim_input);
	  inv_post_std.CopyFromMat(linear_params_std_resize);
	  inv_post_std.InvertElements(); // 1/post_std
	  post_std_deriv.AddMat(-1.0, inv_post_std); // post_std/prior_std^2 - 1/post_std
	  if (use_exp_std_) {
		  post_std_deriv.MulElements(linear_params_std_resize);
	  }
	  
	  BaseFloat learning_rate_KL_ = -1.0*learning_rate_*num_frame*KL_scale_;
	  linear_params_mean_.AddMat(learning_rate_KL_, post_mean_deriv);
	  linear_params_std_.AddMatBlocks(learning_rate_KL_, post_std_deriv, kNoTrans);
	  
	  if (update_prior_) {
		  prior_mean_deriv.CopyFromMat(post_mean_deriv);
		  prior_mean_deriv.Scale(-1.0); // - (post_mean - prior_mean)/prior_std
	  
		  prior_std_deriv.CopyFromMat(diff_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
		  if (!use_exp_std_) {
			prior_std_deriv.MulElements(inv_prior_std); // (post_mean - prior_mean)^2/prior_std^3 + post_std^2/prior_std^3 - 1/prior_std
		  }
		  prior_std_deriv.Scale(-1.0); // -(post_mean - prior_mean)^2/prior_std^3 - post_std^2/prior_std^3 + 1/prior_std
		  
		  linear_prior_mean_.AddMat(learning_rate_KL_, prior_mean_deriv);
		  linear_prior_std_.AddMatBlocks(learning_rate_KL_, prior_std_deriv, kNoTrans);
	  }
  }
}

void BayesAffineComponent::Backprop(const std::string &debug_info,
                               const ComponentPrecomputedIndexes *indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               void *memo,
                               Component *to_update_in,
                               CuMatrixBase<BaseFloat> *in_deriv) const {
  BayesAffineComponent *to_update = dynamic_cast<BayesAffineComponent*>(to_update_in);
  CuMatrix<BaseFloat> *rand_memo = reinterpret_cast<CuMatrix<BaseFloat>*>(memo);
  int32 dim_input = linear_params_mean_.NumCols();
  int32 dim_output = linear_params_mean_.NumRows();
  CuMatrix<BaseFloat> rand_mat_(dim_output, dim_input);
  CuMatrix<BaseFloat> linear_params_(dim_output, dim_input);
  rand_mat_.CopyFromMat(rand_memo->ColRange(0,dim_input));
  linear_params_.CopyFromMat(rand_memo->ColRange(dim_input,dim_input));
  // Propagate the derivative back to the input.
  // add with coefficient 1.0 since property kBackpropAdds is true.
  // If we wanted to add with coefficient 0.0 we'd need to zero the
  // in_deriv, in case of infinities.
  if (in_deriv)
    in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, linear_params_, kNoTrans,
                        1.0);

  if (to_update != NULL) {
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    if (to_update->is_gradient_)
      to_update->UpdateSimple(in_value, out_deriv, rand_mat_);
    else  // the call below is to a virtual function that may be re-implemented
      to_update->Update(debug_info, in_value, out_deriv, rand_mat_);  // by child classes.
  }
}

void BayesAffineComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<LinearParamsMean>");
  linear_params_mean_.Read(is, binary);
  ExpectToken(is, binary, "<LinearParamsStd>");
  linear_params_std_.Read(is, binary);
  ExpectToken(is, binary, "<LinearPriorMean>");
  linear_prior_mean_.Read(is, binary);
  ExpectToken(is, binary, "<LinearPriorStd>");
  linear_prior_std_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  if (PeekToken(is, binary) == 'I') {
    // for back compatibility; we don't write this here any
    // more as it's written and read in Write/ReadUpdatableCommon
    ExpectToken(is, binary, "<IsGradient>");
    ReadBasicType(is, binary, &is_gradient_);
  }
  if (PeekToken(is, binary) == 'O') {
    ExpectToken(is, binary, "<OrthonormalConstraint>");
    ReadBasicType(is, binary, &orthonormal_constraint_);
  } else {
    orthonormal_constraint_ = 0.0;
  }
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "<ShareStdInput>");
  ReadBasicType(is, binary, &share_std_input_);
  ExpectToken(is, binary, "<ShareStdOutput>");
  ReadBasicType(is, binary, &share_std_output_);
  ExpectToken(is, binary, "<ShareStdInputSampling>");
  ReadBasicType(is, binary, &share_std_input_sampling_);
  ExpectToken(is, binary, "<ShareStdOutputSampling>");
  ReadBasicType(is, binary, &share_std_output_sampling_);
  ExpectToken(is, binary, "<UseExpStd>");
  ReadBasicType(is, binary, &use_exp_std_);
  ExpectToken(is, binary, "<UpdatePrior>");
  ReadBasicType(is, binary, &update_prior_);
  ExpectToken(is, binary, "<KLScale>");
  ReadBasicType(is, binary, &KL_scale_);
  ExpectToken(is, binary, "</BayesAffineComponent>");
}

void BayesAffineComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<LinearParamsMean>");
  linear_params_mean_.Write(os, binary);
  WriteToken(os, binary, "<LinearParamsStd>");
  linear_params_std_.Write(os, binary);
  WriteToken(os, binary, "<LinearPriorMean>");
  linear_prior_mean_.Write(os, binary);
  WriteToken(os, binary, "<LinearPriorStd>");
  linear_prior_std_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  if (orthonormal_constraint_ != 0.0) {
    WriteToken(os, binary, "<OrthonormalConstraint>");
    WriteBasicType(os, binary, orthonormal_constraint_);
  }
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "<ShareStdInput>");
  WriteBasicType(os, binary, share_std_input_);
  WriteToken(os, binary, "<ShareStdOutput>");
  WriteBasicType(os, binary, share_std_output_);
  WriteToken(os, binary, "<ShareStdInputSampling>");
  WriteBasicType(os, binary, share_std_input_sampling_);
  WriteToken(os, binary, "<ShareStdOutputSampling>");
  WriteBasicType(os, binary, share_std_output_sampling_);
  WriteToken(os, binary, "<UseExpStd>");
  WriteBasicType(os, binary, use_exp_std_);
  WriteToken(os, binary, "<UpdatePrior>");
  WriteBasicType(os, binary, update_prior_);
  WriteToken(os, binary, "<KLScale>");
  WriteBasicType(os, binary, KL_scale_);
  WriteToken(os, binary, "</BayesAffineComponent>");
}

int32 BayesAffineComponent::NumParameters() const {
  int32 in_dim_std = linear_params_std_.NumCols(), out_dim_std = linear_params_std_.NumRows();
  return (InputDim()*2 + 1) * OutputDim() + in_dim_std*out_dim_std*2;
}
void BayesAffineComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  int32 in_dim_std = linear_params_std_.NumCols(), 
  out_dim_std = linear_params_std_.NumRows(), 
  pose_current;
  
  params->Range(0, InputDim() * OutputDim()).CopyRowsFromMat(linear_params_mean_);
  pose_current = InputDim() * OutputDim();
  params->Range(pose_current, in_dim_std*out_dim_std).CopyRowsFromMat(linear_params_std_);
  pose_current += in_dim_std*out_dim_std;
  params->Range(pose_current, InputDim() * OutputDim()).CopyRowsFromMat(linear_prior_mean_);
  pose_current += InputDim() * OutputDim();
  params->Range(pose_current, in_dim_std*out_dim_std).CopyRowsFromMat(linear_prior_std_);
  pose_current += in_dim_std*out_dim_std;
  params->Range(pose_current, OutputDim()).CopyFromVec(bias_params_);
}
void BayesAffineComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  int32 in_dim_std = linear_params_std_.NumCols(), 
  out_dim_std = linear_params_std_.NumRows(), 
  pose_current;
  
  linear_params_mean_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
  pose_current = InputDim() * OutputDim();
  linear_params_std_.CopyRowsFromVec(params.Range(pose_current, in_dim_std*out_dim_std));
  pose_current += in_dim_std*out_dim_std;
  linear_prior_mean_.CopyRowsFromVec(params.Range(pose_current, InputDim() * OutputDim()));
  pose_current += InputDim() * OutputDim();
  linear_prior_std_.CopyRowsFromVec(params.Range(pose_current, in_dim_std*out_dim_std));
  pose_current += in_dim_std*out_dim_std;
  bias_params_.CopyFromVec(params.Range(pose_current, OutputDim()));
}

////////////////////////////////////////////////////////////////////////////////////


void NaturalGradientBayesAffineComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<LinearParamsMean>");
  linear_params_mean_.Read(is, binary);
  ExpectToken(is, binary, "<LinearParamsStd>");
  linear_params_std_.Read(is, binary);
  ExpectToken(is, binary, "<LinearPriorMean>");
  linear_prior_mean_.Read(is, binary);
  ExpectToken(is, binary, "<LinearPriorStd>");
  linear_prior_std_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "<ShareStdInput>");
  ReadBasicType(is, binary, &share_std_input_);
  ExpectToken(is, binary, "<ShareStdOutput>");
  ReadBasicType(is, binary, &share_std_output_);
  ExpectToken(is, binary, "<ShareStdInputSampling>");
  ReadBasicType(is, binary, &share_std_input_sampling_);
  ExpectToken(is, binary, "<ShareStdOutputSampling>");
  ReadBasicType(is, binary, &share_std_output_sampling_);
  ExpectToken(is, binary, "<UseExpStd>");
  ReadBasicType(is, binary, &use_exp_std_);
  ExpectToken(is, binary, "<UpdatePrior>");
  ReadBasicType(is, binary, &update_prior_);
  ExpectToken(is, binary, "<KLScale>");
  ReadBasicType(is, binary, &KL_scale_);

  BaseFloat num_samples_history, alpha;
  int32 rank_in, rank_out, update_period;

  ExpectToken(is, binary, "<RankIn>");
  ReadBasicType(is, binary, &rank_in);
  ExpectToken(is, binary, "<RankOut>");
  ReadBasicType(is, binary, &rank_out);
  if (PeekToken(is, binary) == 'O') {
    ExpectToken(is, binary, "<OrthonormalConstraint>");
    ReadBasicType(is, binary, &orthonormal_constraint_);
  } else {
    orthonormal_constraint_ = 0.0;
  }
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period);
  ExpectToken(is, binary, "<NumSamplesHistory>");
  ReadBasicType(is, binary, &num_samples_history);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha);

  preconditioner_in_.SetNumSamplesHistory(num_samples_history);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history);
  preconditioner_in_.SetAlpha(alpha);
  preconditioner_out_.SetAlpha(alpha);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetUpdatePeriod(update_period);
  preconditioner_out_.SetUpdatePeriod(update_period);

  if (PeekToken(is, binary) == 'M') {
    // MaxChangePerSample, long ago removed; back compatibility.
    ExpectToken(is, binary, "<MaxChangePerSample>");
    BaseFloat temp;
    ReadBasicType(is, binary, &temp);
  }
  if (PeekToken(is, binary) == 'I') {
    // for back compatibility; we don't write this here any
    // more as it's written and read in Write/ReadUpdatableCommon
    ExpectToken(is, binary, "<IsGradient>");
    ReadBasicType(is, binary, &is_gradient_);
  }
  if (PeekToken(is, binary) == 'U') {
    ExpectToken(is, binary, "<UpdateCount>");
    // back-compatibility branch (these configs were added and then removed).
    double temp;
    ReadBasicType(is, binary, &temp);
    ExpectToken(is, binary, "<ActiveScalingCount>");
    ReadBasicType(is, binary, &temp);
    ExpectToken(is, binary, "<MaxChangeScaleStats>");
    ReadBasicType(is, binary, &temp);
  }
  std::string token;
  ReadToken(is, binary, &token);
  // the following has to handle a couple variants of
  if (token.find("NaturalGradientBayesAffineComponent>") == std::string::npos)
    KALDI_ERR << "Expected <NaturalGradientBayesAffineComponent> or "
              << "</NaturalGradientBayesAffineComponent>, got " << token;
}


NaturalGradientBayesAffineComponent::NaturalGradientBayesAffineComponent(
    const CuMatrixBase<BaseFloat> &linear_params_mean,
		const CuMatrixBase<BaseFloat> &linear_params_std,
		const CuMatrixBase<BaseFloat> &linear_prior_mean,
		const CuMatrixBase<BaseFloat> &linear_prior_std,
        const CuVectorBase<BaseFloat> &bias_params):
    BayesAffineComponent(linear_params_mean, linear_params_std, linear_prior_mean, linear_prior_std, bias_params, 0.001) {
  KALDI_ASSERT(bias_params.Dim() == linear_params_mean.NumRows() &&
               bias_params.Dim() != 0);

  // set some default natural gradient configs.
  preconditioner_in_.SetRank(20);
  preconditioner_out_.SetRank(80);
  preconditioner_in_.SetUpdatePeriod(4);
  preconditioner_out_.SetUpdatePeriod(4);
}

void NaturalGradientBayesAffineComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;

  is_gradient_ = false;  // not configurable; there's no reason you'd want this

  test_mode_ = false;
  share_std_input_ = false;
  share_std_output_ = false;
  share_std_input_sampling_ = false;
  share_std_output_sampling_ = false;
  use_exp_std_ = false;
  update_prior_ = false;
  KL_scale_ = 0.00001;
  
  InitLearningRatesFromConfig(cfl);

  cfl->GetValue("test-mode", &test_mode_);
  cfl->GetValue("share-std-input", &share_std_input_);
  cfl->GetValue("share-std-output", &share_std_output_);
  cfl->GetValue("share-std-input-sampling", &share_std_input_sampling_);
  cfl->GetValue("share-std-output-sampling", &share_std_output_sampling_);
  cfl->GetValue("use-exp-std", &use_exp_std_);
  cfl->GetValue("use-exp-std", &update_prior_);
  cfl->GetValue("KL-scale", &KL_scale_);
  
  if (cfl->GetValue("matrix", &matrix_filename)) {
    CuMatrix<BaseFloat> mat;
    ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
    KALDI_ASSERT(mat.NumCols() >= 2);
    
	int32 output_dim = mat.NumRows();
	int32 input_dim = (mat.NumCols() - 1)/4;
	linear_params_mean_.Resize(output_dim, input_dim);
	linear_prior_mean_.Resize(output_dim, input_dim);
	int32 input_dim_re = input_dim;
	int32 output_dim_re = output_dim;
	if (share_std_input_) {
		input_dim_re = 1;
	}
	if (share_std_output_) {
		output_dim_re = 1;
	}
	
    linear_params_std_.Resize(output_dim_re, input_dim_re);
    linear_prior_std_.Resize(output_dim_re, input_dim_re);
    bias_params_.Resize(output_dim);
  
    linear_params_mean_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
    linear_params_std_.CopyFromMat(mat.Range(0, output_dim_re, input_dim, input_dim_re));
    linear_prior_mean_.CopyFromMat(mat.Range(0, output_dim, 2*input_dim, input_dim));
    linear_prior_std_.CopyFromMat(mat.Range(0, output_dim_re, 3*input_dim, input_dim_re));
    bias_params_.CopyColFromMat(mat, 4*input_dim);
	
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    int32 input_dim = -1, output_dim = -1;

    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    if (!ok)
      KALDI_ERR << "Bad initializer " << cfl->WholeLine();
    BaseFloat param_mean_stddev = 1.0 / std::sqrt(input_dim),
		param_std_stddev = 1.0 / std::sqrt(input_dim),
		prior_mean = 0, prior_std,
        bias_stddev = 1.0, bias_mean = 0.0;
	if (!use_exp_std_) {
		prior_std = 1.0;
	} else {
		prior_std = 0.0;
	}
    cfl->GetValue("param-stddev", &param_mean_stddev);
	cfl->GetValue("param-std-stddev", &param_std_stddev);
	cfl->GetValue("prior-mean", &prior_mean);
	cfl->GetValue("prior-std", &prior_std);
    cfl->GetValue("bias-stddev", &bias_stddev);
    cfl->GetValue("bias-mean", &bias_mean);
	
    linear_params_mean_.Resize(output_dim, input_dim);
    linear_prior_mean_.Resize(output_dim, input_dim);
    int32 input_dim_re = input_dim;
    int32 output_dim_re = output_dim;
    if (share_std_input_) {
	    input_dim_re = 1;
    }
    if (share_std_output_) {
	    output_dim_re = 1;
    }
    linear_params_std_.Resize(output_dim_re, input_dim_re);
    linear_prior_std_.Resize(output_dim_re, input_dim_re);
    bias_params_.Resize(output_dim);
    KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_mean_stddev >= 0.0 && param_std_stddev >= 0.0 &&
                 bias_stddev >= 0.0);
    linear_params_mean_.SetRandn(); // sets to random normally distributed noise.
    linear_params_mean_.Scale(param_mean_stddev);
  
    linear_params_std_.SetRandn(); // sets to random normally distributed noise.
    linear_params_std_.Scale(param_std_stddev);
    if (!use_exp_std_) {
	    linear_params_std_.Add(1.0); // init around 1
    }
  
    linear_prior_mean_.Set(prior_mean);
    linear_prior_std_.Set(prior_std);
    bias_params_.SetRandn();
    bias_params_.Scale(bias_stddev);
    bias_params_.Add(bias_mean);
  }

  orthonormal_constraint_ = 0.0;
  cfl->GetValue("orthonormal-constraint", &orthonormal_constraint_);

  // Set natural-gradient configs.
  BaseFloat num_samples_history = 2000.0,
      alpha = 4.0;
  int32 rank_in = -1, rank_out = -1,
      update_period = 4;
  cfl->GetValue("num-samples-history", &num_samples_history);
  cfl->GetValue("alpha", &alpha);
  cfl->GetValue("rank-in", &rank_in);
  cfl->GetValue("rank-out", &rank_out);
  cfl->GetValue("update-period", &update_period);

  if (rank_in < 0)
    rank_in = std::min<int32>(20, (InputDim() + 1) / 2);
  if (rank_out < 0)
    rank_out = std::min<int32>(80, (OutputDim() + 1) / 2);

  preconditioner_in_.SetNumSamplesHistory(num_samples_history);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history);
  preconditioner_in_.SetAlpha(alpha);
  preconditioner_out_.SetAlpha(alpha);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetUpdatePeriod(update_period);
  preconditioner_out_.SetUpdatePeriod(update_period);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void NaturalGradientBayesAffineComponent::Write(std::ostream &os,
                                           bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<LinearParamsMean>");
  linear_params_mean_.Write(os, binary);
  WriteToken(os, binary, "<LinearParamsStd>");
  linear_params_std_.Write(os, binary);
  WriteToken(os, binary, "<LinearPriorMean>");
  linear_prior_mean_.Write(os, binary);
  WriteToken(os, binary, "<LinearPriorStd>");
  linear_prior_std_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "<ShareStdInput>");
  WriteBasicType(os, binary, share_std_input_);
  WriteToken(os, binary, "<ShareStdOutput>");
  WriteBasicType(os, binary, share_std_output_);
  WriteToken(os, binary, "<ShareStdInputSampling>");
  WriteBasicType(os, binary, share_std_input_sampling_);
  WriteToken(os, binary, "<ShareStdOutputSampling>");
  WriteBasicType(os, binary, share_std_output_sampling_);
  WriteToken(os, binary, "<UseExpStd>");
  WriteBasicType(os, binary, use_exp_std_);
  WriteToken(os, binary, "<UpdatePrior>");
  WriteBasicType(os, binary, update_prior_);
  WriteToken(os, binary, "<KLScale>");
  WriteBasicType(os, binary, KL_scale_);
  WriteToken(os, binary, "<RankIn>");
  WriteBasicType(os, binary, preconditioner_in_.GetRank());
  WriteToken(os, binary, "<RankOut>");
  WriteBasicType(os, binary, preconditioner_out_.GetRank());
  if (orthonormal_constraint_ != 0.0) {
    WriteToken(os, binary, "<OrthonormalConstraint>");
    WriteBasicType(os, binary, orthonormal_constraint_);
  }
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, preconditioner_in_.GetUpdatePeriod());
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, preconditioner_in_.GetNumSamplesHistory());
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, preconditioner_in_.GetAlpha());
  WriteToken(os, binary, "</NaturalGradientBayesAffineComponent>");
}

std::string NaturalGradientBayesAffineComponent::Info() const {
  std::ostringstream stream;
  stream << BayesAffineComponent::Info();
  stream << ", rank-in=" << preconditioner_in_.GetRank()
         << ", rank-out=" << preconditioner_out_.GetRank()
         << ", num-samples-history=" << preconditioner_in_.GetNumSamplesHistory()
         << ", update-period=" << preconditioner_in_.GetUpdatePeriod()
         << ", alpha=" << preconditioner_in_.GetAlpha();
  return stream.str();
}

Component* NaturalGradientBayesAffineComponent::Copy() const {
  return new NaturalGradientBayesAffineComponent(*this);
}

NaturalGradientBayesAffineComponent::NaturalGradientBayesAffineComponent(
    const NaturalGradientBayesAffineComponent &other):
    BayesAffineComponent(other),
    preconditioner_in_(other.preconditioner_in_),
    preconditioner_out_(other.preconditioner_out_) { }

void NaturalGradientBayesAffineComponent::Update(
    const std::string &debug_info,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
	const CuMatrixBase<BaseFloat> &rand_mat) {
  CuMatrix<BaseFloat> in_value_temp;

  in_value_temp.Resize(in_value.NumRows(),
                       in_value.NumCols() + 1, kUndefined);
  in_value_temp.Range(0, in_value.NumRows(),
                      0, in_value.NumCols()).CopyFromMat(in_value);

  // Add the 1.0 at the end of each row "in_value_temp"
  in_value_temp.Range(0, in_value.NumRows(),
                      in_value.NumCols(), 1).Set(1.0);

  CuMatrix<BaseFloat> out_deriv_temp(out_deriv);

  // These "scale" values get will get multiplied into the learning rate (faster
  // than having the matrices scaled inside the preconditioning code).
  BaseFloat in_scale, out_scale;

  preconditioner_in_.PreconditionDirections(&in_value_temp, &in_scale);
  preconditioner_out_.PreconditionDirections(&out_deriv_temp, &out_scale);

  // "scale" is a scaling factor coming from the PreconditionDirections calls
  // (it's faster to have them output a scaling factor than to have them scale
  // their outputs).
  BaseFloat scale = in_scale * out_scale;

  CuSubMatrix<BaseFloat> in_value_precon_part(in_value_temp,
                                              0, in_value_temp.NumRows(),
                                              0, in_value_temp.NumCols() - 1);
  // this "precon_ones" is what happens to the vector of 1's representing
  // offsets, after multiplication by the preconditioner.
  CuVector<BaseFloat> precon_ones(in_value_temp.NumRows());

  precon_ones.CopyColFromMat(in_value_temp, in_value_temp.NumCols() - 1);

  BaseFloat local_lrate = scale * learning_rate_;

  bias_params_.AddMatVec(local_lrate, out_deriv_temp, kTrans,
                         precon_ones, 1.0);
  //linear_params_.AddMatMat(local_lrate, out_deriv_temp, kTrans,
  //                         in_value_precon_part, kNoTrans, 1.0);
  
  int32 dim_input = linear_params_mean_.NumCols();
  int32 dim_output = linear_params_mean_.NumRows();
  CuMatrix<BaseFloat> linear_deriv(dim_output, dim_input);
  linear_deriv.AddMatMat(1.0, out_deriv_temp, kTrans,
                           in_value_precon_part, kNoTrans, 0.0);
  linear_params_mean_.AddMat(local_lrate, linear_deriv);
  linear_deriv.MulElements(rand_mat);
  if (use_exp_std_) {
	  CuMatrix<BaseFloat> linear_params_std_exp(linear_params_std_);
	  linear_params_std_exp.ApplyCeiling(46.05);
	  linear_params_std_exp.ApplyExp();
	  linear_deriv.MulElements(linear_params_std_exp);
  }
  linear_params_std_.AddMatBlocks(local_lrate, linear_deriv, kNoTrans);
  
  // original
  if (KL_scale_ != 0) {
	  int32 num_frame = in_value.NumRows();
	  
	  CuMatrix<BaseFloat> inv_prior_std(dim_output, dim_input);
	  CuMatrix<BaseFloat> diff(dim_output, dim_input);
	  CuMatrix<BaseFloat> diff_power_2(dim_output, dim_input);
	  CuMatrix<BaseFloat> rate(dim_output, dim_input);
	  
	  CuMatrix<BaseFloat> post_mean_deriv(dim_output, dim_input);
	  CuMatrix<BaseFloat> post_std_deriv(dim_output, dim_input);
	  CuMatrix<BaseFloat> prior_mean_deriv(dim_output, dim_input);
	  CuMatrix<BaseFloat> prior_std_deriv(dim_output, dim_input);
	  
	  CuMatrix<BaseFloat> linear_params_std_resize(dim_output, dim_input);
	  CuMatrix<BaseFloat> linear_prior_std_resize(dim_output, dim_input);
	  linear_params_std_resize.AddMatBlocks(1.0, linear_params_std_, kNoTrans);
	  linear_prior_std_resize.AddMatBlocks(1.0, linear_prior_std_, kNoTrans);
	  
	  if (use_exp_std_) {
		  linear_params_std_resize.ApplyCeiling(46.05); // avoid inf
		  linear_params_std_resize.ApplyExp(); // std = exp()
		  linear_prior_std_resize.ApplyCeiling(46.05); // avoid inf
		  linear_prior_std_resize.ApplyExp(); // std = exp()
	  }
  
	  inv_prior_std.CopyFromMat(linear_prior_std_resize);
	  inv_prior_std.InvertElements(); // 1/prior_std
  
	  rate.CopyFromMat(linear_params_std_resize);
	  rate.MulElements(inv_prior_std); // post_std/prior_std
	  CuMatrix<BaseFloat> rate_power_2(rate);
	  rate_power_2.MulElements(rate); // post_std^2/prior_std^2
  
	  diff.CopyFromMat(linear_params_mean_);
	  diff.AddMat(-1.0, linear_prior_mean_); // post_mean - prior_mean
	  diff.MulElements(inv_prior_std); // (post_mean - prior_mean)/prior_std
  
	  diff_power_2.CopyFromMat(diff);
	  diff_power_2.MulElements(diff); // (post_mean - prior_mean)^2/prior_std^2
	  diff_power_2.AddMat(1.0, rate_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2
	  diff_power_2.Add(-1.0); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
	  
	  post_mean_deriv.CopyFromMat(diff); // (post_mean - prior_mean)/prior_std
	  post_mean_deriv.MulElements(inv_prior_std); // // (post_mean - prior_mean)/prior_std^2
	  
	  post_std_deriv.CopyFromMat(rate); // post_std/prior_std
	  post_std_deriv.MulElements(inv_prior_std); // post_std/prior_std^2
	  CuMatrix<BaseFloat> inv_post_std(dim_output, dim_input);
	  inv_post_std.CopyFromMat(linear_params_std_resize);
	  inv_post_std.InvertElements(); // 1/post_std
	  post_std_deriv.AddMat(-1.0, inv_post_std); // post_std/prior_std^2 - 1/post_std
	  if (use_exp_std_) {
		  post_std_deriv.MulElements(linear_params_std_resize);
	  }
	  
	  BaseFloat learning_rate_KL_ = -1.0*learning_rate_*num_frame*KL_scale_;
	  linear_params_mean_.AddMat(learning_rate_KL_, post_mean_deriv);
	  linear_params_std_.AddMatBlocks(learning_rate_KL_, post_std_deriv, kNoTrans);
	  
	  if (update_prior_) {
		  prior_mean_deriv.CopyFromMat(post_mean_deriv);
		  prior_mean_deriv.Scale(-1.0); // - (post_mean - prior_mean)/prior_std
	  
		  prior_std_deriv.CopyFromMat(diff_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
		  if (!use_exp_std_) {
			prior_std_deriv.MulElements(inv_prior_std); // (post_mean - prior_mean)^2/prior_std^3 + post_std^2/prior_std^3 - 1/prior_std
		  }
		  prior_std_deriv.Scale(-1.0); // -(post_mean - prior_mean)^2/prior_std^3 - post_std^2/prior_std^3 + 1/prior_std
		  
		  linear_prior_mean_.AddMat(learning_rate_KL_, prior_mean_deriv);
		  linear_prior_std_.AddMatBlocks(learning_rate_KL_, prior_std_deriv, kNoTrans);
	  }
  }
}

void NaturalGradientBayesAffineComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    // If scale == 0.0 we call SetZero() which will get rid of NaN's and inf's.
    linear_params_mean_.SetZero();
	linear_params_std_.SetZero();
	linear_prior_mean_.SetZero();
	linear_prior_std_.SetZero();
    bias_params_.SetZero();
  } else {
	linear_params_mean_.Scale(scale);
	linear_params_std_.Scale(scale);
	linear_prior_mean_.Scale(scale);
	linear_prior_std_.Scale(scale);
    bias_params_.Scale(scale);
  }
}

void NaturalGradientBayesAffineComponent::Add(BaseFloat alpha, const Component &other_in) {
  const NaturalGradientBayesAffineComponent *other =
      dynamic_cast<const NaturalGradientBayesAffineComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_mean_.AddMat(alpha, other->linear_params_mean_);
  linear_prior_mean_.AddMat(alpha, other->linear_prior_mean_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

void NaturalGradientBayesAffineComponent::FreezeNaturalGradient(bool freeze) {
  preconditioner_in_.Freeze(freeze);
  preconditioner_out_.Freeze(freeze);
}

void NaturalGradientBayesAffineComponent::ConsolidateMemory() {
  OnlineNaturalGradient temp_in(preconditioner_in_);
  preconditioner_in_.Swap(&temp_in);
  OnlineNaturalGradient temp_out(preconditioner_out_);
  preconditioner_out_.Swap(&temp_out);
}

////////////////////////////////////////////////////////////////////////////////////


void LinearSelectColComponent::Read(std::istream &is, bool binary) {
  std::string token = ReadUpdatableCommon(is, binary);
  KALDI_ASSERT(token == "");
  ExpectToken(is, binary, "<Params>");
  params_.Read(is, binary);
  if (PeekToken(is, binary) == 'O') {
    ExpectToken(is, binary, "<OrthonormalConstraint>");
    ReadBasicType(is, binary, &orthonormal_constraint_);
  } else {
    orthonormal_constraint_ = 0.0;
  }
  ExpectToken(is, binary, "<UseNaturalGradient>");
  ReadBasicType(is, binary, &use_natural_gradient_);

  // Read various natural-gradient-related configs.
  int32 rank_in,  rank_out, update_period;
  BaseFloat alpha, num_samples_history;
  ExpectToken(is, binary, "<RankInOut>");
  ReadBasicType(is, binary, &rank_in);
  ReadBasicType(is, binary, &rank_out);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha);
  ExpectToken(is, binary, "<NumSamplesHistory>");
  ReadBasicType(is, binary, &num_samples_history);
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period);

  preconditioner_in_.SetAlpha(alpha);
  preconditioner_out_.SetAlpha(alpha);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history);
  preconditioner_in_.SetUpdatePeriod(update_period);
  preconditioner_out_.SetUpdatePeriod(update_period);

  ExpectToken(is, binary, "</LinearSelectColComponent>");
}

void LinearSelectColComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  is_gradient_ = false;  // not configurable; there's no reason you'd want this

  InitLearningRatesFromConfig(cfl);

  int32 input_dim = -1, output_dim = -1, col_number = 1;
  if (cfl->GetValue("matrix", &matrix_filename)) {
    ReadKaldiObject(matrix_filename, &params_); // will abort on failure.
    KALDI_ASSERT(params_.NumRows() != 0);
    if (cfl->GetValue("col-num", &col_number))
      KALDI_ASSERT(col_number == params_.NumCols() &&
                   "col-num mismatch vs. matrix.");
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == 1 &&
                   "input-dim must be 1.");
	if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("col-num", &col_number);
	ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    if (!ok)
      KALDI_ERR << "Bad initializer " << cfl->WholeLine();
	BaseFloat param_stddev = 1.0 / std::sqrt(col_number), param_mean = 0.0;
	cfl->GetValue("param-mean", &param_mean);
    cfl->GetValue("param-stddev", &param_stddev);
    params_.Resize(output_dim, col_number);
    KALDI_ASSERT(output_dim > 0 && input_dim == 1 && col_number > 0 && param_stddev >= 0.0);
    params_.SetRandn(); // sets to random normally distributed noise.
    params_.Scale(param_stddev);
	params_.Add(param_mean);
  }
  // Read various natural-gradient-related configs.
  int32 rank_in = -1, rank_out = -1, update_period = 4;
  BaseFloat alpha = 4.0,
      num_samples_history = 2000.0;

  use_natural_gradient_ = false;

  cfl->GetValue("num-samples-history", &num_samples_history);
  cfl->GetValue("alpha", &alpha);
  cfl->GetValue("rank-in", &rank_in);
  cfl->GetValue("rank-out", &rank_out);
  cfl->GetValue("update-period", &update_period);
  cfl->GetValue("use-natural-gradient", &use_natural_gradient_);

  if (rank_in < 0)
    rank_in = std::min<int32>(20, (InputDim() + 1) / 2);
  if (rank_out < 0)
    rank_out = std::min<int32>(80, (OutputDim() + 1) / 2);

  preconditioner_in_.SetAlpha(alpha);
  preconditioner_out_.SetAlpha(alpha);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history);
  preconditioner_in_.SetUpdatePeriod(update_period);
  preconditioner_out_.SetUpdatePeriod(update_period);

  orthonormal_constraint_ = 0.0;
  cfl->GetValue("orthonormal-constraint", &orthonormal_constraint_);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
}


void LinearSelectColComponent::Write(std::ostream &os,
                            bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
  WriteToken(os, binary, "<Params>");
  params_.Write(os, binary);
  if (orthonormal_constraint_ != 0.0) {
    WriteToken(os, binary, "<OrthonormalConstraint>");
    WriteBasicType(os, binary, orthonormal_constraint_);
  }
  WriteToken(os, binary, "<UseNaturalGradient>");
  WriteBasicType(os, binary, use_natural_gradient_);

  int32 rank_in = preconditioner_in_.GetRank(),
      rank_out = preconditioner_out_.GetRank(),
      update_period = preconditioner_in_.GetUpdatePeriod();
  BaseFloat alpha = preconditioner_in_.GetAlpha(),
      num_samples_history = preconditioner_in_.GetNumSamplesHistory();
  WriteToken(os, binary, "<RankInOut>");
  WriteBasicType(os, binary, rank_in);
  WriteBasicType(os, binary, rank_out);
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha);
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, num_samples_history);
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, update_period);
  WriteToken(os, binary, "</LinearSelectColComponent>");
}

std::string LinearSelectColComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info();
  stream << ", col-num=" << params_.NumCols();
  PrintParameterStats(stream, "params", params_,
                      true, // include_mean
                      true, // include_row_norms
                      true, // include_column_norms
                      GetVerboseLevel() >= 2); // include_singular_values
  if (orthonormal_constraint_ != 0.0)
    stream << ", orthonormal-constraint=" << orthonormal_constraint_;
  stream << ", use-natural-gradient="
         << (use_natural_gradient_ ? "true" : "false")
         << ", rank-in=" << preconditioner_in_.GetRank()
         << ", rank-out=" << preconditioner_out_.GetRank()
         << ", num-samples-history="
         << preconditioner_in_.GetNumSamplesHistory()
         << ", update-period=" << preconditioner_in_.GetUpdatePeriod()
         << ", alpha=" << preconditioner_in_.GetAlpha();
  return stream.str();
}

void* LinearSelectColComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == 1);
  int32 num_row = in.NumRows();
  int32 col_number = params_.NumCols();
  CuMatrix<BaseFloat> in_id;
  in_id.Resize(num_row,col_number);
  in_id.SelectOneFromIdRow(in);
  out->AddMatMat(1.0, in_id, kNoTrans, params_, kTrans, 1.0);
  
  CuMatrix<BaseFloat> *memo = new CuMatrix<BaseFloat>(num_row, col_number);
  memo->CopyFromMat(in_id);
  return memo;
}

void LinearSelectColComponent::Backprop(const std::string &debug_info,
                               const ComponentPrecomputedIndexes *indexes,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               void *memo,
                               Component *to_update_in,
                               CuMatrixBase<BaseFloat> *in_deriv) const {
  LinearSelectColComponent *to_update = dynamic_cast<LinearSelectColComponent*>(to_update_in);
  CuMatrix<BaseFloat> *in_id = reinterpret_cast<CuMatrix<BaseFloat>*>(memo);
  // Propagate the derivative back to the input.  add with coefficient 1.0 since
  // property kBackpropAdds is true.  If we wanted to add with coefficient 0.0
  // we'd need to zero the in_deriv, in case of infinities.
  if (to_update != NULL) {
    if (!to_update->is_gradient_) {
      CuMatrix<BaseFloat> in_value_temp(*in_id), out_deriv_temp(out_deriv);
      // These "scale" values get will get multiplied into the learning rate (faster
      // than having the matrices scaled inside the preconditioning code).
      BaseFloat in_scale, out_scale;
      to_update->preconditioner_in_.PreconditionDirections(&in_value_temp,
                                                           &in_scale);
      to_update->preconditioner_out_.PreconditionDirections(&out_deriv_temp,
                                                            &out_scale);
      BaseFloat local_lrate = in_scale * out_scale * to_update->learning_rate_;

      to_update->params_.AddMatMat(local_lrate, out_deriv_temp, kTrans,
                                   in_value_temp, kNoTrans, 1.0);
    } else {
	  to_update->params_.AddMatMat(to_update->learning_rate_,
                                   out_deriv, kTrans,
                                   *in_id, kNoTrans, 1.0);
    }
  }
}


Component* LinearSelectColComponent::Copy() const {
  return new LinearSelectColComponent(*this);
}

LinearSelectColComponent::LinearSelectColComponent(
    const LinearSelectColComponent &other):
    UpdatableComponent(other),
    params_(other.params_),
    orthonormal_constraint_(other.orthonormal_constraint_),
    use_natural_gradient_(other.use_natural_gradient_),
    preconditioner_in_(other.preconditioner_in_),
    preconditioner_out_(other.preconditioner_out_) { }

LinearSelectColComponent::LinearSelectColComponent(const CuMatrix<BaseFloat> &params):
    params_(params),
    orthonormal_constraint_(0.0),
    use_natural_gradient_(false) {
  // Set defaults for natural gradient.
  preconditioner_in_.SetRank(40);
  preconditioner_out_.SetRank(80);
  preconditioner_in_.SetUpdatePeriod(4);
  preconditioner_out_.SetUpdatePeriod(4);
  // the component-level defaults of alpha and num_samples_history, at 4.0 and
  // 2000.0, are the same as in the NaturalGradientOnline code, so there is no
  // need to set those here.
}

void LinearSelectColComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) params_.SetZero();
  else params_.Scale(scale);
}

void LinearSelectColComponent::Add(BaseFloat alpha, const Component &other_in) {
  const LinearSelectColComponent *other =
      dynamic_cast<const LinearSelectColComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  params_.AddMat(alpha, other->params_);
}

void LinearSelectColComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_params(params_);
  temp_params.SetRandn();
  params_.AddMat(stddev, temp_params);
}
int32 LinearSelectColComponent::NumParameters() const {
  return params_.NumRows() * params_.NumCols();
}
void LinearSelectColComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->CopyRowsFromMat(params_);
}
void LinearSelectColComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  params_.CopyRowsFromVec(params);
}
BaseFloat LinearSelectColComponent::DotProduct(const UpdatableComponent &other_in) const {
  const LinearSelectColComponent *other =
      dynamic_cast<const LinearSelectColComponent*>(&other_in);
  return TraceMatMat(params_, other->params_, kTrans);
}

void LinearSelectColComponent::FreezeNaturalGradient(bool freeze) {
  preconditioner_in_.Freeze(freeze);
  preconditioner_out_.Freeze(freeze);
}

void LinearSelectColComponent::ConsolidateMemory() {
  OnlineNaturalGradient temp_in(preconditioner_in_);
  preconditioner_in_.Swap(&temp_in);
  OnlineNaturalGradient temp_out(preconditioner_out_);
  preconditioner_out_.Swap(&temp_out);
}


//////////

void VecL2NormComponent::Init(int32 input_dim, int32 output_dim, BaseFloat scale, bool input_frame_scale, bool inv_frame_scale)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  scale_ = scale;
  input_frame_scale_ = input_frame_scale;
  inv_frame_scale_ = inv_frame_scale;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0);
  if (input_frame_scale_) {
	  KALDI_ASSERT(input_dim_ == output_dim_+1);
  } else {
	  KALDI_ASSERT(input_dim_ == output_dim_);
  }
}

void VecL2NormComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  BaseFloat scale = 1.0;
  bool input_frame_scale = false;
  bool inv_frame_scale = false;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || output_dim <= 0)
    KALDI_ERR << "Invalid dim for layer of type "
              << Type();
  cfl->GetValue("scale", &scale);
  cfl->GetValue("input-frame-scale", &input_frame_scale);
  cfl->GetValue("inv-frame-scale", &inv_frame_scale);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim, scale, input_frame_scale, inv_frame_scale);
}

void* VecL2NormComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in.ColRange(0,output_dim_));
  return NULL;
}

void VecL2NormComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {

  in_deriv->SetZero();
  (in_deriv->ColRange(0,output_dim_)).CopyFromMat(out_deriv);

  if (scale_ != 0) {
	  CuMatrix<BaseFloat> in(in_value.NumRows(), output_dim_);
	  in.CopyFromMat(in_value.ColRange(0,output_dim_));
	  
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == output_dim_+1);
		  CuVector<BaseFloat> frame_scale(in_value.NumRows());
		  frame_scale.CopyColFromMat(in_value, output_dim_);
		  if (inv_frame_scale_) {
			  frame_scale.InvertElements();
		  }
		  in.MulRowsVec(frame_scale);
	  }

	  (in_deriv->ColRange(0,output_dim_)).AddMat(-1.0*scale_, in);
  }
}

void VecL2NormComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<VecL2NormComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "<InputeFrameScale>");
  ReadBasicType(is, binary, &input_frame_scale_);
  ExpectToken(is, binary, "<InvFrameScale>");
  ReadBasicType(is, binary, &inv_frame_scale_);
  ExpectToken(is, binary, "</VecL2NormComponent>");
}

void VecL2NormComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<VecL2NormComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "<InputeFrameScale>");
  WriteBasicType(os, binary, input_frame_scale_);
  WriteToken(os, binary, "<InvFrameScale>");
  WriteBasicType(os, binary, inv_frame_scale_);
  WriteToken(os, binary, "</VecL2NormComponent>");
}

std::string VecL2NormComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", scale=" << scale_
	<< ", input-frame-scale=" << (input_frame_scale_ ? "true" : "false")
	<< ", inv-frame-scale=" << (inv_frame_scale_ ? "true" : "false");
  return stream.str();
}


//////////

void VeclogNormComponent::Init(int32 input_dim, int32 output_dim, BaseFloat scale, bool input_frame_scale, bool inv_frame_scale)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  scale_ = scale;
  input_frame_scale_ = input_frame_scale;
  inv_frame_scale_ = inv_frame_scale;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0);
  if (input_frame_scale_) {
	  KALDI_ASSERT(input_dim_ == output_dim_+1);
  } else {
	  KALDI_ASSERT(input_dim_ == output_dim_);
  }
}

void VeclogNormComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  BaseFloat scale = 1.0;
  bool input_frame_scale = false;
  bool inv_frame_scale = false;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || output_dim <= 0)
    KALDI_ERR << "Invalid dim for layer of type "
              << Type();
  cfl->GetValue("scale", &scale);
  cfl->GetValue("input-frame-scale", &input_frame_scale);
  cfl->GetValue("inv-frame-scale", &inv_frame_scale);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim, scale, input_frame_scale, inv_frame_scale);
}

void* VeclogNormComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in.ColRange(0,output_dim_));
  return NULL;
}

void VeclogNormComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {

  in_deriv->SetZero();
  (in_deriv->ColRange(0,output_dim_)).CopyFromMat(out_deriv);

  if (scale_ != 0) {
	  CuMatrix<BaseFloat> in(in_value.NumRows(), output_dim_);
	  in.CopyFromMat(in_value.ColRange(0,output_dim_));
	  in.InvertElements();
	  
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == output_dim_+1);
		  CuVector<BaseFloat> frame_scale(in_value.NumRows());
		  frame_scale.CopyColFromMat(in_value, output_dim_);
		  if (inv_frame_scale_) {
			  frame_scale.InvertElements();
		  }
		  in.MulRowsVec(frame_scale);
	  }

	  (in_deriv->ColRange(0,output_dim_)).AddMat(-1.0*scale_, in);
  }
}

void VeclogNormComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<VeclogNormComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "<InputeFrameScale>");
  ReadBasicType(is, binary, &input_frame_scale_);
  ExpectToken(is, binary, "<InvFrameScale>");
  ReadBasicType(is, binary, &inv_frame_scale_);
  ExpectToken(is, binary, "</VeclogNormComponent>");
}

void VeclogNormComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<VeclogNormComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "<InputeFrameScale>");
  WriteBasicType(os, binary, input_frame_scale_);
  WriteToken(os, binary, "<InvFrameScale>");
  WriteBasicType(os, binary, inv_frame_scale_);
  WriteToken(os, binary, "</VeclogNormComponent>");
}

std::string VeclogNormComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", scale=" << scale_
	<< ", input-frame-scale=" << (input_frame_scale_ ? "true" : "false")
	<< ", inv-frame-scale=" << (inv_frame_scale_ ? "true" : "false");
  return stream.str();
}

/////////////////////////////////////

PnormSumBlockComponent::PnormSumBlockComponent(const PnormSumBlockComponent &other):
    input_dim_(other.input_dim_), output_dim_(other.output_dim_),
    scale_(other.scale_) { }

void PnormSumBlockComponent::InitFromConfig(ConfigLine *cfl) {
  scale_ = 1.0;
  bool ok = cfl->GetValue("input-dim", &input_dim_) &&
      cfl->GetValue("output-dim", &output_dim_);
  if (!ok)
    KALDI_ERR << "input-dim and output-dim must both be provided.";
  if (input_dim_ <= 0 || input_dim_ % output_dim_ != 0)
    KALDI_ERR << "Invalid values input-dim=" << input_dim_
              << " output-dim=" << output_dim_;
  cfl->GetValue("scale", &scale_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
}

void PnormSumBlockComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PnormSumBlockComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "</PnormSumBlockComponent>");
}

void PnormSumBlockComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PnormSumBlockComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "</PnormSumBlockComponent>");
}

std::string PnormSumBlockComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", scale=" << scale_;
  return stream.str();
}

void* PnormSumBlockComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(out->NumRows() == in.NumRows() &&
               out->NumCols() == output_dim_ &&
               in.NumCols() == input_dim_);
  
  CuMatrix<BaseFloat> in_pow2(in.NumRows(), in.NumCols());
  in_pow2.CopyFromMat(in);
  in_pow2.MulElements(in_pow2); // x^2,y^2
  out->AddMatBlocks(scale_*scale_, in_pow2, kNoTrans); // a^2 * (x^2 + y^2)
  out->ApplyPow(0.5); // a * (x^2 + y^2)^0.5
  
  return NULL;
}

void PnormSumBlockComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv) {
    CuMatrix<BaseFloat> tmp_deriv(out_value.NumRows(), out_value.NumCols());
	tmp_deriv.CopyFromMat(out_value);
	tmp_deriv.Add(1e-20); // keep > 0
	tmp_deriv.InvertElements(); // 1/(a*(x^2 + y^2)^0.5)
	tmp_deriv.MulElements(out_deriv); // deriv/(a*(x^2 + y^2)^0.5)
	in_deriv->AddMatBlocks(scale_*scale_, tmp_deriv, kNoTrans); // a*deriv/(x^2 + y^2)^0.5
	in_deriv->MulElements(in_value); // a*deriv*x / (x^2 + y^2)^0.5, a*deriv*y / (x^2 + y^2)^0.5
  }
}

/////////////////////////////////////

WeightSumBlockComponent::WeightSumBlockComponent(const WeightSumBlockComponent &other):
    input_dim_(other.input_dim_), output_dim_(other.output_dim_),
    scale_(other.scale_), mix_num_(other.mix_num_), weight_deriv_scale_(other.weight_deriv_scale_) { }

void WeightSumBlockComponent::InitFromConfig(ConfigLine *cfl) {
  scale_ = 1.0;
  mix_num_ = 1;
  weight_deriv_scale_ = 1.0;
  bool ok = cfl->GetValue("input-dim", &input_dim_) &&
      cfl->GetValue("output-dim", &output_dim_);
  if (!ok)
    KALDI_ERR << "input-dim and output-dim must both be provided.";
  if (input_dim_ <= 0 || input_dim_ % (output_dim_+1) != 0)
    KALDI_ERR << "Invalid values input-dim=" << input_dim_
              << " output-dim=" << output_dim_;
  cfl->GetValue("scale", &scale_);
  cfl->GetValue("mix-num", &mix_num_);
  KALDI_ASSERT(input_dim_/(output_dim_+1) == mix_num_);
  cfl->GetValue("weight-deriv-scale", &weight_deriv_scale_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
}

void WeightSumBlockComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<WeightSumBlockComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "<MixNum>");
  ReadBasicType(is, binary, &mix_num_);
  ExpectToken(is, binary, "<WeightDerivScale>");
  ReadBasicType(is, binary, &weight_deriv_scale_);
  ExpectToken(is, binary, "</WeightSumBlockComponent>");
}

void WeightSumBlockComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<WeightSumBlockComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "<MixNum>");
  WriteBasicType(os, binary, mix_num_);
  WriteToken(os, binary, "<WeightDerivScale>");
  WriteBasicType(os, binary, weight_deriv_scale_);
  WriteToken(os, binary, "</WeightSumBlockComponent>");
}

std::string WeightSumBlockComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", scale=" << scale_
		 << ", mix-num=" << mix_num_;
  return stream.str();
}

void* WeightSumBlockComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(out->NumRows() == in.NumRows() &&
               out->NumCols() == output_dim_ &&
               in.NumCols() == input_dim_);
  
  int32 in_val_dim = input_dim_ - mix_num_;
  
  CuMatrix<BaseFloat> in_val(in.NumRows(), in_val_dim);
  in_val.CopyFromMat(in.ColRange(0, in_val_dim));
  CuSubMatrix<BaseFloat> in_weight(in.ColRange(in_val_dim, mix_num_));
  
  CuVector<BaseFloat> weight_vec(in.NumRows());
  for (int32 i=0; i<mix_num_; i++) {
	  weight_vec.CopyColFromMat(in_weight, i);
	  (in_val.ColRange(i*output_dim_, output_dim_)).MulRowsVec(weight_vec); // weight_i*in_i
  }
  
  out->AddMatBlocks(scale_, in_val, kNoTrans); // sum_i weight_i*in_i
  
  return NULL;
}

void WeightSumBlockComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv) {
	int32 in_val_dim = input_dim_ - mix_num_;
	
	CuMatrix<BaseFloat> in_val(in_value.NumRows(), in_val_dim);
    in_val.CopyFromMat(in_value.ColRange(0, in_val_dim));
	CuSubMatrix<BaseFloat> in_weight(in_value.ColRange(in_val_dim, mix_num_));
	CuSubMatrix<BaseFloat> deriv_in_val(in_deriv->ColRange(0, in_val_dim));
	CuSubMatrix<BaseFloat> deriv_in_weight(in_deriv->ColRange(in_val_dim, mix_num_));
	
	deriv_in_val.AddMatBlocks(scale_, out_deriv, kNoTrans);
	in_val.MulElements(deriv_in_val); // [in_i*deriv_i]
	
	CuVector<BaseFloat> weight_vec(in_value.NumRows());
	for (int32 i=0; i<mix_num_; i++) {
		weight_vec.CopyColFromMat(in_weight, i);
		(deriv_in_val.ColRange(i*output_dim_, output_dim_)).MulRowsVec(weight_vec); // weight_i*deriv_i
		weight_vec.AddColSumMat(1.0, in_val.ColRange(i*output_dim_, output_dim_), 0.0); // sum_j in_i_j*deriv_i_j
		deriv_in_weight.CopyColFromVec(weight_vec, i);
	}
	if (weight_deriv_scale_ != 1.0) {
		deriv_in_weight.Scale(weight_deriv_scale_);
	}
  }
}

/////////////////////////////////////

WeightSumBlockFixComponent::WeightSumBlockFixComponent(const WeightSumBlockFixComponent &other):
    input_dim_(other.input_dim_), output_dim_(other.output_dim_),
    scale_(other.scale_), mix_num_(other.mix_num_), weight_deriv_scale_(other.weight_deriv_scale_) { }

void WeightSumBlockFixComponent::InitFromConfig(ConfigLine *cfl) {
  scale_ = 1.0;
  mix_num_ = 1;
  weight_deriv_scale_ = 1.0;
  bool ok = cfl->GetValue("input-dim", &input_dim_) &&
      cfl->GetValue("output-dim", &output_dim_);
  if (!ok)
    KALDI_ERR << "input-dim and output-dim must both be provided.";
  if (input_dim_ <= 0 || input_dim_ % (output_dim_+1) != 0)
    KALDI_ERR << "Invalid values input-dim=" << input_dim_
              << " output-dim=" << output_dim_;
  cfl->GetValue("scale", &scale_);
  cfl->GetValue("mix-num", &mix_num_);
  KALDI_ASSERT(input_dim_/(output_dim_+1) == mix_num_);
  cfl->GetValue("weight-deriv-scale", &weight_deriv_scale_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
}

void WeightSumBlockFixComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<WeightSumBlockFixComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "<MixNum>");
  ReadBasicType(is, binary, &mix_num_);
  ExpectToken(is, binary, "<WeightDerivScale>");
  ReadBasicType(is, binary, &weight_deriv_scale_);
  ExpectToken(is, binary, "</WeightSumBlockFixComponent>");
}

void WeightSumBlockFixComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<WeightSumBlockFixComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "<MixNum>");
  WriteBasicType(os, binary, mix_num_);
  WriteToken(os, binary, "<WeightDerivScale>");
  WriteBasicType(os, binary, weight_deriv_scale_);
  WriteToken(os, binary, "</WeightSumBlockFixComponent>");
}

std::string WeightSumBlockFixComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", scale=" << scale_
		 << ", mix-num=" << mix_num_;
  return stream.str();
}

void* WeightSumBlockFixComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(out->NumRows() == in.NumRows() &&
               out->NumCols() == output_dim_ &&
               in.NumCols() == input_dim_);
  
  int32 in_val_dim = input_dim_ - mix_num_;
  
  CuMatrix<BaseFloat> in_val(in.NumRows(), in_val_dim);
  in_val.CopyFromMat(in.ColRange(0, in_val_dim));
  CuSubMatrix<BaseFloat> in_weight(in.ColRange(in_val_dim, mix_num_));
  
  CuVector<BaseFloat> weight_vec(in.NumRows());
  for (int32 i=0; i<mix_num_; i++) {
	  weight_vec.CopyColFromMat(in_weight, i);
	  (in_val.ColRange(i*output_dim_, output_dim_)).MulRowsVec(weight_vec); // weight_i*in_i
  }
  
  out->AddMatBlocks(scale_, in_val, kNoTrans); // sum_i weight_i*in_i
  
  return NULL;
}

void WeightSumBlockFixComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv) {
	int32 in_val_dim = input_dim_ - mix_num_;
	
	CuMatrix<BaseFloat> in_val(in_value.NumRows(), in_val_dim);
    in_val.CopyFromMat(in_value.ColRange(0, in_val_dim));
	CuSubMatrix<BaseFloat> in_weight(in_value.ColRange(in_val_dim, mix_num_));
	CuSubMatrix<BaseFloat> deriv_in_val(in_deriv->ColRange(0, in_val_dim));
	CuSubMatrix<BaseFloat> deriv_in_weight(in_deriv->ColRange(in_val_dim, mix_num_));
	
	deriv_in_val.AddMatBlocks(scale_, out_deriv, kNoTrans);
	in_val.MulElements(deriv_in_val); // [in_i*deriv_i]
	
	CuVector<BaseFloat> weight_vec(in_value.NumRows());
	for (int32 i=0; i<mix_num_; i++) {
		weight_vec.CopyColFromMat(in_weight, i);
		//(deriv_in_val.ColRange(i*output_dim_, output_dim_)).MulRowsVec(weight_vec); // weight_i*deriv_i
		weight_vec.AddColSumMat(1.0, in_val.ColRange(i*output_dim_, output_dim_), 0.0); // sum_j in_i_j*deriv_i_j
		deriv_in_weight.CopyColFromVec(weight_vec, i);
	}
	if (weight_deriv_scale_ != 1.0) {
		deriv_in_weight.Scale(weight_deriv_scale_);
	}
  }
}

/////////////////////////////////////

WeightSumBlockRandomComponent::WeightSumBlockRandomComponent(const WeightSumBlockRandomComponent &other):
    input_dim_(other.input_dim_), output_dim_(other.output_dim_),
    scale_(other.scale_), mix_num_(other.mix_num_), weight_deriv_scale_(other.weight_deriv_scale_),
	rand_num_(other.rand_num_), rand_per_frame_(other.rand_per_frame_), test_time_sample_(other.test_time_sample_) { }

void WeightSumBlockRandomComponent::InitFromConfig(ConfigLine *cfl) {
  scale_ = 1.0;
  mix_num_ = 1;
  weight_deriv_scale_ = 1.0;
  rand_num_ = 1;
  test_mode_ = false;
  rand_per_frame_ = false;
  test_time_sample_ = false;
  bool ok = cfl->GetValue("input-dim", &input_dim_) &&
      cfl->GetValue("output-dim", &output_dim_);
  if (!ok)
    KALDI_ERR << "input-dim and output-dim must both be provided.";
  cfl->GetValue("scale", &scale_);
  cfl->GetValue("test-time-sample", &test_time_sample_);
  cfl->GetValue("rand-per-frame", &rand_per_frame_);
  cfl->GetValue("test-mode", &test_mode_);
  cfl->GetValue("mix-num", &mix_num_);
  cfl->GetValue("rand-num", &rand_num_);
  if (input_dim_ <= 0 || (input_dim_-mix_num_) % output_dim_ != 0 || ((input_dim_-mix_num_)/output_dim_) % mix_num_ != 0)
    KALDI_ERR << "Invalid values input-dim=" << input_dim_
              << " output-dim=" << output_dim_
			  << " mixture number=" << mix_num_;
  KALDI_ASSERT(rand_num_ <= (input_dim_-mix_num_)/output_dim_);
  cfl->GetValue("weight-deriv-scale", &weight_deriv_scale_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
}

void WeightSumBlockRandomComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<WeightSumBlockRandomComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "<MixNum>");
  ReadBasicType(is, binary, &mix_num_);
  ExpectToken(is, binary, "<RandNum>");
  ReadBasicType(is, binary, &rand_num_);
  ExpectToken(is, binary, "<WeightDerivScale>");
  ReadBasicType(is, binary, &weight_deriv_scale_);
  ExpectToken(is, binary, "<RandPerFrame>");
  ReadBasicType(is, binary, &rand_per_frame_);
  ExpectToken(is, binary, "<TestTimeSample>");
  ReadBasicType(is, binary, &test_time_sample_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "</WeightSumBlockRandomComponent>");
}

void WeightSumBlockRandomComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<WeightSumBlockRandomComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "<MixNum>");
  WriteBasicType(os, binary, mix_num_);
  WriteToken(os, binary, "<RandNum>");
  WriteBasicType(os, binary, rand_num_);
  WriteToken(os, binary, "<WeightDerivScale>");
  WriteBasicType(os, binary, weight_deriv_scale_);
  WriteToken(os, binary, "<RandPerFrame>");
  WriteBasicType(os, binary, rand_per_frame_);
  WriteToken(os, binary, "<TestTimeSample>");
  WriteBasicType(os, binary, test_time_sample_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "</WeightSumBlockRandomComponent>");
}

std::string WeightSumBlockRandomComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", scale=" << scale_
		 << ", mix-num=" << mix_num_
		 << ", rand-num=" << rand_num_
		 << ", weight-deriv-scale=" << weight_deriv_scale_
		 << ", test-time-sample=" << (test_time_sample_ ? "true" : "false")
		 << ", rand-per-frame=" << (rand_per_frame_ ? "true" : "false")
		 << ", test-mode=" << (test_mode_ ? "true" : "false");
  return stream.str();
}

void* WeightSumBlockRandomComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(out->NumRows() == in.NumRows() &&
               out->NumCols() == output_dim_ &&
               in.NumCols() == input_dim_);
  
  int32 in_val_dim = input_dim_ - mix_num_;
  int32 mix_num_all = in_val_dim/output_dim_;
  int32 num_per_mix = mix_num_all/mix_num_;
  int32 dim_per_mix = in_val_dim/mix_num_;
  BaseFloat average_scale = mix_num_/mix_num_all;
  
  CuMatrix<BaseFloat> in_val(in.NumRows(), in_val_dim);
  in_val.CopyFromMat(in.ColRange(0, in_val_dim));
  CuSubMatrix<BaseFloat> in_weight(in.ColRange(in_val_dim, mix_num_));
  
  CuMatrix<BaseFloat> One_mix_all(in.NumRows(), mix_num_all);
  One_mix_all.Set(1.0);
  Memo *memo = new Memo;
  
  if (!test_mode_ || test_time_sample_) {
	  average_scale = mix_num_/rand_num_;
	  
	  CuMatrix<BaseFloat> rand_mat(in.NumRows(), mix_num_all);
	  CuVector<BaseFloat> rand_vec(mix_num_all);
	  if (!rand_per_frame_) {
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(&rand_vec);
		  BaseFloat max1;		  
		  for (int32 i=0; i<rand_num_; i++) {
			  max1 = rand_vec.Max(); // the current max
			  rand_vec.ReplaceValue(max1, -1.0); // the max one will be -1
		  }
		  rand_mat.CopyRowsFromVec(rand_vec);
		  rand_mat.ApplyHeaviside(); // the selected will be 0, others 1
		  One_mix_all.AddMat(-1.0, rand_mat); // 1 || 0
	  } else {
		  CuMatrix<BaseFloat> rand_mat_tp(mix_num_all, in.NumRows());
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(&rand_mat_tp);
		  BaseFloat max1;
		  for (int32 j=0; j<in.NumRows(); j++) {
			  rand_vec.CopyColFromMat(rand_mat_tp, j); // for jth row	  
			  for (int32 i=0; i<rand_num_; i++) {
				  max1 = rand_vec.Max(); // the current max
				  rand_vec.ReplaceValue(max1, -1.0); // the max one will be -1
			  }
			  rand_mat_tp.CopyColFromVec(rand_vec, j);
		  }
		  rand_mat.CopyFromMat(rand_mat_tp, kTrans);
		  rand_mat.ApplyHeaviside(); // the selected will be 0, others 1
		  One_mix_all.AddMat(-1.0, rand_mat); // 1 || 0
	  }
  }
  
  memo->rand_mask = One_mix_all;
  
  CuVector<BaseFloat> weight_vec(in.NumRows());
  for (int32 i=0; i<mix_num_; i++) {
	  for (int32 j=0; j<num_per_mix; j++) {
		weight_vec.CopyColFromMat(in_weight, i);
		(in_val.ColRange(i*dim_per_mix+j*output_dim_, output_dim_)).MulRowsVec(weight_vec); // weight_i*in_i
		weight_vec.CopyColFromMat(One_mix_all, i*num_per_mix+j);
		(in_val.ColRange(i*dim_per_mix+j*output_dim_, output_dim_)).MulRowsVec(weight_vec); // weight_i*in_i*mask
	  }
  }
  
  out->AddMatBlocks(scale_*average_scale, in_val, kNoTrans); // sum_i weight_i*in_i*mask/N
  
  return static_cast<void*>(memo);
}

void WeightSumBlockRandomComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv) {
	int32 in_val_dim = input_dim_ - mix_num_;
	int32 mix_num_all = in_val_dim/output_dim_;
	int32 num_per_mix = mix_num_all/mix_num_;
	int32 dim_per_mix = in_val_dim/mix_num_;
	BaseFloat average_scale = mix_num_/rand_num_;
	
	Memo *memo_in = static_cast<Memo*>(memo);
	KALDI_ASSERT(memo_in != NULL && "memo not passed into backprop");
	
	CuMatrix<BaseFloat> One_mix_all = memo_in->rand_mask;
	
	CuMatrix<BaseFloat> in_val(in_value.NumRows(), in_val_dim);
	in_val.CopyFromMat(in_value.ColRange(0, in_val_dim));
	CuSubMatrix<BaseFloat> in_weight(in_value.ColRange(in_val_dim, mix_num_));
	CuSubMatrix<BaseFloat> deriv_in_val(in_deriv->ColRange(0, in_val_dim));
	CuSubMatrix<BaseFloat> deriv_in_weight(in_deriv->ColRange(in_val_dim, mix_num_));
	
	deriv_in_val.AddMatBlocks(scale_*average_scale, out_deriv, kNoTrans); // deriv/N
	in_val.MulElements(deriv_in_val); // [in_i*deriv_i/N]
	
	CuVector<BaseFloat> weight_vec(in_value.NumRows());
	for (int32 i=0; i<mix_num_; i++) {
		for (int32 j=0; j<num_per_mix; j++) {
			weight_vec.CopyColFromMat(in_weight, i);
			(deriv_in_val.ColRange(i*dim_per_mix+j*output_dim_, output_dim_)).MulRowsVec(weight_vec); // weight_i*deriv_i/N
			weight_vec.CopyColFromMat(One_mix_all, i*num_per_mix+j);
			(deriv_in_val.ColRange(i*dim_per_mix+j*output_dim_, output_dim_)).MulRowsVec(weight_vec); // weight_i*deriv_i*mask/N
			(in_val.ColRange(i*dim_per_mix+j*output_dim_, output_dim_)).MulRowsVec(weight_vec); // in_i*deriv_i*mask/N
		}
		weight_vec.AddColSumMat(1.0, in_val.ColRange(i*dim_per_mix, dim_per_mix), 0.0); // sum_j in_i_j*deriv_i_j*mask/N
		deriv_in_weight.CopyColFromVec(weight_vec, i);
	}
	if (weight_deriv_scale_ != 1.0) {
		deriv_in_weight.Scale(weight_deriv_scale_);
	}
  }
}

//////////

void SoftmaxKLComponent::Init(int32 input_dim, int32 output_dim, BaseFloat scale, bool input_frame_scale, bool inv_frame_scale)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  scale_ = scale;
  input_frame_scale_ = input_frame_scale;
  inv_frame_scale_ = inv_frame_scale;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0);
  if (input_frame_scale_) {
	  KALDI_ASSERT(input_dim_ == output_dim_+1);
  } else {
	  KALDI_ASSERT(input_dim_ == output_dim_);
  }
}

void SoftmaxKLComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  BaseFloat scale = 1.0;
  bool input_frame_scale = false;
  bool inv_frame_scale = false;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || output_dim <= 0)
    KALDI_ERR << "Invalid dim for layer of type "
              << Type();
  cfl->GetValue("scale", &scale);
  cfl->GetValue("input-frame-scale", &input_frame_scale);
  cfl->GetValue("inv-frame-scale", &inv_frame_scale);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim, scale, input_frame_scale, inv_frame_scale);
}

void* SoftmaxKLComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  // Apply softmax function to each row of the output...
  // for that row, we do
  // x_i = exp(x_i) / sum_j exp(x_j).
  out->SoftMaxPerRow(in.ColRange(0,output_dim_));

  // This floor on the output helps us deal with
  // almost-zeros in a way that doesn't lead to overflow.
  out->ApplyFloor(1.0e-20);

  return NULL;
}

void SoftmaxKLComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {

  if (in_deriv == NULL)
    return;
  /*
    Note on the derivative of the softmax function: let it be
    p_i = exp(x_i) / sum_i exp_i
    The [matrix-valued] Jacobian of this function is
    diag(p) - p p^T
    Let the derivative vector at the output be e, and at the input be
    d.  We have
    d = diag(p) e - p (p^T e).
    d_i = p_i e_i - p_i (p^T e).
  */
  in_deriv->SetZero();
  (in_deriv->ColRange(0,output_dim_)).DiffSoftmaxPerRow(out_value, out_deriv);

  if (scale_ != 0) {
	  CuMatrix<BaseFloat> plogp(out_value.NumRows(), output_dim_);
	  CuVector<BaseFloat> sumplogp(out_value.NumRows());
	  plogp.CopyFromMat(out_value); // p
	  CuMatrix<BaseFloat> psum = plogp;
	  plogp.ApplyLog(); // logp
	  plogp.MulElements(out_value); // plogp
	  sumplogp.AddColSumMat(1.0, plogp, 0.0); // sum plogp
	  psum.MulRowsVec(sumplogp); // p*sum plogp
	  plogp.AddMat(-1.0, psum); // plogp - p*sum plogp
	  
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == output_dim_+1);
		  CuVector<BaseFloat> frame_scale(in_value.NumRows());
		  frame_scale.CopyColFromMat(in_value, output_dim_);
		  if (inv_frame_scale_) {
			  frame_scale.InvertElements();
		  }
		  plogp.MulRowsVec(frame_scale);
	  }

	  (in_deriv->ColRange(0,output_dim_)).AddMat(-1.0*scale_, plogp);
  }
}

void SoftmaxKLComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SoftmaxKLComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "<InputeFrameScale>");
  ReadBasicType(is, binary, &input_frame_scale_);
  ExpectToken(is, binary, "<InvFrameScale>");
  ReadBasicType(is, binary, &inv_frame_scale_);
  ExpectToken(is, binary, "</SoftmaxKLComponent>");
}

void SoftmaxKLComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SoftmaxKLComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "<InputeFrameScale>");
  WriteBasicType(os, binary, input_frame_scale_);
  WriteToken(os, binary, "<InvFrameScale>");
  WriteBasicType(os, binary, inv_frame_scale_);
  WriteToken(os, binary, "</SoftmaxKLComponent>");
}

std::string SoftmaxKLComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", scale=" << scale_
	<< ", input-frame-scale=" << (input_frame_scale_ ? "true" : "false")
	<< ", inv-frame-scale=" << (inv_frame_scale_ ? "true" : "false");
  return stream.str();
}

//////////

void SumOneKLComponent::Init(int32 input_dim, int32 output_dim, BaseFloat scale, bool input_frame_scale, bool inv_frame_scale)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  scale_ = scale;
  input_frame_scale_ = input_frame_scale;
  inv_frame_scale_ = inv_frame_scale;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0);
  if (input_frame_scale_) {
	  KALDI_ASSERT(input_dim_ == output_dim_+1);
  } else {
	  KALDI_ASSERT(input_dim_ == output_dim_);
  }
}

void SumOneKLComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  BaseFloat scale = 1.0;
  bool input_frame_scale = false;
  bool inv_frame_scale = false;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || output_dim <= 0)
    KALDI_ERR << "Invalid dim for layer of type "
              << Type();
  cfl->GetValue("scale", &scale);
  cfl->GetValue("input-frame-scale", &input_frame_scale);
  cfl->GetValue("inv-frame-scale", &inv_frame_scale);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim, scale, input_frame_scale, inv_frame_scale);
}

void* SumOneKLComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {

  out->CopyFromMat(in.ColRange(0,output_dim_)); // a_m
  out->ApplyFloor(1.0e-20);
  
  CuVector<BaseFloat> suma(in.NumRows());
  suma.AddColSumMat(1.0, *out, 0.0); // sum_m a_m
  out->DivRowsVec(suma); // a_m / sum_m a_m

  return NULL;
}

void SumOneKLComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {

  if (in_deriv == NULL)
    return;

  in_deriv->SetZero();
  (in_deriv->ColRange(0,output_dim_)).CopyFromMat(out_deriv);
  
  CuMatrix<BaseFloat> dp(out_value.NumRows(), output_dim_);
  dp.CopyFromMat(out_deriv);
  dp.MulElements(out_value); // deriv*p
  
  CuVector<BaseFloat> suma(out_value.NumRows());
  suma.AddColSumMat(1.0, dp, 0.0); // sum deriv*p
  (in_deriv->ColRange(0,output_dim_)).AddVecToCols(-1.0, suma, 1.0); // deriv - sum deriv*p
  
  suma.AddColSumMat(1.0, out_value, 0.0); // sum_m a_m
  (in_deriv->ColRange(0,output_dim_)).DivRowsVec(suma); // (deriv - sum deriv*p) / sum_m a_m

  if (scale_ != 0) {
	  CuMatrix<BaseFloat> plogp(out_value.NumRows(), output_dim_);
	  CuVector<BaseFloat> sumplogp(out_value.NumRows());
	  plogp.CopyFromMat(out_value); // p
	  CuMatrix<BaseFloat> psum = plogp;
	  plogp.ApplyLog(); // logp
	  sumplogp.AddColSumMat(1.0, plogp, 0.0); // sum logp
	  psum.MulRowsVec(sumplogp); // p*sum logp
	  plogp.AddMat(-1.0, psum); // logp - p*sum logp
	  plogp.DivRowsVec(suma); // (logp - p*sum logp) / sum_m a_m
	  
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == output_dim_+1);
		  CuVector<BaseFloat> frame_scale(in_value.NumRows());
		  frame_scale.CopyColFromMat(in_value, output_dim_);
		  if (inv_frame_scale_) {
			  frame_scale.InvertElements();
		  }
		  plogp.MulRowsVec(frame_scale);
	  }

	  (in_deriv->ColRange(0,output_dim_)).AddMat(-1.0*scale_, plogp);
  }
}

void SumOneKLComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SumOneKLComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "<InputeFrameScale>");
  ReadBasicType(is, binary, &input_frame_scale_);
  ExpectToken(is, binary, "<InvFrameScale>");
  ReadBasicType(is, binary, &inv_frame_scale_);
  ExpectToken(is, binary, "</SumOneKLComponent>");
}

void SumOneKLComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SumOneKLComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "<InputeFrameScale>");
  WriteBasicType(os, binary, input_frame_scale_);
  WriteToken(os, binary, "<InvFrameScale>");
  WriteBasicType(os, binary, inv_frame_scale_);
  WriteToken(os, binary, "</SumOneKLComponent>");
}

std::string SumOneKLComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", scale=" << scale_
	<< ", input-frame-scale=" << (input_frame_scale_ ? "true" : "false")
	<< ", inv-frame-scale=" << (inv_frame_scale_ ? "true" : "false");
  return stream.str();
}

//////////

void BLHUCGMMComponent::Read(std::istream &is, bool binary) {
  std::string token = ReadUpdatableCommon(is, binary);
  KALDI_ASSERT(token == "");
  ExpectToken(is, binary, "<Params>");
  params_.Read(is, binary);
  ExpectToken(is, binary, "<KLScale>");
  ReadBasicType(is, binary, &KL_scale_);
  ExpectToken(is, binary, "<MeanUpdateScale>");
  ReadBasicType(is, binary, &mean_update_scale_);
  ExpectToken(is, binary, "<StdUpdateScale>");
  ReadBasicType(is, binary, &std_update_scale_);
  ExpectToken(is, binary, "<PriorMeanUpdateScale>");
  ReadBasicType(is, binary, &prior_mean_update_scale_);
  ExpectToken(is, binary, "<PriorStdUpdateScale>");
  ReadBasicType(is, binary, &prior_std_update_scale_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "<ShareStdDim>");
  ReadBasicType(is, binary, &share_std_dim_);
  ExpectToken(is, binary, "<ShareMixSampling>");
  ReadBasicType(is, binary, &share_mix_sampling_);
  ExpectToken(is, binary, "<InputFrameScale>");
  ReadBasicType(is, binary, &input_frame_scale_);
  ExpectToken(is, binary, "<InvFrameScale>");
  ReadBasicType(is, binary, &inv_frame_scale_);
  ExpectToken(is, binary, "<Activation>");
  ReadBasicType(is, binary, &activation_);
  ExpectToken(is, binary, "<MixNum>");
  ReadBasicType(is, binary, &mix_num_);
  ExpectToken(is, binary, "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  if (PeekToken(is, binary) == 'O') {
    ExpectToken(is, binary, "<OrthonormalConstraint>");
    ReadBasicType(is, binary, &orthonormal_constraint_);
  } else {
    orthonormal_constraint_ = 0.0;
  }
  ExpectToken(is, binary, "<UseNaturalGradient>");
  ReadBasicType(is, binary, &use_natural_gradient_);

  // Read various natural-gradient-related configs.
  int32 rank_in,  rank_out, update_period;
  BaseFloat alpha, num_samples_history;
  ExpectToken(is, binary, "<RankInOut>");
  ReadBasicType(is, binary, &rank_in);
  ReadBasicType(is, binary, &rank_out);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha);
  ExpectToken(is, binary, "<NumSamplesHistory>");
  ReadBasicType(is, binary, &num_samples_history);
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period);

  preconditioner_in_.SetAlpha(alpha);
  preconditioner_out_.SetAlpha(alpha);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history);
  preconditioner_in_.SetUpdatePeriod(update_period);
  preconditioner_out_.SetUpdatePeriod(update_period);

  ExpectToken(is, binary, "</BLHUCGMMComponent>");
}

void BLHUCGMMComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  is_gradient_ = false;  // not configurable; there's no reason you'd want this

  InitLearningRatesFromConfig(cfl);

  input_dim_ = -1;
  output_dim_ = -1;
  KL_scale_ = 1.0;
  mean_update_scale_ = 1.0;
  std_update_scale_ = 1.0;
  prior_mean_update_scale_ = 1.0;
  prior_std_update_scale_ = 1.0;
  test_mode_ = false;
  share_std_dim_ = false;
  share_mix_sampling_ = false;
  input_frame_scale_ = false;
  inv_frame_scale_ = false;
  activation_ = 0; // sigmoid=0, relu=1, exp=2, lin=3
  mix_num_ = -1;
  cfl->GetValue("mix-num", &mix_num_);
  cfl->GetValue("KL-scale", &KL_scale_);
  cfl->GetValue("mean-update-scale", &mean_update_scale_);
  cfl->GetValue("std-update-scale", &std_update_scale_);
  cfl->GetValue("prior-mean-update-scale", &prior_mean_update_scale_);
  cfl->GetValue("prior-std-update-scale", &prior_std_update_scale_);
  cfl->GetValue("test-mode", &test_mode_);
  cfl->GetValue("share-std-dim", &share_std_dim_);
  cfl->GetValue("share-mix-sampling", &share_mix_sampling_);
  cfl->GetValue("input-frame-scale", &input_frame_scale_);
  cfl->GetValue("inv-frame-scale", &inv_frame_scale_);
  cfl->GetValue("activation", &activation_);
  if (cfl->GetValue("matrix", &matrix_filename)) {
    ReadKaldiObject(matrix_filename, &params_); // will abort on failure.
    KALDI_ASSERT(params_.NumRows() != 0);
    if (cfl->GetValue("output-dim", &output_dim_))
      KALDI_ASSERT(output_dim_ == params_.NumRows()/2 &&
                   "output-dim mismatch vs. matrix.");
	if (cfl->GetValue("input-dim", &input_dim_))
      KALDI_ASSERT(input_dim_ == params_.NumCols()+2*output_dim_ + (input_frame_scale_?1:0) && mix_num_ == params_.NumCols() &&
                   "input-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim_);
    ok = ok && cfl->GetValue("output-dim", &output_dim_);
    if (!ok)
      KALDI_ERR << "Bad initializer " << cfl->WholeLine();
    KALDI_ASSERT(input_dim_-2*output_dim_-(input_frame_scale_?1:0) == mix_num_);
	BaseFloat mean_mean = 0;
	BaseFloat mean_stddev = 1.0 / std::sqrt(mix_num_);
	BaseFloat std_mean = 1;
	BaseFloat std_stddev = 1.0 / std::sqrt(mix_num_);
    cfl->GetValue("mean-mean", &mean_mean);
	cfl->GetValue("mean-stddev", &mean_stddev);
	cfl->GetValue("std-mean", &std_mean);
	cfl->GetValue("std-stddev", &std_stddev);
    params_.Resize(output_dim_*2, mix_num_);
    KALDI_ASSERT(output_dim_ > 0 && input_dim_ > 0 && mix_num_ > 0 && mean_stddev >= 0.0 & std_stddev >= 0.0);
    params_.SetRandn(); // sets to random normally distributed noise.
    (params_.RowRange(0,output_dim_)).Scale(mean_stddev);
	(params_.RowRange(0,output_dim_)).Add(mean_mean);
	(params_.RowRange(output_dim_,output_dim_)).Scale(std_stddev);
	(params_.RowRange(output_dim_,output_dim_)).Add(std_mean);
  }
  // Read various natural-gradient-related configs.
  int32 rank_in = -1, rank_out = -1, update_period = 4;
  BaseFloat alpha = 4.0,
      num_samples_history = 2000.0;

  use_natural_gradient_ = true;

  cfl->GetValue("num-samples-history", &num_samples_history);
  cfl->GetValue("alpha", &alpha);
  cfl->GetValue("rank-in", &rank_in);
  cfl->GetValue("rank-out", &rank_out);
  cfl->GetValue("update-period", &update_period);
  cfl->GetValue("use-natural-gradient", &use_natural_gradient_);

  if (rank_in < 0)
    rank_in = std::min<int32>(20, (params_.NumCols() + 1) / 2);
  if (rank_out < 0)
    rank_out = std::min<int32>(80, (params_.NumRows() + 1) / 2);

  preconditioner_in_.SetAlpha(alpha);
  preconditioner_out_.SetAlpha(alpha);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history);
  preconditioner_in_.SetUpdatePeriod(update_period);
  preconditioner_out_.SetUpdatePeriod(update_period);

  orthonormal_constraint_ = 0.0;
  cfl->GetValue("orthonormal-constraint", &orthonormal_constraint_);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
}


void BLHUCGMMComponent::Write(std::ostream &os,
                            bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
  WriteToken(os, binary, "<Params>");
  params_.Write(os, binary);
  WriteToken(os, binary, "<KLScale>");
  WriteBasicType(os, binary, KL_scale_);
  WriteToken(os, binary, "<MeanUpdateScale>");
  WriteBasicType(os, binary, mean_update_scale_);
  WriteToken(os, binary, "<StdUpdateScale>");
  WriteBasicType(os, binary, std_update_scale_);
  WriteToken(os, binary, "<PriorMeanUpdateScale>");
  WriteBasicType(os, binary, prior_mean_update_scale_);
  WriteToken(os, binary, "<PriorStdUpdateScale>");
  WriteBasicType(os, binary, prior_std_update_scale_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "<ShareStdDim>");
  WriteBasicType(os, binary, share_std_dim_);
  WriteToken(os, binary, "<ShareMixSampling>");
  WriteBasicType(os, binary, share_mix_sampling_);
  WriteToken(os, binary, "<InputFrameScale>");
  WriteBasicType(os, binary, input_frame_scale_);
  WriteToken(os, binary, "<InvFrameScale>");
  WriteBasicType(os, binary, inv_frame_scale_);
  WriteToken(os, binary, "<Activation>");
  WriteBasicType(os, binary, activation_);
  WriteToken(os, binary, "<MixNum>");
  WriteBasicType(os, binary, mix_num_);
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  if (orthonormal_constraint_ != 0.0) {
    WriteToken(os, binary, "<OrthonormalConstraint>");
    WriteBasicType(os, binary, orthonormal_constraint_);
  }
  WriteToken(os, binary, "<UseNaturalGradient>");
  WriteBasicType(os, binary, use_natural_gradient_);

  int32 rank_in = preconditioner_in_.GetRank(),
      rank_out = preconditioner_out_.GetRank(),
      update_period = preconditioner_in_.GetUpdatePeriod();
  BaseFloat alpha = preconditioner_in_.GetAlpha(),
      num_samples_history = preconditioner_in_.GetNumSamplesHistory();
  WriteToken(os, binary, "<RankInOut>");
  WriteBasicType(os, binary, rank_in);
  WriteBasicType(os, binary, rank_out);
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha);
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, num_samples_history);
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, update_period);
  WriteToken(os, binary, "</BLHUCGMMComponent>");
}

std::string BLHUCGMMComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info();
  CuMatrix<BaseFloat> subparams(output_dim_, mix_num_);
  subparams.CopyFromMat(params_.RowRange(0, output_dim_));
  PrintParameterStats(stream, "mean", subparams,
                      true, // include_mean
                      true, // include_row_norms
                      true, // include_column_norms
                      GetVerboseLevel() >= 2); // include_singular_values
  subparams.CopyFromMat(params_.RowRange(output_dim_, output_dim_));
  PrintParameterStats(stream, "std", subparams,
                      true, // include_mean
                      true, // include_row_norms
                      true, // include_column_norms
                      GetVerboseLevel() >= 2); // include_singular_values
  if (orthonormal_constraint_ != 0.0)
    stream << ", orthonormal-constraint=" << orthonormal_constraint_;
  stream << ", use-natural-gradient="
         << (use_natural_gradient_ ? "true" : "false")
         << ", rank-in=" << preconditioner_in_.GetRank()
         << ", rank-out=" << preconditioner_out_.GetRank()
         << ", num-samples-history="
         << preconditioner_in_.GetNumSamplesHistory()
         << ", update-period=" << preconditioner_in_.GetUpdatePeriod()
         << ", alpha=" << preconditioner_in_.GetAlpha();
  return stream.str();
}

void* BLHUCGMMComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  
  CuMatrix<BaseFloat> post_mean(output_dim_, mix_num_);
  post_mean.CopyFromMat(params_.RowRange(0, output_dim_));
  CuMatrix<BaseFloat> post_std(output_dim_, mix_num_);
  post_std.CopyFromMat(params_.RowRange(output_dim_, output_dim_));
  CuSubMatrix<BaseFloat> in_mix(in.ColRange(0, mix_num_));
  
  Memo *memo = new Memo;
	  
  if (!test_mode_) {
	  if (share_mix_sampling_) {
		  (memo->rand_vec).Resize(output_dim_);
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(&(memo->rand_vec));
		  post_std.MulRowsVec(memo->rand_vec); // std * rand
	  } else {
		  (memo->rand_mat).Resize(output_dim_, mix_num_);
		  const_cast<CuRand<BaseFloat>&>(random_generator_).RandGaussian(&(memo->rand_mat));
		  post_std.MulElements(memo->rand_mat); // std * rand
	  }
	  
	  post_mean.AddMat(1.0, post_std); // mean + std * rand
  }
  
  (memo->LHUC_out) = post_mean;
  if (activation_ == 0) {
	  (memo->LHUC_out).Sigmoid(post_mean);
  } else if (activation_ == 1) {
	  (memo->LHUC_out).ApplyFloor(0.0);
  } else if (activation_ == 2) {
	  (memo->LHUC_out).ApplyCeiling(46.05); // avoid inf
	  (memo->LHUC_out).ApplyExp();
  } // else linear
  
  out->AddMatMat(1.0, in_mix, kNoTrans, (memo->LHUC_out), kTrans, 1.0);
  
  if (!test_mode_) {
	return static_cast<void*>(memo);
  } else {
	return NULL;
  }
}

void BLHUCGMMComponent::Backprop(const std::string &debug_info,
                               const ComponentPrecomputedIndexes *indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> & out_value,
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               void *memo,
                               Component *to_update_in,
                               CuMatrixBase<BaseFloat> *in_deriv) const {
  BLHUCGMMComponent *to_update = dynamic_cast<BLHUCGMMComponent*>(to_update_in);
  
  Memo *memo_in = static_cast<Memo*>(memo);
  KALDI_ASSERT(memo_in != NULL && "memo not passed into backprop");

  CuSubMatrix<BaseFloat> post_mean(params_.RowRange(0, output_dim_));
  CuSubMatrix<BaseFloat> post_std(params_.RowRange(output_dim_, output_dim_));
  CuSubMatrix<BaseFloat> in_mix(in_value.ColRange(0, mix_num_));
  
  CuMatrix<BaseFloat> KL_mat(output_dim_, mix_num_);
  CuMatrix<BaseFloat> KL_post_mean_deriv(output_dim_, mix_num_);
  CuMatrix<BaseFloat> KL_post_std_deriv(output_dim_, mix_num_);
  CuMatrix<BaseFloat> KL_prior_mean_deriv(output_dim_, mix_num_);
  CuMatrix<BaseFloat> KL_prior_std_deriv(output_dim_, mix_num_);
  CuVector<BaseFloat> prior_mean_vec(output_dim_);
  CuVector<BaseFloat> prior_std_vec(output_dim_);
  CuMatrix<BaseFloat> diff(output_dim_, mix_num_);
  CuMatrix<BaseFloat> diff_power_2(output_dim_, mix_num_);
  CuMatrix<BaseFloat> rate(output_dim_, mix_num_);
  CuMatrix<BaseFloat> prior_mean(output_dim_, in_value.NumRows());
  CuMatrix<BaseFloat> prior_std(output_dim_, in_value.NumRows());

  if (KL_scale_ != 0) {
	  prior_mean.CopyFromMat(in_value.ColRange(mix_num_, output_dim_), kTrans);
	  prior_std.CopyFromMat(in_value.ColRange(output_dim_+mix_num_, output_dim_), kTrans);
	  prior_mean_vec.CopyColFromMat(prior_mean, 0); // prior is the same
	  prior_std_vec.CopyColFromMat(prior_std, 0); // prior is the same
	  
	  rate.CopyFromMat(post_std);
	  rate.DivRowsVec(prior_std_vec); // post_std/prior_std
	  CuMatrix<BaseFloat> rate_power_2 = rate;
	  rate_power_2.MulElements(rate); // post_std^2/prior_std^2
	  
	  diff.CopyFromMat(post_mean);
	  diff.AddVecToCols(-1.0, prior_mean_vec, 1.0); // post_mean - prior_mean
	  diff.DivRowsVec(prior_std_vec); // (post_mean - prior_mean)/prior_std
	  
	  diff_power_2.CopyFromMat(diff);
	  diff_power_2.MulElements(diff); // (post_mean - prior_mean)^2/prior_std^2
	  diff_power_2.AddMat(1.0, rate_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2
	  diff_power_2.Add(-1.0); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
	  
	  KL_mat.CopyFromMat(diff_power_2);
	  rate_power_2.Add(1e-20); // keep > 0
	  rate_power_2.ApplyLog(); // log (post_std^2/prior_std^2)
	  KL_mat.AddMat(-1.0, rate_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - log (post_std^2/prior_std^2) - 1
	  KL_mat.Scale(0.5*KL_scale_);
	  
	  KL_post_mean_deriv.CopyFromMat(diff); // (post_mean - prior_mean)/prior_std
	  KL_post_mean_deriv.DivRowsVec(prior_std_vec); // (post_mean - prior_mean)/prior_std^2
	  KL_post_mean_deriv.Scale(KL_scale_);
	  
	  KL_prior_mean_deriv.CopyFromMat(KL_post_mean_deriv);
	  KL_prior_mean_deriv.Scale(-1.0*KL_scale_);
	  
	  KL_post_std_deriv.CopyFromMat(rate); // post_std/prior_std
	  KL_post_std_deriv.DivRowsVec(prior_std_vec); // post_std/prior_std^2
	  CuMatrix<BaseFloat> inv_post_std(post_std);
	  inv_post_std.InvertElements(); // 1/post_std
	  KL_post_std_deriv.AddMat(-1.0, inv_post_std); // post_std/prior_std^2 - 1/post_std
	  KL_post_std_deriv.Scale(KL_scale_);
	  
	  KL_prior_std_deriv.CopyFromMat(diff_power_2); // (post_mean - prior_mean)^2/prior_std^2 + post_std^2/prior_std^2 - 1
	  KL_prior_std_deriv.DivRowsVec(prior_std_vec); // (post_mean - prior_mean)^2/prior_std^3 + post_std^2/prior_std^3 - 1/prior_std
	  KL_prior_std_deriv.Scale(-1.0*KL_scale_);
  }
  
  if (in_deriv) {
	in_deriv->SetZero();
	
	CuSubMatrix<BaseFloat> in_mix_deriv(in_deriv->ColRange(0, mix_num_));
	CuSubMatrix<BaseFloat> in_prior_mean_deriv(in_deriv->ColRange(mix_num_, output_dim_));
	CuSubMatrix<BaseFloat> in_prior_std_deriv(in_deriv->ColRange(output_dim_+mix_num_, output_dim_));
	
	in_mix_deriv.AddMatMat(1.0, out_deriv, kNoTrans, memo_in->LHUC_out, kNoTrans, 1.0);
	
	if (KL_scale_ != 0) {
		CuVector<BaseFloat> KL_vec(mix_num_);
		CuVector<BaseFloat> KL_prior_mean_deriv_vec(output_dim_);
		CuVector<BaseFloat> KL_prior_std_deriv_vec(output_dim_);
		
		KL_vec.AddRowSumMat(1.0, KL_mat, 0.0); // sum_d KL_d
		CuMatrix<BaseFloat> KL_in_mix_deriv(in_value.NumRows(), mix_num_);
		KL_in_mix_deriv.AddVecToRows(1.0, KL_vec, 0.0); // sum_d KL_d
		
		KL_prior_mean_deriv_vec.AddColSumMat(prior_mean_update_scale_, KL_prior_mean_deriv, 0.0); // sum_m
		in_prior_mean_deriv.AddVecToRows(-1.0, KL_prior_mean_deriv_vec, 0); // - sum_m
		
		KL_prior_std_deriv_vec.AddColSumMat(prior_std_update_scale_, KL_prior_std_deriv, 0.0); // sum_m
		in_prior_std_deriv.AddVecToRows(-1.0, KL_prior_std_deriv_vec, 0); // - sum_m
		
		if (input_frame_scale_) {
		  CuVector<BaseFloat> frame_scale(in_value.NumRows());
		  frame_scale.CopyColFromMat(in_value, 2*output_dim_+mix_num_);
		  if (inv_frame_scale_) {
			  frame_scale.InvertElements();
		  }
		  KL_in_mix_deriv.MulRowsVec(frame_scale); // sum_d KL_d/N
		  in_prior_mean_deriv.MulRowsVec(frame_scale); // -sum_m /N
		  in_prior_std_deriv.MulRowsVec(frame_scale); // -sum_m /N
	    }
		
		in_mix_deriv.AddMat(-1.0, KL_in_mix_deriv); // deriv - sum_d KL_d/N
	}
  }

  CuMatrix<BaseFloat> in_gmm_deriv(output_dim_, mix_num_);
  BaseFloat local_lrate;
  if (to_update != NULL) {
    if (!to_update->is_gradient_) {
      CuMatrix<BaseFloat> in_mix_temp(in_mix), out_deriv_temp(out_deriv);
      // These "scale" values get will get multiplied into the learning rate (faster
      // than having the matrices scaled inside the preconditioning code).
      BaseFloat in_scale, out_scale;
      to_update->preconditioner_in_.PreconditionDirections(&in_mix_temp,
                                                           &in_scale);
      to_update->preconditioner_out_.PreconditionDirections(&out_deriv_temp,
                                                            &out_scale);
      local_lrate = in_scale * out_scale * to_update->learning_rate_;

      in_gmm_deriv.AddMatMat(1.0, out_deriv_temp, kTrans, in_mix_temp, kNoTrans, 1.0); // deriv*p_m
    } else {
      local_lrate = to_update->learning_rate_;
	  in_gmm_deriv.AddMatMat(1.0, out_deriv, kTrans, in_mix, kNoTrans, 1.0); // deriv*p_m
    }
	
	CuMatrix<BaseFloat> in_gmm_deriv_post = in_gmm_deriv;
	if (activation_ == 0) {
		in_gmm_deriv_post.DiffSigmoid(memo_in->LHUC_out, in_gmm_deriv); // deriv sigmoid
	} else if (activation_ == 1) {
		in_gmm_deriv_post.Heaviside(memo_in->LHUC_out);
		in_gmm_deriv_post.MulElements(in_gmm_deriv);// deriv relu
	} else if (activation_ == 2) {
		in_gmm_deriv_post.MulElements(memo_in->LHUC_out);// deriv exp
	}
	((to_update->params_).RowRange(0, output_dim_)).AddMat(local_lrate*mean_update_scale_, in_gmm_deriv_post);
	
	if (KL_scale_ != 0) {
		((to_update->params_).RowRange(0, output_dim_)).AddMat(-1.0*local_lrate*mean_update_scale_, KL_post_mean_deriv);
	}
	
	if (!test_mode_) {
		if (share_mix_sampling_) {
			in_gmm_deriv_post.MulRowsVec(memo_in->rand_vec); // std * rand
		} else {
			in_gmm_deriv_post.MulElements(memo_in->rand_mat); // std * rand
		}
		((to_update->params_).RowRange(output_dim_, output_dim_)).AddMat(local_lrate*std_update_scale_, in_gmm_deriv_post);
		
		if (KL_scale_ != 0) {
			((to_update->params_).RowRange(output_dim_, output_dim_)).AddMat(-1.0*local_lrate*mean_update_scale_, KL_post_std_deriv);
		}
	}
  }
}


Component* BLHUCGMMComponent::Copy() const {
  return new BLHUCGMMComponent(*this);
}

BLHUCGMMComponent::BLHUCGMMComponent(
    const BLHUCGMMComponent &other):
    UpdatableComponent(other),
    params_(other.params_),
	KL_scale_(other.KL_scale_),
	mean_update_scale_(other.mean_update_scale_),
	std_update_scale_(other.std_update_scale_),
	prior_mean_update_scale_(other.prior_mean_update_scale_),
	prior_std_update_scale_(other.prior_std_update_scale_),
	test_mode_(other.test_mode_),
	share_std_dim_(other.share_std_dim_),
	share_mix_sampling_(other.share_mix_sampling_),
	input_frame_scale_(other.input_frame_scale_),
	inv_frame_scale_(other.inv_frame_scale_),
	activation_(other.activation_),
	mix_num_(other.mix_num_),
	input_dim_(other.input_dim_),
	output_dim_(other.output_dim_),
    orthonormal_constraint_(other.orthonormal_constraint_),
    use_natural_gradient_(other.use_natural_gradient_),
    preconditioner_in_(other.preconditioner_in_),
    preconditioner_out_(other.preconditioner_out_) { }

BLHUCGMMComponent::BLHUCGMMComponent(const CuMatrix<BaseFloat> &params, BaseFloat KL_scale, 
	BaseFloat mean_update_scale, BaseFloat std_update_scale, BaseFloat prior_mean_update_scale, BaseFloat prior_std_update_scale,
	bool test_mode, bool share_std_dim, bool share_mix_sampling, bool input_frame_scale, bool inv_frame_scale,
	int32 activation, int32 mix_num, int32 input_dim, int32 output_dim):
    params_(params),
	KL_scale_(KL_scale),
	mean_update_scale_(mean_update_scale),
	std_update_scale_(std_update_scale),
	prior_mean_update_scale_(prior_mean_update_scale),
	prior_std_update_scale_(prior_std_update_scale),
	test_mode_(test_mode),
	share_std_dim_(share_std_dim),
	share_mix_sampling_(share_mix_sampling),
	input_frame_scale_(input_frame_scale),
	inv_frame_scale_(inv_frame_scale),
	activation_(activation),
	mix_num_(mix_num),
	input_dim_(input_dim),
	output_dim_(output_dim),
    orthonormal_constraint_(0.0),
    use_natural_gradient_(true) {
  // Set defaults for natural gradient.
  preconditioner_in_.SetRank(40);
  preconditioner_out_.SetRank(80);
  preconditioner_in_.SetUpdatePeriod(4);
  preconditioner_out_.SetUpdatePeriod(4);
  // the component-level defaults of alpha and num_samples_history, at 4.0 and
  // 2000.0, are the same as in the NaturalGradientOnline code, so there is no
  // need to set those here.
}

void BLHUCGMMComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) params_.SetZero();
  else params_.Scale(scale);
}

void BLHUCGMMComponent::Add(BaseFloat alpha, const Component &other_in) {
  const BLHUCGMMComponent *other =
      dynamic_cast<const BLHUCGMMComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  params_.AddMat(alpha, other->params_);
}

void BLHUCGMMComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_params(params_);
  temp_params.SetRandn();
  params_.AddMat(stddev, temp_params);
}
int32 BLHUCGMMComponent::NumParameters() const {
  return params_.NumRows() * params_.NumCols();
}
void BLHUCGMMComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->CopyRowsFromMat(params_);
}
void BLHUCGMMComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  params_.CopyRowsFromVec(params);
}
BaseFloat BLHUCGMMComponent::DotProduct(const UpdatableComponent &other_in) const {
  const BLHUCGMMComponent *other =
      dynamic_cast<const BLHUCGMMComponent*>(&other_in);
  return TraceMatMat(params_, other->params_, kTrans);
}

void BLHUCGMMComponent::FreezeNaturalGradient(bool freeze) {
  preconditioner_in_.Freeze(freeze);
  preconditioner_out_.Freeze(freeze);
}

void BLHUCGMMComponent::ConsolidateMemory() {
  OnlineNaturalGradient temp_in(preconditioner_in_);
  preconditioner_in_.Swap(&temp_in);
  OnlineNaturalGradient temp_out(preconditioner_out_);
  preconditioner_out_.Swap(&temp_out);
}


//////////

void* KLAdaptComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in,
                              CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in.ColRange(0,dim_)); // in = [ log p_sd, p_si ]
  return NULL;
}

void KLAdaptComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &in_value,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             void *memo,
                             Component *to_update, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  
  CuMatrix<BaseFloat> out_deriv_adapt(out_deriv.NumRows(),out_deriv.NumCols());
  out_deriv_adapt.CopyFromMat(out_deriv); // p_lab
  if (SI_scale_ != 0.0) {
	out_deriv_adapt.Scale(1 - SI_scale_); // (1-rho) p_lab
	out_deriv_adapt.AddMat(SI_scale_, in_value.ColRange(dim_,dim_)); // p_adapt = (1-rho) p_lab + rho p_si
  }
  
  in_deriv->SetZero();
  (in_deriv->ColRange(0,dim_)).CopyFromMat(out_deriv_adapt);
}

void KLAdaptComponent::InitFromConfig(ConfigLine *cfl) {
  SI_scale_ = 0.0;
  cfl->GetValue("SI-scale", &SI_scale_);
  if (!cfl->GetValue("dim", &dim_) ||
      dim_ <= 0 || cfl->HasUnusedValues()) {
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  }
}

std::string KLAdaptComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_;
  if (SI_scale_ != 0.0)
    stream << ", backprop-scale=" << SI_scale_;
  return stream.str();
}

void KLAdaptComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<KLAdaptComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<SIScale>");
  WriteBasicType(os, binary, SI_scale_);
  WriteToken(os, binary, "</KLAdaptComponent>");
}

void KLAdaptComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<KLAdaptComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);

  if (PeekToken(is, binary) == 'V') {
    // This is the old format, from when KLAdaptComponent inherited from
    // NonlinearComponent.
    SI_scale_ = 0.0;
    ExpectToken(is, binary, "<ValueAvg>");
    CuVector<BaseFloat> temp_vec;
    temp_vec.Read(is, binary);
    ExpectToken(is, binary, "<DerivAvg>");
    temp_vec.Read(is, binary);
    ExpectToken(is, binary, "<Count>");
    BaseFloat temp_float;
    ReadBasicType(is, binary, &temp_float);
    if (PeekToken(is, binary) == 'O') {
      ExpectToken(is, binary, "<OderivRms>");
      temp_vec.Read(is, binary);
      ExpectToken(is, binary, "<OderivCount>");
      ReadBasicType(is, binary, &temp_float);
    }
    std::string token;
    ReadToken(is, binary, &token);
    if (token[0] != '<') {
      // this should happen only rarely, in case we couldn't push back the
      // '<' to the stream in PeekToken().
      token = '<' + token;
    }
    if (token == "<NumDimsSelfRepaired>") {
      ReadBasicType(is, binary, &temp_float);
      ReadToken(is, binary, &token);
    }
    if (token == "<NumDimsProcessed>") {
      ReadBasicType(is, binary, &temp_float);
      ReadToken(is, binary, &token);
    }
    KALDI_ASSERT(token == "</KLAdaptComponent>");
    return;
  } else {
    ExpectToken(is, binary, "<SIScale>");
    ReadBasicType(is, binary, &SI_scale_);
    ExpectToken(is, binary, "</KLAdaptComponent>");
  }
}


/////////////////////////////////////


GumbelSoftmax::GumbelSoftmax(const GumbelSoftmax &other):
    RandomComponent(other),
    dim_(other.dim_),
    temperature_(other.temperature_),
    apply_log_(other.apply_log_),
	temperature_decrease_(other.temperature_decrease_),
	temperature_decrease_proportion_(other.temperature_decrease_proportion_),
	temperature_decrease_minus_(other.temperature_decrease_minus_) { }

Component* GumbelSoftmax::Copy() const {
  GumbelSoftmax *ans = new GumbelSoftmax(*this);
  return ans;
}

void GumbelSoftmax::Init(int32 dim, BaseFloat temperature,
                            bool apply_log, int32 temperature_decrease, BaseFloat temperature_decrease_proportion, BaseFloat temperature_decrease_minus) {
  temperature_ = temperature;
  apply_log_ = apply_log;
  dim_ = dim;
  temperature_decrease_ = temperature_decrease;
  temperature_decrease_proportion_ = temperature_decrease_proportion;
  temperature_decrease_minus_ = temperature_decrease_minus;
}

void GumbelSoftmax::InitFromConfig(ConfigLine *cfl) {
  int32 dim = 0;
  BaseFloat temperature = 1.0;
  bool apply_log = true;
  test_mode_ = false;
  int32 temperature_decrease = 0;
  BaseFloat temperature_decrease_proportion = 1.0;
  BaseFloat temperature_decrease_minus = 0.0;
  bool ok = cfl->GetValue("dim", &dim) &&
    cfl->GetValue("temperature", &temperature);
  cfl->GetValue("apply-log", &apply_log);
  // It only makes sense to set test-mode in the config for testing purposes.
  cfl->GetValue("test-mode", &test_mode_);
  cfl->GetValue("temperature-decrease", &temperature_decrease);
  cfl->GetValue("temperature-decrease-proportion", &temperature_decrease_proportion);
  cfl->GetValue("temperature-decrease-minus", &temperature_decrease_minus);
    // for this stage, dropout is hard coded in
    // normal mode if not declared in config
  if (!ok || cfl->HasUnusedValues() || dim <= 0 ||
      temperature <= 0.0)
       KALDI_ERR << "Invalid initializer for layer of type "
                 << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(dim, temperature, apply_log, temperature_decrease, temperature_decrease_proportion, temperature_decrease_minus);
}

std::string GumbelSoftmax::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_
         << ", temperature=" << temperature_
         << ", apply-log=" << (apply_log_ ? "true" : "false")
		 << ", temperature-decrease=" << temperature_decrease_
		 << ", temperature-decrease-proportion=" << temperature_decrease_proportion_
		 << ", temperature-decrease-minus=" << temperature_decrease_minus_;
  return stream.str();
}

void* GumbelSoftmax::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(out->NumRows() == in.NumRows() && out->NumCols() == in.NumCols()
               && in.NumCols() == dim_);
  
/*  if (temperature_decrease_ == 1) {
	  temperature_ = temperature_ * temperature_decrease_proportion_;
  } else if (temperature_decrease_ == 2) {
	  temperature_ = temperature_ - temperature_decrease_minus_;
  } */
  
  KALDI_ASSERT(temperature_ > 0.0);
  
  CuMatrix<BaseFloat> G(in.NumRows(), in.NumCols());
  const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(&G); // U ~ U(0,1)
  G.ApplyLog();
  G.Scale(-1.0);
  G.ApplyLog();
  G.Scale(-1.0); // G = -log(-log(U))
  
  CuMatrix<BaseFloat> alpha(in.NumRows(), in.NumCols());
  alpha.CopyFromMat(in);
  
  if (apply_log_) {
	  alpha.ApplyFloor(1e-20);
	  alpha.ApplyLog(); // log(alpha)
  }
  
  alpha.AddMat(1.0, G); // x = log(alpha) + G
  alpha.Scale(1.0/temperature_); // x = (log(alpha) + G)/T
  
  out->SoftMaxPerRow(alpha);
  
  return NULL;
}


void GumbelSoftmax::Backprop(const std::string &debug_info,
                                const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &in_value,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                void *memo,
                                Component *to_update,
                                CuMatrixBase<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(in_value.NumRows() == out_value.NumRows() &&
               in_value.NumCols() == out_value.NumCols());

  KALDI_ASSERT(in_value.NumRows() == out_deriv.NumRows() &&
               in_value.NumCols() == out_deriv.NumCols());
  
  CuMatrix<BaseFloat> alpha_deriv(out_deriv.NumRows(), out_deriv.NumCols());
  alpha_deriv.DiffSoftmaxPerRow(out_value, out_deriv);
  alpha_deriv.Scale(1.0/temperature_); // d Softmax(x)/dx * 1/T
  
  if (apply_log_) {
	  CuMatrix<BaseFloat> alpha(in_value.NumRows(), in_value.NumCols());
	  alpha.CopyFromMat(in_value);
	  alpha.ApplyFloor(1e-20);
	  alpha_deriv.DivElements(alpha); // d Softmax(x)/dx * 1/T * 1/alpha
  }
  
  in_deriv->CopyFromMat(alpha_deriv);
}



void GumbelSoftmax::Read(std::istream &is, bool binary) {
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<GumbelSoftmax>") {
    ReadToken(is, binary, &token);
  }
  KALDI_ASSERT(token == "<Dim>");
  ReadBasicType(is, binary, &dim_);  // read dimension.
  ReadToken(is, binary, &token);
  KALDI_ASSERT(token == "<Temperature>");
  ReadBasicType(is, binary, &temperature_);  // read dropout rate
  ExpectToken(is, binary, "<TemperatureDescrease>");
  ReadBasicType(is, binary, &temperature_decrease_);
  ExpectToken(is, binary, "<TemperatureDescreaseProportion>");
  ReadBasicType(is, binary, &temperature_decrease_proportion_);
  ExpectToken(is, binary, "<TemperatureDescreaseMinus>");
  ReadBasicType(is, binary, &temperature_decrease_minus_);
  ReadToken(is, binary, &token);
  if (token == "<ApplyLog>") {
    ReadBasicType(is, binary, &apply_log_);  // read dropout mode
    ReadToken(is, binary, &token);
  } else {
    apply_log_ = false;
  }
  if (token == "<TestMode>") {
    ReadBasicType(is, binary, &test_mode_);  // read test mode
    ExpectToken(is, binary, "</GumbelSoftmax>");
  } else {
    test_mode_ = false;
    KALDI_ASSERT(token == "</GumbelSoftmax>");
  }
}

void GumbelSoftmax::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<GumbelSoftmax>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<Temperature>");
  WriteBasicType(os, binary, temperature_);
  WriteToken(os, binary, "<TemperatureDescrease>");
  WriteBasicType(os, binary, temperature_decrease_);
  WriteToken(os, binary, "<TemperatureDescreaseProportion>");
  WriteBasicType(os, binary, temperature_decrease_proportion_);
  WriteToken(os, binary, "<TemperatureDescreaseMinus>");
  WriteBasicType(os, binary, temperature_decrease_minus_);
  WriteToken(os, binary, "<ApplyLog>");
  WriteBasicType(os, binary, apply_log_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "</GumbelSoftmax>");
}


//////////

void SoftmaxKLPriorComponent::Init(int32 input_dim, int32 output_dim, BaseFloat scale, bool input_frame_scale, bool inv_frame_scale)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  scale_ = scale;
  input_frame_scale_ = input_frame_scale;
  inv_frame_scale_ = inv_frame_scale;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0);
  if (input_frame_scale_) {
	  KALDI_ASSERT(input_dim_ == 2*output_dim_+1);
  } else {
	  KALDI_ASSERT(input_dim_ == 2*output_dim_);
  }
}

void SoftmaxKLPriorComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  BaseFloat scale = 1.0;
  bool input_frame_scale = false;
  bool inv_frame_scale = false;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || output_dim <= 0)
    KALDI_ERR << "Invalid dim for layer of type "
              << Type();
  cfl->GetValue("scale", &scale);
  cfl->GetValue("input-frame-scale", &input_frame_scale);
  cfl->GetValue("inv-frame-scale", &inv_frame_scale);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim, scale, input_frame_scale, inv_frame_scale);
}

void* SoftmaxKLPriorComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  // Apply softmax function to each row of the output...
  // for that row, we do
  // x_i = exp(x_i) / sum_j exp(x_j).
  out->SoftMaxPerRow(in.ColRange(0,output_dim_));

  // This floor on the output helps us deal with
  // almost-zeros in a way that doesn't lead to overflow.
  out->ApplyFloor(1.0e-20);

  return NULL;
}

void SoftmaxKLPriorComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {

  if (in_deriv == NULL)
    return;
  /*
    Note on the derivative of the softmax function: let it be
    p_i = exp(x_i) / sum_i exp_i
    The [matrix-valued] Jacobian of this function is
    diag(p) - p p^T
    Let the derivative vector at the output be e, and at the input be
    d.  We have
    d = diag(p) e - p (p^T e).
    d_i = p_i e_i - p_i (p^T e).
  */
  in_deriv->SetZero();
  (in_deriv->ColRange(0,output_dim_)).DiffSoftmaxPerRow(out_value, out_deriv);

  if (scale_ != 0) {
	  CuMatrix<BaseFloat> plogp(out_value.NumRows(), output_dim_);
	  CuVector<BaseFloat> sumplogp(out_value.NumRows());
	  plogp.CopyFromMat(out_value); // p
	  CuMatrix<BaseFloat> psum = plogp;
	  plogp.ApplyLog(); // logp
	  plogp.MulElements(out_value); // plogp
	  sumplogp.AddColSumMat(1.0, plogp, 0.0); // sum plogp
	  psum.MulRowsVec(sumplogp); // p*sum plogp
	  plogp.AddMat(-1.0, psum); // plogp - p*sum plogp
	  
	  CuSubMatrix<BaseFloat> log_prior(in_value.ColRange(output_dim_,output_dim_)); // log p_0
	  
	  CuMatrix<BaseFloat> deriv_add(out_value.NumRows(), output_dim_);
	  deriv_add.DiffSoftmaxPerRow(out_value, log_prior); // sum_k log p_0_k deriv_softmax_k
	  plogp.AddMat(-1.0, deriv_add); // dKL/dp = plogp - p*sum plogp - sum_k log p_0_k deriv_softmax_k
	  
	  CuMatrix<BaseFloat> deriv_prior(out_value.NumRows(), output_dim_);
	  deriv_prior.CopyFromMat(out_value);
	  deriv_prior.Scale(-1.0); // dKL/d log p_0 = -p
	  
	  if (input_frame_scale_) {
		  KALDI_ASSERT(input_dim_ == 2*output_dim_+1);
		  CuVector<BaseFloat> frame_scale(in_value.NumRows());
		  frame_scale.CopyColFromMat(in_value, 2*output_dim_);
		  if (inv_frame_scale_) {
			  frame_scale.InvertElements();
		  }
		  plogp.MulRowsVec(frame_scale);
		  deriv_prior.MulRowsVec(frame_scale);
	  }

	  (in_deriv->ColRange(0,output_dim_)).AddMat(-1.0*scale_, plogp);
	  (in_deriv->ColRange(output_dim_,output_dim_)).AddMat(-1.0*scale_, deriv_prior);
  }
}

void SoftmaxKLPriorComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SoftmaxKLPriorComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "<InputeFrameScale>");
  ReadBasicType(is, binary, &input_frame_scale_);
  ExpectToken(is, binary, "<InvFrameScale>");
  ReadBasicType(is, binary, &inv_frame_scale_);
  ExpectToken(is, binary, "</SoftmaxKLPriorComponent>");
}

void SoftmaxKLPriorComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SoftmaxKLPriorComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "<InputeFrameScale>");
  WriteBasicType(os, binary, input_frame_scale_);
  WriteToken(os, binary, "<InvFrameScale>");
  WriteBasicType(os, binary, inv_frame_scale_);
  WriteToken(os, binary, "</SoftmaxKLPriorComponent>");
}

std::string SoftmaxKLPriorComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_
         << ", scale=" << scale_
	<< ", input-frame-scale=" << (input_frame_scale_ ? "true" : "false")
	<< ", inv-frame-scale=" << (inv_frame_scale_ ? "true" : "false");
  return stream.str();
}

//////////

void PriorSelectKLGaussianComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    // If scale == 0.0 we call SetZero() which will get rid of NaN's and inf's.
    linear_params_.SetZero();
    bias_params_.SetZero();
  } else {
    linear_params_.Scale(scale);
    bias_params_.Scale(scale);
  }
}

void PriorSelectKLGaussianComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  bias_params_.Resize(output_dim);
  linear_params_.Resize(output_dim, input_dim);
}

void PriorSelectKLGaussianComponent::Add(BaseFloat alpha, const Component &other_in) {
  const PriorSelectKLGaussianComponent *other =
      dynamic_cast<const PriorSelectKLGaussianComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

PriorSelectKLGaussianComponent::PriorSelectKLGaussianComponent(const PriorSelectKLGaussianComponent &component):
    UpdatableComponent(component),
    linear_params_(component.linear_params_),
    bias_params_(component.bias_params_),
    orthonormal_constraint_(component.orthonormal_constraint_),
	mix_num_(component.mix_num_),
	input_mix_num_(component.input_mix_num_),
	output_weight_(component.output_weight_),
	input_dim_(component.input_dim_),
	output_dim_(component.output_dim_) { }

PriorSelectKLGaussianComponent::PriorSelectKLGaussianComponent(const CuMatrixBase<BaseFloat> &linear_params,
                                 const CuVectorBase<BaseFloat> &bias_params,
                                 BaseFloat learning_rate):
    linear_params_(linear_params),
    bias_params_(bias_params),
    orthonormal_constraint_(0.0),
	mix_num_(1),
	input_mix_num_(1),
	output_weight_(false),
	input_dim_(0),
	output_dim_(0) {
  SetUnderlyingLearningRate(learning_rate);
  KALDI_ASSERT(linear_params.NumRows() == bias_params.Dim()&&
               bias_params.Dim() != 0);
}

void PriorSelectKLGaussianComponent::SetParams(const CuVectorBase<BaseFloat> &bias,
                                const CuMatrixBase<BaseFloat> &linear) {
  bias_params_ = bias;
  linear_params_ = linear;
  KALDI_ASSERT(bias_params_.Dim() == linear_params_.NumRows());
}

void PriorSelectKLGaussianComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);

  CuVector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

std::string PriorSelectKLGaussianComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info();
  if (orthonormal_constraint_ != 0.0)
    stream << ", orthonormal-constraint=" << orthonormal_constraint_;
  int32 mix_dim = linear_params_.NumCols()/2;
  stream << ", mix-num=" << mix_num_
		   << ", input-mix-num=" << input_mix_num_
		   << ", output-weight=" << output_weight_
		   << ", input-dim=" << input_dim_
		   << ", output-dim=" << input_dim_
		   << ", mix-dim=" << mix_dim;
  CuMatrix<BaseFloat> temp(linear_params_.NumRows(),mix_dim);
  temp.CopyFromMat(linear_params_.ColRange(0,mix_dim));
  PrintParameterStats(stream, "mean-params", temp,
                      true, // include_mean
                      true, // include_row_norms
                      true, // include_column_norms
                      GetVerboseLevel() >= 2); // include_singular_values
  temp.CopyFromMat(linear_params_.ColRange(mix_dim,mix_dim));
  PrintParameterStats(stream, "std-params", temp,
                      true, // include_mean
                      true, // include_row_norms
                      true, // include_column_norms
                      GetVerboseLevel() >= 2); // include_singular_values
  PrintParameterStats(stream, "bias", bias_params_, true);
  return stream.str();
}

Component* PriorSelectKLGaussianComponent::Copy() const {
  PriorSelectKLGaussianComponent *ans = new PriorSelectKLGaussianComponent(*this);
  return ans;
}

BaseFloat PriorSelectKLGaussianComponent::DotProduct(const UpdatableComponent &other_in) const {
  const PriorSelectKLGaussianComponent *other =
      dynamic_cast<const PriorSelectKLGaussianComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans)
      + VecVec(bias_params_, other->bias_params_);
}

void PriorSelectKLGaussianComponent::Init(int32 mix_dim, int32 mix_num_,
                           BaseFloat param_stddev, BaseFloat bias_stddev, BaseFloat mean_init, BaseFloat std_init) {
  linear_params_.Resize(mix_num_, 2*mix_dim);
  bias_params_.Resize(mix_num_);
  KALDI_ASSERT(mix_num_ > 0 && mix_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  (linear_params_.ColRange(1,mix_dim)).Add(mean_init);
  (linear_params_.ColRange(mix_dim,mix_dim)).Add(std_init);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
}

void PriorSelectKLGaussianComponent::Init(std::string matrix_filename) {
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  bias_params_.CopyColFromMat(mat, input_dim);
}

void PriorSelectKLGaussianComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  input_dim_ = -1, output_dim_ = -1;
  ok = ok && cfl->GetValue("input-dim", &input_dim_);
  ok = ok && cfl->GetValue("output-dim", &output_dim_);
  InitLearningRatesFromConfig(cfl);
  ok = ok && cfl->GetValue("mix-num", &mix_num_);
  ok = ok && cfl->GetValue("input-mix-num", &input_mix_num_);
  ok = ok && cfl->GetValue("output-weight", &output_weight_);
  int32 mix_dim;
  if (output_weight_) {
	  KALDI_ASSERT(input_dim_ == output_dim_-input_mix_num_ &&
                   "input-dim and output-dim mismatch.");
  } else {
	  KALDI_ASSERT(input_dim_ == output_dim_ &&
                   "input-dim and output-dim mismatch.");
  }
  KALDI_ASSERT(input_dim_ % (2*input_mix_num_) == 0);
  mix_dim = input_dim_/(2*input_mix_num_);
  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(matrix_filename);
    KALDI_ASSERT(linear_params_.NumCols()/2 == mix_dim &&
                   "mix-dim mismatch.");
    KALDI_ASSERT(linear_params_.NumRows() == mix_num_ &&
                   "mix-num mismatch.");
  } else {
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim_),
        bias_stddev = 1.0,
		mean_init = 0.0,
		std_init = 1.0;
    cfl->GetValue("param-stddev", &param_stddev);
	cfl->GetValue("mean-init", &mean_init);
	cfl->GetValue("std-init", &std_init);
    cfl->GetValue("bias-stddev", &bias_stddev);
    Init(mix_dim, mix_num_,
         param_stddev, bias_stddev, mean_init, std_init);
  }
  cfl->GetValue("orthonormal-constraint", &orthonormal_constraint_);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}




void* PriorSelectKLGaussianComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {

  Memo *memo = new Memo;
  int32 mix_dim = linear_params_.NumCols()/2;
  int32 input_mix_dim = input_dim_/input_mix_num_;
  int32 output_mix_dim = output_dim_/input_mix_num_;
  
  CuMatrix<BaseFloat> prior_mean(mix_num_,mix_dim);
  prior_mean.CopyFromMat(linear_params_.ColRange(0,mix_dim)); // u_0
  
  CuMatrix<BaseFloat> prior_mean_power2 = prior_mean;
  prior_mean_power2.MulElements(prior_mean); // u_0^2
  
  CuMatrix<BaseFloat> inv_prior_var(mix_num_,mix_dim);
  inv_prior_var.CopyFromMat(linear_params_.ColRange(mix_dim,mix_dim)); // std_0
  inv_prior_var.MulElements(inv_prior_var); // std_0^2
  inv_prior_var.ApplyFloor(1e-20);
  
  CuMatrix<BaseFloat> log_prior_var = inv_prior_var;
  log_prior_var.ApplyLog(); // log std_0^2
  
  inv_prior_var.InvertElements(); // 1/std_0^2
  
  prior_mean.MulElements(inv_prior_var); // u_0/std_0^2
  prior_mean_power2.MulElements(inv_prior_var); // u_0^2/std_0^2
  
  CuVector<BaseFloat> add_vec_prior(mix_num_);
  add_vec_prior.AddColSumMat(1.0, prior_mean_power2, 0.0); // sum_d u_0^2/std_0^2
  add_vec_prior.AddColSumMat(1.0, log_prior_var, 1.0); // sum_d u_0^2/std_0^2 + sum_d log std_0^2
  
  CuVector<BaseFloat> log_mix_weight(mix_num_);
  log_mix_weight.CopyFromVec(bias_params_);
  log_mix_weight.ApplySoftMax(); // pi_0
  log_mix_weight.ApplyFloor(1e-20);
  log_mix_weight.ApplyLog(); // log pi_0
  
  int32 N = in.NumRows();
  CuMatrix<BaseFloat> post_mean(N,mix_dim);
  CuMatrix<BaseFloat> post_mean_power2(N,mix_dim);
  CuMatrix<BaseFloat> post_var(N,mix_dim);
  CuMatrix<BaseFloat> log_post_var(N,mix_dim);
  
  CuMatrix<BaseFloat> select_val(N, mix_num_);
  CuVector<BaseFloat> add_vec_post(N);
  
  (memo->select_id).Resize(N,mix_num_*input_mix_num_);
  
  for (int32 i=0; i<input_mix_num_; i++) {
	  int32 start_pos = i*input_mix_dim;
	  
	  post_mean.CopyFromMat(in.ColRange(start_pos,mix_dim)); // u_s
	  post_mean_power2.CopyFromMat(post_mean);
	  post_mean_power2.MulElements(post_mean_power2); // u_s^2
	  
	  post_var.CopyFromMat(in.ColRange(start_pos+mix_dim,mix_dim)); // std_s
	  post_var.MulElements(post_var); // std_s^2
	  log_post_var.CopyFromMat(post_var);
	  log_post_var.ApplyFloor(1e-20);
	  log_post_var.ApplyLog(); // log std_s^2
	  add_vec_post.AddColSumMat(1.0, log_post_var, 0.0); // sum_d log std_s^2
	  
	  select_val.AddMatMat(0.5, post_mean_power2, kNoTrans, inv_prior_var, kTrans, 0.0); // 0.5 sum_d u_s^2/std_0^2
	  select_val.AddMatMat(-1.0, post_mean, kNoTrans, prior_mean, kTrans, 1.0); // 0.5 sum_d u_s^2/std_0^2 - sum_d u_s u_0/std_0^2
	  select_val.AddMatMat(0.5, post_var, kNoTrans, inv_prior_var, kTrans, 1.0); // 0.5 sum_d u_s^2/std_0^2 - sum_d u_s u_0/std_0^2 + 0.5 sum_d std_s^2/std_0^2
	  select_val.AddVecToCols(-0.5, add_vec_post, 1.0); // 0.5 sum_d u_s^2/std_0^2 - sum_d u_s u_0/std_0^2 + 0.5 sum_d std_s^2/std_0^2 - 0.5 sum_d log std_s^2
	  select_val.AddVecToRows(0.5, add_vec_prior, 1.0); // 0.5 sum_d u_s^2/std_0^2 - sum_d u_s u_0/std_0^2 + 0.5 sum_d std_s^2/std_0^2 - 0.5 sum_d log std_s^2 + 0.5 sum_d u_0^2/std_0^2 + 0.5 sum_d log std_0^2
	  select_val.Add(- mix_dim); // KL = 0.5 sum_d u_s^2/std_0^2 - sum_d u_s u_0/std_0^2 + 0.5 sum_d std_s^2/std_0^2 - 0.5 sum_d log std_s^2 + 0.5 sum_d u_0^2/std_0^2 + 0.5 sum_d log std_0^2 - D
	  
	  select_val.AddVecToRows(-1.0, log_mix_weight, 1.0); // KL - log pi_0
	  
	  for (int32 n=0; n<N; n++) {
		  CuSubMatrix<BaseFloat> select_val_n(select_val.RowRange(n,1));
		  BaseFloat min = select_val_n.Min();
		  select_val_n.Add(- min); // y = x - min >= 0, y(min_id) = 0
	  }
	  
	  CuSubMatrix<BaseFloat> select_id_i((memo->select_id).ColRange(i*mix_num_,mix_num_));
	  select_id_i.Heaviside(select_val); // select_id = [id1, id2, ...] x>0? 1: 0
	  select_id_i.Add(-1.0); // x>0? 0: -1
	  select_id_i.Scale(-1.0); // x>0? 0: 1
	  
	  add_vec_post.AddColSumMat(1.0, select_id_i, 0.0); // sum(id)
	  select_id_i.DivRowsVec(add_vec_post); // ensure to sum to 1
	  
	  int32 start_pos_out = i*output_mix_dim;
	  
	  (out->ColRange(start_pos_out,2*mix_dim)).AddMatMat(1.0, select_id_i, kNoTrans, linear_params_, kNoTrans, 0.0);
	  
	  if (output_weight_) {
		  add_vec_post.AddMatVec(1.0, select_id_i, kNoTrans, bias_params_, 0.0);
		  out->CopyColFromVec(add_vec_post, start_pos_out+2*mix_dim);
	  }
  }
  
  return static_cast<void*>(memo);
}

void PriorSelectKLGaussianComponent::UpdateSimple(const CuMatrixBase<BaseFloat> &select_id,
                                   const CuMatrixBase<BaseFloat> &out_deriv) {
  
  int32 mix_dim = linear_params_.NumCols()/2;
  int32 output_mix_dim = output_dim_/input_mix_num_;
  CuMatrix<BaseFloat> gradient(mix_num_, output_mix_dim);
  gradient.SetZero();
  
  for (int32 i=0; i<input_mix_num_; i++) {
	  gradient.AddMatMat(1.0, select_id.ColRange(i*mix_num_,mix_num_), kTrans, out_deriv.ColRange(i*output_mix_dim,output_mix_dim), kNoTrans, 1.0);
  }
  
  linear_params_.AddMat(learning_rate_, gradient.ColRange(0, 2*mix_dim));
  if (output_weight_) {
	  bias_params_.AddColSumMat(learning_rate_, gradient.ColRange(2*mix_dim, 1), 1.0);
  }
}

void PriorSelectKLGaussianComponent::Backprop(const std::string &debug_info,
                               const ComponentPrecomputedIndexes *indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               void *memo,
                               Component *to_update_in,
                               CuMatrixBase<BaseFloat> *in_deriv) const {
  PriorSelectKLGaussianComponent *to_update = dynamic_cast<PriorSelectKLGaussianComponent*>(to_update_in);

  // Propagate the derivative back to the input.
  // add with coefficient 1.0 since property kBackpropAdds is true.
  // If we wanted to add with coefficient 0.0 we'd need to zero the
  // in_deriv, in case of infinities.
  
  Memo *memo_in = static_cast<Memo*>(memo);
  KALDI_ASSERT(memo_in != NULL && "memo not passed into backprop");
  
  in_deriv->SetZero();

  if (to_update != NULL) {
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    if (to_update->is_gradient_)
      to_update->UpdateSimple(memo_in->select_id, out_deriv);
    else  // the call below is to a virtual function that may be re-implemented
      to_update->Update(debug_info, memo_in->select_id, out_deriv);  // by child classes.
  }
}

void PriorSelectKLGaussianComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<MixNum>");
  ReadBasicType(is, binary, &mix_num_);
  ExpectToken(is, binary, "<InputMixNum>");
  ReadBasicType(is, binary, &input_mix_num_);
  ExpectToken(is, binary, "<OutputWeight>");
  ReadBasicType(is, binary, &output_weight_);
  ExpectToken(is, binary, "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  if (PeekToken(is, binary) == 'I') {
    // for back compatibility; we don't write this here any
    // more as it's written and read in Write/ReadUpdatableCommon
    ExpectToken(is, binary, "<IsGradient>");
    ReadBasicType(is, binary, &is_gradient_);
  }
  if (PeekToken(is, binary) == 'O') {
    ExpectToken(is, binary, "<OrthonormalConstraint>");
    ReadBasicType(is, binary, &orthonormal_constraint_);
  } else {
    orthonormal_constraint_ = 0.0;
  }
  ExpectToken(is, binary, "</PriorSelectKLGaussianComponent>");
}

void PriorSelectKLGaussianComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<MixNum>");
  WriteBasicType(os, binary, mix_num_);
  WriteToken(os, binary, "<InputMixNum>");
  WriteBasicType(os, binary, input_mix_num_);
  WriteToken(os, binary, "<OutputWeight>");
  WriteBasicType(os, binary, output_weight_);
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  if (orthonormal_constraint_ != 0.0) {
    WriteToken(os, binary, "<OrthonormalConstraint>");
    WriteBasicType(os, binary, orthonormal_constraint_);
  }
  WriteToken(os, binary, "</PriorSelectKLGaussianComponent>");
}

int32 PriorSelectKLGaussianComponent::NumParameters() const {
  return (InputDim() + 1) * OutputDim();
}
void PriorSelectKLGaussianComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->Range(0, InputDim() * OutputDim()).CopyRowsFromMat(linear_params_);
  params->Range(InputDim() * OutputDim(),
                OutputDim()).CopyFromVec(bias_params_);
}
void PriorSelectKLGaussianComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
  bias_params_.CopyFromVec(params.Range(InputDim() * OutputDim(),
                                        OutputDim()));
}


////////////////////////////////////////


void* InvertComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
  out->ApplyFloor(1e-20);
  out->InvertElements();
  return NULL;
}

void InvertComponent::Backprop(const std::string &debug_info,
                                const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                void *memo,
                                Component *to_update_in,
                                CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
    in_deriv->CopyFromMat(out_value); // in^-1
	in_deriv->MulElements(*in_deriv); // in^-2
	in_deriv->Scale(-1.0); // -in^-2
	in_deriv->MulElements(out_deriv); // de/dout * -in^-2
	
    InvertComponent *to_update = dynamic_cast<InvertComponent*>(to_update_in);
    if (to_update != NULL) {
      RepairGradients(out_value, in_deriv, to_update);
      to_update->StoreBackpropStats(out_deriv);
    }
  }
}

void InvertComponent::RepairGradients(
    const CuMatrixBase<BaseFloat> &out_value,
    CuMatrixBase<BaseFloat> *in_deriv,
    InvertComponent *to_update) const {}



void InvertComponent::StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_value,
                                  void *memo) {}


////////////////////////////////////////////////////////


void* SpecMaskOnlineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in,
                              CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
  
  
	int32 dim = in.NumCols();
	int32 num_row = in.NumRows();
	
	CuVector<BaseFloat> mean(dim);
	mean.AddRowSumMat(1.0/num_row, in, 0.0);
	
	BaseFloat p;
	
	if (mask_filter_) {
		int32 f1, f0;
		for (int32 i=0; i<num_filter_mask_; i++) {
			p =  rand()/(RAND_MAX + 1);
			if (p < mask_prob_) {
				f1 = (rand() % (width_filter_max_))+1; // f1 \in (0, width_filter_max_]
				f0 = rand() % (dim-f1); // f0 \in [0, dim-f1)
				// KALDI_LOG << "Setting f0 = " << f0 << " and f1 = " << f1 << ".\n";
				if (mask_as_mean_) {
					(out->ColRange(f0,f1)).AddVecToRows(1.0, mean.Range(f0,f1), 0.0);
				} else {
					(out->ColRange(f0,f1)).Set(mask_value_);
				}
			}
		}
	}
	
	if (mask_time_) {
		int32 t1, t0;
		int32 max_mask_time = floor(rate_time_max_ * num_row);
		for (int32 i=0; i<num_time_mask_; i++) {
			p =  rand()/(RAND_MAX + 1);
			if (p < mask_prob_) {
				t1 = (rand() % (width_time_max_))+1; // t1 \in (0, width_time_max_]
				t1 = (t1>max_mask_time)?max_mask_time:t1; // t1 <= max_mask_time
				t0 = rand() % (num_row-t1); // t0 \in [0, num_row-t1)
				// KALDI_LOG << "Setting t0 = " << t0 << " and t1 = " << t1 << ".\n";
				if (mask_as_mean_) {
					(out->RowRange(t0,t1)).AddVecToRows(1.0, mean, 0.0);
				} else {
					(out->RowRange(t0,t1)).Set(mask_value_);
				}
			}
		}
	}
  
  return NULL;
}

void SpecMaskOnlineComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             void *memo,
                             Component *to_update, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  in_deriv->CopyFromMat(out_deriv);
}

void SpecMaskOnlineComponent::InitFromConfig(ConfigLine *cfl) {
  width_filter_max_ = 15;
	width_time_max_ = 70;
	num_filter_mask_ = 2;
	num_time_mask_ = 2;
	rate_time_max_ = 0.2;
	mask_value_ = 0.0;
	mask_prob_ = 1.0;
	mask_filter_ = true;
	mask_time_ = true;
	mask_as_mean_ = true;
  cfl->GetValue("width-filter-max", &width_filter_max_);
  cfl->GetValue("width-time-max", &width_time_max_);
  cfl->GetValue("num-filter-mask", &num_filter_mask_);
  cfl->GetValue("num-time-mask", &num_time_mask_);
  cfl->GetValue("rate-time-max", &rate_time_max_);
  cfl->GetValue("mask-value", &mask_value_);
  cfl->GetValue("mask-prob", &mask_prob_);
  cfl->GetValue("mask-filter", &mask_filter_);
  cfl->GetValue("mask-time", &mask_time_);
  cfl->GetValue("mask-as-mean", &mask_as_mean_);
  if (!cfl->GetValue("dim", &dim_) ||
      dim_ <= 0 || cfl->HasUnusedValues()) {
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  }
}

std::string SpecMaskOnlineComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_;
  stream << ", width-filter-max=" << width_filter_max_;
  stream << ", width-time-max=" << width_time_max_;
  stream << ", num-filter-mask=" << num_filter_mask_;
  stream << ", num-time-mask=" << num_time_mask_;
  stream << ", rate-time-max=" << rate_time_max_;
  stream << ", mask-value=" << mask_value_;
  stream << ", mask-prob=" << mask_prob_;
  stream << ", mask-filter=" << mask_filter_;
  stream << ", mask-time=" << mask_time_;
  stream << ", mask-as-mean=" << mask_as_mean_;
  return stream.str();
}

void SpecMaskOnlineComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SpecMaskOnlineComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<WidthFilterMax>");
  WriteBasicType(os, binary, width_filter_max_);
  WriteToken(os, binary, "<WidthTimeMax>");
  WriteBasicType(os, binary, width_time_max_);
  WriteToken(os, binary, "<NumFilterMask>");
  WriteBasicType(os, binary, num_filter_mask_);
  WriteToken(os, binary, "<NumTimeMask>");
  WriteBasicType(os, binary, num_time_mask_);
  WriteToken(os, binary, "<RateTimeMax>");
  WriteBasicType(os, binary, rate_time_max_);
  WriteToken(os, binary, "<MaskValue>");
  WriteBasicType(os, binary, mask_value_);
  WriteToken(os, binary, "<MaskProb>");
  WriteBasicType(os, binary, mask_prob_);
  WriteToken(os, binary, "<MaskFilter>");
  WriteBasicType(os, binary, mask_filter_);
  WriteToken(os, binary, "<MaskTime>");
  WriteBasicType(os, binary, mask_time_);
  WriteToken(os, binary, "<MaskAsMean>");
  WriteBasicType(os, binary, mask_as_mean_);
  WriteToken(os, binary, "</SpecMaskOnlineComponent>");
}

void SpecMaskOnlineComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SpecMaskOnlineComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);

  if (PeekToken(is, binary) == 'V') {
    // This is the old format, from when SpecMaskOnlineComponent inherited from
    // NonlinearComponent.
    width_filter_max_ = 15;
	width_time_max_ = 70;
	num_filter_mask_ = 2;
	num_time_mask_ = 2;
	rate_time_max_ = 0.2;
	mask_value_ = 0.0;
	mask_prob_ = 1.0;
	mask_filter_ = true;
	mask_time_ = true;
	mask_as_mean_ = true;
    ExpectToken(is, binary, "<ValueAvg>");
    CuVector<BaseFloat> temp_vec;
    temp_vec.Read(is, binary);
    ExpectToken(is, binary, "<DerivAvg>");
    temp_vec.Read(is, binary);
    ExpectToken(is, binary, "<Count>");
    BaseFloat temp_float;
    ReadBasicType(is, binary, &temp_float);
    if (PeekToken(is, binary) == 'O') {
      ExpectToken(is, binary, "<OderivRms>");
      temp_vec.Read(is, binary);
      ExpectToken(is, binary, "<OderivCount>");
      ReadBasicType(is, binary, &temp_float);
    }
    std::string token;
    ReadToken(is, binary, &token);
    if (token[0] != '<') {
      // this should happen only rarely, in case we couldn't push back the
      // '<' to the stream in PeekToken().
      token = '<' + token;
    }
    if (token == "<NumDimsSelfRepaired>") {
      ReadBasicType(is, binary, &temp_float);
      ReadToken(is, binary, &token);
    }
    if (token == "<NumDimsProcessed>") {
      ReadBasicType(is, binary, &temp_float);
      ReadToken(is, binary, &token);
    }
    KALDI_ASSERT(token == "</SpecMaskOnlineComponent>");
    return;
  } else {
    ExpectToken(is, binary, "<WidthFilterMax>");
    ReadBasicType(is, binary, &width_filter_max_);
	ExpectToken(is, binary, "<WidthTimeMax>");
    ReadBasicType(is, binary, &width_time_max_);
	ExpectToken(is, binary, "<NumFilterMask>");
    ReadBasicType(is, binary, &num_filter_mask_);
	ExpectToken(is, binary, "<NumTimeMask>");
    ReadBasicType(is, binary, &num_time_mask_);
	ExpectToken(is, binary, "<RateTimeMax>");
    ReadBasicType(is, binary, &rate_time_max_);
	ExpectToken(is, binary, "<MaskValue>");
    ReadBasicType(is, binary, &mask_value_);
	ExpectToken(is, binary, "<MaskProb>");
    ReadBasicType(is, binary, &mask_prob_);
	ExpectToken(is, binary, "<MaskFilter>");
    ReadBasicType(is, binary, &mask_filter_);
	ExpectToken(is, binary, "<MaskTime>");
    ReadBasicType(is, binary, &mask_time_);
	ExpectToken(is, binary, "<MaskAsMean>");
    ReadBasicType(is, binary, &mask_as_mean_);
    ExpectToken(is, binary, "</SpecMaskOnlineComponent>");
  }
}

///////////////////////////////////////////////////////////////////////

void FramewiseLinearComponent::Init(int32 input_dim, int32 output_dim, int32 feat_dim)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim; // M
  feat_dim_ = feat_dim; // N
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ >= 0);
  KALDI_ASSERT(input_dim_ > output_dim_);
  KALDI_ASSERT(input_dim_ > feat_dim_);
  KALDI_ASSERT(input_dim_ / (output_dim_+1) == feat_dim_); // input_dim_ = N*(M+1) = N + N*M
}

void FramewiseLinearComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  int32 feat_dim = 0;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim) &&
	  cfl->GetValue("feat-dim", &feat_dim);
  if (!ok || cfl->HasUnusedValues() || output_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim, feat_dim);
}

void* FramewiseLinearComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == input_dim_);
  
  int32 row_num = in.NumRows();
  int32 param_dim = feat_dim_*output_dim_; // [ feat_dim feat_dim ... feat_dim ]
  
  CuSubMatrix<BaseFloat> feat(in.ColRange(0, feat_dim_));
  CuMatrix<BaseFloat> param(row_num, param_dim);
  param.CopyFromMat(in.ColRange(feat_dim_, param_dim));
//  CuSubMatrix<BaseFloat> param(in.ColRange(feat_dim_, param_dim));
  CuSubMatrix<BaseFloat> param_reshape(param.Data(), row_num*output_dim_, feat_dim_, feat_dim_);

/* for (int32 k = 0; k < row_num; k++) {
	std::cout << "###############" << k << "\n";
	std::cout << "############### param_reshape dim: " << (param_reshape.RowRange(k,1)).NumCols() << "\n";
	for (int32 i = 0; i < output_dim_; i++) {
		std::cout << "###############" << i << ": ";
		for (int32 j = 0; j < feat_dim_; j++) {
			std::cout << " " << (param_reshape.Range(k*output_dim_+i,1,j,1)).Sum();
		}
		std::cout << "\n";
	}
	std::cout << "############### param dim: " << (param.RowRange(k,1)).NumCols() << "\n";
	for (int32 i = 0; i < output_dim_; i++) {
		std::cout << "###############" << i << ": ";
		for (int32 j = 0; j < feat_dim_; j++) {
			std::cout << " " << (param.Range(k,1,i*feat_dim_+j,1)).Sum();
		}
		std::cout << "\n";
	}
	if (k>10) break;
}
KALDI_ERR << "here1"; */

  std::vector<CuSubMatrix<BaseFloat>* > sub_param, sub_feat, sub_out;
  
  for (int32 i = 0; i < row_num; i++)  {
	CuSubMatrix<BaseFloat> *sub_param_tmp = new CuSubMatrix<BaseFloat>(param_reshape.RowRange(i*output_dim_, output_dim_));
	sub_param.push_back(sub_param_tmp);
	CuSubMatrix<BaseFloat> *sub_feat_tmp = new CuSubMatrix<BaseFloat>(feat.RowRange(i, 1));
	sub_feat.push_back(sub_feat_tmp);
	CuSubMatrix<BaseFloat> *sub_out_tmp = new CuSubMatrix<BaseFloat>(out->RowRange(i, 1));
	sub_out.push_back(sub_out_tmp);
  }

/* for (int32 k = 0; k < row_num; k++) {
	std::cout << "###############" << k << "\n";
	
	std::cout << "############### param_reshape\n";
	for (int32 i = 0; i < output_dim_; i++) {
		std::cout << "###############" << i << ": ";
		for (int32 j = 0; j < feat_dim_; j++) {
			std::cout << " " << (param_reshape.Range(k*output_dim_+i,1,j,1)).Sum();
		}
		std::cout << "\n";
	}
	
	std::cout << "############### sub_param\n";
	for (int32 i = 0; i < output_dim_; i++) {
		std::cout << "###############" << i << ": ";
		for (int32 j = 0; j < feat_dim_; j++) {
			std::cout << " " << (sub_param[k]->Range(i,1,j,1)).Sum();
		}
		std::cout << "\n";
	}
	
	std::cout << "############### feat\n";
	for (int32 j = 0; j < feat_dim_; j++) {
		std::cout << " " << (feat.Range(k,1,j,1)).Sum();
	}
	std::cout << "\n";
	
	std::cout << "############### sub_feat\n";
	for (int32 j = 0; j < feat_dim_; j++) {
		std::cout << " " << (sub_feat[k]->Range(0,1,j,1)).Sum();
	}
	std::cout << "\n";
	
	if (k>10) break;
}
KALDI_ERR << "here1";  */

  AddMatMatBatched<BaseFloat>(1.0, sub_out,
					sub_feat, kNoTrans,
					sub_param, kTrans, 1.0);
  
  return NULL;
}

void FramewiseLinearComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {
  if (!in_deriv)  return;
  
  int32 row_num = in_value.NumRows();
  int32 param_dim = feat_dim_*output_dim_;
  
  CuSubMatrix<BaseFloat> feat(in_value.ColRange(0, feat_dim_));
  CuMatrix<BaseFloat> param(row_num, param_dim);
  param.CopyFromMat(in_value.ColRange(feat_dim_, param_dim));
//  CuSubMatrix<BaseFloat> param(in_value.ColRange(feat_dim_, param_dim));
  CuSubMatrix<BaseFloat> param_reshape(param.Data(), row_num*output_dim_, feat_dim_, feat_dim_);
  
  CuSubMatrix<BaseFloat> feat_deriv(in_deriv->ColRange(0, feat_dim_));
  CuMatrix<BaseFloat> param_deriv_reshape(row_num*output_dim_, feat_dim_);
//  CuSubMatrix<BaseFloat> param_deriv(in_deriv->ColRange(feat_dim_, param_dim));
//  CuSubMatrix<BaseFloat> param_deriv_reshape(param_deriv.Data(), row_num*output_dim_, feat_dim_, feat_dim_);
  
  std::vector<CuSubMatrix<BaseFloat>* > sub_param, sub_feat, sub_param_deriv, sub_feat_deriv, sub_out_deriv;
//  std::vector<CuSubMatrix<BaseFloat>* > sub_out;
  
  for (int32 i = 0; i < row_num; i++)  {
    CuSubMatrix<BaseFloat> *sub_param_tmp = new CuSubMatrix<BaseFloat>(param_reshape.RowRange(i*output_dim_, output_dim_));
	sub_param.push_back(sub_param_tmp);
	CuSubMatrix<BaseFloat> *sub_feat_tmp = new CuSubMatrix<BaseFloat>(feat.RowRange(i, 1));
	sub_feat.push_back(sub_feat_tmp);
//	CuSubMatrix<BaseFloat> *sub_out_tmp = new CuSubMatrix<BaseFloat>(out_value.RowRange(i, 1));
//	sub_out.push_back(sub_out_tmp);
	CuSubMatrix<BaseFloat> *sub_param_deriv_tmp = new CuSubMatrix<BaseFloat>(param_deriv_reshape.RowRange(i*output_dim_, output_dim_));
	sub_param_deriv.push_back(sub_param_deriv_tmp);
	CuSubMatrix<BaseFloat> *sub_feat_deriv_tmp = new CuSubMatrix<BaseFloat>(feat_deriv.RowRange(i, 1));
	sub_feat_deriv.push_back(sub_feat_deriv_tmp);
	CuSubMatrix<BaseFloat> *sub_out_deriv_tmp = new CuSubMatrix<BaseFloat>(out_deriv.RowRange(i, 1));
	sub_out_deriv.push_back(sub_out_deriv_tmp);
  }
  
  AddMatMatBatched<BaseFloat>(1.0, sub_feat_deriv,
					sub_out_deriv, kNoTrans,
					sub_param, kNoTrans, 1.0);

  AddMatMatBatched<BaseFloat>(1.0, sub_param_deriv,
					sub_out_deriv, kTrans,
					sub_feat, kNoTrans, 0.0);
  
  CuMatrix<BaseFloat> param_deriv(row_num, param_dim);
  CuSubMatrix<BaseFloat> param_deriv_reshape1(param_deriv.Data(), row_num*output_dim_, feat_dim_, feat_dim_);
  param_deriv_reshape1.CopyFromMat(param_deriv_reshape);
//  CuSubMatrix<BaseFloat> param_deriv(param_deriv_reshape.Data(), row_num, param_dim, param_dim);
  (in_deriv->ColRange(feat_dim_, param_dim)).AddMat(1.0, param_deriv);
  
/* for (int32 k = 0; k < row_num; k++) {
	std::cout << "###############" << k << "\n";
	std::cout << "############### param_deriv_reshape\n";
	for (int32 i = 0; i < output_dim_; i++) {
		for (int32 j = 0; j < feat_dim_; j++) {
			std::cout << " " << (param_deriv_reshape.Range(k*output_dim_+i,1,j,1)).Sum();
		}
	}
	std::cout << "\n";
	std::cout << "############### param_deriv\n";
	for (int32 i = 0; i < output_dim_; i++) {
		for (int32 j = 0; j < feat_dim_; j++) {
			std::cout << " " << (param_deriv.Range(k,1,i*feat_dim_+j,1)).Sum();
		}
	}
	std::cout << "\n";
	
	std::cout << "############### feat_deriv\n";
	for (int32 j = 0; j < feat_dim_; j++) {
		std::cout << " " << (feat_deriv.Range(k,1,j,1)).Sum();
	}
	std::cout << "\n";
	
	std::cout << "############### sub_feat_deriv\n";
	for (int32 j = 0; j < feat_dim_; j++) {
		std::cout << " " << (sub_feat_deriv[k]->Range(0,1,j,1)).Sum();
	}
	std::cout << "\n";
	
	if (k>10) break;
}
KALDI_ERR << "here1"; */

}

void FramewiseLinearComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<FramewiseLinearComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<FeatDim>");
  ReadBasicType(is, binary, &feat_dim_);
  ExpectToken(is, binary, "</FramewiseLinearComponent>");
}

void FramewiseLinearComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<FramewiseLinearComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<FeatDim>");
  WriteBasicType(os, binary, feat_dim_);
  WriteToken(os, binary, "</FramewiseLinearComponent>");
}




} // namespace nnet3
} // namespace kaldi
