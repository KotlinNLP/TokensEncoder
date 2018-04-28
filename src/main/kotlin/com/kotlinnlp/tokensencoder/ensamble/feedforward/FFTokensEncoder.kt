/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensamble.feedforward

import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.simplednn.encoders.sequenceencoder.SequenceFeedforwardEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoderParameters
import com.kotlinnlp.tokensencoder.ensamble.concat.ConcatTokensEncoder
import com.kotlinnlp.tokensencoder.ensamble.concat.ConcatTokensEncoderParams

/**
 * The tokens-encoder that encodes a token using a feed-forward network which has in input the concatenation of the
 * results of other tokens-encoders.
 *
 * @property model the model of this tokens-encoder
 * @property trainingMode whether the encoder is being trained
 */
class FFTokensEncoder(
  private val model: FFTokensEncoderModel,
  trainingMode: Boolean
) : ConcatTokensEncoder(
  model = model,
  trainingMode = trainingMode
) {

  /**
   * The feed-forward encoder used to merge the results of the other tokens-encoders.
   */
  private val outputEncoder = SequenceFeedforwardEncoder<DenseNDArray>(this.model.tokenEncodingNetwork)

  /**
   * Encode a list of tokens.
   *
   * @param tokens a list of [Token]
   *
   * @return a list of the same size of the [tokens] with their encoded representation
   */
  override fun encode(tokens: List<Token>): Array<DenseNDArray> = this.outputEncoder.encode(super.encode(tokens))

  /**
   * Propagate the errors.
   *
   * @param errors the errors of the current encoding
   */
  override fun backward(errors: Array<DenseNDArray>) {

    this.outputEncoder.backward(errors, propagateToInput = true)
    super.backward(this.outputEncoder.getInputSequenceErrors(copy = false))
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [TokensEncoderParameters] parameters
   */
  override fun getParamsErrors(copy: Boolean): TokensEncoderParameters {

    val paramsErrors = super.getParamsErrors(copy) as ConcatTokensEncoderParams

    return FFTokensEncoderParams(
      encodersParams = paramsErrors.params,
      networkParams = this.outputEncoder.getParamsErrors(copy))
  }
}