/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.reduction

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder

/**
 * The [TokensEncoder] that encodes a token using another tokens encoder as input and then reduces its output vectors.
 *
 * @property model the model of this tokens encoder
 * @property useDropout whether to apply the dropout
 * @property id an identification number useful to track a specific processor*
 */
class ReductionEncoder<TokenType: Token, SentenceType: Sentence<TokenType>>(
  override val model: ReductionEncoderModel<TokenType, SentenceType>,
  override val useDropout: Boolean,
  override val id: Int = 0
) : TokensEncoder<TokenType, SentenceType>(model) {

  /**
   * The tokens encoder of input.
   */
  private val inputEncoder = this.model.inputEncoderModel.buildEncoder(useDropout = this.useDropout)

  /**
   * The reduction neural processor.
   */
  private val reductionProcessor = BatchFeedforwardProcessor<DenseNDArray>(
    neuralNetwork = this.model.reductionNetwork,
    useDropout = this.useDropout,
    propagateToInput = true)

  /**
   * Encode a list of tokens.
   *
   * @param input an input sentence
   *
   * @return a list of dense encoded representations of the given sentence tokens
   */
  override fun forward(input: SentenceType): List<DenseNDArray> =
    this.reductionProcessor.forward(this.inputEncoder.forward(input))

  /**
   * Propagate the errors.
   *
   * @param outputErrors the errors of the current encoding
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.reductionProcessor.backward(outputErrors)
    this.inputEncoder.backward(this.reductionProcessor.getInputErrors(copy = false))
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the model parameters
   */
  override fun getParamsErrors(copy: Boolean) = ReductionEncoderParams(
    inputParams = this.inputEncoder.getParamsErrors(copy = copy),
    reductionParams = this.reductionProcessor.getParamsErrors(copy = copy)
  )

  /**
   * @param copy whether to return by value or by reference
   *
   * @return the input errors of the last backward
   */
  override fun getInputErrors(copy: Boolean) = NeuralProcessor.NoInputErrors
}
