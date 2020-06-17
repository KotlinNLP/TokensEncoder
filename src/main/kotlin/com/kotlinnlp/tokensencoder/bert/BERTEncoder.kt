/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.bert

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.deeplearning.transformers.BERT
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder

/**
 * The [TokensEncoder] that encodes a token using a BERT model.
 *
 * @property model the encoder model
 * @property id an identification number useful to track a specific processor
 * @property propagateToInput whether to propagate the errors to the input during the [backward] (default false)
 */
class BERTEncoder<TokenType: FormToken, SentenceType: Sentence<TokenType>>(
  override val model: BERTEncoderModel<TokenType, SentenceType>,
  override val id: Int = 0,
  propagateToInput: Boolean = false
) : TokensEncoder<TokenType, SentenceType>() {

  /**
   * The BERT transformer.
   */
  private val bert = BERT(
    model = this.model.bert,
    fineTuning = true,
    autoPadding = true,
    propagateToInput = propagateToInput)

  /**
   * The accumulated params errors.
   */
  private val paramsErrorsAccumulator = ParamsErrorsAccumulator()

  /**
   * Encode a list of tokens.
   *
   * @param input an input sentence
   *
   * @return a list of dense encoded representations of the given sentence tokens
   */
  override fun forward(input: SentenceType): List<DenseNDArray> =
    this.bert.forward(input.tokens.map { it.form })

  /**
   * Propagate the errors.
   *
   * @param outputErrors the errors of the current encoding
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.bert.backward(outputErrors)

    this.paramsErrorsAccumulator.accumulate(this.bert.getParamsErrors(copy = false))
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the model parameters
   */
  override fun getParamsErrors(copy: Boolean) = this.paramsErrorsAccumulator.getParamsErrors(copy)

  /**
   * @param copy whether to return by value or by reference
   *
   * @return the input errors of the last backward
   */
  override fun getInputErrors(copy: Boolean) = this.bert.getInputErrors(copy)
}
