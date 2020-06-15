/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.wrapper

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder

/**
 * A [TokensEncoder] combined with a [SentenceConverter] that wraps the conversion of the input sentence.
 *
 * @property model the model of this encoder
 * @property id an identification number useful to track a specific processor
 */
class TokensEncoderWrapper<
  FromTokenType: Token,
  FromSentenceType: Sentence<FromTokenType>,
  ToTokenType: Token,
  ToSentenceType: Sentence<ToTokenType>>
(
  override val model: TokensEncoderWrapperModel<FromTokenType, FromSentenceType, ToTokenType, ToSentenceType>,
  override val id: Int = 0
) : TokensEncoder<FromTokenType, FromSentenceType>() {

  /**
   * The tokens encoder wrapped.
   */
  private val encoder: TokensEncoder<ToTokenType, ToSentenceType> = this.model.model.buildEncoder(this.id)

  /**
   * Encode a list of tokens.
   *
   * @param input an input sentence
   *
   * @return a list of dense encoded representations of the given sentence tokens
   */
  override fun forward(input: FromSentenceType): List<DenseNDArray> =
    this.encoder.forward(this.model.converter.convert(input))

  /**
   * The Backward.
   *
   * @param outputErrors the errors of the current encoding
   */
  override fun backward(outputErrors: List<DenseNDArray>) = this.encoder.backward(outputErrors)

  /**
   * @param copy a boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the model parameters
   */
  override fun getInputErrors(copy: Boolean): NeuralProcessor.NoInputErrors = this.encoder.getInputErrors(copy)

  /**
   * @param copy whether to return by value or by reference (default true)
   *
   * @return the input errors of the last backward
   */
  override fun getParamsErrors(copy: Boolean): ParamsErrorsList = this.encoder.getParamsErrors(copy)
}
