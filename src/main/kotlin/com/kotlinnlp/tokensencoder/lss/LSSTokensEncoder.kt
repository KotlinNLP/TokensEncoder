/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.lss

import com.kotlinnlp.linguisticdescription.sentence.SentenceIdentificable
import com.kotlinnlp.linguisticdescription.sentence.token.TokenIdentificable
import com.kotlinnlp.lssencoder.LSSEncoder
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder
import java.lang.RuntimeException

/**
 * An encoder of the tokens of a sentence that uses the Latent Syntactic Structure of an [LSSEncoder].
 *
 * @property model the tokens encoder model
 * @property useDropout whether to apply the dropout during the [forward]
 * @property id an identification number useful to track a specific encoder
 */
class LSSTokensEncoder<TokenType : TokenIdentificable, SentenceType : SentenceIdentificable<TokenType>>(
  override val model: LSSTokensEncoderModel<TokenType, SentenceType>,
  override val useDropout: Boolean,
  override val id: Int = 0
) : TokensEncoder<TokenType, SentenceType>(model) {

  /**
   * The encoder of the Latent Syntactic Structure of a sentence.
   */
  private val lssEncoder = LSSEncoder(model = this.model.lssModel, useDropout = this.useDropout)

  /**
   * Encode the token forms concatenating word embeddings, latent head representations and context vectors.
   *
   * @param input the input sentence of form tokens
   *
   * @return the list of encodings, one per token
   */
  override fun forward(input: SentenceType): List<DenseNDArray> =
    this.lssEncoder.forward(input).latentSyntacticEncodings

  /**
   * Lss encoder backward.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    val splitErrors: Pair<List<DenseNDArray>, List<DenseNDArray>> = outputErrors
      .map { errors -> errors.splitV(errors.length / 2).let { it[0] to it[1] } }
      .unzip()

    this.lssEncoder.backward(outputErrors = LSSEncoder.OutputErrors(
      contextVectors = splitErrors.first,
      latentHeads = splitErrors.second
    ))
  }

  /**
   * This method should not be used because the input is a sentence.
   */
  override fun getInputErrors(copy: Boolean): NeuralProcessor.NoInputErrors {
    throw RuntimeException(
      "The input errors of the LSS Tokens Encoder cannot be obtained because the input is a sentence.")
  }

  /**
   * Return the params errors of the last backward.
   *
   * @param copy whether the returned errors must be a copy or a reference (default true)
   *
   * @return the parameters errors
   */
  override fun getParamsErrors(copy: Boolean): LSSTokensEncoderParams =
    LSSTokensEncoderParams(lssParams = this.lssEncoder.getParamsErrors(copy = copy))
}
