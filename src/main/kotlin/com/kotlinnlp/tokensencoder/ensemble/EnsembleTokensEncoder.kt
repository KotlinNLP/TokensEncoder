/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensemble

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.tokensencoder.TokensEncoderFactory
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The tokens-encoder that encodes a token by concatenating the results of other [TokensEncoder]s.
 *
 * @property model the model of this tokens encoder
 * @property useDropout whether to apply the dropout
 * @property id an identification number useful to track a specific processor
 */
class EnsembleTokensEncoder(
  override val model: EnsembleTokensEncoderModel,
  override val useDropout: Boolean,
  override val id: Int = 0
) : TokensEncoder<Token, Sentence<Token>>(model) {

  /**
   * List of tokens encoder builders.
   */
  private val encoders: List<TokensEncoder<*, *>> = this.model.models.map {
    TokensEncoderFactory(it, useDropout = this.useDropout)
  }

  /**
   * The processor of the output merge network.
   */
  private val outputMergeProcessors = BatchFeedforwardProcessor<DenseNDArray>(
    neuralNetwork = this.model.outputMergeNetwork,
    useDropout = this.useDropout,
    propagateToInput = true)

  /**
   * Encode a list of tokens.
   *
   * @param input an input sentence
   *
   * @return a list of dense encoded representations of the given sentence tokens
   */
  override fun forward(input: Sentence<Token>): List<DenseNDArray> {

    val tokenEncodings = List<MutableList<DenseNDArray>>(size = input.tokens.size, init = { mutableListOf() })

    this.encoders.forEach { encoder ->

      encoder.forward(input).forEachIndexed { tokenId, values ->
        tokenEncodings[tokenId].add(values)
      }
    }

    val arrayList: ArrayList<List<DenseNDArray>> = ArrayList(tokenEncodings.toList()) // TODO: fix this
    return this.outputMergeProcessors.forward(arrayList)
  }

  /**
   * Propagate the errors.
   *
   * @param outputErrors the errors of the current encoding
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.outputMergeProcessors.backward(outputErrors)

    val inputErrors: List<List<DenseNDArray>> = this.outputMergeProcessors.getInputsErrors(copy = false)

    this.encoders.forEachIndexed { encoderIndex, encoder ->
      encoder.backward(inputErrors.map { it[encoderIndex] } )
    }
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [TokensEncoderParameters] parameters
   */
  override fun getParamsErrors(copy: Boolean): TokensEncoderParameters =
    EnsembleTokensEncoderParams(
      encodersParams = this.encoders.map { it.getParamsErrors(copy = copy) },
      outputMergeParams = this.outputMergeProcessors.getParamsErrors(copy = copy))

  /**
   * Return the input errors of the last backward.
   *
   * @param copy whether to return by value or by reference (default true)
   *
   * @return the input errors
   */
  override fun getInputErrors(copy: Boolean) = NeuralProcessor.NoInputErrors
}
