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
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder

/**
 * The tokens-encoder that encodes a token by concatenating the results of other [TokensEncoder]s.
 *
 * @property model the model of this tokens encoder
 * @property useDropout whether to apply the dropout
 * @property id an identification number useful to track a specific processor
 */
class EnsembleTokensEncoder<TokenType: Token, SentenceType: Sentence<TokenType>>(
  override val model: EnsembleTokensEncoderModel<TokenType, SentenceType>,
  override val useDropout: Boolean,
  override val id: Int = 0
) : TokensEncoder<TokenType, SentenceType>() {

  /**
   * List of tokens encoder builders.
   */
  private val encoders: List<TokensEncoder<TokenType, SentenceType>> = this.model.components.map {
    it.buildEncoder(useDropout = this.useDropout, id = 0)
  }

  /**
   * The processor of the output merge network.
   */
  private val outputMergeProcessors = BatchFeedforwardProcessor<DenseNDArray>(
    model = this.model.outputMergeNetwork,
    useDropout = this.useDropout,
    propagateToInput = this.model.components.any { it.trainable })

  /**
   * Encode a list of tokens.
   *
   * @param input an input sentence
   *
   * @return a list of dense encoded representations of the given sentence tokens
   */
  override fun forward(input: SentenceType): List<DenseNDArray> {

    val tokenEncodings = List<MutableList<DenseNDArray>>(size = input.tokens.size, init = { mutableListOf() })

    this.encoders.forEach {
      it.forward(input).forEachIndexed { tokenId, values ->
        tokenEncodings[tokenId].add(values)
      }
    }

    return this.outputMergeProcessors.forward(tokenEncodings.map { it.toList() }.toTypedArray())
  }

  /**
   * Propagate the errors.
   *
   * @param outputErrors the errors of the current encoding
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.outputMergeProcessors.backward(outputErrors)

    if (this.outputMergeProcessors.propagateToInput) {

      this.outputMergeProcessors.getInputsErrors(copy = false).let { inputErrors ->

        this.encoders.forEachIndexed { encoderIndex, encoder ->
          if (this.model.components[encoderIndex].trainable)
            encoder.backward(inputErrors.map { it[encoderIndex] })
        }
      }
    }
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the model parameters
   */
  override fun getParamsErrors(copy: Boolean): ParamsErrorsList =
    this.outputMergeProcessors.getParamsErrors(copy) +
      this.getEncodersParamsErrors(copy)

  /**
   * @param copy whether to return by value or by reference
   *
   * @return the input errors of the last backward
   */
  override fun getInputErrors(copy: Boolean) = NeuralProcessor.NoInputErrors

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the params errors of the trainable encoders
   */
  private fun getEncodersParamsErrors(copy: Boolean): ParamsErrorsList =
    this.encoders.mapIndexed { encoderIndex, encoder ->
      if (this.model.components[encoderIndex].trainable)
        encoder.getParamsErrors(copy = copy)
      else
        emptyList()
    }.flatten()
}
