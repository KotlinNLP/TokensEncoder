/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.embeddings

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.embeddings.Embedding
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder

/**
 * The [TokensEncoder] that encodes a token using the word embeddings
 *
 * @property model the model of this tokens encoder
 * @property useDropout whether to apply the dropout
 * @property id an identification number useful to track a specific processor
 */
class EmbeddingsEncoder(
  private val model: EmbeddingsEncoderModel,
  override val useDropout: Boolean,
  override val id: Int = 0
) : TokensEncoder() {

  /**
   * The word embeddings of the last encoded sentence.
   */
  private lateinit var lastEmbeddings: List<Embedding>

  /**
   * The errors accumulated during the last backward.
   */
  private var lastEmbeddingsErrors = mutableListOf<Embedding>()

  /**
   * Encode a list of tokens.
   *
   * @param input an input sentence
   *
   * @return a list of dense encoded representations of the given sentence tokens
   */
  override fun forward(input: Sentence<*>): List<DenseNDArray> {

    this.lastEmbeddings = (0 until input.tokens.size).map {

      this.model.embeddingsMap.get(
        element = this.model.embeddingKeyExtractor.getKey(input, it),
        dropoutCoefficient = if (this.useDropout) this.model.dropoutCoefficient else 0.0)
    }

    return this.lastEmbeddings.map { it.array.values }
  }

  /**
   * Propagate the errors.
   *
   * @param outputErrors the errors of the current encoding
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    require(outputErrors.size == this.lastEmbeddings.size)

    this.lastEmbeddingsErrors.clear()

    outputErrors.forEachIndexed { i, tokenErrors ->
      this.accumulateTokenErrors(tokenIndex = i, errors = tokenErrors)
    }
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [EmbeddingsEncoderParams] parameters
   */
  override fun getParamsErrors(copy: Boolean): EmbeddingsEncoderParams {

    return EmbeddingsEncoderParams(this.lastEmbeddingsErrors) // TODO: fix copy
  }

  /**
   * Accumulate the [errors] of a given token.
   *
   * @param tokenIndex the index of a token
   * @param errors the errors to accumulate
   */
  private fun accumulateTokenErrors(tokenIndex: Int, errors: DenseNDArray) {

    this.lastEmbeddingsErrors.add(Embedding(
      id = this.lastEmbeddings[tokenIndex].id,
      array = UpdatableDenseArray(errors.copy())))
  }

  /**
   * Return the input errors of the last backward.
   *
   * @param copy whether to return by value or by reference (default true)
   *
   * @return the input errors
   */
  override fun getInputErrors(copy: Boolean) = NeuralProcessor.NoInputErrors
}
