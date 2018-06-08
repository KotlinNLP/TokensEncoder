/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.embeddings

import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.embeddings.Embedding
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder

/**
 * The [TokensEncoder] that encodes a token using the word embeddings
 *
 * @property model the model of this tokens encoder
 * @property trainingMode whether the encoder is being trained
 */
abstract class EmbeddingsEncoder(
  private val model: EmbeddingsEncoderModel,
  private val trainingMode: Boolean
) : TokensEncoder {

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
   * @param tokens a list of [Token]
   *
   * @return a list of the same size of the [tokens] with their encoded representation
   */
  override fun encode(tokens: List<Token>): List<DenseNDArray> {

    this.lastEmbeddings = tokens.map { token -> this.getEmbedding(token = token) }

    return this.lastEmbeddings.map { it.array.values }
  }

  /**
   * Propagate the errors.
   *
   * @param errors the errors of the current encoding
   */
  override fun backward(errors: List<DenseNDArray>) {

    require(errors.size == this.lastEmbeddings.size)

    this.lastEmbeddingsErrors.clear()

    errors.forEachIndexed { i, tokenErrors ->
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
   * @param token a token
   *
   * @return the word embedding of the given [token]
   */
  protected abstract fun getEmbedding(token: Token): Embedding

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
}