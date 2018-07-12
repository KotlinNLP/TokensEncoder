/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.charactersattention

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.deeplearning.attention.han.HANEncoder
import com.kotlinnlp.simplednn.deeplearning.attention.han.HANEncodersPool
import com.kotlinnlp.simplednn.deeplearning.attention.han.HierarchySequence
import com.kotlinnlp.simplednn.core.embeddings.Embedding
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.deeplearning.attention.han.HANParameters
import com.kotlinnlp.tokensencoder.TokensEncoder

/**
 * The [TokensEncoder] that encodes a token using an [HANEncoder] on its characters.
 *
 * @property model the model of this tokens encoder
 * @property useDropout whether to apply the dropout
 * @property id an identification number useful to track a specific processor
 */
class CharsAttentionEncoder(
  private val model: CharsAttentionEncoderModel,
  override val useDropout: Boolean,
  override val id: Int = 0
) : TokensEncoder() {

  /**
   * The characters embeddings of the last encoding.
   */
  private lateinit var charsEmbeddings: List<List<Embedding>>

  /**
   * A [HANEncodersPool] to encode the chars of a token.
   */
  private val hanEncodersPool = HANEncodersPool<DenseNDArray>(
    model = this.model.charactersNetwork,
    useDropout = this.useDropout,
    propagateToInput = true)

  /**
   * The list of [HANEncoder]s used in the last encoding.
   */
  private val usedEncoders = mutableListOf<HANEncoder<DenseNDArray>>()

  /**
   * Encode a list of tokens.
   *
   * @param input an input sentence
   *
   * @return a list of dense encoded representations of the given sentence tokens
   */
  override fun forward(input: Sentence<*>): List<DenseNDArray> {

    @Suppress("UNCHECKED_CAST")
    input as Sentence<FormToken>

    this.charsEmbeddings = input.tokens.map {
      it.form.map { char -> this.model.charsEmbeddings.get(char) }
    }

    return this.encodeTokensByChars(tokens = input.tokens, charsEmbeddings = this.charsEmbeddings)
  }

  /**
   * Propagate the errors.
   *
   * @param outputErrors the errors of the current encoding
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    outputErrors.forEachIndexed { tokenIndex, tokenErrors ->
      this.usedEncoders[tokenIndex].backward(outputErrors = tokenErrors)
    }
  }

  /**
   * Encode tokens by chars using a HAN with one level.
   *
   * @param tokens the list of tokens of a sentence
   * @param charsEmbeddings the list of lists of chars embeddings, one per token
   *
   * @return a pair with the list of used encoders and the parallel list of tokens encodings by chars
   */
  private fun encodeTokensByChars(
    tokens: List<FormToken>,
    charsEmbeddings: List<List<Embedding>>
  ): List<DenseNDArray> {

    this.usedEncoders.clear()
    this.hanEncodersPool.releaseAll()

    return List(
      size = tokens.size,
      init = { i ->

        val charsVectors = HierarchySequence(*charsEmbeddings[i].map { it.array.values }.toTypedArray())

        this.usedEncoders.add(this.hanEncodersPool.getItem())

        this.usedEncoders[i].forward(charsVectors)
      })
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [CharsAttentionEncoder] parameters
   */
  override fun getParamsErrors(copy: Boolean): CharsAttentionEncoderParams {

    val hanParamsErrors = mutableListOf<HANParameters>()
    val embeddingsParamsErrors = mutableListOf<Embedding>()

    this.usedEncoders.forEachIndexed { tokenIndex, hanEncoder ->

      hanParamsErrors.add(hanEncoder.getParamsErrors(copy = copy))

      (hanEncoder.getInputErrors(copy = copy) as HierarchySequence<*>)
        .forEachIndexed { charIndex, charsErrors ->

        embeddingsParamsErrors.add(Embedding(
          id = this.charsEmbeddings[tokenIndex][charIndex].id,
          array = UpdatableDenseArray((charsErrors as DenseNDArray).copy())))
      }
    }

    return CharsAttentionEncoderParams(
      hanParameters = hanParamsErrors,
      embeddingsParams = embeddingsParamsErrors)
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
