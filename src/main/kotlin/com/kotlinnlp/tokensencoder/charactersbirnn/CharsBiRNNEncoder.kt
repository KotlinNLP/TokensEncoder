/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.charactersbirnn

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncodersPool
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters
import com.kotlinnlp.simplednn.core.embeddings.Embedding
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder

/**
 * The [TokensEncoder] that encodes a token using a BiRNN on its characters.
 *
 * @property model the model of this tokens encoder
 * @property useDropout whether to apply the dropout
 * @property id an identification number useful to track a specific processor*
 */
class CharsBiRNNEncoder(
  override val model: CharsBiRNNEncoderModel,
  override val useDropout: Boolean,
  override val id: Int = 0
) : TokensEncoder<FormToken, Sentence<FormToken>>(model) {

  /**
   * The characters embeddings of the last encoding.
   */
  private lateinit var charsEmbeddings: List<List<Embedding>>

  /**
   * A [BiRNNEncodersPool] to encode the chars of a token.
   */
  private val biRNNEncodersPool = BiRNNEncodersPool<DenseNDArray>(
    network = this.model.biRNN,
    useDropout = this.useDropout,
    propagateToInput = true)

  /**
   * The list of [BiRNNEncoder]s used in the last encoding.
   */
  private val usedEncoders = mutableListOf<BiRNNEncoder<DenseNDArray>>()

  /**
   * Encode a list of tokens.
   *
   * @param input an input sentence
   *
   * @return a list of dense encoded representations of the given sentence tokens
   */
  override fun forward(input: Sentence<FormToken>): List<DenseNDArray> {

    this.charsEmbeddings = input.tokens.map {
      it.form.map { char -> this.model.charsEmbeddings[char] }
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

      val splitErrors: List<DenseNDArray> = tokenErrors.splitV(this.model.tokenEncodingSize)

      this.usedEncoders[tokenIndex].backwardLastOutput(
        leftToRightErrors = splitErrors[0],
        rightToLeftErrors = splitErrors[1])
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
    this.biRNNEncodersPool.releaseAll()

    return List(
      size = tokens.size,
      init = { i ->
        this.usedEncoders.add(this.biRNNEncodersPool.getItem())
        this.usedEncoders[i].forward(charsEmbeddings[i].map { it.array.values })
        this.usedEncoders[i].getLastOutput(copy = true).let { concatVectorsV(it.first, it.second) }
      })
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the model parameters
   */
  override fun getParamsErrors(copy: Boolean): CharsBiRNNEncoderParams {

    val birnnParamsErrors = mutableListOf<BiRNNParameters>()
    val embeddingsParamsErrors = mutableListOf<Embedding>()

    this.usedEncoders.forEachIndexed { tokenIndex, birnnEncoder ->

      birnnParamsErrors.add(birnnEncoder.getParamsErrors(copy = copy))

      birnnEncoder.getInputErrors(copy = copy).forEachIndexed { charIndex, charsErrors ->

        embeddingsParamsErrors.add(Embedding(
          id = this.charsEmbeddings[tokenIndex][charIndex].id,
          array = UpdatableDenseArray(charsErrors.copy())))
      }
    }

    return CharsBiRNNEncoderParams(
      biRNNParameters = birnnParamsErrors,
      embeddingsParams = embeddingsParamsErrors)
  }

  /**
   * @param copy whether to return by value or by reference
   *
   * @return the input errors of the last backward
   */
  override fun getInputErrors(copy: Boolean) = NeuralProcessor.NoInputErrors
}
