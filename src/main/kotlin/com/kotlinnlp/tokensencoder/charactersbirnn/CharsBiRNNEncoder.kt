/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.charactersbirnn

import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncodersPool
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters
import com.kotlinnlp.simplednn.core.embeddings.Embedding
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder

/**
 * The [TokensEncoder] that encodes a token using a BiRNN on its characters.
 *
 * @property model the model of this tokens encoder
 * @property trainingMode whether the encoder is being trained
 */
class CharsBiRNNEncoder(
  private val model: CharsBiRNNEncoderModel,
  private val trainingMode: Boolean
) : TokensEncoder {

  /**
   * The characters embeddings of the last encoding.
   */
  private lateinit var charsEmbeddings: List<List<Embedding>>

  /**
   * A [BiRNNEncodersPool] to encode the chars of a token.
   */
  private val biRNNEncodersPool = BiRNNEncodersPool<DenseNDArray>(this.model.biRNN)

  /**
   * The list of [BiRNNEncoder]s used in the last encoding.
   */
  private val usedEncoders = mutableListOf<BiRNNEncoder<DenseNDArray>>()

  /**
   * Encode a list of tokens.
   *
   * @param tokens a list of [Token]
   *
   * @return a list of the same size of the [tokens] with their encoded representation
   */
  override fun encode(tokens: List<Token>): List<DenseNDArray>{

    this.charsEmbeddings = tokens.map {
      it.word.map { char -> this.model.charsEmbeddings.get(char) }
    }

    return this.encodeTokensByChars(tokens = tokens, charsEmbeddings = this.charsEmbeddings)
  }

  /**
   * Propagate the errors.
   *
   * @param errors the errors of the current encoding
   */
  override fun backward(errors: List<DenseNDArray>){

    errors.forEachIndexed { tokenIndex, tokenErrors ->

      val splitErrors: List<DenseNDArray> = tokenErrors.splitV(this.model.tokenEncodingSize)

      this.usedEncoders[tokenIndex].backwardLastOutput(
        leftToRightErrors = splitErrors[0],
        rightToLeftErrors = splitErrors[1],
        propagateToInput = true)
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
    tokens: List<Token>,
    charsEmbeddings: List<List<Embedding>>
  ): List<DenseNDArray> {

    this.usedEncoders.clear()
    this.biRNNEncodersPool.releaseAll()

    return List(
      size = tokens.size,
      init = { i ->
        this.usedEncoders.add(this.biRNNEncodersPool.getItem())
        this.usedEncoders[i].encode(charsEmbeddings[i].map { it.array.values })
        this.usedEncoders[i].getLastOutput(copy = true).let { concatVectorsV(it.first, it.second) }
      })
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [CharsBiRNNEncoder] parameters
   */
  override fun getParamsErrors(copy: Boolean): CharsBiRNNEncoderParams {

    val birnnParamsErrors = mutableListOf<BiRNNParameters>()
    val embeddingsParamsErrors = mutableListOf<Embedding>()

    this.usedEncoders.forEachIndexed { tokenIndex, birnnEncoder ->

      birnnParamsErrors.add(birnnEncoder.getParamsErrors(copy = copy))

      birnnEncoder.getInputSequenceErrors(copy = copy).forEachIndexed { charIndex, charsErrors ->

        embeddingsParamsErrors.add(Embedding(
          id = this.charsEmbeddings[tokenIndex][charIndex].id,
          array = UpdatableDenseArray(charsErrors.copy())))
      }
    }

    return CharsBiRNNEncoderParams(
      biRNNParameters = birnnParamsErrors,
      embeddingsParams = embeddingsParamsErrors)
  }
}
