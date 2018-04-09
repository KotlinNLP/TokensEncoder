/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.ensamble.concat

import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.utils.SplitVHelper
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.tokensencoder.TokensEncoderBuilder
import com.kotlinnlp.tokensencoder.TokensEncoderFactory
import com.kotlinnlp.tokensencoder.TokensEncoderParameters

/**
 * The tokens-encoder that encodes a token by concatenating the results of other [TokensEncoder]s.
 *
 * @property model the model of this tokens encoder
 * @property trainingMode whether the encoder is being trained
 */
open class ConcatTokensEncoder(
  private val model: ConcatTokensEncoderModel,
  private val trainingMode: Boolean
) : TokensEncoder {

  /**
   * List of tokens encoder builders.
   */
  private val encodersBuilder: List<TokensEncoderBuilder> = model.models.map {
    require(it !is ConcatTokensEncoderModel) // avoid recursion
    TokensEncoderFactory(it, this.trainingMode)
  }

  /**
   * The list of [TokensEncoder] used in the last encoding.
   */
  private val usedEncoders = mutableListOf<TokensEncoder>()

  /**
   * The backward helper to split the output errors.
   */
  private val errorsSplitter = SplitVHelper(*this.model.models.map { it.tokenEncodingSize }.toIntArray())

  /**
   * Encode a list of tokens.
   *
   * @param tokens a list of [Token]
   *
   * @return a list of the same size of the [tokens] with their encoded representation
   */
  override fun encode(tokens: List<Token>): Array<DenseNDArray> {

    this.usedEncoders.clear()
    this.usedEncoders.addAll(this.encodersBuilder.map { it.invoke() })

    val concatEncoding = Array<MutableList<DenseNDArray>>(size = tokens.size, init = { mutableListOf() })

    this.usedEncoders.forEach { encoder ->

      encoder.encode(tokens).forEachIndexed { tokenId, values ->
        concatEncoding[tokenId].add(values)
      }
    }

    return concatEncoding.map { concatVectorsV(*it.toTypedArray()) }.toTypedArray()
  }

  /**
   * Propagate the errors.
   *
   * @param errors the errors of the current encoding
   */
  override fun backward(errors: Array<DenseNDArray>) {

    val splitErrors: List<List<DenseNDArray>> = errors.map { this.errorsSplitter.split(it) }

    this.usedEncoders.forEachIndexed { encoderIndex, encoder ->
      encoder.backward(splitErrors.map { it[encoderIndex] }.toTypedArray())
    }
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [TokensEncoderParameters] parameters
   */
  override fun getParamsErrors(copy: Boolean): TokensEncoderParameters =
    ConcatTokensEncoderParams(this.usedEncoders.map { it.getParamsErrors() })
}
